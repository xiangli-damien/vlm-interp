"""
Backend component for Logit Lens analysis in VLM interpretability.
Projects hidden states through the LM head to analyze token predictions.
"""

import torch
from typing import Dict, List, Any, Optional
import logging

from runtime.cache import TracingCache
from enum import Enum
from preprocess.mapper import VisionMapper

# Configure logging
logger = logging.getLogger("logit_backend")
logger.setLevel(logging.INFO)

class TokenType(Enum):
    """Types of tokens in the input sequence."""
    TEXT = "text"
    IMAGE = "image" 
    GENERATED = "generated"
    UNKNOWN = "unknown"

class LogitBackend:
    """
    Backend component for Logit Lens analysis.
    Projects hidden states through the LM head to analyze token predictions.
    """
    
    def __init__(self, model: torch.nn.Module, cache: TracingCache, device: torch.device):
        """
        Initialize the Logit Lens backend with appropriate dtype handling.
        
        Args:
            model: The model being analyzed
            cache: Tracing cache instance
            device: Device for computation
        """
        self.model = model
        self.cache = cache
        self.device = device
        
        # Get model dtype for consistent projection
        self.model_dtype = next(model.parameters()).dtype
        logger.info(f"Model using dtype: {self.model_dtype}")
        
        # Get the language model head for projecting hidden states
        self.lm_head = None
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
            self.lm_head = model.language_model.lm_head
            logger.info("Found lm_head at model.language_model.lm_head")
        elif hasattr(model, 'lm_head'):
            self.lm_head = model.lm_head
            logger.info("Found lm_head at model.lm_head")
        else:
            # Try other common locations in transformer models
            if hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'lm_head'):
                self.lm_head = model.language_model.model.lm_head
                logger.info("Found lm_head at model.language_model.model.lm_head")
            elif hasattr(model, 'language_model') and hasattr(model.language_model, 'decoder') and hasattr(model.language_model.decoder, 'lm_head'):
                self.lm_head = model.language_model.decoder.lm_head
                logger.info("Found lm_head at model.language_model.decoder.lm_head")
            else:
                logger.warning("Could not find lm_head in model. Logit Lens will not work properly.")
        
        # Ensure lm_head is on the right device and has the same dtype as the model
        if self.lm_head is not None:
            # Move to correct device if needed
            if next(self.lm_head.parameters()).device != self.device:
                self.lm_head = self.lm_head.to(self.device)
                logger.info(f"Moved lm_head to {self.device}")
                
            # Ensure consistent dtype with model
            self.lm_head_dtype = next(self.lm_head.parameters()).dtype
            if self.lm_head_dtype != self.model_dtype:
                logger.warning(f"lm_head dtype ({self.lm_head_dtype}) differs from model dtype ({self.model_dtype})")
                # Note: We don't convert dtype here as it might affect model behavior
            
        self._last_inputs = None  # Cache the last inputs for re-computation
        
    def ensure_cache(self, inputs: Dict[str, torch.Tensor], layers: Optional[List[int]] = None) -> None:
        """
        Ensure hidden states for all layers are cached.
        
        Args:
            inputs: Model input tensors
            layers: Optional list of specific layer indices to cache
        """
        # Cache the inputs for potential reuse
        self._last_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                        for k, v in inputs.items()}
        
        # Get all hidden states in a single forward pass
        self.model.eval()
        
        try:
            # Try to get the token embeddings (layer -1) first
            embeddings = VisionMapper.get_embedding_projection(self.model, inputs["input_ids"])
            if embeddings is not None:
                # Store embeddings as layer -1
                self.cache.set(-1, "hidden", embeddings)
                logger.info("Cached token embeddings as layer -1")
            
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Check if we got hidden states
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Store hidden states for each layer
                    for idx, hidden_state in enumerate(outputs.hidden_states):
                        if layers is None or idx in layers:
                            self.cache.set(idx, "hidden", hidden_state)
                    
                    # Also store the final logits (properly detached and on CPU)
                    if hasattr(outputs, 'logits'):
                        self.cache.set_custom("final_logits", outputs.logits.detach().cpu())
                        logger.info("Cached final logits")
                else:
                    logger.warning("Model did not return hidden states")
        
        except Exception as e:
            logger.error(f"Error ensuring hidden states cache: {e}")
            raise
            
    def trace_layer(self, 
                   layer_idx: int, 
                   target_tokens: Dict[int, float],
                   selection_config: Any,
                   top_k: int = 5,
                   threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        Project hidden states at a specific layer through the LM head
        to analyze token predictions.
        
        Args:
            layer_idx: Index of the layer to trace (including -1 for embeddings)
            target_tokens: Dictionary mapping target token indices to their weights
            selection_config: Configuration for token selection (unused in this backend)
            top_k: Number of top predictions to return per token
            threshold: Minimum probability threshold for predictions
            
        Returns:
            List of source information dictionaries
        """
        if self.lm_head is None:
            logger.error("LM head not available for LogitBackend.trace_layer")
            return []
            
        # Check if we have the hidden states for this layer
        if not self.cache.has(layer_idx, "hidden"):
            logger.warning(f"Hidden states for layer {layer_idx} not in cache")
            
            # Try to recompute if we have inputs
            if self._last_inputs is not None:
                self.ensure_cache(self._last_inputs, [layer_idx])
                
                # Check again after computation
                if not self.cache.has(layer_idx, "hidden"):
                    logger.error(f"Could not compute hidden states for layer {layer_idx}")
                    return []
            else:
                return []
                
        # Get the hidden states for this layer
        hidden_states = self.cache.get(layer_idx, "hidden", self.device)
        
        # Process each target token
        results = []
        for token_idx, token_weight in target_tokens.items():
            # Skip if token_idx is out of bounds
            if token_idx >= hidden_states.shape[1]:
                logger.warning(f"Token index {token_idx} out of bounds for hidden states of shape {hidden_states.shape}")
                continue
                
            # Get hidden state for this token
            token_hidden = hidden_states[:, token_idx:token_idx+1, :]
            
            # Check batch dimension
            batch_size = token_hidden.shape[0]
            if batch_size == 0:
                logger.warning(f"Empty batch dimension for token {token_idx}")
                continue
            
            # Project through LM head
            with torch.no_grad():
                logits = self.lm_head(token_hidden)  # [batch_size, 1, vocab_size]
                logits = logits.view(batch_size, -1)  # [batch_size, vocab_size]
                probs = torch.softmax(logits, dim=-1)
                
                # Apply threshold before top-k to improve efficiency
                threshold_mask = probs > threshold
                valid_probs_count = threshold_mask.sum(dim=-1).max().item()
                
                # If no probabilities above threshold, get at least one result
                effective_k = max(1, min(top_k, valid_probs_count))
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs, k=effective_k)
            
            # Convert to CPU for processing
            top_probs = top_probs.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            logits_cpu = logits.cpu()
            
            # Create source info for each top prediction
            for batch_idx in range(top_indices.shape[0]):
                for k in range(top_indices.shape[1]):
                    prob = float(top_probs[batch_idx, k])
                    
                    # Skip if below threshold (shouldn't happen with our optimized approach)
                    if prob < threshold:
                        continue
                        
                    token_id = int(top_indices[batch_idx, k])
                    logit_value = float(logits_cpu[batch_idx, token_id])
                    
                    results.append({
                        "index": token_idx,
                        "token_id": token_id,
                        "prob": prob,
                        "logit": logit_value,
                        "weight": prob * token_weight,
                        "layer": layer_idx,
                        "rank": k + 1,
                        "type": TokenType.UNKNOWN.value
                    })
        
        return results
        
    def project_token(self, 
                    layer_idx: int, 
                    token_idx: int, 
                    tokenizer=None, 
                    top_k: int = 5) -> Dict[str, Any]:
        """
        Project a single token through the LM head and return detailed prediction info.
        
        Args:
            layer_idx: Index of the layer to analyze (including -1 for embeddings)
            token_idx: Index of the token to project
            tokenizer: Optional tokenizer for decoding token IDs
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with token projection results
        """
        if self.lm_head is None:
            logger.error("LM head not available for LogitBackend.project_token")
            return {}
        
        # Check if we have the hidden states for this layer
        if not self.cache.has(layer_idx, "hidden"):
            logger.warning(f"Hidden states for layer {layer_idx} not in cache")
            return {}
            
        # Get the hidden states for this layer
        hidden_states = self.cache.get(layer_idx, "hidden", self.device)
        
        # Check if token_idx is valid
        if token_idx >= hidden_states.shape[1]:
            logger.warning(f"Token index {token_idx} out of bounds for hidden states of shape {hidden_states.shape}")
            return {}
            
        # Get hidden state for this token
        token_hidden = hidden_states[:, token_idx:token_idx+1, :]
        
        # Project through LM head
        with torch.no_grad():
            logits = self.lm_head(token_hidden).view(token_hidden.shape[0], -1)  # [batch_size, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
        
        # Convert to CPU for processing
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        logits_cpu = logits.cpu()
        
        # Create result dictionary
        result = {
            "token_idx": token_idx,
            "layer_idx": layer_idx,
            "predictions": []
        }
        
        # Add top-k predictions
        for batch_idx in range(top_indices.shape[0]):
            for k in range(top_indices.shape[1]):
                token_id = int(top_indices[batch_idx, k])
                logit_value = float(logits_cpu[batch_idx, token_id])
                
                pred = {
                    "token_id": token_id,
                    "prob": float(top_probs[batch_idx, k]),
                    "logit": logit_value,
                    "rank": k + 1
                }
                
                # Add decoded text if tokenizer is provided
                if tokenizer is not None:
                    pred["text"] = tokenizer.decode([pred["token_id"]])
                    
                result["predictions"].append(pred)
                
        return result
    
    def project_tokens_batch(self,
                           layer_idx: int,
                           token_indices: List[int],
                           tokenizer=None,
                           top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        """
        Project multiple tokens through the LM head in a single batch for efficiency.
        
        Args:
            layer_idx: Index of the layer to analyze (including -1 for embeddings)
            token_indices: List of token indices to project
            tokenizer: Optional tokenizer for decoding token IDs
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary mapping token indices to projection results
        """
        if self.lm_head is None:
            logger.error("LM head not available for LogitBackend.project_tokens_batch")
            return {}
        
        # Check if we have the hidden states for this layer
        if not self.cache.has(layer_idx, "hidden"):
            logger.warning(f"Hidden states for layer {layer_idx} not in cache")
            return {}
            
        # Get the hidden states for this layer
        hidden_states = self.cache.get(layer_idx, "hidden", self.device)
        
        # Filter out invalid token indices
        seq_len = hidden_states.shape[1]
        valid_indices = [idx for idx in token_indices if 0 <= idx < seq_len]
        
        if not valid_indices:
            logger.warning(f"No valid token indices provided for layer {layer_idx}")
            return {}
        
        # Collect hidden states for all tokens
        batch_size = hidden_states.shape[0]
        batch_hiddens = []
        
        for token_idx in valid_indices:
            # Get hidden state for this token
            token_hidden = hidden_states[:, token_idx:token_idx+1, :]  # [batch_size, 1, hidden_dim]
            batch_hiddens.append(token_hidden)
        
        # Stack to form a single batch
        stacked_hiddens = torch.cat(batch_hiddens, dim=1)  # [batch_size, num_tokens, hidden_dim]
        
        # Reshape for projection
        hidden_size = stacked_hiddens.shape[-1]
        reshaped_hiddens = stacked_hiddens.view(-1, hidden_size)  # [batch_size*num_tokens, hidden_dim]
        
        # Project through LM head in a single pass
        with torch.no_grad():
            batch_logits = self.lm_head(reshaped_hiddens)  # [batch_size*num_tokens, vocab_size]
            batch_probs = torch.softmax(batch_logits, dim=-1)
            
            # Get top-k predictions for each token
            top_probs, top_indices = torch.topk(batch_probs, k=min(top_k, batch_probs.size(-1)))
        
        # Convert to CPU for processing
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        batch_logits_cpu = batch_logits.cpu()
        
        # Create result dictionary for each token
        results = {}
        
        # Process each token's results
        for i, token_idx in enumerate(valid_indices):
            idx_offset = i * batch_size
            token_results = {
                "token_idx": token_idx,
                "layer_idx": layer_idx,
                "predictions": []
            }
            
            # Process each batch item for this token
            for batch_idx in range(batch_size):
                flat_idx = idx_offset + batch_idx
                
                # Process top-k predictions
                for k in range(top_indices.shape[1]):
                    token_id = int(top_indices[flat_idx, k])
                    logit_value = float(batch_logits_cpu[flat_idx, token_id])
                    
                    pred = {
                        "token_id": token_id,
                        "prob": float(top_probs[flat_idx, k]),
                        "logit": logit_value,
                        "rank": k + 1
                    }
                    
                    # Add decoded text if tokenizer is provided
                    if tokenizer is not None:
                        pred["text"] = tokenizer.decode([pred["token_id"]])
                        
                    token_results["predictions"].append(pred)
            
            results[token_idx] = token_results
        
        return results