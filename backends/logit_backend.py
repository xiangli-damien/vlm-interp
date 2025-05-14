"""
Backend component for Logit Lens analysis in VLM interpretability.
Projects hidden states through the LM head to analyze token predictions.
"""

import torch
from typing import Dict, List, Any, Optional
import logging
import gc

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
    
    def __init__(self, model: torch.nn.Module, cache: TracingCache, device: torch.device, concepts: Optional[List[str]] = None):
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
        
        # Default concepts to track (can be overridden)
        self.concepts = concepts or [
            "France"
        ]
        
        logger.info(f"Initializing LogitBackend with concepts: {self.concepts}")

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
        
    # In your LogitBackend class
    def project_token(self, 
                    layer_idx: int, 
                    token_idx: int, 
                    tokenizer=None, 
                    top_k: int = 3) -> Dict[str, Any]:
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
            print("LM head not available for LogitBackend.project_token")
            return {}
        
        # Create token decoder if possible
        token_decoder = None
        if tokenizer is not None:
            try:
                from runtime.decode import TokenDecoder
                token_decoder = TokenDecoder(tokenizer)
            except ImportError:
                pass
        
        # Check if we have the hidden states for this layer
        if not self.cache.has(layer_idx, "hidden"):
            print(f"Hidden states for layer {layer_idx} not in cache")
            return {}
            
        # Get the hidden states for this layer
        hidden_states = self.cache.get(layer_idx, "hidden", self.device)
        
        # Check if token_idx is valid
        if token_idx >= hidden_states.shape[1]:
            print(f"Token index {token_idx} out of bounds for hidden states of shape {hidden_states.shape}")
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
                
                # Get token text using TokenDecoder if available
                if token_decoder is not None:
                    token_text = token_decoder.decode_token(token_id)
                elif tokenizer is not None:
                    token_text = tokenizer.decode([token_id])
                else:
                    token_text = f"<tok_{token_id}>"
                    
                pred = {
                    "token_id": token_id,
                    "prob": float(top_probs[batch_idx, k]),
                    "logit": logit_value,
                    "rank": k + 1,
                    "text": token_text
                }
                    
                result["predictions"].append(pred)
                
        return result
    
    def project_tokens_batch(self,
                       layer_idx: int,
                       token_indices: List[int],
                       tokenizer=None,
                       top_k: int = 5) -> Dict[int, Dict[str, Any]]:
        """
        Memory-optimized version that projects tokens through the LM head in small batches.
        
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
        
        # Initialize results dictionary
        results = {}
        
        # MEMORY OPTIMIZATION: Process tokens in small batches
        # This prevents loading all token hidden states at once
        max_batch_size = 10  # Adjust based on your GPU memory
        
        # Filter invalid token indices first to avoid unnecessary processing
        valid_indices = []
        try:
            # Get sequence length without loading full tensor to device
            hidden_states = self.cache.get(layer_idx, "hidden")
            seq_len = hidden_states.shape[1]
            valid_indices = [idx for idx in token_indices if 0 <= idx < seq_len]
            
            if not valid_indices:
                logger.warning(f"No valid token indices provided for layer {layer_idx}")
                return {}
        except Exception as e:
            logger.error(f"Error checking valid indices: {e}")
            return {}
        
        # Process in small batches
        for i in range(0, len(valid_indices), max_batch_size):
            batch_indices = valid_indices[i:i+max_batch_size]
            logger.debug(f"Processing token batch {i//max_batch_size + 1}/{(len(valid_indices)-1)//max_batch_size + 1}")
            
            try:
                # MEMORY OPTIMIZATION: Load hidden states for only this batch of tokens
                # Get hidden states selectively - only move needed tokens to device
                hidden_states = self.cache.get(layer_idx, "hidden")
                
                # Extract only the required token positions from CPU tensor
                batch_hiddens = []
                for token_idx in batch_indices:
                    # Select only this token's hidden state
                    token_hidden = hidden_states[:, token_idx:token_idx+1, :]
                    batch_hiddens.append(token_hidden)
                
                # Stack token hidden states for this small batch
                if batch_hiddens:
                    # Move stacked tensor to device only after stacking to minimize transfers
                    stacked_hiddens = torch.cat(batch_hiddens, dim=1).to(self.device)
                    
                    # MEMORY OPTIMIZATION: Use half precision if possible
                    if stacked_hiddens.dtype == torch.float32:
                        stacked_hiddens = stacked_hiddens.to(torch.float16)
                    
                    # Reshape for projection
                    hidden_size = stacked_hiddens.shape[-1]
                    batch_size = stacked_hiddens.shape[0]
                    num_tokens = len(batch_indices)
                    reshaped_hiddens = stacked_hiddens.view(-1, hidden_size)
                    
                    # Project through LM head
                    with torch.no_grad():
                        # MEMORY OPTIMIZATION: Use autocast for mixed precision if available
                        if hasattr(torch, 'autocast') and torch.cuda.is_available():
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                batch_logits = self.lm_head(reshaped_hiddens)
                        else:
                            batch_logits = self.lm_head(reshaped_hiddens)
                        
                        # Calculate probabilities and get top-k in separate steps to reduce peak memory
                        # MEMORY OPTIMIZATION: Process one token at a time within the batch
                        for i, token_idx in enumerate(batch_indices):
                            token_start = i * batch_size
                            token_end = (i + 1) * batch_size
                            
                            # Get logits for this token
                            token_logits = batch_logits[token_start:token_end]
                            
                            # Calculate softmax and get top-k predictions
                            token_probs = torch.softmax(token_logits, dim=-1)
                            top_probs, top_indices = torch.topk(token_probs, k=min(top_k, token_probs.size(-1)))
                            
                            # Move to CPU for processing
                            top_probs_cpu = top_probs.cpu()
                            top_indices_cpu = top_indices.cpu()
                            token_logits_cpu = token_logits.cpu()
                            
                            # Create result for this token
                            token_results = {
                                "token_idx": token_idx,
                                "layer_idx": layer_idx,
                                "predictions": []
                            }
                            
                            # Process each prediction
                            for batch_idx in range(batch_size):
                                for k in range(top_indices_cpu.shape[1]):
                                    token_id = int(top_indices_cpu[batch_idx, k].item())
                                    logit_value = float(token_logits_cpu[batch_idx, token_id].item())
                                    
                                    pred = {
                                        "token_id": token_id,
                                        "prob": float(top_probs_cpu[batch_idx, k].item()),
                                        "logit": logit_value,
                                        "rank": k + 1
                                    }
                                    
                                    # Add decoded text if tokenizer is provided
                                    if tokenizer is not None:
                                        pred["text"] = tokenizer.decode([pred["token_id"]])
                                        
                                    token_results["predictions"].append(pred)
                            
                            # Store results for this token
                            results[token_idx] = token_results
                            
                            # MEMORY OPTIMIZATION: Explicitly delete tensors
                            del token_logits, token_probs, top_probs, top_indices
                    
                    # MEMORY OPTIMIZATION: Explicitly delete batch tensors
                    del batch_logits, stacked_hiddens, reshaped_hiddens
                    
                    # MEMORY OPTIMIZATION: Force garbage collection
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing token batch {i//max_batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
            
            # MEMORY OPTIMIZATION: Clear batch-specific variables
            del batch_hiddens
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results