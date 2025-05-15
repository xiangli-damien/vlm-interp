"""
Logit lens backend for semantic tracing.
Analyzes hidden states by projecting through the language model head.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch

from runtime.cache import TracingCache
from runtime.hooks import TraceHookManager
from backends.base_backend import BaseBackend


class LogitBackend(BaseBackend):
    """
    Backend for projecting hidden states through the language model head.
    Provides token predictions at intermediate layers.
    """
    
    def __init__(self, model: torch.nn.Module, layer_names: List[str],
                 cache: TracingCache, device: torch.device,
                 concepts: Optional[List[str]] = None, top_k: int = 5):
        """
        Initialize the logit backend.
        
        Args:
            model: The model to analyze
            layer_names: Names of layers to trace
            cache: Tensor cache for data sharing
            device: Computation device
            concepts: Optional list of concept strings to track
            top_k: Number of top predictions to return
        """
        super().__init__(model, layer_names, cache, device)
        self.concepts = concepts or []
        self.top_k = top_k
    
    def ensure_cache(self, inputs: Dict[str, torch.Tensor], target_indices: List[int]) -> None:
        """
        Ensure hidden states are cached for all layers.
        This is used when logit_backend is directly called.
        
        Args:
            inputs: Model input tensors
            target_indices: Indices of target tokens to analyze
        """
        self.ensure_hidden(inputs)
    
    def ensure_hidden(self, inputs: Dict[str, torch.Tensor]) -> None:
        """
        Ensure hidden states are cached for all layers.
        
        Args:
            inputs: Model input tensors
        """
        # Skip if hidden states are already cached for first layer
        if self.cache.has(0, "hidden"):
            return
        
        # Set up hook manager
        hook_mgr = TraceHookManager(self.model, self.cache)
        
        # Register hooks for all layers
        for i, name in enumerate(self.layer_names):
            hook_mgr.add_layer(name, ("hidden",), i)
        
        # Install hooks and run forward pass
        hook_mgr.install()
        hook_mgr.run(inputs, loss_fn=None)
        hook_mgr.clear()
    
    def project_tokens(self, layer_idx: int, token_idx_list: List[int], 
                       tokenizer: Any) -> Dict[int, Dict[str, Any]]:
        """
        Project hidden states for specific tokens through language model head.
        
        Args:
            layer_idx: Index of the layer to analyze
            token_idx_list: List of token indices to project
            tokenizer: Tokenizer for converting IDs to text
            
        Returns:
            Dictionary mapping token indices to projection results
        """
        # Get hidden states from cache
        hidden = self.cache.get(layer_idx, "hidden", self.device)
        
        if hidden is None:
            print(f"Warning: No hidden states found for layer {layer_idx}")
            return {}
        
        # Extract hidden states for requested tokens
        if hidden.dim() == 3:  # [batch, seq, dim]
            slice = hidden[:, token_idx_list]
        else:  # [seq, dim]
            slice = hidden[token_idx_list]
        
        # Project through language model head
        logits = self.model.language_model.lm_head(slice).float()
        
        # Convert to probabilities
        if logits.dim() == 3:  # [batch, tokens, vocab]
            probs = torch.softmax(logits, -1)[0]  # Remove batch dimension
        else:  # [tokens, vocab]
            probs = torch.softmax(logits, -1)
        
        # Get top-k predictions
        topk = torch.topk(probs, self.top_k, dim=-1)
        
        # Prepare result dictionary
        result = {}
        
        for i, idx in enumerate(token_idx_list):
            token_indices = topk.indices[i].tolist()
            token_probs = topk.values[i].tolist()
            
            # Create prediction entries with text
            predictions = [
                {
                    "token": tokenizer.decode([t]),
                    "prob": float(p),
                    "token_id": t
                }
                for t, p in zip(token_indices, token_probs)
            ]
            
            # Store predictions for this token
            result[idx] = {"predictions": predictions}
            
            # Add concept predictions if requested
            if self.concepts:
                concept_predictions = {}
                
                for concept in self.concepts:
                    # Tokenize the concept
                    concept_token_ids = tokenizer.encode(concept, add_special_tokens=False)
                    
                    if concept_token_ids:
                        # Get probability for this concept's tokens
                        concept_probs = [probs[i, t].item() for t in concept_token_ids]
                        max_prob = max(concept_probs)
                        
                        concept_predictions[concept] = {
                            "probability": max_prob,
                            "token_ids": concept_token_ids
                        }
                
                result[idx]["concept_predictions"] = concept_predictions
        
        return result
        
    def trace_layer(self, layer_idx: int, 
                   targets: Dict[int, float],
                   sel_cfg: Any) -> List[Dict[str, Any]]:
        """
        This backend doesn't actually trace, but fulfills the interface. 
        Instead, it computes projections that are used by other backends.
        
        Args:
            layer_idx: Index of the layer to analyze
            targets: Dictionary mapping target token indices to weights
            sel_cfg: Configuration for token selection and pruning
            
        Returns:
            Always returns an empty list (used only via project_tokens)
        """
        return []