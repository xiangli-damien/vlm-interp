"""
Attention-based semantic tracing backend.
Analyzes information flow using attention weights without gradient computation.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch

from runtime.cache import TracingCache
from runtime.selection import SelectionConfig, SelectionStrategy
from runtime.hooks import TraceHookManager
from backends.base_backend import BaseBackend


class AttentionBackend(BaseBackend):
    """
    Backend for analyzing information flow using attention weights.
    Uses direct attention values without gradient computation.
    """
    
    def ensure_cache(self, inputs: Dict[str, torch.Tensor], target_indices: List[int]) -> None:
        """
        Ensure attention weights are cached for all layers.
        
        Args:
            inputs: Model input tensors
            target_indices: Indices of target tokens to analyze
        """
        # Skip if attention is already cached for first layer
        if self.cache.has(0, "attn"):
            return
        
        # Set up hook manager
        hook_mgr = TraceHookManager(self.model, self.cache)
        
        # Register hooks for all layers
        for i, name in enumerate(self.layer_names):
            hook_mgr.add_layer(name, ("attention",), layer_idx=i)
        
        # Install hooks and run forward pass
        hook_mgr.install()
        hook_mgr.run(inputs, loss_fn=None)
        hook_mgr.clear()
        
        # Verify that attention was actually cached
        if not self.cache.has(0, "attn"):
            print("Warning: Attention weights were not cached properly. Check hook implementation.")
    
    def trace_layer(self, layer_idx: int, 
                   targets: Dict[int, float],
                   sel_cfg: SelectionConfig) -> List[Dict[str, Any]]:
        """
        Trace a specific layer using attention weights.
        
        Args:
            layer_idx: Index of the layer to trace
            targets: Dictionary mapping target token indices to weights
            sel_cfg: Configuration for token selection and pruning
            
        Returns:
            List of source token records with importance weights
        """
        # Get attention weights from cache
        att = self.cache.get(layer_idx, "attn", self.device)
        
        if att is None:
            print(f"Warning: No attention weights found for layer {layer_idx}")
            return []
        
        # Average over batch and head dimensions if present
        if att.ndim == 4:  # [batch, head, seq, seq]
            att = att.mean((0, 1))
        elif att.ndim == 3:  # [batch, seq, seq] or [head, seq, seq]
            att = att.mean(0)
        
        # Process each target token
        result = []
        
        for tgt, w in targets.items():
            # Get attention vector for this target (causal: only consider past tokens)
            vec = att[tgt, :tgt]
            
            # Skip if empty (e.g., first token)
            if vec.numel() == 0:
                continue
            
            # Select important source tokens
            srcs = SelectionStrategy.select_sources(vec, sel_cfg)
            
            # Record selected sources
            for sidx, sw in srcs:
                result.append({
                    "index": sidx,          # Token index
                    "weight": sw * w,       # Global importance weight
                    "raw": sw,              # Raw importance score
                    "target": tgt,          # Which token this is a source for
                    "attention_score": sw,  # Same as raw, for clarity
                    "type": "unknown"       # Will be filled later by workflow
                })
        
        # Apply layer-level pruning
        result = self._prune_layer(result, sel_cfg)
        
        # Clear attention tensor to save memory
        self.cache.clear_single(layer_idx, "attn")
        
        return result