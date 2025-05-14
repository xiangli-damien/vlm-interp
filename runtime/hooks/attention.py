# runtime/hooks/attention.py
"""
Hooks for capturing attention weights from model layers.
Provides non-intrusive ways to store attention maps in the cache.
"""

import torch
import torch.nn as nn
from typing import Any, Tuple

from runtime.cache import TracingCache


class SaveAttnHook:
    """
    Hook that saves attention weights to the cache without modifying them.
    Enables attention analysis without requiring gradients.
    """
    
    def __init__(self, layer_idx: int, cache: TracingCache):
        """
        Initialize the attention hook.
        
        Args:
            layer_idx: Index of the layer being hooked
            cache: Tensor cache for storing the attention weights
        """
        self.layer_idx = layer_idx
        self.cache = cache
    
    def __call__(self, module: nn.Module, inp: Tuple[torch.Tensor, ...], out: Any) -> Any:
        """
        Forward hook function applied to attention module.
        
        Args:
            module: The module being hooked
            inp: Input tensors to the module
            out: Output from the module
            
        Returns:
            Unmodified output (hook is non-intrusive)
        """
        # Extract attention weights from output
        # For transformer layers, attention weights are typically the second output
        attn = out[1] if isinstance(out, tuple) and len(out) > 1 else out
        
        # Store in cache
        self.cache.set(self.layer_idx, "attn", attn)
        
        # Return unmodified output (non-intrusive hook)
        return out