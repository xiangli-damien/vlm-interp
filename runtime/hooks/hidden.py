"""
Hook implementation for capturing hidden states from model layers.
Part of the memory-efficient execution strategy for semantic tracing.
"""

import torch
import torch.nn as nn
from typing import Any, Tuple

from runtime.cache import TracingCache


class HiddenHook:
    """
    Hook for capturing hidden states from model layers.
    Stores results in the tracing cache for later use.
    """
    
    def __init__(self, layer_idx: int, cache: TracingCache):
        """
        Initialize the hidden state hook.
        
        Args:
            layer_idx: Index of the layer being hooked
            cache: Tensor cache for storing the hidden states
        """
        self.layer_idx = layer_idx
        self.cache = cache
    
    def __call__(self, module: nn.Module, inp: Tuple[torch.Tensor, ...], out: Any) -> Any:
        """
        Forward hook
        """
        hidden = out[0] if isinstance(out, tuple) else out

        cached = hidden.detach()
        self.cache.set(self.layer_idx, "hidden",
                       cached.cpu() if self.cache.cpu_offload else cached)

        # return a detached view to free GPU mem
        if isinstance(out, tuple):
            return (hidden.detach(),) + out[1:]
        return hidden.detach()