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
        Forward hook function applied to model layer.
        
        Args:
            module: The module being hooked
            inp: Input tensors to the module
            out: Output from the module
            
        Returns:
            Unmodified output (hook is non-intrusive)
        """
        # Extract hidden state from output
        # For transformer layers, hidden state is typically the first output or the output itself
        hid = out[0] if isinstance(out, tuple) else out

        # ► detach & move‑to‑CPU before caching – avoids a *copy* later
        self.cache.set(self.layer_idx, "hidden", hid, detach=True)

        # Important: return a *detached* tensor back to the graph so that
        # PyTorch doesn’t hold the original GPU buffer alive.  If the layer
        # returns a tuple we must rebuild it; otherwise just the tensor.
        if isinstance(out, tuple):
            # keep same structure (first elem replaced by detached view)
            return (hid.detach(),) + out[1:]
        else:
            return hid.detach()