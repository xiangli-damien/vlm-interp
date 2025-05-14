"""
Lightweight gradient hooks for efficient saliency computation.
Implements custom autograd Function to calculate |attention * gradient|
during backward pass without retaining the full computation graph.
"""

import torch
import torch.nn as nn
from typing import Any, Tuple, Dict, Optional, List

# Import the global saliency cache
from runtime.cache import global_sal_cache


class LightAttnFn(torch.autograd.Function):
    """
    Custom autograd function for efficient saliency computation.
    Computes |attention * gradient| during backward pass and stores in global cache.
    """
    
    @staticmethod
    def forward(ctx, attn):
        """
        Forward pass - simply passes through the attention tensor.
        
        Args:
            ctx: Context for storing information for backward pass
            attn: Attention tensor
            
        Returns:
            The unmodified attention tensor
        """
        ctx.save_for_backward(attn)
        ctx.layer = getattr(attn, 'layer_idx', -1)
        return attn
    
    @staticmethod
    def backward(ctx, grad):
        """
        Backward pass - computes saliency score and stores in global cache.
        
        Args:
            ctx: Context containing saved tensors
            grad: Gradient tensor
            
        Returns:
            The input gradient tensor (unchanged for regular backprop)
        """
        attn, = ctx.saved_tensors
        
        # Compute saliency score |attention * gradient|
        # Taking mean across batch and head dimensions if present
        sal = (attn * grad).abs()
        if sal.dim() >= 4:  # [batch, head, seq, seq]
            sal = sal.mean((0, 1))
        elif sal.dim() == 3:  # [batch, seq, seq] or [head, seq, seq]
            sal = sal.mean(0)
        
        # Store in global cache
        if ctx.layer != -1:
            # Convert to float16 and move to CPU to save memory
            global_sal_cache.store(ctx.layer, sal)
        
        return grad


class GradAttnHook:
    """
    Hook that wraps attention outputs with LightAttnFn for saliency computation.
    """
    
    def __init__(self, layer_idx: int):
        """
        Initialize the gradient attention hook.
        
        Args:
            layer_idx: Index of the layer being hooked
        """
        self.layer_idx = layer_idx
    
    def __call__(self, m: nn.Module, inp: Tuple[torch.Tensor, ...], out: Any) -> Any:
        """
        Forward hook function applied to attention module.
        
        Args:
            m: The module being hooked
            inp: Input tensors to the module
            out: Output from the module
            
        Returns:
            Modified output with wrapped attention tensor
        """
        # Extract attention weights from output
        # For LLaVA-Next, attention is typically the second element in a tuple
        attn = out[1] if isinstance(out, tuple) and len(out) > 1 else out
        
        # Add layer index attribute to attention tensor for backward reference
        attn.layer_idx = self.layer_idx
        
        # Apply custom autograd function
        wrapped = LightAttnFn.apply(attn)
        
        # Return with same structure as input
        if isinstance(out, tuple):
            return (out[0], wrapped, *out[2:])
        
        return wrapped