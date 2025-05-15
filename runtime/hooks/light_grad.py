# runtime/hooks/light_grad.py
"""
Lightweight gradient hooks for efficient saliency computation.
Implements custom autograd Function to calculate |attention * gradient| during backward pass.
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
    def forward(ctx, attn, layer_idx):
        """
        Forward pass - simply passes through the attention tensor.
        
        Args:
            ctx: Context for storing information for backward pass
            attn: Attention tensor
            layer_idx: Index of current layer
            
        Returns:
            The unmodified attention tensor
        """
        ctx.save_for_backward(attn)
        ctx.layer_idx = layer_idx
        
        # Create wrapped tensor and add _base attribute
        attn_wrapped = attn  # Reference to maintain computational graph
        attn_wrapped._base = attn  # Add reference to original tensor
        
        # Important: We need to ensure the same tensor is returned
        # to maintain the computational graph
        return attn_wrapped
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass - computes saliency score and stores in global cache.
        
        Args:
            ctx: Context containing saved tensors
            grad_output: Gradient tensor
            
        Returns:
            The input gradient tensor (unchanged for regular backprop)
        """
        attn, = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        
        print(f"[DEBUG][LightAttnFn] backward called for layer {layer_idx}; grad.shape={tuple(grad_output.shape)}")
        
        # Ensure matching shapes and device
        if attn.shape != grad_output.shape:
            print(f"[WARNING][LightAttnFn] Shape mismatch: attn {tuple(attn.shape)}, grad {tuple(grad_output.shape)}")
            try:
                # Try to match dimensions by taking means or expanding
                if grad_output.dim() > attn.dim():
                    # Reduce dimensions by taking means
                    reduced_grad = grad_output
                    while reduced_grad.dim() > attn.dim():
                        reduced_grad = reduced_grad.mean(0)
                    grad_output = reduced_grad
                elif grad_output.dim() < attn.dim():
                    # Expand dimensions if needed
                    expanded_grad = grad_output
                    while expanded_grad.dim() < attn.dim():
                        expanded_grad = expanded_grad.unsqueeze(0)
                    grad_output = expanded_grad.expand_as(attn)
                    
                # Final adjustment to match exact shape
                if grad_output.shape != attn.shape:
                    grad_output = grad_output.expand_as(attn)
            except Exception as e:
                print(f"[ERROR][LightAttnFn] Error adjusting gradient shape: {e}")
                # If we can't match shapes, return gradient as is
                return grad_output, None
        
        # Match device if needed
        if attn.device != grad_output.device:
            grad_output = grad_output.to(attn.device)
            
        # Compute saliency score |attention * gradient|
        try:
            # Element-wise multiplication followed by absolute value
            saliency = (attn * grad_output).abs()  # Changed variable name from sal to saliency
            
            # Average over batch and head dimensions if present
            if saliency.dim() == 4:  # [batch, head, seq, seq]
                saliency = saliency.mean((0, 1))
            elif saliency.dim() == 3:  # [batch, seq, seq] or [head, seq, seq]
                saliency = saliency.mean(0)
            
            print(f"[DEBUG][LightAttnFn] computed saliency for layer {layer_idx}; saliency.shape={tuple(saliency.shape)}")
            
            # Validate saliency - ensure it's not all zeros or NaNs
            if torch.isnan(saliency).any():
                print(f"[WARNING][LightAttnFn] NaN values detected in saliency map for layer {layer_idx}")
                saliency = torch.where(torch.isnan(saliency), torch.zeros_like(saliency), saliency)
                
            if saliency.abs().sum().item() < 1e-10:
                print(f"[WARNING][LightAttnFn] Near-zero saliency map for layer {layer_idx}")
                # Add small epsilon to ensure non-zero values
                saliency = saliency + torch.ones_like(saliency) * 1e-10
                
            # Store in global cache
            if layer_idx != -1:
                print(f"[DEBUG][LightAttnFn] storing saliency for layer {layer_idx}; saliency.shape={tuple(saliency.shape)}")
                # Store the saliency map in global cache
                global_sal_cache.store(layer_idx, saliency)
        except Exception as e:
            print(f"[ERROR][LightAttnFn] Error computing saliency: {e}")
        
        # Return gradient unchanged - this is critical for proper backprop
        return grad_output, None  # Also return None for the layer_idx gradient

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
        if not isinstance(out, tuple) or len(out) <= 1:
            print(f"[WARNING][GradAttnHook] Unexpected output structure for layer {self.layer_idx}: {type(out)}")
            return out
        
        attn_idx = 1  # Typically attention maps are the second element in output tuple
        
        if len(out) <= attn_idx:
            print(f"[WARNING][GradAttnHook] Output tuple too short for layer {self.layer_idx}: {len(out)}")
            return out
        
        attn = out[attn_idx]
        if not torch.is_tensor(attn):
            print(f"[WARNING][GradAttnHook] Expected attention tensor, got {type(attn)}")
            return out
        
        # Only apply the function if we can compute gradients
        if attn.requires_grad:
            print(f"[DEBUG][GradAttnHook] Wrapping attention for layer {self.layer_idx}")
            # Wrap with custom autograd function, explicitly passing layer_idx
            wrapped_attn = LightAttnFn.apply(attn, self.layer_idx)
            
            # Return with modified attention tensor
            new_out = list(out)
            new_out[attn_idx] = wrapped_attn
            return tuple(new_out)
        else:
            print(f"[WARNING][GradAttnHook] Attention tensor doesn't require gradients for layer {self.layer_idx}")
            return out
