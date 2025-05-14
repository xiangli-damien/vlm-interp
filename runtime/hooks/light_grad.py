"""
Light-weight gradient hooks for memory-efficient saliency tracing.
Implements custom autograd functions to compute saliency immediately during backward pass.
"""

import torch
import logging
from typing import Optional, Tuple, Any

# Configure logging
logger = logging.getLogger("light_grad")

class AttnGradFn(torch.autograd.Function):
    """
    Custom autograd function for attention gradients.
    Immediately processes attention matrices and gradients during backward pass,
    saving saliency scores to CPU to minimize GPU memory usage.
    """
    
    @staticmethod
    def forward(ctx, attn):
        """
        Simply pass through the attention tensor for forward pass.
        
        Args:
            ctx: Context object for storing info needed in backward
            attn: Attention tensor
            
        Returns:
            The original attention tensor (preserving computational graph)
        """
        ctx.save_for_backward(attn)
        # Store layer_idx from the tensor attribute (set by the hook)
        ctx.layer_idx = getattr(attn, 'layer_idx', None)
        return attn
        
    @staticmethod
    def backward(ctx, grad):
        """
        Immediately compute saliency during backward, offload to CPU, and free memory.
        
        Args:
            ctx: Context with saved tensors
            grad: Gradient flowing back through this function
            
        Returns:
            The original gradient (for continued backprop)
        """
        # Safety check - if layer_idx is None, we can't store in cache
        if ctx.layer_idx is None:
            logger.debug("No layer_idx available in gradient hook, skipping saliency computation")
            return grad
            
        attn, = ctx.saved_tensors
        
        try:
            # Compute saliency only on CPU to save GPU memory
            sal = (attn * grad).abs().mean(dim=(0,1)).to(torch.float16).cpu()
            
            # Store in global cache
            from runtime.cache import global_saliency_cache
            global_saliency_cache.store(ctx.layer_idx, sal)
            
            # Immediately release tensors to free memory
            del attn, sal
        except Exception as e:
            logger.warning(f"Error in gradient backward hook: {e}")
        
        # Return original gradient for continued backpropagation
        return grad


class LightAttnHook:
    """
    Light-weight hook for attention matrices.
    Wraps attention tensors with custom autograd function.
    """
    def __init__(self, layer_idx):
        """
        Initialize with layer index for proper storage.
        
        Args:
            layer_idx: Index of the layer this hook is attached to
        """
        self.layer_idx = layer_idx
        
    def __call__(self, module, inputs, outputs):
        """
        Hook called during forward pass to wrap attention with our custom Function.
        
        Args:
            module: The module being processed
            inputs: Input tensors
            outputs: Output tensors from the module
            
        Returns:
            Modified output with wrapped attention
        """
        # Handle different output structures
        attn_weights = None
        
        # Extract attention from tuple outputs
        if isinstance(outputs, tuple) and len(outputs) > 1:
            hidden, attn = outputs[0], outputs[1]
            
            # Check if attn has the right shape (4D attention matrix)
            if isinstance(attn, torch.Tensor) and len(attn.shape) == 4 and attn.shape[-1] == attn.shape[-2]:
                attn_weights = attn
            # Special case pattern: some models put attention matrix at position 2
            elif len(outputs) > 2 and isinstance(outputs[2], torch.Tensor) and len(outputs[2].shape) == 4:
                attn_weights = outputs[2]
                hidden, attn = outputs[0], outputs[2]  # Reassign for later use
                
        # Handle BaseModelOutput structures
        elif hasattr(outputs, "attentions") and outputs.attentions is not None:
            # Models with HF BaseModelOutput format
            if isinstance(outputs.attentions, tuple) and len(outputs.attentions) > 0:
                attn_weights = outputs.attentions[0]  # Use first attention layer
                hidden = outputs.last_hidden_state
        
        # Process attention weights with our custom autograd function
        if attn_weights is not None:
            # Pass layer index to tensor as attribute for retrieval in Function
            # Ensure it's an integer to avoid type issues in global_saliency_cache
            attn_weights.layer_idx = int(self.layer_idx)
            
            # Wrap attention with our custom Function
            wrapped = AttnGradFn.apply(attn_weights)
            
            # Return modified output keeping the same structure
            if isinstance(outputs, tuple):
                if outputs[1] is attn_weights:
                    return (hidden, wrapped, *outputs[2:])
                elif len(outputs) > 2 and outputs[2] is attn_weights:
                    return (outputs[0], outputs[1], wrapped, *outputs[3:])
            elif hasattr(outputs, "attentions"):
                # For HF-style outputs, we need to create a new object with same attributes
                result = outputs
                # Replace the attention tuple with a new one containing our wrapped attention
                if isinstance(outputs.attentions, tuple):
                    result.attentions = (wrapped,) + outputs.attentions[1:]
                return result
        else:
            # Log when we can't find attention weights
            logger.debug(f"No attention weights found in layer {self.layer_idx} output structure")
            
        # If we couldn't identify or modify attention, return the original outputs
        return outputs