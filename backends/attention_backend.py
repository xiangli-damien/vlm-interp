# backends/attention_backend.py
"""
Attention backend for capturing raw attention weights from a model.
"""

import torch
import gc
from typing import Dict, List, Any, Optional, Tuple
import logging
from runtime.hooks import register_hooks, remove_hooks
from backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)

class AttentionBackend(BaseBackend):
    """
    Backend for capturing raw attention weights from model layers.
    More lightweight than saliency as it doesn't require gradient computation.
    """
    
    def __init__(self, model, device=None, cpu_offload=True):
        """
        Initialize the attention backend.
        
        Args:
            model: The model to analyze
            device: Device to run computations on
            cpu_offload: Whether to offload tensors to CPU when possible
        """
        super().__init__(model, device, cpu_offload)
        self.hooks = []
        self.attention_maps = {}
        self.hooked_layers = set()
        
    def setup(self, layer_names: List[str]) -> None:
        """
        Set up hooks for the specified layers.
        
        Args:
            layer_names: List of layer names to analyze
        """
        self.cleanup()  # Clear any existing hooks
        self.hooked_layers = set(layer_names)
        
        # Register forward hooks for all specified layers
        hooks = register_hooks(
            self.model,
            layer_names,
            forward_hook=self._create_forward_hook
        )
        
        self.hooks.extend(hooks)
        logger.info(f"Registered attention hooks for {len(self.hooks)} layers")
        
    def _create_forward_hook(self, layer_name: str):
        """Create a forward hook for capturing attention weights."""
        def hook(module, inputs, outputs):
            attn_weights = None
            
            # Identify attention weights in outputs
            if isinstance(outputs, tuple) and len(outputs) > 0:
                # Pattern 1: (hidden_state, attn_weights, ...)
                if len(outputs) > 1 and isinstance(outputs[1], torch.Tensor) and outputs[1].ndim == 4:
                    attn_weights = outputs[1]
                # Pattern 2: (hidden_state, cache, attn_weights, ...)
                elif len(outputs) > 2 and isinstance(outputs[2], torch.Tensor) and outputs[2].ndim == 4:
                    attn_weights = outputs[2]
            
            if attn_weights is not None:
                # Store the weights tensor, detached to avoid memory issues
                processed_attn = attn_weights.detach()
                if self.cpu_offload:
                    processed_attn = processed_attn.cpu()
                self.attention_maps[layer_name] = processed_attn
        
        return hook
        
    def compute(self, inputs: Dict[str, torch.Tensor], target_indices: List[int]) -> Dict[str, Any]:
        """
        Compute attention weights for the given inputs.
        Target indices are provided for API compatibility but not used, as attention
        is the same regardless of which tokens we're analyzing.
        
        Args:
            inputs: Model inputs
            target_indices: Indices of target tokens (not used for pure attention)
            
        Returns:
            Dictionary with attention maps per layer
        """
        self.clear_results()
        
        # Single forward pass to capture attention for all layers
        with torch.no_grad():
            _ = self.model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                          output_attentions=True,
                          use_cache=False)
        
        return {"attention_maps": self.attention_maps}
        
    def clear_results(self) -> None:
        """Clear previous computation results."""
        self.attention_maps.clear()
        
    def cleanup(self) -> None:
        """Remove all hooks and free resources."""
        remove_hooks(self.hooks)
        self.hooks.clear()
        
        self.clear_results()
        self.clear_cache()
        
        self.hooked_layers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()