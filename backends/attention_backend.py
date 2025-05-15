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
        self.use_single_pass = False  # FIX: Add support for single forward pass mode
        
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
        Supports two modes:
        1. Standard mode: Uses hooks to capture attention for each layer individually
        2. Single-pass mode: Gets all layer attention maps in one forward pass
        
        Args:
            inputs: Model inputs
            target_indices: Indices of target tokens (not used for pure attention)
            
        Returns:
            Dictionary with attention maps per layer
        """
        self.clear_results()
        
        # Ensure model in eval mode
        self.model.eval()
        
        # FIX: Use single forward pass if requested
        if self.use_single_pass:
            logger.info("Using single forward pass to compute all attention maps")
            with torch.no_grad():
                outputs = self.model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                    output_attentions=True,
                                    return_dict=True)
                
                if hasattr(outputs, "attentions") and outputs.attentions:
                    for idx, attention in enumerate(outputs.attentions):
                        layer_name = self._layer_idx_to_name(idx)
                        if layer_name in self.hooked_layers:
                            if self.cpu_offload:
                                self.attention_maps[layer_name] = attention.detach().cpu()
                            else:
                                self.attention_maps[layer_name] = attention.detach()
                else:
                    logger.warning("Model did not output attention maps despite output_attentions=True")
        else:
            # Original implementation: use hooks to capture attention
            with torch.no_grad():
                _ = self.model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                              output_attentions=True)
        
        return {"attention_maps": self.attention_maps}
    
    def _layer_idx_to_name(self, idx: int) -> str:
        """
        Convert layer index to layer name based on hooked layers.
        
        Args:
            idx: The layer index
            
        Returns:
            The corresponding layer name from hooked_layers, or a default name
        """
        # Simple implementation - in practice we would need a better mapping
        for name in self.hooked_layers:
            if f".{idx}." in name or name.endswith(f".{idx}"):
                return name
        return f"layer_{idx}"
        
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