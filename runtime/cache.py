# runtime/cache.py
"""
Caching utilities for model activations and intermediate results.
"""

import torch
import gc
from typing import Dict, Any, Optional, List

class ActivationCache:
    """
    Cache for storing model activations (hidden states) during inference.
    Provides utilities for efficient memory management.
    """
    
    def __init__(self, cpu_offload: bool = True):
        """
        Initialize the activation cache.
        
        Args:
            cpu_offload: Whether to offload tensors to CPU when possible
        """
        self.hidden_states: Dict[str, torch.Tensor] = {}
        self.attention_maps: Dict[str, torch.Tensor] = {}
        self.cpu_offload = cpu_offload
    
    def store_hidden_state(self, layer_name: str, hidden_state: torch.Tensor) -> None:
        """
        Store a hidden state tensor for a specific layer.
        
        Args:
            layer_name: Name of the layer
            hidden_state: The hidden state tensor
        """
        # Detach to avoid memory leaks
        processed = hidden_state.detach()
        
        # Offload to CPU if requested
        if self.cpu_offload:
            processed = processed.cpu()
        
        self.hidden_states[layer_name] = processed
    
    def store_attention_map(self, layer_name: str, attention_map: torch.Tensor) -> None:
        """
        Store an attention map tensor for a specific layer.
        
        Args:
            layer_name: Name of the layer
            attention_map: The attention map tensor
        """
        # Detach to avoid memory leaks
        processed = attention_map.detach()
        
        # Offload to CPU if requested
        if self.cpu_offload:
            processed = processed.cpu()
        
        self.attention_maps[layer_name] = processed
    
    def get_hidden_state(self, layer_name: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Get a hidden state tensor for a specific layer.
        
        Args:
            layer_name: Name of the layer
            device: Device to move the tensor to
            
        Returns:
            The hidden state tensor, or None if not found
        """
        if layer_name not in self.hidden_states:
            return None
        
        tensor = self.hidden_states[layer_name]
        
        # Move to requested device if provided
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
        
        return tensor
    
    def get_attention_map(self, layer_name: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Get an attention map tensor for a specific layer.
        
        Args:
            layer_name: Name of the layer
            device: Device to move the tensor to
            
        Returns:
            The attention map tensor, or None if not found
        """
        if layer_name not in self.attention_maps:
            return None
        
        tensor = self.attention_maps[layer_name]
        
        # Move to requested device if provided
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
        
        return tensor
    
    def store_model_outputs(self, outputs: Any, output_types: List[str] = None) -> None:
        """
        Store model outputs in the cache.
        
        Args:
            outputs: The model outputs
            output_types: Types of outputs to store (default: ["hidden_states", "attentions"])
        """
        if output_types is None:
            output_types = ["hidden_states", "attentions"]
        
        # Store hidden states if available
        if "hidden_states" in output_types and hasattr(outputs, "hidden_states"):
            hidden_states = outputs.hidden_states
            if hidden_states is not None:
                for idx, hidden_state in enumerate(hidden_states):
                    layer_name = f"layer_{idx}"
                    self.store_hidden_state(layer_name, hidden_state)
        
        # Store attention maps if available
        if "attentions" in output_types and hasattr(outputs, "attentions"):
            attentions = outputs.attentions
            if attentions is not None:
                for idx, attention in enumerate(attentions):
                    layer_name = f"layer_{idx}"
                    self.store_attention_map(layer_name, attention)
    
    def clear_hidden_states(self) -> None:
        """Clear all cached hidden states."""
        self.hidden_states.clear()
        gc.collect()
    
    def clear_attention_maps(self) -> None:
        """Clear all cached attention maps."""
        self.attention_maps.clear()
        gc.collect()
    
    def clear_all(self) -> None:
        """Clear all cached tensors."""
        self.clear_hidden_states()
        self.clear_attention_maps()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()