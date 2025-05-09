# runtime/cache.py
"""
Caching system for efficient model activation storage during semantic tracing.
"""

import torch
import gc
from typing import Dict, Any, Optional, Union, List


class TracingCache:
    """
    Caches model activations (hidden states, attention, saliency) during semantic tracing.
    Provides automatic CPU offloading to conserve GPU memory during deep analysis.
    """
    
    def __init__(self, cpu_offload: bool = True, pin_memory: bool = False):
        """
        Initialize the tracing cache.
        
        Args:
            cpu_offload: Whether to offload tensors to CPU
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        self.hidden_states: Dict[int, torch.Tensor] = {}
        self.attention: Dict[int, torch.Tensor] = {}
        self.saliency: Dict[int, torch.Tensor] = {}
        self.grad: Dict[int, torch.Tensor] = {}  # New: Store gradients
        self.custom: Dict[str, Any] = {}  # New: Store custom tensors/objects
        self.grad_missing: Dict[int, bool] = {}  # New: Track missing gradients
        
        self.cpu_offload = cpu_offload
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
    def set(self, layer_idx: int, cache_type: str, tensor: torch.Tensor, detach: bool = True) -> None:
        """
        Store a tensor in the cache.
        
        Args:
            layer_idx: Layer index to use as key
            cache_type: Type of cache ("hidden", "attention", "saliency", "grad")
            tensor: Tensor to store
            detach: Whether to detach the tensor (for gradient computation)
        """
        if cache_type not in ["hidden", "attention", "saliency", "grad"]:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        # Process tensor (detach and move to CPU if needed)
        processed = tensor
        if detach:
            processed = processed.detach()
        if self.cpu_offload:
            processed = processed.cpu()
            if self.pin_memory:
                processed = processed.pin_memory()
                
        # Store in the appropriate cache dictionary
        if cache_type == "hidden":
            self.hidden_states[layer_idx] = processed
        elif cache_type == "attention":
            self.attention[layer_idx] = processed
        elif cache_type == "saliency":
            self.saliency[layer_idx] = processed
        elif cache_type == "grad":
            self.grad[layer_idx] = processed
            # Clear grad_missing flag if it was set
            if layer_idx in self.grad_missing:
                del self.grad_missing[layer_idx]
    
    def set_custom(self, tag: str, obj: Any) -> None:
        """
        Store a custom object in the cache.
        
        Args:
            tag: Unique tag/name for the object
            obj: Object to store (can be any type)
        """
        # If it's a tensor, process it like other tensors
        if isinstance(obj, torch.Tensor):
            processed = obj.detach()
            if self.cpu_offload:
                processed = processed.cpu()
                if self.pin_memory:
                    processed = processed.pin_memory()
            self.custom[tag] = processed
        else:
            # Store non-tensor objects directly
            self.custom[tag] = obj
        
    def get(self, layer_idx: int, cache_type: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor from the cache.
        
        Args:
            layer_idx: Layer index key
            cache_type: Type of cache ("hidden", "attention", "saliency", "grad")
            device: Optional device to move tensor to
            
        Returns:
            Cached tensor or None if not found
        """
        if cache_type not in ["hidden", "attention", "saliency", "grad"]:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        # Get the appropriate cache dictionary
        cache_dict = None
        if cache_type == "hidden":
            cache_dict = self.hidden_states
        elif cache_type == "attention":
            cache_dict = self.attention
        elif cache_type == "saliency":
            cache_dict = self.saliency
        elif cache_type == "grad":
            cache_dict = self.grad
        
        if layer_idx not in cache_dict:
            return None
            
        tensor = cache_dict[layer_idx]
        
        return tensor
    
    def get_custom(self, tag: str, device: Optional[torch.device] = None) -> Optional[Any]:
        """
        Retrieve a custom object from the cache.
        
        Args:
            tag: Tag/name of the object
            device: Optional device to move tensor to (if object is a tensor)
            
        Returns:
            Cached object or None if not found
        """
        if tag not in self.custom:
            return None
            
        obj = self.custom[tag]
        
        # If it's a tensor, handle device movement
        if isinstance(obj, torch.Tensor) and device is not None and obj.device != device:
            obj = obj.to(device)
            
        return obj
        
    def has(self, layer_idx: int, cache_type: str) -> bool:
        """
        Check if a tensor exists in the cache.
        
        Args:
            layer_idx: Layer index key
            cache_type: Type of cache ("hidden", "attention", "saliency", "grad")
            
        Returns:
            True if the tensor exists in the cache
        """
        if cache_type not in ["hidden", "attention", "saliency", "grad"]:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        # Check the appropriate cache dictionary
        if cache_type == "hidden":
            return layer_idx in self.hidden_states
        elif cache_type == "attention":
            return layer_idx in self.attention
        elif cache_type == "saliency":
            return layer_idx in self.saliency
        elif cache_type == "grad":
            return layer_idx in self.grad
        
        return False
    
    def has_custom(self, tag: str) -> bool:
        """
        Check if a custom object exists in the cache.
        
        Args:
            tag: Tag/name of the object
            
        Returns:
            True if the object exists in the cache
        """
        return tag in self.custom
    
    def clear_single(self, layer_idx: int, cache_type: str) -> None:
        """
        Clear a single tensor from the cache.
        
        Args:
            layer_idx: Layer index key
            cache_type: Type of cache ("hidden", "attention", "saliency", "grad")
        """
        if cache_type not in ["hidden", "attention", "saliency", "grad"]:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        # Remove from the appropriate cache dictionary
        if cache_type == "hidden" and layer_idx in self.hidden_states:
            del self.hidden_states[layer_idx]
        elif cache_type == "attention" and layer_idx in self.attention:
            del self.attention[layer_idx]
        elif cache_type == "saliency" and layer_idx in self.saliency:
            del self.saliency[layer_idx]
        elif cache_type == "grad" and layer_idx in self.grad:
            del self.grad[layer_idx]
    
    def clear_custom(self, tag: str) -> None:
        """
        Clear a custom object from the cache.
        
        Args:
            tag: Tag/name of the object
        """
        if tag in self.custom:
            del self.custom[tag]
        
    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            cache_type: Optional type of cache to clear. If None, clears all caches.
        """
        if cache_type is None:
            self.hidden_states.clear()
            self.attention.clear()
            self.saliency.clear()
            self.grad.clear()
            self.custom.clear()
            self.grad_missing.clear()
        elif cache_type == "hidden":
            self.hidden_states.clear()
        elif cache_type == "attention":
            self.attention.clear()
        elif cache_type == "saliency":
            self.saliency.clear()
        elif cache_type == "grad":
            self.grad.clear()
        elif cache_type == "custom":
            self.custom.clear()
        elif cache_type == "grad_missing":
            self.grad_missing.clear()
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
            
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()