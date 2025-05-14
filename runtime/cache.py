# runtime/cache.py
"""
Tensor caching system optimized for memory efficiency in semantic tracing.
Provides centralized tensor management with CPU offloading capabilities.
"""

import torch
import gc
from typing import Dict, Union, Optional, List, Any

class TracingCache:
    """
    Memory-efficient tensor cache for semantic tracing.
    Handles automatic detachment and CPU offloading of large tensors.
    """
    
    def __init__(self, cpu_offload: bool = True):
        """
        Initialize the tensor cache.
        
        Args:
            cpu_offload: Whether to move tensors to CPU to conserve GPU memory
        """
        self.hidden: Dict[int, torch.Tensor] = {}
        self.attn: Dict[int, torch.Tensor] = {}
        self.grad: Dict[int, torch.Tensor] = {}
        self.saliency: Dict[int, torch.Tensor] = {}  # Changed from 'sal' to 'saliency'
        self.grad_missing: Dict[int, bool] = {}
        self.cpu_offload = cpu_offload
        
        # Size threshold for aggressive cleanup (1M elements)
        self.large_tensor_threshold = 1e6
    
    def set(self, layer: int, kind: str, tensor: torch.Tensor, detach: bool = True) -> None:
        """
        Store a tensor in the cache with automatic detachment and offloading.
        
        Args:
            layer: Layer index
            kind: Tensor type ('hidden', 'attn', 'grad', or 'saliency')
            tensor: The tensor to store
            detach: Whether to detach the tensor from the computation graph
        """
        # Validate kind parameter
        if kind not in ['hidden', 'attn', 'grad', 'saliency']:  # Updated validation list
            raise ValueError(f"Invalid tensor kind: {kind}")
        
        # Process tensor (detach and convert to CPU if needed)
        if detach and tensor.requires_grad:
            tensor = tensor.detach()
            
        # Convert to float16 for memory savings and send to CPU if enabled
        if self.cpu_offload:
            # For numerical stability, use float32 for gradient and saliency 
            if kind in ['grad', 'saliency']:
                tensor = tensor.to(torch.float32).cpu()
            else:
                tensor = tensor.to(torch.float16).cpu()
        
        # Store tensor in appropriate dictionary
        cache_dict = getattr(self, kind)
        cache_dict[layer] = tensor
        
        # Aggressive cleanup for large tensors
        if tensor.numel() > self.large_tensor_threshold and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get(self, layer: int, kind: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """
        Retrieve a tensor from the cache, optionally moving it to the specified device.
        
        Args:
            layer: Layer index
            kind: Tensor type ('hidden', 'attn', 'grad', or 'saliency')
            device: Optional device to move the tensor to
            
        Returns:
            The cached tensor, or None if not found
        """
        # Validate kind parameter
        if kind not in ['hidden', 'attn', 'grad', 'saliency']:  # Updated validation list
            raise ValueError(f"Invalid tensor kind: {kind}")
        
        # Get tensor from appropriate dictionary
        cache_dict = getattr(self, kind)
        tensor = cache_dict.get(layer)
        
        if tensor is None:
            return None
        
        # Move to requested device if specified
        if device is not None and tensor.device != device:
            tensor = tensor.to(device)
            
        return tensor
    
    def has(self, layer: int, kind: str) -> bool:
        """
        Check if a tensor exists in the cache.
        
        Args:
            layer: Layer index
            kind: Tensor type ('hidden', 'attn', 'grad', or 'saliency')
            
        Returns:
            True if the tensor exists, False otherwise
        """
        # Validate kind parameter
        if kind not in ['hidden', 'attn', 'grad', 'saliency']:  # Updated validation list
            raise ValueError(f"Invalid tensor kind: {kind}")
        
        # Check appropriate dictionary
        cache_dict = getattr(self, kind)
        return layer in cache_dict
    
    def clear_single(self, layer: int, kind: str) -> None:
        """
        Remove a specific tensor from the cache.
        
        Args:
            layer: Layer index
            kind: Tensor type ('hidden', 'attn', 'grad', or 'saliency')
        """
        # Validate kind parameter
        if kind not in ['hidden', 'attn', 'grad', 'saliency']:  # Updated validation list
            raise ValueError(f"Invalid tensor kind: {kind}")
        
        # Remove from appropriate dictionary
        cache_dict = getattr(self, kind)
        if layer in cache_dict:
            del cache_dict[layer]
            
        # Clean up after deletion
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def clear(self, kind: Optional[str] = None) -> None:
        """
        Clear the entire cache or a specific kind of tensors.
        
        Args:
            kind: Optional tensor type to clear ('hidden', 'attn', 'grad', or 'saliency').
                  If None, clears all tensors.
        """
        if kind is None:
            # Clear all caches
            self.hidden.clear()
            self.attn.clear()
            self.grad.clear()
            self.saliency.clear()  # Updated from 'sal' to 'saliency'
            self.grad_missing.clear()
        else:
            # Validate kind parameter
            if kind not in ['hidden', 'attn', 'grad', 'saliency']:  # Updated validation list
                raise ValueError(f"Invalid tensor kind: {kind}")
            
            # Clear specific cache
            cache_dict = getattr(self, kind)
            cache_dict.clear()
        
        # Clean up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# runtime/cache.py (just add/update the _GlobalSalCache class)

class _GlobalSalCache:
    """
    Global singleton cache for saliency scores calculated during backward passes.
    Used to communicate saliency values computed by custom autograd functions.
    """
    
    def __init__(self):
        """Initialize the global saliency cache."""
        self._cache: Dict[int, torch.Tensor] = {}
    
    def store(self, layer: int, sal: torch.Tensor) -> None:
        """
        Store a saliency tensor for a specific layer.
        
        Args:
            layer: Layer index
            sal: Saliency tensor (|attention * gradient|)
        """
        # Make sure we store a detached copy to avoid memory leaks
        # Use float32 for numerical stability
        if sal.requires_grad:
            sal = sal.detach()
            
        # Store with CPU offloading for memory efficiency
        self._cache[layer] = sal.to(torch.float32).cpu()
        print(f"[DEBUG][GlobalSalCache] Stored saliency for layer {layer}, shape: {tuple(sal.shape)}")
    
    def pop(self, layer: int) -> Optional[torch.Tensor]:
        """
        Retrieve and remove a saliency tensor from the cache.
        
        Args:
            layer: Layer index
            
        Returns:
            The cached saliency tensor, or None if not found
        """
        if layer in self._cache:
            result = self._cache.pop(layer)
            print(f"[DEBUG][GlobalSalCache] Popped saliency for layer {layer}")
            return result
        return None

# Global singleton instance
global_sal_cache = _GlobalSalCache()