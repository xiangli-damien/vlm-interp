# runtime/hooks/manager.py
"""
Hook manager for coordinating model introspection and gradient capture.
Provides a centralized system for registering, executing, and cleaning up hooks.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Optional, Any, Set, Union

from runtime.cache import TracingCache, global_sal_cache
from runtime.hooks.light_grad import GradAttnHook
from runtime.hooks.hidden import HiddenHook
from runtime.model_utils import get_module_by_name
from runtime.hooks.attention import SaveAttnHook


class TraceHookManager:
    """
    Manages registration and execution of hooks for model introspection.
    Coordinates hidden state capture and gradient computations with memory efficiency.
    """
    
    def __init__(self, model: nn.Module, cache: TracingCache, cpu_offload: bool = True):
        """
        Initialize the hook manager.
        
        Args:
            model: The model to hook
            cache: Cache for storing tensor results
            cpu_offload: Whether to offload tensors to CPU
        """
        self.model = model
        self.cache = cache
        self.cpu_offload = cpu_offload
        
        # Track registered hooks for cleanup
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Track which layers have hooks installed
        self.layer_hooks: Dict[str, Dict[str, bool]] = {}
        
        # Device for computation
        self.device = next(model.parameters()).device
    
    def add_layer(self, layer_name: str, capture: Tuple[str, ...] = ("attention",), 
                  layer_idx: int = 0) -> None:
        """
        Register hooks for a specific layer.
        
        Args:
            layer_name: Name of the layer module
            capture: Types of data to capture ('attention', 'hidden', 'grad')
            layer_idx: Index of the layer for reference
        """
        # Get the module by name
        module = get_module_by_name(self.model, layer_name)
        
        if module is None:
            print(f"Warning: Module '{layer_name}' not found. Skipping hook registration.")
            return
        
        # Track this layer's hooks
        if layer_name not in self.layer_hooks:
            self.layer_hooks[layer_name] = {}
        
        # Register appropriate hooks based on capture types
        for cap_type in capture:
            # Skip if already registered
            if cap_type in self.layer_hooks[layer_name]:
                continue
            
            # Use the appropriate hook class based on capture type
            if cap_type == "attention":
                # For attention, use SaveAttnHook that explicitly saves to cache
                hook = SaveAttnHook(layer_idx, self.cache)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
                self.layer_hooks[layer_name][cap_type] = True
            
            elif cap_type == "grad":
                # For gradient, use GradAttnHook that wraps with autograd function
                # Skip if we've already registered an attention hook for this layer
                if 'attention' in self.layer_hooks[layer_name]:
                    continue
                    
                hook = GradAttnHook(layer_idx)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
                self.layer_hooks[layer_name][cap_type] = True
                
            elif cap_type == "hidden":
                # For hidden state capture, use HiddenHook
                hook = HiddenHook(layer_idx, self.cache)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
                self.layer_hooks[layer_name][cap_type] = True
    
    def install(self) -> int:
        """
        Complete the hook installation process. Should be called after all add_layer calls.
        
        Returns:
            Number of hooks installed
        """
        return len(self.hooks)
    
    def clear(self, keep_cache: bool = False) -> None:
        """
        Remove all registered hooks and optionally clear the cache.
        
        Args:
            keep_cache: If True, preserves cached tensors
        """
        # Remove all hooks
        for handle in self.hooks:
            handle.remove()
        
        self.hooks = []
        self.layer_hooks = {}
        
        # Clear cache if requested
        if not keep_cache:
            self.cache.clear()
    
    def run(self, inputs: Dict[str, torch.Tensor], loss_fn: Optional[Callable] = None) -> Any:
        """
        Run the model with installed hooks.
        
        Args:
            inputs: Model input tensors
            loss_fn: Optional function that computes loss for backward pass
            
        Returns:
            Model outputs
        """
        # Reset grad if doing backward pass
        if loss_fn is not None:
            self.model.zero_grad(set_to_none=True)
        
        # Forward pass
        with torch.set_grad_enabled(loss_fn is not None):
            outputs = self.model(**inputs)
            
            # Backward pass if loss function provided
            if loss_fn is not None:
                loss = loss_fn(outputs)
                if loss.requires_grad:
                    loss.backward()
                    
                # After backward pass, check for saliency results from global_sal_cache
                # and transfer them to the main cache
                for layer_idx in list(global_sal_cache._cache.keys()):
                    sal = global_sal_cache.pop(layer_idx)
                    if sal is not None:
                        self.cache.set(layer_idx, "saliency", sal)
        
        return outputs