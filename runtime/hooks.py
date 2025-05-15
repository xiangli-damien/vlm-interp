# runtime/hooks.py
"""
Utilities for managing PyTorch hooks in a modular way.
"""

import torch
import torch.nn as nn
from typing import List, Callable, Dict, Any, Optional, Set
import logging
from runtime.model_utils import get_module_by_name

logger = logging.getLogger(__name__)

def register_hooks(
    model: nn.Module,
    layer_names: List[str],
    forward_hook: Optional[Callable[[str], Callable]] = None,
    backward_hook: Optional[Callable[[str], Callable]] = None
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register forward and/or backward hooks for specified model layers.
    
    Args:
        model: The model to attach hooks to
        layer_names: List of layer names to hook
        forward_hook: Function that returns a forward hook given a layer name
        backward_hook: Function that returns a backward hook given a layer name
        
    Returns:
        List of hook handles for later removal
    """
    hooks = []
    
    # Count successful registrations
    registered = 0
    
    for name in layer_names:
        module = get_module_by_name(model, name)
        
        if module is None:
            logger.warning(f"Module '{name}' not found. Skipping hook registration.")
            continue
            
        # Register forward hook if provided
        if forward_hook is not None:
            hook_fn = forward_hook(name)
            handle = module.register_forward_hook(hook_fn)
            hooks.append(handle)
        
        # Register backward hook if provided
        if backward_hook is not None:
            hook_fn = backward_hook(name)
            handle = module.register_full_backward_hook(hook_fn)
            hooks.append(handle)
            
        registered += 1
    
    if registered < len(layer_names):
        logger.warning(f"Registered hooks for {registered}/{len(layer_names)} layers")
    else:
        logger.info(f"Successfully registered hooks for all {registered} layers")
    
    return hooks

def remove_hooks(hooks: List[torch.utils.hooks.RemovableHandle]) -> None:
    """
    Remove all hooks in the provided list.
    
    Args:
        hooks: List of hook handles to remove
    """
    if not hooks:
        return
    
    logger.info(f"Removing {len(hooks)} hooks")
    for handle in hooks:
        handle.remove()