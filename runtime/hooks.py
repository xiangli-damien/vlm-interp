"""
Unified hook management system for model interpretation and tracing.
"""

import torch
import torch.nn as nn
import gc
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from runtime.cache import TracingCache
import logging
from runtime.model_utils import get_module_by_name

# Configure logging
logger = logging.getLogger("hook_manager")
logger.setLevel(logging.INFO)



class TraceHookManager:
    """
    Manages hooks for capturing and routing tensors through a model.
    
    Provides a unified interface for registering hooks on model layers,
    capturing hidden states, attention weights, gradients, and custom tensors,
    and routing them to appropriate caches.
    """
    
    def __init__(self, model: nn.Module, cpu_offload: bool = True, pin_memory: bool = False,
                 detach_after_forward: bool = True):
        """
        Initialize the hook manager.
        
        Args:
            model: The model to attach hooks to
            cpu_offload: Whether to offload tensors to CPU to save GPU memory
            pin_memory: Whether to pin memory for faster GPU transfer
            detach_after_forward: Whether to detach tensors after forward pass
        """
        self.model = model
        self.cpu_offload = cpu_offload
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.detach_after_forward = detach_after_forward
        
        # Initialize cache for storing tensors
        self.cache = TracingCache(cpu_offload=cpu_offload, pin_memory=pin_memory)
        
        # Track registered hooks
        self._forward_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._backward_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._tensor_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        
        # Track layer information
        self._layer_info: Dict[str, Dict[str, Any]] = {}
        self._layer_aliases: Dict[str, str] = {}  # Map from alias to actual name
        self._idx_to_name: Dict[int, str] = {}    # Map from layer_idx to layer_name
        
        # Track custom tensor getters
        self._custom_getters: Dict[str, Callable[[], torch.Tensor]] = {}
        
        # State flags
        self._installed = False
        self._compiling = False
        
        # Safely detect torch compile
        try:
            if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "is_compiling"):
                self._compiling = torch._dynamo.is_compiling()
        except:
            self._compiling = False
        
        if self._compiling:
            logger.warning("Hooks may be ignored during torch.compile. Consider using eager mode for tracing.")
    
    def add_layer(self, layer_name: str, capture: Union[List[str], Tuple[str, ...]] = ("hidden", "attention", "grad"), 
                 alias: Optional[str] = None, layer_idx: Optional[int] = None) -> bool:
        """
        Register a layer for hook installation.
        
        Args:
            layer_name: Full path to the layer in the model
            capture: What to capture from this layer ("hidden", "attention", "grad", "saliency", "custom")
            alias: Optional alias for the layer (useful for consistent naming)
            layer_idx: Optional numeric index for the layer
            
        Returns:
            Success flag
        """
        # Validate capture types
        valid_types = {"hidden", "attention", "grad", "saliency", "custom"}
        for c_type in capture:
            if c_type not in valid_types:
                logger.warning(f"Ignoring unknown capture type '{c_type}'")
                capture = [c for c in capture if c in valid_types]
        
        # Check if layer exists in model
        module = get_module_by_name(self.model, layer_name)
        if module is None:
            logger.warning(f"Layer '{layer_name}' not found in model")
            return False
        
        # Store layer info
        self._layer_info[layer_name] = {
            "capture": list(capture),
            "module": module,
            "index": layer_idx
        }
        
        # Store alias mapping if provided
        if alias is not None:
            self._layer_aliases[alias] = layer_name
            self._layer_info[layer_name]["alias"] = alias
            
        # Store index to name mapping if index provided
        if layer_idx is not None:
            self._idx_to_name[layer_idx] = layer_name
            
        return True
    
    def add_layers(self, layer_names: List[str], **kwargs) -> int:
        """
        Register multiple layers for hook installation.
        
        Args:
            layer_names: List of layer names
            **kwargs: Additional arguments to pass to add_layer
            
        Returns:
            Number of successfully added layers
        """
        success_count = 0
        for i, name in enumerate(layer_names):
            # Update layer_idx if not explicitly provided
            if "layer_idx" not in kwargs:
                current_kwargs = dict(kwargs, layer_idx=i)
            else:
                current_kwargs = kwargs
            
            if self.add_layer(name, **current_kwargs):
                success_count += 1
                
        return success_count
    
    def install(self) -> int:
        """
        Install all registered hooks.
        
        Returns:
            Number of hooks installed
        """
        if self._installed:
            logger.info("Hooks already installed")
            return 0
        
        # Auto-assign indices to layers that don't have them
        self._assign_missing_indices()
            
        hook_count = 0
        
        # Install hooks for each registered layer
        for layer_name, info in self._layer_info.items():
            module = info["module"]
            capture = info["capture"]
            layer_idx = info["index"]
            
            # Skip layers without indices
            if layer_idx is None:
                logger.warning(f"Layer '{layer_name}' has no index assigned, skipping hook installation")
                continue
                
            # Register forward hook if needed
            if any(c in capture for c in ["hidden", "attention"]):
                hook_func = self._create_forward_hook(layer_name, capture)
                handle = module.register_forward_hook(hook_func)
                self._forward_hooks[layer_name] = handle
                hook_count += 1
                
        self._installed = True
        logger.info(f"Installed {hook_count} hooks")
        return hook_count
    
    def run(self, inputs: Dict[str, torch.Tensor], loss_fn: Optional[Callable] = None) -> Any:
        """
        Extremely memory-optimized run method using batch-wise execution and precise memory control.
        """
        if not self._installed:
            logger.warning("Hooks not installed. Installing...")
            self.install()

        # Force memory cleanup before execution
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Always use eval mode, even when computing gradients
        original_mode = self.model.training
        self.model.eval()

        try:
            # Disable gradient tracking unless loss_fn is provided
            torch.set_grad_enabled(loss_fn is not None)

            # Create a minimal input dictionary retaining only required keys
            minimal_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    # Use shallow copy with .to() instead of .clone() for large tensors
                    device = v.device
                    minimal_inputs[k] = v.to(device, non_blocking=True)

                    # If the tensor is float32 and large, convert to float16 to save memory
                    if v.dtype == torch.float32 and v.numel() > 1000:
                        minimal_inputs[k] = v.to(torch.float16)
                else:
                    minimal_inputs[k] = v

            # Use mixed precision to reduce memory usage
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = self.model(
                    **minimal_inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

            # If a loss function is provided, compute gradients
            if loss_fn is not None:
                try:
                    # Compute the loss in mixed precision
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                        loss = loss_fn(outputs)

                    # Check if loss requires gradients
                    if not loss.requires_grad:
                        logger.warning("Loss does not require gradients. Skipping backward pass.")
                        return outputs

                    # Fully zero out gradients
                    self.model.zero_grad(set_to_none=True)

                    # Wrap backward pass in try-except to gracefully handle OOM
                    try:
                        # Key optimization: disable retain_graph and use smaller gradient norm (if applicable)
                        loss.backward(retain_graph=False)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # On OOM, attempt to recover with cleanup and fallback
                            logger.warning(f"OOM during backward pass: {e}. Attempting fallback...")

                            # Release references to large objects
                            del loss, outputs
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                            # Fall back to CPU computation or warn user that tracing is incomplete
                            logger.error("Failed to compute gradients due to OOM. Results may be incomplete.")
                        else:
                            # Re-raise non-OOM errors
                            raise

                except Exception as e:
                    logger.error(f"Error during backward pass: {e}")
                    import traceback
                    traceback.print_exc()

            return outputs

        finally:
            # Restore original model mode (train/eval)
            if self.model.training != original_mode:
                if original_mode:
                    self.model.train()
                else:
                    self.model.eval()

            # Cleanup and release all memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    
    def _assign_missing_indices(self) -> None:
        """
        Assign indices to layers that don't have them.
        This ensures all layers can be properly tracked in the cache.
        """
        # Find the maximum existing index
        max_idx = -1
        if self._idx_to_name:
            max_idx = max(self._idx_to_name.keys())
        
        # Assign new indices to layers without them
        next_idx = max_idx + 1
        for layer_name, info in self._layer_info.items():
            if info["index"] is None:
                info["index"] = next_idx
                self._idx_to_name[next_idx] = layer_name
                logger.info(f"Auto-assigned index {next_idx} to layer '{layer_name}'")
                next_idx += 1
    
    def clear(self) -> None:
        """Remove all hooks and clear cache."""
        # Remove forward hooks
        for handle in self._forward_hooks.values():
            handle.remove()
        self._forward_hooks = {}
        
        # Remove backward hooks
        for handle in self._backward_hooks.values():
            handle.remove()
        self._backward_hooks = {}
        
        # Remove tensor hooks
        for handle in self._tensor_hooks.values():
            handle.remove()
        self._tensor_hooks = {}
        
        # Clear cache
        self.cache.clear()

        for info in self._layer_info.values():
            info.pop("live_attn", None)
            info.pop("live_hidden", None)

        self._installed = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("All hooks removed and cache cleared")
    
    def snapshot(self) -> TracingCache:
        """
        Get the current cache snapshot.
        
        Returns:
            A reference to the current cache
        """
        # Capture any pending custom tensors
        self._capture_custom_tensors()
        
        return self.cache
    
    def compute_saliency(self) -> Dict[Union[int, str], torch.Tensor]:
        """
        Compute saliency (|attention * gradient|) for layers with both captured.
        
        Returns:
            Dictionary mapping layer identifiers to saliency tensors
        """
        saliency_results = {}
        
        # Process each layer
        for layer_name, info in self._layer_info.items():
            layer_idx = info.get("index")
            if layer_idx is None:
                continue
                
            # Check if we have both attention and gradients
            has_attn = self.cache.has(layer_idx, "attention")
            has_grad = self.cache.has(layer_idx, "grad")
            
            if has_attn and has_grad:
                # Get tensors
                attn = self.cache.get(layer_idx, "attention")
                grad = self.cache.get(layer_idx, "grad")
                
                # Compute saliency if shapes match
                if attn.shape == grad.shape:
                    # Compute |attention * gradient|
                    saliency = torch.abs(attn * grad)
                    
                    # Store in cache
                    self.cache.set(layer_idx, "saliency", saliency)
                    
                    # Add to results
                    saliency_results[layer_idx] = saliency
                    
                    # Clean up to save memory
                    if self.cpu_offload:
                        # We can remove the grad since we've computed saliency
                        self.cache.clear_single(layer_idx, "grad")
                else:
                    logger.warning(f"Shape mismatch for layer {layer_idx}: attention {attn.shape}, gradient {grad.shape}")
            else:
                if not has_attn:
                    logger.debug(f"Missing attention for layer {layer_idx}")
                if not has_grad:
                    logger.debug(f"Missing gradient for layer {layer_idx}")
                    # Mark as missing for potential fallback
                    self.cache.grad_missing[layer_idx] = True
        
        return saliency_results
    
    def _capture_custom_tensors(self) -> None:
        """Capture all registered custom tensors."""
        for tag, getter in self._custom_getters.items():
            try:
                tensor = getter()
                self.cache.set_custom(tag, tensor)
            except Exception as e:
                logger.error(f"Error capturing custom tensor '{tag}': {e}")
                
    def _create_forward_hook(self, layer_name: str, capture: List[str]) -> Callable:
        """
        Create a forward hook function for the specified layer.
        
        Args:
            layer_name: Name of the layer
            capture: List of tensor types to capture
            
        Returns:
            Hook function
        """
        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], 
                   outputs: Any) -> None:
            info = self._layer_info[layer_name]
            layer_idx = info.get("index")
            
            # Skip if no layer index
            if layer_idx is None:
                return
                
            # Process hidden state if requested
            if "hidden" in capture:
                hidden_state = None
                
                # Try to find hidden state in output
                if isinstance(outputs, torch.Tensor):
                    hidden_state = outputs
                elif isinstance(outputs, tuple) and len(outputs) > 0:
                    if isinstance(outputs[0], torch.Tensor):
                        hidden_state = outputs[0]
                        
                # Store hidden state if found
                if hidden_state is not None:
                    self.cache.set(
                        layer_idx,
                        "hidden",
                        hidden_state,
                        detach=self.detach_after_forward
                    )
                    # Store original tensor if we need gradients
                    if not self.detach_after_forward:
                        info["live_hidden"] = hidden_state
            
            # Process attention weights if requested
            if "attention" in capture or "grad" in capture:
                attn_weights = None
                
                # Try to find attention weights in output
                if isinstance(outputs, tuple) and len(outputs) > 1:
                    # Pattern 1: (hidden_state, attention_weights, ...)
                    if (isinstance(outputs[1], torch.Tensor) and 
                        len(outputs[1].shape) == 4 and 
                        outputs[1].shape[-1] == outputs[1].shape[-2]):
                        attn_weights = outputs[1]
                    # Pattern 2: (hidden_state, present_key_value, attention_weights, ...)
                    elif (len(outputs) > 2 and 
                          isinstance(outputs[2], torch.Tensor) and 
                          len(outputs[2].shape) == 4 and 
                          outputs[2].shape[-1] == outputs[2].shape[-2]):
                        attn_weights = outputs[2]
                
                # Store attention weights if found
                if attn_weights is not None:
                    # Save the original tensor for gradient hooks
                    if "grad" in capture and attn_weights.requires_grad:
                        # Store the live tensor in layer info
                        info["live_attn"] = attn_weights
                        
                        # Register gradient hook on the original tensor
                        tensor_hook_key = f"{layer_name}_attn_grad"
                        
                        # Remove previous hook if it exists
                        if tensor_hook_key in self._tensor_hooks:
                            self._tensor_hooks[tensor_hook_key].remove()
                            
                        # Define gradient capture function
                        def grad_hook(grad: torch.Tensor) -> None:
                            self.cache.set(layer_idx, "grad", grad)
                            if self.detach_after_forward:
                                attn = info.pop("live_attn")
                                self.cache.set(layer_idx, "attention", attn.detach())
                            
                        # Register the hook
                        handle = attn_weights.register_hook(grad_hook)
                        self._tensor_hooks[tensor_hook_key] = handle
                        
                    # Store the tensor in cache (detached if needed)
                    info["live_attn"] = attn_weights
                    if not self.detach_after_forward:
                        self.cache.set(layer_idx, "attention", attn_weights, detach=False)
                        
        return hook_fn