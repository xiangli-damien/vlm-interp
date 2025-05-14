"""
Main hook management system for model interpretation and tracing.
"""

import torch
import torch.nn as nn
import gc
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from runtime.cache import TracingCache
import logging
# Import required model utility functions
from runtime.model_utils import get_module_by_name, get_llm_attention_layer_names

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
                 detach_after_forward: bool = True):  # Default changed to True
        """
        Initialize the hook manager with memory-efficient settings.
        
        Args:
            model: The model to attach hooks to
            cpu_offload: Whether to offload tensors to CPU to save GPU memory
            pin_memory: Whether to pin memory for faster GPU transfer
            detach_after_forward: Whether to detach tensors after forward pass
                                 (now True by default for memory efficiency)
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
    
    def add_layer(self, layer_name: str, capture: Union[List[str], Tuple[str, ...]] = ("attention", "grad"), 
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
        Ultra memory-optimized run method with proper error handling and fallbacks.
        
        Args:
            inputs: Dictionary of input tensors
            loss_fn: Optional loss function for gradient computation
            
        Returns:
            Model outputs or error dictionary
        """
        if not self._installed:
            logger.warning("Hooks not installed. Installing...")
            self.install()

        # Force memory cleanup before execution
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Always use eval mode for consistency, even when computing gradients
        original_mode = self.model.training
        self.model.eval()
        
        # Initialize outputs to None to avoid reference errors
        outputs = None

        try:
            # Disable gradient tracking unless loss_fn is provided
            with torch.set_grad_enabled(loss_fn is not None):
                # Create a minimal input dictionary with mixed precision to save memory
                minimal_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        device = v.device
                        # Use lower precision for large float tensors
                        if v.dtype == torch.float32 and v.numel() > 1000:
                            minimal_inputs[k] = v.to(torch.float16, device=device, non_blocking=True)
                        else:
                            minimal_inputs[k] = v.to(device, non_blocking=True)
                    else:
                        minimal_inputs[k] = v

                # Only use autocast for forward pass to reduce memory without sacrificing gradient accuracy
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                    try:
                        outputs = self.model(
                            **minimal_inputs,
                            output_hidden_states=False,
                            return_dict=True
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning(f"OOM during forward pass: {e}")
                            # Clear memory and return early with failure indicator
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            return {"error": "OOM during forward pass"}
                        else:
                            raise

                # Clean up forward pass memory immediately
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # If a loss function is provided, compute gradients with proper error handling
                if loss_fn is not None and outputs is not None:
                    try:
                        # Compute loss without autocast to allow proper gradient flow
                        loss = loss_fn(outputs)
                            
                        # Check if loss requires gradients
                        if not loss.requires_grad:
                            logger.warning("Loss does not require gradients. Skipping backward pass.")
                            return outputs

                        # Zero gradients with set_to_none for better memory efficiency
                        self.model.zero_grad(set_to_none=True)
                        
                        # Use retain_graph=False to minimize memory usage during backward pass
                        loss.backward(retain_graph=False)
                        
                        # Clear memory immediately after backward
                        del loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Handle OOM during backward pass gracefully
                            logger.warning(f"OOM during backward pass: {e}. Cleaning up...")
                            
                            # Release references to tensors
                            if 'loss' in locals():
                                del loss
                            
                            # Force garbage collection
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Update outputs to include error information
                            if isinstance(outputs, dict):
                                outputs["error"] = "OOM during backward pass"
                            else:
                                # Create a new dict with the error if outputs is not a dict
                                temp = {"original_outputs": outputs, "error": "OOM during backward pass"}
                                outputs = temp
                        else:
                            # Re-raise other errors
                            raise

                return outputs

        except Exception as e:
            logger.error(f"Error during hook manager run: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
            
        finally:
            # Restore original model mode (train/eval)
            if self.model.training != original_mode:
                if original_mode:
                    self.model.train()
                else:
                    self.model.eval()

            # Final cleanup
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
    
    def clear(self, keep_cache: bool = False) -> None:
        """
        Remove all hooks and optionally clear cache.
        
        Args:
            keep_cache: If True, preserve the cache contents (useful when
                    the cache has been transferred to another object)
        """
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
        
        # Clear layer-specific live tensors
        for info in self._layer_info.values():
            info.pop("live_attn", None)
            info.pop("live_hidden", None)
        
        # Only clear cache if keep_cache is False
        if not keep_cache:
            self.cache.clear()

        self._installed = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("All hooks removed" + (" and cache cleared" if not keep_cache else " (cache preserved)"))
    
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
        Convert every *(attention, gradient)* pair still present in the cache
        into a *saliency tensor* and *immediately* discard the source tensors.

        This method is idempotent and can be called multiple times; layers that
        already have a saliency tensor are skipped.

        Returns
        -------
        Dict[int | str, torch.Tensor]
            Mapping from *layer index* **or** custom alias to the newly
            generated saliency tensor.  (Layers that were already processed are
            omitted from the dictionary.)
        """
        new_saliency: Dict[Union[int, str], torch.Tensor] = {}

        for layer_name, info in self._layer_info.items():
            idx = info.get("index")
            if idx is None:
                # Should never happen, but we guard anyway.
                continue

            # Skip if saliency already exists or prerequisites are missing.
            if self.cache.has(idx, "saliency"):
                continue
            if not (self.cache.has(idx, "attention") and self.cache.has(idx, "grad")):
                if not self.cache.has(idx, "grad"):
                    # Mark missing gradient – helpful for fall‑back logic upstream.
                    self.cache.grad_missing[idx] = True
                continue

            # Retrieve tensors (they are on CPU if `cpu_offload=True`).
            attn = self.cache.get(idx, "attention")
            grad = self.cache.get(idx, "grad")

            if attn.shape != grad.shape:
                logger.warning(
                    "TraceHookManager.compute_saliency – shape mismatch "
                    f"in layer {idx}: attn {attn.shape} vs grad {grad.shape}"
                )
                continue

            sal = torch.abs(attn * grad)
            self.cache.set(idx, "saliency", sal)
            new_saliency[idx] = sal

            # Free up memory – attention & grads are no longer needed.
            self.cache.clear_single(idx, "attention")
            self.cache.clear_single(idx, "grad")

        return new_saliency

    
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
        Create a memory-efficient forward hook function for the specified layer.
        
        Args:
            layer_name: Name of the layer
            capture: List of tensor types to capture
            
        Returns:
            Hook function
        """
        def hook_fn(module: nn.Module, inputs: Tuple[torch.Tensor, ...], outputs: Any) -> Any:
            info = self._layer_info[layer_name]
            layer_idx = info.get("index")
            
            # Skip if no layer index
            if layer_idx is None:
                return outputs
                
            # Extract hidden state and attention for consistent handling
            hidden_state = None
            attn_weights = None
            
            # Try to find hidden state in output
            if isinstance(outputs, torch.Tensor):
                hidden_state = outputs
            elif isinstance(outputs, tuple) and len(outputs) > 0:
                if isinstance(outputs[0], torch.Tensor):
                    hidden_state = outputs[0]
                    
            # Try to find attention weights
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
                    
            # IMPORTANT FIX: Process hidden state BEFORE attention processing
            # This ensures hidden state is captured even when LightAttnHook is used
            if "hidden" in capture and hidden_state is not None:
                # Immediately detach and move to CPU for memory efficiency
                cpu_hidden = hidden_state.detach().to(torch.float16).cpu()
                self.cache.set(layer_idx, "hidden", cpu_hidden, detach=False)
            
            # Process attention weights if needed and found
            if ("attention" in capture or "grad" in capture) and attn_weights is not None:
                # Use the LightAttnHook for efficient attention gradient processing
                from runtime.hooks import LightAttnHook
                return LightAttnHook(layer_idx)(module, inputs, outputs)
                    
            # Return outputs unchanged if no attention processing happened
            return outputs
                
        return hook_fn

    def _register_hooks(self):
        """Register hooks for all attention layers."""
        # Get attention layer names
        attn_layer_names = get_llm_attention_layer_names(self.model)
        logger.info(f"Found {len(attn_layer_names)} attention layer names")
        
        # Filter to ensure we only get language model layers, not vision layers
        lm_attn_layers = []
        for name in attn_layer_names:
            # Check for language model path components
            if any(pattern in name for pattern in ["language_model", "lm_model", "text_model"]):
                lm_attn_layers.append(name)
        
        # If we didn't find any with those patterns, use all layers
        if not lm_attn_layers:
            lm_attn_layers = attn_layer_names
            logger.warning("Could not identify specific language model layers, using all attention layers")
        
        # Build an explicit mapping from index to layer name
        self._idx_to_name = {}
        for idx, name in enumerate(lm_attn_layers):
            if idx in self.layers:
                self._idx_to_name[idx] = name
                # Register hooks with explicit layer index
                self.hooks.add_layer(
                    name, 
                    capture=["hidden", "attention", "grad"], 
                    layer_idx=idx
                )
        
        # Important: Set detach_after_forward=False to enable gradient flow
        # Fix: Directly set the attribute instead of using _detach_after_forward which doesn't exist
        self.hooks.detach_after_forward = False
        
        # Install hooks
        num_installed = self.hooks.install()
        logger.info(f"Installed {num_installed} hooks on language model layers")