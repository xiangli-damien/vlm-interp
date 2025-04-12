# -*- coding: utf-8 -*-
"""
PyTorch Hook Utilities for Model Analysis.

Provides classes to capture intermediate activations, attention weights,
and gradients during model forward and backward passes using hooks.
"""

import torch
import torch.nn as nn
import gc
from typing import Dict, Any, Optional, List, Tuple, Callable, Set

# Import necessary utility from model_utils (assuming it exists/will exist)
# If model_utils.py is not yet created, this import will fail.
# You might temporarily define get_module_by_name here or ensure model_utils.py exists.
try:
    from .model_utils import get_module_by_name
except ImportError:
    print("Warning: Could not import 'get_module_by_name' from '.model_utils'. Ensure model_utils.py exists.")
    # Define a placeholder or raise an error if needed immediately
    def get_module_by_name(model, name):
        print("Error: get_module_by_name is not available. Define it in utils/model_utils.py.")
        return None # Or raise NotImplementedError


class ActivationCache:
    """
    Captures activations (hidden states) and optionally attention weights
    from specified model layers using PyTorch forward hooks.

    Provides a context manager interface for easy hook registration and removal.
    Stores captured tensors in detached mode on CPU by default to save memory,
    but behavior can be adjusted.
    """
    def __init__(self, cpu_offload: bool = True):
        """
        Initializes the ActivationCache.

        Args:
            cpu_offload (bool): If True, move captured tensors to CPU and detach them
                                immediately to conserve GPU memory. Defaults to True.
        """
        self.activations: Dict[str, torch.Tensor] = {} # Stores hidden states
        self.attentions: Dict[str, torch.Tensor] = {}  # Stores attention weights
        self._hooks: List[torch.utils.hooks.RemovableHandle] = [] # Tracks registered hook handles
        self.cpu_offload = cpu_offload
        self._layers_hooked: Set[str] = set()

    def _create_hook_fn(self, layer_name: str, capture_attention: bool = False) -> Callable:
        """
        Factory function to create a forward hook callback for a specific layer.

        Args:
            layer_name (str): The name identifier for the layer being hooked.
            capture_attention (bool): If True, the hook will also attempt to capture
                                     attention weights from the layer's output.

        Returns:
            Callable: The hook function to be registered.
        """
        def hook(module: nn.Module, input_args: Tuple[torch.Tensor], output: Any):
            """
            The actual hook function executed during the forward pass.
            """
            # --- Capture Hidden State ---
            hidden_state: Optional[torch.Tensor] = None
            if isinstance(output, torch.Tensor):
                # Simplest case: output is the hidden state tensor
                hidden_state = output
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                 # Common case: output is a tuple, first element is the hidden state
                 # (e.g., Transformer block output: hidden_state, attentions, cache)
                 hidden_state = output[0]
            # Add more checks here if models have different output structures

            if hidden_state is not None:
                 # Detach and optionally move to CPU
                 processed_hs = hidden_state.detach()
                 if self.cpu_offload:
                     processed_hs = processed_hs.cpu()
                 self.activations[layer_name] = processed_hs
            else:
                 print(f"Warning: Could not identify hidden state tensor in output of layer '{layer_name}' (type: {type(output)}).")


            # --- Optionally Capture Attention Weights ---
            if capture_attention:
                attn_weights: Optional[torch.Tensor] = None
                # Look for attention weights in common output structures
                # Typically shape [batch_size, num_heads, seq_len_q, seq_len_k]
                if isinstance(output, tuple) and len(output) > 1:
                     # Common pattern 1: (hidden_state, attention_weights, ...)
                     if isinstance(output[1], torch.Tensor) and output[1].ndim == 4 and output[1].shape[-1] == output[1].shape[-2]:
                          attn_weights = output[1]
                     # Common pattern 2: (hidden_state, present_key_value, attention_weights, ...) - Less common for direct layer output
                     elif len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim == 4 and output[2].shape[-1] == output[2].shape[-2]:
                          attn_weights = output[2]

                if attn_weights is not None:
                     processed_attn = attn_weights.detach()
                     if self.cpu_offload:
                         processed_attn = processed_attn.cpu()
                     self.attentions[layer_name] = processed_attn
                # else: # Be less verbose if attention isn't found unless debugging
                #      print(f"Note: Did not find expected attention pattern in output of layer '{layer_name}'.")

        return hook

    def capture_outputs(self, model: nn.Module, layers_to_hook: List[str], attention_layers: Optional[List[str]] = None):
        """
        Registers forward hooks on the specified layers of the model.

        Clears any previously registered hooks and captured data before registering new ones.

        Args:
            model (nn.Module): The model instance to attach hooks to.
            layers_to_hook (List[str]): A list of layer names (e.g., 'model.layers.0.mlp')
                                         from which to capture hidden states.
            attention_layers (Optional[List[str]]): A list of layer names (e.g., 'model.layers.0.self_attn')
                                                   from which to capture attention weights. Defaults to None.

        Returns:
            self: Returns the ActivationCache instance for potential chaining.
        """
        self.clear() # Clear previous state

        if attention_layers is None:
             attention_layers = []

        all_target_layers = set(layers_to_hook + attention_layers)
        self._layers_hooked = all_target_layers.copy() # Store names for reference

        print(f"Registering hooks for {len(all_target_layers)} target layers...")

        for name, module in model.named_modules():
            if name in all_target_layers:
                capture_attn = name in attention_layers
                hook_fn = self._create_hook_fn(name, capture_attention=capture_attn)
                handle = module.register_forward_hook(hook_fn)
                self._hooks.append(handle)
                # print(f"  Hook registered for layer: '{name}' (Capture Attention: {capture_attn})") # Verbose

        print(f"Successfully registered {len(self._hooks)} hooks.")
        if len(self._hooks) != len(all_target_layers):
             print(f"Warning: Expected to register {len(all_target_layers)} hooks, but registered {len(self._hooks)}. Some layer names might be incorrect.")

        return self

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Returns the dictionary of captured activation tensors."""
        return self.activations

    def get_attentions(self) -> Dict[str, torch.Tensor]:
        """Returns the dictionary of captured attention weight tensors."""
        return self.attentions

    def clear_hooks(self):
        """Removes all registered forward hooks."""
        if not self._hooks:
            return # Nothing to remove
        print(f"Removing {len(self._hooks)} registered hooks...")
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._layers_hooked = set()
        print("Hooks removed.")

    def clear_cache(self):
        """Clears the stored activation and attention data."""
        print("Clearing cached activations and attentions...")
        self.activations = {}
        self.attentions = {}
        gc.collect() # Trigger garbage collection
        print("Cache cleared.")

    def clear(self):
        """Clears both hooks and cached data."""
        self.clear_hooks()
        self.clear_cache()

    def __enter__(self):
        """Allows using the cache as a context manager."""
        # Hook registration should happen *before* entering the context usually
        # e.g., with ActivationCache(...) as cache: cache.capture_outputs(...)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Ensures hooks are cleared when exiting the context manager block."""
        self.clear_hooks()
        # Optionally clear cache too, or leave it for inspection
        # self.clear_cache()
        print("Exiting ActivationCache context, hooks cleared.")


class GradientAttentionCapture:
    """
    Captures attention weights during the forward pass and their corresponding
    gradients during the backward pass for specific attention modules.

    Designed for saliency analysis methods like |attention * gradient|.
    Includes memory optimizations by processing gradients immediately and
    offloading results to CPU.

    Uses `register_full_backward_hook` for module-level gradient capture,
    and registers tensor hooks within the backward hook to capture gradients
    with respect to the attention weights tensor itself.
    """
    def __init__(self, cpu_offload: bool = True):
        """
        Initializes the GradientAttentionCapture.

        Args:
            cpu_offload (bool): If True, computed saliency scores are moved to CPU
                                immediately to save GPU memory. Defaults to True.
        """
        self.attention_weights: Dict[str, torch.Tensor] = {}  # Stores weights from forward pass (on original device)
        self.attention_grads: Dict[str, torch.Tensor] = {}    # Stores gradients w.r.t weights (on original device)
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._tensor_grad_hooks: Dict[str, torch.utils.hooks.RemovableHandle] = {} # Tracks tensor-specific grad hooks
        self._hooked_layers: Set[str] = set()
        self.cpu_offload = cpu_offload
        self.saliency_scores: Dict[str, torch.Tensor] = {} # Stores computed saliency scores


    def _create_forward_hook_fn(self, layer_name: str) -> Callable:
        """Creates a forward hook function to capture attention weights."""
        def hook(module: nn.Module, input_args: Tuple[torch.Tensor], output: Any):
            """Forward hook: Captures attention weights."""
            attn_weights: Optional[torch.Tensor] = None
            # Identify attention weights (common patterns)
            if isinstance(output, tuple) and len(output) > 0:
                 # Pattern 1: (hidden_state, attn_weights, ...)
                if len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].ndim == 4 and output[1].shape[-1] == output[1].shape[-2]:
                    attn_weights = output[1]
                 # Pattern 2: (hidden_state, cache, attn_weights, ...)
                elif len(output) > 2 and isinstance(output[2], torch.Tensor) and output[2].ndim == 4 and output[2].shape[-1] == output[2].shape[-2]:
                    attn_weights = output[2]

            if attn_weights is not None:
                 # Store the weights tensor itself. It needs requires_grad=True
                 # for the backward hook on the tensor to work later.
                 # Do NOT detach here.
                 if not attn_weights.requires_grad:
                      # This can happen if the model is run in no_grad mode *before* the backward pass
                      # Or if the layer itself doesn't have gradients enabled properly.
                      print(f"Warning: Attention weights for layer '{layer_name}' do not require grad. Gradient capture may fail.")
                 self.attention_weights[layer_name] = attn_weights
            # else:
            #      print(f"Debug: Did not find attention weights in output of '{layer_name}'.")

        return hook

    def _create_backward_hook_fn(self, layer_name: str) -> Callable:
        """
        Creates a *module* backward hook function. This hook's primary role is
        to register a *tensor* hook on the corresponding attention weights captured
        during the forward pass, right before the gradients are computed for that tensor.
        """
        def module_backward_hook(module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]):
            """Module backward hook: Registers a hook on the attention tensor."""
            # Check if we captured weights for this layer in the forward pass
            if layer_name in self.attention_weights:
                attn_weights_tensor = self.attention_weights[layer_name]

                # Ensure the tensor requires gradients for the hook to fire
                if attn_weights_tensor.requires_grad:
                    # Define the tensor hook function inline
                    def _capture_tensor_grad(grad: torch.Tensor):
                        # Store the gradient (detached) for later saliency calculation
                        # print(f"Debug: Capturing gradient for tensor of layer '{layer_name}' (Shape: {grad.shape})")
                        self.attention_grads[layer_name] = grad.detach() # Detach to prevent holding graph history

                    # Register the hook *on the tensor* itself.
                    # This hook will execute when the gradient *for this specific tensor* is computed.
                    # Use register_hook for non-leaf tensors if needed, but usually weights are parameters (leaves)
                    # Important: Only register *once* per backward pass if possible.
                    # Check if already registered to avoid duplicates (though handles are usually removed)
                    if layer_name not in self._tensor_grad_hooks:
                         # print(f"Debug: Registering tensor grad hook for '{layer_name}'")
                         handle = attn_weights_tensor.register_hook(_capture_tensor_grad)
                         self._tensor_grad_hooks[layer_name] = handle
                    # else:
                    #      print(f"Debug: Tensor grad hook already registered for '{layer_name}'")

                # else: # Debugging print
                     # print(f"Debug: Skipping tensor hook registration for '{layer_name}' - weights tensor does not require grad.")
            # else: # Debugging print
                 # print(f"Debug: Skipping backward processing for '{layer_name}' - no attention weights found in forward pass.")

        return module_backward_hook


    def register_hooks(self, model: nn.Module, layer_names: List[str]):
        """
        Registers forward and backward hooks on the specified module names.

        Args:
            model (nn.Module): The model instance.
            layer_names (List[str]): List of attention module names to hook.

        Returns:
            self: The GradientAttentionCapture instance.
        """
        self.clear() # Clear previous state
        self._hooked_layers = set(layer_names)
        print(f"Registering gradient capture hooks for {len(layer_names)} layers...")

        hook_count = 0
        for name in layer_names:
            module = get_module_by_name(model, name)
            if module is not None and isinstance(module, nn.Module):
                # Register forward hook to capture weights
                f_handle = module.register_forward_hook(self._create_forward_hook_fn(name))
                self._forward_hooks.append(f_handle)

                # Register *module* backward hook to trigger tensor grad hook registration
                b_handle = module.register_full_backward_hook(self._create_backward_hook_fn(name))
                self._backward_hooks.append(b_handle)
                hook_count += 1
            else:
                print(f"Warning: Module '{name}' not found or is not an nn.Module. Cannot register hooks.")

        print(f"Registered {hook_count * 2} hooks ({hook_count} forward, {hook_count} backward).")
        if hook_count != len(layer_names):
             print(f"Warning: Expected to register hooks for {len(layer_names)} layers, but only did for {hook_count}.")
        return self

    def compute_saliency(self):
        """
        Computes saliency scores (|attention * gradient|) after a backward pass.

        Assumes `register_hooks` was called, a forward pass was run, and a backward
        pass (e.g., `loss.backward()`) triggering the hooks has completed.

        Stores results in `self.saliency_scores` (offloaded to CPU if configured).
        Clears the stored weights and gradients afterwards to free memory.

        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping layer names to saliency score tensors.
                                    Tensors are on CPU if `cpu_offload` is True.
        """
        print("Computing saliency scores (|Attention * Gradient|)...")
        self.saliency_scores = {} # Clear previous scores
        processed_layers = set()

        # Iterate through layers for which gradients were captured
        captured_grad_layers = list(self.attention_grads.keys()) # Iterate over copy of keys
        print(f"Found gradients for {len(captured_grad_layers)} layers.")

        for layer_name in captured_grad_layers:
            if layer_name in self.attention_weights:
                attn_weights = self.attention_weights[layer_name]
                grad = self.attention_grads[layer_name]

                # Ensure tensors are compatible (should be if captured correctly)
                if attn_weights.shape != grad.shape:
                    print(f"Warning: Shape mismatch for layer '{layer_name}'! Weights: {attn_weights.shape}, Grad: {grad.shape}. Skipping.")
                    continue
                if attn_weights.device != grad.device:
                     print(f"Warning: Device mismatch for layer '{layer_name}'! Weights: {attn_weights.device}, Grad: {grad.device}. Attempting to move grad.")
                     try:
                          grad = grad.to(attn_weights.device)
                     except Exception as e:
                          print(f"  Error moving gradient: {e}. Skipping layer.")
                          continue

                # Compute saliency: |Attention * Gradient|
                # Use float() for calculation to avoid potential low-precision issues
                saliency = torch.abs(attn_weights.float() * grad.float())

                # Store the result, optionally offloading to CPU
                if self.cpu_offload:
                    self.saliency_scores[layer_name] = saliency.detach().cpu()
                else:
                    self.saliency_scores[layer_name] = saliency.detach() # Keep on original device

                processed_layers.add(layer_name)

                # --- Memory Cleanup within loop ---
                # Remove the stored weight and gradient tensor for this layer now that saliency is computed
                del self.attention_weights[layer_name]
                del self.attention_grads[layer_name]

            else:
                # This case might happen if grad hook fired but forward hook didn't capture weights somehow
                print(f"Warning: Gradient found for layer '{layer_name}', but no corresponding attention weights were captured. Cannot compute saliency.")


        print(f"Computed saliency scores for {len(self.saliency_scores)} layers.")

        # Final cleanup of any remaining weights/grads (should be empty if loop worked)
        self.attention_weights.clear()
        self.attention_grads.clear()
        gc.collect()
        if torch.cuda.is_available():
             torch.cuda.empty_cache()

        return self.saliency_scores


    def clear_hooks(self):
        """Removes all registered forward, backward, and tensor hooks."""
        if not self._forward_hooks and not self._backward_hooks and not self._tensor_grad_hooks:
             return
        print(f"Removing gradient capture hooks ({len(self._forward_hooks)} Fwd, {len(self._backward_hooks)} Bwd, {len(self._tensor_grad_hooks)} Tensor)...")
        for handle in self._forward_hooks: handle.remove()
        for handle in self._backward_hooks: handle.remove()
        for handle in self._tensor_grad_hooks.values(): handle.remove() # Remove tensor hooks too!

        self._forward_hooks = []
        self._backward_hooks = []
        self._tensor_grad_hooks = {}
        self._hooked_layers = set()
        print("Gradient capture hooks removed.")


    def clear_cache(self):
        """Clears stored attention weights, gradients, and computed saliency scores."""
        print("Clearing gradient capture cache (weights, grads, saliency)...")
        self.attention_weights = {}
        self.attention_grads = {}
        self.saliency_scores = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Gradient capture cache cleared.")

    def clear(self):
        """Clears hooks and all cached data."""
        self.clear_hooks()
        self.clear_cache()

    def __enter__(self):
        """Context manager entry."""
        # Hooks should be registered before entering usually
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Context manager exit: ensures hooks are cleared."""
        self.clear_hooks()
        print("Exiting GradientAttentionCapture context, hooks cleared.")

    def __del__(self):
         """Ensure cleanup when the object is deleted."""
         self.clear_hooks() # Mostly important to remove hooks