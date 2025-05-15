# backends/saliency_backend.py
"""
Saliency backend for computing gradient-based attention saliency maps.
"""

import torch
import torch.nn as nn
import gc
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from runtime.hooks import register_hooks, remove_hooks
from backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)

class SaliencyBackend(BaseBackend):
    """
    Backend for computing gradient-based saliency scores (|attention * gradient|).
    Captures attention weights and their gradients during model execution.
    """
    
    def __init__(self, model, device=None, cpu_offload=True, epsilon=1e-7, batch_size=2):
        """
        Initialize the saliency backend.
        
        Args:
            model: The model to analyze
            device: Device to run computations on
            cpu_offload: Whether to offload tensors to CPU when possible
            epsilon: Small value for numerical stability
            batch_size: Number of layers to process in a batch
        """
        super().__init__(model, device, cpu_offload)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.attention_weights = {}
        self.attention_grads = {}
        self.hooks = []
        self.tensor_hooks = {}
        self.hooked_layers = set()
        self.saliency_scores = {}
    
    def setup(self, layer_names: List[str]) -> None:
        """
        Set up hooks for the specified layers.
        
        Args:
            layer_names: List of layer names to analyze
        """
        self.cleanup() # Clear any existing hooks
        self.hooked_layers = set(layer_names)
        
        # Register hooks for specified layers
        hooks = register_hooks(
            self.model, 
            layer_names, 
            forward_hook=self._create_forward_hook,
            backward_hook=self._create_backward_hook
        )
        
        self.hooks.extend(hooks)
        logger.info(f"Registered hooks for {len(self.hooks)} layers")
    
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
                # Check if gradients are enabled
                if not attn_weights.requires_grad:
                    logger.warning(f"Attention weights for layer '{layer_name}' do not require grad. Gradient capture may fail.")
                
                # Store the weights tensor
                self.attention_weights[layer_name] = attn_weights
        
        return hook
    
    def _create_backward_hook(self, layer_name: str):
        """Create a backward hook for capturing attention gradients."""
        def hook(module, grad_input, grad_output):
            # Check if we captured weights for this layer in the forward pass
            if layer_name in self.attention_weights:
                attn_weights_tensor = self.attention_weights[layer_name]
                
                # Ensure tensor requires gradients
                if attn_weights_tensor.requires_grad:
                    # Define the tensor hook function
                    def _capture_grad(grad):
                        # Store the gradient (detached)
                        self.attention_grads[layer_name] = grad.detach()
                    
                    # Register the hook on the tensor
                    if layer_name not in self.tensor_hooks:
                        handle = attn_weights_tensor.register_hook(_capture_grad)
                        self.tensor_hooks[layer_name] = handle
        
        return hook
    
    def compute(self, inputs: Dict[str, torch.Tensor], target_indices: List[int]) -> Dict[str, Any]:
        """
        Compute saliency scores for the given inputs and targets.
        
        Args:
            inputs: Model inputs
            target_indices: Indices of target tokens to analyze
            
        Returns:
            Dictionary with saliency scores per layer
        """
        self.clear_results()
        
        if not target_indices:
            logger.warning("No target indices provided. Cannot compute saliency.")
            return {}
        
        # Process targets in batches for memory efficiency
        all_layers = list(self.hooked_layers)
        
        # Calculate loss function based on target tokens
        def compute_loss(outputs, target_idxs):
            loss = 0
            input_ids = inputs["input_ids"][0]  # Assuming batch size 1
            
            for target_idx in target_idxs:
                if target_idx >= outputs.logits.shape[1]:
                    logger.warning(f"Target index {target_idx} exceeds sequence length. Skipping.")
                    continue
                
                logits = outputs.logits[0, target_idx]
                target_token_id = input_ids[target_idx].item()
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                loss = loss - log_probs[target_token_id]
            
            return loss / len(target_idxs)  # Normalize by number of targets
        
        # Compute saliency for all layers at once if requested
        if self.batch_size >= len(all_layers):
            logger.info(f"Computing saliency for all {len(all_layers)} layers in a single pass")
            
            # Perform forward and backward pass
            self.model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = self.model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                    output_attentions=True,
                                    use_cache=False)
                
                loss = compute_loss(outputs, target_indices)
                
                if loss.requires_grad:
                    loss.backward()
                    # ★ NEW: Free memory immediately after backward pass
                    del outputs, loss
                    torch.cuda.empty_cache()
                else:
                    logger.warning("Loss doesn't require gradients. Check model setup.")
                    return {}
            
            # Compute saliency scores
            self._compute_saliency_scores()
            
        else:
            # Process layers in batches
            for batch_start in range(0, len(all_layers), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(all_layers))
                batch_layers = all_layers[batch_start:batch_end]
                
                logger.info(f"Computing saliency for layers {batch_start+1}-{batch_end} of {len(all_layers)}")
                
                # Setup hooks only for the current batch of layers
                self.cleanup()
                self.setup(batch_layers)
                
                # Perform forward and backward pass
                self.model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    outputs = self.model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                        output_attentions=True,
                                        use_cache=False)
                    
                    loss = compute_loss(outputs, target_indices)
                    
                    if loss.requires_grad:
                        loss.backward()
                        # ★ NEW: Free memory immediately after backward pass
                        del outputs, loss
                        torch.cuda.empty_cache()
                    else:
                        logger.warning("Loss doesn't require gradients for batch.")
                        continue
                
                # Compute saliency scores for this batch
                batch_scores = self._compute_saliency_scores()
                
                # Update the overall saliency scores
                self.saliency_scores.update(batch_scores)
                
                # Clean up memory
                if self.cpu_offload and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
        
        return {"saliency_scores": self.saliency_scores}
    
    def _compute_saliency_scores(self) -> Dict[str, torch.Tensor]:
        """
        Compute |attention * gradient| for all captured layers.
        
        Returns:
            Dictionary mapping layer names to saliency tensors
        """
        scores = {}
        captured_grad_layers = list(self.attention_grads.keys())
        
        for layer_name in captured_grad_layers:
            if layer_name in self.attention_weights:
                attn_weights = self.attention_weights[layer_name]
                grad = self.attention_grads[layer_name]
                
                # Ensure tensors are compatible
                if attn_weights.shape != grad.shape:
                    logger.warning(f"Shape mismatch for layer '{layer_name}'. Skipping.")
                    continue
                
                if attn_weights.device != grad.device:
                    try:
                        grad = grad.to(attn_weights.device)
                    except Exception as e:
                        logger.warning(f"Error moving gradient for layer '{layer_name}': {e}. Skipping.")
                        continue
                
                # Compute saliency: |Attention * Gradient|
                saliency = torch.abs(attn_weights.float() * grad.float())
                
                # Store the result, optionally offloading to CPU
                if self.cpu_offload:
                    scores[layer_name] = saliency.detach().cpu()
                else:
                    scores[layer_name] = saliency.detach()
                
                # Cleanup
                del self.attention_weights[layer_name]
                del self.attention_grads[layer_name]
            
            else:
                logger.warning(f"Gradient found for layer '{layer_name}', but no weights were captured.")
        
        # Update the instance's scores
        self.saliency_scores.update(scores)
        
        # Final cleanup
        self.attention_weights.clear()
        self.attention_grads.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return scores
    
    def clear_results(self) -> None:
        """Clear previous computation results."""
        self.attention_weights.clear()
        self.attention_grads.clear()
        self.saliency_scores.clear()
    
    def cleanup(self) -> None:
        """Remove all hooks and free resources."""
        # Remove tensor hooks
        for hook in self.tensor_hooks.values():
            hook.remove()
        self.tensor_hooks.clear()
        
        # Remove layer hooks
        remove_hooks(self.hooks)
        self.hooks.clear()
        
        # Clear cached data
        self.clear_results()
        self.clear_cache()
        
        # Additional cleanup
        self.hooked_layers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()