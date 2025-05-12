"""
Backend component for gradient-based saliency semantic tracing in VLM interpretability.
"""

import torch
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from runtime.selection import SelectionStrategy, SelectionConfig
from enum import Enum
from runtime.hooks import TraceHookManager  # Use only this import, remove conflicting import

# Configure logging
logger = logging.getLogger("saliency_backend")
logger.setLevel(logging.INFO)

class TokenType(Enum):
    """Types of tokens in the input sequence."""
    TEXT = "text"
    IMAGE = "image" 
    GENERATED = "generated"
    UNKNOWN = "unknown"

class SaliencyBackend:
    """
    Backend component for gradient-based saliency semantic tracing.
    Computes and processes attention gradients to identify token contribution patterns.
    """
    
    def __init__(self, model: torch.nn.Module, layer_names: List[str], cache, device: torch.device):
        """
        Initialize the saliency backend.
        
        Args:
            model: The model being traced
            layer_names: List of attention layer names
            cache: Tracing cache instance
            device: Device for computation
        """
        self.model = model
        self.layer_names = layer_names
        self.cache = cache
        self.device = device
        self.layer_name_map = {i: name for i, name in enumerate(layer_names)}
        self._last_inputs = None   # Cache last inputs for re-computation if needed
        
    def trace_layer(self, layer_idx: int, target_tokens: Dict[int, float], 
                    selection_config: SelectionConfig, batch_compute: bool = True) -> List[Dict[str, Any]]:
        """
        Trace saliency patterns for a specific layer.
        
        Args:
            layer_idx: Index of the layer to trace
            target_tokens: Dictionary mapping target token indices to their weights
            selection_config: Configuration for token selection
            batch_compute: Whether to compute saliency in batches
            
        Returns:
            List of source information dictionaries
        """
        # Get saliency scores, computing if necessary
        saliency_matrix = self._get_saliency_matrix(layer_idx, list(target_tokens.keys()), batch_compute)
        if saliency_matrix is None:
            return []
            
        # Process saliency for each target token
        all_sources = []
        
        for target_idx, target_weight in target_tokens.items():
            if target_idx >= saliency_matrix.size(2):
                continue
                
            # Extract saliency from target to all prior tokens (causal mask)
            saliency_vector = self._prepare_saliency_vector(saliency_matrix, target_idx)
            
            # Skip if no valid saliency values
            if saliency_vector.numel() == 0:
                continue
                
            # Select important source tokens
            sources = SelectionStrategy.select_sources(saliency_vector, selection_config)
            
            # Create source info objects
            for src_idx, importance in sources:
                # Scale importance by target weight
                scaled_importance = importance * target_weight
                
                all_sources.append({
                    "index": int(src_idx),
                    "weight": float(scaled_importance),
                    "raw_score": float(importance),
                    "target": int(target_idx),
                    "type": TokenType.UNKNOWN.value
                })
                
        # Aggregate sources with the same index
        aggregated_sources = self._aggregate_sources(all_sources)
                
        # Apply layer-level pruning
        sources_list = [(s["index"], s["weight"]) for s in aggregated_sources.values()]
        pruned_sources = SelectionStrategy.prune_layer(sources_list, selection_config)
        
        # Create final source information list
        result = []
        for idx, weight in pruned_sources:
            source = aggregated_sources[idx]
            source["weight"] = weight  # Update with pruned weight
            result.append(source)
            
        return result
    
    def _prepare_saliency_vector(self, saliency_matrix: torch.Tensor, target_idx: int) -> torch.Tensor:
        """
        Extract and prepare the saliency vector for a specific target.
        Pure function that handles extraction with causal masking.
        
        Args:
            saliency_matrix: The full saliency matrix
            target_idx: The target token index
            
        Returns:
            Saliency vector for the target
        """
        vec = saliency_matrix[:, :, target_idx, :target_idx].mean(dim=(0, 1))
        if self.device.type == "cuda":
            vec = vec.to(self.device, non_blocking=True)
        return vec
    
    def _aggregate_sources(self, sources: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Aggregate sources with the same index by combining weights.
        Pure function to centralize aggregation logic.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            Dictionary mapping index to aggregated source
        """
        index_to_source = {}
        for source in sources:
            idx = source["index"]
            if idx not in index_to_source:
                index_to_source[idx] = source
            else:
                # Combine weights
                index_to_source[idx]["weight"] += source["weight"]
                # Keep the maximum raw score
                index_to_source[idx]["raw_score"] = max(
                    index_to_source[idx]["raw_score"], source["raw_score"]
                )
                
        return index_to_source
        
    def _get_saliency_matrix(self, layer_idx: int, target_indices: List[int], 
                            batch_compute: bool = True) -> Optional[torch.Tensor]:
        """
        Get saliency matrix for a specific layer, computing if needed.
        
        Args:
            layer_idx: Index of the layer
            target_indices: Indices of target tokens
            batch_compute: Whether to compute saliency in batches
            
        Returns:
            Saliency matrix tensor or None if unavailable
        """
        # Check if already in cache
        if self.cache.has(layer_idx, "saliency"):
            return self.cache.get(layer_idx, "saliency", self.device)
            
        # If not in cache but we have both attention and gradients, compute saliency
        if self.cache.has(layer_idx, "attention") and self.cache.has(layer_idx, "grad"):
            attention = self.cache.get(layer_idx, "attention", self.device)
            grad = self.cache.get(layer_idx, "grad", self.device)
            
            # Compute saliency using the product of attention and gradient
            saliency = torch.abs(attention.float() * grad.float())
            self.cache.set(layer_idx, "saliency", saliency)
            return saliency
            
        # Fallback: If we only have attention but not gradients, use attention as fallback
        elif self.cache.has(layer_idx, "attention"):
            logger.warning(f"No gradients for layer {layer_idx}, using attention as fallback")
            attention = self.cache.get(layer_idx, "attention", self.device)
            self.cache.set(layer_idx, "saliency", attention)
            return attention
        
        # If not in cache but we have inputs, compute it
        if self._last_inputs is not None and target_indices:
            # Compute for all target indices
            self.compute_batch_saliency(target_indices, self._last_inputs)
            
            # Check again after computation
            if self.cache.has(layer_idx, "saliency"):
                return self.cache.get(layer_idx, "saliency", self.device)
            elif self.cache.has(layer_idx, "attention"):
                logger.warning(f"No saliency for layer {layer_idx} after computation, using attention as fallback")
                attention = self.cache.get(layer_idx, "attention", self.device)
                self.cache.set(layer_idx, "saliency", attention)
                return attention
                
        # We can't compute saliency without inputs
        return None
        
    def ensure_cache(
        self,
        inputs: Dict[str, torch.Tensor],
        target_indices: Optional[List[int]] = None,
        single_pass: bool = False
    ):  
        """
        Ensure that all tensors required for saliency analysis are cached in advance.
        This is a memory-optimized implementation consistent with the legacy version.

        Args:
            inputs (Dict[str, torch.Tensor]): Model input dictionary (must include 'input_ids', etc.)
            target_indices (List[int], optional): Token indices to be used as saliency targets.
            single_pass (bool): Whether to compute saliency for all layers in one forward-backward pass.
                                Warning: this consumes a large amount of GPU memory.
        """
        # 1) Clone and store the original inputs for reuse during multiple passes
        self._last_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # 2) If target tokens are specified and single_pass is False, use memory-efficient batch mode
        if target_indices and not single_pass:
            logger.info(f"Using batch-mode saliency computation for {len(target_indices)} target tokens.")

            # For memory efficiency, compute one layer at a time (layer_batch_size=1)
            self.compute_batch_saliency(
                target_indices=target_indices,
                inputs=self._last_inputs,
                layer_batch_size=1
            )
            return

        # 3) If single_pass=True, attempt to compute all saliency maps in a single forward+backward pass
        if single_pass and target_indices:
            logger.info(f"Using single-pass mode for {len(target_indices)} target tokens.")
            logger.warning(
                "Single-pass mode may cause out-of-memory (OOM) errors. "
                "Switch to batch mode if this fails."
            )

            # NOTE: To ensure memory safety, we override the 'single pass' request
            # and still process one token at a time, across all layers.
            for target_idx in target_indices:
                # Compute saliency for each target independently across all layers
                self.compute_batch_saliency(
                    target_indices=[target_idx],
                    inputs=self._last_inputs,
                    layer_batch_size=1
                )

                # Force garbage collection between targets to free memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _run_hook_manager(self, hook_mgr, target_indices):
        """Helper method to run hook manager with appropriate loss function."""
        # Build a combined loss over all requested target tokens
        def loss_fn(outputs):
            # outputs.logits: [B, seq_len, vocab_size]
            logprobs = torch.log_softmax(outputs.logits.float(), dim=-1)
            loss = None
            input_ids = self._last_inputs["input_ids"][0]
            for t in target_indices:
                if t >= len(input_ids) or t <= 0:
                    continue
                token_id = input_ids[t].item()
                # use logits at position t-1 to predict token at t
                this = -logprobs[0, t-1, token_id]
                loss = this if loss is None else loss + this
            return loss * 1.0 if loss is not None else torch.tensor(0.0, requires_grad=True)

        # Run the model with hooks
        hook_mgr.run(self._last_inputs, loss_fn)
        
        # Compute saliency scores
        hook_mgr.compute_saliency()

        # Push everything into our main cache
        snapshot = hook_mgr.snapshot()
        for idx in range(len(self.layer_names)):
            if snapshot.has(idx, "saliency"):
                self.cache.set(idx, "saliency", snapshot.get(idx, "saliency"))
            else:
                # fallback: cache raw attention & grad if saliency missing
                if snapshot.has(idx, "attention"):
                    self.cache.set(idx, "attention", snapshot.get(idx, "attention"))
                if snapshot.has(idx, "grad"):
                    self.cache.set(idx, "grad", snapshot.get(idx, "grad"))
        
    def compute_batch_saliency(self, target_indices, inputs, layer_batch_size=1):
        """
        Ultra memory-optimized version: processes each token and each layer individually 
        using mixed precision and progressive offloading.
        """
        # Return early if there are no valid targets
        if not target_indices:
            return

        # Filter out invalid token indices
        max_idx = inputs["input_ids"].size(1) - 1
        valid_targets = [idx for idx in target_indices if 0 < idx <= max_idx]

        if not valid_targets:
            return

        # Force memory cleanup before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get list of all layer indices
        all_layer_indices = list(range(len(self.layer_names)))

        # Process each target token independently
        for target_idx in valid_targets:
            logger.info(f"Computing batch saliency for token {target_idx}")

            # Process each layer individually to minimize memory usage
            for layer_idx in all_layer_indices:
                layer_name = self.layer_names[layer_idx]
                logger.debug(f"Processing layer {layer_idx}/{len(all_layer_indices)}: {layer_name}")

                # Create a new hook manager for this specific layer
                hook_mgr = TraceHookManager(
                    self.model,
                    cpu_offload=True,
                    detach_after_forward=True  # IMPORTANT: always detach after forward
                )

                # Register only the current layer, not multiple layers
                hook_mgr.add_layer(
                    layer_name,
                    capture=["attention", "grad"],
                    layer_idx=layer_idx
                )

                # Install hooks
                hook_mgr.install()

                try:
                    # Define the loss function for the current target token only
                    def loss_fn(outputs):
                        # Extract logits from the model output
                        logits = outputs.logits  # [B, seq_len, vocab_size]

                        # Compute log probabilities in float32 for numerical stability
                        log_probs = torch.log_softmax(logits, dim=-1)

                        # Compute loss based on the token at the previous position
                        prev_idx = target_idx - 1
                        input_ids = inputs["input_ids"][0]

                        # Safety check: skip invalid indices
                        if target_idx >= len(input_ids):
                            return torch.tensor(0.0, device=logits.device, requires_grad=True)

                        true_token_id = input_ids[target_idx].item()

                        # Compute negative log-probability for the true token
                        token_loss = -log_probs[0, prev_idx, true_token_id]

                        return token_loss

                    # Prepare a minimal input dictionary with lower precision to save memory
                    minimal_inputs = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            if v.dtype == torch.float32 and v.numel() > 1000:
                                # Convert large float32 tensors to float16 to reduce memory
                                minimal_inputs[k] = v.to(torch.float16)
                            else:
                                # Use shallow copy for small tensors
                                minimal_inputs[k] = v.to(v.device, non_blocking=True)
                        else:
                            minimal_inputs[k] = v

                    # Enable mixed precision for forward pass
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                        # Run the model using the hook-enabled version
                        hook_mgr.run(minimal_inputs, loss_fn)

                    # Retrieve snapshot of all collected tensors
                    snapshot = hook_mgr.snapshot()

                    # Transfer saliency for the current layer into the main cache
                    if snapshot.has(layer_idx, "saliency"):
                        self.cache.set(layer_idx, "saliency", snapshot.get(layer_idx, "saliency"))
                    elif snapshot.has(layer_idx, "attention") and snapshot.has(layer_idx, "grad"):
                        # Manual compute if needed
                        attention = snapshot.get(layer_idx, "attention")
                        grad = snapshot.get(layer_idx, "grad")
                        saliency = torch.abs(attention * grad).to(torch.float16).cpu()
                        self.cache.set(layer_idx, "saliency", saliency)
                        # Clean up immediately
                        del attention, grad
                    else:
                        # Fallback: store components separately
                        if snapshot.has(layer_idx, "attention"):
                            self.cache.set(layer_idx, "attention", snapshot.get(layer_idx, "attention"))
                        if snapshot.has(layer_idx, "grad"):
                            self.cache.set(layer_idx, "grad", snapshot.get(layer_idx, "grad"))

                    # Force memory cleanup after each layer
                    del minimal_inputs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(f"Error computing batch saliency for token {target_idx}, layer {layer_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Clear all registered hooks
                    hook_mgr.clear()

                    # Force memory cleanup after hook teardown
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Ensure model is reset to eval mode after tracing
        self.model.eval()



    @staticmethod
    def mask_sources(saliency_matrix: torch.Tensor, 
                    text_indices: torch.Tensor, 
                    image_indices: torch.Tensor, 
                    target_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masks for different source token types in a saliency matrix.
        Handles empty indices by ensuring they are on the correct device.
        
        Args:
            saliency_matrix: The full saliency matrix
            text_indices: Indices of text tokens
            image_indices: Indices of image tokens  
            target_idx: Index of the target token
            
        Returns:
            Tuple of (text_mask, image_mask, generated_mask, causal_mask)
        """
        S = saliency_matrix.shape[0]  # Sequence length
        device = saliency_matrix.device
        
        # Ensure indices are on the correct device, handling empty tensors properly
        if text_indices.numel() > 0 and text_indices.device != device:
            text_indices = text_indices.to(device)
        elif text_indices.numel() == 0:
            text_indices = torch.empty(0, dtype=torch.long, device=device)
            
        if image_indices.numel() > 0 and image_indices.device != device:
            image_indices = image_indices.to(device)
        elif image_indices.numel() == 0:
            image_indices = torch.empty(0, dtype=torch.long, device=device)
        
        # Create boolean masks for each token type
        text_mask = torch.zeros(S, dtype=torch.bool, device=device)
        image_mask = torch.zeros(S, dtype=torch.bool, device=device)
        
        if len(text_indices) > 0: 
            text_mask[text_indices] = True
        if len(image_indices) > 0: 
            image_mask[image_indices] = True
        
        # Generated/Other mask includes tokens that are neither text nor image
        generated_mask = ~(text_mask | image_mask)
        
        # Causal mask: source index must be less than target index
        causal_mask = torch.arange(S, device=device) < target_idx
        
        return text_mask, image_mask, generated_mask, causal_mask
    
    @staticmethod
    def aggregate_flow(saliency_matrix: torch.Tensor, 
                      target_idx: int,
                      text_mask: torch.Tensor, 
                      image_mask: torch.Tensor, 
                      generated_mask: torch.Tensor,
                      causal_mask: torch.Tensor,
                      top_k_image_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Aggregate flow metrics from different token types to a target token.
        
        Args:
            saliency_matrix: The full saliency matrix
            target_idx: Index of the target token
            text_mask: Boolean mask for text tokens
            image_mask: Boolean mask for image tokens
            generated_mask: Boolean mask for generated tokens
            causal_mask: Boolean mask for tokens preceding the target
            top_k_image_tokens: If provided, only consider top-k image tokens
            
        Returns:
            Dictionary of aggregated flow metrics
        """
        metrics = {}
        
        # Extract target row from saliency matrix
        target_row = saliency_matrix[target_idx, :]
        
        # Calculate metrics for each source type
        total_flow_sum_causal = 0.0  # For normalization
        
        for prefix, source_mask in [("Stq", text_mask), ("Siq", image_mask), ("Sgq", generated_mask)]:
            # Combine source type mask with causal mask
            valid_sources_mask = source_mask & causal_mask
            count = valid_sources_mask.sum().item()
            metrics[f"{prefix}_count"] = count
            
            if count > 0:
                values = target_row[valid_sources_mask]
                flow_sum = values.sum().item()
                
                # For image tokens, optionally consider only top-k
                if prefix == "Siq" and top_k_image_tokens is not None and count > top_k_image_tokens:
                    # Get top-k values for image tokens
                    top_k_values, _ = torch.topk(values, k=min(top_k_image_tokens, count))
                    metrics[f"{prefix}_top{top_k_image_tokens}_mean"] = top_k_values.mean().item()
                    metrics[f"{prefix}_top{top_k_image_tokens}_sum"] = top_k_values.sum().item()
                
                # Store standard metrics regardless
                metrics[f"{prefix}_mean"] = values.mean().item()
                metrics[f"{prefix}_sum"] = flow_sum
                total_flow_sum_causal += flow_sum
            else:
                metrics[f"{prefix}_sum"] = 0.0
                metrics[f"{prefix}_mean"] = 0.0
                if prefix == "Siq" and top_k_image_tokens is not None:
                    metrics[f"{prefix}_top{top_k_image_tokens}_mean"] = 0.0
                    metrics[f"{prefix}_top{top_k_image_tokens}_sum"] = 0.0
        
        metrics["total_flow_sum_causal"] = total_flow_sum_causal
        
        # Add token counts to metrics
        metrics["token_counts"] = {
            "text": text_mask.sum().item(),
            "image": image_mask.sum().item(),
            "generated": generated_mask.sum().item(),
            "total": saliency_matrix.shape[0]
        }
        
        return metrics
    
    @staticmethod
    def normalize_flow(metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Normalize flow metrics to percentages of total flow.
        
        Args:
            metrics: Dictionary of aggregated flow metrics
            
        Returns:
            Dictionary of normalized flow metrics
        """
        normalized_metrics = dict(metrics)  # Make a copy
        
        total_flow_sum_causal = metrics.get("total_flow_sum_causal", 0.0)
        
        if total_flow_sum_causal > 1e-8:  # Avoid division by zero
            # Normalize standard metrics
            for prefix in ["Stq", "Siq", "Sgq"]:
                normalized_metrics[f"{prefix}_percent"] = (metrics[f"{prefix}_sum"] / total_flow_sum_causal) * 100.0
                
                # Normalize top-k metrics if they exist
                for k in [10, 20, 50]:  # Common top-k values
                    top_k_key = f"{prefix}_top{k}_sum"
                    if top_k_key in metrics:
                        normalized_metrics[f"{prefix}_top{k}_percent"] = (metrics[top_k_key] / total_flow_sum_causal) * 100.0
        else:
            # Set percentages to zero if total flow is zero
            for prefix in ["Stq", "Siq", "Sgq"]:
                normalized_metrics[f"{prefix}_percent"] = 0.0
                
                # Set top-k percentages to zero
                for k in [10, 20, 50]:  # Common top-k values
                    if f"{prefix}_top{k}_sum" in metrics:
                        normalized_metrics[f"{prefix}_top{k}_percent"] = 0.0
        
        return normalized_metrics
    
    @staticmethod
    def prepare_matrix(saliency_tensor: torch.Tensor) -> torch.Tensor:
        """
        Prepare and normalize a saliency tensor for flow analysis.
        
        Args:
            saliency_tensor: Raw saliency tensor
            
        Returns:
            Prepared 2D saliency matrix
        """
        # Average saliency over batch and head dimensions if present
        if saliency_tensor.ndim == 4:  # [B, H, S, S]
            saliency_matrix_2d = saliency_tensor.mean(dim=(0, 1)).float()
        elif saliency_tensor.ndim == 3:  # [B, S, S] or [H, S, S]
            saliency_matrix_2d = saliency_tensor.mean(dim=0).float()
        elif saliency_tensor.ndim == 2:  # [S, S]
            saliency_matrix_2d = saliency_tensor.float()
        else:
            raise ValueError(f"Unexpected saliency tensor shape: {saliency_tensor.shape}")
            
        return saliency_matrix_2d
    
    @staticmethod
    def calculate_saliency_scores(attention_weights: torch.Tensor, 
                                 attention_grads: torch.Tensor) -> torch.Tensor:
        """
        Calculate saliency scores using attention weights and gradients.
        Uses efficient einsum implementation for the calculation.
        
        Args:
            attention_weights: Attention weight tensor
            attention_grads: Attention gradient tensor
            
        Returns:
            Saliency score tensor
        """
        # Validate shapes match
        if attention_weights.shape != attention_grads.shape:
            raise ValueError(f"Shape mismatch: weights {attention_weights.shape}, gradients {attention_grads.shape}")
        
        # Compute |A * dA/dL| efficiently
        saliency = torch.abs(torch.einsum("bhij,bhij->bhij", attention_weights.float(), attention_grads.float()))
        
        return saliency