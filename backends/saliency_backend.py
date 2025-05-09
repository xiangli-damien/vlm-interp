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
        # Extract saliency from target to all prior tokens (causal mask)
        saliency_vector = saliency_matrix[:, :, target_idx, :target_idx].mean(dim=(0, 1))
        return saliency_vector
    
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
        Ensure all necessary tensors are cached for later saliency analysis.

        If single_pass=True AND target_indices is provided, performs exactly one
        combined forward+backward over the entire model to capture attention+grad
        for every layer, computes saliency for each, and caches them all.

        Otherwise, if target_indices is given, falls back to compute_batch_saliency();
        if neither, just saves inputs for on-demand per-layer computation later.

        Args:
            inputs:         Model inputs (must include 'input_ids', etc.)
            target_indices: List of token indices to treat as loss targets
            single_pass:    Whether to do a single “forward+backward for all layers”
        """
        # 1) Clone & stash the original inputs so we can reuse them
        self._last_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # 2) True “single-pass” mode: compute all layers’ saliency in one shot
        if single_pass and target_indices:
            # a) Set up a hook manager to capture both attention & grads
            hook_mgr = TraceHookManager(
                self.model,
                cpu_offload=True,
                detach_after_forward=False
            )
            for layer_idx, layer_name in enumerate(self.layer_names):
                hook_mgr.add_layer(
                    layer_name,
                    capture=["attention", "grad"],
                    layer_idx=layer_idx
                )
            hook_mgr.install()

            # b) Build a combined loss over all requested target tokens
            def loss_fn(outputs):
                # outputs.logits: [B, seq_len, vocab_size]
                logprobs = torch.log_softmax(outputs.logits.float(), dim=-1)
                loss = None
                input_ids = self._last_inputs["input_ids"][0]
                for t in target_indices:
                    token_id = input_ids[t].item()
                    # use logits at position t-1 to predict token at t
                    this = -logprobs[0, t-1, token_id]
                    loss = this if loss is None else loss + this
                return loss

            # c) One forward+backward to populate all attention & grad hooks
            hook_mgr.run(self._last_inputs, loss_fn)
            hook_mgr.compute_saliency()

            # d) Push everything into our main cache
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

            # e) Clean up hooks & return to eval mode
            hook_mgr.clear()
            self.model.eval()

        # 3) Otherwise, if they gave us targets but didn’t want single_pass:
        elif target_indices:
            # compute saliency layer-by-layer in batches as before
            self.compute_batch_saliency(target_indices, self._last_inputs)

        # 4) Neither single-pass nor targets → just cache inputs for on-demand use
        else:
            # no-op beyond storing self._last_inputs
            pass
        
    def compute_batch_saliency(self, target_indices, inputs, layer_batch_size=4):
        """
        Compute saliency for multiple layers efficiently.
        Processes layers in batches to minimize hook reinstallation overhead.
        
        Args:
            target_indices: Indices of target tokens
            inputs: Model input tensors
            layer_batch_size: Number of layers to process in each batch
        """
        # Check if we have valid targets
        if not target_indices:
            return
            
        # Filter out invalid target indices
        max_idx = inputs["input_ids"].size(1) - 1
        valid_targets = [idx for idx in target_indices if 0 < idx <= max_idx]  # Skip idx 0
        
        if not valid_targets:
            return
        
        # Get all layer indices at once
        all_layer_indices = list(range(len(self.layer_names)))
        
        # Process layers in batches
        for batch_start in range(0, len(all_layer_indices), layer_batch_size):
            batch_end = min(batch_start + layer_batch_size, len(all_layer_indices))
            current_layer_indices = all_layer_indices[batch_start:batch_end]
            current_layer_names = [self.layer_names[i] for i in current_layer_indices]
            
            # Initialize hook manager for this batch of layers
            hook_mgr = TraceHookManager(self.model, cpu_offload=True, detach_after_forward=False)
            
            # Register all layers in the batch at once
            for idx, layer_name in zip(current_layer_indices, current_layer_names):
                hook_mgr.add_layer(
                    layer_name,
                    capture=["attention", "grad"],
                    layer_idx=idx
                )
            
            # Install hooks only once per batch
            hook_mgr.install()
            
            try:
                # Define loss function for all target tokens
                def loss_fn(outputs):
                    # Compute aggregated loss for all target tokens
                    logits = outputs.logits  # [B, seq_len, vocab_size]
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    
                    # Initialize total loss
                    total_loss = None
                    input_ids = inputs["input_ids"][0]  # Get the actual token IDs
                    
                    for target_idx in valid_targets:
                        # For each target, we want to compute the loss at the previous position
                        # that would predict this token
                        prev_idx = target_idx - 1
                        
                        # Get the true token ID at the target position
                        true_token_id = input_ids[target_idx].item()
                        
                        # Get the negative log probability of this token at the previous position
                        token_loss = -log_probs[0, prev_idx, true_token_id]
                        
                        # Add to total loss
                        if total_loss is None:
                            total_loss = token_loss
                        else:
                            total_loss = total_loss + token_loss
                    
                    return total_loss
                
                # Run model with hooks through hook manager
                hook_mgr.run(inputs, loss_fn)
                
                # Compute saliency scores
                hook_mgr.compute_saliency()
                
                # Get captured tensors from hook manager
                cache = hook_mgr.snapshot()
                
                # Transfer to our cache
                for layer_idx in current_layer_indices:
                    # First check for saliency
                    if cache.has(layer_idx, "saliency"):
                        self.cache.set(layer_idx, "saliency", cache.get(layer_idx, "saliency"))
                    # Then check for individual components
                    else:
                        if cache.has(layer_idx, "attention"):
                            self.cache.set(layer_idx, "attention", cache.get(layer_idx, "attention"))
                        if cache.has(layer_idx, "grad"):
                            self.cache.set(layer_idx, "grad", cache.get(layer_idx, "grad"))
                
            except Exception as e:
                logger.error(f"Error in batch saliency computation: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clear hooks
                hook_mgr.clear()
                
                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Reset model to eval mode
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