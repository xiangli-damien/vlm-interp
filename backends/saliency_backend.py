"""
Backend component for gradient-based saliency semantic tracing in VLM interpretability.
"""

import torch
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

from runtime.selection import SelectionStrategy, SelectionConfig
from enum import Enum
from runtime.hooks import TraceHookManager, LightAttnHook
from runtime.cache import global_saliency_cache
from runtime.model_utils import freeze_non_attention_params

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
        Trace saliency patterns for a specific layer with lazy computation and fallback mechanism.
        
        Args:
            layer_idx: Index of the layer to trace
            target_tokens: Dictionary mapping target token indices to their weights
            selection_config: Configuration for token selection
            batch_compute: Whether to compute saliency in batches (kept for API compatibility)
                
        Returns:
            List of source information dictionaries
        """
        # Lazy computation: Check if saliency exists, compute if needed
        if not self.cache.has(layer_idx, "saliency"):
            # Compute saliency only for this specific layer
            self.compute_batch_saliency(
                target_indices=list(target_tokens.keys()),
                inputs=self._last_inputs,
                layer_batch_size=1,  # Force single layer to avoid OOM
                offload_tensors=True,
                restrict_layers=[layer_idx]  # Only process this layer
            )
        
        # Get saliency scores after computation
        saliency_matrix = self.cache.get(layer_idx, "saliency", self.device)
        
        # FALLBACK: If saliency is still missing (which can happen if gradient hooks weren't triggered)
        # try to compute it directly from attention if available
        if saliency_matrix is None and self.cache.has(layer_idx, "attention"):
            logger.warning(f"Saliency computation failed for layer {layer_idx}, falling back to attention weights")
            attention_matrix = self.cache.get(layer_idx, "attention", self.device)
            
            # Use attention weights directly as a fallback
            # This happens if the layer doesn't have requires_grad=True for attention weights
            saliency_matrix = attention_matrix.abs()
            
            # Cache the computed fallback saliency
            self.cache.set(layer_idx, "saliency", saliency_matrix.detach().to(torch.float16).cpu())
        
        # Final check - if still no saliency, return empty list
        if saliency_matrix is None:
            logger.error(f"Failed to compute saliency for layer {layer_idx} - no attention or gradient available")
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
        Get saliency matrix for a layer, with improved memory efficiency.
        First checks global cache from light hooks before computing on demand.
        
        Args:
            layer_idx: Index of the layer to get saliency for
            target_indices: Token indices for which to compute saliency
            batch_compute: Whether to compute in batch (kept for compatibility)
            
        Returns:
            Saliency tensor or None if unavailable
        """
        # First check global saliency cache
        sal = global_saliency_cache.pop(layer_idx)
        if sal is not None:
            self.cache.set(layer_idx, "saliency", sal)   # CPU tensor
            return sal.to(self.device)
        
        # Check if saliency is already in the regular cache
        if self.cache.has(layer_idx, "saliency"):
            return self.cache.get(layer_idx, "saliency", self.device)

        # We have attention + gradient → build saliency now.
        if self.cache.has(layer_idx, "attention") and self.cache.has(layer_idx, "grad"):
            attn = self.cache.get(layer_idx, "attention", self.device)
            grad = self.cache.get(layer_idx, "grad", self.device)

            sal = torch.abs(attn.float() * grad.float())
            self.cache.set(layer_idx, "saliency", sal)

            # Free the bulky auxiliaries – they are no longer needed.
            self.cache.clear_single(layer_idx, "attention")
            self.cache.clear_single(layer_idx, "grad")
            return sal

        # Nothing cached – request a **single‑layer** computation.
        if self._last_inputs is not None and target_indices:
            self.compute_batch_saliency(
                target_indices   = target_indices,
                inputs           = self._last_inputs,
                layer_batch_size = 1,            # one block at a time
                offload_tensors  = True,
                restrict_layers  = [layer_idx],  # Only process this layer
            )
            if self.cache.has(layer_idx, "saliency"):
                return self.cache.get(layer_idx, "saliency", self.device)

        # Could not create a saliency tensor (e.g., no gradients available).
        return None


    def ensure_cache(
        self,
        inputs: Dict[str, torch.Tensor],
        target_indices: Optional[List[int]] = None,
        single_pass: bool = False
    ):  
        """
        Minimalist cache initialization - only store essential inputs to save memory.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Model input dictionary
            target_indices (List[int], optional): Token indices for saliency targets
            single_pass (bool): Flag maintained for API compatibility (ignored)
        """
        # Only store the essential inputs needed for computation
        self._last_inputs = {}
        
        # Always preserve input_ids and attention_mask which are needed for loss computation
        for key in ["input_ids", "attention_mask"]:
            if key in inputs:
                # Use detach() instead of clone() to save memory
                self._last_inputs[key] = inputs[key].detach()
        
        # Store other tensors only if small or necessary
        for k, v in inputs.items():
            if k not in self._last_inputs:  # Skip already copied essentials
                if isinstance(v, torch.Tensor):
                    if v.numel() < 1000:  # Only store small tensors
                        self._last_inputs[k] = v.detach()
                else:
                    # Non-tensor values can be stored directly
                    self._last_inputs[k] = v
        
        # Return immediately without pre-computing any saliency
        # Actual computation will be performed on-demand in trace_layer
        return

    def _run_hook_manager(self, hook_mgr, target_indices):
        """Helper method to run hook manager with appropriate loss function."""
        # Ensure consistent use of detach_after_forward=True
        if not hook_mgr.detach_after_forward:
            logger.warning("Inconsistent hook manager configuration: detach_after_forward=False may cause OOM issues")
            hook_mgr.detach_after_forward = True
        
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
        
        # Copy results from hook_mgr.cache to our main cache
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
        
    def compute_batch_saliency(
        self,
        target_indices: List[int],
        inputs: Dict[str, torch.Tensor],
        layer_batch_size: int = 1,  # Force single-layer processing
        offload_tensors: bool = True,
        restrict_layers: Optional[List[int]] = None,
    ) -> None:
        """
        Memory-optimized implementation that computes saliency for a small set of layers.
        Always processes one layer at a time for memory efficiency.
        
        Args:
            target_indices: List of token positions for gradient computation
            inputs: Model input tensors
            layer_batch_size: Number of layers to process at once (always 1)
            offload_tensors: Whether to offload tensors to CPU
            restrict_layers: Optional list of specific layers to process
        """
        if not target_indices:
            return  # Nothing to do
            
        # Always force layer_batch_size to 1 for memory efficiency
        layer_batch_size = 1
        
        # Validate target indices
        seq_len = inputs["input_ids"].size(1)
        valid_targets = [t for t in target_indices if 0 < t < seq_len]
        if not valid_targets:
            logger.warning("No valid target positions.")
            return
            
        # Determine which layers to process
        layers_to_run = (
            restrict_layers
            if restrict_layers is not None
            else list(range(len(self.layer_names)))
        )
        
        # Create half-precision view of inputs for memory efficiency
        minimal_inputs = {}
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32 and tensor.numel() > 1000:
                minimal_inputs[key] = tensor.to(torch.float16)
            else:
                minimal_inputs[key] = v
        
        # Process one layer at a time
        for layer_idx in layers_to_run:
            # Skip if already computed
            if self.cache.has(layer_idx, "saliency"):
                continue
                
            # Create hook manager for this layer only
            hook_mgr = TraceHookManager(
                self.model,
                cpu_offload=True,
                detach_after_forward=True,  # Detach to save memory
            )
            
            # Register only the current layer
            hook_mgr.add_layer(
                self.layer_names[layer_idx],
                capture=["attention", "grad"],
                layer_idx=layer_idx,
            )
            hook_mgr.install()
            
            # Use the existing _run_hook_manager method which properly handles cache merging
            self._run_hook_manager(hook_mgr, valid_targets)
            
            # Alternatively, if you prefer direct control, you can use:
            """
            # Define loss function
            def loss_fn(outputs):
                logits = outputs.logits
                # Use float32 for numerical stability
                logprob = torch.log_softmax(logits.float(), dim=-1)
                
                ids = minimal_inputs["input_ids"][0]
                loss = torch.zeros((), device=logprob.device, requires_grad=True)
                for t in valid_targets:
                    if t < logprob.size(1) and t < ids.size(0):
                        loss = loss - logprob[0, t - 1, ids[t].item()]
                return loss / max(len(valid_targets), 1)
            
            # Forward + backward with memory optimization
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                hook_mgr.run(minimal_inputs, loss_fn)
                
            # Copy results from hook_mgr.cache to our main cache
            snapshot = hook_mgr.snapshot()
            # Merge snapshots - critical step!
            self._merge_cache_snapshot(snapshot)
            """
            
            # Clear CUDA cache immediately
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Clean up regardless of result
            hook_mgr.clear(keep_cache=True)
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    def _merge_cache_snapshot(self, snapshot):
        """Merge a cache snapshot into the main cache."""
        # Copy all saliency tensors
        for layer_idx in range(len(self.layer_names)):
            if snapshot.has(layer_idx, "saliency"):
                self.cache.set(layer_idx, "saliency", snapshot.get(layer_idx, "saliency"))
            else:
                # fallback: cache raw attention & grad if saliency missing
                if snapshot.has(layer_idx, "attention"):
                    self.cache.set(layer_idx, "attention", snapshot.get(layer_idx, "attention"))
                if snapshot.has(layer_idx, "grad"):
                    self.cache.set(layer_idx, "grad", snapshot.get(layer_idx, "grad"))

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