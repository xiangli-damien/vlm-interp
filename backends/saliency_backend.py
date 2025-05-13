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
        Trace saliency patterns for a specific layer with lazy computation.
        
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
        
    def _get_saliency_matrix(  # noqa: C901 – complexity is intentional for clarity
        self,
        layer_idx: int,
        target_indices: List[int],
        batch_compute: bool = True,          # kept for API compatibility – no longer used
    ) -> Optional[torch.Tensor]:
        """
        Return a *saliency tensor* for ``layer_idx`` – computing it on‑demand
        while keeping GPU memory low.

        The search order is:

        1. **Hit:** tensor already cached → immediately return it.
        2. **Derive:** both *attention* **and** *gradient* are cached →
           compute ``|A * dA/dL|``; cache; delete the auxiliaries.
        3. **Miss:** nothing cached → call :py:meth:`compute_batch_saliency`
           *restricted to this single layer*; look again; fallback to ``None``.

        Parameters
        ----------
        layer_idx:
            Index of the transformer block whose saliency we want.
        target_indices:
            Sequence positions whose prediction loss should be used to create
            gradients.  They are forwarded unmodified to the batch routine
            when a computation is required.
        batch_compute:
            Ignored – retained only so external callers do not break.

        Returns
        -------
        torch.Tensor | None
            The 4‑D tensor ``[B, H, S, S]`` on *self.device* or *None* when
            saliency cannot be produced.
        """
        # ------------------------------------------------------------------ #
        # 1) Fast path – already have the answer.                            #
        # ------------------------------------------------------------------ #
        if self.cache.has(layer_idx, "saliency"):
            return self.cache.get(layer_idx, "saliency", self.device)

        # ------------------------------------------------------------------ #
        # 2) We have attention + gradient → build saliency now.              #
        # ------------------------------------------------------------------ #
        if self.cache.has(layer_idx, "attention") and self.cache.has(layer_idx, "grad"):
            attn = self.cache.get(layer_idx, "attention", self.device)
            grad = self.cache.get(layer_idx, "grad",      self.device)

            sal  = torch.abs(attn.float() * grad.float())
            self.cache.set(layer_idx, "saliency", sal)

            # Free the bulky auxiliaries – they are no longer needed.
            self.cache.clear_single(layer_idx, "attention")
            self.cache.clear_single(layer_idx, "grad")
            return sal

        # ------------------------------------------------------------------ #
        # 3) Nothing cached – request a **single‑layer** computation.        #
        # ------------------------------------------------------------------ #
        if self._last_inputs is not None and target_indices:
            self.compute_batch_saliency(
                target_indices   = target_indices,
                inputs           = self._last_inputs,
                layer_batch_size = 1,            # one block at a time
                offload_tensors  = True,
                restrict_layers  = [layer_idx],  # ← ***new***
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
        Minimalist cache initialization - only store inputs without pre-computing saliency.
        This avoids the expensive computation that was causing OOM errors.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Model input dictionary
            target_indices (List[int], optional): Token indices for saliency targets (not used for pre-computation)
            single_pass (bool): Flag maintained for API compatibility (ignored)
        """
        # Simply clone and store the original inputs for reuse during layer-by-layer computation
        self._last_inputs = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        
        # Return immediately without pre-computing any saliency
        # Actual computation will be performed on-demand in trace_layer
        return

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
        
    def compute_batch_saliency(
        self,
        target_indices: List[int],
        inputs: Dict[str, torch.Tensor],
        layer_batch_size: int = 1,
        offload_tensors: bool = True,
        restrict_layers: Optional[List[int]] = None,
    ) -> None:
        """
        Memory-optimized implementation that computes saliency for a small set of layers.
        Processes one layer at a time by default to minimize memory usage.

        Parameters
        ----------
        target_indices:
            List of sequence positions that act as "prediction locations" for
            the self-supervised loss used to obtain attention gradients.
        inputs:
            The pre-built input dictionary used for normal decoding.
            Tensors are referenced, not copied.
        layer_batch_size:
            Number of transformer blocks to hook simultaneously.
            Default: 1 (recommended for memory efficiency)
        offload_tensors:
            If True, tensors are converted to float16 and moved to CPU
            immediately after creation. Default: True
        restrict_layers:
            Optional whitelist of layer indices to process. Default: process
            all layers
        """
        if not target_indices:
            return  # nothing to do

        # ─────────────────── validate target indices ──────────────────── #
        seq_len = inputs["input_ids"].size(1)
        valid_targets = [t for t in target_indices if 0 < t < seq_len]
        if not valid_targets:
            logger.warning("compute_batch_saliency: no valid target positions.")
            return

        layers_to_run = (
            restrict_layers
            if restrict_layers is not None
            else list(range(len(self.layer_names)))
        )

        # Force layer_batch_size to 1 for maximum memory efficiency
        # (This is a key change to prevent OOM)
        layer_batch_size = 1

        # Build a *view* of `inputs` that uses half precision where possible
        minimal_inputs: Dict[str, torch.Tensor] = {}
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32 and tensor.numel() > 1000:
                minimal_inputs[key] = tensor.to(torch.float16)
            else:
                minimal_inputs[key] = tensor

        # ───────────────────── layer-by-layer processing ──────────────────── #
        for batch_start in range(0, len(layers_to_run), layer_batch_size):
            batch_layers = layers_to_run[batch_start : batch_start + layer_batch_size]

            if all(self.cache.has(l, "saliency") for l in batch_layers):
                continue  # already done

            # Create a new hook manager for each layer batch
            hook_mgr = TraceHookManager(
                self.model,
                cpu_offload=True,
                detach_after_forward=True,  # Crucial: detach tensors immediately
            )

            # Register only the layers we want to process now
            for l in batch_layers:
                hook_mgr.add_layer(
                    self.layer_names[l],
                    capture=["attention", "grad"],
                    layer_idx=l,
                )
            hook_mgr.install()

            # Loss = negative log-probability of requested targets
            def loss_fn(outputs):
                logits = outputs.logits
                # Use float32 for numerical stability but keep on same device
                logprob = torch.log_softmax(logits.float(), dim=-1)

                ids = minimal_inputs["input_ids"][0]
                loss = torch.zeros((), device=logprob.device, requires_grad=True)
                for t in valid_targets:
                    loss = loss - logprob[0, t - 1, ids[t].item()]
                return loss / max(len(valid_targets), 1)

            # Forward + backward (gradients captured by hooks)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                result = hook_mgr.run(minimal_inputs, loss_fn)
            
            # Clear CUDA cache immediately after backward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            if isinstance(result, dict) and result.get("error"):
                logger.warning(f"Hook-run aborted: {result['error']}")
                hook_mgr.clear(keep_cache=True)  # Don't clear the cache
                continue

            # Saliency tensors already computed in gradient hooks, no need to call compute_saliency again

            # Clear hooks but keep the cache
            hook_mgr.clear(keep_cache=True)

            # Force aggressive memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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