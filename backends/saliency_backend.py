# backends/saliency_backend.py
"""
Saliency-based semantic tracing backend.
Analyzes information flow using gradient-based saliency (attention * gradient).
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn.functional as F

from runtime.cache import TracingCache
from runtime.selection import SelectionConfig, SelectionStrategy
from runtime.hooks import TraceHookManager
from backends.base_backend import BaseBackend


class SaliencyBackend(BaseBackend):
    """
    Backend for analyzing information flow using gradient-based saliency.
    Computes |attention * gradient| for more accurate importance weights.
    """
    
    def ensure_cache(self, inputs: Dict[str, torch.Tensor], target_indices: List[int]) -> None:
        """
        Prepare inputs for lazy computation.
        Saliency computation is deferred until specific layers are requested.
        
        Args:
            inputs: Model input tensors
            target_indices: Indices of target tokens to analyze
        """
        # --- DEBUG LOG ---
        print(f"[DEBUG][SaliencyBackend] ensure_cache inputs.keys(): {list(inputs.keys())}")
        print(f"[DEBUG][SaliencyBackend] target_indices: {target_indices}")
        self.last_inputs = {
            k: v for k, v in inputs.items()
            if k not in ("past_key_values", "use_cache")
        }
        self.last_inputs["use_cache"] = False
        self.last_inputs["output_attentions"] = True
        self.target_indices = target_indices
    
    def _compute_single_layer(self, layer_idx: int, target_indices: List[int]) -> None:
        """
        Compute saliency scores for a specific layer.
        
        Args:
            layer_idx: Index of the layer to analyze
            target_indices: Indices of target tokens
        """
        # Set up hook manager
        print(f"[DEBUG][SaliencyBackend] _compute_single_layer layer {layer_idx}, targets {target_indices}")
        hook_mgr = TraceHookManager(self.model, self.cache, cpu_offload=self.cache.cpu_offload)
        
        # Make sure we have a valid layer name
        if layer_idx >= len(self.layer_names):
            print(f"[ERROR] Layer index {layer_idx} is out of range (max: {len(self.layer_names)-1})")
            return
            
        layer_name = self.layer_names[layer_idx]
        print(f"[DEBUG][SaliencyBackend] Registering hooks for layer '{layer_name}'")
        
        # Register both attention and grad hooks
        hook_mgr.add_layer(
            layer_name,
            capture=("grad", "attention"),
            layer_idx=layer_idx,
        )
        
        # Filter to valid targets (position > 0) for loss computation
        valid_targets = target_indices
        
        # Define loss function for backward pass
        def loss_fn(outputs):
            print(f"[DEBUG][SaliencyBackend] loss_fn called for layer {layer_idx}")
            logits = outputs.logits.float()
            
            # Use float32 for more stable softmax computation
            if logits.dtype != torch.float32:
                logits = logits.float()
            
            # Apply log_softmax for numerical stability
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get input IDs
            ids = self.last_inputs["input_ids"][0]
            
            # Initialize loss as a tensor with requires_grad=True
            # Create a dummy tensor for the case where no valid targets contribute
            dummy_tensor = torch.zeros(1, device=logits.device, dtype=logits.dtype, requires_grad=True)
            
            # Use all target indices directly (no filtering for position > 0)
            valid_targets = target_indices
            
            if valid_targets:
                # Compute loss for all target tokens (use t directly, not t-1)
                loss = sum(-log_probs[0, t, ids[t]] for t in valid_targets)
            else:
                # If no valid targets, use the dummy tensor
                print(f"[WARNING][SaliencyBackend] No valid targets found. Using dummy loss.")
                loss = dummy_tensor
            
            print(f"[DEBUG][SaliencyBackend] loss created with gradient tracking: {loss.requires_grad}")
            return loss
        
        # Install hooks and run with loss function
        hook_mgr.install()
        print(f"[DEBUG][SaliencyBackend] Running forward+backward for layer {layer_idx}")
        hook_mgr.run(self.last_inputs, loss_fn)
        print(f"[DEBUG][SaliencyBackend] Completed run for layer {layer_idx}")
        
        # Force garbage collection to free memory but keep cached data
        hook_mgr.clear(keep_cache=True)
        
        # Check if we have saliency or can compute it from attention and gradients
        has_sal = self.cache.has(layer_idx, "saliency")
        has_attn = self.cache.has(layer_idx, "attn")
        has_grad = self.cache.has(layer_idx, "grad")
        
        print(f"[DEBUG][SaliencyBackend] has_saliency={has_sal}, has_attn={has_attn}, has_grad={has_grad}")
        
        # Fallback: if LightAttnFn didn't work but we have both attention and gradients
        if not has_sal and has_attn and has_grad:
            print(f"[DEBUG][SaliencyBackend] Running fallback A×grad computation for layer {layer_idx}")
            
            # Get attention and gradient tensors
            attn = self.cache.get(layer_idx, "attn", self.device)
            grad = self.cache.get(layer_idx, "grad", self.device)
            
            if attn is not None and grad is not None:
                # Make sure shapes match
                if attn.shape == grad.shape:
                    # Compute saliency |attention * gradient|
                    saliency = (attn * grad).abs()  # Changed variable name from sal to saliency
                    
                    # Average over batch and heads if needed
                    if saliency.dim() == 4:  # [batch, head, seq, seq]
                        saliency = saliency.mean((0, 1))
                    elif saliency.dim() == 3:  # [batch, seq, seq] or [head, seq, seq]
                        saliency = saliency.mean(0)
                    
                    # Cache result
                    print(f"[DEBUG][SaliencyBackend] Computed fallback saliency for layer {layer_idx}, shape: {tuple(saliency.shape)}")
                    self.cache.set(layer_idx, "saliency", saliency)
                else:
                    print(f"[ERROR] Shape mismatch: attention {tuple(att.shape)} vs gradient {tuple(grad.shape)}")
            else:
                print(f"[WARNING] Missing tensors for fallback computation")
                
        # Always clean up individual tensors after computing saliency
        if has_attn:
            self.cache.clear_single(layer_idx, "attn")
        if has_grad:
            self.cache.clear_single(layer_idx, "grad")
    
    def trace_layer(self, layer_idx: int, 
                    targets: Dict[int, float],
                    sel_cfg: SelectionConfig) -> List[Dict[str, Any]]:
        """
        Trace a specific layer using saliency scores.
        
        Args:
            layer_idx: Index of the layer to trace
            targets: Dictionary mapping target token indices to weights
            sel_cfg: Configuration for token selection and pruning
            
        Returns:
            List of source token records with importance weights
        """
        print(f"[DEBUG][SaliencyBackend.trace_layer] entering layer {layer_idx}")
        
        # Cache original targets before filtering
        all_targets = targets.copy()
        
        # For loss computation, we need valid targets (index > 0)
        # This is because token 0 has no previous context to compute loss against
        valid_targets = {k: v for k, v in targets.items() if k > 0}
        
        # If we have no valid targets, we can still proceed with computation
        # but we need to handle the empty case for saliency computation
        compute_targets = valid_targets if valid_targets else all_targets
        
        # Compute saliency if not already cached
        if not self.cache.has(layer_idx, "saliency"):
            self._compute_single_layer(layer_idx, list(compute_targets.keys()))
        
        # Get saliency scores from cache
        saliency = self.cache.get(layer_idx, "saliency", self.device)
        print(f"[DEBUG][SaliencyBackend.trace_layer] saliency map for layer {layer_idx}: {'found' if saliency is not None else 'None'}")

        if saliency is None:
            print(f"[WARNING][SaliencyBackend.trace_layer] Failed to compute saliency for layer {layer_idx}")
            return []

        # Process EACH target token - including index 0 targets
        # This follows the original implementation which processed all targets
        results = []

        for tgt, w in all_targets.items():
            # Get saliency vector for this target (causal: only consider past tokens)
            if tgt < saliency.shape[0]:
                # For token 0, there are no previous tokens (empty saliency vector)
                vec = saliency[tgt, :tgt]
                
                # Skip if empty (e.g., first token)
                if vec.numel() == 0:
                    print(f"[INFO] Token {tgt} has no previous context, skipping.")
                    continue
                
                # Select important source tokens - no validation here
                srcs = SelectionStrategy.select_sources(vec, sel_cfg)
                
                # Record selected sources
                for i, sw in srcs:
                    results.append({
                        "index": i,             # Token index
                        "weight": sw * w,       # Global importance weight
                        "raw": sw,              # Raw importance score
                        "target": tgt,          # Which token this is a source for
                        "saliency_score": sw,   # Same as raw, but explicitly labeled
                        "type": "unknown"       # Will be filled later by the workflow
                    })
            else:
                print(f"[WARNING] Target index {tgt} is out of bounds for saliency shape {sal.shape}")
        
        # Apply layer-level pruning if we have results
        if results:
            results = self._prune_layer(results, sel_cfg)
        
        # Clean up after use
        self.cache.clear_single(layer_idx, "saliency")
        
        return results