# backends/saliency_backend.py
"""
Saliency-based semantic tracing backend.
Analyzes information flow using gradient-based saliency (attention * gradient).
"""

from typing import Dict, List, Optional, Any, Tuple
import torch

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
            k: v.detach()
            for k, v in inputs.items()
            if k not in ("past_key_values", "use_cache")
        }
        self.last_inputs["use_cache"] = False
        self.last_inputs["output_attentions"] = True
    
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
        
        print(f"[DEBUG][SaliencyBackend] Registering hooks for layer '{self.layer_names[layer_idx]}'")
        hook_mgr.add_layer(
            self.layer_names[layer_idx],
            capture=("grad", "attention"),
            layer_idx=layer_idx,
        )
        
        # Define loss function for backward pass
        def loss_fn(outputs):
            print(f"[DEBUG][SaliencyBackend] loss_fn called for layer {layer_idx}")
            logits = outputs.logits.float()
            logp = torch.log_softmax(logits, -1)
            ids = self.last_inputs["input_ids"][0]
            
            # Sum negative log probabilities for all target tokens
            # Corrected: Use t-1 to get probability of predicting token t
            loss = sum(-logp[0, t-1, ids[t]] for t in target_indices if t > 0)
            return loss
        
        # Install hooks, run forward/backward, and clear hooks
        hook_mgr.install()
        print(f"[DEBUG][SaliencyBackend] Running forward+backward for layer {layer_idx}")
        hook_mgr.run(self.last_inputs, loss_fn)
        print(f"[DEBUG][SaliencyBackend] Completed run for layer {layer_idx}")
        hook_mgr.clear(keep_cache=True)  # Keep cached saliency
        
        # Fallback: extremely rare – only if LightAttnFn path failed
        has_sal = self.cache.has(layer_idx, "saliency")
        has_attn = self.cache.has(layer_idx, "attn")
        # we don't store raw grad by default; if you do, check has_grad
        print(f"[DEBUG][SaliencyBackend] has_saliency={has_sal}, has_attn={has_attn}")
        if (not has_sal and has_attn and self.cache.has(layer_idx, "grad")):

            # Get attention and gradient tensors
            att = self.cache.get(layer_idx, "attn", self.device)
            grad = self.cache.get(layer_idx, "grad", self.device)
            
            # Compute saliency |attention * gradient|
            print(f"[DEBUG][SaliencyBackend] Running fallback A×grad for layer {layer_idx}")
            sal = (att * grad).abs()
            
            # Cache result and clean up intermediates
            self.cache.set(layer_idx, "saliency", sal)
            self.cache.clear_single(layer_idx, "attn")
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
        # Compute saliency if not already cached
        print(f"[DEBUG][SaliencyBackend.trace_layer] entering layer {layer_idx}")
        if not self.cache.has(layer_idx, "saliency"):
            self._compute_single_layer(layer_idx, list(targets.keys()))
        
        # Get saliency scores from cache
        sal = self.cache.get(layer_idx, "saliency", self.device)
        print(f"[DEBUG][SaliencyBackend.trace_layer] saliency map for layer {layer_idx}: { 'found' if sal is not None else 'None' }")
        
        if sal is None:
            print(f"Warning: Failed to compute saliency for layer {layer_idx}")
            return []
        
        # Average over batch and head dimensions if present
        if sal.ndim == 4:  # [batch, head, seq, seq]
            sal = sal.mean((0, 1))
        elif sal.ndim == 3:  # [batch, seq, seq] or [head, seq, seq]
            sal = sal.mean(0)
        
        # Process each target token
        results = []
        
        for tgt, w in targets.items():
            # Get saliency vector for this target (causal: only consider past tokens)
            vec = sal[tgt, :tgt]
            
            # Skip if empty (e.g., first token)
            if vec.numel() == 0:
                continue
            
            # Select important source tokens
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
        
        # Apply layer-level pruning
        results = self._prune_layer(results, sel_cfg)
        
        # After the saliency vector is consumed, make sure any stray
        # attention/grad tensors are gone as well (safety net).
        self.cache.clear_single(layer_idx, "saliency")
        self.cache.clear_single(layer_idx, "attn")
        self.cache.clear_single(layer_idx, "grad")
        
        return results