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
            k: v.detach().clone()  # Make sure we have our own copy
            for k, v in inputs.items()
            if k not in ("past_key_values", "use_cache")
        }
        self.last_inputs["use_cache"] = False
        self.last_inputs["output_attentions"] = True
        
        # Store target indices for later use
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
        
        # Define loss function for backward pass
        def loss_fn(outputs):
            print(f"[DEBUG][SaliencyBackend] loss_fn called for layer {layer_idx}")
            logits = outputs.logits
            
            # Use float32 for more stable softmax computation
            if logits.dtype != torch.float32:
                logits = logits.float()
            
            # Apply log_softmax for numerical stability
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Get input IDs
            ids = self.last_inputs["input_ids"][0]
            
            # Compute loss for target tokens
            loss = 0
            for t in target_indices:
                # Skip tokens at position 0 (no previous prediction)
                if t > 0:
                    # Get probability of predicting token t given context up to t-1
                    loss = loss - log_probs[0, t-1, ids[t]]
            
            if loss.requires_grad:
                print(f"[DEBUG][SaliencyBackend] loss created with gradient tracking")
            else:
                print(f"[WARNING][SaliencyBackend] loss doesn't have requires_grad=True!")
                
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
            att = self.cache.get(layer_idx, "attn", self.device)
            grad = self.cache.get(layer_idx, "grad", self.device)
            
            if att is not None and grad is not None:
                # Make sure shapes match
                if att.shape == grad.shape:
                    # Compute saliency |attention * gradient|
                    sal = (att * grad).abs()
                    
                    # Average over batch and heads if needed
                    if sal.dim() == 4:  # [batch, head, seq, seq]
                        sal = sal.mean((0, 1))
                    elif sal.dim() == 3:  # [batch, seq, seq] or [head, seq, seq]
                        sal = sal.mean(0)
                    
                    # Cache result
                    print(f"[DEBUG][SaliencyBackend] Computed fallback saliency for layer {layer_idx}, shape: {tuple(sal.shape)}")
                    self.cache.set(layer_idx, "saliency", sal)
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
        # Compute saliency if not already cached
        print(f"[DEBUG][SaliencyBackend.trace_layer] entering layer {layer_idx}")
        if not self.cache.has(layer_idx, "saliency"):
            self._compute_single_layer(layer_idx, list(targets.keys()))
        
        # Get saliency scores from cache
        sal = self.cache.get(layer_idx, "saliency", self.device)
        print(f"[DEBUG][SaliencyBackend.trace_layer] saliency map for layer {layer_idx}: {'found' if sal is not None else 'None'}")
        
        if sal is None:
            print(f"Warning: Failed to compute saliency for layer {layer_idx}")
            return []
        
        # Process each target token
        results = []
        
        for tgt, w in targets.items():
            # Get saliency vector for this target (causal: only consider past tokens)
            if tgt < sal.shape[0]:
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
            else:
                print(f"[WARNING] Target index {tgt} is out of bounds for saliency shape {sal.shape}")
        
        # Apply layer-level pruning
        results = self._prune_layer(results, sel_cfg)
        
        # Clean up after use
        self.cache.clear_single(layer_idx, "saliency")
        
        return results