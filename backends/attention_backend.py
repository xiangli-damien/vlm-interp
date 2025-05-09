# backends/attention_backend.py
"""
Backend component for attention-based semantic tracing in VLM interpretability.
"""

import torch
from typing import Dict, List, Optional, Any
import logging
from runtime.selection import SelectionStrategy, SelectionConfig
from enum import Enum
from runtime.hooks import TraceHookManager

# Configure logging
logger = logging.getLogger("attention_backend")
logger.setLevel(logging.INFO)

class TokenType(Enum):
    """Types of tokens in the input sequence."""
    TEXT = "text"
    IMAGE = "image" 
    GENERATED = "generated"
    UNKNOWN = "unknown"

class AttentionBackend:
    """
    Backend component for attention-based semantic tracing.
    Extracts and processes attention weights to identify token contribution patterns.
    """
    
    def __init__(self, model: torch.nn.Module, layer_names: List[str], cache, device: torch.device):
        """
        Initialize the attention backend.
        
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
        self._last_inputs = None  # Cache the last inputs for re-computation
        
    def trace_layer(self, layer_idx: int, target_tokens: Dict[int, float], 
                    selection_config: SelectionConfig) -> List[Dict[str, Any]]:
        """
        Trace attention patterns for a specific layer.
        
        Args:
            layer_idx: Index of the layer to trace
            target_tokens: Dictionary mapping target token indices to their weights
            selection_config: Configuration for token selection
            
        Returns:
            List of source information dictionaries
        """
        # Get attention weights, caching if necessary
        attn_weights = self._get_attention_weights(layer_idx)
        if attn_weights is None:
            return []
            
        # Process attention for each target token
        all_sources = []
        
        for target_idx, target_weight in target_tokens.items():
            # Skip if target_idx is out of bounds
            if target_idx >= attn_weights.size(2):
                continue
                
            # Extract attention from target to all prior tokens (causal mask)
            attn_vector = attn_weights[:, :, target_idx, :target_idx].mean(dim=(0, 1))
            
            # Skip if no valid attention values
            if attn_vector.numel() == 0:
                continue
                
            # Select important source tokens
            sources = SelectionStrategy.select_sources(attn_vector, selection_config)
            
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
        index_to_source = {}
        for source in all_sources:
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
                
        # Apply layer-level pruning
        sources_list = [(s["index"], s["weight"]) for s in index_to_source.values()]
        pruned_sources = SelectionStrategy.prune_layer(sources_list, selection_config)
        
        # Create final source information list
        result = []
        for idx, weight in pruned_sources:
            source = index_to_source[idx]
            source["weight"] = weight  # Update with pruned weight
            result.append(source)
            
        return result
        
    def _get_attention_weights(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Get attention weights for a specific layer, using cache or computing if needed.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            Attention weights tensor or None if unavailable
        """
        # Check if already in cache
        if self.cache.has(layer_idx, "attention"):
            return self.cache.get(layer_idx, "attention", self.device)
        
        # If not in cache but we have inputs, compute it
        if self._last_inputs is not None:
            self.ensure_cache(self._last_inputs)
            
            # Check again after computation
            if self.cache.has(layer_idx, "attention"):
                return self.cache.get(layer_idx, "attention", self.device)
        
        # Still not in cache, return None
        return None
    
    def ensure_cache(self, inputs: Dict[str, torch.Tensor]) -> None:
        """
        Ensure all necessary tensors are cached for later analysis.
        
        Args:
            inputs: Model input tensors
        """
        # Cache the inputs for potential reuse
        self._last_inputs = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                           for k, v in inputs.items()}
        
        # Capture attention weights in a single forward pass
        self._ensure_all_attention_weights(inputs)
        
    def _ensure_all_attention_weights(self, inputs: Dict[str, torch.Tensor]) -> None:
        """
        Run a forward pass to ensure all attention weights are cached.
        
        Args:
            inputs: Model input tensors
        """
        
        # Initialize hook manager
        hook_mgr = TraceHookManager(self.model, cpu_offload=True)
        
        # Register layers for attention capture
        for layer_idx, layer_name in self.layer_name_map.items():
            hook_mgr.add_layer(
                layer_name,
                capture=["attention"],
                layer_idx=layer_idx
            )
        
        # Set model to eval mode
        self.model.eval()
        
        try:
            # Run forward pass with attention output
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True,
                    return_dict=True
                )
                
            # Check if we got attention weights directly
            if hasattr(outputs, 'attentions') and outputs.attentions:
                # Store attention weights for each layer
                for layer_idx, attn_tensor in enumerate(outputs.attentions):
                    if layer_idx < len(self.layer_names):
                        self.cache.set(layer_idx, "attention", attn_tensor)
            else:
                # If no attentions in outputs, use the hook manager
                # Run model with hooks
                hook_mgr.run(inputs)
                
                # Get captured tensors
                cache = hook_mgr.snapshot()
                
                # Transfer to our cache
                for layer_idx in range(len(self.layer_names)):
                    if cache.has(layer_idx, "attention"):
                        self.cache.set(layer_idx, "attention", cache.get(layer_idx, "attention"))
        
        except Exception as e:
            logger.error(f"Error capturing attention weights: {e}")
        finally:
            # Clean up hooks
            hook_mgr.clear()
            
            # Ensure model is back in eval mode
            self.model.eval()