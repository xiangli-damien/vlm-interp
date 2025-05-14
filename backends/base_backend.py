# backends/base_backend.py
"""
Base interface for all semantic tracing backends.
Defines the common API for different analysis methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple

import torch

from runtime.cache import TracingCache
from runtime.selection import SelectionConfig


class BaseBackend(ABC):
    """
    Abstract base class for semantic tracing backends.
    Defines common functionality for different analysis approaches.
    """
    
    def __init__(self, model: torch.nn.Module, layer_names: List[str],
                 cache: TracingCache, device: torch.device):
        """
        Initialize the backend.
        
        Args:
            model: The model to analyze
            layer_names: Names of layers to trace
            cache: Tensor cache for data sharing
            device: Computation device
        """
        self.model = model
        self.layer_names = layer_names
        self.cache = cache
        self.device = device
    
    @abstractmethod
    def ensure_cache(self, inputs: Dict[str, torch.Tensor], target_indices: List[int]) -> None:
        """
        Ensure required data is cached before tracing.
        
        Args:
            inputs: Model input tensors
            target_indices: Indices of target tokens to analyze
        """
        pass
    
    @abstractmethod
    def trace_layer(self, layer_idx: int, 
                    target_tokens: Dict[int, float],
                    sel_cfg: SelectionConfig) -> List[Dict[str, Any]]:
        """
        Trace a specific layer, analyzing influences on target tokens.
        
        Args:
            layer_idx: Index of the layer to trace
            target_tokens: Dictionary mapping target token indices to weights
            sel_cfg: Configuration for token selection and pruning
            
        Returns:
            List of source token records with importance weights
        """
        pass
    
    def _prune_layer(self, sources: List[Dict[str, Any]], sel_cfg: SelectionConfig) -> List[Dict[str, Any]]:
        """
        Apply layer-level pruning to reduce the number of active source nodes.
        
        Args:
            sources: List of source token records
            sel_cfg: Selection configuration parameters
            
        Returns:
            Pruned list of source token records
        """
        if not sources:
            return []
        
        # Sort sources by absolute weight (descending)
        sources = sorted(sources, key=lambda x: abs(x["weight"]), reverse=True)
        
        # Calculate cumulative coverage
        weights = [abs(s["weight"]) for s in sources]
        total = sum(weights)
        
        if total <= 0:
            # If total weight is zero, keep just a minimum number
            return sources[:sel_cfg.min_keep_layer]
        
        # Calculate cumulative coverage
        cumsum = 0
        keep_idx = 0
        
        for i, w in enumerate(weights):
            cumsum += w
            coverage = cumsum / total
            
            # Find cutoff index based on coverage
            if coverage >= sel_cfg.beta_layer:
                keep_idx = i + 1  # Include the current index
                break
        
        # Apply min/max constraints
        keep_idx = max(keep_idx, sel_cfg.min_keep_layer)
        keep_idx = min(keep_idx, sel_cfg.max_keep_layer, len(sources))
        
        return sources[:keep_idx]