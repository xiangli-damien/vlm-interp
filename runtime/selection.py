# runtime/selection.py
"""
Selection strategies for pruning and filtering token sources in interpretability analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np
import logging

# Configure logging
logger = logging.getLogger("selection_utils")
logger.setLevel(logging.INFO)

@dataclass
class SelectionConfig:
    """Configuration parameters for token selection and pruning."""
    beta_target: float = 0.8  # Coverage threshold for selecting source nodes
    beta_layer: float = 1.0   # Coverage threshold for pruning at layer level
    min_keep: int = 4         # Minimum nodes to keep per target
    max_keep: int = 8        # Maximum nodes to keep per target
    min_keep_layer: int = 8   # Minimum nodes to keep per layer
    max_keep_layer: int = 400  # Maximum nodes to keep per layer


class SelectionStrategy:
    """
    Strategies for selecting and pruning tokens when tracing information flow.
    Handles both coverage-based and top-k selection methods.
    """

    @staticmethod
    def select_sources(scores: torch.Tensor, config: SelectionConfig) -> List[Tuple[int, float]]:
        """
        Select source tokens based on importance scores using a coverage threshold.
        Handles both positive and negative importance values by selecting based on absolute magnitude.
        
        Args:
            scores: Tensor of importance scores for source tokens
            config: Selection configuration parameters
            
        Returns:
            List of (token_idx, weight) tuples for selected sources
        """
        if scores.numel() == 0:
            return []
        
        # Convert to numpy for easier manipulation
        if isinstance(scores, torch.Tensor):
            # Take absolute values for importance-based sorting, but preserve original values
            values = scores.cpu().abs().numpy()  
            indices = np.arange(len(values))
            original_values = scores.cpu().numpy()
        else:
            # Support numpy input
            values = np.abs(scores)  
            indices = np.arange(len(values))
            original_values = scores
        
        # Sort by importance scores (descending)
        sorted_order = np.argsort(-values)  # Negative for descending
        sorted_values = values[sorted_order]
        sorted_indices = indices[sorted_order]
        sorted_original = original_values[sorted_order]
        
        # Calculate cumulative coverage
        total = sorted_values.sum()
        
        # Handle edge case: near-zero total sum
        if total <= 1e-10:
            # Fallback: return a limited number of tokens with equal weights
            max_idx = min(config.min_keep, len(sorted_indices))
            return [(int(sorted_indices[i]), 1.0/max_idx) for i in range(max_idx)]
        
        # Calculate cumulative sum for coverage threshold
        cumsum = np.cumsum(sorted_values) / total
        
        # Find cutoff based on coverage threshold
        cutoff_idx = np.searchsorted(cumsum, config.beta_target, side='right')
        
        # Apply min/max constraints
        cutoff_idx = min(max(cutoff_idx, config.min_keep), config.max_keep, len(sorted_indices))
        
        # Get selected indices and values
        selected_indices = sorted_indices[:cutoff_idx]
        selected_original = sorted_original[:cutoff_idx]
        selected_values = sorted_values[:cutoff_idx]
        
        # Renormalize to make weights sum to 1.0 
        sum_selected = np.sum(selected_values)
        if sum_selected > 0:
            normalized_weights = selected_values / sum_selected
        else:
            normalized_weights = np.ones_like(selected_values) / len(selected_values)
        
        # Create result with original signs preserved but normalized magnitude
        # This preserves the directionality of influence while normalizing importance
        result = []
        for idx, orig_val, norm_w in zip(selected_indices, selected_original, normalized_weights):
            sign = np.sign(orig_val) if abs(orig_val) > 1e-10 else 1.0
            result.append((int(idx), float(sign * norm_w)))
        
        return result
    
    @staticmethod
    def prune_layer(sources: List[Tuple[int, float]], config: SelectionConfig) -> List[Tuple[int, float]]:
        """
        Prune sources at layer level using coverage-based threshold.
        
        Args:
            sources: List of (token_idx, weight) tuples for all sources at this layer
            config: Selection configuration parameters
            
        Returns:
            Pruned list of (token_idx, weight) tuples
        """
        if not sources:
            return []
            
        # Sort sources by weight
        sorted_sources = sorted(sources, key=lambda x: abs(x[1]), reverse=True)
        
        # Calculate cumulative weight sum
        weights = np.array([abs(s[1]) for s in sorted_sources])
        total_sum = weights.sum()
        
        if total_sum <= 0:
            return sorted_sources[:config.min_keep_layer] if sorted_sources else []
            
        cumsum = np.cumsum(weights) / total_sum
        
        # Find cutoff based on coverage threshold (beta_layer)
        cutoff_idx = np.searchsorted(cumsum, config.beta_layer, side='right')
        cutoff_idx = min(max(cutoff_idx, config.min_keep_layer), config.max_keep_layer, len(sorted_sources))
        
        return sorted_sources[:cutoff_idx]
    
    @staticmethod
    def renormalize(token_weights: Dict[int, float], config: Optional[SelectionConfig] = None, 
                   apply_layer_prune: bool = False) -> Dict[int, float]:
        """
        Renormalize weights to sum to 1.0 and optionally apply pruning.
        
        Args:
            token_weights: Dictionary mapping token indices to weights
            config: Optional selection configuration
            apply_layer_prune: Whether to apply layer-level pruning
            
        Returns:
            Dictionary of renormalized weights
        """
        if not token_weights:
            return {}
            
        # Apply pruning if requested and config provided
        if apply_layer_prune and config is not None:
            sources = [(idx, weight) for idx, weight in token_weights.items()]
            pruned_sources = SelectionStrategy.prune_layer(sources, config)
            token_weights = {idx: weight for idx, weight in pruned_sources}
            
            if not token_weights:
                return {}
                
        # Renormalize remaining weights
        total = sum(abs(w) for w in token_weights.values())
        if total <= 1e-10:
            # If total is zero or very small, use uniform weights
            return {idx: 1.0 / len(token_weights) for idx in token_weights}
            
        return {idx: weight / total for idx, weight in token_weights.items()}