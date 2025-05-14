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
    min_keep: int = 2         # Minimum nodes to keep per target
    max_keep: int = 8        # Maximum nodes to keep per target
    min_keep_layer: int = 6   # Minimum nodes to keep per layer
    max_keep_layer: int = 500  # Maximum nodes to keep per layer


class SelectionStrategy:
    """
    Strategies for selecting and pruning tokens when tracing information flow.
    Handles both coverage-based and top-k selection methods.
    """

    @staticmethod
    def select_sources(scores: torch.Tensor, config: SelectionConfig) -> List[Tuple[int, float]]:
        """
        Select source tokens based on importance scores using a coverage threshold.
        Attempts to include non-zero indices when possible, while respecting the 
        coverage parameter.
        
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
            values = scores.cpu().abs().numpy()  # Take absolute values for importance
            indices = np.arange(len(values))
        else:
            values = np.abs(scores)  # Support numpy input
            indices = np.arange(len(values))
        
        # Identify non-zero indices
        nonzero_indices = np.where(indices > 0)[0]
        
        # Sort by importance scores (descending)
        sorted_order = np.argsort(-values)  # Negative for descending
        sorted_values = values[sorted_order]
        sorted_indices = indices[sorted_order]
        
        # Calculate cumulative coverage
        total = sorted_values.sum()
        if total <= 0:
            # Fallback for zero sum: return the highest value or first index
            return [(int(sorted_indices[0]), 1.0)] if len(sorted_indices) > 0 else []
        
        cumsum = np.cumsum(sorted_values) / total
        
        # Find cutoff based on coverage threshold
        cutoff_idx = np.searchsorted(cumsum, config.beta_target, side='right')
        
        # Apply min/max constraints
        cutoff_idx = min(max(cutoff_idx, config.min_keep), config.max_keep, len(sorted_indices))
        
        # Create initial selection
        selected_indices = sorted_indices[:cutoff_idx]
        selected_values = sorted_values[:cutoff_idx]
        
        # Check if we have non-zero indices in our selection
        has_nonzero = np.any(selected_indices > 0)
        
        # If we don't have any non-zero indices but they exist in the data,
        # force inclusion of the highest scoring non-zero indices
        if not has_nonzero and len(nonzero_indices) > 0:
            # Find highest scoring non-zero indices
            nonzero_scores = values[nonzero_indices]
            best_nonzero_idx = nonzero_indices[np.argmax(nonzero_scores)]
            
            # Include this index if not already selected
            if best_nonzero_idx not in selected_indices:
                # Append to selection
                selected_indices = np.append(selected_indices, best_nonzero_idx)
                selected_values = np.append(selected_values, values[best_nonzero_idx])
        
        # Re-normalize values based on selection
        selected_sum = selected_values.sum()
        if selected_sum > 0:
            normalized_values = selected_values / selected_sum
        else:
            normalized_values = np.ones_like(selected_values) / len(selected_values)
        
        # Create final list of (index, weight) tuples
        result = []
        for idx, val, norm_val in zip(selected_indices, values[selected_indices], normalized_values):
            # Use the original score sign but normalized magnitude
            sign = np.sign(scores[idx]) if isinstance(scores, torch.Tensor) else np.sign(scores[idx])
            result.append((int(idx), float(sign * norm_val)))
        
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
        sorted_sources = sorted(sources, key=lambda x: x[1], reverse=True)
        
        # Calculate cumulative weight sum
        weights = np.array([s[1] for s in sorted_sources])
        total_sum = weights.sum()
        
        if total_sum <= 0:
            return sorted_sources[:1] if sorted_sources else []
            
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
        total = sum(token_weights.values())
        if total <= 0:
            # If total is zero or negative, use uniform weights
            return {idx: 1.0 / len(token_weights) for idx in token_weights}
            
        return {idx: weight / total for idx, weight in token_weights.items()}