# base_backend.py
"""Base protocol for VLM interpretability backends."""

from typing import Dict, List, Any, Protocol

class Backend(Protocol):
    """Protocol defining the interface for all interpretability backends."""
    
    def ensure_cache(self, inputs: Dict[str, Any], **kwargs) -> None:
        """
        Ensure all necessary tensors are cached for later analysis.
        
        Args:
            inputs: Model input tensors
            **kwargs: Additional backend-specific parameters
        """
        ...
        
    def trace_layer(
        self,
        layer_idx: int,
        target_tokens: Dict[int, float],
        selection_config: Any,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Trace contributions at a specific layer.
        
        Args:
            layer_idx: Index of the layer to trace
            target_tokens: Dictionary mapping target token indices to their weights
            selection_config: Configuration for token selection
            **kwargs: Additional backend-specific parameters
            
        Returns:
            List of source information dictionaries
        """
        ...