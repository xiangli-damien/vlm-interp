# backends/base_backend.py
"""
Base backend for all tracing methods.
Defines the common interface and utilities for backend implementations.
"""

from abc import ABC, abstractmethod
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set

logger = logging.getLogger(__name__)

class BaseBackend(ABC):
    """
    Abstract base class for all tracing backends.
    Defines the common interface that all backends must implement.
    """
    
    def __init__(self, model, device=None, cpu_offload=True):
        """
        Initialize the backend.
        
        Args:
            model: The model to analyze
            device: Device to run computations on
            cpu_offload: Whether to offload tensors to CPU when possible
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.cpu_offload = cpu_offload
        self.cache = {}
        
    @abstractmethod
    def setup(self, layer_names: List[str]) -> None:
        """
        Set up the backend for the specified layers.
        
        Args:
            layer_names: List of layer names to analyze
        """
        pass
    
    @abstractmethod
    def compute(self, inputs: Dict[str, torch.Tensor], target_indices: List[int]) -> Dict[str, Any]:
        """
        Compute the backend's specific outputs for the given inputs and targets.
        
        Args:
            inputs: Model inputs
            target_indices: Indices of target tokens to analyze
            
        Returns:
            Dictionary with backend-specific outputs
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the backend."""
        pass
    
    def clear_cache(self) -> None:
        """Clear the backend's cache."""
        self.cache.clear()
        
    def __enter__(self):
        """Support context manager interface."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        self.cleanup()