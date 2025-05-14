"""
Runtime utilities for VLM interpretability, including caching, hook management, I/O, and model utilities.
"""

from runtime.cache import TracingCache, global_saliency_cache
from runtime.hooks.light_grad import AttnGradFn, LightAttnHook
from runtime.hooks.manager import TraceHookManager
from runtime.io import TraceIO
from runtime.model_utils import load_model, get_module_by_name
from runtime.selection import SelectionConfig, SelectionStrategy
from runtime.decode import TokenDecoder

__all__ = [
    "TracingCache",
    "global_saliency_cache",
    "AttnGradFn",
    "LightAttnHook",
    "TraceHookManager",
    "TraceIO",
    "load_model",
    "get_module_by_name",
    "SelectionConfig",
    "SelectionStrategy",
    "TokenDecoder",
]
