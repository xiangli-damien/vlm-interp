"""
Runtime utilities for VLM interpretability, including caching, hook management, I/O, and model utilities.
"""

from runtime.cache import TracingCache, _GlobalSalCache
from runtime.hooks.light_grad import LightAttnFn, GradAttnHook
from runtime.hooks.manager import TraceHookManager
from runtime.io import TraceIO
from runtime.model_utils import load_model, get_module_by_name
from runtime.selection import SelectionConfig, SelectionStrategy
from runtime.decode import TokenDecoder
from runtime.hooks.attention import SaveAttnHook

__all__ = [
    "TracingCache",
    "_GlobalSalCache",
    "LightAttnFn",
    "GradAttnHook",
    "TraceHookManager",
    "TraceIO",
    "load_model",
    "get_module_by_name",
    "SelectionConfig",
    "SelectionStrategy",
    "TokenDecoder",
    "SaveAttnHook",
]
