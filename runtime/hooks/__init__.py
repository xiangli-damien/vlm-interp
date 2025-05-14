"""
Hook modules for model introspection.
Provides components for capturing activations and gradients with minimal memory impact.
"""

from runtime.hooks.manager import TraceHookManager
from runtime.hooks.light_grad import GradAttnHook, LightAttnFn
from runtime.hooks.hidden import HiddenHook
from runtime.hooks.attention import SaveAttnHook

__all__ = [
    'TraceHookManager',
    'GradAttnHook',
    'LightAttnFn',
    'HiddenHook',
    'SaveAttnHook'
]