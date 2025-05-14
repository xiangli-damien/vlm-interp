"""
Unified hook exports for model interpretation and tracing.
"""

from runtime.hooks.manager import TraceHookManager
from runtime.hooks.light_grad import AttnGradFn, LightAttnHook

# Make LightGradHook an alias for LightAttnHook for backward compatibility
LightGradHook = LightAttnHook

__all__ = ["TraceHookManager", "LightAttnHook", "AttnGradFn", "LightGradHook"]