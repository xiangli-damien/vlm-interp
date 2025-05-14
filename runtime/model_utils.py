# runtime/model_utils.py
"""
Model handling and architecture analysis utilities.

Includes functions for:
- Loading VLM models (LLaVA-Next) with configuration options
- Finding modules within a model by name
- Identifying attention layers in the language model component
- Analyzing and summarizing the model's overall architecture
"""

import torch
import torch.nn as nn
import time
from typing import Optional, Tuple, Union, List, Dict, Any
from packaging import version
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

# Import necessary components from transformers
from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig
)

# Model ID mapping
MODEL_OPTIONS = {
    "mistral_7b": {
        "id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "name": "LLaVA-v1.6-Mistral-7B"
    },
    "vicuna_7b": {
         "id": "llava-hf/llava-v1.6-vicuna-7b-hf",
         "name": "LLaVA-v1.6-Vicuna-7B"
    },
    "llava_34b": {
        "id": "llava-hf/llava-v1.6-34b-hf",
        "name": "LLaVA-v1.6-34B"
    }
}

FLASH_MIN_VERSION = (2, 5)  # flash-attn >= 2.5 required for compatibility

def _is_flash_attn_supported() -> bool:
    """
    Returns True if flash-attn is installed and its version meets the minimum.
    """
    try:
        import flash_attn
        ver = tuple(int(v) for v in flash_attn.__version__.split('.')[:2])
        return ver >= FLASH_MIN_VERSION
    except Exception:
        return False

def _create_4bit_config() -> BitsAndBytesConfig:
    """
    Constructs a default 4-bit quantization config (nf4, double quant).
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def load_model(
    model_id: str,
    *,
    use_flash_attn: bool = False,
    load_in_4bit: bool = False,
    enable_gradients: bool = False,
    device_map: Optional[str] = "auto",
    **hf_kwargs: Dict[str, Any],
) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """
    Load a LLaVA-Next model + processor with optional flash-attention and 4-bit quantization.

    Args:
        model_id: Hugging Face model identifier.
        use_flash_attn: Enable flash-attn backend if available.
        load_in_4bit: Enable 4-bit quantization (requires CUDA).
        enable_gradients: If True, unfreeze attention projections for training.
        device_map: Device placement strategy (e.g., 'auto' or explicit dict).
        hf_kwargs: Additional kwargs forwarded to `from_pretrained`.

    Returns:
        (model, processor)
    """
    t0 = time.time()
    print(f"[INFO] Loading LLaVA-Next '{model_id}'")

    # 1. Processor
    processor = LlavaNextProcessor.from_pretrained(model_id)
    print("[INFO] Processor loaded")

    # 2. Attention implementation
    if use_flash_attn and torch.cuda.is_available() and _is_flash_attn_supported():
        attn_impl = "flash_attention_2"
        print("[INFO] Using flash-attention backend")
    else:
        attn_impl = "eager"
        if use_flash_attn:
            print("[WARNING] flash-attn unavailable or incompatible; using eager")

    # 3. Quantization config
    quant_config = None
    if load_in_4bit and torch.cuda.is_available():
        try:
            quant_config = _create_4bit_config()
            print("[INFO] 4-bit quantization enabled (nf4)")
        except Exception as e:
            print(f"[WARNING] 4-bit config failed ({e}); falling back to fp16")
    elif load_in_4bit:
        print("[WARNING] 4-bit quantization requires CUDA; ignoring")

    # 4. Assemble load args
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    effective_map = device_map if torch.cuda.is_available() else None
    load_args: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": effective_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": attn_impl,
        "quantization_config": quant_config,
    }
    load_args.update(hf_kwargs)

    print(f"[DEBUG] load_args = {{'dtype': {dtype}, 'device_map': {effective_map}, 'attn_impl': '{attn_impl}'}}")
    model = LlavaNextForConditionalGeneration.from_pretrained(model_id, **load_args)
    print("[INFO] Model weights loaded")

    # 5. Gradient settings
    if enable_gradients and not load_in_4bit:
        model.train()
        for name, param in model.named_parameters():
            if any(tag in name for tag in (".q_proj", ".k_proj", ".v_proj", ".o_proj")):
                param.requires_grad_(True)
        print("[INFO] Gradients enabled for attention projections only")
    else:
        model.eval()

    dt = time.time() - t0
    print(f"[INFO] Completed in {dt:.2f}s | dtype={model.dtype}")
    return model, processor
    

def get_module_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """
    Retrieves a submodule from a model using its fully qualified name.

    Args:
        model: The parent model instance
        name: The dot-separated path to th e target submodule

    Returns:
        The submodule if found, otherwise None
    """
    names = name.split('.')
    module: Union[nn.Module, nn.Sequential, nn.ModuleList] = model

    try:
        for n in names:
            if n.isdigit():
                # Index into ModuleList or Sequential
                module = module[int(n)]
            else:
                # Access attribute (submodule)
                module = getattr(module, n)
                
        # Ensure the final result is an nn.Module
        if isinstance(module, nn.Module):
             return module
        else:
             print(f"Warning: Path '{name}' leads to type {type(module)}, not nn.Module.")
             return None
    except (AttributeError, IndexError, TypeError):
        return None


def matches_pattern(name: str, pattern: str) -> bool:
    """
    Checks if a module name matches a simple pattern with a wildcard '*'.

    Args:
        name: The full module name
        pattern: A pattern containing potentially one or more '*' wildcards

    Returns:
        True if the name matches the pattern, False otherwise
    """
    pattern_parts = pattern.split('.')
    name_parts = name.split('.')

    # Lengths must match for the pattern to apply
    if len(name_parts) != len(pattern_parts):
        return False

    # Compare parts element-wise
    for pattern_part, name_part in zip(pattern_parts, name_parts):
        if pattern_part == '*':
            # Wildcard matches any corresponding part
            continue
        elif pattern_part != name_part:
            # Literal parts must match exactly
            return False

    # If all parts matched, the name fits the pattern
    return True


def get_llm_attention_layer_names(model: nn.Module) -> List[str]:
    """
    Extracts the names of likely attention modules within the language model
    component of a VLM.

    Args:
        model: A VLM model instance, expected to have a language_model attribute

    Returns:
        A list of module names identified as attention layers
    """
    attention_layer_names = []

    if not hasattr(model, 'language_model'):
        print("Warning: Model does not have a 'language_model' attribute. Cannot find attention layers.")
        return []

    # Common patterns for attention modules in Hugging Face transformer models
    patterns = [
        'language_model.model.layers.*.self_attn',      # Llama, Mistral, Vicuna style
        'language_model.transformer.h.*.attn',          # GPT-2 style
        'language_model.encoder.layer.*.attention.self',# BERT style
        'language_model.layers.*.attention',            # Some other architectures
    ]

    print("Searching for language model attention layers using patterns:")

    # Iterate through all named modules in the model
    for name, module in model.named_modules():
        # Check if the module name matches any of the defined patterns
        if any(matches_pattern(name, pattern) for pattern in patterns):
             # Basic check: Ensure it looks like an attention mechanism
             is_likely_attention = hasattr(module, 'q_proj') or hasattr(module, 'query') or hasattr(module, 'Wq')
             if is_likely_attention:
                attention_layer_names.append(name)

    if not attention_layer_names:
        print("Warning: No attention layers found matching the known patterns within model.language_model.")
    else:
        print(f"Found {len(attention_layer_names)} potential attention layer names in the language model.")

    return attention_layer_names


def freeze_non_attention_params(model):
    """
    Freeze all parameters except attention-related ones to save memory.
    
    Args:
        model: The model to freeze parameters for
    """
    frozen_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        # Only keep attention parameters for gradient computation
        if ".attn." not in name:  # Ensure qkv/out proj can still compute gradients
            param.requires_grad_(False)
            frozen_count += 1
    
    print(f"Frozen {frozen_count}/{total_count} parameters ({frozen_count/total_count:.1%}) to save memory")
    
    # Disable KV-cache to reduce memory usage
    if hasattr(model, 'config'):
        model.config.use_cache = False
        print("Disabled KV-cache to save memory")
        
    # Disable flash attention if present in LLaMA models
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        flash_disabled = 0
        for block in model.model.layers:
            if hasattr(block, 'self_attn') and hasattr(block.self_attn, '_flash_attn_'):
                block.self_attn._flash_attn_ = False
                flash_disabled += 1
        if flash_disabled > 0:
            print(f"Disabled flash attention in {flash_disabled} layers to ensure proper attention weight calculation")