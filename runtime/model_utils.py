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
from typing import Optional, Tuple, Union, List
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

def load_model(
    model_id: str,
    use_flash_attn: bool = False,
    load_in_4bit: bool = False,
    enable_gradients: bool = False,
    device_map: Optional[str] = "auto",
    # Add new optional parameters for quantization configuration
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_compute_dtype: torch.dtype = torch.float16,
    bnb_4bit_use_double_quant: bool = True,
    **kwargs  # Catch any additional parameters
) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """
    Loads a LLaVA-Next model and processor with flexible support for Flash Attention and 4-bit quantization.

    Args:
        model_id (str): HuggingFace model ID to load.
        use_flash_attn (bool): Whether to enable Flash Attention 2 (requires compatible GPU and library).
        load_in_4bit (bool): Whether to load the model using 4-bit quantization (requires bitsandbytes).
        enable_gradients (bool): If True, enables requires_grad on model parameters.
        device_map (Optional[str]): Device mapping for model loading. Defaults to "auto".
        bnb_4bit_quant_type (str): Quantization type for 4-bit loading ("nf4" or "fp4").
        bnb_4bit_compute_dtype (torch.dtype): Compute dtype for 4-bit quantization.
        bnb_4bit_use_double_quant (bool): Whether to use double quantization for 4-bit loading.
        **kwargs: Additional arguments to pass to the model's from_pretrained method.

    Returns:
        Tuple: (model, processor)
    """
    start_time = time.time()
    print(f"[INFO] Loading model and processor from: {model_id}...")

    # --- Load processor ---
    try:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        print("[✓] Processor loaded.")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to load processor: {e}") from e

    # --- Configure attention ---
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    if use_flash_attn:
        print("[INFO] Flash Attention 2 requested. Will use attn_implementation='flash_attention_2'.")

    # --- Configure precision and device ---
    quantization_config = None
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    effective_device_map = device_map if torch.cuda.is_available() else None

    # --- Configure 4-bit quantization (with fallback) ---
    if load_in_4bit and torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            from bitsandbytes import BitsAndBytesConfig

            if version.parse(bnb.__version__) < version.parse("0.41.1"):
                raise ImportError(f"bitsandbytes version {bnb.__version__} is too old. Please upgrade to >= 0.41.1.")

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype
            )
            print(f"[✓] 4-bit quantization enabled with {bnb_4bit_quant_type} format and {bnb_4bit_compute_dtype} compute dtype.")

        except ImportError as e:
            print(f"[WARN] bitsandbytes not available or outdated: {e}")
            print("[INFO] Falling back to float16 model...")
            load_in_4bit = False
            quantization_config = None
    elif load_in_4bit:
        print("[WARN] 4-bit quantization requires CUDA. Ignoring 'load_in_4bit=True'.")

    # --- Load model ---
    try:
        print(f"[INFO] Loading model with dtype={model_dtype}, device_map={effective_device_map}, attn='{attn_implementation}'...")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map=effective_device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
            **kwargs  # Pass any additional arguments
        )
        print("[✓] Model loaded.")
        if effective_device_map is None:
            print(f"[INFO] Model is on device: {model.device}")
        else:
            print(f"[INFO] Model loaded with device_map: {effective_device_map}")

    except ImportError as e:
        raise ImportError(f"[ERROR] ImportError during model loading: {e}\n"
                          "Check if 'accelerate' is installed, and 'flash-attn' if using flash_attention_2.")
    except Exception as e:
        if "out of memory" in str(e).lower():
            print("[OOM] CUDA out-of-memory. Try enabling load_in_4bit or reduce model size.")
        raise RuntimeError(f"[ERROR] Failed to load model: {e}") from e

    # --- Configure gradients ---
    if enable_gradients:
        print("[INFO] Enabling gradients for model parameters...")
        try:
            model.train()
            for param in model.parameters():
                param.requires_grad = True
            print("[✓] Gradients enabled.")
        except Exception as e:
            print(f"[WARN] Could not enable gradients: {e}")
    else:
        model.eval()

    elapsed = time.time() - start_time
    print(f"[✓] Model and processor loaded in {elapsed:.2f} seconds.")
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