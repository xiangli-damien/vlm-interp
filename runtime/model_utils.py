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
from typing import List, Dict, Tuple, Any, Optional, Union


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
    device_map: Optional[str] = "auto"
) -> Tuple[LlavaNextForConditionalGeneration, LlavaNextProcessor]:
    """
    Loads a LLaVA-Next model and processor with configurable options.

    Args:
        model_id: HuggingFace model ID
        use_flash_attn: Enable Flash Attention 2 if available
        load_in_4bit: Load the model using 4-bit quantization
        enable_gradients: Set requires_grad=True for model parameters
        device_map: Device map strategy for from_pretrained

    Returns:
        (model, processor): The loaded model and processor
    """
    start_time = time.time()
    print(f"Loading model and processor for: {model_id}...")

    # --- Load Processor ---
    try:
        processor = LlavaNextProcessor.from_pretrained(model_id)
        print("Processor loaded successfully.")
    except Exception as e:
        print(f"Error loading processor for {model_id}: {e}")
        raise RuntimeError(f"Failed to load processor for {model_id}") from e

    # --- Configure Model Loading ---
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    if use_flash_attn:
         print(f"Attempting to use attn_implementation='{attn_implementation}'")

    quantization_config = None
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    effective_device_map = device_map if torch.cuda.is_available() else None

    if load_in_4bit:
        if not torch.cuda.is_available():
            print("Warning: load_in_4bit=True requires CUDA. Ignoring quantization.")
        else:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_dtype = torch.float16
                print("Configured 4-bit quantization (nf4, float16 compute).")
                if effective_device_map is None:
                     print("Warning: 4-bit quantization typically requires device_map='auto'. Setting device_map='auto'.")
                     effective_device_map = "auto"

            except ImportError:
                print("Error: bitsandbytes library not found. Cannot use load_in_4bit=True.")
                raise
            except Exception as e:
                 print(f"Error configuring BitsAndBytesConfig: {e}")
                 raise

    # --- Load Model ---
    print(f"Loading model with dtype={model_dtype}, device_map='{effective_device_map}', attn='{attn_implementation}'...")
    try:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map=effective_device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True
        )
        print("Model loaded successfully.")
        print(f"  Model is on device(s): {model.device if effective_device_map is None else 'Multiple (device_map used)'}")

    except ImportError as e:
        print(f"ImportError during model loading: {e}. Ensure 'accelerate' is installed if using device_map or quantization, and 'flash-attn' if using flash_attention_2.")
        raise
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        if "out of memory" in str(e).lower():
             print("CUDA Out-of-Memory error detected. Try using 4-bit quantization (load_in_4bit=True) or ensure you have enough GPU RAM.")
        raise RuntimeError(f"Failed to load model {model_id}") from e

    # --- Post-Loading Configuration (Gradients) ---
    if enable_gradients:
        if load_in_4bit:
            print("Warning: Enabling gradients with 4-bit loaded model. This is experimental and may not work as expected or provide meaningful gradients.")
            try:
                 print("Attempting to set requires_grad=True on parameters...")
                 model.train()
                 for param in model.parameters():
                      param.requires_grad = True
                 print("Note: Full gradient enabling on 4-bit model is complex. Consider using PEFT library for fine-tuning.")
            except Exception as e:
                 print(f"Error enabling gradients on 4-bit model: {e}")
        else:
            print("Enabling gradients for all model parameters...")
            model.train()
            for param in model.parameters():
                 param.requires_grad = True
            print("Gradients enabled.")
    else:
         model.eval()

    end_time = time.time()
    print(f"Model '{model_id}' and processor loaded in {end_time - start_time:.2f} seconds.")

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