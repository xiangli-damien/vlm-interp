"""
Prompt, token, and high-level input helpers.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Union, Optional
import torch
from PIL import Image
import logging

from .image import load_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    # prompt
    "llavanext_build_conversation",
    "build_formatted_prompt",
    # tokens
    "get_token_masks",
    "get_token_indices",
    "get_image_token_spans",
    # end‑to‑end helper
    "prepare_inputs",
]

# ---------------------------------------------------------------------------#
# Prompt helpers
# ---------------------------------------------------------------------------#


def llavanext_build_conversation(
    prompt: str, *, use_dict_format: bool = True
) -> Union[str, List[Dict[str, Any]]]:
    """
    Return a minimal chat structure with one image placeholder.
    
    Args:
        prompt: The text prompt
        use_dict_format: Whether to use dict format (True) or string format (False)
        
    Returns:
        Chat structure as dict list or string
    """
    if use_dict_format:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    return prompt


def build_formatted_prompt(processor, prompt: str) -> str:
    """
    Apply the processor's chat template or fall back to a simple pattern.
    
    Args:
        processor: The model processor
        prompt: The text prompt
        
    Returns:
        Formatted prompt string with image token
    """
    convo = llavanext_build_conversation(prompt)
    try:
        return processor.apply_chat_template(convo, add_generation_prompt=True)
    except Exception:
        image_token = getattr(processor, "image_token", "<image>")
        return f"USER: {image_token}\n{prompt} ASSISTANT:"


# ---------------------------------------------------------------------------#
# Token helpers
# ---------------------------------------------------------------------------#

import torch
import torch.nn.functional as F


def get_token_masks(
    input_ids: torch.Tensor,
    image_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates boolean masks identifying text and image tokens in the input sequence.

    Args:
        input_ids: Tensor of shape [1, seq_len]
        image_token_id: Token ID for image tokens

    Returns:
        (text_mask, image_mask): Boolean tensors of shape [seq_len]
    """
    if input_ids.ndim != 2 or input_ids.size(0) != 1:
        raise ValueError(f"Expected input_ids with shape [1, seq_len], but got {input_ids.shape}")
    
    flat = input_ids[0]
    image_mask = flat == image_token_id
    text_mask = ~image_mask
    return text_mask, image_mask


def get_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int,
    debug: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns index tensors for text and image tokens.

    Args:
        input_ids: Tensor of shape [1, seq_len]
        image_token_id: Token ID for image tokens
        debug: Whether to print diagnostic information

    Returns:
        (text_indices, image_indices): 1D index tensors
    """
    text_mask, image_mask = get_token_masks(input_ids, image_token_id)
    text_indices = torch.where(text_mask)[0]
    image_indices = torch.where(image_mask)[0]

    if debug:
        print(f"[get_token_indices] Found {len(text_indices)} text tokens and {len(image_indices)} image tokens.")

    return text_indices, image_indices


def get_image_token_spans(
    input_ids: torch.Tensor,
    image_token_id: int,
    debug: bool = False
) -> List[Tuple[int, int]]:
    """
    Finds contiguous spans of image tokens (start and end indices inclusive).

    Args:
        input_ids: Tensor of shape [1, seq_len]
        image_token_id: Token ID used for image tokens
        debug: Whether to print span detection details

    Returns:
        A list of (start_idx, end_idx) tuples for contiguous image token spans
    """
    _, image_mask = get_token_masks(input_ids, image_token_id)

    num_image_tokens = image_mask.sum().item()
    if num_image_tokens == 0:
        return []

    # Pad to detect boundaries at the edges
    padded_mask = F.pad(image_mask, (1, 1), value=False).int()
    diff = padded_mask[1:] - padded_mask[:-1]

    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0] - 1  # inclusive end

    spans = []
    for s, e in zip(starts, ends):
        if s > e:
            if debug:
                print(f"[get_image_token_spans] Warning: Invalid span detected: start {s.item()} > end {e.item()}")
            continue
        spans.append((int(s), int(e)))
        if debug:
            print(f"[get_image_token_spans] Span detected: ({int(s)}, {int(e)}), length: {int(e - s + 1)}")

    if not spans and debug:
        print("[get_image_token_spans] Warning: Image tokens present, but no valid contiguous spans found.")

    return spans


# ---------------------------------------------------------------------------#
# High-level single function
# ---------------------------------------------------------------------------#


def prepare_inputs(
    model,
    processor,
    image: Union[str, Image.Image],
    prompt: str,
    resize_to: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Prepare model inputs from raw data.
    
    Args:
        model: The model to prepare inputs for
        processor: The model's processor
        image: PIL image, path, or URL
        prompt: Text prompt
        resize_to: Optional size to resize image to
        
    Returns:
        Dictionary with all necessary inputs and metadata
    """
    # 1. Load and process image
    pil_img = image if isinstance(image, Image.Image) else load_image(
        image, resize_to=resize_to
    )
    
    # 2. Format prompt
    if hasattr(processor, "apply_chat_template"):
        # Modern HF processors use chat templates
        conversation = llavanext_build_conversation(prompt)
        formatted_prompt = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True,
            return_tensors=None
        )
    else:
        # Fallback formatting
        formatted_prompt = f"USER: {prompt}\nASSISTANT:"
    
    # 3. Process inputs
    inputs = processor(
        images=pil_img,
        text=formatted_prompt,
        return_tensors="pt"
    )
    
    # 4. Move to model device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 5. Get image token ID
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        try:
            image_token_id = processor.tokenizer.convert_tokens_to_ids(
                getattr(processor, "image_token", "<image>")
            )
        except Exception:
            image_token_id = 32000  # fallback
            logger.warning("Falling back to default image token ID 32000")
    
    # 6. Get token indices
    text_indices, image_indices = get_token_indices(inputs["input_ids"], image_token_id)
    
    # 7. Get image spans for feature mapping
    image_spans = get_image_token_spans(inputs["input_ids"], image_token_id)
    
    # 8. Create token type dictionary
    token_types = {}
    for i in text_indices:
        token_types[int(i)] = "text"
    for i in image_indices:
        token_types[int(i)] = "image"
    
    logger.info(f"Prepared inputs with {len(text_indices)} text tokens, {len(image_indices)} image tokens")

    return {
    "inputs": inputs,
    "text_indices": text_indices,
    "image_indices": image_indices,
    "token_types": token_types,
    "formatted_prompt": formatted_prompt,
    "original_image": pil_img,
    "original_image_size_hw": (pil_img.height, pil_img.width),
    "image_token_id": int(image_token_id),
    "image_spans": image_spans,
    "token_lengths": {
        "text": len(text_indices),
        "image": len(image_indices),
        "total": inputs["input_ids"].shape[1],
    },
}
