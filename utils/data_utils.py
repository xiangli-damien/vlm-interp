"""
Data handling utilities for VLM analysis.

Includes functions for:
- Memory management (clearing CUDA cache)
- Loading images from various sources
- Building conversation prompts suitable for LLaVA models
- Processing token IDs to identify text and image tokens/spans
"""

import gc
import torch
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from typing import Dict, Any, Optional, Union, List, Tuple

def clean_memory():
    """Attempts to clear GPU memory cache and trigger garbage collection."""
    print("Attempting memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("  CUDA memory cache cleared.")
    print("Memory cleanup routine complete.")

def load_image(
    source: Union[str, BytesIO, Image.Image],
    resize_to: Optional[Tuple[int, int]] = None,
    convert_mode: str = "RGB",
    verbose: bool = True
) -> Image.Image:
    """
    Loads an image from various sources and optionally resizes and converts it.

    Args:
        source: URL, local path, BytesIO object, or PIL Image
        resize_to: Target size (width, height) to resize the image to
        convert_mode: Target PIL image mode
        verbose: Whether to print status messages

    Returns:
        Processed PIL Image
    """
    image: Optional[Image.Image] = None
    source_type = "Unknown"

    try:
        # Case 1: Source is already a PIL Image
        if isinstance(source, Image.Image):
            image = source
            source_type = "PIL Image"
            if verbose: print(f"Using provided PIL Image: Size={image.size}, Mode={image.mode}")

        # Case 2: Source is a string (URL or file path)
        elif isinstance(source, str):
            if source.startswith(('http://', 'https://')):
                source_type = "URL"
                if verbose: print(f"Loading image from URL: {source}")
                response = requests.get(source, stream=True, timeout=20)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                if verbose: print(f"  Successfully loaded from URL.")
            else:
                source_type = "File Path"
                if verbose: print(f"Loading image from file path: {source}")
                image = Image.open(source)
                if verbose: print(f"  Successfully loaded from file.")

        # Case 3: Source is a BytesIO object
        elif isinstance(source, BytesIO):
            source_type = "BytesIO"
            if verbose: print("Loading image from BytesIO object")
            source.seek(0)
            image = Image.open(source)
            if verbose: print(f"  Successfully loaded from BytesIO.")

        # Case 4: Unsupported type
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")

        # --- Post-loading Processing ---
        if image is None:
             raise ValueError("Image loading resulted in None unexpectedly.")

        # Convert mode if specified and different
        if convert_mode and image.mode != convert_mode:
            original_mode = image.mode
            image = image.convert(convert_mode)
            if verbose: print(f"Converted image mode from {original_mode} to {convert_mode}")

        # Resize if specified
        if resize_to:
            original_size = image.size
            image = image.resize(resize_to, Image.Resampling.LANCZOS)
            if verbose: print(f"Resized image from {original_size} to {resize_to}")

        if verbose:
            print(f"Image loading complete. Final dimensions: {image.size}, Mode: {image.mode}")

        return image

    except FileNotFoundError:
        print(f"Error loading image: File not found at path '{source}'")
        raise
    except requests.exceptions.RequestException as e:
        print(f"Error loading image from URL '{source}': {e}")
        raise
    except UnidentifiedImageError:
         print(f"Error loading image: Cannot identify image file from source ({source_type}). It might be corrupted or not an image.")
         raise ValueError(f"Cannot identify image file from {source_type}")
    except Exception as e:
        print(f"An unexpected error occurred during image loading/processing from {source_type}: {e}")
        raise ValueError(f"Failed to load or process image from {source_type}") from e


def build_conversation(prompt_text: str, conversation_format: bool = True) -> Union[List[Dict[str, Any]], str]:
    """
    Builds a conversation structure suitable for LLaVA models.

    Args:
        prompt_text: The user's text query
        conversation_format: If True, returns standard LLaVA list-of-dictionaries format

    Returns:
        Conversation structure for LLaVA models
    """
    if conversation_format:
        # Standard LLaVA-Next conversation format
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            },
        ]
    else:
        # Simpler format (less common for LLaVA-Next)
        print("Warning: Using non-standard simple string format for conversation.")
        return prompt_text
    
def build_formatted_prompt(processor, prompt_text: str) -> str:
    conversation = build_conversation(prompt_text)
    try:
        return processor.apply_chat_template(conversation, add_generation_prompt=True)
    except Exception:
        image_token = getattr(processor, "image_token", "<image>")
        return f"USER: {image_token}\n{prompt_text} ASSISTANT:"


def get_token_masks(input_ids: torch.Tensor, image_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates boolean masks identifying text and image tokens in the input sequence.
    
    Args:
        input_ids: Tensor of token IDs [1, sequence_length]
        image_token_id: Token ID for image tokens
        
    Returns:
        (text_mask, image_mask): Boolean tensors [sequence_length]
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids tensor with shape [1, sequence_length], but got {input_ids.shape}")

    input_ids_1d = input_ids[0]
    image_mask = (input_ids_1d == image_token_id)
    text_mask = ~image_mask

    return text_mask, image_mask


def get_token_indices(input_ids: torch.Tensor, image_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the 1D tensors containing the indices of text and image tokens.
    
    Args:
        input_ids: Tensor of token IDs [1, sequence_length]
        image_token_id: Token ID for image tokens
        
    Returns:
        (text_indices, image_indices): Tensors of indices
    """
    text_mask, image_mask = get_token_masks(input_ids, image_token_id)
    text_indices = torch.where(text_mask)[0]
    image_indices = torch.where(image_mask)[0]

    print(f"Indices generation: Found {len(text_indices)} text tokens and {len(image_indices)} image tokens using ID {image_token_id}.")

    return text_indices, image_indices


def get_image_token_spans(input_ids: torch.Tensor, image_token_id: int) -> List[Tuple[int, int]]:
    """
    Finds contiguous spans (start and end indices) of image tokens.
    
    Args:
        input_ids: Tensor of token IDs [1, sequence_length]
        image_token_id: Token ID for image tokens
        
    Returns:
        List of (start_index, end_index) tuples
    """
    _, image_mask = get_token_masks(input_ids, image_token_id)
    device = input_ids.device

    num_image_tokens = image_mask.sum().item()
    if num_image_tokens == 0:
        return []

    print(f"Span generation: Found {num_image_tokens} image tokens. Identifying spans...")

    spans = []
    padded_mask = torch.cat([
        torch.tensor([False], device=device),
        image_mask,
        torch.tensor([False], device=device)
    ])

    diff = padded_mask[1:].int() - padded_mask[:-1].int()
    change_indices = torch.where(diff != 0)[0]

    for i in range(0, len(change_indices), 2):
        if i + 1 < len(change_indices):
            start_idx = change_indices[i].item()
            end_idx = change_indices[i+1].item() - 1

            if diff[start_idx] == 1 and (end_idx + 1 >= len(diff) or diff[end_idx + 1] == -1):
                span_len = end_idx - start_idx + 1
                if span_len > 0:
                    spans.append((start_idx, end_idx))
                    print(f"  Detected image token span: Indices {start_idx} to {end_idx} (Length: {span_len})")
            else:
                print(f"  Warning: Unexpected transition pattern detected near index {start_idx} for spans. Check input_ids.")

    if num_image_tokens > 0 and not spans:
        print("Warning: Found image tokens, but could not form valid contiguous spans based on transitions.")

    return spans