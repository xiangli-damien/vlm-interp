# -*- coding: utf-8 -*-
"""
Data handling utilities for VLM analysis.

Includes functions for:
- Memory management (clearing CUDA cache).
- Loading images from various sources (URL, path, PIL, BytesIO).
- Building conversation prompts suitable for LLaVA models.
- Processing token IDs to identify text and image tokens/spans.
"""

import gc
import torch
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from typing import Dict, Any, Optional, Union, List, Tuple

def clean_memory():
    """
    Attempts to clear GPU memory cache and trigger garbage collection.

    Useful for freeing up resources, especially during iterative analysis
    or when working with large models in memory-constrained environments.
    Prints status messages indicating completion.
    """
    print("Attempting memory cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect() # This might be needed in specific distributed scenarios
        print("  CUDA memory cache cleared.")
    print("Memory cleanup routine complete.")

def load_image(
    source: Union[str, BytesIO, Image.Image],
    resize_to: Optional[Tuple[int, int]] = None, # Changed default to None
    convert_mode: str = "RGB",
    verbose: bool = True
) -> Image.Image:
    """
    Loads an image from various sources and optionally resizes and converts it.

    Handles URLs, local file paths, BytesIO streams, or existing PIL Image objects.

    Args:
        source (Union[str, BytesIO, Image.Image]): The source of the image.
            Can be a web URL (http/https), a local file system path,
            a BytesIO object containing image data, or a PIL Image object.
        resize_to (Optional[Tuple[int, int]]): Target size (width, height) to resize the image to.
            If None, the image is not resized. Defaults to None.
        convert_mode (str): The target PIL image mode (e.g., "RGB", "L").
            If the loaded image mode differs, it will be converted. Defaults to "RGB".
        verbose (bool): If True, print status messages during loading and processing. Defaults to True.

    Returns:
        Image.Image: The loaded (and potentially processed) PIL Image object.

    Raises:
        ValueError: If the source type is unsupported or if loading fails (e.g., invalid URL/path).
        requests.exceptions.RequestException: If downloading from a URL fails.
        FileNotFoundError: If a local file path does not exist.
        PIL.UnidentifiedImageError: If the data cannot be identified as an image.
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
                response = requests.get(source, stream=True, timeout=20) # Increased timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                image = Image.open(BytesIO(response.content))
                # Ensure content is fully read for BytesIO, context manager handles closing stream
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
            # Ensure the stream position is at the beginning if it has been read before
            source.seek(0)
            image = Image.open(source)
            if verbose: print(f"  Successfully loaded from BytesIO.")

        # Case 4: Unsupported type
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")

        # --- Post-loading Processing ---

        if image is None: # Should not happen if logic above is correct, but as a safeguard
             raise ValueError("Image loading resulted in None unexpectedly.")

        # Convert mode if specified and different
        if convert_mode and image.mode != convert_mode:
            original_mode = image.mode
            image = image.convert(convert_mode)
            if verbose: print(f"Converted image mode from {original_mode} to {convert_mode}")

        # Resize if specified
        if resize_to:
            original_size = image.size
            # Use LANCZOS (a high-quality downsampling filter)
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
        # Catch any other unexpected errors during loading/processing
        print(f"An unexpected error occurred during image loading/processing from {source_type}: {e}")
        raise ValueError(f"Failed to load or process image from {source_type}") from e


def build_conversation(prompt_text: str, conversation_format: bool = True) -> Union[List[Dict[str, Any]], str]:
    """
    Builds a conversation structure suitable for LLaVA models.

    Args:
        prompt_text (str): The user's text query.
        conversation_format (bool): If True, returns the standard LLaVA list-of-dictionaries format
                                    including placeholders for text and image. If False, returns a
                                    simpler string format (less common for LLaVA-Next). Defaults to True.

    Returns:
        Union[List[Dict[str, Any]], str]: The conversation structure.
    """
    if conversation_format:
        # Standard LLaVA-Next conversation format expects a list of turns.
        # A single turn from the user includes both text and an image placeholder.
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"}, # Placeholder processed by LlavaNextProcessor
                ],
            },
            # Optionally add an empty assistant turn if needed for prompting:
            # {"role": "assistant", "content": [{"type": "text", "text": ""}]}
        ]
    else:
        # A simpler, less structured format (might be needed for older models or specific use cases)
        # Assumes an image token placeholder like "<image>" will be appended or handled elsewhere.
        # This format is generally NOT recommended for LLaVA-Next.
        print("Warning: Using non-standard simple string format for conversation.")
        return prompt_text # The processor or calling code needs to handle adding image tokens.


def find_token_indices(input_ids: torch.Tensor, image_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separates input token IDs into text and image token indices.

    Assumes a batch size of 1 for the input_ids tensor.

    Args:
        input_ids (torch.Tensor): Tensor of token IDs, expected shape [1, sequence_length].
        image_token_id (int): The specific token ID used to represent image patches/placeholders.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - text_indices: A 1D tensor of indices corresponding to non-image tokens.
            - image_indices: A 1D tensor of indices corresponding to image tokens.
            Both tensors will be on the same device as the input_ids.

    Raises:
        ValueError: If the input_ids tensor does not have a batch size of 1.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids tensor with shape [1, sequence_length], but got {input_ids.shape}")

    # Keep tensors on the same device
    device = input_ids.device

    # Flatten to 1D for easier processing
    input_ids_1d = input_ids[0]

    # Create boolean masks
    is_image_mask = (input_ids_1d == image_token_id)
    is_text_mask = ~is_image_mask

    # Find indices where masks are True
    text_indices = torch.where(is_text_mask)[0]
    image_indices = torch.where(is_image_mask)[0]

    print(f"Found {len(text_indices)} text tokens and {len(image_indices)} image tokens using ID {image_token_id}.")

    return text_indices, image_indices


def find_image_token_spans(input_ids: torch.Tensor, image_token_id: int) -> List[Tuple[int, int]]:
    """
    Finds contiguous spans (start and end indices) of image tokens in the input sequence.

    Assumes a batch size of 1 for the input_ids tensor. Handles cases where the
    image token might appear in multiple contiguous blocks, though this is less
    common in standard LLaVA processing.

    Args:
        input_ids (torch.Tensor): Tensor of token IDs, expected shape [1, sequence_length].
        image_token_id (int): The specific token ID used to represent image patches/placeholders.

    Returns:
        List[Tuple[int, int]]: A list where each tuple represents a span: (start_index, end_index).
                                Returns an empty list if no image tokens are found.

    Raises:
        ValueError: If the input_ids tensor does not have a batch size of 1.
    """
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids tensor with shape [1, sequence_length], but got {input_ids.shape}")

    device = input_ids.device
    input_ids_1d = input_ids[0]

    # Create boolean mask for image tokens
    image_mask = (input_ids_1d == image_token_id)
    num_image_tokens = image_mask.sum().item()

    if num_image_tokens == 0:
        print(f"Warning: Image token ID {image_token_id} not found in the input sequence.")
        return []

    print(f"Found {num_image_tokens} image tokens with ID {image_token_id}. Identifying spans...")

    # Pad the mask with False at both ends to correctly detect transitions at the boundaries
    padded_mask = torch.cat([
        torch.tensor([False], device=device),
        image_mask,
        torch.tensor([False], device=device)
    ])

    # Find indices where the mask value changes (False->True or True->False)
    # diff gives 1 for False->True, -1 for True->False, 0 for no change
    diff = padded_mask[1:].int() - padded_mask[:-1].int()
    change_indices = torch.where(diff != 0)[0]

    spans = []
    # Iterate through change indices in pairs:
    # - A +1 change indicates the start of a span (at the index *after* the change).
    # - A -1 change indicates the end of a span (at the index *of* the change).
    for i in range(0, len(change_indices), 2):
         # Ensure we have a pair of changes (start and end)
        if i + 1 < len(change_indices):
            start_idx = change_indices[i].item() # Start index is the position of False->True transition
            end_idx = change_indices[i+1].item() - 1 # End index is position *before* True->False transition

            # Sanity check: start index should correspond to +1 diff, end index+1 should be -1 diff
            if diff[start_idx] == 1 and diff[end_idx + 1] == -1:
                span_len = end_idx - start_idx + 1
                if span_len > 0: # Ensure valid span length
                     spans.append((start_idx, end_idx))
                     print(f"  Detected image token span: Indices {start_idx} to {end_idx} (Length: {span_len})")
                else:
                     print(f"  Warning: Detected zero-length span at index {start_idx}. Ignoring.")
            else:
                 # This indicates an unexpected pattern (e.g., consecutive starts or ends)
                 print(f"  Warning: Unexpected transition pattern detected near index {start_idx}. Check input_ids.")


    if not spans:
        print("Warning: Found image tokens, but could not form valid contiguous spans.")

    return spans


def get_token_masks(
    input_ids: torch.Tensor,
    image_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates boolean masks identifying text and image tokens in the input sequence.
    """
    # --- Input Validation ---
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids tensor with shape [1, sequence_length], but got {input_ids.shape}")

    input_ids_1d = input_ids[0] # Flatten to 1D

    # --- Calculate Masks ---
    image_mask = (input_ids_1d == image_token_id)
    text_mask = ~image_mask

    # --- Count Tokens ---
    # num_text = text_mask.sum().item()
    # num_image = image_mask.sum().item()
    # print(f"Mask generation: Found {num_text} text and {num_image} image tokens using ID {image_token_id}.")

    return text_mask, image_mask

def get_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gets the 1D tensors containing the indices of text and image tokens.
    """
    text_mask, image_mask = get_token_masks(input_ids, image_token_id)

    # --- Calculate Indices from Masks ---
    text_indices = torch.where(text_mask)[0]
    image_indices = torch.where(image_mask)[0]

    print(f"Indices generation: Found {len(text_indices)} text tokens and {len(image_indices)} image tokens using ID {image_token_id}.")

    return text_indices, image_indices

def get_image_token_spans(
    input_ids: torch.Tensor,
    image_token_id: int
) -> List[Tuple[int, int]]:
    """
    Finds contiguous spans (start and end indices) of image tokens.
    """
    # --- Get the image mask first ---
    _, image_mask = get_token_masks(input_ids, image_token_id)
    device = input_ids.device # Get device from original tensor

    num_image_tokens = image_mask.sum().item()
    if num_image_tokens == 0:
        return []

    print(f"Span generation: Found {num_image_tokens} image tokens. Identifying spans...")

    # --- Calculate Spans from the image_mask ---
    spans = []
    padded_mask = torch.cat([
        torch.tensor([False], device=device),
        image_mask,
        torch.tensor([False], device=device)
    ])

    # Find indices where the mask value changes
    diff = padded_mask[1:].int() - padded_mask[:-1].int()
    change_indices = torch.where(diff != 0)[0]

    # Iterate through change indices in pairs
    for i in range(0, len(change_indices), 2):
        if i + 1 < len(change_indices):
            start_idx = change_indices[i].item()      # Position of False->True transition
            end_idx = change_indices[i+1].item() - 1 # Position *before* True->False transition

            # Sanity check for valid transitions and span length
            if diff[start_idx] == 1 and (end_idx + 1 >= len(diff) or diff[end_idx + 1] == -1): # Adjusted boundary check for diff
                span_len = end_idx - start_idx + 1
                if span_len > 0:
                    spans.append((start_idx, end_idx))
                    print(f"  Detected image token span: Indices {start_idx} to {end_idx} (Length: {span_len})")
                # else: # Warning for zero-length span already handled if needed
                #     print(f"  Warning: Detected zero-length span at index {start_idx}. Ignoring.")
            else:
                print(f"  Warning: Unexpected transition pattern detected near index {start_idx} for spans. Check input_ids.")

    if num_image_tokens > 0 and not spans:
        print("Warning: Found image tokens, but could not form valid contiguous spans based on transitions.")

    return spans
