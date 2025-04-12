# -*- coding: utf-8 -*-
"""
Generic functions for extracting internal representations from models,
such as hidden states and logits projected from intermediate layers.
"""

import torch
from torch import nn
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict, Any, Union

def get_hidden_states_from_forward_pass(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    layers_to_extract: Optional[List[int]] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Performs a forward pass and extracts hidden states from specified layers.

    Args:
        model (nn.Module): The model to run the forward pass on. Assumed to support `output_hidden_states=True`.
        inputs (Dict[str, torch.Tensor]): The input data prepared for the model.
        layers_to_extract (Optional[List[int]]): A list of layer indices to extract.
                                                  If None, extracts all hidden states returned by the model.
                                                  Layer 0 typically corresponds to the initial embeddings.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple containing the hidden state tensors for the requested layers.

    Raises:
        ValueError: If the model does not return hidden states.
        IndexError: If a requested layer index is out of bounds.
    """
    model.eval() # Ensure model is in evaluation mode
    print(f"Performing forward pass to extract hidden states...")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)

    all_hidden_states = outputs.get("hidden_states")

    if all_hidden_states is None:
        raise ValueError("Model did not return 'hidden_states'. Ensure the model supports this output and 'output_hidden_states=True' was passed.")

    print(f"  Model returned {len(all_hidden_states)} hidden state tensors (including embeddings).")

    if layers_to_extract is None:
        print(f"  Extracting all {len(all_hidden_states)} hidden states.")
        return all_hidden_states
    else:
        extracted_states = []
        max_layer_idx = len(all_hidden_states) - 1
        print(f"  Extracting hidden states for specified layers: {layers_to_extract}")
        for layer_idx in layers_to_extract:
            if 0 <= layer_idx <= max_layer_idx:
                extracted_states.append(all_hidden_states[layer_idx])
                print(f"    Extracted layer {layer_idx} (Shape: {all_hidden_states[layer_idx].shape})")
            else:
                raise IndexError(f"Requested layer index {layer_idx} is out of bounds (0 to {max_layer_idx}).")
        return tuple(extracted_states)


def get_logits_from_hidden_states(
    hidden_states: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]],
    lm_head: nn.Module,
    layers_to_process: Optional[List[int]] = None,
    cpu_offload: bool = True,
    use_float32_for_softmax: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Projects hidden states from specified layers through the language model head (lm_head)
    to compute vocabulary logits for each layer.

    Args:
        hidden_states (Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]):
            A tuple or list of hidden state tensors, typically one per layer
            (e.g., from `get_hidden_states_from_forward_pass` or model output).
            Assumes hidden_states[0] is embeddings, hidden_states[i] is output of layer i.
        lm_head (nn.Module): The language model head (usually a Linear layer) used for projection.
        layers_to_process (Optional[List[int]]): A list of layer indices (corresponding to the indices in `hidden_states`)
                                                 to process. If None, processes all available layers.
        cpu_offload (bool): If True, moves logits tensor to CPU after computation to save GPU memory. Defaults to True.
        use_float32_for_softmax (bool): If True, casts logits to float32 before softmax for numerical stability. Defaults to True.
                                        Note: This returns *probabilities*, not raw logits, despite the function name.

    Returns:
        Dict[int, torch.Tensor]: A dictionary mapping layer index to the computed probability tensor
                                 (shape typically [batch_size, sequence_length, vocab_size]).
                                 Probabilities are on CPU if cpu_offload is True.
    """
    print("Computing probabilities from hidden states using LM head...")
    layer_probabilities: Dict[int, torch.Tensor] = {}
    num_hidden_layers = len(hidden_states)

    if layers_to_process is None:
        layers_to_process = list(range(num_hidden_layers))
    else:
        # Validate layer indices
        layers_to_process = [l for l in layers_to_process if 0 <= l < num_hidden_layers]
        if not layers_to_process:
             print("Warning: No valid layers specified in layers_to_process. Returning empty dictionary.")
             return {}

    print(f"  Processing layers: {layers_to_process}")

    # Determine target device for LM head operation
    try:
        lm_head_device = next(lm_head.parameters()).device
    except StopIteration: # Handle cases where lm_head might not have parameters (unlikely but possible)
         print("Warning: Could not determine LM head device from parameters. Assuming model's main device if possible.")
         # Attempt to find device from a buffer or default to CPU
         try:
             lm_head_device = next(lm_head.buffers()).device
         except StopIteration:
             print("Warning: LM head has no parameters or buffers. Defaulting device to CPU.")
             lm_head_device = torch.device('cpu')


    print(f"  LM head computations will run on device: {lm_head_device}")

    # Ensure lm_head is in eval mode
    lm_head.eval()

    with torch.no_grad():
        for layer_idx in tqdm(layers_to_process, desc="Projecting Layers"):
            layer_hidden = hidden_states[layer_idx]
            original_hidden_device = layer_hidden.device

            # Move hidden state to LM head device if necessary
            if layer_hidden.device != lm_head_device:
                try:
                    layer_hidden = layer_hidden.to(lm_head_device)
                except Exception as e:
                    print(f"Warning: Failed to move hidden state for layer {layer_idx} to device {lm_head_device}. Skipping layer. Error: {e}")
                    continue

            # Project to logits
            try:
                 # Optionally cast hidden state before projection for stability? Usually not needed.
                 # layer_hidden_proj = layer_hidden.float() if use_float32_for_projection else layer_hidden
                 logits = lm_head(layer_hidden)
            except Exception as e:
                print(f"Warning: Error during LM head projection for layer {layer_idx}. Skipping layer. Error: {e}")
                # Clean up potentially moved tensor
                if layer_hidden.device != original_hidden_device: del layer_hidden
                continue

            # Compute probabilities (softmax)
            try:
                if use_float32_for_softmax:
                     probs = torch.softmax(logits.float(), dim=-1)
                else:
                     probs = torch.softmax(logits, dim=-1)
            except Exception as e:
                 print(f"Warning: Error during softmax calculation for layer {layer_idx}. Skipping layer. Error: {e}")
                 # Clean up potentially moved tensor and logits
                 if layer_hidden.device != original_hidden_device: del layer_hidden
                 del logits
                 continue


            # Offload to CPU if requested
            if cpu_offload:
                if probs.device != torch.device('cpu'):
                     try:
                          probs = probs.cpu()
                     except Exception as e:
                          print(f"Warning: Failed to move probabilities for layer {layer_idx} to CPU. Error: {e}")
                          # Keep on current device if move fails

            layer_probabilities[layer_idx] = probs

            # Clean up tensors on the computation device (lm_head_device) if they are not the original hidden state
            if layer_hidden.device != original_hidden_device: del layer_hidden
            if logits.device == lm_head_device: del logits # Delete logits on computation device
            # Delete probs from computation device only if it was *not* moved to CPU
            if not cpu_offload and probs.device == lm_head_device: del probs
            elif cpu_offload and probs.device == lm_head_device: # Should have been moved, but delete if still there
                 del probs

            # Try to clear CUDA cache periodically, especially if not offloading heavily
            if lm_head_device.type == 'cuda' and layer_idx % 5 == 0: # Every 5 layers
                 torch.cuda.empty_cache()


    print("Probability computation complete.")
    return layer_probabilities