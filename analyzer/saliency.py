"""
Saliency analysis utilities for VLM information flow.

Provides functions for analyzing the flow of information between different types
of tokens (text, image, generated) using attention weights and gradients.
"""

import torch
import numpy as np
import gc
from typing import Dict, Any, Optional, List, Tuple, Union
from tqdm import tqdm


def compute_flow_metrics_optimized(
    attention_or_saliency_matrix: torch.Tensor,
    text_indices: torch.Tensor,
    image_indices: torch.Tensor,
    target_idx: int,
    normalize: bool = False,
    top_k_image_tokens: Optional[int] = None  # New parameter
) -> Dict[str, float]:
    """
    Computes information flow metrics from different token types (text, image, generated)
    towards a specific target token, based on an attention or saliency matrix.

    Args:
        attention_or_saliency_matrix: A 2D tensor [seq_len, seq_len] representing
            attention weights or saliency scores. Assumes rows are destinations (target)
            and columns are sources.
        text_indices: 1D tensor of indices for text tokens
        image_indices: 1D tensor of indices for image tokens
        target_idx: The index of the target token (row index in the matrix)
        normalize: If True, normalizes to represent percentages of total flow
        top_k_image_tokens: If provided, only consider the top-k image tokens with 
            highest saliency scores when calculating the mean. If None, uses all image tokens.

    Returns:
        Dict containing flow metrics between different token types and the target
    """
    metrics: Dict[str, float] = {}
    if attention_or_saliency_matrix.ndim != 2:
         raise ValueError(f"Input matrix must be 2D, but got shape {attention_or_saliency_matrix.shape}")

    S = attention_or_saliency_matrix.shape[0] # Sequence length
    device = attention_or_saliency_matrix.device

    # Ensure indices are on the correct device
    if text_indices.device != device: text_indices = text_indices.to(device)
    if image_indices.device != device: image_indices = image_indices.to(device)

    # --- Create Masks ---
    # Boolean masks for source token types
    text_mask_src = torch.zeros(S, dtype=torch.bool, device=device)
    image_mask_src = torch.zeros(S, dtype=torch.bool, device=device)
    if len(text_indices) > 0: text_mask_src[text_indices] = True
    if len(image_indices) > 0: image_mask_src[image_indices] = True
    # Generated/Other mask includes tokens that are neither text nor image
    generated_mask_src = ~(text_mask_src | image_mask_src)

    # Causal mask: source index must be less than target index
    causal_mask = torch.arange(S, device=device) < target_idx

    # --- Extract Target Row ---
    if not (0 <= target_idx < S):
        print(f"Warning: target_idx {target_idx} is out of bounds for sequence length {S}. Returning zero metrics.")
        for prefix in ["Stq", "Siq", "Sgq"]:
             metrics[f"{prefix}_sum"] = 0.0
             metrics[f"{prefix}_mean"] = 0.0
             metrics[f"{prefix}_count"] = 0
             if normalize: metrics[f"{prefix}_percent"] = 0.0
        metrics["token_counts"] = {'text': 0, 'image': 0, 'generated': 0, 'total': S}
        return metrics

    target_row = attention_or_saliency_matrix[target_idx, :] # Shape [S] - Flow from all sources to target

    # --- Calculate Metrics for Each Source Type ---
    total_flow_sum_causal = 0.0 # For normalization

    for prefix, source_mask in [("Stq", text_mask_src), ("Siq", image_mask_src), ("Sgq", generated_mask_src)]:
        # Combine source type mask with causal mask
        valid_sources_mask = source_mask & causal_mask
        count = valid_sources_mask.sum().item()
        metrics[f"{prefix}_count"] = count

        if count > 0:
            values = target_row[valid_sources_mask]
            flow_sum = values.sum().item()
            
            # For image tokens, optionally consider only top-k
            if prefix == "Siq" and top_k_image_tokens is not None and count > top_k_image_tokens:
                # Get top-k values for image tokens
                top_k_values, _ = torch.topk(values, k=min(top_k_image_tokens, count))
                metrics[f"{prefix}_top{top_k_image_tokens}_mean"] = top_k_values.mean().item()
                metrics[f"{prefix}_top{top_k_image_tokens}_sum"] = top_k_values.sum().item()
            
            # Store standard metrics regardless
            metrics[f"{prefix}_mean"] = values.mean().item()
            metrics[f"{prefix}_sum"] = flow_sum
            total_flow_sum_causal += flow_sum
        else:
            metrics[f"{prefix}_sum"] = 0.0
            metrics[f"{prefix}_mean"] = 0.0
            if prefix == "Siq" and top_k_image_tokens is not None:
                metrics[f"{prefix}_top{top_k_image_tokens}_mean"] = 0.0
                metrics[f"{prefix}_top{top_k_image_tokens}_sum"] = 0.0

    # --- Normalization (Optional) ---
    if normalize:
        if total_flow_sum_causal > 1e-8: # Avoid division by zero
             for prefix in ["Stq", "Siq", "Sgq"]:
                 metrics[f"{prefix}_percent"] = (metrics[f"{prefix}_sum"] / total_flow_sum_causal) * 100.0
                 # Also normalize top-k metrics if they exist
                 if prefix == "Siq" and top_k_image_tokens is not None and metrics[f"{prefix}_count"] > top_k_image_tokens:
                     top_k_sum = metrics.get(f"{prefix}_top{top_k_image_tokens}_sum", 0.0)
                     metrics[f"{prefix}_top{top_k_image_tokens}_percent"] = (top_k_sum / total_flow_sum_causal) * 100.0
        else:
             for prefix in ["Stq", "Siq", "Sgq"]:
                 metrics[f"{prefix}_percent"] = 0.0
                 if prefix == "Siq" and top_k_image_tokens is not None and metrics[f"{prefix}_count"] > 0:
                     metrics[f"{prefix}_top{top_k_image_tokens}_percent"] = 0.0

    # --- Token Counts ---
    metrics["token_counts"] = {
        "text": text_mask_src.sum().item(),
        "image": image_mask_src.sum().item(),
        "generated": generated_mask_src.sum().item(),
        "total": S
    }

    return metrics


def calculate_saliency_scores(
    attention_weights: Dict[str, torch.Tensor],
    attention_grads: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Calculate saliency scores as the element-wise absolute product
    of attention weights and their corresponding gradients: S = |A * dA/dL|

    Args:
        attention_weights: Dictionary mapping layer names to attention weight tensors
        attention_grads: Dictionary mapping layer names to gradient tensors

    Returns:
        Dictionary mapping layer names to saliency score tensors
    """
    saliency_scores: Dict[str, torch.Tensor] = {}
    print(f"Calculating saliency scores for {len(attention_grads)} layers with gradients...")

    layers_with_grads = list(attention_grads.keys())

    for layer_name in layers_with_grads:
        if layer_name in attention_weights:
            attn = attention_weights[layer_name]
            grad = attention_grads[layer_name]

            # --- Sanity Checks ---
            if attn.shape != grad.shape:
                print(f"  Warning: Shape mismatch for layer '{layer_name}'! Weights: {attn.shape}, Grad: {grad.shape}. Skipping.")
                continue
            if attn.device != grad.device:
                print(f"  Warning: Device mismatch for layer '{layer_name}'! Weights: {attn.device}, Grad: {grad.device}. Attempting to move grad.")
                try: grad = grad.to(attn.device)
                except Exception as e: print(f"    Error moving gradient: {e}. Skipping layer."); continue

            # --- Compute Saliency ---
            try:
                # Perform calculation in float32 for potentially better precision
                saliency = torch.abs(attn.float() * grad.float())
                saliency_scores[layer_name] = saliency.detach() # Detach result
            except Exception as e:
                 print(f"  Error computing saliency for layer '{layer_name}': {e}. Skipping.")

        else:
            print(f"  Warning: Gradient found for layer '{layer_name}', but no corresponding attention weights were provided. Cannot compute saliency.")

    print(f"Calculated saliency scores for {len(saliency_scores)} layers.")
    return saliency_scores


def analyze_layerwise_saliency_flow(
    saliency_scores: Dict[str, torch.Tensor],
    text_indices: torch.Tensor,
    image_indices: torch.Tensor,
    target_token_idx: int,
    cpu_offload: bool = True,
    top_k_image_tokens: Optional[int] = 10  # New parameter with default value
) -> Dict[int, Dict[str, float]]:
    """
    Analyzes layer-wise information flow based on computed saliency scores.

    For each layer with a saliency score, it averages the saliency matrix over
    batch and head dimensions (if present) and then computes flow metrics
    (Text->Target, Image->Target, Generated->Target).

    Args:
        saliency_scores: Dictionary mapping layer names to saliency tensors
        text_indices: 1D tensor of indices for text tokens
        image_indices: 1D tensor of indices for image tokens
        target_token_idx: The index of the target token for flow analysis
        cpu_offload: If True, attempts to move saliency tensors to CPU before
                     computing metrics to conserve GPU memory
        top_k_image_tokens: If provided, only consider the top-k image tokens with 
                           highest saliency scores when calculating the mean. Default is 10.

    Returns:
        Dictionary mapping layer numerical index to flow metrics dictionary
    """
    layer_flow_metrics: Dict[int, Dict[str, float]] = {}
    print(f"\nAnalyzing layer-wise flow based on saliency scores for target token index: {target_token_idx}")
    if top_k_image_tokens:
        print(f"Using top-{top_k_image_tokens} image tokens for mean saliency calculation")
    
    if not saliency_scores:
         print("Warning: No saliency scores provided for analysis.")
         return {}

    # Sort layer names based on likely numerical index if possible
    def get_layer_num(name):
        parts = name.split('.')
        for p in parts:
             if p.isdigit(): return int(p)
        return -1 # Default if no number found
    sorted_layer_names = sorted(saliency_scores.keys(), key=get_layer_num)

    # Ensure index tensors are on the appropriate device for metric calculation
    target_device = torch.device('cpu') if cpu_offload else next(iter(saliency_scores.values())).device
    print(f"  Index tensors will be moved to: {target_device}")
    try:
         if text_indices.device != target_device: text_indices = text_indices.to(target_device)
         if image_indices.device != target_device: image_indices = image_indices.to(target_device)
    except Exception as e:
         print(f"  Error moving index tensors to {target_device}: {e}. Analysis might fail.")


    for layer_name in tqdm(sorted_layer_names, desc="Analyzing Layer Saliency", ncols=100):
        saliency_tensor = saliency_scores[layer_name]
        layer_num = get_layer_num(layer_name)
        if layer_num == -1:
            print(f"Warning: Could not determine numerical index for layer '{layer_name}'. Skipping.")
            continue

        # --- Prepare Saliency Matrix (Average over Batch/Heads) ---
        saliency_matrix_2d: Optional[torch.Tensor] = None
        if saliency_tensor.ndim == 4: # Assume [B, H, S, S]
            saliency_matrix_2d = saliency_tensor.mean(dim=(0, 1)).float() # Average batch and heads
        elif saliency_tensor.ndim == 3: # Assume [B, S, S] or [H, S, S] -> Average first dim
            saliency_matrix_2d = saliency_tensor.mean(dim=0).float()
        elif saliency_tensor.ndim == 2: # Assume [S, S]
            saliency_matrix_2d = saliency_tensor.float()
        else:
            print(f"Warning: Unexpected saliency tensor shape {saliency_tensor.shape} for layer '{layer_name}'. Skipping.")
            continue

        # --- Offload if requested ---
        if cpu_offload and saliency_matrix_2d.device != torch.device('cpu'):
             try:
                  saliency_matrix_2d = saliency_matrix_2d.cpu()
             except Exception as e:
                  print(f"Warning: Failed to move saliency matrix for layer {layer_num} to CPU: {e}. Trying on original device.")
                  # Ensure index tensors match the saliency matrix device if not offloaded
                  if text_indices.device != saliency_matrix_2d.device: text_indices = text_indices.to(saliency_matrix_2d.device)
                  if image_indices.device != saliency_matrix_2d.device: image_indices = image_indices.to(saliency_matrix_2d.device)
        elif not cpu_offload and (text_indices.device != saliency_matrix_2d.device or image_indices.device != saliency_matrix_2d.device):
             # If not offloading, make sure indices match the matrix device
             print(f"Moving index tensors to device {saliency_matrix_2d.device} for layer {layer_num}")
             try:
                 if text_indices.device != saliency_matrix_2d.device: text_indices = text_indices.to(saliency_matrix_2d.device)
                 if image_indices.device != saliency_matrix_2d.device: image_indices = image_indices.to(saliency_matrix_2d.device)
             except Exception as e:
                 print(f"  Error moving index tensors: {e}. Skipping layer.")
                 del saliency_matrix_2d # Clean up
                 continue

        # --- Compute Flow Metrics ---
        try:
            metrics = compute_flow_metrics_optimized(
                saliency_matrix_2d,
                text_indices,
                image_indices,
                target_token_idx,
                normalize=True, # Calculate percentages as well
                top_k_image_tokens=top_k_image_tokens  # Pass the top-k parameter
            )
            layer_flow_metrics[layer_num] = metrics
        except Exception as e:
            print(f"Error computing flow metrics for layer {layer_num} ('{layer_name}'): {e}")
            import traceback
            traceback.print_exc()

        # --- Cleanup ---
        del saliency_matrix_2d # Explicitly delete the potentially large 2D matrix
        if layer_num % 10 == 0: # Periodic GC
             gc.collect()
             if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"Saliency flow analysis complete for {len(layer_flow_metrics)} layers.")
    return layer_flow_metrics