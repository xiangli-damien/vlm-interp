# -*- coding: utf-8 -*-
"""
Visualization utilities for VLM analysis results.

Includes functions for plotting:
- Information flow metrics across layers.
- Attention weight heatmaps.
- Processed image tensors fed into the vision encoder.
- Logit lens token probability heatmaps and overlays.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
# Ensure necessary PIL components are imported
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import time # Potentially useful for timing sections
import traceback # For detailed error printing

# Try importing skimage for potentially better heatmap resizing, with fallback
try:
    from skimage.transform import resize as skimage_resize
    HAS_SKIMAGE = True
except (ImportError, ModuleNotFoundError):
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Falling back to simpler numpy-based resizing for heatmap visualization.")


def visualize_information_flow(
    metrics: Dict[int, Dict[str, float]],
    title: str = "VLM Information Flow Analysis",
    save_path: Optional[str] = None
):
    """
    Visualizes information flow metrics (mean and sum) across model layers.

    Creates line plots showing how attention flows from text, image, and
    previously generated tokens towards the current target token at each layer.

    Args:
        metrics (Dict[int, Dict[str, float]]): A dictionary where keys are layer indices (int)
            and values are dictionaries containing flow metrics (e.g., 'Siq_mean',
            'Stq_sum', 'Sgq_mean'). Assumes metrics 'Siq', 'Stq', 'Sgq'.
        title (str): The main title for the combined plot figure.
        save_path (Optional[str]): If provided, the path where the plot image will be saved.
    """
    if not metrics:
        print("Warning: No metrics data provided to visualize_information_flow.")
        return

    # Define consistent colors and markers for different flow types
    flow_styles = {
        "Siq_mean": {"color": "#FF4500", "marker": "o", "label": "Image→Target (Mean)"},  # Orange-Red
        "Stq_mean": {"color": "#1E90FF", "marker": "^", "label": "Text→Target (Mean)"},    # Dodger Blue
        "Sgq_mean": {"color": "#32CD32", "marker": "s", "label": "Generated→Target (Mean)"}, # Lime Green (Sgq replaces Soq)
        "Siq_sum": {"color": "#FF4500", "marker": "o", "label": "Image→Target (Sum)", "linestyle": '--'}, # Dashed for Sum
        "Stq_sum": {"color": "#1E90FF", "marker": "^", "label": "Text→Target (Sum)", "linestyle": '--'}, # Dashed for Sum
        "Sgq_sum": {"color": "#32CD32", "marker": "s", "label": "Generated→Target (Sum)", "linestyle": '--'}  # Dashed for Sum (Sgq replaces Soq)
    }

    # Extract layer indices and available metric keys
    layers = sorted(metrics.keys())
    if not layers:
        print("Warning: Metrics dictionary is empty or contains no valid layer indices.")
        return

    available_metric_keys = set()
    for layer_idx in layers:
        if isinstance(metrics[layer_idx], dict):
             available_metric_keys.update(metrics[layer_idx].keys())

    # Collect data for plotting, handling missing metrics gracefully
    plot_data: Dict[str, List[Optional[float]]] = {key: [] for key in flow_styles if key in available_metric_keys}
    for layer in layers:
         layer_metrics = metrics.get(layer, {})
         for metric_key in plot_data.keys():
              plot_data[metric_key].append(layer_metrics.get(metric_key)) # Append value or None

    # Create figure with two subplots (Mean and Sum)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True) # Share x-axis

    # --- Plot 1: Mean Information Flow ---
    ax1.set_title("Mean Information Flow per Layer")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean Attention / Saliency")
    mean_keys = ["Siq_mean", "Stq_mean", "Sgq_mean"]
    for key in mean_keys:
        if key in plot_data:
            style = flow_styles[key]
            valid_layers = [l for l, v in zip(layers, plot_data[key]) if v is not None]
            valid_values = [v for v in plot_data[key] if v is not None]
            if valid_layers:
                 ax1.plot(valid_layers, valid_values, marker=style["marker"], color=style["color"], label=style["label"], linewidth=2)
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Sum Information Flow ---
    ax2.set_title("Total Information Flow per Layer")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Summed Attention / Saliency")
    sum_keys = ["Siq_sum", "Stq_sum", "Sgq_sum"]
    for key in sum_keys:
         if key in plot_data:
             style = flow_styles[key]
             valid_layers = [l for l, v in zip(layers, plot_data[key]) if v is not None]
             valid_values = [v for v in plot_data[key] if v is not None]
             if valid_layers:
                 ax2.plot(valid_layers, valid_values, marker=style["marker"], color=style["color"], label=style["label"], linestyle=style.get("linestyle", '-'), linewidth=2)
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Add overall figure title
    fig.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Save if requested
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Information flow visualization saved to: {save_path}")
        except Exception as e:
            print(f"Error saving information flow plot to {save_path}: {e}")

    plt.show()
    plt.close(fig)


def visualize_attention_heatmap(
    attention_matrix: Union[np.ndarray, torch.Tensor],
    tokens: Optional[List[str]] = None,
    title: str = "Attention Heatmap",
    save_path: Optional[str] = None,
    colormap: str = "viridis",
    max_tokens_display: int = 60
):
    """
    Creates a heatmap visualization of an attention matrix using matplotlib.

    Args:
        attention_matrix (Union[np.ndarray, torch.Tensor]): 2D array/tensor of attention weights
            (Sequence Length x Sequence Length). Assumes weights are from destination (rows)
            attending to source (columns).
        tokens (Optional[List[str]]): List of token strings corresponding to the sequence length.
        title (str): Title for the heatmap plot.
        save_path (Optional[str]): Path to save the generated heatmap image. If None, not saved.
        colormap (str): Matplotlib colormap name.
        max_tokens_display (int): Maximum number of token labels to display on each axis.
    """
    if isinstance(attention_matrix, torch.Tensor):
        attention_data = attention_matrix.detach().cpu().numpy()
    elif isinstance(attention_matrix, np.ndarray):
        attention_data = attention_matrix
    else:
        raise TypeError("attention_matrix must be a NumPy array or PyTorch tensor.")

    if attention_data.ndim != 2:
        raise ValueError(f"attention_matrix must be 2D, but got shape {attention_data.shape}")

    seq_len_dst, seq_len_src = attention_data.shape
    if seq_len_dst != seq_len_src:
        print(f"Warning: Attention matrix shape {attention_data.shape} is not square.")

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attention_data, cmap=colormap, aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Attention Weight")

    if tokens:
        display_tokens = tokens # Assume tokens match dimensions for now
        if len(tokens) != seq_len_dst or len(tokens) != seq_len_src:
             print(f"Warning: Number of tokens ({len(tokens)}) differs from matrix dims ({seq_len_dst}x{seq_len_src}).")
             display_tokens = tokens[:max(seq_len_dst, seq_len_src)]

        num_ticks = min(max_tokens_display, max(seq_len_dst, seq_len_src))
        ticks_src = np.linspace(0, seq_len_src - 1, num_ticks, dtype=int)
        labels_src = [display_tokens[i] if i < len(display_tokens) else '?' for i in ticks_src]
        ticks_dst = np.linspace(0, seq_len_dst - 1, num_ticks, dtype=int)
        labels_dst = [display_tokens[i] if i < len(display_tokens) else '?' for i in ticks_dst]

        ax.set_xticks(ticks_src)
        ax.set_xticklabels(labels_src, rotation=90, fontsize=8)
        ax.set_yticks(ticks_dst)
        ax.set_yticklabels(labels_dst, rotation=0, fontsize=8)

        ax.set_xticks(np.arange(seq_len_src + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(seq_len_dst + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="grey", linestyle='-', linewidth=0.5, alpha=0.3)
        ax.tick_params(which='minor', size=0)
        ax.set_ylabel("Destination Token (Query)", fontsize=10)
        ax.set_xlabel("Source Token (Key/Value)", fontsize=10)
    else:
        ax.set_xlabel("Source Token Index")
        ax.set_ylabel("Destination Token Index")
        ax.grid(True, linestyle=':', alpha=0.4)

    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to: {save_path}")
        except Exception as e:
            print(f"Error saving attention heatmap to {save_path}: {e}")

    plt.show()
    plt.close(fig)


def visualize_processed_image_input(analysis_data: Dict[str, Any], save_dir: Optional[str] = None):
    """
    Visualizes the actual processed image tensor(s) fed into the vision encoder.

    Handles both standard single image inputs ([B, C, H, W]) and tiled inputs
    used in high-resolution processing ([B, N, C, H, W]).

    Args:
        analysis_data (Dict[str, Any]): Dictionary containing analysis results,
            expected to have 'inputs_cpu' key which holds the processed tensors
            moved to CPU, including 'pixel_values'.
        save_dir (Optional[str]): Directory to save the visualization(s). If None, not saved.
    """
    print("\nVisualizing processed image tensor(s) input to vision encoder...")
    inputs_cpu = analysis_data.get("inputs_cpu")
    if not inputs_cpu or "pixel_values" not in inputs_cpu:
        print("Error: Missing 'pixel_values' in 'inputs_cpu' for visualization.")
        return

    pixel_values = inputs_cpu["pixel_values"]
    save_paths = []
    if save_dir:
         os.makedirs(save_dir, exist_ok=True)

    try:
        # Case 1: High-resolution Tiling [B, N, C, H, W]
        if pixel_values.ndim == 5:
            batch_idx = 0
            num_tiles, C, H, W = pixel_values.shape[1:]
            print(f"Input tensor shape: {pixel_values.shape}. Visualizing {num_tiles} tiles.")
            cols = int(np.ceil(np.sqrt(num_tiles)))
            rows = int(np.ceil(num_tiles / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), squeeze=False)
            axes_flat = axes.flatten()
            for i in range(num_tiles):
                tile_tensor = pixel_values[batch_idx, i]
                tile_np = tile_tensor.permute(1, 2, 0).float().numpy()
                min_val, max_val = tile_np.min(), tile_np.max()
                tile_np = (tile_np - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(tile_np)
                ax = axes_flat[i]
                ax.imshow(np.clip(tile_np, 0, 1))
                ax.set_title(f"Tile {i+1}", fontsize=9)
                ax.axis("off")
            for i in range(num_tiles, len(axes_flat)): axes_flat[i].axis("off")
            fig.suptitle(f"Processed Image Input Tiles (from {pixel_values.shape} shape)", fontsize=14)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                fpath = os.path.join(save_dir, "processed_image_tiles.png")
                plt.savefig(fpath, dpi=150)
                save_paths.append(fpath); print(f"  Saved tiled visualization to: {fpath}")
            plt.show(); plt.close(fig)

        # Case 2: Standard Single Image [B, C, H, W]
        elif pixel_values.ndim == 4:
            batch_idx = 0
            C, H, W = pixel_values.shape[1:]
            print(f"Input tensor shape: {pixel_values.shape}. Visualizing single processed image.")
            img_tensor = pixel_values[batch_idx]
            img_np = img_tensor.permute(1, 2, 0).float().numpy()
            min_val, max_val = img_np.min(), img_np.max()
            img_np = (img_np - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(img_np)
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(f"Processed Image Input Tensor\nShape: {list(img_tensor.shape)}")
            ax.axis("off"); plt.tight_layout()
            if save_dir:
                 fpath = os.path.join(save_dir, "processed_image_single.png")
                 plt.savefig(fpath, dpi=150)
                 save_paths.append(fpath); print(f"  Saved single image visualization to: {fpath}")
            plt.show(); plt.close(fig)
        else:
            print(f"Warning: Unexpected pixel_values shape: {pixel_values.shape}. Cannot visualize.")
    except Exception as e:
        print(f"An error occurred during visualization of processed image: {e}")
        import traceback; traceback.print_exc()
    return save_paths


# <<< Added Function >>>
def visualize_token_probabilities(
    token_probs: Dict[int, Dict[str, Any]],
    input_data: Dict[str, Any],
    selected_layers: Optional[List[int]] = None,
    output_dir: str = "logit_lens_visualization",
    colormap: str = "jet", # Colormap for heatmaps
    heatmap_alpha: float = 0.6, # Alpha for heatmap overlay
    generate_composite: bool = True # Flag to generate composite images
):
    """
    Visualize token probability maps from logit lens analysis using heatmaps and line plots.

    Generates individual visualizations per layer for 'base_feature' (grid heatmap overlaid
    on original image), 'patch_feature' (overlay on spatial preview image), and
    'newline_feature' (line plot). Optionally generates composite images combining all
    layer visualizations for 'base' and 'patch' features using internally defined padding
    via the `create_composite_image` helper function (assumed to be defined elsewhere).

    Args:
        token_probs (Dict[int, Dict[str, Any]]): Dictionary mapping layer index to probabilities.
            Expected inner structure: {'base_feature': {concept: np.array}, 'patch_feature': ...}.
            Probabilities are typically max probability for tracked concept token(s).
        input_data (Dict[str, Any]): Dictionary from the analyzer's `prepare_inputs`. Must contain
            'feature_mapping', 'original_image' (PIL), and 'spatial_preview_image' (PIL).
        selected_layers (Optional[List[int]]): List of layer indices to visualize. If None, visualizes all available layers.
        output_dir (str): Directory path to save the visualization images. Subdirectories will be created.
                               This path is typically set by the calling analysis function.
        colormap (str): Matplotlib colormap name for heatmaps.
        heatmap_alpha (float): Alpha blending value for heatmap overlays (0.0 to 1.0).
        generate_composite (bool): If True, creates composite images stitching together
                                   individual layer visualizations for base and patch features.

    Returns:
        List[str]: File paths of all saved visualization images (including composites).
    """
    print(f"\n--- Generating Logit Lens Probability Visualizations ---")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Validate Inputs ---
    if not token_probs:
        print("  Error: No token probabilities data provided. Cannot visualize.")
        return []
    required_input_keys = ['feature_mapping', 'spatial_preview_image', 'original_image']
    if not input_data or not all(k in input_data for k in required_input_keys):
        print(f"  Error: Missing required keys in input_data. Need: {required_input_keys}.")
        print(f"  Available keys: {list(input_data.keys()) if input_data else 'None'}")
        return []
    if not isinstance(input_data.get("original_image"), Image.Image):
         print(f"  Error: 'original_image' in input_data is not a PIL Image object.")
         return []
    if not isinstance(input_data.get("spatial_preview_image"), Image.Image):
         print(f"  Error: 'spatial_preview_image' in input_data is not a PIL Image object.")
         return []

    feature_mapping = input_data["feature_mapping"]
    spatial_preview_image = input_data["spatial_preview_image"] # PIL Image (resized + padded)
    original_image_pil = input_data["original_image"] # Original PIL Image

    if not feature_mapping or not isinstance(feature_mapping, dict):
        print("  Error: 'feature_mapping' in input_data is empty or not a dictionary.")
        return []

    raw_patch_size = feature_mapping.get("patch_size")
    if not isinstance(raw_patch_size, int) or raw_patch_size <= 0:
        print(f"  Error: 'patch_size' ({raw_patch_size}) not found or invalid in feature_mapping.")
        return []

    # --- Determine Layers and Concepts ---
    available_layers = sorted([k for k in token_probs.keys() if isinstance(k, int)])
    if not available_layers:
        print("  Error: token_probs dictionary contains no valid integer layer keys.")
        return []

    if selected_layers is None:
        layers_to_plot = available_layers
    else:
        # Ensure selected_layers contains only valid integers present in available_layers
        layers_to_plot = sorted([l for l in selected_layers if isinstance(l, int) and l in available_layers])
        if not layers_to_plot:
             print(f"  Warning: None of the selected layers {selected_layers} have data in token_probs. Visualizing all available layers instead ({available_layers}).")
             layers_to_plot = available_layers

    # Infer concepts from the first layer's data robustly
    concepts = set() # Use a set to automatically handle duplicates
    first_valid_layer_data = token_probs.get(layers_to_plot[0], {})

    # Check 'base_feature' dictionary
    base_feat_data = first_valid_layer_data.get("base_feature")
    if isinstance(base_feat_data, dict):
        concepts.update(k for k in base_feat_data.keys() if isinstance(k, str))
    # Check 'patch_feature' dictionary
    patch_feat_data = first_valid_layer_data.get("patch_feature")
    if isinstance(patch_feat_data, dict):
        concepts.update(k for k in patch_feat_data.keys() if isinstance(k, str))
    # Check 'newline_feature' dictionary
    newline_feat_data = first_valid_layer_data.get("newline_feature")
    if isinstance(newline_feat_data, dict):
        concepts.update(k for k in newline_feat_data.keys() if isinstance(k, str))

    concepts = sorted(list(concepts)) # Convert back to sorted list

    if not concepts:
        print("  Error: No concepts (string keys) found in the token probability data for the first layer. Cannot visualize.")
        return []

    print(f"  Visualizing for Layers: {layers_to_plot}")
    print(f"  Visualizing for Concepts: {concepts}")

    # --- Prepare Output Directories and Storage for Paths ---
    all_saved_paths = [] # Stores paths of ALL generated images
    # Dictionaries to store paths for composite generation, keyed by concept
    base_feature_paths = {concept: {} for concept in concepts} # Store {layer_idx: path}
    patch_feature_paths = {concept: {} for concept in concepts} # Store {layer_idx: path}

    # Define subdirectories within the main output_dir
    base_dir = os.path.join(output_dir, "base_feature_overlays")
    patch_dir = os.path.join(output_dir, "patch_feature_overlays")
    newline_dir = os.path.join(output_dir, "newline_feature_plots")
    composite_dir = os.path.join(output_dir, "composite_views") # Directory for composite images

    # Create directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(newline_dir, exist_ok=True)
    if generate_composite:
        os.makedirs(composite_dir, exist_ok=True)

    # --- 1. Visualize Base Feature Heatmaps (Overlay on Resized Original Image) ---
    print("  Generating base feature overlay heatmaps...")
    base_feature_map_info = feature_mapping.get("base_feature", {})
    base_grid = base_feature_map_info.get("grid") # Expected shape (grid_h, grid_w)

    if base_grid and isinstance(base_grid, tuple) and len(base_grid) == 2:
        base_grid_h, base_grid_w = base_grid
        target_overlay_size = (336, 336) # Standard size, could be configurable
        try:
            resized_background = original_image_pil.resize(target_overlay_size, Image.Resampling.LANCZOS)
            background_np = np.array(resized_background)
        except Exception as e:
            print(f"  Error resizing original image for base background: {e}. Skipping base overlays.")
            background_np = None # Mark background as unavailable

        if background_np is not None:
            for concept in concepts:
                pbar_base = tqdm(layers_to_plot, desc=f"Base '{concept}' Overlay", leave=False, ncols=100)
                for layer_idx in pbar_base:
                    layer_data = token_probs.get(layer_idx, {}).get("base_feature", {})
                    base_prob_map = layer_data.get(concept) # np.array

                    if base_prob_map is None or not isinstance(base_prob_map, np.ndarray) or base_prob_map.size == 0:
                        continue
                    if base_prob_map.shape != (base_grid_h, base_grid_w):
                        print(f"  Warning: Shape mismatch base layer {layer_idx}, '{concept}'. Expected {base_grid}, got {base_prob_map.shape}. Skipping.")
                        continue

                    # --- Upscale Heatmap ---
                    upscaled_heatmap_base: Optional[np.ndarray] = None
                    try:
                        if HAS_SKIMAGE:
                            upscaled_heatmap_base = skimage_resize(base_prob_map, target_overlay_size, order=1, mode='constant', cval=0, anti_aliasing=True, preserve_range=True)
                        else: # Fallback numpy scaling
                            repeat_y = target_overlay_size[1] // base_grid_h
                            repeat_x = target_overlay_size[0] // base_grid_w
                            upscaled_heatmap_base = np.kron(base_prob_map, np.ones((repeat_y, repeat_x)))
                            upscaled_heatmap_base = upscaled_heatmap_base[:target_overlay_size[1], :target_overlay_size[0]]
                    except Exception as e:
                         print(f"  Warning: Error upscaling base heatmap layer {layer_idx} '{concept}': {e}. Skipping visualization.")
                         continue # Skip this layer/concept if upscaling fails

                    # --- Plotting ---
                    fig, ax = plt.subplots(figsize=(6, 6))
                    try:
                        ax.imshow(background_np, extent=(0, target_overlay_size[0], target_overlay_size[1], 0))
                        im = ax.imshow(upscaled_heatmap_base, alpha=heatmap_alpha, cmap=colormap, vmin=0, vmax=1, extent=(0, target_overlay_size[0], target_overlay_size[1], 0))
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label("Max Probability")
                        ax.set_title(f"Base Feature Overlay: '{concept}' - Layer {layer_idx}", fontsize=12)
                        ax.axis("off")

                        filepath = os.path.join(base_dir, f"layer_{layer_idx:03d}_{concept}_base_overlay.png")
                        plt.savefig(filepath, dpi=150, bbox_inches="tight")
                        all_saved_paths.append(filepath)
                        if generate_composite:
                           base_feature_paths[concept][layer_idx] = filepath # Store path keyed by layer index
                    except Exception as e:
                        print(f"    Error plotting/saving base overlay layer {layer_idx} '{concept}': {e}")
                    finally:
                        plt.close(fig) # Ensure figure is closed
    else:
        print("  Skipping base feature visualization: Grid info invalid or missing in feature_mapping.")


    # --- 2. Visualize Spatial (Patch) Feature Heatmaps (Overlay on Padded Preview Image) ---
    print("  Generating patch feature overlay heatmaps...")
    patch_feature_map_info = feature_mapping.get("patch_feature", {})
    patch_vis_grid = patch_feature_map_info.get("grid_for_visualization")
    patch_unpadded_grid = patch_feature_map_info.get("grid_unpadded")
    resized_dims_wh = feature_mapping.get("resized_dimensions") # W, H after resize, before padding

    if (patch_vis_grid and isinstance(patch_vis_grid, tuple) and len(patch_vis_grid) == 2 and
        patch_unpadded_grid and isinstance(patch_unpadded_grid, tuple) and len(patch_unpadded_grid) == 2 and
        resized_dims_wh and isinstance(resized_dims_wh, tuple) and len(resized_dims_wh) == 2):

        prob_grid_h, prob_grid_w = patch_unpadded_grid
        resized_w_actual, resized_h_actual = resized_dims_wh
        preview_w, preview_h = spatial_preview_image.size

        # Calculate padding based on preview size and actual resized dimensions
        pad_h_total = preview_h - resized_h_actual
        pad_w_total = preview_w - resized_w_actual
        pad_top = max(0, pad_h_total // 2) # Ensure non-negative padding
        pad_left = max(0, pad_w_total // 2)

        for concept in concepts:
            pbar_patch = tqdm(layers_to_plot, desc=f"Patch '{concept}' Overlay", leave=False, ncols=100)
            for layer_idx in pbar_patch:
                layer_data = token_probs.get(layer_idx, {}).get("patch_feature", {})
                patch_prob_map_unpadded = layer_data.get(concept) # np.array

                if patch_prob_map_unpadded is None or not isinstance(patch_prob_map_unpadded, np.ndarray) or patch_prob_map_unpadded.size == 0:
                    continue
                if patch_prob_map_unpadded.shape != (prob_grid_h, prob_grid_w):
                    print(f"  Warning: Shape mismatch patch layer {layer_idx}, '{concept}'. Expected {patch_unpadded_grid}, got {patch_prob_map_unpadded.shape}. Skipping.")
                    continue

                # --- Upscale and Resize Heatmap ---
                resized_heatmap_patch: Optional[np.ndarray] = None
                try:
                    # Upscale probability map by raw patch size first
                    heatmap_unpadded = np.repeat(np.repeat(patch_prob_map_unpadded, raw_patch_size, axis=0), raw_patch_size, axis=1)
                    heatmap_h_unpadded, heatmap_w_unpadded = heatmap_unpadded.shape

                    # Resize this large heatmap to match the actual content area dimensions
                    target_heatmap_size = (resized_h_actual, resized_w_actual)
                    if HAS_SKIMAGE:
                        resized_heatmap_patch = skimage_resize(heatmap_unpadded, target_heatmap_size, order=1, mode='constant', cval=0, anti_aliasing=True, preserve_range=True)
                    else: # Fallback numpy scaling
                         scale_y = target_heatmap_size[0] / heatmap_h_unpadded if heatmap_h_unpadded > 0 else 1
                         scale_x = target_heatmap_size[1] / heatmap_w_unpadded if heatmap_w_unpadded > 0 else 1
                         y_indices = np.clip((np.arange(target_heatmap_size[0]) / scale_y), 0, heatmap_h_unpadded - 1).astype(int)
                         x_indices = np.clip((np.arange(target_heatmap_size[1]) / scale_x), 0, heatmap_w_unpadded - 1).astype(int)
                         resized_heatmap_patch = heatmap_unpadded[y_indices[:, None], x_indices]

                except Exception as e:
                    print(f"  Warning: Error upscaling/resizing patch heatmap layer {layer_idx} '{concept}': {e}. Skipping visualization.")
                    continue # Skip if upscaling fails

                # --- Plotting ---
                fig, ax = plt.subplots(figsize=(8, 8 * preview_h / max(1, preview_w))) # Avoid div by zero
                try:
                    ax.imshow(spatial_preview_image, extent=(0, preview_w, preview_h, 0))
                    # Calculate extent for the overlay using padding and actual dimensions
                    extent = (pad_left, pad_left + resized_w_actual, pad_top + resized_h_actual, pad_top)
                    im = ax.imshow(resized_heatmap_patch, alpha=heatmap_alpha, cmap=colormap, vmin=0, vmax=1, extent=extent)
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Max Probability")
                    ax.set_title(f"Patch Feature Overlay: '{concept}' - Layer {layer_idx}", fontsize=12)
                    ax.axis("off")

                    filepath = os.path.join(patch_dir, f"layer_{layer_idx:03d}_{concept}_patch_overlay.png")
                    plt.savefig(filepath, dpi=150, bbox_inches="tight")
                    all_saved_paths.append(filepath)
                    if generate_composite:
                        patch_feature_paths[concept][layer_idx] = filepath # Store path keyed by layer index
                except Exception as e:
                    print(f"    Error plotting/saving patch overlay layer {layer_idx} '{concept}': {e}")
                finally:
                    plt.close(fig) # Ensure figure is closed
    else:
        print("  Skipping patch feature visualization: Required grid/dimension info invalid or missing in feature_mapping.")
        print(f"    Got: patch_vis_grid={patch_vis_grid}, patch_unpadded_grid={patch_unpadded_grid}, resized_dims_wh={resized_dims_wh}")


    # --- 3. Visualize Newline Token Probabilities (Line Plots) ---
    print("  Generating newline feature line plots...")
    has_newline_data = any(
        isinstance(token_probs.get(l, {}).get("newline_feature"), dict) and token_probs[l]["newline_feature"]
        for l in layers_to_plot
    )

    if has_newline_data:
        max_row_overall = 0
        for layer_idx in layers_to_plot:
            newline_layer_data = token_probs.get(layer_idx, {}).get("newline_feature", {})
            if isinstance(newline_layer_data, dict):
                for concept_probs_dict in newline_layer_data.values():
                    if concept_probs_dict and isinstance(concept_probs_dict, dict):
                        current_max = max(k for k in concept_probs_dict.keys() if isinstance(k, int)) if any(isinstance(k, int) for k in concept_probs_dict.keys()) else -1
                        if current_max >= 0:
                            max_row_overall = max(max_row_overall, current_max)

        for concept in concepts:
            pbar_newline = tqdm(layers_to_plot, desc=f"Newline '{concept}' Plot", leave=False, ncols=100)
            plot_data_exists_for_concept = False # Track if any data plotted for this concept
            for layer_idx in pbar_newline:
                layer_data = token_probs.get(layer_idx, {}).get("newline_feature", {})
                newline_probs_concept = layer_data.get(concept) # Expected: {row_idx: prob}

                if not isinstance(newline_probs_concept, dict) or not newline_probs_concept:
                    continue # Skip if no data or not a dict

                # Filter for integer keys only and sort
                rows_probs = sorted([(r, p) for r, p in newline_probs_concept.items() if isinstance(r, int)])
                if not rows_probs:
                    continue # Skip if no valid integer rows found

                rows, probs = zip(*rows_probs) # Unzip into separate lists
                plot_data_exists_for_concept = True

                # --- Plotting ---
                fig, ax = plt.subplots(figsize=(8, 4))
                try:
                    ax.plot(rows, probs, marker='o', linestyle='-', color='green')
                    ax.set_xlabel("Row Index (Spatial Grid)")
                    ax.set_ylabel("Max Probability")
                    ax.set_title(f"Newline Feature Prob: '{concept}' - Layer {layer_idx}", fontsize=12)
                    ax.set_ylim(-0.05, 1.05)
                    # Set x-axis limits based on actual data or overall max
                    current_max_row = max(rows) if rows else 0
                    xlim_upper = max(max_row_overall, current_max_row) + 0.5
                    ax.set_xlim(-0.5, xlim_upper)
                    # Generate ticks up to the max relevant row index
                    ax.set_xticks(np.arange(0, int(xlim_upper) + 1))
                    ax.grid(True, linestyle=':', alpha=0.6)

                    for r, p in zip(rows, probs):
                        ax.text(r, p + 0.03, f"{p:.3f}", ha="center", va="bottom", fontsize=8)

                    filepath = os.path.join(newline_dir, f"layer_{layer_idx:03d}_{concept}_newline_plot.png")
                    plt.savefig(filepath, dpi=150, bbox_inches="tight")
                    all_saved_paths.append(filepath)
                except Exception as e:
                    print(f"    Error plotting/saving newline plot layer {layer_idx} '{concept}': {e}")
                finally:
                    plt.close(fig) # Ensure figure is closed

            # Optional: Print a message if no data was found for a concept across all layers
            # if not plot_data_exists_for_concept:
            #     print(f"  No valid newline data found to plot for concept '{concept}'.")

    else:
        print("  Skipping newline feature visualization: No valid newline data found for selected layers.")


    # --- 4. Generate Composite Images (using helper function defined elsewhere) ---
    if generate_composite:
        print("\n  Generating composite overview images...")
        # Ensure the helper function is available in the scope
        if 'create_composite_image' not in globals() and 'create_composite_image' not in locals():
             print("  ERROR: 'create_composite_image' helper function not defined. Cannot generate composite images.")
        else:
            for concept in concepts:
                # --- Create Composite for Base Features ---
                # Get paths ordered by layer index
                ordered_base_paths = [base_feature_paths[concept][l] for l in layers_to_plot if l in base_feature_paths[concept]]
                if ordered_base_paths:
                    print(f"    Creating composite base feature image for concept: '{concept}' ({len(ordered_base_paths)} layers)")
                    # We need the layers corresponding ONLY to the paths we have
                    layers_for_base_composite = [l for l in layers_to_plot if l in base_feature_paths[concept]]
                    composite_base_path = create_composite_image( # Call the helper
                        image_paths=ordered_base_paths,
                        layers=layers_for_base_composite,
                        output_filename=os.path.join(composite_dir, f"composite_base_{concept}.png"),
                        title=f"Base Feature Evolution: '{concept}'"
                        # Padding handled internally by helper
                    )
                    if composite_base_path:
                        all_saved_paths.append(composite_base_path)
                        print(f"      Saved composite base image to: {composite_base_path}")
                else:
                    print(f"    Skipping composite base image for concept '{concept}' (no individual images found/matched).")

                # --- Create Composite for Patch Features ---
                 # Get paths ordered by layer index
                ordered_patch_paths = [patch_feature_paths[concept][l] for l in layers_to_plot if l in patch_feature_paths[concept]]
                if ordered_patch_paths:
                    print(f"    Creating composite patch feature image for concept: '{concept}' ({len(ordered_patch_paths)} layers)")
                    # We need the layers corresponding ONLY to the paths we have
                    layers_for_patch_composite = [l for l in layers_to_plot if l in patch_feature_paths[concept]]
                    composite_patch_path = create_composite_image( # Call the helper
                        image_paths=ordered_patch_paths,
                        layers=layers_for_patch_composite,
                        output_filename=os.path.join(composite_dir, f"composite_patch_{concept}.png"),
                        title=f"Patch Feature Evolution: '{concept}'"
                         # Padding handled internally by helper
                    )
                    if composite_patch_path:
                        all_saved_paths.append(composite_patch_path)
                        print(f"      Saved composite patch image to: {composite_patch_path}")
                else:
                    print(f"    Skipping composite patch image for concept '{concept}' (no individual images found/matched).")

    print(f"\n--- Logit Lens Visualizations Generated. Total files saved: {len(all_saved_paths)} ---")
    # Optionally return the directory path as well or instead
    # print(f"Visualizations saved in: {output_dir}")
    return all_saved_paths

# ============================================
# Composite Image Creation Helper Function
# ============================================

def create_composite_image(
    image_paths: List[str],
    layers: List[int],
    output_filename: str,
    title: str,
    background_color: Tuple[int, int, int] = (255, 255, 255) # White background
    # Removed padding and label_padding from signature
) -> Optional[str]:
    """
    Creates a composite image grid from a list of individual image files.

    Arranges the images in a grid using internally defined padding values,
    adds labels (layer index) below each image, and saves the result to a file.

    Args:
        image_paths (List[str]): List of file paths to the individual images to combine.
                                 Assumes these correspond sequentially to the `layers` list.
        layers (List[int]): List of layer indices corresponding to each image path.
                            Used for labeling.
        output_filename (str): Full path where the composite image should be saved.
        title (str): Title to be placed at the top of the composite image.
        background_color (Tuple[int, int, int]): RGB tuple for the background color.

    Returns:
        Optional[str]: The path to the saved composite image if successful, otherwise None.
    """
    # <<< Define internal padding constants >>>
    padding = 10         # Pixel padding between images and around edges
    label_padding = 25   # Extra vertical space below each image for its label

    # (Input validation remains the same)
    if not image_paths:
        print(f"Error creating composite image '{output_filename}': No image paths provided.")
        return None
    if len(image_paths) != len(layers):
        # Match paths and layers based on layer number in filename if lengths mismatch
        print(f"Warning creating composite image '{output_filename}': Mismatch between image paths ({len(image_paths)}) and layers ({len(layers)}). Attempting to match.")
        path_layer_map = {}
        for p in image_paths:
            try:
                # Extract layer number like 'layer_012_'
                layer_num_str = os.path.basename(p).split('_')[1]
                path_layer_map[int(layer_num_str)] = p
            except (IndexError, ValueError):
                print(f"  Could not extract layer number from filename: {os.path.basename(p)}")
        # Rebuild lists based on provided layers
        matched_paths = [path_layer_map.get(l) for l in layers]
        # Filter out layers for which no path was found
        filtered_layers = [l for l, p in zip(layers, matched_paths) if p is not None]
        filtered_paths = [p for p in matched_paths if p is not None]

        if not filtered_paths:
            print(f"Error creating composite image '{output_filename}': No images could be matched to layers after mismatch.")
            return None

        image_paths = filtered_paths
        layers = filtered_layers # Use the filtered list of layers that have images
        print(f"  Proceeding with {len(layers)} matched images/layers.")


    try:
        # (Load first image to get dimensions - remains the same)
        with Image.open(image_paths[0]) as img:
            img_w, img_h = img.size
            img_mode = img.mode
    except FileNotFoundError:
        print(f"Error creating composite image: Could not find first image at {image_paths[0]}")
        return None
    except Exception as e:
        print(f"Error creating composite image: Failed to load first image {image_paths[0]}: {e}")
        return None

    # (Calculate grid dimensions - remains the same)
    num_images = len(image_paths) # Use potentially adjusted number of images
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # (Calculate cell and canvas dimensions using internal padding values - remains the same)
    cell_w = img_w + padding
    cell_h = img_h + padding + label_padding
    title_height = 50
    canvas_w = cols * cell_w + padding
    canvas_h = rows * cell_h + padding + title_height

    # (Create canvas and draw object - remains the same)
    canvas = Image.new(img_mode, (canvas_w, canvas_h), background_color)
    draw = ImageDraw.Draw(canvas)

    # (Add Title - remains the same)
    try:
        # Define default fonts if not already defined
        if 'DEFAULT_FONT' not in globals():
            try:
                DEFAULT_FONT = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                DEFAULT_FONT = ImageFont.load_default()

        if 'DEFAULT_FONT_SMALL' not in globals():
            try:
                DEFAULT_FONT_SMALL = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                DEFAULT_FONT_SMALL = ImageFont.load_default()
        title_bbox = draw.textbbox((0, 0), title, font=DEFAULT_FONT)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
    except AttributeError:
        title_w, title_h = draw.textlength(title, font=DEFAULT_FONT)
    title_x = (canvas_w - title_w) // 2
    title_y = padding
    draw.text((title_x, title_y), title, fill=(0, 0, 0), font=DEFAULT_FONT)

    # (Paste Images and Add Labels loop - remains the same, uses internal padding)
    current_col = 0
    current_row = 0
    # Iterate using the potentially filtered lists
    for i, (img_path, layer_idx) in enumerate(zip(image_paths, layers)):
        try:
            with Image.open(img_path) as img:
                if img.mode != canvas.mode:
                    img = img.convert(canvas.mode)

                paste_x = padding + current_col * cell_w
                paste_y = padding + current_row * cell_h + title_height

                canvas.paste(img, (paste_x, paste_y))

                label_text = f"Layer {layer_idx}"
                try:
                     label_bbox = draw.textbbox((0, 0), label_text, font=DEFAULT_FONT_SMALL)
                     label_w = label_bbox[2] - label_bbox[0]
                except AttributeError:
                     label_w = draw.textlength(label_text, font=DEFAULT_FONT_SMALL)
                label_x = paste_x + (img_w - label_w) // 2
                label_y = paste_y + img_h + (padding // 2)

                draw.text((label_x, label_y), label_text, fill=(50, 50, 50), font=DEFAULT_FONT_SMALL)

        except FileNotFoundError:
            print(f"  Warning: Could not find image file {img_path} for composite. Skipping.")
        except Exception as e:
            print(f"  Warning: Error processing image {img_path} for composite: {e}. Skipping.")

        current_col += 1
        if current_col >= cols:
            current_col = 0
            current_row += 1

    # (Save the final composite image - remains the same)
    try:
        canvas.save(output_filename)
        return output_filename
    except Exception as e:
        print(f"Error saving composite image to {output_filename}: {e}")
        return None
