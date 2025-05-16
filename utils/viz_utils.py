"""
Visualization utilities for VLM analysis results.

Includes functions for plotting:
- Information flow metrics across layers
- Attention weight heatmaps
- Processed image tensors fed into the vision encoder
- Logit lens token probability heatmaps and overlays
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import time
import traceback

# Try importing skimage for better heatmap resizing, with fallback
try:
    from skimage.transform import resize as skimage_resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. Falling back to simpler numpy-based resizing for heatmap visualization.")


def visualize_information_flow(
    metrics: Dict[int, Dict[str, float]],
    title: str = "VLM Information Flow Analysis",
    save_path: Optional[str] = None,
    use_top_k: bool = False
):
    """
    Visualizes information flow metrics (mean and sum) across model layers.

    Args:
        metrics: Dictionary where keys are layer indices and values are
                 dictionaries containing flow metrics
        title:   Main title for the plot figure
        save_path: Path to save the plot image (optional)
        use_top_k: Whether the metrics were computed using top-k image tokens
    """
    if not metrics:
        print("Warning: No metrics data provided to visualize_information_flow.")
        return

    # Define consistent markers and labels for flow types
    flow_styles = {
        "Siq_mean": {"marker": "o", "label": "Image→Target (Mean)"},
        "Stq_mean": {"marker": "^", "label": "Text→Target (Mean)"},
        "Sgq_mean": {"marker": "s", "label": "Generated→Target (Mean)"},
        "Siq_sum": {"marker": "o", "label": "Image→Target (Sum)", "linestyle": '--'},
        "Stq_sum": {"marker": "^", "label": "Text→Target (Sum)", "linestyle": '--'},
        "Sgq_sum": {"marker": "s", "label": "Generated→Target (Sum)", "linestyle": '--'}
    }

    layers = sorted(metrics.keys())
    available_keys = set().union(*(metrics[layer].keys() for layer in layers))

    # Build data for plotting
    plot_data = {
        key: [metrics[layer].get(key) for layer in layers]
        for key in flow_styles
        if key in available_keys
    }

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True)

    # Mean subplot
    ax1.set_title("Mean Information Flow per Layer")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean Attention / Saliency")
    for key in ["Siq_mean", "Stq_mean", "Sgq_mean"]:
        if key in plot_data:
            style = flow_styles[key]
            ax1.plot(layers, plot_data[key], marker=style["marker"], label=style["label"], linewidth=2)
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Sum subplot
    ax2.set_title("Total Information Flow per Layer")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Summed Attention / Saliency")
    for key in ["Siq_sum", "Stq_sum", "Sgq_sum"]:
        if key in plot_data:
            style = flow_styles[key]
            ax2.plot(
                layers, plot_data[key],
                marker=style["marker"], linestyle=style.get("linestyle", '-'),
                label=style["label"], linewidth=2
            )
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Overall title adjustment
    prefix = "Top-k " if use_top_k else ""
    fig.suptitle(f"{prefix}{title}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Save figure
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Information flow visualization saved to: {save_path}")
        except Exception as e:
            print(f"Error saving information flow plot to {save_path}: {e}")

    plt.show()
    plt.close(fig)

def create_composite_image(
    image_paths: List[str],
    layers: List[int],
    output_filename: str,
    title: str,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Optional[str]:
    """
    Creates a composite image grid from a list of individual image files.

    Args:
        image_paths: List of file paths to the individual images to combine
        layers: List of layer indices corresponding to each image path
        output_filename: Full path where the composite image should be saved
        title: Title to be placed at the top of the composite image
        background_color: RGB tuple for the background color

    Returns:
        The path to the saved composite image if successful, otherwise None
    """
    # Internal padding constants
    padding = 10
    label_padding = 25

    if not image_paths:
        print(f"Error creating composite image '{output_filename}': No image paths provided.")
        return None
        
    if len(image_paths) != len(layers):
        # Match paths and layers based on layer number in filename if lengths mismatch
        print(f"Warning creating composite image '{output_filename}': Mismatch between image paths ({len(image_paths)}) and layers ({len(layers)}). Attempting to match.")
        path_layer_map = {}
        for p in image_paths:
            try:
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
        layers = filtered_layers

    try:
        # Load first image to get dimensions
        with Image.open(image_paths[0]) as img:
            img_w, img_h = img.size
            img_mode = img.mode
    except FileNotFoundError:
        print(f"Error creating composite image: Could not find first image at {image_paths[0]}")
        return None
    except Exception as e:
        print(f"Error creating composite image: Failed to load first image {image_paths[0]}: {e}")
        return None

    # Calculate grid dimensions
    num_images = len(image_paths)
    cols = math.ceil(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    # Calculate cell and canvas dimensions
    cell_w = img_w + padding
    cell_h = img_h + padding + label_padding
    title_height = 50
    canvas_w = cols * cell_w + padding
    canvas_h = rows * cell_h + padding + title_height

    # Create canvas and draw object
    canvas = Image.new(img_mode, (canvas_w, canvas_h), background_color)
    draw = ImageDraw.Draw(canvas)

    # Define default fonts
    try:
        DEFAULT_FONT = ImageFont.truetype("arial.ttf", 18)
        DEFAULT_FONT_SMALL = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        DEFAULT_FONT = ImageFont.load_default()
        DEFAULT_FONT_SMALL = ImageFont.load_default()

    # Add Title
    try:
        title_bbox = draw.textbbox((0, 0), title, font=DEFAULT_FONT)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
    except AttributeError:
        title_w, title_h = draw.textlength(title, font=DEFAULT_FONT), 20
        
    title_x = (canvas_w - title_w) // 2
    title_y = padding
    draw.text((title_x, title_y), title, fill=(0, 0, 0), font=DEFAULT_FONT)

    # Paste Images and Add Labels
    current_col = 0
    current_row = 0
    
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

    # Save the final composite image
    try:
        canvas.save(output_filename)
        return output_filename
    except Exception as e:
        print(f"Error saving composite image to {output_filename}: {e}")
        return None


def visualize_token_probabilities(
    token_probs: Dict[int, Dict[str, Any]],
    input_data: Dict[str, Any],
    selected_layers: Optional[List[int]] = None,
    output_dir: str = "logit_lens_visualization",
    colormap: str = "jet",
    heatmap_alpha: float = 0.6,
    interpolation: str = "nearest",
    add_gridlines: bool = True,
    generate_composite: bool = True,
    only_composite: bool = False
) -> List[str]:
    """
    Visualize token probability maps from logit lens analysis using heatmaps and line plots.

    Args:
        token_probs: Dictionary mapping layer index to probabilities
        input_data: Dictionary from the analyzer's prepare_inputs
        selected_layers: List of layer indices to visualize
        output_dir: Directory path to save the visualization images
        colormap: Matplotlib colormap name for heatmaps
        heatmap_alpha: Alpha blending value for heatmap overlays
        interpolation: Interpolation method for imshow
        add_gridlines: Whether to add grid lines to visualizations
        generate_composite: Create composite images combining layer visualizations
        only_composite: Only keep composite images

    Returns:
        File paths of all saved visualization images
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
    spatial_preview_image = input_data["spatial_preview_image"]
    original_image_pil = input_data["original_image"]

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
        layers_to_plot = sorted([l for l in selected_layers if isinstance(l, int) and l in available_layers])
        if not layers_to_plot:
             print(f"  Warning: None of the selected layers {selected_layers} have data in token_probs. Visualizing all available layers instead ({available_layers}).")
             layers_to_plot = available_layers

    # Infer concepts from the first layer's data
    concepts = set()
    first_valid_layer_data = token_probs.get(layers_to_plot[0], {})

    # Check various feature dictionaries
    base_feat_data = first_valid_layer_data.get("base_feature")
    if isinstance(base_feat_data, dict):
        concepts.update(k for k in base_feat_data.keys() if isinstance(k, str))
        
    patch_feat_data = first_valid_layer_data.get("patch_feature")
    if isinstance(patch_feat_data, dict):
        concepts.update(k for k in patch_feat_data.keys() if isinstance(k, str))
        
    newline_feat_data = first_valid_layer_data.get("newline_feature")
    if isinstance(newline_feat_data, dict):
        concepts.update(k for k in newline_feat_data.keys() if isinstance(k, str))

    concepts = sorted(list(concepts))

    if not concepts:
        print("  Error: No concepts (string keys) found in the token probability data for the first layer. Cannot visualize.")
        return []

    print(f"  Visualizing for Layers: {layers_to_plot}")
    print(f"  Visualizing for Concepts: {concepts}")

    # --- Prepare Output Directories and Storage for Paths ---
    all_saved_paths = []
    base_feature_paths = {concept: {} for concept in concepts}
    patch_feature_paths = {concept: {} for concept in concepts}

    # Define subdirectories
    base_dir = os.path.join(output_dir, "base_feature_overlays")
    patch_dir = os.path.join(output_dir, "patch_feature_overlays")
    newline_dir = os.path.join(output_dir, "newline_feature_plots")
    composite_dir = os.path.join(output_dir, "composite_views")

    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(newline_dir, exist_ok=True)
    if generate_composite:
        os.makedirs(composite_dir, exist_ok=True)

    # --- 1. Visualize Base Feature Heatmaps ---
    print("  Generating base feature overlay heatmaps...")
    base_feature_map_info = feature_mapping.get("base_feature", {})
    base_grid = base_feature_map_info.get("grid")

    if base_grid and isinstance(base_grid, tuple) and len(base_grid) == 2:
        base_grid_h, base_grid_w = base_grid
        target_overlay_size = (336, 336)
        try:
            resized_background = original_image_pil.resize(target_overlay_size, Image.Resampling.LANCZOS)
            background_np = np.array(resized_background)
        except Exception as e:
            print(f"  Error resizing original image for base background: {e}. Skipping base overlays.")
            background_np = None

        if background_np is not None:
            for concept in concepts:
                pbar_base = tqdm(layers_to_plot, desc=f"Base '{concept}' Overlay", leave=False, ncols=100)
                for layer_idx in pbar_base:
                    layer_data = token_probs.get(layer_idx, {}).get("base_feature", {})
                    base_prob_map = layer_data.get(concept)

                    if base_prob_map is None or not isinstance(base_prob_map, np.ndarray) or base_prob_map.size == 0:
                        continue
                    if base_prob_map.shape != (base_grid_h, base_grid_w):
                        print(f"  Warning: Shape mismatch base layer {layer_idx}, '{concept}'. Expected {base_grid}, got {base_prob_map.shape}. Skipping.")
                        continue

                    # Simplified upscaling: repeat each heatmap cell into a block
                    repeat_y = target_overlay_size[1] // base_grid_h
                    repeat_x = target_overlay_size[0] // base_grid_w
                    upscaled_heatmap_base = np.kron(
                        base_prob_map,
                        np.ones((repeat_y, repeat_x), dtype=base_prob_map.dtype)
                    )
                    # trim to exact target size
                    upscaled_heatmap_base = upscaled_heatmap_base[
                        :target_overlay_size[1],
                        :target_overlay_size[0]
                    ]


                    # Plotting
                    fig, ax = plt.subplots(figsize=(6, 6))
                    try:
                        ax.imshow(background_np, extent=(0, target_overlay_size[0], target_overlay_size[1], 0))
                        
                        im = ax.imshow(upscaled_heatmap_base, alpha=heatmap_alpha, cmap=colormap, 
                                      vmin=0, vmax=1, 
                                      extent=(0, target_overlay_size[0], target_overlay_size[1], 0),
                                      interpolation=interpolation)
                        
                        # Add grid lines if requested
                        if add_gridlines:
                            cell_height = target_overlay_size[1] / base_grid_h
                            cell_width = target_overlay_size[0] / base_grid_w
                            
                            # Add horizontal grid lines
                            for i in range(1, base_grid_h):
                                y = i * cell_height
                                ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                                
                            # Add vertical grid lines
                            for i in range(1, base_grid_w):
                                x = i * cell_width
                                ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                        
                        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        cbar.set_label("Max Probability")
                        ax.set_title(f"Base Feature Overlay: '{concept}' - Layer {layer_idx}", fontsize=12)
                        ax.axis("off")

                        filepath = os.path.join(base_dir, f"layer_{layer_idx:03d}_{concept}_base_overlay.png")
                        plt.savefig(filepath, dpi=150, bbox_inches="tight")
                        all_saved_paths.append(filepath)
                        if generate_composite:
                           base_feature_paths[concept][layer_idx] = filepath
                    except Exception as e:
                        print(f"    Error plotting/saving base overlay layer {layer_idx} '{concept}': {e}")
                    finally:
                        plt.close(fig)
    else:
        print("  Skipping base feature visualization: Grid info invalid or missing in feature_mapping.")

    # --- 2. Visualize Spatial (Patch) Feature Heatmaps ---
    print("  Generating patch feature overlay heatmaps...")
    patch_feature_map_info = feature_mapping.get("patch_feature", {})
    patch_vis_grid = patch_feature_map_info.get("grid_for_visualization")
    patch_unpadded_grid = patch_feature_map_info.get("grid_unpadded")
    resized_dims_wh = feature_mapping.get("resized_dimensions")

    if (patch_vis_grid and isinstance(patch_vis_grid, tuple) and len(patch_vis_grid) == 2 and
        patch_unpadded_grid and isinstance(patch_unpadded_grid, tuple) and len(patch_unpadded_grid) == 2 and
        resized_dims_wh and isinstance(resized_dims_wh, tuple) and len(resized_dims_wh) == 2):

        prob_grid_h, prob_grid_w = patch_unpadded_grid
        resized_w_actual, resized_h_actual = resized_dims_wh
        preview_w, preview_h = spatial_preview_image.size

        # Calculate padding
        pad_h_total = preview_h - resized_h_actual
        pad_w_total = preview_w - resized_w_actual
        pad_top = max(0, pad_h_total // 2)
        pad_left = max(0, pad_w_total // 2)

        for concept in concepts:
            pbar_patch = tqdm(layers_to_plot, desc=f"Patch '{concept}' Overlay", leave=False, ncols=100)
            for layer_idx in pbar_patch:
                layer_data = token_probs.get(layer_idx, {}).get("patch_feature", {})
                patch_prob_map_unpadded = layer_data.get(concept)

                if patch_prob_map_unpadded is None or not isinstance(patch_prob_map_unpadded, np.ndarray) or patch_prob_map_unpadded.size == 0:
                    continue
                if patch_prob_map_unpadded.shape != (prob_grid_h, prob_grid_w):
                    print(f"  Warning: Shape mismatch patch layer {layer_idx}, '{concept}'. Expected {patch_unpadded_grid}, got {patch_prob_map_unpadded.shape}. Skipping.")
                    continue

                # Upscale and Resize Heatmap
                resized_heatmap_patch: Optional[np.ndarray] = None
                try:
                    # Upscale probability map by raw patch size first
                    heatmap_unpadded = np.repeat(np.repeat(patch_prob_map_unpadded, raw_patch_size, axis=0), raw_patch_size, axis=1)
                    heatmap_h_unpadded, heatmap_w_unpadded = heatmap_unpadded.shape

                    # Resize to match the actual content area dimensions
                    target_heatmap_size = (resized_h_actual, resized_w_actual)
                    if HAS_SKIMAGE:
                        resized_heatmap_patch = skimage_resize(heatmap_unpadded, target_heatmap_size, order=1, 
                                                             mode='constant', cval=0, anti_aliasing=True, 
                                                             preserve_range=True)
                    else:
                         scale_y = target_heatmap_size[0] / heatmap_h_unpadded if heatmap_h_unpadded > 0 else 1
                         scale_x = target_heatmap_size[1] / heatmap_w_unpadded if heatmap_w_unpadded > 0 else 1
                         y_indices = np.clip((np.arange(target_heatmap_size[0]) / scale_y), 0, heatmap_h_unpadded - 1).astype(int)
                         x_indices = np.clip((np.arange(target_heatmap_size[1]) / scale_x), 0, heatmap_w_unpadded - 1).astype(int)
                         resized_heatmap_patch = heatmap_unpadded[y_indices[:, None], x_indices]

                except Exception as e:
                    print(f"  Warning: Error upscaling/resizing patch heatmap layer {layer_idx} '{concept}': {e}. Skipping visualization.")
                    continue

                # Plotting
                fig, ax = plt.subplots(figsize=(8, 8 * preview_h / max(1, preview_w)))
                try:
                    ax.imshow(spatial_preview_image, extent=(0, preview_w, preview_h, 0))
                    # Calculate extent for the overlay using padding and actual dimensions
                    extent = (pad_left, pad_left + resized_w_actual, pad_top + resized_h_actual, pad_top)
                    
                    im = ax.imshow(resized_heatmap_patch, alpha=heatmap_alpha, cmap=colormap, 
                                  vmin=0, vmax=1, extent=extent,
                                  interpolation=interpolation)
                    
                    # Add grid lines if requested
                    if add_gridlines:
                        # Calculate cell sizes for the patch grid
                        cell_height = resized_h_actual / prob_grid_h
                        cell_width = resized_w_actual / prob_grid_w
                        
                        # Add horizontal grid lines
                        for i in range(1, prob_grid_h):
                            y = pad_top + i * cell_height
                            ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                            
                        # Add vertical grid lines
                        for i in range(1, prob_grid_w):
                            x = pad_left + i * cell_width
                            ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                    
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label("Max Probability")
                    ax.set_title(f"Patch Feature Overlay: '{concept}' - Layer {layer_idx}", fontsize=12)
                    ax.axis("off")

                    filepath = os.path.join(patch_dir, f"layer_{layer_idx:03d}_{concept}_patch_overlay.png")
                    plt.savefig(filepath, dpi=150, bbox_inches="tight")
                    all_saved_paths.append(filepath)
                    if generate_composite:
                        patch_feature_paths[concept][layer_idx] = filepath
                except Exception as e:
                    print(f"    Error plotting/saving patch overlay layer {layer_idx} '{concept}': {e}")
                finally:
                    plt.close(fig)
    else:
        print("  Skipping patch feature visualization: Required grid/dimension info invalid or missing in feature_mapping.")
        print(f"    Got: patch_vis_grid={patch_vis_grid}, patch_unpadded_grid={patch_unpadded_grid}, resized_dims_wh={resized_dims_wh}")

    # --- 3. Generate Composite Images ---
    if generate_composite:
        print("\n  Generating composite overview images...")
        
        for concept in concepts:
            # Create Composite for Base Features
            ordered_base_paths = [base_feature_paths[concept][l] for l in layers_to_plot if l in base_feature_paths[concept]]
            if ordered_base_paths:
                print(f"    Creating composite base feature image for concept: '{concept}' ({len(ordered_base_paths)} layers)")
                # Get layers corresponding to the paths we have
                layers_for_base_composite = [l for l in layers_to_plot if l in base_feature_paths[concept]]
                composite_base_path = create_composite_image(
                    image_paths=ordered_base_paths,
                    layers=layers_for_base_composite,
                    output_filename=os.path.join(composite_dir, f"composite_base_{concept}.png"),
                    title=f"Base Feature Evolution: '{concept}'"
                )
                if composite_base_path:
                    all_saved_paths.append(composite_base_path)
                    print(f"      Saved composite base image to: {composite_base_path}")
            else:
                print(f"    Skipping composite base image for concept '{concept}' (no individual images found/matched).")

            # Create Composite for Patch Features
            ordered_patch_paths = [patch_feature_paths[concept][l] for l in layers_to_plot if l in patch_feature_paths[concept]]
            if ordered_patch_paths:
                print(f"    Creating composite patch feature image for concept: '{concept}' ({len(ordered_patch_paths)} layers)")
                # Get layers corresponding to the paths we have
                layers_for_patch_composite = [l for l in layers_to_plot if l in patch_feature_paths[concept]]
                composite_patch_path = create_composite_image(
                    image_paths=ordered_patch_paths,
                    layers=layers_for_patch_composite,
                    output_filename=os.path.join(composite_dir, f"composite_patch_{concept}.png"),
                    title=f"Patch Feature Evolution: '{concept}'"
                )
                if composite_patch_path:
                    all_saved_paths.append(composite_patch_path)
                    print(f"      Saved composite patch image to: {composite_patch_path}")
            else:
                print(f"    Skipping composite patch image for concept '{concept}' (no individual images found/matched).")

    print(f"\n--- Logit Lens Visualizations Generated. Total files saved: {len(all_saved_paths)} ---")
    
    # If only_composite is True and composites were generated, keep only composite images
    if only_composite and generate_composite:
        print("  Only keeping composite images as requested...")
        # Identify composite images
        composite_paths = [p for p in all_saved_paths if "composite" in os.path.basename(p)]
        if composite_paths:
            # Delete all non-composite images
            for path in all_saved_paths:
                if path not in composite_paths:
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"    Warning: Failed to delete {path}: {e}")
            # Update the list of saved paths to only include composites
            all_saved_paths = composite_paths
        else:
            print("    Warning: No composite images found to keep.")
    
    return all_saved_paths