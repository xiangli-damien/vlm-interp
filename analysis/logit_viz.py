# File: analysis/logit_viz.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Optional, Tuple


def create_composite_image(
    image_paths: List[str],
    layers: List[int],
    output_filename: str,
    title: str,
    background_color: Tuple[int,int,int] = (255,255,255)
) -> Optional[str]:
    """
    Create a grid of layer-by-layer images (e.g. heatmaps) and save as one composite.

    Args:
        image_paths:     a list of filepaths for each layer’s image (must align with layers)
        layers:          the layer indices corresponding to each image
        output_filename: the final composite’s save path
        title:           text to draw at the top of the canvas
        background_color: RGB background fill

    Returns:
        The path to the saved composite, or None on failure.
    """
    padding, label_pad, title_pad = 10, 20, 50

    # Validate inputs
    if not image_paths or len(image_paths) != len(layers):
        print(f"[Composite] mismatch paths vs layers: {len(image_paths)} vs {len(layers)}")
        return None

    # Load first image to infer size
    try:
        sample = Image.open(image_paths[0])
        w, h = sample.size
        mode = sample.mode
    except Exception as e:
        print(f"[Composite] failed to open sample image: {e}")
        return None

    # Compute grid dims
    n = len(image_paths)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w = w + padding
    cell_h = h + padding + label_pad
    canvas_w = cols * cell_w + padding
    canvas_h = rows * cell_h + padding + title_pad

    # Make canvas
    canvas = Image.new(mode, (canvas_w, canvas_h), background_color)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        small = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    # Draw title
    w_title, h_title = draw.textsize(title, font=font)
    draw.text(((canvas_w - w_title)//2, padding), title, fill="black", font=font)

    # Paste each image + label
    for idx, (path, layer) in enumerate(zip(image_paths, layers)):
        row, col = divmod(idx, cols)
        x0 = padding + col*cell_w
        y0 = padding + title_pad + row*cell_h

        try:
            img = Image.open(path).convert(mode)
            canvas.paste(img, (x0, y0))
            lbl = f"Layer {layer}"
            w_lbl, h_lbl = draw.textsize(lbl, font=small)
            lx = x0 + (w - w_lbl)//2
            ly = y0 + h + 5
            draw.text((lx, ly), lbl, fill="gray", font=small)
        except Exception as e:
            print(f"[Composite] skip {path}: {e}")

    # Save
    try:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        canvas.save(output_filename)
        return output_filename
    except Exception as e:
        print(f"[Composite] failed to save {output_filename}: {e}")
        return None


def visualize_token_probabilities(
    token_probs: Dict[int, Dict[str, Any]],
    feature_mapping: Dict[str, Any],
    original_image: Image.Image,
    spatial_preview: Image.Image,
    concepts: List[str],
    output_dir: str,
    colormap: str = "jet",
    alpha: float = 0.6,
    interpolation: str = "nearest",
    grid: bool = True,
    composite: bool = True
) -> List[str]:
    """
    Visualizes token-level probability distributions as heatmap overlays on either
    the base feature grid or the spatial patch grid. Optionally, it can also generate
    composite overview images for each concept.

    Args:
        token_probs:       A dictionary mapping layer index to token probabilities, split
                           into "base_feature" and "patch_feature" components per concept.
        feature_mapping:   Dictionary containing base and patch feature grid shapes and positions.
        original_image:    The original scene as a PIL Image used as the background for base overlay.
        spatial_preview:   Resized and padded image aligned with the patch feature grid.
        concepts:          A list of semantic concept strings to be visualized.
        output_dir:        Directory where overlay and composite images will be saved.
        colormap:          Name of the matplotlib colormap to use (default: "jet").
        alpha:             Transparency of the heatmap overlay (0.0–1.0).
        interpolation:     Interpolation method for `imshow` (default: "nearest").
        grid:              Whether to overlay grid lines for each cell (default: True).
        composite:         Whether to generate composite overview images (default: True).

    Returns:
        List of all saved image file paths.
    """
    try:
        # --- Input validation ---
        if not token_probs:
            logger.error("visualize_token_probabilities: Empty token_probs")
            return []
        if not feature_mapping:
            logger.error("visualize_token_probabilities: Empty feature_mapping")
            return []
        if not isinstance(original_image, Image.Image):
            logger.error("visualize_token_probabilities: original_image is not a PIL Image")
            return []
        if not isinstance(spatial_preview, Image.Image):
            logger.error("visualize_token_probabilities: spatial_preview is not a PIL Image")
            return []
        if not concepts:
            logger.error("visualize_token_probabilities: No concepts provided")
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved_paths: List[str] = []

        # --- Base feature overlays ---
        base_info = feature_mapping.get("base_feature", {})
        if not base_info:
            logger.warning("visualize_token_probabilities: No base_feature in feature_mapping")
        else:
            grid_h, grid_w = base_info.get("grid", (0, 0))
            if grid_h > 0 and grid_w > 0:
                try:
                    bg = original_image.resize((336, 336))
                    bg_np = np.array(bg)
                except Exception as e:
                    logger.error(f"Error resizing original image: {e}")
                    bg_np = np.zeros((336, 336, 3), dtype=np.uint8)

                for concept in concepts:
                    concept_dir = os.path.join(output_dir, "base", concept)
                    os.makedirs(concept_dir, exist_ok=True)
                    for layer, data in token_probs.items():
                        try:
                            arr = data.get("base_feature", {}).get(concept)
                            if arr is None or not isinstance(arr, np.ndarray) or arr.shape != (grid_h, grid_w):
                                continue

                            # Upsample to match 336x336 image using nearest neighbor (kron)
                            ry, rx = 336 // grid_h, 336 // grid_w
                            heat = np.kron(arr, np.ones((ry, rx)))[:336, :336]

                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.imshow(bg_np)
                            im = ax.imshow(heat, cmap=colormap, alpha=alpha,
                                           interpolation=interpolation,
                                           extent=(0, 336, 336, 0))
                            if grid:
                                for i in range(1, grid_h): ax.axhline(i * ry, color="white", lw=0.5)
                                for j in range(1, grid_w): ax.axvline(j * rx, color="white", lw=0.5)
                            ax.axis("off")
                            ax.set_title(f"Base '{concept}' L{layer}", fontsize=10)

                            path = os.path.join(concept_dir, f"layer_{layer:03d}.png")
                            fig.savefig(path, dpi=100, bbox_inches="tight")
                            plt.close(fig)
                            saved_paths.append(path)
                        except Exception as e:
                            logger.error(f"Error visualizing base feature for concept '{concept}', layer {layer}: {e}")

        # --- Patch feature overlays ---
        patch_info = feature_mapping.get("patch_feature", {})
        if not patch_info:
            logger.warning("visualize_token_probabilities: No patch_feature in feature_mapping")
        else:
            grid_up = patch_info.get("grid_unpadded", (0, 0))
            if grid_up[0] <= 0 or grid_up[1] <= 0:
                logger.warning(f"Invalid grid_unpadded: {grid_up}")
            else:
                resized_dims = feature_mapping.get("resized_dimensions", (0, 0))
                if resized_dims[0] <= 0 or resized_dims[1] <= 0:
                    logger.warning(f"Invalid resized_dimensions: {resized_dims}")
                else:
                    preview_w, preview_h = spatial_preview.size
                    pad_h = (preview_h - resized_dims[1]) // 2
                    pad_w = (preview_w - resized_dims[0]) // 2

                    for concept in concepts:
                        concept_dir = os.path.join(output_dir, "patch", concept)
                        os.makedirs(concept_dir, exist_ok=True)
                        for layer, data in token_probs.items():
                            try:
                                arr = data.get("patch_feature", {}).get(concept)
                                if arr is None or not isinstance(arr, np.ndarray) or arr.shape != grid_up:
                                    continue

                                raw = feature_mapping.get("patch_size", 1)
                                heat_un = np.repeat(np.repeat(arr, raw, 0), raw, 1)

                                h_target, w_target = resized_dims[1], resized_dims[0]
                                if heat_un.shape[0] == 0 or heat_un.shape[1] == 0 or h_target == 0 or w_target == 0:
                                    logger.warning(f"Invalid dimensions for resizing: heat_un={heat_un.shape}, target=({h_target}, {w_target})")
                                    continue

                                y_idx = np.clip((np.arange(h_target) / h_target * heat_un.shape[0]).astype(int), 0, heat_un.shape[0] - 1)
                                x_idx = np.clip((np.arange(w_target) / w_target * heat_un.shape[1]).astype(int), 0, heat_un.shape[1] - 1)
                                heat = heat_un[y_idx[:, None], x_idx]

                                fig, ax = plt.subplots(figsize=(6, 6 * preview_h / preview_w))
                                ax.imshow(spatial_preview, extent=(0, preview_w, preview_h, 0))
                                im = ax.imshow(heat, cmap=colormap, alpha=alpha,
                                               interpolation=interpolation,
                                               extent=(pad_w, pad_w + w_target, pad_h + h_target, pad_h))
                                if grid:
                                    cell_h = h_target / grid_up[0]
                                    cell_w = w_target / grid_up[1]
                                    for i in range(1, grid_up[0]): ax.axhline(pad_h + i * cell_h, color="white", lw=0.5)
                                    for j in range(1, grid_up[1]): ax.axvline(pad_w + j * cell_w, color="white", lw=0.5)
                                ax.axis("off")
                                ax.set_title(f"Patch '{concept}' L{layer}", fontsize=10)

                                path = os.path.join(concept_dir, f"layer_{layer:03d}.png")
                                fig.savefig(path, dpi=100, bbox_inches="tight")
                                plt.close(fig)
                                saved_paths.append(path)
                            except Exception as e:
                                logger.error(f"Error visualizing patch feature for concept '{concept}', layer {layer}: {e}")
                                import traceback
                                traceback.print_exc()

        # --- Composite image generation ---
        if composite and saved_paths:
            try:
                for concept in concepts:
                    bimgs = [p for p in saved_paths if f"/base/{concept}/" in p]
                    if bimgs:
                        blayers = [int(os.path.basename(p).split("_")[1]) for p in bimgs]
                        out = create_composite_image(
                            image_paths=bimgs,
                            layers=blayers,
                            output_filename=os.path.join(output_dir, f"composite_base_{concept}.png"),
                            title=f"Base Evolution: {concept}"
                        )
                        if out:
                            saved_paths.append(out)
                            logger.info(f"Created base composite image for concept '{concept}'")

                    pimgs = [p for p in saved_paths if f"/patch/{concept}/" in p]
                    if pimgs:
                        players = [int(os.path.basename(p).split("_")[1]) for p in pimgs]
                        out = create_composite_image(
                            image_paths=pimgs,
                            layers=players,
                            output_filename=os.path.join(output_dir, f"composite_patch_{concept}.png"),
                            title=f"Patch Evolution: {concept}"
                        )
                        if out:
                            saved_paths.append(out)
                            logger.info(f"Created patch composite image for concept '{concept}'")
            except Exception as e:
                logger.error(f"Error creating composite images: {e}")
                import traceback
                traceback.print_exc()

        return saved_paths

    except Exception as e:
        logger.error(f"Global error in visualize_token_probabilities: {e}")
        import traceback
        traceback.print_exc()
        return []
