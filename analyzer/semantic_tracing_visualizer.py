import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import requests
import io
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from collections import defaultdict


class EnhancedSemanticTracingVisualizer:
    """
    Visualizes semantic tracing results from CSV data, creating interactive flow graphs and heatmaps.
    Provides improved visualization with interactive exploration features.
    """
    
    def __init__(
        self,
        output_dir: str = "semantic_tracing_visualizations",
        debug_mode: bool = False,
    ):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
            debug_mode: Whether to print additional debug information
        """
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        self.unified_colorscale = False
        self.include_all_token_types = False
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
    
    def download_image(self, image_url: str, save_dir: str) -> Optional[str]:
        """
        Download image from URL and save it locally.
        
        Args:
            image_url: URL of the image
            save_dir: Directory to save the downloaded image
            
        Returns:
            Local path to the downloaded image or None if download failed
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Extract filename from URL or use a default name
            if '/' in image_url:
                filename = image_url.split('/')[-1]
                # Remove query parameters if any
                if '?' in filename:
                    filename = filename.split('?')[0]
            else:
                filename = "downloaded_image.jpg"
            
            # Ensure the filename has an extension
            if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                filename += '.jpg'
            
            local_path = os.path.join(save_dir, filename)
            
            # Check if already downloaded
            if os.path.exists(local_path):
                print(f"Using previously downloaded image at: {local_path}")
                return local_path
            
            # Download the image
            print(f"Downloading image from {image_url}...")
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Save the image
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Image saved to {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def get_local_image_path(self, image_path: str, save_dir: str) -> Optional[str]:
        """
        Ensure we have a local image path, downloading if necessary.
        
        Args:
            image_path: Path or URL to the image
            save_dir: Directory to save downloaded images
            
        Returns:
            Local path to the image or None if invalid
        """
        # Check if it's a URL
        if image_path.startswith(('http://', 'https://')):
            # Download the image
            download_dir = os.path.join(save_dir, "downloaded_images")
            return self.download_image(image_path, download_dir)
        
        # Check if it's a valid local path
        elif os.path.exists(image_path):
            return image_path
            
        else:
            print(f"Warning: Image path is invalid: {image_path}")
            return None
    
    
    def create_heatmaps_from_csv(
        self,
        trace_data: pd.DataFrame,
        target_text: str,
        target_idx: int,
        image_path: str,
        save_dir: str,
        feature_mapping: Dict[str, Any],
        use_grid_visualization: bool = True,
        show_values: bool = True,
        composite_only: bool = False,
        unified_colorscale: bool = False
    ) -> List[str]:
        """
        Create enhanced heatmap visualizations with improved image overlay and unified colorscale options.
        
        Args:
            trace_data: DataFrame of token-trace records
            target_text: Text of the target token
            target_idx: Index of the target token
            image_path: Path to original image
            save_dir: Directory to save output files
            feature_mapping: Metadata mapping for features
            use_grid_visualization: Whether to use grid overlay
            show_values: Whether to annotate cell values
            composite_only: If True, skip individual-layer maps
            unified_colorscale: Whether to use same color scale across all layers
            
        Returns:
            list of saved file paths
        """
        saved_paths = []
        
        # Set unified colorscale property
        self.unified_colorscale = unified_colorscale

        # Create output directories
        base_dir = os.path.join(save_dir, "base_heatmaps")
        patch_dir = os.path.join(save_dir, "patch_heatmaps")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(patch_dir, exist_ok=True)

        # Load the original image with robust error handling
        try:
            img = Image.open(image_path)
            # Ensure RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Create a preview version for patch visualization
            preview = img.copy()
            if "resized_dimensions" in feature_mapping:
                w, h = feature_mapping["resized_dimensions"]
                preview = preview.resize((w, h), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading image: {e}")
            # Create a placeholder image
            img = Image.new('RGB', (336, 336), color='gray')
            preview = img.copy()

        # Get all unique layers in the trace data 
        layers = sorted(trace_data["layer"].unique())
        
        # Dictionaries to store heatmaps for each layer
        layer_base_maps = {}
        layer_patch_maps = {}
        
        # Calculate global max value for unified color scale if requested
        if unified_colorscale:
            # Get max importance weight across all layers for image tokens
            img_tokens = trace_data[trace_data["token_type"] == 2]
            global_max = img_tokens["importance_weight"].max() if not img_tokens.empty else 1.0
            print(f"Using unified color scale with global maximum: {global_max:.4f}")
        else:
            global_max = None

        # Process each layer separately
        for L in layers:
            # Get just the data for this layer
            dfL = trace_data[trace_data["layer"] == L]
            
            # Filter for image tokens (token_type = 2)
            img_toks = dfL[dfL["token_type"] == 2]
            
            if img_toks.empty:
                print(f"Layer {L}: No image tokens found")
                layer_base_maps[L] = None
                layer_patch_maps[L] = None
                continue

            # Use importance_weight column
            if "importance_weight" in img_toks.columns:
                tok_weights = img_toks["importance_weight"].values
            else:
                tok_weights = img_toks["predicted_top_prob"].values
                print(f"Layer {L}: importance_weight column not found, using predicted_top_prob")
                
            layer_max = tok_weights.max() if len(tok_weights) > 0 else 0
            print(f"Layer {L}: Max weight = {layer_max:.4f}, Tokens = {len(img_toks)}")
            
            if layer_max <= 0:
                print(f"Layer {L}: All weights are zero")
                layer_base_maps[L] = None
                layer_patch_maps[L] = None
                continue
                
            # Create normalized weights dictionary for this layer
            # Normalize by global max if using unified color scale, otherwise by layer max
            normalization_factor = global_max if unified_colorscale else layer_max
            weights = {
                int(r.token_index): r.importance_weight / normalization_factor
                for _, r in img_toks.iterrows()
            }

            # 1. Process base feature map
            bf = feature_mapping.get("base_feature", {})
            if bf.get("grid") and bf.get("positions"):
                gh, gw = bf["grid"]
                heat = np.zeros((gh, gw), float)
                mapped_count = 0
                
                for tid, w in weights.items():
                    # Convert string positions back to integers
                    pos = None
                    if str(tid) in bf["positions"]:
                        pos = bf["positions"][str(tid)]
                    elif tid in bf["positions"]:
                        pos = bf["positions"][tid]
                        
                    if pos:
                        r0, c0 = pos
                        if 0 <= r0 < gh and 0 <= c0 < gw:
                            heat[r0, c0] = w
                            mapped_count += 1
                
                # Only use heatmap if it has meaningful data
                if mapped_count > 0 and heat.max() > 0:
                    layer_base_maps[L] = heat
                    if not composite_only:
                        p = self._create_enhanced_base_feature_overlay(
                            heatmap=heat,
                            original_image=img,
                            grid_size=(gh, gw),
                            layer_idx=L,
                            target_idx=target_idx,
                            title=f"Base Influence L{L}",
                            save_path=os.path.join(base_dir, f"base_L{L}.png"),
                            use_grid_visualization=use_grid_visualization,
                            show_values=show_values,
                            vmax=1.0 if unified_colorscale else None  # Use 1.0 for unified scale
                        )
                        if p:
                            saved_paths.append(p)
                            print(f"Created base feature overlay for layer {L}")
                else:
                    print(f"Layer {L}: No valid base feature data (mapped {mapped_count} tokens)")
                    layer_base_maps[L] = None
            else:
                print(f"Layer {L}: Missing base feature mapping information")
                layer_base_maps[L] = None

            # 2. Process patch feature map
            pf = feature_mapping.get("patch_feature", {})
            if pf.get("grid_unpadded") and pf.get("positions"):
                gh, gw = pf["grid_unpadded"]
                heat = np.zeros((gh, gw), float)
                mapped_count = 0
                
                for tid, w in weights.items():
                    # Convert string positions back to integers
                    pos = None
                    if str(tid) in pf["positions"]:
                        pos = pf["positions"][str(tid)]
                    elif tid in pf["positions"]:
                        pos = pf["positions"][tid]
                        
                    if pos:
                        r0, c0 = pos
                        if 0 <= r0 < gh and 0 <= c0 < gw:
                            heat[r0, c0] = w
                            mapped_count += 1
                
                # Only use heatmap if it has meaningful data
                if mapped_count > 0 and heat.max() > 0:
                    layer_patch_maps[L] = heat
                    if not composite_only:
                        p = self._create_enhanced_patch_feature_overlay(
                            heatmap=heat,
                            spatial_preview_image=preview,
                            feature_mapping=feature_mapping,
                            patch_size=feature_mapping.get("patch_size", 14),
                            layer_idx=L,
                            target_idx=target_idx,
                            title=f"Patch Influence L{L}",
                            save_path=os.path.join(patch_dir, f"patch_L{L}.png"),
                            show_values=show_values,
                            vmax=1.0 if unified_colorscale else None  # Use 1.0 for unified scale
                        )
                        if p:
                            saved_paths.append(p)
                            print(f"Created patch feature overlay for layer {L}")
                else:
                    print(f"Layer {L}: No valid patch feature data (mapped {mapped_count} tokens)")
                    layer_patch_maps[L] = None
            else:
                print(f"Layer {L}: Missing patch feature mapping information")
                layer_patch_maps[L] = None

        # Create composite grid visualizations if we have valid maps
        valid_base = [L for L, hm in layer_base_maps.items() if hm is not None]
        if valid_base:
            out = os.path.join(save_dir, f"composite_base_grid_{target_idx}.png")
            p = self._create_enhanced_grid_composite(
                heatmap_maps=layer_base_maps,
                layers=valid_base,
                title=f"Base Influence per Layer for Token {target_idx}",
                save_path=out,
                cmap="hot",
                show_values=show_values,
                unified_colorscale=unified_colorscale
            )
            if p:
                saved_paths.append(p)
                print(f"Created composite base grid with {len(valid_base)} layers")

        valid_patch = [L for L, hm in layer_patch_maps.items() if hm is not None]
        if valid_patch:
            out = os.path.join(save_dir, f"composite_patch_grid_{target_idx}.png")
            p = self._create_enhanced_grid_composite(
                heatmap_maps=layer_patch_maps,
                layers=valid_patch,
                title=f"Patch Influence per Layer for Token {target_idx}",
                save_path=out,
                cmap="hot",
                show_values=show_values,
                unified_colorscale=unified_colorscale
            )
            if p:
                saved_paths.append(p)
                print(f"Created composite patch grid with {len(valid_patch)} layers")

        return saved_paths
        
    def _create_enhanced_base_feature_overlay(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image,
        grid_size: Tuple[int, int],
        layer_idx: int,
        target_idx: int,
        title: str,
        save_path: str,
        target_size: Tuple[int, int] = (336, 336),
        colormap: str = "hot",
        alpha: float = 0.7,
        add_gridlines: bool = True,
        use_grid_visualization: bool = True,
        show_values: bool = True,
        vmax: Optional[float] = None
    ) -> Optional[str]:
        """
        Create an enhanced heatmap overlay visualization for base image features.
        
        Args:
            heatmap: 2D numpy array with heatmap values
            original_image: Original PIL image
            grid_size: Tuple of (height, width) for the grid
            layer_idx: Index of the layer
            target_idx: Index of the target token
            title: Title for the plot
            save_path: Path to save the visualization
            target_size: Target size for the visualization
            colormap: Matplotlib colormap name
            alpha: Alpha blending value for overlay
            add_gridlines: Whether to add grid lines
            use_grid_visualization: Whether to use grid-based visualization
            show_values: Whether to show numeric values in cells
            vmax: Maximum value for colormap scaling (None for auto-scaling)
            
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            
            # Create a new figure with transparent background for overlay
            fig = Figure(figsize=(8, 8), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Convert and resize image for background
            if original_image.mode != 'RGB':
                resized_background = original_image.convert('RGB').resize(target_size, Image.LANCZOS)
            else:
                resized_background = original_image.resize(target_size, Image.LANCZOS)
            
            background_np = np.array(resized_background)
            
            # Display background image with proper extent
            ax.imshow(
                background_np, 
                extent=(0, target_size[0], target_size[1], 0),
                aspect='auto',
                origin='upper'
            )
            
            # Calculate cell dimensions
            grid_h, grid_w = grid_size
            cell_height = target_size[1] / grid_h
            cell_width = target_size[0] / grid_w
            
            if use_grid_visualization:
                # Create a colored grid overlay with improved visibility
                base_alpha = min(0.2, alpha)  # Limit maximum alpha
                cmap = plt.get_cmap(colormap)
                
                # Draw grid cells with color based on heatmap value
                for r in range(grid_h):
                    for c in range(grid_w):
                        # Get value for this cell
                        cell_value = heatmap[r, c] if r < len(heatmap) and c < len(heatmap[0]) else 0
                        
                        if cell_value > 0:  # Only draw cells with influence
                            # Calculate cell boundaries
                            x_start = c * cell_width
                            y_start = r * cell_height
                            
                            # Get color with adjusted alpha based on value
                            cell_color = cmap(cell_value)
                            cell_alpha = min(cell_value * 1.3, base_alpha)
                            
                            # Create rectangle with appropriate alpha
                            rect = plt.Rectangle(
                                (x_start, y_start),
                                cell_width, cell_height,
                                color=cell_color,
                                alpha=cell_alpha,
                                linewidth=0.5 if cell_value > 0.4 else 0
                            )
                            ax.add_patch(rect)
                            
                            # Optionally add text showing value
                            if show_values and cell_value > 0.25:
                                cell_center_x = x_start + cell_width / 2
                                cell_center_y = y_start + cell_height / 2
                                ax.text(
                                    cell_center_x, cell_center_y,
                                    f"{cell_value:.2f}",
                                    ha='center', va='center',
                                    color='white' if cell_value > 0.5 else 'black',
                                    fontsize=8,
                                    bbox=dict(facecolor='none', alpha=0, pad=0)
                                )
            else:
                # Use smoother heatmap overlay with improved interpolation
                try:
                    from skimage.transform import resize as skimage_resize
                    upscaled_heatmap = skimage_resize(
                        heatmap, target_size, order=1, mode='constant', 
                        cval=0, anti_aliasing=True, preserve_range=True
                    )
                except ImportError:
                    # Simple upscaling method
                    repeat_y = target_size[1] // grid_h
                    repeat_x = target_size[0] // grid_w
                    upscaled_heatmap = np.kron(heatmap, np.ones((repeat_y, repeat_x)))
                    upscaled_heatmap = upscaled_heatmap[:target_size[1], :target_size[0]]
                
                # Use adjusted alpha for better overlay
                im = ax.imshow(
                    upscaled_heatmap, 
                    alpha=min(0.6, alpha),
                    cmap=colormap,
                    vmin=0,
                    vmax=vmax,  # Use provided vmax if available
                    extent=(0, target_size[0], target_size[1], 0),
                    interpolation="bilinear"  # Better interpolation
                )
            
            # Add grid lines if requested (with improved visibility)
            if add_gridlines:
                # Horizontal grid lines
                for i in range(grid_h + 1):
                    y = i * cell_height
                    ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5, alpha=0.6)
                
                # Vertical grid lines
                for i in range(grid_w + 1):
                    x = i * cell_width
                    ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5, alpha=0.6)
            
            # Add colorbar with improved formatting
            norm = mpl.colors.Normalize(vmin=0, vmax=vmax if vmax is not None else heatmap.max())
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Influence Weight")
            
            # Set title with improved formatting
            ax.set_title(title, fontsize=12, pad=10)
            ax.axis("off")
            
            # Save figure with improved quality
            fig.tight_layout()
            canvas.draw()
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            
            return save_path
        
        except Exception as e:
            print(f"Error creating enhanced base feature overlay: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_enhanced_patch_feature_overlay(
        self,
        heatmap: np.ndarray,
        spatial_preview_image: Image.Image,
        feature_mapping: Dict[str, Any],
        patch_size: int,
        layer_idx: int,
        target_idx: int,
        title: str,
        save_path: str,
        colormap: str = "hot",
        alpha: float = 0.7,
        add_gridlines: bool = True,
        show_values: bool = True,
        vmax: Optional[float] = None
    ) -> Optional[str]:
        """
        Create an enhanced heatmap overlay visualization for patch image features.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            
            # Get patch feature information
            patch_feature_info = feature_mapping.get("patch_feature", {})
            if not patch_feature_info:
                print("Error: No patch feature information available.")
                return None
            
            prob_grid_h, prob_grid_w = patch_feature_info.get("grid_unpadded", (0, 0))
            if prob_grid_h == 0 or prob_grid_w == 0:
                print("Error: Invalid grid dimensions.")
                return None
            
            # Ensure the preview image is in RGB mode
            if spatial_preview_image.mode != 'RGB':
                preview_image = spatial_preview_image.convert('RGB')
                print(f"Converting image from {spatial_preview_image.mode} to RGB")
            else:
                preview_image = spatial_preview_image
                    
            # Get dimensions
            preview_w, preview_h = preview_image.size
            background_np = np.array(preview_image)
            
            # Debug: Verify image data
            print(f"Debug: Background image shape={background_np.shape}, dtype={background_np.dtype}")
            print(f"Debug: Image min/max values: {background_np.min()}/{background_np.max()}")
            
            # Get content dimensions and padding
            resized_dims_wh = feature_mapping.get("resized_dimensions", (0, 0))
            if resized_dims_wh == (0, 0):
                print("Error: Missing resized dimensions in feature mapping.")
                return None
            
            resized_w_actual, resized_h_actual = resized_dims_wh
            
            # Calculate padding
            pad_h_total = preview_h - resized_h_actual
            pad_w_total = preview_w - resized_w_actual
            pad_top = max(0, pad_h_total // 2)
            pad_left = max(0, pad_w_total // 2)
            
            # Calculate cell dimensions
            cell_height = resized_h_actual / prob_grid_h
            cell_width = resized_w_actual / prob_grid_w
            
            # Create figure with improved aspect ratio
            aspect_ratio = preview_h / max(1, preview_w)
            fig = Figure(figsize=(10, 10 * aspect_ratio), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # CRITICAL FIX: Check for all-zero or very small heatmap values
            if np.all(np.isclose(heatmap, 0, atol=1e-8)):
                print(f"Warning: Heatmap for layer {layer_idx} has all zero values. Adding small values for visibility.")
                # Add small values to make visible
                heatmap = np.ones_like(heatmap) * 0.1
            
            # CRITICAL FIX: Ensure background image is fully visible first
            # 修改: 确保背景图像完全可见且不透明
            ax.imshow(
                background_np, 
                extent=(0, preview_w, preview_h, 0),
                aspect='equal',
                origin='upper',
                alpha=1.0  # 设置为完全不透明
            )
            
            # Debug: Heatmap stats before overlay
            print(f"Debug: Heatmap shape={heatmap.shape}, min/max={heatmap.min()}/{heatmap.max()}")
            
            # Create better colormap with improved alpha handling
            cmap = plt.get_cmap(colormap)
            # 修改: 降低热力图的不透明度，确保背景可见
            base_alpha = min(0.5, alpha)  # 改为0.5，让背景更明显
            
            # Upscale heatmap to content area dimensions
            try:
                from skimage.transform import resize as skimage_resize
                upscaled_hm = skimage_resize(
                    heatmap, 
                    (resized_h_actual, resized_w_actual), 
                    order=1,  # 线性插值，平滑结果
                    mode='constant',
                    cval=0, 
                    anti_aliasing=True, 
                    preserve_range=True
                )
            except ImportError:
                # Fallback upscaling method
                print("Using fallback upscaling method (skimage not available)")
                cell_h = resized_h_actual / prob_grid_h
                cell_w = resized_w_actual / prob_grid_w
                upscaled_hm = np.zeros((int(resized_h_actual), int(resized_w_actual)))
                
                for r in range(prob_grid_h):
                    for c in range(prob_grid_w):
                        if r < heatmap.shape[0] and c < heatmap.shape[1]:
                            r_start = int(r * cell_h)
                            r_end = int((r + 1) * cell_h)
                            c_start = int(c * cell_w)
                            c_end = int((c + 1) * cell_w)
                            upscaled_hm[r_start:r_end, c_start:c_end] = heatmap[r, c]
            
            # CRITICAL FIX: Properly set the extent coordinates for overlay
            # 修改: 确保热力图和背景图像正确对齐
            extent = (
                pad_left,                    # left
                pad_left + resized_w_actual, # right
                pad_top + resized_h_actual,  # bottom (origin='upper'时, y轴从上到下)
                pad_top                      # top
            )
            
            # Debug: Print extent for verification
            print(f"Debug: Heatmap extent={extent}, background extent=(0,{preview_w},{preview_h},0)")
            
            # IMPROVED: Set explicit value range and better alpha
            im = ax.imshow(
                upscaled_hm, 
                cmap=colormap,
                alpha=base_alpha,  # 透明度降低，确保背景可见
                extent=extent,     # 使用正确计算的extent值
                origin='upper',    # 确保与背景图像使用相同的origin
                interpolation='bilinear',
                vmin=0,
                vmax=vmax if vmax is not None else max(0.01, np.max(heatmap))
            )
            
            # Add grid lines with improved visibility
            if add_gridlines:
                # Horizontal grid lines
                for i in range(prob_grid_h + 1):
                    y = pad_top + i * cell_height
                    ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5, alpha=0.7)
                
                # Vertical grid lines
                for i in range(prob_grid_w + 1):
                    x = pad_left + i * cell_width
                    ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5, alpha=0.7)
                
                # Add content area border for clarity
                content_rect = plt.Rectangle(
                    (pad_left, pad_top),
                    resized_w_actual, resized_h_actual,
                    fill=False,
                    edgecolor='cyan',
                    linestyle='--',
                    linewidth=1.5
                )
                ax.add_patch(content_rect)
            
            # IMPROVED: Better colorbar
            norm = mpl.colors.Normalize(
                vmin=0, 
                vmax=vmax if vmax is not None else max(0.01, np.max(heatmap))
            )
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Influence Weight")
            
            # Add informative title
            ax.set_title(f"{title}", fontsize=12, pad=10)
            ax.axis("off")
            
            # DEBUGGING: 生成单独的背景和热力图图像用于调试
            debug_dir = os.path.dirname(save_path)
            os.makedirs(debug_dir, exist_ok=True)
            
            # 保存背景图像
            debug_bg_path = os.path.join(debug_dir, f"debug_bg_layer{layer_idx}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(background_np, origin='upper')
            plt.title("Background Image Only")
            plt.axis('off')
            plt.savefig(debug_bg_path, dpi=150)
            plt.close()
            
            # 保存热力图
            debug_heat_path = os.path.join(debug_dir, f"debug_heat_layer{layer_idx}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(upscaled_hm, cmap=colormap, origin='upper')
            plt.title("Heatmap Only")
            plt.axis('off')
            plt.savefig(debug_heat_path, dpi=150)
            plt.close()
            
            # Save with improved quality
            fig.tight_layout()
            canvas.draw()
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            
            print(f"Success: Saved visualization to {save_path}")
            print(f"Saved debug images to {debug_bg_path} and {debug_heat_path}")
            
            return save_path
        
        except Exception as e:
            print(f"Error creating enhanced patch feature overlay: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_enhanced_grid_composite(
        self,
        heatmap_maps: Dict[int, Optional[np.ndarray]],
        layers: List[int],
        title: str,
        save_path: str,
        cmap: str = "hot",
        show_values: bool = True,
        unified_colorscale: bool = False
    ) -> Optional[str]:
        """
        Arrange each layer's 2D heatmap array into a grid with improved layout and colorscale.
        
        Args:
            heatmap_maps: Dictionary mapping layer index to 2D heatmap array
            layers: List of layer indices to include (each gets its own subplot)
            title: Title for the figure
            save_path: Path to save the composite image
            cmap: Colormap to use
            show_values: Whether to show cell values
            unified_colorscale: Whether to use same colorscale across all layers
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            import numpy as np
            import math
            
            # Only use valid layers (that have heatmaps)
            valid_layers = [L for L in layers if heatmap_maps.get(L) is not None]
            n = len(valid_layers)
            
            if n == 0:
                print("No valid layers with heatmaps. Cannot create composite.")
                return None
    
            # Better grid layout calculation
            if n <= 3:
                ncols, nrows = n, 1
            elif n <= 6:
                ncols, nrows = 3, 2
            elif n <= 9:
                ncols, nrows = 3, 3
            elif n <= 12:
                ncols, nrows = 4, 3
            else:
                # Optimize for square-ish layout
                ncols = math.ceil(math.sqrt(n))
                nrows = math.ceil(n / ncols)
            
            # Determine global max value if using unified colorscale
            if unified_colorscale:
                global_max = 0
                for L in valid_layers:
                    hm = heatmap_maps[L]
                    if hm is not None:
                        layer_max = np.max(hm)
                        global_max = max(global_max, layer_max)
                vmax = global_max
            else:
                vmax = None
            
            # Create figure with better size calculation based on layout
            subplot_size = 2.5  # Base size of each subplot
            fig_width = max(8, ncols * subplot_size + 2)  # Add space for colorbar
            fig_height = max(6, nrows * subplot_size + 1)  # Add space for title
            
            # Create figure with white background
            fig = Figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')
            canvas = FigureCanvas(fig)
            
            # Create subplot grid with improved spacing
            grid = fig.add_gridspec(nrows, ncols, wspace=0.3, hspace=0.3)
            axes = []
            
            for i in range(nrows):
                for j in range(ncols):
                    idx = i * ncols + j
                    if idx < n:
                        axes.append(fig.add_subplot(grid[i, j]))
                    else:
                        # Create empty subplot to maintain grid
                        ax = fig.add_subplot(grid[i, j])
                        ax.axis('off')
                        axes.append(ax)
            
            # Track the shared colormap instance
            cmap_inst = plt.get_cmap(cmap)
            im = None
            
            # Add each layer's heatmap to the grid with consistent colormap
            for idx, L in enumerate(valid_layers):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                hm = heatmap_maps[L]
                
                if hm is None:
                    ax.axis("off")
                    continue
                
                # Use consistent vmin/vmax when unified_colorscale is True
                im = ax.imshow(hm, vmin=0, vmax=vmax, cmap=cmap_inst)
                
                # Add layer title with statistics
                ax_title = f"Layer {L} "
                if hm.max() > 0:
                    ax_title += f"(max: {hm.max():.2f})"
                ax.set_title(ax_title, fontsize=10)
                
                # Add grid lines for better visualization
                ax.grid(False)
                
                # Add row/column labels for orientation
                if hm.shape[0] <= 20 and hm.shape[1] <= 20:
                    ax.set_xticks(np.arange(hm.shape[1]))
                    ax.set_yticks(np.arange(hm.shape[0]))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.tick_params(axis='both', which='both', length=0)
                else:
                    ax.axis("off")
    
                # Show values in cells if requested
                if show_values:
                    H, W = hm.shape
                    if H <= 10 and W <= 10:  # Only for reasonably sized grids
                        for i in range(H):
                            for j in range(W):
                                val = hm[i, j]
                                if val > 0.1:  # Only show significant values
                                    ax.text(j, i, f"{val:.2f}",
                                            ha='center', va='center', fontsize=6,
                                            color='white' if val > 0.5 else 'black')
    
            # Add shared colorbar
            if im is not None:
                # Create a dedicated axis for the colorbar
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                cbar = fig.colorbar(im, cax=cbar_ax)
                cbar.set_label(f"Normalized Influence{' (Unified Scale)' if unified_colorscale else ''}")
    
            # Add overall title
            scale_note = " (Unified Color Scale)" if unified_colorscale else ""
            fig.suptitle(f"{title}{scale_note}", fontsize=14, y=0.98)
    
            # Adjust layout and save with improved quality
            fig.tight_layout(rect=[0, 0, 0.9, 0.95])
            canvas.draw()
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            
            return save_path
            
        except Exception as e:
            print(f"Error creating enhanced grid composite: {e}")
            import traceback
            traceback.print_exc()
            return None