# analyzer/heatmap_visualizer.py
"""
Standalone heatmap visualization utility for semantic tracing results.
Allows for offline generation of heatmaps from CSV data without requiring
the model or GPU resources.
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, List

# Import only the visualization components from the main visualizer
from analyzer.semantic_tracing_visualizer import EnhancedSemanticTracingVisualizer

class HeatmapVisualizer:
    """
    Creates comprehensive heatmap visualizations from semantic tracing CSV data
    without requiring the original model or GPU resources.
    """
    
    def __init__(
        self, 
        csv_path: str, 
        metadata_path: str, 
        image_path: str, 
        out_dir: str, 
        weight_column: str = "importance_weight",
        debug_mode: bool = False
    ):
        """
        Initialize the heatmap visualizer.
        
        Args:
            csv_path: Path to the CSV file with trace data
            metadata_path: Path to JSON metadata file
            image_path: Path to original image used in the trace
            out_dir: Directory to save output visualizations
            weight_column: Column name to use for importance weights
            debug_mode: Whether to print additional debug information
        """
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.image_path = image_path
        self.out_dir = out_dir
        self.weight_column = weight_column
        self.debug_mode = debug_mode
        
        # New configuration options
        self.unified_colorscale = False
        self.include_all_token_types = False
        
        # Create output directory
        os.makedirs(out_dir, exist_ok=True)
        
        # Load CSV data with robust error handling
        print(f"Loading trace data from {csv_path}")
        try:
            # Try to load CSV with explicit numeric conversion for weight column
            df = pd.read_csv(csv_path)
            
            # Check if the specified weight column exists
            if weight_column not in df.columns:
                print(f"Warning: Weight column '{weight_column}' not found in CSV. Available columns: {list(df.columns)}")
                if "predicted_top_prob" in df.columns:
                    print(f"Using 'predicted_top_prob' as fallback weight column")
                    self.weight_column = "predicted_top_prob"
                    weight_column = "predicted_top_prob"
                else:
                    raise ValueError(f"Neither '{weight_column}' nor fallback column 'predicted_top_prob' found in CSV")
            
            # Force numeric conversion of the weight column
            df[weight_column] = pd.to_numeric(df[weight_column], errors="coerce")
            
            # Check for data issues
            if df[weight_column].isna().all():
                print(f"Warning: All values in '{weight_column}' are NaN after numeric conversion")
                # Try to get the raw values to understand the issue
                raw_df = pd.read_csv(csv_path)
                print(f"Sample raw values: {raw_df[weight_column].iloc[:5].tolist()}")
                # Handle all NaN by setting small positive values
                df[weight_column] = 0.01
            elif df[weight_column].max() <= 0:
                # If all weights are zero or negative, add a small value for visibility
                print(f"Warning: All weights are zero or negative. Adding small value for visibility.")
                df[weight_column] = df[weight_column] + 0.01
            
            self.df = df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            # Create empty DataFrame as fallback
            self.df = pd.DataFrame()
            
        # Load metadata
        print(f"Loading metadata from {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                self.meta = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.meta = {}
        
        # Add importance_weight column if using a different weight column
        if self.weight_column != "importance_weight" and not self.df.empty:
            self.df["importance_weight"] = self.df[self.weight_column]
            print(f"Mapped '{self.weight_column}' to 'importance_weight' for visualization")
        
        # Extract feature mapping
        self.feature_mapping = self.meta.get("feature_mapping", {})
        if not self.feature_mapping:
            print("Warning: No feature mapping found in metadata")
        
        # Check image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found at {image_path}")
        
        # Create visualizer instance (only for its drawing functions)
        self.viz = self._create_visualizer(output_dir=out_dir, debug_mode=debug_mode)
    
    def _create_visualizer(self, output_dir, debug_mode):
        """Create semantic tracing visualizer with custom enhancements"""
        visualizer = EnhancedSemanticTracingVisualizer(
            output_dir=output_dir,
            debug_mode=debug_mode
        )
        # Add new property for unified colorscale
        visualizer.unified_colorscale = self.unified_colorscale
        return visualizer
    
    def run(self, composite_only: bool = False, show_values: bool = True) -> List[str]:
        """
        Generate heatmap visualizations.
        
        Args:
            composite_only: If True, only create composite visualizations
            show_values: Whether to show weight values in cells
                
        Returns:
            List of paths to generated visualization files
        """
        # Verify we have target tokens
        if "target_tokens" not in self.meta or not self.meta["target_tokens"]:
            print("Error: No target tokens found in metadata")
            return []
        
        # Verify feature mapping completeness
        if not self._verify_feature_mapping():
            print("Warning: Incomplete feature mapping may affect visualization quality")
        
        # Get target token information
        target_token = self.meta["target_tokens"][0]  # Use first target for demonstration
        target_idx = target_token.get("index", 0)
        target_text = target_token.get("text", "unknown")
        
        print(f"Generating heatmaps for target token '{target_text}' (index: {target_idx})")
        
        # Set unified colorscale on visualizer
        self.viz.unified_colorscale = self.unified_colorscale
        
        # Generate standard heatmaps (for backwards compatibility)
        if not self.include_all_token_types:
            vis_results = self.viz.create_heatmaps_from_csv(
                trace_data=self.df,
                target_text=target_text,
                target_idx=target_idx,
                image_path=self.image_path,
                save_dir=self.out_dir,
                feature_mapping=self.feature_mapping,
                use_grid_visualization=True,
                show_values=show_values,
                composite_only=composite_only,
                unified_colorscale=self.unified_colorscale
            )
            return vis_results
        
        # Generate combined visualizations with all token types
        print("\nGenerating combined visualizations with all token types...")
        
        # Create combined heatmaps with all token types
        combined_heatmaps = self.create_combined_all_token_types_heatmap(
            df=self.df,
            feature_mapping=self.feature_mapping,
            original_image=Image.open(self.image_path),
            target_idx=target_idx,
            target_text=target_text,
            output_dir=self.out_dir,
            unified_colorscale=self.unified_colorscale,
            show_values=show_values
        )
        
        return combined_heatmaps
    
    def _verify_feature_mapping(self) -> bool:
        """
        Verify that feature mapping contains all required components.
        
        Returns:
            True if feature mapping is complete, False otherwise
        """
        # Check base feature
        base_valid = (
            "base_feature" in self.feature_mapping and
            "grid" in self.feature_mapping["base_feature"] and
            "positions" in self.feature_mapping["base_feature"]
        )
        
        # Check patch feature
        patch_valid = (
            "patch_feature" in self.feature_mapping and
            "grid_unpadded" in self.feature_mapping["patch_feature"] and
            "positions" in self.feature_mapping["patch_feature"]
        )
        
        # Check resized dimensions
        dims_valid = "resized_dimensions" in self.feature_mapping
        
        if not base_valid:
            print("Warning: Missing base_feature mapping (grid, positions)")
        if not patch_valid:
            print("Warning: Missing patch_feature mapping (grid_unpadded, positions)")
        if not dims_valid:
            print("Warning: Missing resized_dimensions in feature mapping")
        
        return base_valid and patch_valid and dims_valid

    def create_combined_all_token_types_heatmap(
        self,
        df: pd.DataFrame,
        feature_mapping: Dict[str, Any],
        original_image: Image.Image,
        target_idx: int,
        target_text: str,
        output_dir: str,
        unified_colorscale: bool = False,
        show_values: bool = True
    ) -> List[str]:
        """
        Create comprehensive heatmaps combining base and patch features with all token types 
        (image, text, generated) for each layer.
        
        This creates a visualization with 4 quadrants:
        - Top-left: Base features visualization with background
        - Top-right: Patch features visualization with background
        - Bottom: Token type distribution analysis
        
        Args:
            df: DataFrame containing trace data
            feature_mapping: Dictionary with feature mapping information
            original_image: Original image for background
            target_idx: Target token index
            target_text: Text of the target token
            output_dir: Directory to save output files
            unified_colorscale: Whether to use unified color scale across all token types
            show_values: Whether to show values in cells
            
        Returns:
            List of saved file paths
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        saved_paths = []
        
        # Create output directory
        combined_dir = os.path.join(output_dir, "combined_heatmaps")
        os.makedirs(combined_dir, exist_ok=True)
        
        # Get unique layers
        layers = sorted(df["layer"].unique())
        
        # Process each layer
        for layer_idx in layers:
            # Filter data for this layer
            layer_df = df[df["layer"] == layer_idx]
            
            # Skip if no data for this layer
            if layer_df.empty:
                print(f"No data for layer {layer_idx}. Skipping.")
                continue
                
            print(f"Creating combined heatmap for layer {layer_idx}")
            
            # Divide tokens by their types
            text_tokens = layer_df[layer_df["token_type"] == 1]
            image_tokens = layer_df[layer_df["token_type"] == 2]
            generated_tokens = layer_df[layer_df["token_type"] == 0]
            
            # Determine global max value for unified color scale if requested
            if unified_colorscale:
                all_importance = layer_df["importance_weight"].values
                global_max = all_importance.max() if len(all_importance) > 0 else 1.0
                print(f"Using unified color scale with global maximum: {global_max:.4f}")
            else:
                global_max = None
            
            # Create figure with a 2x2 grid layout
            fig = Figure(figsize=(20, 16), dpi=100, facecolor='white')
            canvas = FigureCanvas(fig)
            
            # Create a 2x2 grid of subplots
            grid = fig.add_gridspec(2, 2, height_ratios=[3, 1])
            
            # 1. Top-left: Base features visualization (with background)
            ax_base = fig.add_subplot(grid[0, 0])
            base_heatmap = self._create_base_features_heatmap_for_layer(
                layer_df=layer_df, 
                feature_mapping=feature_mapping,
                original_image=original_image,
                ax=ax_base,
                layer_idx=layer_idx,
                global_max=global_max if unified_colorscale else None,
                show_values=show_values
            )
            
            # 2. Top-right: Patch features visualization (with background)
            ax_patch = fig.add_subplot(grid[0, 1])
            patch_heatmap = self._create_patch_features_heatmap_for_layer(
                layer_df=layer_df, 
                feature_mapping=feature_mapping,
                original_image=original_image,
                ax=ax_patch,
                layer_idx=layer_idx,
                global_max=global_max if unified_colorscale else None,
                show_values=show_values
            )
            
            # 3. Bottom: Token importance distribution
            ax_dist = fig.add_subplot(grid[1, :])
            self._create_token_distribution_for_layer(
                layer_df=layer_df,
                ax=ax_dist,
                layer_idx=layer_idx,
                global_max=global_max if unified_colorscale else None
            )
            
            # Set overall title
            scale_note = " (Unified Scale)" if unified_colorscale else ""
            fig.suptitle(
                f"Layer {layer_idx} - Combined Base and Patch Features with All Token Types{scale_note}\n"
                f"Target: '{target_text}' (idx: {target_idx})", 
                fontsize=16
            )
            
            # Save the figure
            out_path = os.path.join(combined_dir, f"combined_all_tokens_layer{layer_idx}.png")
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
            canvas.draw()
            fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            
            saved_paths.append(out_path)
            print(f"Created combined heatmap for layer {layer_idx} at {out_path}")
        
        # Create a composite image with all layers in a grid
        if saved_paths:
            composite_path = self._create_all_layers_composite(
                saved_paths=saved_paths,
                layers=layers,
                output_dir=combined_dir,
                target_idx=target_idx,
                target_text=target_text,
                unified_colorscale=unified_colorscale
            )
            if composite_path:
                saved_paths.append(composite_path)
        
        return saved_paths

    def _create_base_features_heatmap_for_layer(
        self,
        layer_df: pd.DataFrame,
        feature_mapping: Dict[str, Any],
        original_image: Image.Image,
        ax,
        layer_idx: int,
        global_max: Optional[float] = None,
        show_values: bool = True
    ) -> bool:
        """
        Create base features heatmap for a specific layer with all token types.
        
        Args:
            layer_df: DataFrame with layer-specific trace data
            feature_mapping: Dictionary with feature mapping information
            original_image: Original image for background
            ax: Matplotlib axis for plotting
            layer_idx: Layer index
            global_max: Maximum value for colormap normalization (for unified scale)
            show_values: Whether to show values in cells
            
        Returns:
            True if successful, False otherwise
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get base feature mapping
        bf = feature_mapping.get("base_feature", {})
        if not bf:
            ax.text(0.5, 0.5, "No base feature mapping available", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return False
        
        # Get grid dimensions
        grid_h, grid_w = bf.get("grid", (0, 0))
        if grid_h == 0 or grid_w == 0:
            ax.text(0.5, 0.5, "Invalid grid dimensions", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return False
        
        # Use consistent size for base images
        target_size = (336, 336)
        
        # Resize image for background
        if original_image.mode != 'RGB':
            resized_background = original_image.convert('RGB').resize(target_size, Image.LANCZOS)
        else:
            resized_background = original_image.resize(target_size, Image.LANCZOS)
        
        background_np = np.array(resized_background)
        
        # Apply slight darkening for better contrast
        darkened_bg = background_np.astype(float) * 0.45
        darkened_bg = np.clip(darkened_bg, 0, 255).astype(np.uint8)
        
        # Show background
        ax.imshow(
            darkened_bg,
            extent=(0, target_size[0], target_size[1], 0),
            aspect='auto',
            origin='upper'
        )
        
        # Calculate cell dimensions
        cell_height = target_size[1] / grid_h
        cell_width = target_size[0] / grid_w
        
        # Get token data divided by types
        text_tokens = layer_df[layer_df["token_type"] == 1]
        image_tokens = layer_df[layer_df["token_type"] == 2]
        generated_tokens = layer_df[layer_df["token_type"] == 0]
        
        # Get token positions mapped to the grid
        token_to_position = {}
        token_to_type = {}
        token_to_weight = {}
        
        # For each token type, extract positions and weights
        for token_type, tokens_df in [
            (2, image_tokens),    # Image tokens
            (1, text_tokens),     # Text tokens
            (0, generated_tokens) # Generated tokens
        ]:
            for _, row in tokens_df.iterrows():
                token_idx = row["token_index"]
                weight = row["importance_weight"]
                
                # Skip tokens with zero importance
                if weight <= 0:
                    continue
                    
                # Map token index to position in grid
                pos = None
                if str(token_idx) in bf["positions"]:
                    pos = bf["positions"][str(token_idx)]
                elif token_idx in bf["positions"]:
                    pos = bf["positions"][token_idx]
                    
                if pos:
                    r, c = pos
                    if 0 <= r < grid_h and 0 <= c < grid_w:
                        # Store position, type and weight
                        token_to_position[token_idx] = (r, c)
                        token_to_type[token_idx] = token_type
                        
                        # If same token appears multiple times, keep the maximum weight
                        if token_idx in token_to_weight:
                            token_to_weight[token_idx] = max(token_to_weight[token_idx], weight)
                        else:
                            token_to_weight[token_idx] = weight
        
        # If using unified scale, normalize weights
        if global_max and global_max > 0:
            for token_idx in token_to_weight:
                token_to_weight[token_idx] = token_to_weight[token_idx] / global_max
        
        # Set colormaps for different token types
        cmap_image = plt.get_cmap('hot')      # Red-yellow for image tokens
        cmap_text = plt.get_cmap('Blues')     # Blue for text tokens
        cmap_generated = plt.get_cmap('Greens')  # Green for generated tokens
        
        # Draw tokens on the grid
        for token_idx, pos in token_to_position.items():
            r, c = pos
            token_type = token_to_type[token_idx]
            weight = token_to_weight[token_idx]
            
            # Skip tokens with very low weight
            if weight < 0.01:
                continue
                
            # Calculate cell position
            x_start = c * cell_width
            y_start = r * cell_height
            
            # Apply gamma correction for better contrast
            contrast_val = weight ** 0.5
            
            # Choose colormap based on token type
            if token_type == 2:  # Image token
                cmap = cmap_image
                edgecolor = 'white'
            elif token_type == 1:  # Text token
                cmap = cmap_text
                edgecolor = 'lightblue'
            else:  # Generated token
                cmap = cmap_generated
                edgecolor = 'lightgreen'
            
            # Create colored rectangle
            rect = plt.Rectangle(
                (x_start, y_start),
                cell_width, cell_height,
                facecolor=cmap(contrast_val),
                edgecolor=edgecolor,
                linewidth=0.5 if weight > 0.5 else 0,
                alpha=min(0.8, contrast_val + 0.3)
            )
            ax.add_patch(rect)
            
            # Show weight value if requested
            if show_values and weight > 0.2:
                cell_center_x = x_start + cell_width / 2
                cell_center_y = y_start + cell_height / 2
                ax.text(
                    cell_center_x, cell_center_y,
                    f"{weight:.2f}",
                    ha='center', va='center',
                    fontsize=7,
                    color='white' if weight > 0.4 else 'black',
                    weight='bold'
                )
        
        # Add grid lines
        for i in range(grid_h + 1):
            y = i * cell_height
            ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        for i in range(grid_w + 1):
            x = i * cell_width
            ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add legend
        import matplotlib.patches as mpatches
        legend_handles = []
        
        # Count tokens of each type for legend
        img_count = len(image_tokens)
        text_count = len(text_tokens)
        gen_count = len(generated_tokens)
        
        # Add legend entries only for token types that are present
        if img_count > 0:
            max_img_weight = image_tokens["importance_weight"].max() if not image_tokens.empty else 0
            if global_max and global_max > 0:
                norm_max_img = max_img_weight / global_max
                img_label = f"Image Tokens ({img_count}) - Max: {max_img_weight:.3f}, Norm: {norm_max_img:.3f}"
            else:
                img_label = f"Image Tokens ({img_count}) - Max: {max_img_weight:.3f}"
            legend_handles.append(mpatches.Patch(color=cmap_image(0.7), label=img_label))
            
        if text_count > 0:
            max_text_weight = text_tokens["importance_weight"].max() if not text_tokens.empty else 0
            if global_max and global_max > 0:
                norm_max_text = max_text_weight / global_max
                text_label = f"Text Tokens ({text_count}) - Max: {max_text_weight:.3f}, Norm: {norm_max_text:.3f}"
            else:
                text_label = f"Text Tokens ({text_count}) - Max: {max_text_weight:.3f}"
            legend_handles.append(mpatches.Patch(color=cmap_text(0.7), label=text_label))
            
        if gen_count > 0:
            max_gen_weight = generated_tokens["importance_weight"].max() if not generated_tokens.empty else 0
            if global_max and global_max > 0:
                norm_max_gen = max_gen_weight / global_max
                gen_label = f"Generated Tokens ({gen_count}) - Max: {max_gen_weight:.3f}, Norm: {norm_max_gen:.3f}"
            else:
                gen_label = f"Generated Tokens ({gen_count}) - Max: {max_gen_weight:.3f}"
            legend_handles.append(mpatches.Patch(color=cmap_generated(0.7), label=gen_label))
        
        # Add legend to plot
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=min(3, len(legend_handles))
            )
        
        # Set title
        ax.set_title(f"Base Features - Layer {layer_idx}", fontsize=12)
        ax.axis('off')
        
        return True

    def _create_patch_features_heatmap_for_layer(
        self,
        layer_df: pd.DataFrame,
        feature_mapping: Dict[str, Any],
        original_image: Image.Image,
        ax,
        layer_idx: int,
        global_max: Optional[float] = None,
        show_values: bool = True
    ) -> bool:
        """
        Create patch features heatmap for a specific layer with all token types.
        
        Args:
            layer_df: DataFrame with layer-specific trace data
            feature_mapping: Dictionary with feature mapping information
            original_image: Original image for background
            ax: Matplotlib axis for plotting
            layer_idx: Layer index
            global_max: Maximum value for colormap normalization (for unified scale)
            show_values: Whether to show values in cells
            
        Returns:
            True if successful, False otherwise
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Get patch feature mapping
        pf = feature_mapping.get("patch_feature", {})
        if not pf:
            ax.text(0.5, 0.5, "No patch feature mapping available", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return False
        
        # Get grid dimensions
        grid_h, grid_w = pf.get("grid_unpadded", (0, 0))
        if grid_h == 0 or grid_w == 0:
            ax.text(0.5, 0.5, "Invalid grid dimensions", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return False
        
        # Create preview image for background
        preview = original_image.copy()
        if "resized_dimensions" in feature_mapping:
            w, h = feature_mapping["resized_dimensions"]
            preview = preview.resize((w, h), Image.LANCZOS)
        
        preview_w, preview_h = preview.size
        background_np = np.array(preview)
        
        # Apply slight darkening for better contrast
        darkened_bg = background_np.astype(float) * 0.45
        darkened_bg = np.clip(darkened_bg, 0, 255).astype(np.uint8)
        
        # Show background
        ax.imshow(
            darkened_bg,
            extent=(0, preview_w, preview_h, 0),
            aspect='equal',
            origin='upper'
        )
        
        # Calculate padding
        resized_dims_wh = feature_mapping.get("resized_dimensions", (0, 0))
        if resized_dims_wh == (0, 0):
            resized_w_actual, resized_h_actual = preview_w, preview_h
        else:
            resized_w_actual, resized_h_actual = resized_dims_wh
        
        pad_h_total = preview_h - resized_h_actual
        pad_w_total = preview_w - resized_w_actual
        pad_top = max(0, pad_h_total // 2)
        pad_left = max(0, pad_w_total // 2)
        
        # Calculate cell dimensions
        cell_height = resized_h_actual / grid_h
        cell_width = resized_w_actual / grid_w
        
        # Get token data divided by types
        text_tokens = layer_df[layer_df["token_type"] == 1]
        image_tokens = layer_df[layer_df["token_type"] == 2]
        generated_tokens = layer_df[layer_df["token_type"] == 0]
        
        # Get token positions mapped to the grid
        token_to_position = {}
        token_to_type = {}
        token_to_weight = {}
        
        # For each token type, extract positions and weights
        for token_type, tokens_df in [
            (2, image_tokens),    # Image tokens
            (1, text_tokens),     # Text tokens
            (0, generated_tokens) # Generated tokens
        ]:
            for _, row in tokens_df.iterrows():
                token_idx = row["token_index"]
                weight = row["importance_weight"]
                
                # Skip tokens with zero importance
                if weight <= 0:
                    continue
                    
                # Map token index to position in grid
                pos = None
                if str(token_idx) in pf["positions"]:
                    pos = pf["positions"][str(token_idx)]
                elif token_idx in pf["positions"]:
                    pos = pf["positions"][token_idx]
                    
                if pos:
                    r, c = pos
                    if 0 <= r < grid_h and 0 <= c < grid_w:
                        # Store position, type and weight
                        token_to_position[token_idx] = (r, c)
                        token_to_type[token_idx] = token_type
                        
                        # If same token appears multiple times, keep the maximum weight
                        if token_idx in token_to_weight:
                            token_to_weight[token_idx] = max(token_to_weight[token_idx], weight)
                        else:
                            token_to_weight[token_idx] = weight
        
        # If using unified scale, normalize weights
        if global_max and global_max > 0:
            for token_idx in token_to_weight:
                token_to_weight[token_idx] = token_to_weight[token_idx] / global_max
        
        # Set colormaps for different token types
        cmap_image = plt.get_cmap('hot')      # Red-yellow for image tokens
        cmap_text = plt.get_cmap('Blues')     # Blue for text tokens
        cmap_generated = plt.get_cmap('Greens')  # Green for generated tokens
        
        # Draw tokens on the grid
        for token_idx, pos in token_to_position.items():
            r, c = pos
            token_type = token_to_type[token_idx]
            weight = token_to_weight[token_idx]
            
            # Skip tokens with very low weight
            if weight < 0.01:
                continue
                
            # Calculate cell position
            x_start = pad_left + c * cell_width
            y_start = pad_top + r * cell_height
            
            # Apply gamma correction for better contrast
            contrast_val = weight ** 0.5
            
            # Choose colormap based on token type
            if token_type == 2:  # Image token
                cmap = cmap_image
                edgecolor = 'white'
            elif token_type == 1:  # Text token
                cmap = cmap_text
                edgecolor = 'lightblue'
            else:  # Generated token
                cmap = cmap_generated
                edgecolor = 'lightgreen'
            
            # Create colored rectangle
            rect = plt.Rectangle(
                (x_start, y_start),
                cell_width, cell_height,
                facecolor=cmap(contrast_val),
                edgecolor=edgecolor,
                linewidth=0.5 if weight > 0.5 else 0,
                alpha=min(0.8, contrast_val + 0.3)
            )
            ax.add_patch(rect)
            
            # Show weight value if requested
            if show_values and weight > 0.2:
                cell_center_x = x_start + cell_width / 2
                cell_center_y = y_start + cell_height / 2
                ax.text(
                    cell_center_x, cell_center_y,
                    f"{weight:.2f}",
                    ha='center', va='center',
                    fontsize=7,
                    color='white' if weight > 0.4 else 'black',
                    weight='bold'
                )
        
        # Add grid lines
        for i in range(grid_h + 1):
            y = pad_top + i * cell_height
            ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        for i in range(grid_w + 1):
            x = pad_left + i * cell_width
            ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add content area border
        content_rect = plt.Rectangle(
            (pad_left, pad_top),
            resized_w_actual, resized_h_actual,
            fill=False,
            edgecolor='cyan',
            linestyle='--',
            linewidth=1.5
        )
        ax.add_patch(content_rect)
        
        # Add legend
        import matplotlib.patches as mpatches
        legend_handles = []
        
        # Count tokens of each type for legend
        img_count = len(image_tokens)
        text_count = len(text_tokens)
        gen_count = len(generated_tokens)
        
        # Add legend entries only for token types that are present
        if img_count > 0:
            max_img_weight = image_tokens["importance_weight"].max() if not image_tokens.empty else 0
            if global_max and global_max > 0:
                norm_max_img = max_img_weight / global_max
                img_label = f"Image Tokens ({img_count}) - Max: {max_img_weight:.3f}, Norm: {norm_max_img:.3f}"
            else:
                img_label = f"Image Tokens ({img_count}) - Max: {max_img_weight:.3f}"
            legend_handles.append(mpatches.Patch(color=cmap_image(0.7), label=img_label))
            
        if text_count > 0:
            max_text_weight = text_tokens["importance_weight"].max() if not text_tokens.empty else 0
            if global_max and global_max > 0:
                norm_max_text = max_text_weight / global_max
                text_label = f"Text Tokens ({text_count}) - Max: {max_text_weight:.3f}, Norm: {norm_max_text:.3f}"
            else:
                text_label = f"Text Tokens ({text_count}) - Max: {max_text_weight:.3f}"
            legend_handles.append(mpatches.Patch(color=cmap_text(0.7), label=text_label))
            
        if gen_count > 0:
            max_gen_weight = generated_tokens["importance_weight"].max() if not generated_tokens.empty else 0
            if global_max and global_max > 0:
                norm_max_gen = max_gen_weight / global_max
                gen_label = f"Generated Tokens ({gen_count}) - Max: {max_gen_weight:.3f}, Norm: {norm_max_gen:.3f}"
            else:
                gen_label = f"Generated Tokens ({gen_count}) - Max: {max_gen_weight:.3f}"
            legend_handles.append(mpatches.Patch(color=cmap_generated(0.7), label=gen_label))
        
        # Add legend to plot
        if legend_handles:
            ax.legend(
                handles=legend_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True,
                ncol=min(3, len(legend_handles))
            )
        
        # Set title
        ax.set_title(f"Patch Features - Layer {layer_idx}", fontsize=12)
        ax.axis('off')
        
        return True

    def _create_token_distribution_for_layer(
        self,
        layer_df: pd.DataFrame,
        ax,
        layer_idx: int,
        global_max: Optional[float] = None
    ) -> bool:
        """
        Create token distribution visualization for a specific layer.
        
        Args:
            layer_df: DataFrame with layer-specific trace data
            ax: Matplotlib axis for plotting
            layer_idx: Layer index
            global_max: Maximum value for colormap normalization (for unified scale)
            
        Returns:
            True if successful, False otherwise
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        import pandas as pd
        
        # Skip if no data
        if layer_df.empty:
            ax.text(0.5, 0.5, "No token data available", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return False
        
        # Get token data divided by types
        text_tokens = layer_df[layer_df["token_type"] == 1]
        image_tokens = layer_df[layer_df["token_type"] == 2]
        generated_tokens = layer_df[layer_df["token_type"] == 0]
        
        # Set colors for different token types
        colors = {0: 'green', 1: 'blue', 2: 'red'}
        token_type_labels = {0: 'Generated', 1: 'Text', 2: 'Image'}
        
        # Aggregate by token index to get maximum weight per token
        tokens_with_max_weight = []
        
        for token_type, tokens_df in [
            (2, image_tokens),    # Image tokens
            (1, text_tokens),     # Text tokens
            (0, generated_tokens) # Generated tokens
        ]:
            if tokens_df.empty:
                continue
                
            for token_idx, group in tokens_df.groupby("token_index"):
                max_weight = group["importance_weight"].max()
                if global_max and global_max > 0:
                    normalized_weight = max_weight / global_max
                else:
                    normalized_weight = max_weight
                    
                tokens_with_max_weight.append({
                    "token_index": token_idx,
                    "token_text": group["token_text"].iloc[0],
                    "token_type": token_type,
                    "weight": max_weight,
                    "normalized_weight": normalized_weight
                })
        
        # Convert to DataFrame for easier plotting
        tokens_df = pd.DataFrame(tokens_with_max_weight)
        
        if tokens_df.empty:
            ax.text(0.5, 0.5, "No tokens with weights available", ha='center', va='center', fontsize=12)
            ax.axis('off')
            return False
        
        # Sort by token index for consistent ordering
        tokens_df = tokens_df.sort_values("token_index")
        
        # Create bar chart of token weights
        bar_width = 0.8
        x = np.arange(len(tokens_df))
        
        # Plot bars with token type colors
        for token_type in sorted(tokens_df["token_type"].unique()):
            type_df = tokens_df[tokens_df["token_type"] == token_type]
            if type_df.empty:
                continue
                
            # Get indices in the overall array
            type_indices = [list(tokens_df["token_index"]).index(idx) for idx in type_df["token_index"]]
            
            # Plot bars
            ax.bar(
                x[type_indices],
                type_df["normalized_weight"] if "normalized_weight" in type_df.columns else type_df["weight"],
                width=bar_width,
                color=colors[token_type],
                label=f"{token_type_labels[token_type]} Tokens ({len(type_df)})"
            )
        
        # Add token texts as x-tick labels
        if len(tokens_df) <= 30:  # Only add labels if not too many tokens
            ax.set_xticks(x)
            ax.set_xticklabels(
                tokens_df["token_text"],
                rotation=45,
                ha="right",
                fontsize=8
            )
        else:
            # Just mark a few positions
            step = max(1, len(tokens_df) // 10)
            ax.set_xticks(x[::step])
            ax.set_xticklabels(
                tokens_df["token_text"].iloc[::step],
                rotation=45,
                ha="right",
                fontsize=8
            )
        
        # Add token indices above the bars
        for i, (_, row) in enumerate(tokens_df.iterrows()):
            ax.text(
                i,
                row["normalized_weight"] if "normalized_weight" in row else row["weight"],
                f"{row['token_index']}",
                ha='center',
                va='bottom',
                fontsize=7,
                color='black'
            )
        
        # Set labels and title
        weight_label = "Normalized Weight" if "normalized_weight" in tokens_df.columns else "Weight"
        ax.set_xlabel("Tokens", fontsize=10)
        ax.set_ylabel(weight_label, fontsize=10)
        ax.set_title(f"Token Distribution - Layer {layer_idx}", fontsize=12)
        
        # Add grid lines for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend
        ax.legend(fontsize=9)
        
        return True

    def _create_all_layers_composite(
        self,
        saved_paths: List[str],
        layers: List[int],
        output_dir: str,
        target_idx: int,
        target_text: str,
        unified_colorscale: bool = False
    ) -> Optional[str]:
        """
        Create a composite visualization of all layers.
        
        Args:
            saved_paths: List of paths to individual layer visualizations
            layers: List of layer indices
            output_dir: Directory to save output
            target_idx: Target token index
            target_text: Text of the target token
            unified_colorscale: Whether unified colorscale was used
            
        Returns:
            Path to saved composite visualization or None if failed
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np
        import math
        import os
        
        try:
            # Determine grid layout
            n = len(saved_paths)
            if n == 0:
                return None
                
            # Use a grid layout that's as square as possible
            ncols = int(math.ceil(math.sqrt(n)))
            nrows = int(math.ceil(n / ncols))
            
            # Create figure
            fig_width = ncols * 6
            fig_height = nrows * 6
            fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
            
            # Make axes a 2D array even if nrows or ncols is 1
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = axes.reshape(1, -1)
            elif ncols == 1:
                axes = axes.reshape(-1, 1)
            
            # Add each layer image to the grid
            for i, path in enumerate(saved_paths):
                row = i // ncols
                col = i % ncols
                
                # Load image
                img = mpimg.imread(path)
                
                # Display in the grid
                axes[row, col].imshow(img)
                axes[row, col].axis('off')
                
                # Add layer number
                if i < len(layers):
                    axes[row, col].set_title(f"Layer {layers[i]}", fontsize=12)
            
            # Turn off empty subplots
            for i in range(len(saved_paths), nrows * ncols):
                row = i // ncols
                col = i % ncols
                axes[row, col].axis('off')
            
            # Add overall title
            scale_note = " (Unified Scale)" if unified_colorscale else ""
            fig.suptitle(
                f"All Layers - Combined Base and Patch Features with All Token Types{scale_note}\n"
                f"Target: '{target_text}' (idx: {target_idx})",
                fontsize=20
            )
            
            # Save figure
            output_path = os.path.join(output_dir, f"all_layers_composite_{target_idx}.png")
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the title
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            print(f"Created all layers composite at {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating all layers composite: {e}")
            return None