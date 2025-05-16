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
        Create comprehensive heatmaps showing all token types across separate visualizations.
        
        For each layer, creates a visualization with 5 distinct components:
        1. Text tokens heatmap - Shows all text tokens in the sequence
        2. Image base tokens heatmap - Shows base image features
        3. Image patch tokens heatmap - Shows patch image features  
        4. Generated tokens heatmap - Shows all generated tokens
        5. Token distribution graph - Shows relative importance weights
        
        All token types are displayed in their own grid with consistent formatting, ensuring
        that even tokens with zero weights are represented in the spatial layout.
        
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
                
            print(f"Creating comprehensive combined heatmap for layer {layer_idx}")
            
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
                # Use separate max values for each token type
                img_max = image_tokens["importance_weight"].max() if not image_tokens.empty else 0.0
                text_max = text_tokens["importance_weight"].max() if not text_tokens.empty else 0.0
                gen_max = generated_tokens["importance_weight"].max() if not generated_tokens.empty else 0.0
                global_max = None
                print(f"Using separate scales - Image max: {img_max:.4f}, Text max: {text_max:.4f}, Generated max: {gen_max:.4f}")
            
            # Create figure with a grid layout for the 5 components
            # Taller figure to accommodate all 5 panels
            fig = Figure(figsize=(20, 30), dpi=100, facecolor='white')
            canvas = FigureCanvas(fig)
            
            # Create a 5-row grid
            grid = fig.add_gridspec(5, 1, height_ratios=[2, 2, 2, 2, 1])
            
            # 1. Text tokens heatmap
            ax_text = fig.add_subplot(grid[0])
            self._create_specific_token_heatmap(
                layer_df=layer_df,
                feature_mapping=feature_mapping,
                original_image=original_image,
                ax=ax_text,
                layer_idx=layer_idx,
                global_max=global_max,
                show_values=show_values,
                token_type=1,  # Text tokens
                title="Text Tokens"
            )
            
            # 2. Image base tokens heatmap
            ax_base = fig.add_subplot(grid[1])
            self._create_specific_token_heatmap(
                layer_df=layer_df,
                feature_mapping=feature_mapping,
                original_image=original_image,
                ax=ax_base,
                layer_idx=layer_idx,
                global_max=global_max,
                show_values=show_values,
                token_type=2,  # Image tokens
                is_patch=False,  # Base features
                title="Image Base Tokens"
            )
            
            # 3. Image patch tokens heatmap
            ax_patch = fig.add_subplot(grid[2])
            self._create_specific_token_heatmap(
                layer_df=layer_df,
                feature_mapping=feature_mapping,
                original_image=original_image,
                ax=ax_patch,
                layer_idx=layer_idx,
                global_max=global_max,
                show_values=show_values,
                token_type=2,  # Image tokens
                is_patch=True,  # Patch features
                title="Image Patch Tokens"
            )
            
            # 4. Generated tokens heatmap
            ax_generated = fig.add_subplot(grid[3])
            self._create_specific_token_heatmap(
                layer_df=layer_df,
                feature_mapping=feature_mapping,
                original_image=original_image,
                ax=ax_generated,
                layer_idx=layer_idx,
                global_max=global_max,
                show_values=show_values,
                token_type=0,  # Generated tokens
                title="Generated Tokens"
            )
            
            # 5. Token distribution analysis
            ax_dist = fig.add_subplot(grid[4])
            self._create_token_distribution_for_layer(
                layer_df=layer_df,
                ax=ax_dist,
                layer_idx=layer_idx,
                global_max=global_max
            )
            
            # Set overall title
            scale_note = " (Unified Scale)" if unified_colorscale else " (Independent Scales)"
            fig.suptitle(
                f"Layer {layer_idx} - Complete Token Influence Map{scale_note}\n"
                f"Target: '{target_text}' (idx: {target_idx})", 
                fontsize=20
            )
            
            # Save the figure
            out_path = os.path.join(combined_dir, f"combined_all_tokens_layer{layer_idx}.png")
            fig.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust for the title
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

    def _create_specific_token_heatmap(
        self,
        layer_df: pd.DataFrame,
        feature_mapping: Dict[str, Any],
        original_image: Image.Image,
        ax,
        layer_idx: int,
        global_max: Optional[float] = None,
        show_values: bool = True,
        token_type: int = 2,  # Default: image tokens
        is_patch: bool = None,  # None=auto, True=patch, False=base
        title: Optional[str] = None
    ) -> bool:
        """
        Create a heatmap for a specific token type, ensuring complete grid visualization.
        
        This function creates a grid visualization where each token is shown in its proper position,
        regardless of whether it has a significant weight. This ensures all tokens of the specified
        type are visualized in their spatial layout.
        
        Different token types are handled differently:
        - Image tokens (type=2): These can be visualized as base or patch features with image background
        - Text tokens (type=1): These are shown as a sequential grid
        - Generated tokens (type=0): These are shown as a sequential grid
        
        Args:
            layer_df: DataFrame with layer-specific trace data
            feature_mapping: Dictionary with feature mapping information
            original_image: Original image for background
            ax: Matplotlib axis for plotting
            layer_idx: Layer index
            global_max: Maximum value for colormap normalization (for unified scale)
            show_values: Whether to show values in cells
            token_type: Token type to display (0=generated, 1=text, 2=image)
            is_patch: For image tokens, whether to use patch features (True) or base features (False)
            title: Optional title override
            
        Returns:
            True if successful, False otherwise
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Filter by token type
        type_df = layer_df[layer_df["token_type"] == token_type]
        
        # Set color map based on token type
        if token_type == 0:  # Generated
            cmap = plt.get_cmap('Greens') 
            token_type_name = "Generated"
        elif token_type == 1:  # Text
            cmap = plt.get_cmap('Blues')
            token_type_name = "Text"
        else:  # Image
            cmap = plt.get_cmap('hot')
            token_type_name = "Image"
        
        # Check if we have data
        if type_df.empty:
            ax.text(0.5, 0.5, f"No {token_type_name} tokens in layer {layer_idx}", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return False
        
        # For text and generated tokens, create a sequential representation
        if token_type in [0, 1]:
            return self._create_sequential_token_heatmap(
                type_df=type_df,
                ax=ax,
                layer_idx=layer_idx,
                global_max=global_max,
                show_values=show_values,
                token_type=token_type,
                cmap=cmap,
                title=title
            )
        
        # For image tokens, determine if we should use base or patch visualization
        if token_type == 2:
            # Auto-detect if not specified
            if is_patch is None:
                # Check if patch feature info exists
                if "patch_feature" in feature_mapping and "positions" in feature_mapping["patch_feature"]:
                    is_patch = True
                else:
                    is_patch = False
            
            # Use appropriate feature mapping based on patch/base selection
            feature_type = "patch_feature" if is_patch else "base_feature"
            feature_info = feature_mapping.get(feature_type, {})
            
            # Check if feature mapping exists
            if not feature_info or "positions" not in feature_info:
                ax.text(0.5, 0.5, f"No {feature_type} mapping available", 
                        ha='center', va='center', fontsize=14)
                ax.axis('off')
                return False
            
            # Get grid dimensions and positions
            if is_patch:
                grid_h, grid_w = feature_info.get("grid_unpadded", (0, 0))
                grid_name = "grid_unpadded"
            else:
                grid_h, grid_w = feature_info.get("grid", (0, 0))
                grid_name = "grid"
            
            positions = feature_info.get("positions", {})
            
            if grid_h == 0 or grid_w == 0 or not positions:
                ax.text(0.5, 0.5, f"Invalid {feature_type} dimensions or positions", 
                        ha='center', va='center', fontsize=14)
                ax.axis('off')
                return False
            
            # Initialize background image based on feature type
            if is_patch:
                # For patch features
                preview = original_image.copy()
                if "resized_dimensions" in feature_mapping:
                    w, h = feature_mapping["resized_dimensions"]
                    preview = preview.resize((w, h), Image.LANCZOS)
                
                preview_w, preview_h = preview.size
                background_np = np.array(preview)
                
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
                
                # Display background
                darkened_bg = background_np.astype(float) * 0.45
                darkened_bg = np.clip(darkened_bg, 0, 255).astype(np.uint8)
                
                ax.imshow(
                    darkened_bg,
                    extent=(0, preview_w, preview_h, 0),
                    aspect='equal',
                    origin='upper'
                )
            else:
                # For base features
                target_size = (336, 336)
                
                # Resize image for background
                if original_image.mode != 'RGB':
                    resized_background = original_image.convert('RGB').resize(target_size, Image.LANCZOS)
                else:
                    resized_background = original_image.resize(target_size, Image.LANCZOS)
                
                background_np = np.array(resized_background)
                darkened_bg = background_np.astype(float) * 0.45
                darkened_bg = np.clip(darkened_bg, 0, 255).astype(np.uint8)
                
                ax.imshow(
                    darkened_bg,
                    extent=(0, target_size[0], target_size[1], 0),
                    aspect='auto',
                    origin='upper'
                )
                
                # No padding for base features
                pad_top, pad_left = 0, 0
                
                # Calculate cell dimensions
                cell_height = target_size[1] / grid_h
                cell_width = target_size[0] / grid_w
            
            # Initialize token position and weight maps
            token_positions = {}  # Maps token indices to grid positions
            token_weights = {}    # Maps token indices to weights
            
            # CRITICAL: Create full grid with all possible token positions from mapping
            # This ensures we show ALL possible image token locations, not just ones with data
            for token_idx_str, pos in positions.items():
                try:
                    # Convert string key to integer if needed
                    token_idx = int(token_idx_str) if isinstance(token_idx_str, str) else token_idx_str
                    r, c = pos
                    
                    # Only add if position is within grid bounds
                    if 0 <= r < grid_h and 0 <= c < grid_w:
                        token_positions[token_idx] = (r, c)
                        token_weights[token_idx] = 0.0  # Default weight is zero
                except (ValueError, TypeError):
                    continue
            
            # Now update with actual data from DataFrame
            for _, row in type_df.iterrows():
                token_idx = row["token_index"]
                weight = row["importance_weight"]
                
                # Check if this token has a position mapping
                if token_idx in token_positions:
                    # If token appears multiple times, use max weight
                    if token_idx in token_weights:
                        token_weights[token_idx] = max(token_weights[token_idx], weight)
                    else:
                        token_weights[token_idx] = weight
            
            # Get stats
            active_tokens = sum(1 for w in token_weights.values() if w > 0.01)
            max_weight = max(token_weights.values()) if token_weights else 0.0
            
            # If using unified scale, normalize weights
            if global_max is not None and global_max > 0:
                for token_idx in token_weights:
                    token_weights[token_idx] = token_weights[token_idx] / global_max
            
            # Draw token positions regardless of weight, ensuring complete grid visualization
            for token_idx, (r, c) in token_positions.items():
                # Get token weight
                weight = token_weights.get(token_idx, 0.0)
                
                # Calculate cell position based on feature type
                if is_patch:
                    x_start = pad_left + c * cell_width
                    y_start = pad_top + r * cell_height
                else:
                    x_start = c * cell_width
                    y_start = r * cell_height
                
                # Apply gamma correction for better contrast
                contrast_val = weight ** 0.5
                
                # Determine alpha and edge properties based on weight
                if weight > 0.01:
                    # Significant weight
                    alpha = min(0.8, contrast_val + 0.3)
                    edgecolor = 'white'
                    linewidth = 0.5 if weight > 0.5 else 0
                else:
                    # Very low or zero weight - show as faint grid outline
                    alpha = 0.15
                    edgecolor = 'gray'
                    linewidth = 0.5
                
                # Create colored rectangle for this token
                rect = plt.Rectangle(
                    (x_start, y_start),
                    cell_width, cell_height,
                    facecolor=cmap(max(0.1, contrast_val)),  # Use at least 0.1 for visibility
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=alpha
                )
                ax.add_patch(rect)
                
                # Show weight value if requested and weight is significant
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
            
            # Add grid lines to show the full grid structure
            for i in range(grid_h + 1):
                y = pad_top + i * cell_height if is_patch else i * cell_height
                ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            
            for i in range(grid_w + 1):
                x = pad_left + i * cell_width if is_patch else i * cell_width
                ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # For patch features, add content area border
            if is_patch:
                content_rect = plt.Rectangle(
                    (pad_left, pad_top),
                    resized_w_actual, resized_h_actual,
                    fill=False,
                    edgecolor='cyan',
                    linestyle='--',
                    linewidth=1.5
                )
                ax.add_patch(content_rect)
            
            # Add legend with stats
            import matplotlib.patches as mpatches
            
            if global_max is not None:
                # Using unified scale, need to show original max
                orig_max = type_df["importance_weight"].max() if not type_df.empty else 0.0
                norm_max = orig_max / global_max if global_max > 0 else 0.0
                legend_text = f"Image Tokens ({active_tokens} active of {len(token_positions)}) - " \
                            f"Max: {orig_max:.3f}, Norm: {norm_max:.3f}"
            else:
                # Using type-specific scale
                legend_text = f"Image Tokens ({active_tokens} active of {len(token_positions)}) - " \
                            f"Max: {max_weight:.3f}"
            
            legend_handles = [mpatches.Patch(color=cmap(0.7), label=legend_text)]
            
            # Add legend to plot
            ax.legend(
                handles=legend_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.05),
                fancybox=True,
                shadow=True
            )
            
            # Set title
            if title:
                ax.set_title(title, fontsize=14)
            else:
                feature_name = "Patch" if is_patch else "Base"
                ax.set_title(f"Image {feature_name} Tokens - Layer {layer_idx}", fontsize=14)
            
            ax.axis('off')
            return True

    def _create_sequential_token_heatmap(
        self,
        type_df: pd.DataFrame,
        ax,
        layer_idx: int,
        global_max: Optional[float] = None,
        show_values: bool = True,
        token_type: int = 1,  # 1=text, 0=generated
        cmap = None,
        title: Optional[str] = None
    ) -> bool:
        """
        Create a sequential heatmap visualization for text or generated tokens.
        
        Unlike image tokens that have a 2D spatial arrangement, text and generated tokens
        are displayed in a more sequential layout, but still with a grid structure to 
        maintain consistency with the image visualizations.
        
        Args:
            type_df: DataFrame with tokens of a specific type
            ax: Matplotlib axis for plotting
            layer_idx: Layer index
            global_max: Maximum value for colormap normalization (for unified scale)
            show_values: Whether to show values in cells
            token_type: Token type (1=text, 0=generated)
            cmap: Colormap to use
            title: Optional title override
            
        Returns:
            True if successful, False otherwise
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import math
        
        if type_df.empty:
            token_type_name = "Text" if token_type == 1 else "Generated"
            ax.text(0.5, 0.5, f"No {token_type_name} tokens in layer {layer_idx}", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return False
        
        # Get token data
        tokens = type_df.sort_values("token_index")
        
        # Get max importance weight
        max_weight = tokens["importance_weight"].max()
        
        # If using unified scale, normalize weights
        if global_max is not None and global_max > 0:
            tokens["normalized_weight"] = tokens["importance_weight"] / global_max
            weight_column = "normalized_weight"
        else:
            weight_column = "importance_weight"
        
        # Create a grid layout for the tokens
        num_tokens = len(tokens)
        
        # Determine grid dimensions - aim for roughly square layout
        grid_width = math.ceil(math.sqrt(num_tokens))
        grid_height = math.ceil(num_tokens / grid_width)
        
        # Create a background with neutral color
        background_color = "#f0f0f0" if token_type == 1 else "#f0fff0"  # Light gray or light green
        
        # Determine token positions in the grid
        positions = {}
        row_heights = []
        col_widths = []
        
        # Fill in positions row by row
        token_iter = tokens.iterrows()
        for row in range(grid_height):
            row_tokens = []
            for col in range(grid_width):
                try:
                    idx, token = next(token_iter)
                    positions[(row, col)] = token
                    row_tokens.append(token)
                except StopIteration:
                    # No more tokens
                    break
            
            # Calculate row height - based on largest token text
            max_len = max([len(str(t["token_text"])) for t in row_tokens]) if row_tokens else 1
            row_heights.append(max(1.0, max_len / 10))  # Scale based on text length
        
        # Ensure all rows have a minimum height
        row_heights = [max(0.8, h) for h in row_heights]
        
        # Create a more functional grid visualization
        import matplotlib.gridspec as gridspec
        
        # Create figure with grid layout
        grid = gridspec.GridSpec(grid_height, grid_width)
        
        # Draw background
        rect = plt.Rectangle(
            (0, 0),
            1, 1,
            facecolor=background_color,
            edgecolor='none',
            alpha=0.3,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Draw tokens in the grid
        for pos, token in positions.items():
            row, col = pos
            
            # Get token info
            token_idx = token["token_index"]
            token_text = token["token_text"]
            weight = token[weight_column]
            
            # Calculate cell position (normalized coordinates)
            x_start = col / grid_width
            y_start = row / grid_height
            width = 1 / grid_width
            height = 1 / grid_height
            
            # Apply gamma correction for better contrast
            contrast_val = weight ** 0.5
            
            # Determine alpha based on weight
            alpha = min(0.9, contrast_val + 0.3) if weight > 0.01 else 0.15
            
            # Create colored rectangle
            rect = plt.Rectangle(
                (x_start, y_start),
                width, height,
                facecolor=cmap(contrast_val),
                edgecolor='black',
                linewidth=0.5,
                alpha=alpha,
                transform=ax.transAxes
            )
            ax.add_patch(rect)
            
            # Add token text and index
            if show_values:
                # Add token_text centered in the cell
                ax.text(
                    x_start + width/2,
                    y_start + height*0.7,  # Positioned near the top
                    str(token_text),
                    ha='center', va='center',
                    fontsize=8,
                    transform=ax.transAxes,
                    color='black'
                )
                
                # Add token index below the text
                ax.text(
                    x_start + width/2,
                    y_start + height*0.5,  # Middle
                    f"idx:{token_idx}",
                    ha='center', va='center',
                    fontsize=7,
                    transform=ax.transAxes,
                    color='darkblue'
                )
                
                # Add weight value at the bottom
                ax.text(
                    x_start + width/2,
                    y_start + height*0.3,  # Near the bottom
                    f"{weight:.3f}",
                    ha='center', va='center',
                    fontsize=7,
                    transform=ax.transAxes,
                    color='white' if weight > 0.4 else 'black',
                    weight='bold'
                )
        
        # Add legend with stats
        import matplotlib.patches as mpatches
        
        token_type_name = "Text" if token_type == 1 else "Generated"
        active_tokens = sum(1 for _, t in tokens.iterrows() if t["importance_weight"] > 0.01)
        
        if global_max is not None:
            # Using unified scale, need to show original max
            orig_max = max_weight
            norm_max = orig_max / global_max if global_max > 0 else 0.0
            legend_text = f"{token_type_name} Tokens ({active_tokens} active of {num_tokens}) - " \
                        f"Max: {orig_max:.3f}, Norm: {norm_max:.3f}"
        else:
            # Using type-specific scale
            legend_text = f"{token_type_name} Tokens ({active_tokens} active of {num_tokens}) - " \
                        f"Max: {max_weight:.3f}"
        
        legend_handles = [mpatches.Patch(color=cmap(0.7), label=legend_text)]
        
        # Add legend to plot
        ax.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            fancybox=True,
            shadow=True
        )
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f"{token_type_name} Tokens - Layer {layer_idx}", fontsize=14)
        
        ax.axis('off')
        return True