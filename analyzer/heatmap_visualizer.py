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
        Draw 5-panel heatmap (text / base / patch / generated / distribution)
        for **each layer**. Key modifications:
        1. Calculate max value for each token-type separately to avoid domination by high-value tokens
        2. When unified_colorscale=True, still use global_max but apply stretch_factor to smaller token types
        3. Pass type_maxes and stretch factors to child functions through kwargs for consistent normalization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        saved_paths = []
        combined_dir = os.path.join(output_dir, "combined_heatmaps")
        os.makedirs(combined_dir, exist_ok=True)

        for layer_idx in sorted(df["layer"].unique()):
            layer_df = df[df["layer"] == layer_idx]
            if layer_df.empty:
                continue

            # ------- 1️⃣ Calculate max values for each token type --------
            # This is critical for proper normalization and visualization
            text_max = layer_df.query("token_type==1")["importance_weight"].max()
            img_max  = layer_df.query("token_type==2")["importance_weight"].max()
            gen_max  = layer_df.query("token_type==0")["importance_weight"].max()
            global_max = layer_df["importance_weight"].max()

            # Avoid NaN values and ensure at least a small positive value
            text_max = 1e-9 if pd.isna(text_max) else max(text_max, 1e-9)
            img_max  = 1e-9 if pd.isna(img_max)  else max(img_max,  1e-9)
            gen_max  = 1e-9 if pd.isna(gen_max)  else max(gen_max,  1e-9)

            # Store all max values in a dictionary for easy access by token_type
            type_maxes = {
                0: gen_max,
                1: text_max,
                2: img_max
            }

            # For unified scale, calculate stretch factors (limit to 5x max enhancement)
            # This prevents text/generated tokens from being too faint when their max values
            # are much smaller than image tokens
            if unified_colorscale:
                stretch = {k: min(global_max / v, 5.0) for k, v in type_maxes.items()}
            else:
                # For independent scales, no stretching needed
                stretch = {k: 1.0 for k in type_maxes}

            # ------- 2️⃣ Create 5 subplot rows ------
            fig = Figure(figsize=(20, 30), dpi=110)
            canvas = FigureCanvas(fig)
            grids = fig.add_gridspec(5, 1, height_ratios=[2, 2, 2, 2, 1])

            # Text Tokens Panel
            ax_text = fig.add_subplot(grids[0])
            self._create_specific_token_heatmap(
                layer_df, feature_mapping, original_image, ax_text,
                layer_idx, type_maxes, stretch, show_values,
                token_type=1, title="Text Tokens"
            )

            # Image-Base Tokens Panel
            ax_base = fig.add_subplot(grids[1])
            self._create_specific_token_heatmap(
                layer_df, feature_mapping, original_image, ax_base,
                layer_idx, type_maxes, stretch, show_values,
                token_type=2, is_patch=False, title="Image Base Tokens"
            )

            # Image-Patch Tokens Panel
            ax_patch = fig.add_subplot(grids[2])
            self._create_specific_token_heatmap(
                layer_df, feature_mapping, original_image, ax_patch,
                layer_idx, type_maxes, stretch, show_values,
                token_type=2, is_patch=True, title="Image Patch Tokens"
            )

            # Generated Tokens Panel
            ax_gen = fig.add_subplot(grids[3])
            self._create_specific_token_heatmap(
                layer_df, feature_mapping, original_image, ax_gen,
                layer_idx, type_maxes, stretch, show_values,
                token_type=0, title="Generated Tokens"
            )

            # Distribution Analysis Panel
            ax_dist = fig.add_subplot(grids[4])
            self._create_token_distribution_for_layer(layer_df, ax_dist, layer_idx, global_max)

            # ---- Save the complete visualization ----
            scale_note = "(Unified Scale)" if unified_colorscale else "(Independent Scale)"
            fig.suptitle(f"Layer {layer_idx} – Complete Token Map {scale_note}\nTarget '{target_text}' (idx:{target_idx})", fontsize=20)
            out_path = os.path.join(combined_dir, f"combined_L{layer_idx}.png")
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            canvas.draw(); fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths.append(out_path)
            print(f"✓ combined heatmap saved: {out_path}")

        return saved_paths

    def _create_token_distribution_for_layer(
        self,
        layer_df: pd.DataFrame,
        ax,
        layer_idx: int,
        global_max: Optional[float] = None
    ) -> None:
        """
        Creates a visualization showing the distribution of token importance weights for a specific layer.
        
        This function generates a bar chart that visualizes how importance weights are distributed
        across different token types (text, image base, image patch, generated) within a specific layer.
        
        Key improvements:
        1. Better handling of global_max normalization references
        2. More informative labels and statistics
        3. Improved visual design for better readability
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Check if we have data
        if layer_df.empty:
            ax.text(0.5, 0.5, f"No token data available for layer {layer_idx}", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Divide tokens by their types
        text_tokens = layer_df[layer_df["token_type"] == 1]
        image_tokens = layer_df[layer_df["token_type"] == 2]
        generated_tokens = layer_df[layer_df["token_type"] == 0]
        
        # Try to divide image tokens into base and patch using feature mapping
        base_tokens = pd.DataFrame()
        patch_tokens = pd.DataFrame()
        
        if not image_tokens.empty and hasattr(self, 'feature_mapping') and self.feature_mapping:
            base_indices = set()
            patch_indices = set()
            
            # Extract base token indices from feature mapping
            if ("base_feature" in self.feature_mapping and 
                "positions" in self.feature_mapping["base_feature"]):
                base_indices = set(int(idx) for idx in self.feature_mapping["base_feature"]["positions"].keys())
                
            # Extract patch token indices from feature mapping
            if ("patch_feature" in self.feature_mapping and 
                "positions" in self.feature_mapping["patch_feature"]):
                patch_indices = set(int(idx) for idx in self.feature_mapping["patch_feature"]["positions"].keys())
            
            # Filter tokens by these indices
            base_tokens = image_tokens[image_tokens["token_index"].isin(base_indices)]
            patch_tokens = image_tokens[image_tokens["token_index"].isin(patch_indices)]
        
        # Fallback if we couldn't separate base/patch
        if base_tokens.empty and patch_tokens.empty:
            base_tokens = image_tokens
        
        # Prepare data for visualization
        token_categories = []
        token_counts = []
        max_weights = []
        mean_weights = []
        median_weights = []
        total_weights = []
        colors = []
        
        # Analyze text tokens
        if not text_tokens.empty:
            token_categories.append("Text")
            token_counts.append(len(text_tokens))
            max_weights.append(text_tokens["importance_weight"].max())
            mean_weights.append(text_tokens["importance_weight"].mean())
            median_weights.append(text_tokens["importance_weight"].median())
            total_weights.append(text_tokens["importance_weight"].sum())
            colors.append('royalblue')
        
        # Analyze base image tokens
        if not base_tokens.empty:
            token_categories.append("Image (Base)")
            token_counts.append(len(base_tokens))
            max_weights.append(base_tokens["importance_weight"].max())
            mean_weights.append(base_tokens["importance_weight"].mean())
            median_weights.append(base_tokens["importance_weight"].median())
            total_weights.append(base_tokens["importance_weight"].sum())
            colors.append('firebrick')
        
        # Analyze patch image tokens
        if not patch_tokens.empty:
            token_categories.append("Image (Patch)")
            token_counts.append(len(patch_tokens))
            max_weights.append(patch_tokens["importance_weight"].max())
            mean_weights.append(patch_tokens["importance_weight"].mean())
            median_weights.append(patch_tokens["importance_weight"].median())
            total_weights.append(patch_tokens["importance_weight"].sum())
            colors.append('orangered')
        
        # Analyze generated tokens
        if not generated_tokens.empty:
            token_categories.append("Generated")
            token_counts.append(len(generated_tokens))
            max_weights.append(generated_tokens["importance_weight"].max())
            mean_weights.append(generated_tokens["importance_weight"].mean())
            median_weights.append(generated_tokens["importance_weight"].median())
            total_weights.append(generated_tokens["importance_weight"].sum())
            colors.append('forestgreen')
        
        # If no valid categories, show message and return
        if not token_categories:
            ax.text(0.5, 0.5, "No token categories available for analysis", 
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            return
        
        # Calculate total influence percentage for each token type
        total_influence = sum(total_weights)
        if total_influence > 0:
            influence_percentages = [100 * w / total_influence for w in total_weights]
        else:
            influence_percentages = [0] * len(total_weights)
        
        # Create positions for the bars
        x_pos = np.arange(len(token_categories))
        width = 0.35  # Width of the bars
        
        # Create bars for token counts (secondary axis)
        ax2 = ax.twinx()
        count_bars = ax2.bar(x_pos - width/2, token_counts, width, alpha=0.3, color='gray', label='Count')
        ax2.set_ylabel('Token Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, max(token_counts) * 1.2 if token_counts else 100)
        
        # Create bars for influence percentage (primary axis)
        influence_bars = ax.bar(x_pos + width/2, influence_percentages, width, color=colors, alpha=0.7, label='% Influence')
        ax.set_ylabel('% of Total Influence', color='black')
        ax.set_ylim(0, max(influence_percentages) * 1.2 if influence_percentages else 100)
        
        # Add data labels on bars
        for i, (count, pct) in enumerate(zip(token_counts, influence_percentages)):
            # Token count label
            ax2.text(i - width/2, count, str(count), 
                    ha='center', va='bottom', fontsize=8, color='dimgray')
            
            # Influence percentage label
            if pct > 0.1:
                ax.text(i + width/2, pct, f"{pct:.1f}%", 
                        ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')
        
        # Set x-axis labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(token_categories)
        
        # Add a title
        ax.set_title(f"Token Distribution Analysis - Layer {layer_idx}", fontsize=14)
        
        # Add a horizontal line for average influence
        if token_categories:
            avg_influence = 100 / len(token_categories)
            ax.axhline(y=avg_influence, color='black', linestyle='--', alpha=0.5)
            ax.text(len(token_categories) - 0.5, avg_influence, f"Avg: {avg_influence:.1f}%", 
                    va='bottom', ha='right', fontsize=8, alpha=0.7)
        
        # Add summary statistics as text
        stats_text = "Token Statistics:\n"
        for i, category in enumerate(token_categories):
            stats_text += f"{category}: {token_counts[i]} tokens, "
            stats_text += f"Max: {max_weights[i]:.3f}, "
            stats_text += f"Mean: {mean_weights[i]:.3f}, "
            stats_text += f"Total: {total_weights[i]:.3f} ({influence_percentages[i]:.1f}%)\n"
        
        # Place text box in bottom right
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        # Add grid lines for readability
        ax.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Adjust layout
        ax.set_axisbelow(True)  # Put grid behind bars
        
        # If global_max is provided, add an annotation about normalization
        if global_max is not None and global_max > 0:
            norm_text = f"Note: Global max weight: {global_max:.3f}"
            ax.text(0.02, 0.98, norm_text, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', horizontalalignment='left', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))

    def _create_specific_token_heatmap(
        self, 
        layer_df, 
        feature_mapping, 
        original_image, 
        ax,
        layer_idx: int,
        type_maxes: Dict[int, float],         # NEW: maximum values by token type
        stretch: Dict[int, float],            # NEW: stretch factors for unified scale
        show_values: bool = True,
        token_type: int = 2,
        is_patch: bool = None,
        title: Optional[str] = None
    ) -> bool:
        """
        Draw one token-type heatmap. Unified or independent scale is determined 
        by the parameters passed through type_maxes/stretch.
        
        Key improvements:
        1. Use separate max values for each token type to avoid dominated visualizations
        2. Apply stretch factors when using unified colorscale
        3. Ensure minimum alpha of 0.3 for better visibility of low-weight tokens
        4. Remove gamma correction (val**0.5) for more accurate representation
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Filter to only tokens of the requested type
        type_df = layer_df[layer_df["token_type"] == token_type]
        if type_df.empty:
            ax.axis('off'); 
            return False

        # ---------- Normalization ----------
        # Get the max value for this token type
        max_w = type_maxes[token_type]
        
        # Create a copy to avoid modifying the original DataFrame
        type_df = type_df.copy()
        
        # Apply normalization with stretch factor
        # The stretch factor allows smaller token types to be more visible in unified scale mode
        type_df["norm_w"] = (type_df["importance_weight"] / max_w) * stretch[token_type]
        
        # Cap at 1.0 to maintain consistent color scale
        type_df["norm_w"] = type_df["norm_w"].clip(upper=1.0)

        # --------- Configure colormap based on token type ----------
        cmap = plt.get_cmap('hot') if token_type == 2 else \
            plt.get_cmap('Blues') if token_type == 1 else \
            plt.get_cmap('Greens')
            
        # For image tokens, use spatial visualization; for text/generated use sequential layout
        if token_type == 2:  # Image tokens
            # Determine if we should use patch or base visualization
            if is_patch is None:
                is_patch = "patch_feature" in feature_mapping
                
            # Use appropriate feature mapping
            feature_type = "patch_feature" if is_patch else "base_feature"
            feature_info = feature_mapping.get(feature_type, {})
            
            if not feature_info or "positions" not in feature_info:
                ax.text(0.5, 0.5, f"No {feature_type} mapping available", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                return False
                
            # Get grid dimensions
            if is_patch:
                grid_dims = feature_info.get("grid_unpadded", (0, 0))
            else:
                grid_dims = feature_info.get("grid", (0, 0))
                
            if not all(grid_dims):
                ax.text(0.5, 0.5, f"Invalid grid dimensions", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                return False
                
            grid_h, grid_w = grid_dims
            positions = feature_info.get("positions", {})
            
            # Prepare image background
            if is_patch:  # Patch visualization
                preview = original_image.copy()
                if "resized_dimensions" in feature_mapping:
                    w, h = feature_mapping["resized_dimensions"]
                    preview = preview.resize((w, h), Image.LANCZOS)
                    
                preview_w, preview_h = preview.size
                background_np = np.array(preview)
                
                # Calculate content area and padding
                resized_dims = feature_mapping.get("resized_dimensions", (0, 0))
                if resized_dims == (0, 0):
                    resized_w, resized_h = preview_w, preview_h
                else:
                    resized_w, resized_h = resized_dims
                    
                pad_h_total = preview_h - resized_h
                pad_w_total = preview_w - resized_w
                pad_top = pad_h_total // 2 if pad_h_total > 0 else 0
                pad_left = pad_w_total // 2 if pad_w_total > 0 else 0
                
                # Slightly darken background for better contrast with heatmap
                darkened_bg = background_np.astype(float) * 0.45
                darkened_bg = np.clip(darkened_bg, 0, 255).astype(np.uint8)
                
                # Display background with appropriate extent
                ax.imshow(
                    darkened_bg,
                    extent=(0, preview_w, preview_h, 0),
                    aspect='equal',
                    origin='upper'
                )
                
                # Calculate cell dimensions
                cell_height = resized_h / grid_h
                cell_width = resized_w / grid_w
            else:  # Base visualization
                # Use consistent size for base features
                target_size = (336, 336)
                
                # Resize image for background
                if original_image.mode != 'RGB':
                    resized_bg = original_image.convert('RGB').resize(target_size, Image.LANCZOS)
                else:
                    resized_bg = original_image.resize(target_size, Image.LANCZOS)
                    
                background_np = np.array(resized_bg)
                
                # Darken background for better contrast
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
            
            # Create dictionaries to map token indices to positions and weights
            token_positions = {}
            token_weights = {}
            
            # Convert positions from string keys to integers if needed
            for token_idx_str, pos in positions.items():
                try:
                    token_idx = int(token_idx_str) if isinstance(token_idx_str, str) else token_idx_str
                    r, c = pos
                    
                    # Only add valid positions
                    if 0 <= r < grid_h and 0 <= c < grid_w:
                        token_positions[token_idx] = (r, c)
                        token_weights[token_idx] = 0.0  # Default to zero
                except (ValueError, TypeError):
                    continue
                    
            # Update weights from DataFrame
            for _, row in type_df.iterrows():
                token_idx = row["token_index"]
                weight = row["importance_weight"]
                norm_weight = row["norm_w"]
                
                if token_idx in token_positions:
                    token_weights[token_idx] = weight
                    
            # Get stats for title
            max_weight = max(token_weights.values()) if token_weights else 0.0
            active_tokens = sum(1 for w in token_weights.values() if w > 0.01)
                    
            # Draw each token cell
            for token_idx, (r, c) in token_positions.items():
                # Get weight and calculate normalized value
                weight = token_weights.get(token_idx, 0.0)
                
                # IMPROVED: Calculate normalized value with stretch factor
                norm_val = weight / max_w * stretch[token_type]
                norm_val = min(norm_val, 1.0)  # Cap at 1.0
                
                # Calculate cell position
                x_start = pad_left + c * cell_width
                y_start = pad_top + r * cell_height
                
                # IMPROVED: Use alpha that ensures low weights are still visible
                # Minimum alpha of 0.3 ensures all cells have some visibility
                alpha = 0.3 + 0.7 * norm_val
                
                # Create cell rectangle
                rect = plt.Rectangle(
                    (x_start, y_start), 
                    cell_width, cell_height,
                    facecolor=cmap(norm_val),  # NO gamma correction (removed val**0.5)
                    edgecolor='white' if norm_val > 0.5 else 'gray',
                    linewidth=0.5,
                    alpha=alpha
                )
                ax.add_patch(rect)
                
                # Show weight value if requested and significant
                if show_values and weight > 0.01:
                    ax.text(
                        x_start + cell_width/2, 
                        y_start + cell_height/2,
                        f"{weight:.2f}", 
                        ha='center', va='center',
                        fontsize=7, 
                        color='white' if norm_val > 0.6 else 'black'
                    )
                    
            # Add grid lines for better visualization
            for i in range(grid_h + 1):
                y = pad_top + i * cell_height
                ax.axhline(y=y, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
                
            for i in range(grid_w + 1):
                x = pad_left + i * cell_width
                ax.axvline(x=x, color='white', linestyle='-', linewidth=0.5, alpha=0.5)
                
            # For patch features, add content area border
            if is_patch:
                content_rect = plt.Rectangle(
                    (pad_left, pad_top),
                    resized_w, resized_h,
                    fill=False,
                    edgecolor='cyan',
                    linestyle='--',
                    linewidth=1.5
                )
                ax.add_patch(content_rect)
                
            # Add legend with statistics
            import matplotlib.patches as mpatches
            
            # IMPROVED: Show original max value rather than normalized value
            legend_text = f"Image Tokens ({active_tokens} active of {len(token_positions)}) - " \
                        f"Max(orig): {max_w:.3f}"
                        
            legend_handles = [mpatches.Patch(color=cmap(0.7), label=legend_text)]
            
            ax.legend(
                handles=legend_handles,
                loc='upper right',
                fancybox=True,
                shadow=True
            )
        else:
            # For text and generated tokens, use sequential visualization
            return self._create_sequential_token_heatmap(
                layer_df, type_df, feature_mapping, ax,
                layer_idx, type_maxes, stretch, show_values,
                token_type=token_type, title=title
            )
            
        # Set title if provided, or create a default one
        if title:
            ax.set_title(title, fontsize=14)
        else:
            feature_name = "Patch" if is_patch else "Base"
            ax.set_title(f"Image {feature_name} Tokens - Layer {layer_idx}", fontsize=14)
            
        ax.axis('off')
        return True

    def _create_sequential_token_heatmap(
        self,
        layer_df,
        type_df,
        feature_mapping,
        ax,
        layer_idx: int,
        type_maxes: Dict[int, float],    # NEW: maximum values by token type
        stretch: Dict[int, float],       # NEW: stretch factors for unified scale
        show_values: bool = True,
        token_type: int = 1,             # 1=text, 0=generated
        title: Optional[str] = None
    ) -> bool:
        """
        Create a grid-based visualization for text or generated tokens in a sequential layout.
        
        Key improvements:
        1. Use fixed grid width based on token count to avoid too many rows
        2. Ensure minimum alpha of 0.2 for better visibility of low-weight tokens
        3. Display token text, index, and weight on separate lines for clarity
        4. Apply stretch factors when using unified colorscale
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
        
        # Sort tokens by index for sequential ordering
        tokens = type_df.sort_values("token_index")
        num_tokens = len(tokens)
        
        # ---- Normalization ----
        # Get max value for this token type
        max_w = type_maxes[token_type]
        
        # IMPROVED: Apply normalization with stretch factor
        tokens = tokens.assign(norm_w = (tokens["importance_weight"] / max_w) * stretch[token_type])
        tokens["norm_w"] = tokens["norm_w"].clip(upper=1.0)  # Cap at 1.0
        
        # IMPROVED: Fixed grid width based on token count
        # This ensures better token layout with reasonable row counts
        grid_width = 12 if num_tokens > 120 else \
                    8  if num_tokens > 64  else \
                    math.ceil(math.sqrt(num_tokens))
        grid_height = math.ceil(num_tokens / grid_width)
        
        # Set colormap based on token type
        cmap = plt.get_cmap('Blues') if token_type == 1 else plt.get_cmap('Greens')
        
        # Create a neutral background
        background_color = "#f0f0f0" if token_type == 1 else "#f0fff0"
        rect = plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=background_color,
            edgecolor='none',
            alpha=0.3,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Arrange tokens in a grid layout
        positions = {}
        idx = 0
        for i in range(grid_height):
            for j in range(grid_width):
                if idx < num_tokens:
                    positions[(i, j)] = tokens.iloc[idx]
                    idx += 1
        
        # Draw tokens in the grid
        for (row, col), token in positions.items():
            # Get token info
            token_idx = token["token_index"]
            token_text = str(token["token_text"]).replace('\n', '⏎')[:15]  # Truncate for display
            weight = token["importance_weight"]
            norm_val = token["norm_w"]
            
            # Calculate cell position in normalized coordinates
            x_start = col / grid_width
            y_start = row / grid_height
            width = 1 / grid_width
            height = 1 / grid_height
            
            # IMPROVED: Ensure minimum alpha for better visibility
            # Alpha ranges from 0.2 to 1.0 based on normalized value
            alpha = 0.2 + 0.8 * norm_val
            
            # Create cell rectangle
            rect = plt.Rectangle(
                (x_start, y_start),
                width, height,
                facecolor=cmap(norm_val),  # NO gamma correction (removed val**0.5)
                edgecolor='black',
                linewidth=0.3,
                alpha=alpha,
                transform=ax.transAxes
            )
            ax.add_patch(rect)
            
            # IMPROVED: Display token information on separate lines for clarity
            # Line 1: Token text
            ax.text(
                x_start + width/2,
                y_start + height*0.75,  # Upper portion
                token_text,
                ha='center', va='center',
                fontsize=8,
                transform=ax.transAxes,
                color='black'
            )
            
            # Line 2: Token index
            ax.text(
                x_start + width/2,
                y_start + height*0.5,  # Middle
                f"idx:{token_idx}",
                ha='center', va='center',
                fontsize=7,
                transform=ax.transAxes,
                color='darkblue'
            )
            
            # Line 3: Weight value
            ax.text(
                x_start + width/2,
                y_start + height*0.25,  # Lower portion
                f"{weight:.2f}",
                ha='center', va='center',
                fontsize=7,
                transform=ax.transAxes,
                color='white' if norm_val > 0.5 else 'black',
                weight='bold'
            )
        
        # Add legend with statistics
        import matplotlib.patches as mpatches
        
        token_type_name = "Text" if token_type == 1 else "Generated"
        active_tokens = sum(1 for _, t in tokens.iterrows() if t["importance_weight"] > 0.01)
        
        # IMPROVED: Show original max rather than normalized value
        legend_text = f"{token_type_name} Tokens ({active_tokens} active of {num_tokens}) - " \
                    f"Max(orig): {max_w:.3f}"
                    
        legend_handles = [mpatches.Patch(color=cmap(0.7), label=legend_text)]
        
        ax.legend(
            handles=legend_handles,
            loc='upper right',
            fancybox=True,
            shadow=True
        )
        
        # Set title if provided, or create a default one
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f"{token_type_name} Tokens - Layer {layer_idx}", fontsize=14)
        
        ax.axis('off')
        return True