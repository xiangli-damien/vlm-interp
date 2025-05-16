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
        
        # Load CSV data
        print(f"Loading trace data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        
        # Check if the specified weight column exists
        if weight_column not in self.df.columns:
            print(f"Warning: Weight column '{weight_column}' not found in CSV. Available columns: {list(self.df.columns)}")
            if "predicted_top_prob" in self.df.columns:
                print(f"Using 'predicted_top_prob' as fallback weight column")
                self.weight_column = "predicted_top_prob"
            else:
                raise ValueError(f"Neither '{weight_column}' nor fallback column 'predicted_top_prob' found in CSV")
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}")
        try:
            with open(metadata_path, 'r') as f:
                self.meta = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.meta = {}
        
        # Add importance_weight column if using a different weight column
        if self.weight_column != "importance_weight":
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
        
        # Generate standard heatmaps
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
        
        # Generate additional visualizations if requested
        if self.include_all_token_types:
            # Create comprehensive token heatmap with all token types
            token_heatmap_path = self.create_comprehensive_token_heatmap(target_idx, target_text)
            if token_heatmap_path:
                vis_results.append(token_heatmap_path)
            
            # Create token analysis visualizations
            analysis_paths = self.create_token_analysis_visualizations()
            vis_results.extend(analysis_paths)
        
        # Print results summary
        total_files = len(vis_results)
        print(f"Generated {total_files} heatmap visualization files")
        
        return vis_results
    
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
    
    def create_comprehensive_token_heatmap(self, target_idx: int, target_text: str) -> Optional[str]:
        """
        Create a comprehensive heatmap visualization that includes text, image, and generated tokens.
        
        Args:
            target_idx: Index of the target token
            target_text: Text of the target token
            
        Returns:
            Path to the generated visualization file, or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from matplotlib import gridspec
            
            # Create output directory
            comp_dir = os.path.join(self.out_dir, "comprehensive_heatmaps")
            os.makedirs(comp_dir, exist_ok=True)
            
            # Divide tokens by their types
            text_tokens = self.df[self.df["token_type"] == 1]
            image_tokens = self.df[self.df["token_type"] == 2]
            generated_tokens = self.df[self.df["token_type"] == 0]
            
            # Group by token index and get the maximum importance weight
            grouped_text = text_tokens.groupby("token_index").agg({
                "token_text": "first", 
                "importance_weight": "max",
                "layer": "first"  # Keep layer info for sorting
            }).reset_index()
            
            grouped_image = image_tokens.groupby("token_index").agg({
                "token_text": "first", 
                "importance_weight": "max",
                "layer": "first"
            }).reset_index()
            
            grouped_generated = generated_tokens.groupby("token_index").agg({
                "token_text": "first", 
                "importance_weight": "max",
                "layer": "first"
            }).reset_index()
            
            # Sort by token index
            grouped_text = grouped_text.sort_values("token_index")
            grouped_image = grouped_image.sort_values("token_index")
            grouped_generated = grouped_generated.sort_values("token_index")
            
            # Get counts
            n_text = len(grouped_text)
            n_image = len(grouped_image)
            n_generated = len(grouped_generated)
            
            # Determine global max for unified color scale
            all_weights = np.concatenate([
                grouped_text["importance_weight"].values,
                grouped_image["importance_weight"].values,
                grouped_generated["importance_weight"].values
            ])
            global_max = np.max(all_weights) if len(all_weights) > 0 else 1.0
            
            # Create figure with gridspec for better layout control
            fig = plt.figure(figsize=(12, 10), facecolor='white')
            gs = gridspec.GridSpec(3, 1, height_ratios=[2, 4, 2])
            
            # 1. Text tokens section
            # 1. Text tokens section
            ax_text = plt.subplot(gs[0])
            if n_text > 0:
                # Create a matrix for text tokens (1 row x n_text columns)
                text_matrix = grouped_text["importance_weight"].values.reshape(1, -1)
                
                # Plot text token heatmap
                im_text = ax_text.imshow(
                    text_matrix, 
                    aspect='auto', 
                    cmap='viridis',
                    vmin=0,
                    vmax=global_max if self.unified_colorscale else None
                )
                
                # Add token text as x-tick labels
                ax_text.set_xticks(range(n_text))
                ax_text.set_xticklabels(
                    grouped_text["token_text"], 
                    rotation=45, 
                    ha="right", 
                    fontsize=8
                )
                ax_text.set_yticks([])
                ax_text.set_title("Text Tokens Importance Weights", fontsize=10)
            else:
                ax_text.text(0.5, 0.5, "No text tokens", ha='center', va='center')
                ax_text.axis('off')
            
            # 2. Image tokens section (in a grid if possible)
            ax_image = plt.subplot(gs[1])
            if n_image > 0:
                # Try to arrange in a grid if we have feature mapping
                if self.feature_mapping and "base_feature" in self.feature_mapping:
                    # Get grid dimensions
                    grid_h, grid_w = self.feature_mapping["base_feature"].get("grid", (1, n_image))
                    
                    # Create an empty grid
                    img_matrix = np.zeros((grid_h, grid_w))
                    
                    # Fill in values from positions
                    positions = self.feature_mapping["base_feature"].get("positions", {})
                    for idx, row in grouped_image.iterrows():
                        token_idx = row["token_index"]
                        # Convert string positions back to integers
                        pos = None
                        if str(token_idx) in positions:
                            pos = positions[str(token_idx)]
                        elif token_idx in positions:
                            pos = positions[token_idx]
                            
                        if pos:
                            r, c = pos
                            if 0 <= r < grid_h and 0 <= c < grid_w:
                                img_matrix[r, c] = row["importance_weight"]
                else:
                    # Simple 1D layout if no grid mapping
                    grid_h, grid_w = 1, n_image
                    img_matrix = grouped_image["importance_weight"].values.reshape(1, -1)
                
                # Load and display the background image
                try:
                    from PIL import Image
                    bg_img = Image.open(self.image_path)
                    ax_image.imshow(bg_img, aspect='equal', alpha=0.5)
                    
                    # Overlay heatmap
                    im_img = ax_image.imshow(
                        img_matrix,
                        aspect='auto',
                        cmap='hot',
                        alpha=0.7,
                        vmin=0,
                        vmax=global_max if self.unified_colorscale else None
                    )
                except Exception as e:
                    print(f"Error loading image: {e}")
                    # Fallback to just the heatmap
                    im_img = ax_image.imshow(
                        img_matrix,
                        aspect='auto',
                        cmap='hot',
                        vmin=0,
                        vmax=global_max if self.unified_colorscale else None
                    )
                
                ax_image.set_title("Image Tokens Importance (Overlay)", fontsize=10)
                ax_image.axis('off')
            else:
                ax_image.text(0.5, 0.5, "No image tokens", ha='center', va='center')
                ax_image.axis('off')
            
            # 3. Generated tokens section
            ax_gen = plt.subplot(gs[2])
            if n_generated > 0:
                # Create a matrix for generated tokens (1 row x n_generated columns)
                gen_matrix = grouped_generated["importance_weight"].values.reshape(1, -1)
                
                # Plot generated token heatmap
                im_gen = ax_gen.imshow(
                    gen_matrix, 
                    aspect='auto', 
                    cmap='plasma',
                    vmin=0,
                    vmax=global_max if self.unified_colorscale else None
                )
                
                # Add token text as x-tick labels
                ax_gen.set_xticks(range(n_generated))
                ax_gen.set_xticklabels(
                    grouped_generated["token_text"], 
                    rotation=45, 
                    ha="right", 
                    fontsize=8
                )
                ax_gen.set_yticks([])
                ax_gen.set_title("Generated Tokens Importance Weights", fontsize=10)
            else:
                ax_gen.text(0.5, 0.5, "No generated tokens", ha='center', va='center')
                ax_gen.axis('off')
            
            # Add shared colorbar
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            colorbar = plt.colorbar(im_image if 'im_image' in locals() else 
                                    (im_text if 'im_text' in locals() else 
                                     im_gen if 'im_gen' in locals() else None), 
                                   cax=cbar_ax)
            colorbar.set_label('Importance Weight')
            
            # Set overall title
            plt.suptitle(f"Comprehensive Token Importance for Target '{target_text}' (idx: {target_idx})", 
                         fontsize=14)
            
            # Adjust layout and save
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Make room for suptitle and colorbar
            out_path = os.path.join(comp_dir, f"comprehensive_heatmap_{target_idx}.png")
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            
            print(f"Created comprehensive token heatmap: {out_path}")
            return out_path
            
        except Exception as e:
            print(f"Error creating comprehensive token heatmap: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_token_analysis_visualizations(self) -> List[str]:
        """
        Create additional visualizations analyzing token usage patterns.
        
        Returns:
            List of paths to generated visualization files
        """
        output_files = []
        
        # Create output directory
        analysis_dir = os.path.join(self.out_dir, "token_analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 1. Token frequency and importance visualization
            vis_path = self._create_token_frequency_importance_plot(analysis_dir)
            if vis_path:
                output_files.append(vis_path)
            
            # 2. Layer-wise token importance patterns
            vis_path = self._create_layerwise_importance_plot(analysis_dir)
            if vis_path:
                output_files.append(vis_path)
            
            # 3. Token type distribution
            vis_path = self._create_token_type_distribution_plot(analysis_dir)
            if vis_path:
                output_files.append(vis_path)
            
            # 4. Correlation between predicted probability and importance
            vis_path = self._create_probability_importance_correlation_plot(analysis_dir)
            if vis_path:
                output_files.append(vis_path)
            
        except Exception as e:
            print(f"Error creating token analysis visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        return output_files
    
    def _create_token_frequency_importance_plot(self, save_dir: str) -> Optional[str]:
        """
        Create a visualization showing token frequency vs. importance.
        
        Args:
            save_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization, or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            # Group by token index to get frequency and average importance
            token_stats = self.df.groupby(['token_index', 'token_text', 'token_type']).agg({
                'importance_weight': ['mean', 'max', 'count']
            }).reset_index()
            
            # Flatten the column names
            token_stats.columns = ['_'.join(col) if col[1] else col[0] for col in token_stats.columns.values]
            
            # Rename to more intuitive column names
            token_stats = token_stats.rename(columns={
                'importance_weight_mean': 'avg_importance',
                'importance_weight_max': 'max_importance',
                'importance_weight_count': 'frequency'
            })
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Color mapping for token types
            token_type_colors = {0: 'green', 1: 'blue', 2: 'red'}
            token_type_labels = {0: 'Generated', 1: 'Text', 2: 'Image'}
            
            # Create scatter plot
            for token_type, color in token_type_colors.items():
                subset = token_stats[token_stats['token_type'] == token_type]
                if len(subset) > 0:
                    plt.scatter(
                        subset['frequency'], 
                        subset['max_importance'],
                        c=color, 
                        alpha=0.7,
                        s=subset['frequency'] * 20,  # Size based on frequency
                        label=token_type_labels[token_type]
                    )
            
            # Add labels for top tokens
            top_tokens = token_stats.nlargest(10, 'max_importance')
            for _, row in top_tokens.iterrows():
                plt.annotate(
                    row['token_text'],
                    xy=(row['frequency'], row['max_importance']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
            
            # Set axis labels and title
            plt.xlabel('Token Frequency (Count across Layers)')
            plt.ylabel('Maximum Importance Weight')
            plt.title('Token Frequency vs. Importance Analysis')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Token Type")
            
            # Save figure
            out_path = os.path.join(save_dir, 'token_frequency_importance.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            
            print(f"Created token frequency vs. importance plot: {out_path}")
            return out_path
            
        except Exception as e:
            print(f"Error creating token frequency vs. importance plot: {e}")
            return None
    
    def _create_layerwise_importance_plot(self, save_dir: str) -> Optional[str]:
        """
        Create a visualization showing how token importance changes across layers.
        
        Args:
            save_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization, or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import pandas as pd
            
            # Get all layers
            layers = sorted(self.df['layer'].unique())
            
            # Find top tokens by maximum importance
            top_tokens = self.df.groupby('token_index').agg({
                'token_text': 'first',
                'token_type': 'first',
                'importance_weight': 'max'
            }).nlargest(8, 'importance_weight')
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Color mapping for token types
            token_type_colors = {0: 'green', 1: 'blue', 2: 'red'}
            token_type_markers = {0: 'o', 1: 's', 2: '^'}
            
            # For each top token, plot importance across layers
            for _, row in top_tokens.iterrows():
                token_idx = row.name
                token_text = row['token_text']
                token_type = row['token_type']
                
                # Get importance values across layers
                layer_values = []
                for layer in layers:
                    layer_row = self.df[(self.df['token_index'] == token_idx) & 
                                        (self.df['layer'] == layer)]
                    if not layer_row.empty:
                        layer_values.append(layer_row['importance_weight'].iloc[0])
                    else:
                        layer_values.append(0)
                
                # Plot line for this token
                plt.plot(
                    layers, 
                    layer_values,
                    marker=token_type_markers[token_type],
                    color=token_type_colors[token_type],
                    label=f"{token_text} (idx: {token_idx})",
                    alpha=0.8,
                    linewidth=2
                )
            
            # Set axis labels and title
            plt.xlabel('Layer')
            plt.ylabel('Importance Weight')
            plt.title('Layer-wise Token Importance Patterns for Top Tokens')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Tokens", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Save figure
            out_path = os.path.join(save_dir, 'layerwise_importance_patterns.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            print(f"Created layer-wise importance patterns plot: {out_path}")
            return out_path
            
        except Exception as e:
            print(f"Error creating layer-wise importance patterns plot: {e}")
            return None
    
    def _create_token_type_distribution_plot(self, save_dir: str) -> Optional[str]:
        """
        Create a visualization showing the distribution of token types and their importance.
        
        Args:
            save_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization, or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Token type labels
            token_type_labels = {0: 'Generated', 1: 'Text', 2: 'Image'}
            token_type_colors = {0: 'green', 1: 'blue', 2: 'red'}
            
            # 1. Distribution of token counts by type
            token_counts = self.df.groupby(['layer', 'token_type']).size().reset_index(name='count')
            
            # Pivot for stacked bar chart
            pivot_counts = token_counts.pivot(index='layer', columns='token_type', values='count').fillna(0)
            
            # Ensure all token types are represented
            for token_type in [0, 1, 2]:
                if token_type not in pivot_counts.columns:
                    pivot_counts[token_type] = 0
            
            # Sort columns for consistent order
            pivot_counts = pivot_counts.reindex(sorted(pivot_counts.columns), axis=1)
            
            # Create stacked bar chart
            pivot_counts.plot(
                kind='bar', 
                stacked=True, 
                ax=ax1,
                color=[token_type_colors.get(col, 'gray') for col in pivot_counts.columns]
            )
            
            # Replace numeric token types with labels in legend
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(
                handles, 
                [token_type_labels.get(int(float(label)), f'Type {label}') for label in labels],
                title="Token Type"
            )
            
            ax1.set_xlabel('Layer')
            ax1.set_ylabel('Token Count')
            ax1.set_title('Distribution of Token Types Across Layers')
            ax1.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 2. Boxplot of importance by token type
            sns.boxplot(
                x='token_type', 
                y='importance_weight', 
                data=self.df,
                palette=token_type_colors,
                ax=ax2
            )
            
            ax2.set_xlabel('Token Type')
            ax2.set_ylabel('Importance Weight')
            ax2.set_title('Distribution of Importance Weights by Token Type')
            ax2.set_xticklabels([token_type_labels.get(i, f'Type {i}') for i in sorted(token_type_labels.keys())])
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            out_path = os.path.join(save_dir, 'token_type_distribution.png')
            plt.savefig(out_path, dpi=150)
            plt.close()
            
            print(f"Created token type distribution plot: {out_path}")
            return out_path
            
        except Exception as e:
            print(f"Error creating token type distribution plot: {e}")
            return None
    
    def _create_probability_importance_correlation_plot(self, save_dir: str) -> Optional[str]:
        """
        Create a visualization showing the correlation between predicted probability and importance weight.
        
        Args:
            save_dir: Directory to save the visualization
            
        Returns:
            Path to the saved visualization, or None if failed
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Check if we have prediction probability data
            if 'predicted_top_prob' not in self.df.columns:
                print("No predicted_top_prob column found for correlation analysis")
                return None
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Token type labels and colors
            token_type_labels = {0: 'Generated', 1: 'Text', 2: 'Image'}
            token_type_colors = {0: 'green', 1: 'blue', 2: 'red'}
            
            # Create scatter plot with regression line for each token type
            for token_type, color in token_type_colors.items():
                subset = self.df[self.df['token_type'] == token_type]
                if len(subset) > 0:
                    sns.regplot(
                        x='predicted_top_prob',
                        y='importance_weight',
                        data=subset,
                        scatter_kws={'alpha': 0.5, 's': 20},
                        line_kws={'color': color, 'lw': 2},
                        color=color,
                        label=token_type_labels[token_type]
                    )
            
            # Calculate overall correlation
            correlation = self.df['predicted_top_prob'].corr(self.df['importance_weight'])
            
            # Set axis labels and title
            plt.xlabel('Predicted Top Token Probability')
            plt.ylabel('Importance Weight')
            plt.title(f'Correlation between Prediction Probability and Importance Weight\nOverall r = {correlation:.3f}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(title="Token Type")
            
            # Save figure
            out_path = os.path.join(save_dir, 'probability_importance_correlation.png')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close()
            
            print(f"Created probability-importance correlation plot: {out_path}")
            return out_path
            
        except Exception as e:
            print(f"Error creating probability-importance correlation plot: {e}")
            return None

def run_multi_target_heatmap_analysis(
    trace_csv: str,
    metadata_json: str,
    image_path: str,
    output_dir: str,
    target_indices: Optional[List[int]] = None,
    weight_column: str = "importance_weight",
    composite_only: bool = True,
    unified_colorscale: bool = False,
    include_all_token_types: bool = False,
    debug: bool = False
) -> Dict[int, List[str]]:
    """
    Run heatmap analysis for multiple target tokens in the trace.
    
    Args:
        trace_csv: Path to CSV file with trace data
        metadata_json: Path to metadata JSON file
        image_path: Path to the image
        output_dir: Directory to save results
        target_indices: Specific target token indices to analyze (None for all)
        weight_column: Column to use for importance weights
        composite_only: Whether to only generate composite visualizations
        unified_colorscale: Whether to use unified color scale across layers
        include_all_token_types: Whether to include text and generated tokens
        debug: Whether to print debug information
        
    Returns:
        Dictionary mapping target indices to lists of generated visualization paths
    """
    hv = HeatmapVisualizer(
        csv_path=trace_csv,
        metadata_path=metadata_json,
        image_path=image_path,
        out_dir=output_dir,
        weight_column=weight_column,
        debug_mode=debug
    )
    
    # Configure visualizer
    hv.unified_colorscale = unified_colorscale
    hv.include_all_token_types = include_all_token_types
    
    # Run multi-target analysis
    results = hv.run_multi_target(
        target_indices=target_indices,
        composite_only=composite_only,
        show_values=not composite_only
    )
    
    return results