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
    Creates heatmap visualizations from semantic tracing CSV data without requiring
    the model or GPU resources.
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
        self.viz = EnhancedSemanticTracingVisualizer(
            output_dir=out_dir,
            debug_mode=debug_mode
        )
    
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
        
        # Generate heatmaps
        vis_results = self.viz.create_heatmaps_from_csv(
            trace_data=self.df,
            target_text=target_text,
            target_idx=target_idx,
            image_path=self.image_path,
            save_dir=self.out_dir,
            feature_mapping=self.feature_mapping,
            use_grid_visualization=True,
            show_values=show_values,
            composite_only=composite_only
        )
        
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
    
    def run_multi_target(
        self, 
        target_indices: Optional[List[int]] = None,
        composite_only: bool = True,
        show_values: bool = True
    ) -> Dict[int, List[str]]:
        """
        Generate heatmaps for multiple target tokens.
        
        Args:
            target_indices: List of target token indices to visualize (None for all)
            composite_only: If True, only create composite visualizations
            show_values: Whether to show weight values in cells
            
        Returns:
            Dictionary mapping target indices to lists of generated file paths
        """
        # Get list of target tokens from metadata
        if "target_tokens" not in self.meta or not self.meta["target_tokens"]:
            print("Error: No target tokens found in metadata")
            return {}
        
        all_targets = self.meta["target_tokens"]
        
        # Filter targets if indices provided
        if target_indices:
            targets = [t for t in all_targets if t.get("index") in target_indices]
        else:
            targets = all_targets
        
        if not targets:
            print("Error: No matching target tokens found")
            return {}
        
        print(f"Processing {len(targets)} target tokens")
        
        # Process each target
        results = {}
        for target in targets:
            target_idx = target.get("index", 0)
            target_text = target.get("text", "unknown")
            
            # Create subdirectory for this target
            target_dir = os.path.join(self.out_dir, f"token_{target_idx}")
            os.makedirs(target_dir, exist_ok=True)
            
            print(f"\nGenerating heatmaps for target token '{target_text}' (index: {target_idx})")
            
            # Generate heatmaps for this target
            target_viz = EnhancedSemanticTracingVisualizer(output_dir=target_dir, debug_mode=self.debug_mode)
            
            # Filter dataframe to focus on this target's tracing
            # This is important if CSV contains multiple traces with different trace_ids
            if "trace_id" in self.df.columns:
                trace_ids = self.df["trace_id"].unique()
                if len(trace_ids) > 1:
                    # Find records where this target is marked as a target
                    target_df = self.df[
                        (self.df["is_target"] == True) & 
                        (self.df["token_index"] == target_idx)
                    ]
                    # Get the trace_id for this target
                    if not target_df.empty:
                        target_trace_id = target_df["trace_id"].iloc[0]
                        # Filter to just this trace_id
                        target_df = self.df[self.df["trace_id"] == target_trace_id]
                        print(f"Using trace_id {target_trace_id} for target {target_idx}")
                    else:
                        target_df = self.df  # Fallback to all data
                else:
                    target_df = self.df  # Only one trace_id, use all data
            else:
                target_df = self.df  # No trace_id column, use all data
            
            # Generate visualizations
            vis_paths = target_viz.create_heatmaps_from_csv(
                trace_data=target_df,
                target_text=target_text,
                target_idx=target_idx,
                image_path=self.image_path,
                save_dir=target_dir,
                feature_mapping=self.feature_mapping,
                use_grid_visualization=True,
                show_values=show_values,
                composite_only=composite_only
            )
            
            results[target_idx] = vis_paths
            print(f"Generated {len(vis_paths)} visualization files for target {target_idx}")
        
        return results