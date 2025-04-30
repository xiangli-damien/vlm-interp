"""
Visualizer for semantic tracing results. This module provides visualization tools
for semantic tracing data, allowing for the creation of flow graphs and heatmaps
from saved CSV data, decoupled from the tracing process.
"""

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
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from collections import defaultdict


class SemanticTracingVisualizer:
    """
    Visualizes semantic tracing results from CSV data, creating flow graphs and heatmaps.
    Works independently from the tracing process, allowing for visualization without rerunning analysis.
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
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
    
    def visualize_from_csv(
    self,
    csv_path: str,
    metadata_path: Optional[str] = None,
    image_path: Optional[str] = None,
    target_token: Optional[Dict[str, Any]] = None,
    flow_graph_params: Optional[Dict[str, Any]] = None,
    heatmap_params: Optional[Dict[str, Any]] = None,
) -> List[str]:
        """
        Generate visualizations from a saved CSV trace file.
        
        Args:
            csv_path: Path to the CSV file with trace data
            metadata_path: Optional path to JSON metadata file
            image_path: Path to original image used in the trace (required for heatmaps)
            target_token: Optional information about the target token
            flow_graph_params: Parameters for flow graph visualization
            heatmap_params: Parameters for heatmap visualization
            
        Returns:
            List of paths to generated visualization files
        """
        print(f"Generating visualizations from CSV: {csv_path}")
        saved_paths = []
        
        # Set default parameters
        if flow_graph_params is None:
            flow_graph_params = {
                "output_format": "both",
                "align_tokens_by_layer": True,
                "show_orphaned_nodes": False,
                "min_edge_weight": 0.05,
                "use_variable_node_size": True,
                "debug_mode": self.debug_mode
            }
        
        if heatmap_params is None:
            heatmap_params = {
                "use_grid_visualization": True,
                "show_values": True,
                "composite_only": True
            }
        
        # Load CSV data
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return saved_paths
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded trace data with {len(df)} rows")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return saved_paths
        
        # Load metadata if available
        metadata = {}
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"Loaded metadata from {metadata_path}")
            except Exception as e:
                print(f"Error loading metadata: {e}")
        
        # Extract target token information
        if target_token is None:
            target_token = self._extract_target_token_info(df, metadata)
        
        # Create a directory for the visualizations
        target_idx = target_token.get("index", "unknown")
        target_text = target_token.get("text", "unknown")
        save_dir = os.path.join(self.output_dir, f"token_{target_idx}_{target_text}")
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Create flow graph visualization
        try:
            print("Creating token flow graph visualization...")
            flow_graph_paths = self.create_flow_graph(
                df, target_text, target_idx, save_dir, **flow_graph_params
            )
            saved_paths.extend(flow_graph_paths)
        except Exception as e:
            print(f"Error creating flow graph visualization: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Create heatmap visualizations if we have an image and feature mapping
        image_exists = image_path is not None and os.path.exists(image_path)
        feature_mapping_exists = "feature_mapping" in metadata and metadata["feature_mapping"]
        
        if not image_exists:
            print(f"Warning: Image path is missing or invalid: {image_path}")
        if not feature_mapping_exists:
            print(f"Warning: Feature mapping not found in metadata")
        
        if image_exists and feature_mapping_exists:
            try:
                print(f"Creating heatmap visualizations with image path: {image_path}")
                heatmap_paths = self.create_heatmaps(
                    df, target_text, target_idx, image_path, 
                    save_dir, metadata["feature_mapping"], **heatmap_params
                )
                saved_paths.extend(heatmap_paths)
            except Exception as e:
                print(f"Error creating heatmap visualizations: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Skipping heatmap visualizations: missing image path or feature mapping")
        
        # 3. Create trace data visualizations
        try:
            print("Creating trace data visualizations...")
            data_vis_paths = self.create_trace_data_visualizations(
                df, target_text, target_idx, save_dir
            )
            saved_paths.extend(data_vis_paths)
        except Exception as e:
            print(f"Error creating trace data visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"Visualization complete. Generated {len(saved_paths)} files.")
        return saved_paths
    
    def _extract_target_token_info(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract target token information from DataFrame and metadata.
        
        Args:
            df: DataFrame with trace data
            metadata: Metadata dictionary
            
        Returns:
            Dictionary with target token information
        """
        # First try to get from metadata
        if "target_tokens" in metadata and metadata["target_tokens"]:
            return metadata["target_tokens"][0]
        
        # Otherwise, extract from DataFrame
        target_tokens = df[df["is_target"] == True]
        if not target_tokens.empty:
            # Take the highest token index as the main target
            target_idx = target_tokens["token_index"].max()
            target_row = target_tokens[target_tokens["token_index"] == target_idx].iloc[0]
            
            return {
                "index": int(target_idx),
                "text": target_row["token_text"],
                "id": int(target_row["token_id"])
            }
        
        # Fallback to using the highest token index in the DataFrame
        highest_idx = df["token_index"].max()
        highest_row = df[df["token_index"] == highest_idx].iloc[0]
        
        return {
            "index": int(highest_idx),
            "text": highest_row["token_text"],
            "id": int(highest_row["token_id"])
        }
    
    def create_flow_graph(
        self,
        trace_data: pd.DataFrame,
        target_text: str,
        target_idx: int,
        save_dir: str,
        output_format: str = "both",
        align_tokens_by_layer: bool = True,
        show_orphaned_nodes: bool = False,  
        min_edge_weight: float = 0.05,
        use_variable_node_size: bool = True,
        min_node_size: int = 600,
        max_node_size: int = 1500,
        debug_mode: bool = False,
        dpi: int = 150,
        show_continuation_edges: bool = False,
        use_exponential_scaling: bool = True
    ) -> List[str]:
        """
        Create a flow graph visualization from trace data.
        
        Args:
            trace_data: DataFrame with trace data
            target_text: Text of the target token
            target_idx: Index of the target token
            save_dir: Directory to save the visualization
            output_format: Output format ("png", "svg", or "both")
            align_tokens_by_layer: Whether to align tokens in strict columns by layer
            show_orphaned_nodes: Whether to show nodes with no connections
            min_edge_weight: Minimum edge weight to display (filters weak connections)
            use_variable_node_size: Whether to vary node size based on weight
            min_node_size: Minimum node size for visualization
            max_node_size: Maximum node size for visualization
            debug_mode: Whether to print debug information
            dpi: DPI for PNG output
            show_continuation_edges: Whether to show dashed continuation edges between layers
            use_exponential_scaling: Whether to use exponential scaling for weights
            
        Returns:
            List of paths to saved visualizations
        """
        saved_paths = []
        
        # Check if we have data
        if trace_data.empty:
            print("No trace data available for flow graph visualization.")
            return saved_paths
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Extract unique layers ordered from lowest to highest
        layers = sorted(trace_data["layer"].unique())
        if debug_mode:
            print(f"Found {len(layers)} layers: {layers}")
        
        # First, create a mapping of node information
        node_map = {}  # {(layer, token_index): node_id}
        node_metadata = {}  # {node_id: metadata}
        
        # Process all rows to create nodes
        for _, row in trace_data.iterrows():
            layer = row["layer"]
            token_idx = row["token_index"]
            token_text = row["token_text"]
            token_type = row["token_type"]
            is_target = row["is_target"]
            
            # Create node ID in format "L{layer}_T{token_idx}"
            node_id = f"L{layer}_T{token_idx}"
            
            # Store mapping
            node_map[(layer, token_idx)] = node_id
            
            # Extract top prediction
            top_pred = ""
            if "predicted_top_token" in row and not pd.isna(row["predicted_top_token"]):
                top_pred_text = row["predicted_top_token"]
                top_pred_text = self._sanitize_text_for_display(top_pred_text)
                top_pred = f"→{top_pred_text}"
            
            # Clean token text for display
            display_text = self._sanitize_text_for_display(token_text)
            
            # Store node metadata
            node_metadata[node_id] = {
                "type": token_type,
                "text": display_text,
                "idx": token_idx,
                "layer": layer,
                "weight": row["predicted_top_prob"] if "predicted_top_prob" in row else 1.0,
                "top_pred": top_pred,
                "is_target": is_target
            }
        
        # Now add edges based on source-target relationships
        # First, add all nodes to the graph
        for node_id, metadata in node_metadata.items():
            G.add_node(node_id)
        
        # Process rows that have source information
        for _, row in trace_data.iterrows():
            if not row["is_target"] or pd.isna(row["sources_indices"]):
                continue
            
            layer = row["layer"]
            token_idx = row["token_index"]
            target_node_id = node_map.get((layer, token_idx))
            
            if not target_node_id:
                continue
            
            # Parse source indices and weights
            try:
                source_indices = [int(idx) for idx in row["sources_indices"].split(",") if idx.strip()]
                source_weights = [float(w) for w in row["sources_weights"].split(",") if w.strip()]
                
                # Ensure equal length
                if len(source_indices) != len(source_weights):
                    source_weights = [1.0] * len(source_indices)
            except:
                continue
            
            # Find previous layers
            prev_layers = [l for l in layers if l < layer]
            
            # Add edges for each source
            for src_idx, weight in zip(source_indices, source_weights):
                # Find the source in a previous layer
                source_node_id = None
                
                for prev_layer in reversed(prev_layers):
                    if (prev_layer, src_idx) in node_map:
                        source_node_id = node_map[(prev_layer, src_idx)]
                        break
                
                if source_node_id and weight >= min_edge_weight:
                    G.add_edge(
                        source_node_id,
                        target_node_id,
                        weight=weight,
                        saliency=weight
                    )
                    
                    if debug_mode:
                        print(f"Added edge: {source_node_id} -> {target_node_id} with weight {weight:.3f}")
        
        # Add continuation edges if requested
        if show_continuation_edges:
            token_layer_map = defaultdict(list)
            
            # Group token occurrences by layer
            for (layer, token_idx), node_id in node_map.items():
                token_layer_map[token_idx].append(layer)
            
            # Add continuation edges for tokens that appear in consecutive layers
            for token_idx, token_layers in token_layer_map.items():
                token_layers = sorted(token_layers)
                
                for i in range(len(token_layers) - 1):
                    current_layer = token_layers[i]
                    next_layer = token_layers[i+1]
                    
                    # Only if the layers are adjacent in our selected layers
                    if layers.index(next_layer) == layers.index(current_layer) + 1:
                        src_node_id = node_map.get((current_layer, token_idx))
                        dst_node_id = node_map.get((next_layer, token_idx))
                        
                        if src_node_id and dst_node_id and src_node_id in G and dst_node_id in G:
                            G.add_edge(
                                src_node_id,
                                dst_node_id,
                                weight=0.3,
                                is_continuation=True
                            )
        
        # Remove orphaned nodes if requested
        if not show_orphaned_nodes:
            orphaned_nodes = [n for n in G.nodes() if G.degree(n) == 0]
            G.remove_nodes_from(orphaned_nodes)
            
            if debug_mode:
                print(f"Removed {len(orphaned_nodes)} orphaned nodes")
        
        # Check if we have any nodes left
        if len(G.nodes()) == 0:
            print("No nodes left in graph after filtering. Cannot create visualization.")
            return saved_paths
        
        # Calculate positions for nodes
        pos = self._calculate_flow_graph_node_positions(G, node_metadata, align_tokens_by_layer)
        
        # Calculate node sizes based on weights
        node_sizes = self._calculate_flow_graph_node_sizes(
            G, node_metadata, use_variable_node_size, min_node_size, max_node_size, use_exponential_scaling
        )
        
        # Get node colors based on token type
        token_colors = {
            0: "#2ecc71",  # Generated = green
            1: "#3498db",  # Text = blue
            2: "#e74c3c"   # Image = red
        }
        
        node_colors = [
            token_colors.get(node_metadata.get(node, {}).get("type", 0), "#7f8c8d") 
            for node in G.nodes()
        ]
        
        # Create the visualization
        fig_width, fig_height = self._calculate_flow_graph_dimensions(G, node_metadata)
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Try to set a font that supports special characters
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Draw edges with width based on weight
        edge_width_multiplier = 5.0
        
        # First draw white "halo" for contrast
        for u, v, data in G.edges(data=True):
            if data.get('is_continuation', False):
                continue
                    
            weight = data.get("weight", 0.0)
            
            # Apply exponential scaling if requested
            if use_exponential_scaling:
                width = math.sqrt(weight) * edge_width_multiplier
            else:
                width = weight * edge_width_multiplier
                
            if width >= min_edge_weight * edge_width_multiplier:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=width + 1.5,
                    alpha=0.3,
                    edge_color='white'
                )
        
        # Then draw actual edges
        for u, v, data in G.edges(data=True):
            if data.get('is_continuation', False):
                continue
                    
            weight = data.get("weight", 0.0)
            
            # Apply exponential scaling if requested
            if use_exponential_scaling:
                width = math.sqrt(weight) * edge_width_multiplier
            else:
                width = weight * edge_width_multiplier
                
            if width >= min_edge_weight * edge_width_multiplier:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=max(width, 0.8),
                    alpha=min(0.8, max(0.2, weight * 1.5)),
                    arrows=True,
                    arrowsize=10,
                    connectionstyle="arc3,rad=0.1"
                )
        
        # Draw continuation edges
        if show_continuation_edges:
            continuation_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_continuation', False)]
            if continuation_edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=continuation_edges,
                    width=1.0,
                    alpha=0.5,
                    edge_color='gray',
                    style='dashed',
                    arrows=True,
                    arrowsize=8
                )
        
        # Create labels
        labels = {}
        for node in G.nodes():
            meta = node_metadata.get(node, {})
            token_text = meta.get('text', '')
            top_pred = meta.get('top_pred', '')
            if top_pred:
                labels[node] = f"{token_text} {top_pred}"
            else:
                labels[node] = token_text
        
        # Calculate font sizes
        base_font_size = 10
        font_sizes = []
        for i, node in enumerate(G.nodes()):
            is_target = node_metadata.get(node, {}).get("is_target", False)
            if use_variable_node_size:
                font_size = base_font_size * (node_sizes[i] / min_node_size) ** 0.25
            else:
                font_size = base_font_size * 1.2 if is_target else base_font_size
            font_sizes.append(min(font_size, 12))
        
        # Draw each label separately for better control
        for i, node in enumerate(G.nodes()):
            x, y = pos[node]
            label = labels.get(node, "")
            plt.text(
                x, y,
                label,
                fontsize=font_sizes[i],
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                zorder=100
            )
        
        # Add layer labels
        for i, layer in enumerate(sorted(set(node_metadata[n]["layer"] for n in G.nodes()))):
            layer_nodes = [n for n in G.nodes() if node_metadata[n]["layer"] == layer]
            if layer_nodes:
                layer_x = np.mean([pos[n][0] for n in layer_nodes])
                max_y = max([pos[n][1] for n in layer_nodes]) if layer_nodes else 0
                plt.text(layer_x, max_y + 1.5, f"Layer {layer}", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Set title
        title = f"Semantic Trace Flow Graph for Token '{target_text}' (idx: {target_idx})"
        plt.title(title, fontsize=16)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#3498db", markersize=10, label='Text Token'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#e74c3c", markersize=10, label='Image Token'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#2ecc71", markersize=10, label='Generated Token')
        ]
        
        if use_variable_node_size:
            legend_elements.extend([
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Low Influence'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='High Influence')
            ])
        
        # Add edge type legend
        if show_continuation_edges:
            legend_elements.extend([
                plt.Line2D([0], [0], linestyle='-', color='black', linewidth=2, label='Token Influence'),
                plt.Line2D([0], [0], linestyle='--', color='gray', linewidth=1, label='Token Continuation')
            ])
        else:
            legend_elements.extend([
                plt.Line2D([0], [0], linestyle='-', color='black', linewidth=2, label='Token Influence')
            ])
        
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Remove axis
        plt.axis('off')
        plt.tight_layout(pad=3.0)
        
        # Save in requested formats
        if output_format in ["png", "both"]:
            png_path = os.path.join(save_dir, f"flow_graph_{target_idx}.png")
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            saved_paths.append(png_path)
        
        if output_format in ["svg", "both"]:
            # Use non-font embedding for SVG
            mpl.rcParams['svg.fonttype'] = 'none'
            
            svg_path = os.path.join(save_dir, f"flow_graph_{target_idx}.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            saved_paths.append(svg_path)
        
        plt.close()
        
        print(f"Flow graph visualization saved: {saved_paths}")
        return saved_paths
    
    def _calculate_flow_graph_node_positions(self, G, node_metadata, align_tokens_by_layer):
        """Calculate positions for nodes in the flow graph with optimized spacing"""
        # Get all layers present in the graph
        graph_layers = sorted(set(node_metadata[n]["layer"] for n in G.nodes()))
        
        # Organize nodes by layer
        layer_nodes = {}
        for node in G.nodes():
            layer = node_metadata[node]["layer"]
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(node)
        
        # Calculate vertical spacing based on the maximum nodes in any layer
        max_nodes_in_layer = max(len(nodes) for nodes in layer_nodes.values())
        x_spacing = 4.0  # Reduced from 6.0
        y_spacing = 2.0  # Reduced from 3.0
        
        # Position nodes with fixed x by layer, distributed y with increased spacing
        pos = {}
        for layer_idx, layer_node_list in layer_nodes.items():
            # Layer-specific x-coordinate
            layer_x = graph_layers.index(layer_idx) * x_spacing
            
            # Sort nodes by token index for consistent order
            layer_node_list.sort(key=lambda n: node_metadata[n]["idx"])
            
            # Distribute nodes vertically
            num_nodes = len(layer_node_list)
            if num_nodes > 0:
                # Calculate vertical spacing
                total_height = (num_nodes - 1) * y_spacing
                start_y = -total_height / 2
                
                for i, node_id in enumerate(layer_node_list):
                    pos[node_id] = (layer_x, start_y + i * y_spacing)
        
        return pos
    
    def _calculate_flow_graph_node_sizes(
        self, G, node_metadata, use_variable_node_size, min_node_size, max_node_size, use_exponential_scaling
    ):
        """Calculate node sizes for the flow graph based on weights"""
        if use_variable_node_size:
            # Get all weights
            all_weights = [
                node_metadata.get(node, {}).get("weight", 1.0) 
                for node in G.nodes()
            ]
            
            if all_weights:
                max_weight = max(all_weights)
                min_weight = min(all_weights)
                weight_range = max_weight - min_weight
                
                # Calculate size for each node
                node_sizes = []
                for node in G.nodes():
                    weight = node_metadata.get(node, {}).get("weight", 1.0)
                    
                    # Scale node size based on weight
                    if weight_range > 0:
                        # Normalized weight between 0 and 1
                        norm_weight = (weight - min_weight) / weight_range
                        
                        if use_exponential_scaling:
                            # Exponential scaling (stronger emphasis on higher weights)
                            scaled_weight = math.pow(norm_weight, 0.5)
                            size = min_node_size + (max_node_size - min_node_size) * scaled_weight
                        else:
                            # Standard linear scaling
                            size = min_node_size + (max_node_size - min_node_size) * norm_weight
                    else:
                        size = (min_node_size + max_node_size) / 2
                    
                    node_sizes.append(size)
            else:
                node_sizes = [min_node_size] * len(G.nodes())
        else:
            # Use fixed size based on node type
            node_sizes = []
            for node in G.nodes():
                is_target = node_metadata.get(node, {}).get("is_target", False)
                size = max_node_size if is_target else min_node_size
                node_sizes.append(size)
        
        return node_sizes
    
    def _calculate_flow_graph_dimensions(self, G, node_metadata):
        # Get all layers present in the graph
        graph_layers = sorted(set(node_metadata[n]["layer"] for n in G.nodes()))
        
        # Count nodes per layer
        nodes_per_layer = {}
        for node in G.nodes():
            layer = node_metadata[node]["layer"]
            if layer not in nodes_per_layer:
                nodes_per_layer[layer] = 0
            nodes_per_layer[layer] += 1
        
        max_nodes_in_layer = max(nodes_per_layer.values()) if nodes_per_layer else 1
        
        fig_width = min(16, len(graph_layers) * 2 + 2)  # Width scales with layers, max 16 inches
        fig_height = min(12, max(6, max_nodes_in_layer * 0.8))  # Height scales with max nodes, min 6, max 12 inches
        
        return fig_width, fig_height
    
    def create_heatmaps(
        self,
        trace_data: pd.DataFrame,
        target_text: str,
        target_idx: int,
        image_path: str,
        save_dir: str,
        feature_mapping: Dict[str, Any],
        use_grid_visualization: bool = True,
        show_values: bool = True,
        composite_only: bool = True
    ) -> List[str]:
        """
        Create heatmap visualizations for image token influence.
        
        Args:
            trace_data: DataFrame with trace data
            target_text: Text of the target token
            target_idx: Index of the target token
            image_path: Path to the original image
            save_dir: Directory to save visualizations
            feature_mapping: Dictionary with feature mapping information
            use_grid_visualization: Whether to use grid-based visualization
            show_values: Whether to show numeric values in cells
            composite_only: Whether to only generate composite heatmaps
            
        Returns:
            List of paths to saved visualizations
        """
        saved_paths = []
        
        # Check if we have necessary data
        if trace_data.empty:
            print("No trace data available for heatmap visualization.")
            return saved_paths
        
        # Check if we have feature mapping
        if not feature_mapping:
            print("No feature mapping available for heatmap visualization.")
            return saved_paths
        
        # Prepare output directories if we're creating individual heatmaps
        if not composite_only:
            base_heatmap_dir = os.path.join(save_dir, "base_feature_heatmaps")
            patch_heatmap_dir = os.path.join(save_dir, "patch_feature_heatmaps")
            os.makedirs(base_heatmap_dir, exist_ok=True)
            os.makedirs(patch_heatmap_dir, exist_ok=True)
        
        # Load the original image
        try:
            original_image = Image.open(image_path)
            print(f"Loaded image: {image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return saved_paths
        
        # Create a spatial preview image (resized to match feature mapping)
        spatial_preview_image = original_image.copy()
        if "resized_dimensions" in feature_mapping:
            width, height = feature_mapping["resized_dimensions"]
            spatial_preview_image = spatial_preview_image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Extract base and patch feature information
        base_feature_info = feature_mapping.get("base_feature", {})
        patch_feature_info = feature_mapping.get("patch_feature", {})
        
        # Tracking for valid layers
        base_valid_layers = []
        patch_valid_layers = []
        base_heatmap_paths = []
        patch_heatmap_paths = []
        
        # Extract unique layers
        layers = sorted(trace_data["layer"].unique())
        
        # Process each layer
        for layer_idx in layers:
            # Find all image tokens in this layer
            layer_data = trace_data[trace_data["layer"] == layer_idx]
            
            # Extract image token weights
            image_tokens = layer_data[layer_data["token_type"] == 2]
            
            if image_tokens.empty:
                continue
            
            # Calculate normalized weights for visualization
            max_weight = image_tokens["predicted_top_prob"].max()
            if max_weight <= 0:
                continue
                
            image_token_weights = {}
            for _, row in image_tokens.iterrows():
                token_idx = row["token_index"]
                weight = row["predicted_top_prob"] / max_weight  # Normalize
                image_token_weights[token_idx] = weight
            
            # 1. Create base feature heatmap if available
            if base_feature_info and "grid" in base_feature_info and "positions" in base_feature_info:
                base_grid_h, base_grid_w = base_feature_info["grid"]
                positions = base_feature_info["positions"]
                
                # Convert string keys to integers if necessary
                if all(isinstance(k, str) for k in positions.keys()):
                    positions = {int(k): v for k, v in positions.items()}
                
                # Initialize empty heatmap
                base_heatmap = np.zeros((base_grid_h, base_grid_w), dtype=np.float32)
                
                # Fill the heatmap with normalized weights
                mapped_tokens = 0
                for token_idx, weight in image_token_weights.items():
                    position = positions.get(token_idx)
                    if position:
                        r, c = position
                        if 0 <= r < base_grid_h and 0 <= c < base_grid_w:
                            base_heatmap[r, c] = weight
                            mapped_tokens += 1
                
                # Create visualization if we have data and not composite_only
                if mapped_tokens > 0 and np.max(base_heatmap) > 0 and not composite_only:
                    base_path = self._create_base_feature_overlay(
                        heatmap=base_heatmap,
                        original_image=original_image,
                        grid_size=(base_grid_h, base_grid_w),
                        layer_idx=layer_idx,
                        target_idx=target_idx,
                        title=f"Base Image Token Influence - Layer {layer_idx}",
                        save_path=os.path.join(base_heatmap_dir, f"base_influence_layer_{layer_idx}_{target_idx}.png"),
                        use_grid_visualization=use_grid_visualization
                    )
                    if base_path:
                        base_heatmap_paths.append(base_path)
                        base_valid_layers.append(layer_idx)
                elif mapped_tokens > 0 and np.max(base_heatmap) > 0:
                    # Just track valid layers for composite
                    base_valid_layers.append(layer_idx)
            
            # 2. Create patch feature heatmap if available
            if patch_feature_info and "grid_unpadded" in patch_feature_info and "positions" in patch_feature_info:
                prob_grid_h, prob_grid_w = patch_feature_info["grid_unpadded"]
                positions = patch_feature_info["positions"]
                
                # Convert string keys to integers if necessary
                if all(isinstance(k, str) for k in positions.keys()):
                    positions = {int(k): v for k, v in positions.items()}
                
                # Initialize empty heatmap
                patch_heatmap = np.zeros((prob_grid_h, prob_grid_w), dtype=np.float32)
                
                # Fill the heatmap with normalized weights
                mapped_tokens = 0
                for token_idx, weight in image_token_weights.items():
                    position = positions.get(token_idx)
                    if position:
                        r, c = position
                        if 0 <= r < prob_grid_h and 0 <= c < prob_grid_w:
                            patch_heatmap[r, c] = weight
                            mapped_tokens += 1
                
                # Create visualization if we have data and not composite_only
                if mapped_tokens > 0 and np.max(patch_heatmap) > 0 and not composite_only:
                    # Get required dimensions
                    patch_size = feature_mapping.get("patch_size", 14)
                    
                    patch_path = self._create_patch_feature_overlay(
                        heatmap=patch_heatmap,
                        spatial_preview_image=spatial_preview_image,
                        feature_mapping=feature_mapping,
                        patch_size=patch_size,
                        layer_idx=layer_idx,
                        target_idx=target_idx,
                        title=f"Patch Image Token Influence - Layer {layer_idx}",
                        save_path=os.path.join(patch_heatmap_dir, f"patch_influence_layer_{layer_idx}_{target_idx}.png"),
                        show_values=show_values
                    )
                    if patch_path:
                        patch_heatmap_paths.append(patch_path)
                        patch_valid_layers.append(layer_idx)
                elif mapped_tokens > 0 and np.max(patch_heatmap) > 0:
                    # Just track valid layers for composite
                    patch_valid_layers.append(layer_idx)
        
        # Create composite visualizations if we have multiple layers
        if base_valid_layers:
            try:
                if not composite_only and base_heatmap_paths:
                    # Create composite from individual heatmaps
                    base_composite_path = os.path.join(save_dir, f"composite_base_influence_{target_idx}.png")
                    base_composite = self._create_composite_heatmap(
                        base_heatmap_paths, 
                        base_valid_layers, 
                        base_composite_path,
                        f"Base Image Token Influence Across Layers for Target '{target_text}'"
                    )
                    if base_composite:
                        saved_paths.append(base_composite)
                else:
                    # Create composite directly from trace data
                    base_composite_path = os.path.join(save_dir, f"composite_base_influence_{target_idx}.png")
                    base_composite = self._create_direct_composite_heatmap(
                        trace_data, 
                        original_image,
                        feature_mapping,
                        "base",
                        base_valid_layers,
                        target_text,
                        target_idx,
                        base_composite_path,
                        use_grid_visualization
                    )
                    if base_composite:
                        saved_paths.append(base_composite)
            except Exception as e:
                print(f"Error creating base composite: {e}")
                import traceback
                traceback.print_exc()
        
        if patch_valid_layers:
            try:
                if not composite_only and patch_heatmap_paths:
                    # Create composite from individual heatmaps
                    patch_composite_path = os.path.join(save_dir, f"composite_patch_influence_{target_idx}.png")
                    patch_composite = self._create_composite_heatmap(
                        patch_heatmap_paths, 
                        patch_valid_layers, 
                        patch_composite_path,
                        f"Patch Image Token Influence Across Layers for Target '{target_text}'"
                    )
                    if patch_composite:
                        saved_paths.append(patch_composite)
                else:
                    # Create composite directly from trace data
                    patch_composite_path = os.path.join(save_dir, f"composite_patch_influence_{target_idx}.png")
                    patch_composite = self._create_direct_composite_heatmap(
                        trace_data, 
                        spatial_preview_image,
                        feature_mapping,
                        "patch",
                        patch_valid_layers,
                        target_text,
                        target_idx,
                        patch_composite_path,
                        use_grid_visualization,
                        show_values
                    )
                    if patch_composite:
                        saved_paths.append(patch_composite)
            except Exception as e:
                print(f"Error creating patch composite: {e}")
                import traceback
                traceback.print_exc()
        
        return saved_paths
    
    def _create_direct_composite_heatmap(
        self,
        trace_data: pd.DataFrame,
        image: Image.Image,
        feature_mapping: Dict[str, Any],
        feature_type: str,  # "base" or "patch"
        valid_layers: List[int],
        target_text: str,
        target_idx: int,
        save_path: str,
        use_grid_visualization: bool = True,
        show_values: bool = False
    ) -> Optional[str]:
        """
        Create a composite heatmap directly from trace data without generating individual heatmaps first.
        
        Args:
            trace_data: DataFrame with trace data
            image: Original or spatial preview image
            feature_mapping: Feature mapping information
            feature_type: Type of feature map ("base" or "patch")
            valid_layers: List of valid layer indices
            target_text: Text of the target token
            target_idx: Index of the target token
            save_path: Path to save the composite heatmap
            use_grid_visualization: Whether to use grid-based visualization
            show_values: Whether to show numeric values in cells
            
        Returns:
            Path to saved composite heatmap or None if failed
        """
        if not valid_layers:
            return None
        
        # Get the appropriate feature information
        if feature_type == "base":
            feature_info = feature_mapping.get("base_feature", {})
            grid_key = "grid"
            title = f"Base Image Token Influence Across Layers for Target '{target_text}'"
        else:  # patch
            feature_info = feature_mapping.get("patch_feature", {})
            grid_key = "grid_unpadded"
            title = f"Patch Image Token Influence Across Layers for Target '{target_text}'"
        
        if not feature_info or grid_key not in feature_info or "positions" not in feature_info:
            return None
        
        # Get grid dimensions
        grid_h, grid_w = feature_info[grid_key]
        positions = feature_info["positions"]
        
        # Convert string keys to integers if necessary
        if all(isinstance(k, str) for k in positions.keys()):
            positions = {int(k): v for k, v in positions.items()}
        
        # Create a composite heatmap by averaging across layers
        composite_heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        layer_count = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Process each layer
        for layer_idx in valid_layers:
            # Find all image tokens in this layer
            layer_data = trace_data[trace_data["layer"] == layer_idx]
            image_tokens = layer_data[layer_data["token_type"] == 2]
            
            if image_tokens.empty:
                continue
            
            # Calculate normalized weights
            max_weight = image_tokens["predicted_top_prob"].max()
            if max_weight <= 0:
                continue
            
            # Process each image token
            for _, row in image_tokens.iterrows():
                token_idx = row["token_index"]
                weight = row["predicted_top_prob"] / max_weight  # Normalize
                
                position = positions.get(token_idx)
                if position:
                    r, c = position
                    if 0 <= r < grid_h and 0 <= c < grid_w:
                        composite_heatmap[r, c] += weight
                        layer_count[r, c] += 1
        
        # Average the heatmap
        mask = layer_count > 0
        composite_heatmap[mask] /= layer_count[mask]
        
        # Normalize the composite heatmap
        max_val = np.max(composite_heatmap)
        if max_val > 0:
            composite_heatmap /= max_val
        
        # Create the visualization
        if feature_type == "base":
            result_path = self._create_base_feature_overlay(
                heatmap=composite_heatmap,
                original_image=image,
                grid_size=(grid_h, grid_w),
                layer_idx=-1,  # Composite
                target_idx=target_idx,
                title=title,
                save_path=save_path,
                use_grid_visualization=use_grid_visualization
            )
        else:  # patch
            patch_size = feature_mapping.get("patch_size", 14)
            result_path = self._create_patch_feature_overlay(
                heatmap=composite_heatmap,
                spatial_preview_image=image,
                feature_mapping=feature_mapping,
                patch_size=patch_size,
                layer_idx=-1,  # Composite
                target_idx=target_idx,
                title=title,
                save_path=save_path,
                show_values=show_values
            )
        
        return result_path
    
    def _create_base_feature_overlay(
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
        use_grid_visualization: bool = True
    ) -> Optional[str]:
        """
        Create a heatmap overlay visualization for base image features.
        
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
            
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            # Resize original image for overlay
            resized_background = original_image.resize(target_size, Image.Resampling.LANCZOS)
            background_np = np.array(resized_background)
            
            # Get grid dimensions
            grid_h, grid_w = grid_size
            
            # Calculate cell dimensions
            cell_height = target_size[1] / grid_h
            cell_width = target_size[0] / grid_w
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot background image
            ax.imshow(background_np, extent=(0, target_size[0], target_size[1], 0))
            
            if use_grid_visualization:
                # Grid-based approach for consistency with patch visualization
                # Create a colored grid overlay
                for r in range(grid_h):
                    for c in range(grid_w):
                        # Get heatmap value for this cell (between 0 and 1)
                        cell_value = heatmap[r, c] if r < len(heatmap) and c < len(heatmap[0]) else 0
                        
                        if cell_value > 0:  # Only draw cells with influence
                            # Calculate cell boundaries
                            x_start = c * cell_width
                            y_start = r * cell_height
                            
                            # Create colored rectangle
                            cmap = plt.get_cmap(colormap)
                            cell_color = cmap(cell_value)
                            
                            # Create rectangle with appropriate alpha
                            rect = plt.Rectangle(
                                (x_start, y_start),
                                cell_width, cell_height,
                                color=cell_color,
                                alpha=min(cell_value * 1.5, alpha),  # Scale alpha by value, capped at max alpha
                                linewidth=0
                            )
                            ax.add_patch(rect)
                            
                            # Optionally add text showing value
                            if cell_value > 0.25:  # Only show text for more significant cells
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
                # Original smooth visualization approach
                # Use scikit-image if available, otherwise fall back to simple method
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
                
                # Plot heatmap overlay
                im = ax.imshow(
                    upscaled_heatmap, 
                    alpha=alpha,
                    cmap=colormap,
                    vmin=0,
                    vmax=1,
                    extent=(0, target_size[0], target_size[1], 0),
                    interpolation="nearest"
                )
            
            # Add grid lines regardless of visualization type
            if add_gridlines:
                # Horizontal grid lines
                for i in range(grid_h + 1):
                    y = i * cell_height
                    ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Vertical grid lines
                for i in range(grid_w + 1):
                    x = i * cell_width
                    ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add colorbar that works for both approaches
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Influence Weight")
            
            # Set title and remove axes
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            return save_path
        
        except Exception as e:
            print(f"Error creating base feature overlay: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_patch_feature_overlay(
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
        show_values: bool = True
    ) -> Optional[str]:
        """
        Create a heatmap overlay visualization for patch image features.
        
        Args:
            heatmap: 2D numpy array with heatmap values
            spatial_preview_image: Preprocessed spatial image
            feature_mapping: Feature mapping dictionary
            patch_size: Raw patch size for the vision model
            layer_idx: Index of the layer
            target_idx: Index of the target token
            title: Title for the plot
            save_path: Path to save the visualization
            colormap: Matplotlib colormap name
            alpha: Alpha blending value for overlay
            add_gridlines: Whether to add grid lines
            show_values: Whether to show numeric values in cells
            
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            # Get patch feature information
            patch_feature_info = feature_mapping.get("patch_feature", {})
            if not patch_feature_info:
                print("Error: No patch feature information available.")
                return None
            
            prob_grid_h, prob_grid_w = patch_feature_info.get("grid_unpadded", (0, 0))
            if prob_grid_h == 0 or prob_grid_w == 0:
                print("Error: Invalid grid dimensions.")
                return None
            
            # Get dimensions for the visualization
            preview_w, preview_h = spatial_preview_image.size
            background_np = np.array(spatial_preview_image)
            
            # Get the actual content dimensions and padding
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
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(8, 8 * preview_h / max(1, preview_w)))
            
            # Plot background image
            ax.imshow(background_np, extent=(0, preview_w, preview_h, 0))
            
            # Create colormap
            cmap = plt.get_cmap(colormap)
            
            # Create grid-based overlay
            for r in range(prob_grid_h):
                for c in range(prob_grid_w):
                    # Get heatmap value for this cell (between 0 and 1)
                    cell_value = heatmap[r, c] if r < len(heatmap) and c < len(heatmap[0]) else 0
                    
                    if cell_value > 0:  # Only draw cells with influence
                        # Calculate cell boundaries in image coordinates
                        x_start = pad_left + c * cell_width
                        y_start = pad_top + r * cell_height
                        
                        # Get color from colormap
                        cell_color = cmap(cell_value)
                        
                        # Create rectangle with appropriate alpha
                        rect = plt.Rectangle(
                            (x_start, y_start),
                            cell_width, cell_height,
                            color=cell_color,
                            alpha=min(cell_value * 1.5, alpha),  # Scale alpha by value, capped at max alpha
                            linewidth=0
                        )
                        ax.add_patch(rect)
                        
                        # Optionally add text showing value
                        if show_values and cell_value > 0.25:  # Only show text for more significant cells
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
            
            # Add grid lines if requested
            if add_gridlines:
                # Horizontal grid lines
                for i in range(prob_grid_h + 1):
                    y = pad_top + i * cell_height
                    ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Vertical grid lines
                for i in range(prob_grid_w + 1):
                    x = pad_left + i * cell_width
                    ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add colorbar
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Influence Weight")
            
            # Set title and remove axes
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            return save_path
        
        except Exception as e:
            print(f"Error creating patch feature overlay: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_composite_heatmap(
        self,
        image_paths: List[str],
        layers: List[int],
        output_filename: str,
        title: str,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Optional[str]:
        """
        Creates a composite image grid from individual heatmap visualizations.
        
        Args:
            image_paths: List of file paths to the individual images
            layers: List of layer indices corresponding to each image
            output_filename: Path to save the composite image
            title: Title for the composite image
            background_color: RGB background color
            
        Returns:
            Path to saved composite image or None if failed
        """
        try:
            # Internal padding constants
            padding = 10
            label_padding = 25
            
            if not image_paths:
                print(f"Error: No image paths provided for composite.")
                return None
            
            if len(image_paths) != len(layers):
                print(f"Warning: Mismatch between image paths ({len(image_paths)}) and layers ({len(layers)}).")
                # Match paths and layers based on filename
                path_layer_map = {}
                for p in image_paths:
                    try:
                        # More robust layer extraction from filename
                        filename = os.path.basename(p)
                        parts = filename.split('_')
                        layer_num = None
                        
                        # Look for "layer_X" pattern in filename
                        for i, part in enumerate(parts):
                            if part == "layer" and i+1 < len(parts) and parts[i+1].isdigit():
                                layer_num = int(parts[i+1])
                                break
                        
                        # Fallback: Try other patterns commonly found in filenames
                        if layer_num is None:
                            for part in parts:
                                if part.isdigit() and 0 <= int(part) < 100:  # Reasonable layer range
                                    layer_num = int(part)
                                    break
                        
                        if layer_num is not None:
                            path_layer_map[layer_num] = p
                        else:
                            print(f"Could not extract layer number from filename: {filename}")
                    except (IndexError, ValueError) as e:
                        print(f"Could not extract layer number from filename: {os.path.basename(p)}: {e}")
                    
                # Rebuild lists based on layers
                matched_paths = [path_layer_map.get(l) for l in layers]
                filtered_layers = [l for l, p in zip(layers, matched_paths) if p is not None]
                filtered_paths = [p for p in matched_paths if p is not None]
                
                if not filtered_paths:
                    print(f"Error: No images could be matched to layers.")
                    return None
                
                image_paths = filtered_paths
                layers = filtered_layers
            
            # Load first image to get dimensions
            with Image.open(image_paths[0]) as img:
                img_w, img_h = img.size
                img_mode = img.mode
            
            # Calculate grid dimensions
            num_images = len(image_paths)
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)
            
            # Calculate canvas dimensions
            cell_w = img_w + padding
            cell_h = img_h + padding + label_padding
            title_height = 50
            canvas_w = cols * cell_w + padding
            canvas_h = rows * cell_h + padding + title_height
            
            # Create canvas and draw object
            canvas = Image.new(img_mode, (canvas_w, canvas_h), background_color)
            draw = ImageDraw.Draw(canvas)
            
            # Define fonts
            try:
                DEFAULT_FONT = ImageFont.truetype("arial.ttf", 18)
                DEFAULT_FONT_SMALL = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                DEFAULT_FONT = ImageFont.load_default()
                DEFAULT_FONT_SMALL = ImageFont.load_default()
            
            # Add title
            try:
                title_bbox = draw.textbbox((0, 0), title, font=DEFAULT_FONT)
                title_w = title_bbox[2] - title_bbox[0]
                title_h = title_bbox[3] - title_bbox[1]
            except AttributeError:
                title_w, title_h = draw.textlength(title, font=DEFAULT_FONT), 20
            
            title_x = (canvas_w - title_w) // 2
            title_y = padding
            draw.text((title_x, title_y), title, fill=(0, 0, 0), font=DEFAULT_FONT)
            
            # Paste images and add labels
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
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                
                current_col += 1
                if current_col >= cols:
                    current_col = 0
                    current_row += 1
            
            # Save the composite image
            canvas.save(output_filename)
            print(f"Saved composite image to: {output_filename}")
            return output_filename
        
        except Exception as e:
            print(f"Error creating composite image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_trace_data_visualizations(
        self,
        trace_data: pd.DataFrame,
        target_text: str,
        target_idx: int,
        save_dir: str
    ) -> List[str]:
        """Visualize trace data with various plots"""
        saved_paths = []
        
        if trace_data.empty:
            print("No trace data available for visualizations.")
            return saved_paths
                
        try:
            # Create directory for trace visualizations
            trace_vis_dir = os.path.join(save_dir, "trace_data_plots")
            os.makedirs(trace_vis_dir, exist_ok=True)
            
            # 1. Top predicted tokens by layer
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Focus on unique token positions
            token_positions = trace_data["token_index"].unique()
            
            # Create color map for token positions
            cmap = plt.cm.get_cmap('tab20', len(token_positions))
            color_map = {pos: cmap(i) for i, pos in enumerate(token_positions)}
            
            # Group by layer and get top predicted tokens
            layers = sorted(trace_data["layer"].unique())
            
            # For each unique token position, plot its top prediction probability across layers
            for pos in token_positions:
                token_df = trace_data[trace_data["token_index"] == pos]
                if len(token_df) >= len(layers) * 0.5:  # Only include if present in at least half the layers
                    token_text = token_df["token_text"].iloc[0]
                    layer_probs = []
                    
                    for layer in layers:
                        layer_row = token_df[token_df["layer"] == layer]
                        if not layer_row.empty:
                            prob = layer_row["predicted_top_prob"].iloc[0]
                            layer_probs.append(prob)
                        else:
                            layer_probs.append(None)
                    
                    # Plot with positions that have data
                    valid_indices = [i for i, p in enumerate(layer_probs) if p is not None]
                    valid_layers = [layers[i] for i in valid_indices]
                    valid_probs = [layer_probs[i] for i in valid_indices]
                    
                    if valid_probs:
                        ax.plot(valid_layers, valid_probs, marker='o', label=f"{pos}: '{token_text}'", 
                                color=color_map[pos], linewidth=2, alpha=0.8)
            
            ax.set_xlabel("Layer")
            ax.set_ylabel("Top Prediction Probability")
            ax.set_title(f"Top Token Prediction Confidence by Layer for Target '{target_text}'")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(layers)
            
            # Add legend with reasonable size
            if len(token_positions) > 15:
                # Too many tokens, use a compact legend
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
            else:
                ax.legend(loc='best')
                
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(trace_vis_dir, f"top_predictions_by_layer_{target_idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_paths.append(save_path)
            
            plt.close(fig)
            
            # 2. Create heatmap of all concept probabilities for key tokens
            concept_cols = [col for col in trace_data.columns if col.startswith("concept_") and col.endswith("_prob")]
            if concept_cols:
                token_probs = trace_data.groupby('token_index').agg({
                    'token_text': 'first',
                    'token_type': 'first',
                    **{col: 'max' for col in concept_cols}
                }).reset_index()
                
                # Select tokens with at least one significant concept probability
                token_probs['max_concept_prob'] = token_probs[concept_cols].max(axis=1)
                significant_tokens = token_probs[token_probs['max_concept_prob'] > 0.05]
                
                if len(significant_tokens) > 0:
                    # Create a heatmap
                    plt.figure(figsize=(12, max(6, len(significant_tokens) * 0.4)))
                    
                    # Prepare data for heatmap
                    heatmap_data = significant_tokens.set_index('token_text')[concept_cols]
                    # Rename columns to just concept names
                    heatmap_data.columns = [col.replace("concept_", "").replace("_prob", "") for col in concept_cols]
                    
                    # Create heatmap
                    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Probability'})
                    plt.title(f'Maximum Concept Probabilities for Key Tokens - Target {target_idx}: "{target_text}"')
                    plt.tight_layout()
                    
                    # Save figure
                    save_path = os.path.join(trace_vis_dir, f"concept_heatmap_{target_idx}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    saved_paths.append(save_path)
                    
                    plt.close()
            
            return saved_paths
        
        except Exception as e:
            print(f"Error creating trace data visualizations: {e}")
            import traceback
            traceback.print_exc()
            return saved_paths
    
    def _sanitize_text_for_display(self, text):
        """
        Sanitize text to avoid font rendering issues with special characters.
        
        Args:
            text: Input text that may contain special characters (can be string or any other type)
            
        Returns:
            Sanitized text that should render properly in matplotlib
        """
        # Convert to string if not already a string
        if not isinstance(text, str):
            text = str(text)
            
        if not text:
            return ""
            
        # Special handling for common tokens
        if text in ["<s>", "<pad>", "<bos>", "<eos>", "<image>"]:
            return text
            
        # Replace common problematic characters
        replacements = {
            # Replace various quotes
            '"': '"', '"': '"', ''': "'", ''': "'",
            # Replace emoji and special symbols with simple alternatives
            '→': '->',
            '←': '<-',
            '⟨': '<',
            '⟩': '>',
            '⇒': '=>',
            '⇐': '<=',
            '≤': '<=',
            '≥': '>=',
            '…': '...',
        }
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Filter out other potentially problematic characters
        result = ""
        for char in text:
            # Keep ASCII characters and common symbols
            if ord(char) < 128 or char in '°±²³½¼¾×÷':
                result += char
            else:
                # Replace other non-ASCII characters with a placeholder
                result += '·'  # Middle dot as placeholder
        
        return result