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
    
    def visualize_from_csv(
        self,
        csv_path: str,
        metadata_path: Optional[str] = None,
        image_path: Optional[str] = None,
        target_token: Optional[Dict[str, Any]] = None,
        create_flow_graph: bool = True,
        create_heatmaps: bool = True,
        create_data_vis: bool = True,
        flow_graph_params: Optional[Dict[str, Any]] = None,
        heatmap_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[str]]:
        """
        Generate visualizations from a saved CSV trace file.
        
        Args:
            csv_path: Path to the CSV file with trace data
            metadata_path: Optional path to JSON metadata file
            image_path: Path or URL to original image used in the trace
            target_token: Optional information about the target token
            create_flow_graph: Whether to create flow graph visualization
            create_heatmaps: Whether to create heatmap visualizations
            create_data_vis: Whether to create trace data visualizations
            flow_graph_params: Parameters for flow graph visualization
            heatmap_params: Parameters for heatmap visualization
            
        Returns:
            Dictionary mapping visualization types to lists of generated file paths
        """
        print(f"Generating visualizations from CSV: {csv_path}")
        results = {
            "flow_graph": [],
            "heatmap": [],
            "data_vis": []
        }
        
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
            return results
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded trace data with {len(df)} rows")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return results
        
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
        
        # 1. Create flow graph visualization if requested
        if create_flow_graph:
            try:
                print("Creating interactive token flow graph visualization...")
                flow_graph_paths = self.create_interactive_flow_graph_from_csv(
                    df, target_text, target_idx, save_dir, **flow_graph_params
                )
                results["flow_graph"] = flow_graph_paths
            except Exception as e:
                print(f"Error creating flow graph visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # 2. Create heatmap visualizations if requested and if we have an image
        if create_heatmaps and image_path:
            try:
                # Get local image path (download if needed)
                local_image_path = self.get_local_image_path(image_path, save_dir)
                
                if local_image_path and os.path.exists(local_image_path):
                    print(f"Creating heatmap visualizations with image: {local_image_path}")
                    
                    # Check if we have feature mapping
                    feature_mapping_exists = "feature_mapping" in metadata and metadata["feature_mapping"]
                    
                    if feature_mapping_exists:
                        heatmap_paths = self.create_heatmaps_from_csv(
                            df, target_text, target_idx, local_image_path,
                            save_dir, metadata["feature_mapping"], **heatmap_params
                        )
                        results["heatmap"] = heatmap_paths
                    else:
                        print("Skipping heatmap visualizations: missing feature mapping")
                else:
                    print(f"Skipping heatmap visualizations: image not available")
            except Exception as e:
                print(f"Error creating heatmap visualizations: {e}")
                import traceback
                traceback.print_exc()
        
        # 3. Create trace data visualizations if requested
        if create_data_vis:
            try:
                print("Creating trace data visualizations...")
                data_vis_paths = self.create_trace_data_visualizations_from_csv(
                    df, target_text, target_idx, save_dir
                )
                results["data_vis"] = data_vis_paths
            except Exception as e:
                print(f"Error creating trace data visualizations: {e}")
                import traceback
                traceback.print_exc()
        
        # Count total visualizations
        total_files = sum(len(files) for files in results.values())
        print(f"Visualization complete. Generated {total_files} files.")
        return results
    
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
    
    #
    # INTERACTIVE FLOW GRAPH VISUALIZATION
    #
    def create_interactive_flow_graph_from_csv(
        self,
        trace_data: pd.DataFrame,
        target_text: str,
        target_idx: int,
        save_dir: str,
        min_edge_weight: float = 0.05,
        use_variable_node_size: bool = True,
        min_node_size: int = 10,
        max_node_size: int = 30,
        debug_mode: bool = False,
        use_exponential_scaling: bool = True,
        output_format: str = "both",
        show_orphaned_nodes: bool = False,
        horizontal_spacing: float = 1.0,
        vertical_spacing: float = 0.5,
        **kwargs
    ) -> List[str]:
        """
        Creates an interactive flow graph visualization from trace CSV data using Plotly.
        
        This function builds a layered visualization with:
        - One column per model layer
        - Tokens as nodes colored by type (image=red, text=blue, generated=green)
        - Connections between source and target tokens across layers
        - Interactive features:
            - Click nodes to highlight connected paths
            - Hover over nodes for token details
            - Zoom, pan, and export options
        
        Args:
            trace_data: DataFrame of token trace records
            target_text: Text of the target token
            target_idx: Index of the target token
            save_dir: Directory to save visualization files
            min_edge_weight: Minimum edge weight to include
            use_variable_node_size: Whether to vary node size based on weight
            min_node_size: Minimum node size for visualization
            max_node_size: Maximum node size for visualization
            debug_mode: Whether to print debug information
            use_exponential_scaling: Whether to use exponential scaling for node sizes
            output_format: Output format (png, svg, or both)
            show_orphaned_nodes: Whether to show nodes with no connections
            horizontal_spacing: Multiplier for horizontal spacing between layers
            vertical_spacing: Multiplier for vertical spacing between nodes
            **kwargs: Additional arguments for future extensions
            
        Returns:
            List of paths to saved visualization files
        """
        import plotly.graph_objects as go
        import networkx as nx
        import math
        
        saved_paths = []
        if trace_data.empty:
            print("No data for interactive flow graph.")
            return saved_paths

        # Extract all unique layers from the data
        layers = sorted(trace_data["layer"].unique())
        if debug_mode:
            print(f"Detected {len(layers)} layers: {layers}")

        # Build directed graph
        G = nx.DiGraph()
        
        # Map from (layer, token_index) to node_id and store metadata
        node_map = {}
        node_metadata = {}
        
        # Track tokens by layer for positioning
        layer_tokens = {layer_idx: [] for layer_idx in layers}
        
        # STEP 1: Build node mapping and metadata
        for _, row in trace_data.iterrows():
            layer_idx = row["layer"]
            token_idx = row["token_index"]
            node_id = f"L{layer_idx}_T{token_idx}"
            node_map[(layer_idx, token_idx)] = node_id

            # Add token to its layer group for positioning
            if token_idx not in layer_tokens[layer_idx]:
                layer_tokens[layer_idx].append(token_idx)

            # Sanitize display text and predicted top token
            txt = self._sanitize_text_for_display(row["token_text"])
            pred = row.get("predicted_top_token", "")
            if pd.notna(pred) and pred:
                pred = f"→{self._sanitize_text_for_display(pred)}"

            # Store node metadata
            node_metadata[node_id] = {
                "type": row["token_type"],
                "text": txt,
                "top_pred": pred,
                "weight": row.get("predicted_top_prob", 1.0),
                "layer": layer_idx,
                "is_target": row["is_target"],
                "token_idx": token_idx  # Store original token index for highlighting
            }
            G.add_node(node_id)

        # STEP 2: Build edges based on source-target relationships
        for _, row in trace_data.iterrows():
            if not row["is_target"] or pd.isna(row.get("sources_indices", None)):
                continue

            target_node = node_map[(row["layer"], row["token_index"])]
            
            # Parse source indices and weights from the CSV
            try:
                src_indices = [int(i) for i in str(row["sources_indices"]).split(",") if i]
                src_weights = [float(w) for w in str(row["sources_weights"]).split(",") if w]
            except (ValueError, AttributeError):
                continue
                
            for src_idx, weight in zip(src_indices, src_weights):
                if weight < min_edge_weight:
                    continue
                    
                # Find the latest previous layer that contains this source token
                for prev_layer in reversed([l for l in layers if l < row["layer"]]):
                    key = (prev_layer, src_idx)
                    if key in node_map:
                        # Add edge with weight attribute
                        G.add_edge(node_map[key], target_node, weight=weight)
                        break

        # STEP 3: Remove orphaned nodes if requested
        if not show_orphaned_nodes:
            orphans = [n for n in G.nodes() if G.degree(n) == 0]
            G.remove_nodes_from(orphans)
            if debug_mode:
                print(f"Removed {len(orphans)} orphaned nodes")
        
        # If graph is empty after filtering, return
        if len(G) == 0:
            print("Graph empty after filtering.")
            return saved_paths

        # STEP 4: Compute improved node positions with better spacing
        # This is a critical part that needs improvement
        pos = {}
        
        # Calculate total width and height of the graph area
        # Use wider spacing to prevent crowding
        graph_width = len(layers) * horizontal_spacing * 4.0  # Increase horizontal space
        
        # Count maximum nodes in any layer for height calculation
        max_nodes_per_layer = max(len(nodes) for nodes in layer_tokens.values())
        graph_height = max_nodes_per_layer * vertical_spacing * 2.0  # Increase vertical space
        
        # IMPORTANT: Position nodes in a grid layout by layer with better spacing
        for layer_idx in layers:
            # Get tokens in this layer that are in the graph
            layer_nodes_ids = []
            for token_idx in layer_tokens[layer_idx]:
                node_id = node_map.get((layer_idx, token_idx))
                if node_id in G.nodes():
                    layer_nodes_ids.append(node_id)
            
            # Skip empty layers
            if not layer_nodes_ids:
                continue
            
            # Calculate x-coordinate for this layer
            layer_idx_norm = layers.index(layer_idx)  # Normalized layer index (0 to len(layers)-1)
            x_pos = layer_idx_norm * graph_width / max(1, len(layers) - 1)
            
            # Sort nodes by token index for consistent ordering
            layer_nodes_ids.sort(key=lambda n: node_metadata[n]["token_idx"])
            
            # Distribute nodes vertically with proper spacing
            num_nodes = len(layer_nodes_ids)
            if num_nodes > 0:
                # Use the full height for better distribution
                total_height = graph_height
                
                # Calculate step size and starting y position
                y_step = total_height / max(1, num_nodes - 1) if num_nodes > 1 else 0
                y_start = -total_height / 2 if num_nodes > 1 else 0
                
                # Position each node
                for i, node_id in enumerate(layer_nodes_ids):
                    y_pos = y_start + i * y_step
                    pos[node_id] = (x_pos, y_pos)
                    if debug_mode and i == 0:
                        print(f"Layer {layer_idx}: positioned at x={x_pos:.1f}, y from {y_start:.1f}")

        # STEP 5: Compute node sizes based on weights
        sizes = self._calculate_flow_graph_node_sizes(
            G, node_metadata, use_variable_node_size,
            min_node_size, max_node_size, use_exponential_scaling
        )
        
        # STEP 6: Prepare node and edge traces for Plotly
        # Define color map for token types
        color_map = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}  # green, blue, red
        token_type_labels = {0: "Generated", 1: "Text", 2: "Image"}
        
        # Create node traces (one per token type for legend grouping)
        node_traces = {}
        for token_type in [0, 1, 2]:  # Generated, Text, Image
            node_traces[token_type] = go.Scatter(
                x=[],
                y=[],
                mode='markers',
                name=token_type_labels[token_type],
                marker=dict(
                    color=color_map[token_type],
                    size=[],
                    line=dict(width=1, color='black')
                ),
                hoverinfo='text',
                hovertext=[],
                ids=[],  # Store node IDs for callbacks
                customdata=[]  # Store token indices for path highlighting
            )
        
        # Populate node traces
        for node in G.nodes():
            x, y = pos[node]
            token_type = node_metadata[node]["type"]
            token_text = node_metadata[node]["text"]
            top_pred = node_metadata[node]["top_pred"]
            layer = node_metadata[node]["layer"]
            token_idx = node_metadata[node]["token_idx"]
            is_target = node_metadata[node]["is_target"]
            
            # Add node to appropriate trace
            node_traces[token_type].x = list(node_traces[token_type].x) + [x]
            node_traces[token_type].y = list(node_traces[token_type].y) + [y]
            
            # Size
            idx = list(G.nodes()).index(node)
            node_traces[token_type].marker.size = list(node_traces[token_type].marker.size) + [sizes[idx]]
            
            # Hover text
            text_label = f"{token_text} {top_pred}" if top_pred else token_text
            hover_text = f"Layer: {layer}<br>Token: '{text_label}'<br>Index: {token_idx}<br>Type: {token_type_labels[token_type]}"
            if is_target:
                hover_text += "<br>Is Target: Yes"
                
            node_traces[token_type].hovertext = list(node_traces[token_type].hovertext) + [hover_text]
            node_traces[token_type].ids = list(node_traces[token_type].ids) + [node]
            
            # Store token index in customdata for path highlighting
            node_traces[token_type].customdata = list(node_traces[token_type].customdata) + [[token_idx, layer]]
        
        # Create edge traces with proper curvature and width scaling
        edge_traces = []
        edge_data = {}  # Store all edge data for highlighting
        
        for u, v, data in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            weight = data.get('weight', 0.0)
            
            # Apply exponential scaling to width if requested
            if use_exponential_scaling:
                width = math.sqrt(weight) * 3  # Adjust multiplier as needed
            else:
                width = weight * 3
            
            # Make sure width is at least 0.5 for visibility, but not more than 10
            width = max(0.5, min(10, width))
            
            # Create curved edge by adding midpoint with offset
            xmid = (x0 + x1) / 2
            ymid = (y0 + y1) / 2
            # Add slight curvature - increase curve factor for more pronounced curves
            curve_factor = 0.2
            xmid += curve_factor * (y1 - y0)
            ymid -= curve_factor * (x1 - x0)
            
            # Hover text for this edge
            u_text = node_metadata[u]["text"]
            v_text = node_metadata[v]["text"]
            u_layer = node_metadata[u]["layer"]
            v_layer = node_metadata[v]["layer"]
            edge_text = f"Source: '{u_text}' (L{u_layer})<br>Target: '{v_text}' (L{v_layer})<br>Weight: {weight:.3f}"
            
            # Store source-target token indices for path highlighting
            source_idx = node_metadata[u]["token_idx"]
            target_idx = node_metadata[v]["token_idx"]
            
            # Edge alpha scaled by weight, but never below 0.2 for visibility
            alpha = min(0.9, max(0.2, weight))
            
            # Create edge trace for this specific edge
            edge_trace = go.Scatter(
                x=[x0, xmid, x1, None],
                y=[y0, ymid, y1, None],
                mode='lines',
                line=dict(
                    width=width,  # Single width for this edge
                    color=f'rgba(136, 136, 136, {alpha})'
                ),
                hoverinfo='text',
                hovertext=[edge_text, edge_text, edge_text, None],
                showlegend=False,
                customdata=[[source_idx, target_idx], [source_idx, target_idx], [source_idx, target_idx], None]
            )
            
            edge_traces.append(edge_trace)
            
            # Store edge data for highlighting
            edge_key = f"{u}_{v}"
            edge_data[edge_key] = {
                "source_idx": source_idx,
                "target_idx": target_idx,
                "weight": weight,
                "trace_index": len(edge_traces) - 1
            }

        # STEP 7: Combine all traces and create the figure
        # Edge traces first, then node traces
        data = edge_traces + list(node_traces.values())
        
        # Create figure with proper size
        fig = go.Figure(data=data)
        
        # STEP 8: Configure layout with proper layer labels and spacing
        title = f"Interactive Semantic Trace Flow Graph for Token '{target_text}' (idx: {target_idx})"
        
        # Add layer labels as vertical lines and text annotations
        shapes = []
        annotations = []
        
        # Extract unique layers actually present in the graph
        graph_layers = sorted(set(node_metadata[n]["layer"] for n in G.nodes()))
        
        # Map from actual layer index to normalized index for positioning
        layer_to_norm_idx = {layer: i for i, layer in enumerate(graph_layers)}
        
        # For each layer, add a vertical line and label
        for layer_idx in graph_layers:
            # Get normalized x-position 
            norm_idx = layer_to_norm_idx[layer_idx]
            x_pos = norm_idx * graph_width / max(1, len(graph_layers) - 1)
            
            # Add vertical line (not for the first layer)
            if norm_idx > 0:
                shapes.append(dict(
                    type="line",
                    x0=x_pos - horizontal_spacing,  # Adjust for better spacing
                    y0=-graph_height/2 - 1,  # Extend below the bottom-most node
                    x1=x_pos - horizontal_spacing,
                    y1=graph_height/2 + 1,   # Extend above the top-most node
                    line=dict(
                        color="rgba(150, 150, 150, 0.4)",
                        width=1,
                        dash="dash"
                    )
                ))
            
            # Add layer label
            annotations.append(dict(
                x=x_pos,
                y=graph_height/2 + 1.5,  # Position above the top-most node
                xref="x",
                yref="y",
                text=f"Layer {layer_idx}",
                showarrow=False,
                font=dict(size=14, color="black"),
            ))
        
        # Configure the figure layout
        fig.update_layout(
            title=title,
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            # Explicitly set plot area size to avoid overcrowding
            width=max(800, len(graph_layers) * 100),  # Scale width by number of layers
            height=max(600, max_nodes_per_layer * 50), # Scale height by max nodes
            margin=dict(b=100, l=50, r=50, t=100),
            annotations=annotations,
            shapes=shapes,
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-1, graph_width + 1]  # Add padding around the graph
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="x",  # Keep aspect ratio consistent
                scaleratio=1,
                range=[-graph_height/2 - 1, graph_height/2 + 3]  # Add padding
            ),
            legend=dict(
                title="Token Types",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                font=dict(size=12)
            ),
            # Add custom buttons for highlighting and resetting
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            args=[{"visible": [True] * len(data)}],
                            label="Reset View",
                            method="update"
                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=False,
                    x=0.1,
                    xanchor="left",
                    y=0,
                    yanchor="top"
                )
            ]
        )

        # STEP 9: Add JavaScript for interactivity
        # Add annotation for user guidance
        fig.update_layout(
            clickmode='event',
            annotations=[
                *fig.layout.annotations,
                dict(
                    x=0.5,
                    y=-0.1,
                    xref="paper",
                    yref="paper",
                    text="Click on any node to highlight its path. Double-click to reset.",
                    showarrow=False,
                    font=dict(size=12)
                )
            ]
        )

        # JavaScript for handling node clicking and highlighting
        onclick_js = """
        // This JavaScript handles interactive node highlighting
        const gd = document.getElementById('{plot_id}');
        
        // Store original trace data to enable resetting
        let originalOpacity = {};
        let originalWidth = {};
        let originalColor = {};
        let originalSize = {};
        let firstClick = true;
        let numEdgeTraces = 0;  // Will be set on first click
        
        gd.on('plotly_click', function(data) {
            // Get the clicked point
            const point = data.points[0];
            
            // Initialize storage on first click
            if (firstClick) {
                // Count edge traces (all traces before the node traces, which are the last 3)
                numEdgeTraces = gd.data.length - 3;
                
                // Store original values
                // For edges (each edge is a separate trace)
                for (let i = 0; i < numEdgeTraces; i++) {
                    let edgeTrace = gd.data[i];
                    originalOpacity[i] = edgeTrace.line.color;
                    originalWidth[i] = edgeTrace.line.width;
                }
                
                // For nodes
                for (let i = numEdgeTraces; i < gd.data.length; i++) {
                    originalSize[i] = [...gd.data[i].marker.size];
                    originalColor[i] = gd.data[i].marker.color;
                }
                firstClick = false;
            }
            
            // Reset to original state
            resetHighlighting();
            
            // If we clicked a node (not an edge), highlight its path
            if (point.customdata && point.curveNumber >= numEdgeTraces) {
                const clickedTokenIdx = point.customdata[0];
                const clickedLayer = point.customdata[1];
                
                // Fade all nodes and edges
                fadeAllElements();
                
                // Find connected paths and highlight them
                highlightNodePath(clickedTokenIdx, clickedLayer);
            }
        });
        
        gd.on('plotly_doubleclick', function() {
            // Reset all highlighting on double-click
            resetHighlighting();
        });
        
        function fadeAllElements() {
            // Fade edges - each edge is a separate trace
            for (let i = 0; i < numEdgeTraces; i++) {
                let edgeTrace = gd.data[i];
                // Get original color and make it transparent
                let origColor = originalOpacity[i];
                let fadeColor = origColor;
                if (origColor.startsWith('rgba')) {
                    // Parse rgba and change alpha
                    let colorParts = origColor.replace('rgba(', '').replace(')', '').split(',');
                    fadeColor = `rgba(${colorParts[0]},${colorParts[1]},${colorParts[2]},0.15)`;
                } else {
                    // Default color with low alpha
                    fadeColor = 'rgba(136,136,136,0.15)';
                }
                
                Plotly.restyle(gd, {
                    'line.color': fadeColor,
                    'line.width': Math.min(1, originalWidth[i]/3)
                }, [i]);
            }
            
            // Fade nodes
            for (let i = numEdgeTraces; i < gd.data.length; i++) {
                Plotly.restyle(gd, {
                    'marker.opacity': 0.15,
                    'marker.line.width': 0.5
                }, [i]);
            }
        }
        
        function highlightNodePath(tokenIdx, layerIdx) {
            // Highlight this specific node
            for (let i = numEdgeTraces; i < gd.data.length; i++) {
                const trace = gd.data[i];
                const highlightIndices = [];
                
                // Find all points in this trace matching our token
                trace.customdata.forEach((data, idx) => {
                    if (data && data[0] === tokenIdx && data[1] === layerIdx) {
                        highlightIndices.push(idx);
                    }
                });
                
                if (highlightIndices.length > 0) {
                    // Create arrays of the same length as the trace points
                    const opacity = Array(trace.x.length).fill(0.15);
                    const lineWidth = Array(trace.x.length).fill(0.5);
                    const sizes = [...originalSize[i]];
                    
                    // Update for highlighted points
                    highlightIndices.forEach(idx => {
                        opacity[idx] = 1;
                        lineWidth[idx] = 2;
                        sizes[idx] = sizes[idx] * 1.3;  // Make highlighted node larger
                    });
                    
                    // Apply the updates
                    Plotly.restyle(gd, {
                        'marker.opacity': opacity,
                        'marker.line.width': lineWidth,
                        'marker.size': sizes
                    }, [i]);
                }
            }
            
            // Highlight edges connected to this node
            // Each edge is a separate trace
            for (let i = 0; i < numEdgeTraces; i++) {
                const edgeTrace = gd.data[i];
                const edgeData = edgeTrace.customdata[0];  // First point has the data
                
                if (!edgeData) continue;
                
                // Check if this edge connects to our token
                if (edgeData[0] === tokenIdx || edgeData[1] === tokenIdx) {
                    // Highlight this edge
                    let origColor = originalOpacity[i];
                    let highlightColor = origColor;
                    
                    if (origColor.startsWith('rgba')) {
                        // Parse rgba and increase alpha
                        let colorParts = origColor.replace('rgba(', '').replace(')', '').split(',');
                        highlightColor = `rgba(${colorParts[0]},${colorParts[1]},${colorParts[2]},0.9)`;
                    } else {
                        // Default highlight color
                        highlightColor = 'rgba(136,136,136,0.9)';
                    }
                    
                    Plotly.restyle(gd, {
                        'line.color': highlightColor,
                        'line.width': originalWidth[i] * 1.5  // Make highlighted edge thicker
                    }, [i]);
                }
            }
        }
        
        function resetHighlighting() {
            // Only proceed if we've stored original values
            if (firstClick) return;
            
            // Reset edges
            for (let i = 0; i < numEdgeTraces; i++) {
                Plotly.restyle(gd, {
                    'line.color': originalOpacity[i],
                    'line.width': originalWidth[i]
                }, [i]);
            }
            
            // Reset nodes
            for (let i = numEdgeTraces; i < gd.data.length; i++) {
                Plotly.restyle(gd, {
                    'marker.opacity': 1,
                    'marker.line.width': 1,
                    'marker.size': originalSize[i],
                    'marker.color': originalColor[i]
                }, [i]);
            }
        }
        """
        
        # STEP 10: Save visualizations
        # Create output directories
        html_dir = os.path.join(save_dir, "interactive")
        os.makedirs(html_dir, exist_ok=True)
        
        # Save interactive HTML figure
        html_path = os.path.join(html_dir, f"interactive_flow_graph_{target_idx}.html")
        fig.write_html(
            html_path, 
            include_plotlyjs='cdn',
            post_script=onclick_js.replace('{plot_id}', 'graph')
        )
        saved_paths.append(html_path)
        
        # Try to save static images
        try:
            if output_format in ["png", "both"]:
                png_path = os.path.join(save_dir, f"flow_graph_{target_idx}.png")
                try:
                    # Try to use kaleido if available
                    fig.write_image(png_path, width=1200, height=800, scale=2)
                    saved_paths.append(png_path)
                except Exception as e:
                    # Fallback to alternative method - save a screenshot of div
                    print(f"PNG export error (kaleido may be missing): {e}")
                    print(f"Tip: Install kaleido package with 'pip install -U kaleido' for better image export")
                    
                    # Try plotly.io export with 'svg' engine as fallback
                    try:
                        import plotly.io as pio
                        pio.write_image(fig, png_path, format='png', engine='svg')
                        saved_paths.append(png_path)
                    except:
                        print("Could not save PNG, using HTML only")
                    
            if output_format in ["svg", "both"]:
                svg_path = os.path.join(save_dir, f"flow_graph_{target_idx}.svg")
                try:
                    # Try to use kaleido if available
                    fig.write_image(svg_path, format="svg", width=1200, height=800)
                    saved_paths.append(svg_path)
                except Exception as e:
                    # Fallback to alternative method
                    print(f"SVG export error (kaleido may be missing): {e}")
                    print(f"Tip: Install kaleido package with 'pip install -U kaleido' for better image export")
                    
                    # Try plotly.io export with 'svg' engine as fallback
                    try:
                        import plotly.io as pio
                        pio.write_image(fig, svg_path, format='svg', engine='svg')
                        saved_paths.append(svg_path)
                    except:
                        print("Could not save SVG, using HTML only")
        except Exception as export_error:
            print(f"Error exporting static images: {export_error}")
            print("Interactive HTML visualization still available")
        
        print(f"Created interactive flow graph: {html_path}")
        if len(saved_paths) > 1:
            print(f"Created {len(saved_paths) - 1} static image files for reference")
        
        return saved_paths


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
    
    def _sanitize_text_for_display(self, text):
        """
        Sanitize text to avoid rendering issues with special characters.
        
        Args:
            text: Input text that may contain special characters (can be string or any other type)
            
        Returns:
            Sanitized text that should render properly in plotly/html
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
    
    #
    # STANDALONE HEATMAP VISUALIZATION
    #
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
        composite_only: bool = False
    ) -> List[str]:
        """
        Create per-layer and grid-composite heatmaps for base and patch features.
        
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
        Returns:
            list of saved file paths
        """
        saved_paths = []

        # Create output directories
        base_dir = os.path.join(save_dir, "base_heatmaps")
        patch_dir = os.path.join(save_dir, "patch_heatmaps")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(patch_dir, exist_ok=True)

        # Load the original image
        try:
            img = Image.open(image_path)
            # Create a preview version for patch visualization
            preview = img.copy()
            if "resized_dimensions" in feature_mapping:
                w, h = feature_mapping["resized_dimensions"]
                preview = preview.resize((w, h), Image.LANCZOS)
        except Exception as e:
            print(f"Error loading image: {e}")
            return saved_paths

        # Get all unique layers in the trace data 
        layers = sorted(trace_data["layer"].unique())
        
        # Dictionaries to store heatmaps for each layer
        layer_base_maps = {}
        layer_patch_maps = {}

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

            # MODIFIED: Use importance_weight instead of predicted_top_prob
            # This change ensures we use the correct weights reflecting token importance
            # across the semantic tracing process
            if "importance_weight" in img_toks.columns:
                tok_weights = img_toks["importance_weight"].values
                print(f"Layer {L}: Using importance_weight column for heatmap")
            else:
                tok_weights = img_toks["predicted_top_prob"].values
                print(f"Layer {L}: importance_weight column not found, using predicted_top_prob")
                
            mx = tok_weights.max() if len(tok_weights) > 0 else 0
            print(f"Layer {L}: Max weight = {mx:.4f}, Tokens = {len(img_toks)}")
            
            if mx <= 0:
                print(f"Layer {L}: All weights are zero")
                layer_base_maps[L] = None
                layer_patch_maps[L] = None
                continue
                
            # Create normalized weights dictionary for this layer
            weights = {
                int(r.token_index): r.importance_weight / mx
                for _, r in img_toks.iterrows()
            }
            
            # Debug: Print some token weights for verification
            if self.debug_mode and weights:
                sample_keys = list(weights.keys())[:3]
                print(f"Layer {L} sample weights: " + 
                    ", ".join([f"Token {k}: {weights[k]:.3f}" for k in sample_keys]))

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
                        p = self._create_base_feature_overlay(
                            heatmap=heat,
                            original_image=img,
                            grid_size=(gh, gw),
                            layer_idx=L,
                            target_idx=target_idx,
                            title=f"Base Influence L{L}",
                            save_path=os.path.join(base_dir, f"base_L{L}.png"),
                            use_grid_visualization=use_grid_visualization,
                            show_values=show_values
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
                        p = self._create_patch_feature_overlay(
                            heatmap=heat,
                            spatial_preview_image=preview,
                            feature_mapping=feature_mapping,
                            patch_size=feature_mapping.get("patch_size", 14),
                            layer_idx=L,
                            target_idx=target_idx,
                            title=f"Patch Influence L{L}",
                            save_path=os.path.join(patch_dir, f"patch_L{L}.png"),
                            show_values=show_values
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
            p = self._create_grid_composite(
                heatmap_maps=layer_base_maps,
                layers=valid_base,  # FIXED: Only use layers with valid data
                title=f"Base Influence per Layer for Token {target_idx}",
                save_path=out,
                cmap="hot",
                show_values=show_values
            )
            if p:
                saved_paths.append(p)
                print(f"Created composite base grid with {len(valid_base)} layers")

        valid_patch = [L for L, hm in layer_patch_maps.items() if hm is not None]
        if valid_patch:
            out = os.path.join(save_dir, f"composite_patch_grid_{target_idx}.png")
            p = self._create_grid_composite(
                heatmap_maps=layer_patch_maps,
                layers=valid_patch,  # FIXED: Only use layers with valid data
                title=f"Patch Influence per Layer for Token {target_idx}",
                save_path=out,
                cmap="hot",
                show_values=show_values
            )
            if p:
                saved_paths.append(p)
                print(f"Created composite patch grid with {len(valid_patch)} layers")

        return saved_paths


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
        use_grid_visualization: bool = True,
        show_values: bool = True
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
            show_values: Whether to show numeric values in cells
            
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            # FIXED: Ensure the image is resized correctly and converted to RGB for consistent display
            if original_image.mode != 'RGB':
                resized_background = original_image.convert('RGB').resize(target_size, Image.LANCZOS)
            else:
                resized_background = original_image.resize(target_size, Image.LANCZOS)
            
            background_np = np.array(resized_background)
            
            # Verify image data is valid
            if background_np.min() == background_np.max():
                print(f"Warning: Image has no variation (constant value: {background_np.min()})")
            
            # Calculate cell dimensions
            grid_h, grid_w = grid_size
            cell_height = target_size[1] / grid_h
            cell_width = target_size[0] / grid_w
            
            # Create figure with explicit size and dpi
            plt.figure(figsize=(8, 8), dpi=100, facecolor='white')
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # FIXED: Display background image with explicit extent and origin
            # The issue might be with the extent parameter, so let's be explicit
            ax.imshow(
                background_np, 
                extent=(0, target_size[0], target_size[1], 0),
                aspect='auto',
                origin='upper'
            )
            
            # Debug print
            print(f"Background image shape: {background_np.shape}, Type: {background_np.dtype}")
            print(f"Heat map shape: {heatmap.shape}, Min: {heatmap.min()}, Max: {heatmap.max()}")
            
            if use_grid_visualization:
                # FIXED: Use slightly more transparent overlay for better visibility
                base_alpha = min(0.6, alpha)  # Limit maximum alpha
                
                # Create a colored grid overlay
                for r in range(grid_h):
                    for c in range(grid_w):
                        # Get heatmap value for this cell (between 0 and 1)
                        cell_value = heatmap[r, c] if r < len(heatmap) and c < len(heatmap[0]) else 0
                        
                        if cell_value > 0:  # Only draw cells with influence
                            # Calculate cell boundaries
                            x_start = c * cell_width
                            y_start = r * cell_height
                            
                            # Create colored rectangle with better visibility
                            cmap = plt.get_cmap(colormap)
                            cell_color = cmap(cell_value)
                            
                            # FIXED: Scale alpha more aggressively for better visibility
                            # Use a lower alpha value for cells with low values
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
            else:
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
                
                # FIXED: Lower alpha for better background visibility
                lowered_alpha = min(0.6, alpha)
                
                # Plot heatmap overlay with fixed alpha
                im = ax.imshow(
                    upscaled_heatmap, 
                    alpha=lowered_alpha,
                    cmap=colormap,
                    vmin=0,
                    vmax=1,
                    extent=(0, target_size[0], target_size[1], 0),
                    interpolation="nearest"
                )
            
            # Add grid lines if requested
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
            
            # Save figure with explicit bbox_inches to prevent cropping
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            plt.close('all')  # Close any other open figures
            
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
            
            # FIXED: Ensure the preview image is in RGB mode for consistent display
            if spatial_preview_image.mode != 'RGB':
                preview_image = spatial_preview_image.convert('RGB')
            else:
                preview_image = spatial_preview_image
                
            # Get dimensions for the visualization
            preview_w, preview_h = preview_image.size
            background_np = np.array(preview_image)
            
            # Verify image data is valid
            if background_np.min() == background_np.max():
                print(f"Warning: Preview image has no variation (constant value: {background_np.min()})")
            
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
            
            # Create figure with explicit size and dpi
            plt.figure(figsize=(8, 8 * preview_h / max(1, preview_w)), dpi=100, facecolor='white')
            fig, ax = plt.subplots(figsize=(8, 8 * preview_h / max(1, preview_w)))
            
            # FIXED: Display background image with explicit extent and origin
            ax.imshow(
                background_np, 
                extent=(0, preview_w, preview_h, 0),
                aspect='auto',
                origin='upper'
            )
            
            # Debug print
            print(f"Preview image shape: {background_np.shape}, Type: {background_np.dtype}")
            print(f"Patch heatmap shape: {heatmap.shape}, Min: {heatmap.min()}, Max: {heatmap.max()}")
            print(f"Padding: top={pad_top}, left={pad_left}, Content size: {resized_w_actual}x{resized_h_actual}")
            
            # FIXED: Use slightly more transparent overlay for better visibility
            base_alpha = min(0.6, alpha)  # Limit maximum alpha
            
            # Create colormap
            cmap = plt.get_cmap(colormap)
            
            # FIXED: Keep track of whether we're adding any cells
            cells_added = 0
            
            # Create grid-based overlay
            for r in range(prob_grid_h):
                for c in range(prob_grid_w):
                    # Get heatmap value for this cell (between 0 and 1)
                    cell_value = heatmap[r, c] if r < len(heatmap) and c < len(heatmap[0]) else 0
                    
                    # FIXED: Lower threshold for visibility
                    if cell_value > 0.01:  # Show even cells with small influence
                        # Calculate cell boundaries in image coordinates
                        x_start = pad_left + c * cell_width
                        y_start = pad_top + r * cell_height
                        
                        # Get color from colormap
                        cell_color = cmap(cell_value)
                        
                        # FIXED: Scale alpha more aggressively for better visibility
                        # Use a lower alpha value for cells with low values
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
                        cells_added += 1
                        
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
            
            print(f"Added {cells_added} cells to patch visualization")
            
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
                
                # Add border around content area
                content_rect = plt.Rectangle(
                    (pad_left, pad_top),
                    resized_w_actual, resized_h_actual,
                    fill=False,
                    edgecolor='blue',
                    linestyle='--',
                    linewidth=1
                )
                ax.add_patch(content_rect)
            
            # Add colorbar
            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Influence Weight")
            
            # Set title and remove axes
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            
            # Save figure with explicit bbox_inches to prevent cropping
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
            plt.close(fig)
            plt.close('all')  # Close any other open figures
            
            return save_path
        
        except Exception as e:
            print(f"Error creating patch feature overlay: {e}")
            import traceback
            traceback.print_exc()
            return None


    def _create_grid_composite(
        self,
        heatmap_maps: Dict[int, Optional[np.ndarray]],
        layers: List[int],
        title: str,
        save_path: str,
        cmap: str = "hot",
        show_values: bool = True
    ) -> Optional[str]:
        """
        Arrange each layer's 2D heatmap array into a square grid of subplots.
        
        Args:
            heatmap_maps: Dictionary mapping layer index to 2D heatmap array
            layers: List of layer indices to include (each gets its own subplot)
            title: Title for the figure
            save_path: Path to save the composite image
            cmap: Colormap to use
            show_values: Whether to show cell values
            
        Returns:
            Path to saved image or None if failed
        """
        # Only use valid layers (that have heatmaps)
        valid_layers = [L for L in layers if heatmap_maps.get(L) is not None]
        n = len(valid_layers)
        
        if n == 0:
            print("No valid layers with heatmaps. Cannot create composite.")
            return None

        # Determine grid layout
        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)
        
        # Create figure with white background
        fig, axes = plt.subplots(
            nrows, ncols, 
            figsize=(ncols*3, nrows*3),
            facecolor='white'
        )
        axes = np.array(axes).flatten()  # Handle both 1D and 2D axes arrays

        # FIXED: Verify all heatmaps are different using a hash
        heatmap_hashes = {}
        for L in valid_layers:
            hm = heatmap_maps[L]
            if hm is not None:
                # Create a simple hash of the heatmap
                hm_hash = hash(hm.tobytes())
                if hm_hash in heatmap_hashes:
                    print(f"WARNING: Layer {L} has identical heatmap to layer {heatmap_hashes[hm_hash]}")
                else:
                    heatmap_hashes[hm_hash] = L

        # Track the common colormap instance for consistent colors
        im = None
        
        # Add each layer's heatmap to the grid
        for idx, L in enumerate(valid_layers):
            ax = axes[idx]
            hm = heatmap_maps[L]
            
            if hm is None:
                ax.axis("off")
                continue
            
            # FIXED: Use consistent vmin/vmax across all subplots
            im = ax.imshow(hm, vmin=0, vmax=1, cmap=cmap)
            ax.set_title(f"Layer {L}", fontsize=10)
            
            # FIXED: Add more detailed layer stats
            ax_title = f"Layer {L} "
            if hm.max() > 0:
                ax_title += f"(max: {hm.max():.2f})"
            ax.set_title(ax_title, fontsize=10)
            
            ax.axis("off")

            # Show values in cells if requested
            if show_values:
                H, W = hm.shape
                for i in range(H):
                    for j in range(W):
                        val = hm[i, j]
                        if val > 0.1:  # Only show significant values
                            ax.text(j, i, f"{val:.2f}",
                                    ha='center', va='center', fontsize=6,
                                    color='white' if val > 0.5 else 'black')

        # Hide any unused subplots
        for ax in axes[n:]:
            ax.axis("off")

        # Add title and colorbar
        fig.suptitle(title, fontsize=14)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
            cbar.set_label("Normalized Influence", fontsize=10)

        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for title
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        return save_path
    
    #
    # STANDALONE TRACE DATA VISUALIZATION
    #
    def create_trace_data_visualizations_from_csv(
        self,
        trace_data: pd.DataFrame,
        target_text: str,
        target_idx: int,
        save_dir: str
    ) -> List[str]:
        """
        Visualize trace data with various plots.
        
        Args:
            trace_data: DataFrame with trace data
            target_text: Text of the target token
            target_idx: Index of the target token
            save_dir: Directory to save visualizations
            
        Returns:
            List of paths to saved visualizations
        """
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