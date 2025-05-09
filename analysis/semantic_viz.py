# analysis/semantic_viz.py
import os
import pandas as pd
import numpy as np
import json
import math
from typing import Dict, Any, Optional, List, Tuple, Set
from collections import defaultdict

class FlowGraphVisualizer:
    """
    Enhanced visualization tool for creating interactive semantic trace flow graphs.
    This implementation uses Cytoscape.js for better performance and improved interactivity.
    Handles complex token relationships in semantic tracing data, including tokens that are
    both sources and targets, and accurately calculates aggregate token weights.
    """

    def __init__(self, output_dir="flow_graph_output", debug_mode=False):
        """
        Initialize the visualizer.

        Args:
            output_dir: Directory to save visualizations
            debug_mode: Whether to print additional debug information
        """
        self.output_dir = output_dir
        self.debug_mode = debug_mode

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def visualize_from_csv(
        self,
        csv_path: str,
        metadata_path: Optional[str] = None,
        target_token: Optional[Dict[str, Any]] = None,
        output_format: str = "both",
        min_edge_weight: float = 0.05,
        show_orphaned_nodes: bool = False,
        use_variable_node_size: bool = True,
        max_nodes_per_layer: int = 100,
        layout_name: str = "grid"
    ) -> List[str]:
        """
        Generate flow graph visualization from a CSV trace file.

        Args:
            csv_path: Path to CSV file containing trace data
            metadata_path: Optional path to JSON metadata file
            target_token: Optional target token information
            output_format: Output format (html, png, or both)
            min_edge_weight: Minimum edge weight to include
            show_orphaned_nodes: Whether to show nodes with no connections
            use_variable_node_size: Whether to vary node size by weight
            max_nodes_per_layer: Maximum number of nodes to display per layer (for performance)
            layout_name: Layout algorithm to use ('grid', 'dagre', or 'cose')

        Returns:
            List of paths to generated visualization files
        """
        print(f"Generating flow graph from CSV: {csv_path}")
        saved_paths = []

        # Load CSV data
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found at {csv_path}")
            return saved_paths

        try:
            # Preprocess CSV file to handle common issues
            df = self._preprocess_csv(csv_path)
            print(f"Loaded {len(df)} rows of trace data")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return saved_paths

        # Load metadata (if available)
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

        # Create output directory
        target_idx = target_token.get("index", "unknown")
        target_text = target_token.get("text", "unknown")

        # Create a directory with a meaningful name
        if isinstance(target_text, str) and len(target_text) > 0:
            # Clean up filename
            target_text_clean = "".join(c if c.isalnum() else "_" for c in target_text).strip("_")
        else:
            target_text_clean = "unknown"

        save_dir = os.path.join(self.output_dir, f"token_{target_idx}_{target_text_clean}")
        os.makedirs(save_dir, exist_ok=True)

        # Create flow graph visualization
        try:
            print("Creating interactive Cytoscape.js token flow graph visualization...")
            flow_graph_params = {
                "output_format": output_format,
                "show_orphaned_nodes": show_orphaned_nodes,
                "min_edge_weight": min_edge_weight,
                "use_variable_node_size": use_variable_node_size,
                "max_nodes_per_layer": max_nodes_per_layer,
                "layout_name": layout_name,
                "debug_mode": self.debug_mode
            }

            flow_graph_paths = self.create_cytoscape_flow_graph_from_csv(
                df, target_text, target_idx, save_dir, **flow_graph_params
            )
            saved_paths.extend(flow_graph_paths)

        except Exception as e:
            print(f"Error creating flow graph visualization: {e}")
            import traceback
            traceback.print_exc()

        print(f"Visualization complete. Generated {len(saved_paths)} files.")
        return saved_paths

    def _preprocess_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Preprocess the CSV file to fix common issues and prepare data for visualization.
        Handles parsing of comma-separated lists, conversion of boolean values, and sets default
        values for missing columns.

        Args:
            csv_path: Path to the CSV file

        Returns:
            Processed DataFrame
        """
        # Read CSV file with boolean conversion
        df = pd.read_csv(
            csv_path,
            true_values=['True', 'true', '1'],
            false_values=['False', 'false', '0'],
        )

        # Check for predicted columns and debug
        missing = [c for c in df.columns if 'predicted' in c]
        if self.debug_mode:
            print(f"Prediction-related columns found: {missing}")

        # Ensure boolean columns are properly handled
        boolean_columns = ['is_target', 'is_node', 'is_edge']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
                print(f"Converted '{col}' column to boolean")

        # Fix issues in token_text column
        if 'token_text' in df.columns:
            # Fix NaN values
            nan_mask = df['token_text'].isna()
            if nan_mask.any():
                print(f"Fixed {nan_mask.sum()} NaN values")
                df.loc[nan_mask, 'token_text'] = "(nan)"

            # Fix empty strings
            empty_mask = df['token_text'] == ""
            if empty_mask.any():
                print(f"Fixed {empty_mask.sum()} empty strings")
                df.loc[empty_mask, 'token_text'] = "(empty)"

            # Special case: normalize special token representations
            special_tokens = ["<s>", "<pad>", "<bos>", "<eos>", "<image>"]
            for token in special_tokens:
                # Ensure these tokens remain as is
                token_mask = df['token_text'] == token
                if token_mask.any():
                    print(f"Preserved {token_mask.sum()} '{token}' tokens")

        # Define function to parse comma-separated lists
        def _parse_comma_list(x, dtype=float):
            if pd.isna(x) or str(x).strip() == '':
                return []
            return [dtype(i.strip()) for i in str(x).split(',') if i.strip()]

        # Convert sources_indices and sources_weights to actual lists
        for col in ['sources_indices', 'sources_weights']:
            if col in df.columns:
                if col == 'sources_indices':
                    df[col] = df[col].apply(lambda x: _parse_comma_list(x, int))
                else:
                    df[col] = df[col].apply(lambda x: _parse_comma_list(x, float))
                print(f"Converted '{col}' column to Python lists")

        # Handle predicted_top_token
        if 'predicted_top_token' not in df.columns:
            print("No 'predicted_top_token' column found, adding default value")
            df['predicted_top_token'] = '?'

        # Handle global_weight - calculated from source weights if not present
        if 'global_weight' not in df.columns:
            print("No 'global_weight' column found, will calculate from source connections")
            # Initially set to 0; we'll calculate actual values during visualization
            df['global_weight'] = 0.0

        # Ensure global_weight is float
        if 'global_weight' in df.columns:
            df['global_weight'] = df['global_weight'].astype(float)
            print("Ensured 'global_weight' column is float type")

        # Normalize edge weights within each target token's sources
        if 'sources_weights' in df.columns:
            def normalize_weights(weights):
                if not weights or len(weights) == 0:
                    return []
                weights_sum = sum(weights)
                if weights_sum <= 0:
                    return weights
                return [w / weights_sum for w in weights]

            # Add normalized weights column
            df['sources_weights_normalized'] = df['sources_weights'].apply(normalize_weights)
            print("Added 'sources_weights_normalized' column for node-local edge weight normalization")

        return df

    def _extract_target_token_info(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract target token information from DataFrame and metadata.
        Prioritizes metadata if available, otherwise selects the highest target token index.

        Args:
            df: DataFrame containing trace data
            metadata: Metadata dictionary

        Returns:
            Dictionary containing target token information
        """
        # First try to get from metadata
        if "target_tokens" in metadata and metadata["target_tokens"]:
            return metadata["target_tokens"][0]

        # Otherwise, extract from DataFrame
        target_tokens = df[df["is_target"] == True]
        if not target_tokens.empty:
            # Take highest token index as the main target
            target_idx = target_tokens["token_index"].max()
            target_row = target_tokens[target_tokens["token_index"] == target_idx].iloc[0]

            return {
                "index": int(target_idx),
                "text": target_row["token_text"],
                "id": int(target_row["token_id"])
            }

        # Fallback to using highest token index in DataFrame
        highest_idx = df["token_index"].max()
        highest_row = df[df["token_index"] == highest_idx].iloc[0]

        return {
            "index": int(highest_idx),
            "text": highest_row["token_text"],
            "id": int(highest_row["token_id"])
        }

    def create_cytoscape_flow_graph_from_csv(
        self,
        trace_data: pd.DataFrame,
        target_text: str,
        target_idx: int,
        save_dir: str,
        min_edge_weight: float = 0.05,
        use_variable_node_size: bool = True,
        min_node_size: int = 14,
        max_node_size: int = 80,     # Adjusted to 46px maximum
        debug_mode: bool = False,
        use_exponential_scaling: bool = True,
        output_format: str = "both",
        show_orphaned_nodes: bool = False,
        max_nodes_per_layer: int = 100,
        horizontal_spacing: float = 120.0,  # Increased default to 120px
        vertical_spacing: float = 50.0,
        **kwargs
    ) -> List[str]:
        """
        Create an interactive flow graph using Cytoscape.js with diamond-shaped layout
        and edge connections between adjacent token layers.
        Handles complex token relationships and calculates global weights based on
        accumulated source connections.

        Args:
            trace_data: DataFrame of token trace records
            target_text: Text of the target token
            target_idx: Index of the target token
            save_dir: Directory to save visualization files
            min_edge_weight: Minimum edge weight to include
            use_variable_node_size: Whether to vary node size by weight
            min_node_size: Minimum node size for visualization (in px)
            max_node_size: Maximum node size for visualization (in px)
            debug_mode: Whether to print debug info
            use_exponential_scaling: Whether to use exponential scaling for node sizes
            output_format: Output format (html, png, or both)
            show_orphaned_nodes: Whether to show nodes without connections
            max_nodes_per_layer: Maximum number of nodes to display per layer
            horizontal_spacing: Space between layers (in px)
            vertical_spacing: Space between nodes in a layer (in px)
            **kwargs: Other parameters for future expansion

        Returns:
            List of paths to saved visualization files
        """
        saved_paths = []

        # Check for empty data
        if trace_data.empty:
            print("No data for interactive flow graph.")
            return saved_paths

        # Extract all unique layers from the data and convert to native Python types
        # This is important for JSON serialization later
        layers = sorted([int(layer) for layer in trace_data["layer"].unique()])
        if debug_mode:
            print(f"Detected {len(layers)} layers: {layers}")

        #---------------------------------------------------------------------------
        # STEP 1: Pre-process the data and calculate accumulated weights for nodes
        #---------------------------------------------------------------------------
        # Create a copy of the DataFrame to avoid modifying the original
        df = trace_data.copy()

        # Initialize the global accumulated weight dictionary
        # This will store the total weight for each token based on all its connections
        token_global_weights = defaultdict(float)

        # Phase 1: Calculate global weights by summing the weights for each token
        # A token's global weight is the sum of weights from all its outgoing connections
        target_rows = df[df['is_target'] == True]
        for _, row in target_rows.iterrows():
            layer = int(row['layer'])
            token_idx = int(row['token_index'])

            # If this target token has source connections, add their weights to the global weights
            sources_indices = row.get('sources_indices', [])
            sources_weights = row.get('sources_weights', [])

            # Skip if no sources or weights data
            if not isinstance(sources_indices, list) or not isinstance(sources_weights, list):
                continue

            # Ensure we have valid data
            if len(sources_indices) > 0 and len(sources_weights) > 0:

                # For each source, add its weighted contribution to the global weight
                min_len = min(len(sources_indices), len(sources_weights))
                for i in range(min_len):
                    src_idx = sources_indices[i]
                    weight = sources_weights[i]

                    # Sources are in the previous layer
                    src_layer = layer - 1

                    # Add this source's weight to its global weight
                    token_global_weights[(src_layer, src_idx)] += weight
                    token_global_weights[(layer, token_idx)] += weight

        if debug_mode:
            print(f"Calculated global weights for {len(token_global_weights)} tokens")

            # Show the top weighted tokens for debugging
            top_weights = sorted(
                [(layer, idx, w) for (layer, idx), w in token_global_weights.items()],
                key=lambda x: x[2],
                reverse=True
            )[:10]

            print("Top 10 weighted tokens:")
            for layer, idx, weight in top_weights:
                token_row = df[(df['layer'] == layer) & (df['token_index'] == idx)]
                if not token_row.empty:
                    token_text = token_row.iloc[0]['token_text']
                    print(f"  Layer {layer}, Token {idx} ('{token_text}'): {weight:.4f}")


        for (layer, token_idx), weight in token_global_weights.items():
            mask = (df['layer'] == layer) & (df['token_index'] == token_idx)
            if any(mask):
                df.loc[mask, 'global_weight'] = weight

        #---------------------------------------------------------------------------
        # STEP 2: Group tokens by layer and apply limits for better performance
        #---------------------------------------------------------------------------
        # This step primarily organizes data by layers and applies per-layer limits
        layer_buckets = {}
        for layer_idx in layers:
            # Get tokens for this layer
            layer_tokens = df[df["layer"] == layer_idx]

            # Apply maximum nodes per layer limit if needed
            if len(layer_tokens) > max_nodes_per_layer:
                # Prioritize target tokens and calculate importance for non-targets
                target_tokens = layer_tokens[layer_tokens["is_target"] == True]
                non_target_tokens = layer_tokens[layer_tokens["is_target"] == False]

                # Add global weights to dataframe for sorting
                for idx, row in non_target_tokens.iterrows():
                    token_idx = int(row['token_index'])
                    global_weight = token_global_weights.get((layer_idx, token_idx), 0.0)
                    df.at[idx, 'global_weight'] = global_weight

                # Keep all target tokens and fill remaining slots with most important non-target tokens
                remaining_slots = max_nodes_per_layer - len(target_tokens)
                if remaining_slots > 0:
                    # Sort non-target tokens by global weight (descending)
                    sorted_non_targets = non_target_tokens.sort_values('global_weight', ascending=False)

                    # Take only the top tokens to fill remaining slots
                    selected_non_targets = sorted_non_targets.head(remaining_slots)
                    layer_tokens = pd.concat([target_tokens, selected_non_targets])
                else:
                    # If we have more target tokens than max, just keep all targets
                    layer_tokens = target_tokens

            # Store filtered tokens for this layer
            layer_buckets[layer_idx] = layer_tokens

        #---------------------------------------------------------------------------
        # STEP 3: Find all relevant tokens and edges to include in the visualization
        #---------------------------------------------------------------------------
        # Track all target tokens and their sources
        target_to_sources = {}  # Maps (layer, token_idx) -> list of source (layer, token_idx)

        # Analyze target tokens to find edge connections
        for layer_idx, layer_df in layer_buckets.items():
            target_rows = layer_df[layer_df['is_target'] == True]

            for _, row in target_rows.iterrows():
                token_idx = int(row['token_index'])
                tgt_key = (layer_idx, token_idx)

                # Get this target's sources
                sources_indices = row.get('sources_indices', [])
                sources_weights = row.get('sources_weights', [])

                if not isinstance(sources_indices, list) or not isinstance(sources_weights, list):
                    continue

                # Skip if no sources
                if len(sources_indices) == 0 or len(sources_weights) == 0:
                    continue

                # Add each valid source connection
                sources = []
                min_len = min(len(sources_indices), len(sources_weights))
                for i in range(min_len):
                    src_idx = sources_indices[i]
                    weight = sources_weights[i]

                    # Apply minimum edge weight filter
                    if weight < min_edge_weight:
                        continue

                    # Source tokens are in the previous layer
                    src_layer = layer_idx - 1
                    src_key = (src_layer, src_idx)

                    # Ensure the source layer exists
                    if src_layer not in layers:
                        if debug_mode:
                            print(f"Warning: Source layer {src_layer} not found in layers: {layers}")
                        continue

                    # Add source info
                    sources.append({
                        'key': src_key,
                        'index': src_idx,
                        'weight': weight
                    })

                # Store the sources for this target
                if sources:
                    target_to_sources[tgt_key] = sources

        if debug_mode:
            print(f"Found {len(target_to_sources)} target tokens with valid sources")

            # Count unique source tokens
            all_sources = set()
            for sources in target_to_sources.values():
                for s in sources:
                    all_sources.add(s['key'])

            print(f"Found {len(all_sources)} unique source tokens")

        #---------------------------------------------------------------------------
        # STEP 4: Ensure all connected tokens are included in the visualization
        #---------------------------------------------------------------------------
        # Create sets to track all tokens that need to be included
        all_target_keys = set(target_to_sources.keys())
        all_source_keys = set()

        for target_key, sources in target_to_sources.items():
            for source in sources:
                all_source_keys.add(source['key'])

        # Combine all keys that need nodes
        all_keys = all_target_keys.union(all_source_keys)

        if debug_mode:
            print(f"Total targets: {len(all_target_keys)}")
            print(f"Total sources: {len(all_source_keys)}")
            print(f"Total unique tokens to display: {len(all_keys)}")
            print(f"Tokens that are both source and target: {len(all_target_keys.intersection(all_source_keys))}")

        #---------------------------------------------------------------------------
        # STEP 5: Calculate node positions with diamond-shaped layout
        #---------------------------------------------------------------------------
        coords = {}

        # For each layer, calculate node positions in a diamond layout
        for layer_idx in layers:
            # Filter tokens for this layer that need to be displayed
            layer_tokens = []

            # Include target tokens at this layer
            layer_targets = [key for key in all_target_keys if key[0] == layer_idx]
            for tgt_key in layer_targets:
                token_row = df[(df['layer'] == tgt_key[0]) & (df['token_index'] == tgt_key[1])]
                if not token_row.empty:
                    layer_tokens.append(token_row.iloc[0])

            # Include source tokens at this layer
            layer_sources = [key for key in all_source_keys if key[0] == layer_idx]
            for src_key in layer_sources:
                if src_key in layer_targets:
                    # Already included as a target
                    continue

                token_row = df[(df['layer'] == src_key[0]) & (df['token_index'] == src_key[1])]
                if not token_row.empty:
                    layer_tokens.append(token_row.iloc[0])

            # Skip empty layers
            if not layer_tokens:
                continue

            # Sort tokens by index for consistent ordering
            layer_tokens.sort(key=lambda x: x['token_index'])

            # Center the layer vertically to create diamond shape
            offset = -(len(layer_tokens) - 1) / 2.0

            # Calculate position for each token in this layer
            for rank, row in enumerate(layer_tokens):
                token_idx = int(row["token_index"])

                # Position nodes in a centered grid
                x = float(layer_idx * horizontal_spacing)
                y = float((rank + offset) * vertical_spacing)

                # Store coordinates keyed by (layer, token_index)
                coords[(layer_idx, token_idx)] = (x, y)

        if debug_mode:
            print(f"Calculated positions for {len(coords)} tokens")

        #---------------------------------------------------------------------------
        # STEP 6: Build Cytoscape nodes and edges
        #---------------------------------------------------------------------------
        nodes = []
        edges = []

        # Create mapping from token keys to node IDs
        token_id_map = {}

        def calculate_node_size(global_weight: float) -> float:
            """
            Calculate node size based purely on its global accumulated weight.

            Args:
                global_weight: The global accumulated weight of the node.

            Returns:
                The computed node size in pixels.
            """
            w = max(global_weight, 0.00001)
            size = min_node_size + (max_node_size - min_node_size) * math.sqrt(w)

            return float(max(min_node_size, min(max_node_size, size)))


        # 1. Create nodes for all tokens
        for (layer_idx, token_idx), (x, y) in coords.items():
            # Get token data from DataFrame
            token_row = df[(df['layer'] == layer_idx) & (df['token_index'] == token_idx)]

            if token_row.empty:
                if debug_mode:
                    print(f"Warning: No data for token at layer {layer_idx}, index {token_idx}")
                continue

            row = token_row.iloc[0]

            # Determine if this token is a target, source, or both
            key = (layer_idx, token_idx)
            is_target = key in all_target_keys
            is_source = key in all_source_keys

            # Get token type and text
            token_type = int(row['token_type'])
            token_text = row['token_text']

            # Get global weight from DataFrame (which now includes ALL tokens' weights)
            global_weight = float(row['global_weight'])

            # Get top prediction probability (keep separate from global weight)
            top_pred_prob = float(row.get('predicted_top_prob', 0.0))

            # IMPORTANT: Calculate node size based on global weight
            node_size = calculate_node_size(global_weight)

            # Create node ID
            node_id = f"L{layer_idx}_T{token_idx}"
            token_id_map[key] = node_id

            # Clean text for display
            clean_text = self._sanitize_text_for_display(token_text)

            # Create abbreviated label for display
            label = token_text
            if label and len(label.strip()) > 3:
                label = label[:1] + "..."

            # Get token type label
            token_type_label = {0: "Generated", 1: "Text", 2: "Image"}.get(token_type, "Other")

            # Get predicted_top_token if available
            predicted_top_token = row.get("predicted_top_token", "?")
            if not isinstance(predicted_top_token, str):
                predicted_top_token = str(predicted_top_token)

            # Create node with all necessary data
            node = {
                "data": {
                    "id": node_id,
                    "label": label,
                    "full_text": clean_text,
                    "layer": int(layer_idx),
                    "token_idx": int(token_idx),
                    "type": token_type_label,
                    "type_code": token_type,
                    "top_pred_prob": top_pred_prob,  # Keep prediction probability separate
                    "global_weight": global_weight,  # Use actual global weight for visualization
                    "size": node_size,  # IMPORTANT: Add calculated size to node data
                    "predicted_top_token": predicted_top_token,
                    "is_target": bool(is_target),
                    "is_source": bool(is_source)
                },
                "position": {
                    "x": float(x),
                    "y": float(y)
                }
            }

            nodes.append(node)

        if debug_mode:
            print(f"Created {len(nodes)} nodes")
            targets = [n for n in nodes if n['data']['is_target']]
            sources = [n for n in nodes if n['data']['is_source']]
            both = [n for n in nodes if n['data']['is_target'] and n['data']['is_source']]
            print(f"  Targets: {len(targets)}")
            print(f"  Sources: {len(sources)}")
            print(f"  Both: {len(both)}")

        # 2. Create edges between tokens
        for target_key, sources in target_to_sources.items():
            # Skip if target isn't in the token map (shouldn't happen)
            if target_key not in token_id_map:
                continue

            target_node_id = token_id_map[target_key]

            # Create edges to each source
            for source in sources:
                source_key = source['key']

                # Skip if source isn't in the token map (shouldn't happen)
                if source_key not in token_id_map:
                    continue

                source_node_id = token_id_map[source_key]
                weight = source['weight']

                # Create edge
                edge_id = f"{source_node_id}_{target_node_id}"
                edge = {
                    "data": {
                        "id": edge_id,
                        "source": source_node_id,
                        "target": target_node_id,
                        "weight": float(weight)
                    }
                }

                edges.append(edge)

        if debug_mode:
            print(f"Created {len(edges)} edges")

        # Return early if no nodes or edges
        if not nodes or not edges:
            print("No nodes or edges to display. Visualization would be empty.")
            return saved_paths

        #---------------------------------------------------------------------------
        # STEP 8: Create the HTML visualization with Cytoscape.js
        #---------------------------------------------------------------------------
        html_dir = os.path.join(save_dir, "interactive")
        os.makedirs(html_dir, exist_ok=True)
        html_path = os.path.join(html_dir, f"flow_graph_{target_idx}.html")

        # Create the title for the visualization
        title = f"Interactive Semantic Trace Flow Graph for '{target_text}' (idx: {target_idx})"

        # Create style for Cytoscape.js - using mapData for proper value mapping
        stylesheet = [
            # Node styles by type
            {
                "selector": 'node[type_code = 0]',  # Generated tokens
                "style": {
                    "background-color": "#2ecc71",
                    "border-width": 2,
                    "border-color": "#27ae60"
                }
            },
            {
                "selector": 'node[type_code = 1]',  # Text tokens
                "style": {
                    "background-color": "#3498db",
                    "border-width": 2,
                    "border-color": "#2980b9"
                }
            },
            {
                "selector": 'node[type_code = 2]',  # Image tokens
                "style": {
                    "background-color": "#e74c3c",
                    "border-width": 2,
                    "border-color": "#c0392b"
                }
            },
            # Node label style
            {
                "selector": 'node',
                "style": {
                    "label": "data(label)",
                    "text-valign": "center",
                    "text-halign": "center",
                    "color": "#fff",
                    "font-size": "12px",
                    "text-outline-width": 1,
                    "text-outline-color": "#555",
                    # Use size for sizing
                    "width": "mapData(size, 14, 46, 14, 46)",
                    "height": "mapData(size, 14, 46, 14, 46)",
                    # Border width proportional to global_weight
                    "border-width": "mapData(global_weight, 0, 1, 1, 4)",
                    "text-wrap": "wrap",
                    "text-max-width": "80px"
                }
            },
            # Highlight target tokens
            {
                "selector": 'node[is_target = true]',
                "style": {
                    "border-width": 4,
                    "border-color": "#f39c12",
                    "border-style": "double"
                }
            },
            # Style for source tokens
            {
                "selector": 'node[is_source = true][is_target = false]',
                "style": {
                    "border-color": "#95a5a6",
                    "border-width": 2
                }
            },
            # Style for tokens that are both source and target
            {
                "selector": 'node[is_source = true][is_target = true]',
                "style": {
                    "border-color": "#f39c12",
                    "border-width": 4,
                    "border-style": "dashed"
                }
            },
            # Edge style - use weights
            {
                "selector": 'edge',
                "style": {
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "arrow-scale": 0.8,
                    "line-color": "#aaa",
                    "target-arrow-color": "#aaa",
                    "width": "mapData(weight, 0, 1, 1, 6)",  # Maps weight to width range
                    "opacity": "mapData(weight, 0, 1, 0.15, 1)"  # Maps weight to opacity range
                }
            },
            # Highlighted elements
            {
                "selector": '.highlighted',
                "style": {
                    "background-color": "data(color)",
                    "line-color": "#f39c12",
                    "target-arrow-color": "#f39c12",
                    "transition-property": "background-color, line-color, target-arrow-color, opacity",
                    "transition-duration": "0.2s",
                    "opacity": 1,
                    "z-index": 999
                }
            },
            # Faded elements
            {
                "selector": '.faded',
                "style": {
                    "opacity": 0.1,
                    "z-index": 1,
                    "transition-property": "opacity",
                    "transition-duration": "0.2s"
                }
            }
        ]

        # Custom JSON encoder to handle NumPy types properly
        class NumpyEncoder(json.JSONEncoder):
            """Custom JSON encoder for NumPy types"""
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        # IMPORTANT: Convert to flat JSON list for embedding in HTML (not an object with nodes/edges)
        elements_list = nodes + edges
        elements_json = json.dumps(elements_list, indent=2, cls=NumpyEncoder)
        stylesheet_json = json.dumps(stylesheet, indent=2)

        # Create HTML with embedded Cytoscape.js
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- Cytoscape.js dependencies -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.21/lodash.min.js"></script>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f8f8;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}

        .header {{
            background-color: #fff;
            border-bottom: 1px solid #ddd;
            padding: 10px 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            margin: 0;
            padding: 0;
            font-size: 20px;
            color: #333;
        }}

        .container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        .sidebar {{
            width: 280px;
            background-color: #fff;
            border-right: 1px solid #ddd;
            padding: 15px;
            overflow-y: auto;
        }}

        .main-content {{
            flex: 1;
            position: relative;
            overflow: hidden;
        }}

        #cy {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }}

        .controls {{
            margin-top: 15px;
        }}

        .control-group {{
            margin-bottom: 15px;
            border: 1px solid #eee;
            border-radius: 4px;
            padding: 10px;
        }}

        .control-group h3 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 16px;
            color: #333;
        }}

        button {{
            padding: 6px 10px;
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            margin-right: 5px;
            margin-bottom: 5px;
        }}

        button:hover {{
            background-color: #2980b9;
        }}

        button.active {{
            background-color: #27ae60;
        }}

        .slider-container {{
            margin-top: 10px;
            margin-bottom: 5px;
        }}

        .slider-container label {{
            display: block;
            margin-bottom: 5px;
            font-size: 13px;
            color: #555;
        }}

        .slider {{
            width: 100%;
            margin-bottom: 5px;
        }}

        .slider-value {{
            text-align: center;
            font-size: 12px;
            color: #777;
        }}

        .search-box {{
            width: 100%;
            padding: 6px;
            margin-bottom: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}

        .node-info {{
            margin-top: 15px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 4px;
            font-size: 13px;
            display: none;
        }}

        .node-info h3 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 15px;
            color: #333;
        }}

        .info-item {{
            margin-bottom: 6px;
        }}

        .info-label {{
            font-weight: bold;
            color: #555;
        }}

        .legend {{
            margin-top: 15px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 4px;
        }}

        .legend h3 {{
            margin-top: 0;
            margin-bottom: 8px;
            font-size: 15px;
            color: #333;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 6px;
        }}

        .legend-color {{
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 8px;
        }}

        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(255, 255, 255, 0.8);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }}

        .spinner {{
            border: 4px solid rgba(0, 0, 0, 0.1);
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border-left-color: #3498db;
            animation: spin 1s linear infinite;
            margin-bottom: 10px;
        }}

        .loading-text {{
            font-size: 14px;
            color: #333;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        #error-message {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: #f44336;
            color: white;
            padding: 15px 20px;
            border-radius: 4px;
            font-size: 16px;
            z-index: 10000;
            display: none;
        }}

        /* Export button */
        #export-png {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            z-index: 100;
            background-color: #27ae60;
        }}

        .dropdown-select {{
            width: 100%;
            padding: 6px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #fff;
            font-size: 13px;
            margin-bottom: 5px;
        }}

        /* Tooltip for node hover */
        .tooltip {{
            position: absolute;
            z-index: 1000;
            background-color: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            pointer-events: none;
            display: none;
            max-width: 250px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}

        .tooltip-row {{
            display: flex;
            margin-bottom: 5px;
        }}

        .tooltip-label {{
            font-weight: bold;
            margin-right: 5px;
            min-width: 60px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
    </div>

    <div class="container">
        <div class="sidebar">
            <div class="controls">
                <div class="control-group">
                    <h3>Navigation</h3>
                    <button id="reset-view">Reset View</button>
                    <button id="fit-view">Fit to Screen</button>
                </div>

                <div class="control-group">
                    <h3>Layout</h3>

                    <!-- Layer spacing slider -->
                    <div class="slider-container">
                        <label>Layer Spacing:</label>
                        <input type="range" min="60" max="200" value="{horizontal_spacing}" class="slider" id="spacing-slider">
                        <div id="spacing-value" class="slider-value">{horizontal_spacing}px</div>
                    </div>

                    <!-- Node size controls -->
                    <div class="slider-container">
                        <label>Min Node Size:</label>
                        <input type="range" min="10" max="30" value="{min_node_size}" class="slider" id="min-size-slider">
                        <div id="min-size-value" class="slider-value">{min_node_size}px</div>
                    </div>

                    <div class="slider-container">
                        <label>Max Node Size:</label>
                        <input type="range" min="30" max="60" value="{max_node_size}" class="slider" id="max-size-slider">
                        <div id="max-size-value" class="slider-value">{max_node_size}px</div>
                    </div>
                </div>

                <div class="control-group">
                    <h3>Layer Filter</h3>
                    <div class="slider-container">
                        <input type="range" min="0" max="{len(layers) - 1}" value="0" class="slider" id="layer-slider">
                        <div id="layer-value" class="slider-value">Current Layer: 0</div>
                    </div>
                    <button id="show-all-layers">Show All Layers</button>
                    <button id="show-current-layer">Show Current Layer</button>
                </div>

                <!-- Trace Mode selector -->
                <div class="control-group">
                    <h3>Trace Depth</h3>
                    <select id="trace-mode" class="dropdown-select">
                        <option value="direct">Direct Neighbors</option>
                        <option value="upstream">Full Upstream</option>
                        <option value="downstream">Full Downstream</option>
                        <option value="both">Both Directions</option>
                    </select>
                </div>

                <div class="control-group">
                    <h3>Find Token</h3>
                    <input type="text" id="search-box" class="search-box" placeholder="Search by token text or predicted token...">
                    <div id="search-results"></div>
                </div>
            </div>

            <div id="node-info" class="node-info">
                <h3>Token Information</h3>
                <div class="info-item">
                    <span class="info-label">Text:</span>
                    <span id="info-text"></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Layer:</span>
                    <span id="info-layer"></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Index:</span>
                    <span id="info-index"></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Type:</span>
                    <span id="info-type"></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Role:</span>
                    <span id="info-role"></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Top Prob:</span>
                    <span id="info-top-prob"></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Global Weight:</span>
                    <span id="info-global-weight"></span>
                </div>
                <div class="info-item">
                    <span class="info-label">Top Token:</span>
                    <span id="info-top-token"></span>
                </div>
            </div>

            <div class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #2ecc71;"></div>
                    <span>Generated Tokens</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #3498db;"></div>
                    <span>Text Tokens</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e74c3c;"></div>
                    <span>Image Tokens</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: white; border: 3px solid #f39c12;"></div>
                    <span>Target Token</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: white; border: 2px solid #95a5a6;"></div>
                    <span>Source Token</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: white; border: 4px dashed #f39c12;"></div>
                    <span>Both Source & Target</span>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div id="cy"></div>
            <!-- Node tooltip for hover display -->
            <div id="node-tooltip" class="tooltip"></div>
            <button id="export-png">Export PNG</button>
            <div id="loading-overlay" class="loading-overlay">
                <div class="spinner"></div>
                <div class="loading-text">Loading graph...</div>
            </div>
            <div id="error-message"></div>
        </div>
    </div>

    <script>
        // Error handling for debugging
        window.onerror = function(message, source, lineno, colno, error) {{
            console.error('Error:', message, 'at', source, lineno, colno);
            document.getElementById('loading-overlay').style.display = 'none';

            const errorMessageElement = document.getElementById('error-message');
            errorMessageElement.textContent = 'Error: ' + message;
            errorMessageElement.style.display = 'block';

            return true;
        }};

        // Parse elements and stylesheet from embedded JSON
        const elements = {elements_json};
        const stylesheet = {stylesheet_json};

        console.log('Initializing with', elements.length, 'elements');

        // Safety timeout to ensure loading overlay disappears
        setTimeout(() => {{
            document.getElementById('loading-overlay').style.display = 'none';
        }}, 10000);

        // Register the plugin properly before instantiating Cytoscape
        if (window.cytoscapeDagre) {{
            cytoscape.use(cytoscapeDagre);
            console.log('Registered dagre plugin');
        }}

        // Initialize Cytoscape when the page is loaded
        document.addEventListener('DOMContentLoaded', function() {{
            // Get reference to the loading overlay
            const loadingOverlay = document.getElementById('loading-overlay');

            try {{
                console.log('Creating Cytoscape instance...');

                // Create Cytoscape instance
                const cy = cytoscape({{
                    container: document.getElementById('cy'),
                    elements: elements,
                    style: stylesheet,
                    wheelSensitivity: 0.3,
                    minZoom: 0.1,
                    maxZoom: 3,
                    layout: {{
                        name: 'preset'  // Use the predefined positions
                    }}
                }});

                console.log('Cytoscape instance created successfully');

                // Track the most recently clicked node
                let lastClickedNode = null;

                // Store current layout parameters
                let currentHorizontalSpacing = {horizontal_spacing};
                let currentMinNodeSize = {min_node_size};
                let currentMaxNodeSize = {max_node_size};

                // Store current trace mode
                let currentTraceMode = 'direct';

                // Track mouse position for tooltip placement
                let mouseX = 0;
                let mouseY = 0;
                cy.container().addEventListener('mousemove', function(event) {{
                    mouseX = event.offsetX;
                    mouseY = event.offsetY;
                }});

                // Layer filtering functionality
                const allLayers = [...new Set(cy.nodes().map(node => node.data('layer')))].sort((a, b) => a - b);

                // Function to dynamically relocate layers based on spacing
                function relocateLayers(spacing) {{
                    // Update current spacing
                    currentHorizontalSpacing = spacing;

                    // Update the spacing value display
                    document.getElementById('spacing-value').textContent = spacing + 'px';

                    // Batch all position changes to avoid multiple redraws
                    cy.batch(() => {{
                        cy.nodes().forEach(node => {{
                            const layer = node.data('layer');
                            // Get current Y position
                            const y = node.position('y');
                            // Calculate new X position based on new spacing
                            const x = layer * spacing;
                            // Update node position
                            node.position({{
                                x: x,
                                y: y
                            }});
                        }});
                    }});
                }}

                // Function to update node sizes
                function updateNodeSizes(minSize, maxSize) {{
                    // Update current size values
                    currentMinNodeSize = minSize;
                    currentMaxNodeSize = maxSize;

                    // Update the size value displays
                    document.getElementById('min-size-value').textContent = minSize + 'px';
                    document.getElementById('max-size-value').textContent = maxSize + 'px';

                    // Update style to use new size range
                    cy.style()
                        .selector('node')
                        .style({{
                            'width': `mapData(size, ${{minSize}}, ${{maxSize}}, ${{minSize}}, ${{maxSize}})`,
                            'height': `mapData(size, ${{minSize}}, ${{maxSize}}, ${{minSize}}, ${{maxSize}})`
                        }})
                        .update();
                }}

                // Function to show or hide layers based on selection
                function updateLayerDisplay(currentLayer, showAll = false) {{
                    if (showAll) {{
                        // Show all layers
                        cy.elements().removeClass('faded');
                        document.getElementById('layer-value').textContent = 'Showing All Layers';
                    }} else {{
                        // Show only the selected layer
                        const layerNodes = cy.nodes().filter(node => node.data('layer') == currentLayer);
                        const connectedEdges = layerNodes.connectedEdges();

                        // Fade everything
                        cy.elements().addClass('faded');

                        // Highlight the current layer
                        layerNodes.removeClass('faded');
                        connectedEdges.removeClass('faded');

                        document.getElementById('layer-value').textContent = `Current Layer: ${{currentLayer}}`;
                    }}
                }}

                // Function to collect nodes based on trace mode
                function collectNodesForHighlight(node, mode) {{
                    const connected = cy.collection();

                    // Always include the selected node itself
                    connected.merge(node);

                    if (mode === 'direct') {{
                        // Direct neighbors mode - just immediate neighbors
                        const prevLayerNodes = node.incomers('node');
                        const nextLayerNodes = node.outgoers('node');

                        connected.merge(prevLayerNodes);
                        connected.merge(nextLayerNodes);
                    }}
                    else if (mode === 'upstream') {{
                        // Full upstream mode - all predecessors
                        connected.merge(node.predecessors('node'));
                    }}
                    else if (mode === 'downstream') {{
                        // Full downstream mode - all successors
                        connected.merge(node.successors('node'));
                    }}
                    else {{
                        // Both directions mode - all predecessors and successors
                        connected.merge(node.predecessors('node'));
                        connected.merge(node.successors('node'));
                    }}

                    // Also include relevant edges
                    const connectedEdges = connected.connectedEdges().filter(edge => {{
                        return connected.contains(edge.source()) && connected.contains(edge.target());
                    }});
                    connected.merge(connectedEdges);

                    return connected;
                }}

                // Layer slider event handler
                const layerSlider = document.getElementById('layer-slider');
                layerSlider.addEventListener('input', function() {{
                    updateLayerDisplay(parseInt(this.value), false);
                }});

                // Show all layers button
                document.getElementById('show-all-layers').addEventListener('click', function() {{
                    updateLayerDisplay(0, true);
                }});

                // Show current layer button
                document.getElementById('show-current-layer').addEventListener('click', function() {{
                    updateLayerDisplay(parseInt(layerSlider.value), false);
                }});

                // Horizontal spacing slider
                const spacingSlider = document.getElementById('spacing-slider');
                spacingSlider.addEventListener('input', function() {{
                    relocateLayers(parseInt(this.value));
                }});

                // Min node size slider
                const minSizeSlider = document.getElementById('min-size-slider');
                minSizeSlider.addEventListener('input', function() {{
                    const minSize = parseInt(this.value);
                    const maxSize = parseInt(document.getElementById('max-size-slider').value);
                    updateNodeSizes(minSize, maxSize);
                }});

                // Max node size slider
                const maxSizeSlider = document.getElementById('max-size-slider');
                maxSizeSlider.addEventListener('input', function() {{
                    const minSize = parseInt(document.getElementById('min-size-slider').value);
                    const maxSize = parseInt(this.value);
                    updateNodeSizes(minSize, maxSize);
                }});

                // Trace mode selector
                const traceModeSelect = document.getElementById('trace-mode');
                traceModeSelect.addEventListener('change', function() {{
                    currentTraceMode = this.value;

                    // If we have a selected node, update the highlighting
                    if (lastClickedNode) {{
                        highlightConnectedNodes(lastClickedNode);
                    }}
                }});

                // Hide loading overlay after initialization
                setTimeout(() => {{
                    loadingOverlay.style.display = 'none';
                    console.log('Ready!');

                    // If we have a target token, highlight it
                    const targetNodes = cy.nodes().filter(node => node.data('is_target'));
                    if (targetNodes.length > 0) {{
                        highlightConnectedNodes(targetNodes[0]);
                    }}
                }}, 100);

                // Helper function to get node role string
                function getNodeRoleString(node) {{
                    const isTarget = node.data('is_target');
                    const isSource = node.data('is_source');

                    if (isTarget && isSource) return "Target & Source";
                    if (isTarget) return "Target";
                    if (isSource) return "Source";
                    return "Other";
                }}

                // Node hover events for tooltip - UPDATED LABELS
                cy.on('mouseover', 'node', function(evt) {{
                    const node = evt.target;
                    const tooltip = document.getElementById('node-tooltip');

                    // Create tooltip content with updated labels
                    tooltip.innerHTML = `
                        <div class="tooltip-row">
                            <div class="tooltip-label">Text:</div>
                            <div>${{node.data('full_text')}}</div>
                        </div>
                        <div class="tooltip-row">
                            <div class="tooltip-label">Layer:</div>
                            <div>${{node.data('layer')}}</div>
                        </div>
                        <div class="tooltip-row">
                            <div class="tooltip-label">Index:</div>
                            <div>${{node.data('token_idx')}}</div>
                        </div>
                        <div class="tooltip-row">
                            <div class="tooltip-label">Type:</div>
                            <div>${{node.data('type')}}</div>
                        </div>
                        <div class="tooltip-row">
                            <div class="tooltip-label">Role:</div>
                            <div>${{getNodeRoleString(node)}}</div>
                        </div>
                        <div class="tooltip-row">
                            <div class="tooltip-label">Top Prob:</div>
                            <div>${{node.data('top_pred_prob').toFixed(4)}}</div>
                        </div>
                        <div class="tooltip-row">
                            <div class="tooltip-label">Global Weight:</div>
                            <div>${{node.data('global_weight').toFixed(4)}}</div>
                        </div>
                        <div class="tooltip-row">
                            <div class="tooltip-label">Top Token:</div>
                            <div>${{node.data('predicted_top_token')}}</div>
                        </div>
                    `;

                    // Position tooltip near cursor but with margins
                    const offsetX = 10;
                    const offsetY = 10;
                    const tooltipWidth = tooltip.offsetWidth;
                    const tooltipHeight = tooltip.offsetHeight;
                    const containerWidth = cy.width();
                    const containerHeight = cy.height();

                    // Adjust position to avoid going off-screen
                    let posX = mouseX + offsetX;
                    let posY = mouseY + offsetY;

                    if (posX + tooltipWidth > containerWidth) {{
                        posX = mouseX - tooltipWidth - offsetX;
                    }}

                    if (posY + tooltipHeight > containerHeight) {{
                        posY = mouseY - tooltipHeight - offsetY;
                    }}

                    tooltip.style.left = posX + 'px';
                    tooltip.style.top = posY + 'px';
                    tooltip.style.display = 'block';
                }});

                cy.on('mouseout', 'node', function() {{
                    document.getElementById('node-tooltip').style.display = 'none';
                }});

                // Keep tooltip positioned correctly while hovering over node
                cy.on('mousemove', 'node', function() {{
                    const tooltip = document.getElementById('node-tooltip');
                    if (tooltip.style.display === 'block') {{
                        const offsetX = 10;
                        const offsetY = 10;
                        const tooltipWidth = tooltip.offsetWidth;
                        const tooltipHeight = tooltip.offsetHeight;
                        const containerWidth = cy.width();
                        const containerHeight = cy.height();

                        let posX = mouseX + offsetX;
                        let posY = mouseY + offsetY;

                        if (posX + tooltipWidth > containerWidth) {{
                            posX = mouseX - tooltipWidth - offsetX;
                        }}

                        if (posY + tooltipHeight > containerHeight) {{
                            posY = mouseY - tooltipHeight - offsetY;
                        }}

                        tooltip.style.left = posX + 'px';
                        tooltip.style.top = posY + 'px';
                    }}
                }});

                // Node click event
                cy.on('tap', 'node', function(evt) {{
                    const node = evt.target;
                    highlightConnectedNodes(node);
                    // Critical: Prevent the event from propagating to the background
                    evt.stopPropagation();
                }});

                // Background click to reset
                cy.on('tap', function(evt) {{
                    if (evt.target === cy) {{
                        // Reset highlighting
                        cy.elements().removeClass('highlighted faded');
                        updateNodeInfo(null);
                        lastClickedNode = null;
                    }}
                }});

                // Highlight function that uses the current trace mode
                function highlightConnectedNodes(node) {{
                    if (!node) {{
                        return;
                    }}

                    // Reset previous highlighting
                    cy.elements().removeClass('highlighted faded');

                    // Store original background colors
                    cy.nodes().forEach(n => {{
                        if (!n.data('originalColor')) {{
                            n.data('color', n.style('background-color'));
                        }}
                    }});

                    // Collect nodes based on current trace mode
                    const connected = collectNodesForHighlight(node, currentTraceMode);

                    // Highlight connected elements and fade others
                    connected.addClass('highlighted');
                    cy.elements().not(connected).addClass('faded');

                    // Update node info panel
                    updateNodeInfo(node);

                    // Pan to the clicked node
                    cy.animate({{
                        center: {{
                            eles: node
                        }},
                        duration: 300,
                        easing: 'ease-in-out-cubic'
                    }});

                    // Store the clicked node
                    lastClickedNode = node;

                    // Debug output
                    console.log(`Highlighted node ${{node.id()}} using mode ${{currentTraceMode}}`);
                }}

                // Update node information panel - UPDATED LABELS
                function updateNodeInfo(node) {{
                    const nodeInfo = document.getElementById('node-info');
                    const infoText = document.getElementById('info-text');
                    const infoLayer = document.getElementById('info-layer');
                    const infoIndex = document.getElementById('info-index');
                    const infoType = document.getElementById('info-type');
                    const infoRole = document.getElementById('info-role');
                    const infoTopProb = document.getElementById('info-top-prob');
                    const infoGlobalWeight = document.getElementById('info-global-weight');
                    const infoTopToken = document.getElementById('info-top-token');

                    if (node) {{
                        infoText.textContent = node.data('full_text');
                        infoLayer.textContent = node.data('layer');
                        infoIndex.textContent = node.data('token_idx');
                        infoType.textContent = node.data('type');
                        infoRole.textContent = getNodeRoleString(node);
                        infoTopProb.textContent = node.data('top_pred_prob').toFixed(4);
                        infoGlobalWeight.textContent = node.data('global_weight').toFixed(4);
                        infoTopToken.textContent = node.data('predicted_top_token');

                        nodeInfo.style.display = 'block';
                    }} else {{
                        nodeInfo.style.display = 'none';
                    }}
                }}

                // Search for nodes by text or predicted token - ENHANCED SEARCH
                let searchTimeout = null;
                function searchNodes(searchText) {{
                    /**
                    * Search for nodes by token text or predicted token text
                    * Allows finding tokens by either their actual text or what they're predicted to produce
                    *
                    * @param {{string}} searchText - The text to search for
                    * @returns {{Array}} - Array of matching cytoscape nodes
                    */
                    if (!searchText || searchText.trim() === '') {{
                        return [];
                    }}

                    searchText = searchText.toLowerCase();

                    return cy.nodes().filter(node => {{
                        // Check if the token text contains the search text
                        const fullText = node.data('full_text');
                        if (fullText && fullText.toLowerCase().includes(searchText)) {{
                            return true;
                        }}

                        // Also check if the predicted token contains the search text
                        const predictedToken = node.data('predicted_top_token');
                        if (predictedToken && predictedToken.toLowerCase().includes(searchText)) {{
                            return true;
                        }}

                        return false;
                    }});
                }}

                function displaySearchResults(results) {{
                    /**
                    * Display search results in the sidebar with clear indication of match type
                    * Clearly shows whether the match was on the token text or predicted token
                    *
                    * @param {{Array}} results - Array of cytoscape nodes matching the search
                    */
                    const searchResults = document.getElementById('search-results');
                    searchResults.innerHTML = '';

                    if (results.length === 0) {{
                        searchResults.innerHTML = '<div style="padding: 8px; color: #777;">No matching tokens found</div>';
                        return;
                    }}

                    // Create results list
                    const resultsList = document.createElement('div');
                    resultsList.style.maxHeight = '200px';
                    resultsList.style.overflowY = 'auto';
                    resultsList.style.backgroundColor = '#f5f5f5';
                    resultsList.style.borderRadius = '4px';
                    resultsList.style.marginTop = '8px';

                    // Add each result
                    results.forEach(node => {{
                        const resultItem = document.createElement('div');
                        resultItem.style.padding = '6px';
                        resultItem.style.borderBottom = '1px solid #ddd';
                        resultItem.style.cursor = 'pointer';

                        // Include both token text and predicted token in the results
                        // with clear highlighting to indicate where the match occurred
                        const fullText = node.data('full_text');
                        const predictedToken = node.data('predicted_top_token');
                        const searchText = document.getElementById('search-box').value.toLowerCase();

                        let tokenMatch = false;
                        let predMatch = false;

                        // Check which field matched
                        if (fullText && fullText.toLowerCase().includes(searchText)) {{
                            tokenMatch = true;
                        }}
                        if (predictedToken && predictedToken.toLowerCase().includes(searchText)) {{
                            predMatch = true;
                        }}

                        // Build result HTML with appropriate visual feedback about match type
                        resultItem.innerHTML = `
                            <div><strong>${{tokenMatch ? `<span style="background-color: #fff7cf;">${{fullText}}</span>` : fullText}}</strong></div>
                            <div style="font-size: 12px; color: #666;">
                                Layer ${{node.data('layer')}}, Index ${{node.data('token_idx')}}
                                ${{predictedToken ? ` | Predicts: ${{predMatch ? `<span style="background-color: #e8f4ff;">${{predictedToken}}</span>` : predictedToken}}` : ''}}
                            </div>
                        `;

                        // Click event to highlight node
                        resultItem.addEventListener('click', function() {{
                            highlightConnectedNodes(node);
                        }});

                        // Hover effect
                        resultItem.addEventListener('mouseenter', function() {{
                            this.style.backgroundColor = '#e5e5e5';
                        }});
                        resultItem.addEventListener('mouseleave', function() {{
                            this.style.backgroundColor = 'transparent';
                        }});

                        resultsList.appendChild(resultItem);
                    }});

                    searchResults.appendChild(resultsList);
                }}

                // Reset view button
                document.getElementById('reset-view').addEventListener('click', function() {{
                    // Reset highlighting
                    cy.elements().removeClass('highlighted faded');
                    updateNodeInfo(null);
                    lastClickedNode = null;

                    // Fit to view
                    cy.fit(cy.elements(), 50);
                }});

                // Fit view button
                document.getElementById('fit-view').addEventListener('click', function() {{
                    cy.fit(cy.elements(), 50);
                }});

                // Export PNG button
                document.getElementById('export-png').addEventListener('click', function() {{
                    // Create PNG and download
                    const png64 = cy.png({{
                        output: 'blob',
                        scale: 2.0,
                        bg: '#ffffff',
                        full: true
                    }});

                    // Create download link
                    const downloadLink = document.createElement('a');
                    downloadLink.href = URL.createObjectURL(png64);
                    downloadLink.download = `flow_graph_${{target_idx}}.png`;
                    downloadLink.click();
                }});

                // Search functionality with debouncing
                const searchBox = document.getElementById('search-box');
                searchBox.addEventListener('input', function() {{
                    // Clear previous timeout
                    if (searchTimeout) {{
                        clearTimeout(searchTimeout);
                    }}

                    // Set new timeout (200ms delay)
                    searchTimeout = setTimeout(() => {{
                        const searchText = this.value;
                        const results = searchNodes(searchText);
                        displaySearchResults(results);
                    }}, 200);
                }});

            }} catch (error) {{
                console.error('Error initializing Cytoscape:', error);
                loadingOverlay.style.display = 'none';

                const errorMessageElement = document.getElementById('error-message');
                errorMessageElement.textContent = 'Error initializing visualization: ' + error.message;
                errorMessageElement.style.display = 'block';
            }}
        }});
    </script>
</body>
</html>
"""

        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        saved_paths.append(html_path)
        print(f"Created interactive Cytoscape.js flow graph with diamond layout: {html_path}")

        return saved_paths

    def _sanitize_text_for_display(self, text):
        """
        Improved sanitization function to avoid rendering issues with special characters.
        Escapes all angle brackets to ensure HTML/special tokens display correctly.

        Args:
            text: Input text that may contain special characters (can be string or any other type)

        Returns:
            Sanitized text that should render correctly in plotly/html
        """
        import pandas as pd

        # Handle None and NaN
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return "(nan)"

        # Force to string
        text = str(text)

        # Handle whitespace
        if text.strip() == "":
            return "(empty)"

        # Special token exact mappings
        special_tokens = {
            "<s>": "&lt;s&gt;",
            "</s>": "&lt;/s&gt;",
            "<pad>": "&lt;pad&gt;",
            "<bos>": "&lt;bos&gt;",
            "<eos>": "&lt;eos&gt;",
            "<image>": "&lt;image&gt;",
            "<|im_start|>": "&lt;|im_start|&gt;",
            "<|im_end|>": "&lt;|im_end|&gt;"
        }
        if text in special_tokens:
            return special_tokens[text]

        # Escape all < and >
        text = text.replace("<", "&lt;").replace(">", "&gt;")

        # Replace common symbols
        replacements = {
            "→": "->",
            "←": "<-",
            "⟨": "<",
            "⟩": ">",
            "⇒": "=>",
            "⇐": "<=",
            "≤": "<=",
            "≥": ">=",
            "…": "...",
            "\n": "\\n",
            "\t": "\\t"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        # Only keep ASCII, optionally replacing non-ASCII with middle dots
        result = ""
        for ch in text:
            if ord(ch) < 128:
                result += ch
            else:
                result += "·"
        return result or "(empty)"