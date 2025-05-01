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
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from collections import defaultdict


class EnhancedSemanticTracingVisualizer:
    """
    Visualizes semantic tracing results from CSV data, creating flow graphs and heatmaps.
    Provides improved spacing for flow graphs and modular visualization functions.
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
                print("Creating token flow graph visualization...")
                flow_graph_paths = self.create_flow_graph_from_csv(
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
    # STANDALONE FLOW GRAPH VISUALIZATION
    #
    def create_flow_graph_from_csv(
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
        dpi: int = 100,
        show_continuation_edges: bool = False,
        use_exponential_scaling: bool = True
    ) -> List[str]:
        """
        Create a token-flow graph from trace CSV using a multipartite layout.

        This implementation constructs a mapping from layer index to the list of nodes
        in that layer, which satisfies NetworkX's requirement that subset_key.values()
        be sequences. Other improvements include single-pass edge drawing and lowered DPI.

        Args:
            trace_data: DataFrame of token-trace records.
            target_text: Text of the target token.
            target_idx: Index of the target token.
            save_dir: Directory in which to save output files.
            output_format: 'png', 'svg', or 'both'.
            align_tokens_by_layer: If True, use multipartite layout for even vertical spacing.
            show_orphaned_nodes: If False, remove isolated nodes.
            min_edge_weight: Minimum edge weight to include.
            use_variable_node_size: Scale node sizes by weight.
            min_node_size: Minimum node size.
            max_node_size: Maximum node size.
            debug_mode: If True, print debug information.
            dpi: DPI for raster output.
            show_continuation_edges: Draw dashed edges for token continuations (not used here).
            use_exponential_scaling: If True, apply square-root scaling to weights.

        Returns:
            A list of saved file paths.
        """
        saved_paths = []
        if trace_data.empty:
            print("No data for flow graph.")
            return saved_paths

        # Build directed graph
        G = nx.DiGraph()
        layers = sorted(trace_data["layer"].unique())
        if debug_mode:
            print(f"Detected layers: {layers}")

        # Map from (layer, token_index) to node_id, and store metadata
        node_map = {}
        node_metadata = {}
        for _, row in trace_data.iterrows():
            layer_idx = row["layer"]
            token_idx_i = row["token_index"]
            node_id = f"L{layer_idx}_T{token_idx_i}"
            node_map[(layer_idx, token_idx_i)] = node_id

            # Sanitize display text and predicted top token
            txt = self._sanitize_text_for_display(row["token_text"])
            pred = row.get("predicted_top_token", "")
            if pd.notna(pred) and pred:
                pred = f"→{self._sanitize_text_for_display(pred)}"

            node_metadata[node_id] = {
                "type": row["token_type"],
                "text": txt,
                "top_pred": pred,
                "weight": row.get("predicted_top_prob", 1.0),
                "layer": layer_idx,
                "is_target": row["is_target"]
            }
            G.add_node(node_id)

        # Build edges based on source-target relationships and weight threshold
        for _, row in trace_data.iterrows():
            if not row["is_target"] or pd.isna(row.get("sources_indices", None)):
                continue

            target_node = node_map[(row["layer"], row["token_index"])]
            src_indices = [int(i) for i in str(row["sources_indices"]).split(",") if i]
            src_weights = [float(w) for w in str(row["sources_weights"]).split(",") if w]

            for src_idx, weight in zip(src_indices, src_weights):
                if weight < min_edge_weight:
                    continue
                # Find the latest previous layer that contains this source token
                for prev_layer in reversed(layers):
                    key = (prev_layer, src_idx)
                    if key in node_map:
                        G.add_edge(node_map[key], target_node, weight=weight)
                        break

        # Optionally remove isolated nodes
        if not show_orphaned_nodes:
            orphans = [n for n in G.nodes() if G.degree(n) == 0]
            G.remove_nodes_from(orphans)
        if len(G) == 0:
            print("Graph empty after filtering.")
            return saved_paths

        # Determine node positions
        if align_tokens_by_layer:
            # Build mapping from layer -> list of node_ids for multipartite layout
            layer_nodes = defaultdict(list)
            for node in G.nodes():
                layer_nodes[node_metadata[node]["layer"]].append(node)
            pos = nx.multipartite_layout(G, subset_key=layer_nodes)
        else:
            pos = nx.spring_layout(G, k=1.0, iterations=50)

        # Compute node sizes and colors
        sizes = self._calculate_flow_graph_node_sizes(
            G, node_metadata, use_variable_node_size,
            min_node_size, max_node_size, use_exponential_scaling
        )
        color_map = {0: "#2ecc71", 1: "#3498db", 2: "#e74c3c"}
        colors = [color_map[node_metadata[n]["type"]] for n in G.nodes()]

        # Draw nodes
        plt.figure(figsize=(12, 8), dpi=dpi)
        nx.draw_networkx_nodes(
            G, pos,
            node_size=sizes,
            node_color=colors,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.8
        )

        # Draw edges in one pass
        edge_kwargs = dict(arrows=True, arrowsize=8, connectionstyle="arc3,rad=0.1")
        for u, v, data in G.edges(data=True):
            w = data["weight"]
            width = (math.sqrt(w) if use_exponential_scaling else w) * 5.0
            if width >= min_edge_weight * 5.0:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=width,
                    alpha=0.6,
                    **edge_kwargs
                )

        # Draw labels
        labels = {
            n: (node_metadata[n]["text"] + " " + node_metadata[n]["top_pred"]).strip()
            for n in G.nodes()
        }
        nx.draw_networkx_labels(
            G, pos, labels,
            font_size=8,
            font_family='DejaVu Sans',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2')
        )

        plt.title(f"Flow Graph for '{target_text}' (idx {target_idx})", fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        # Save outputs
        if output_format in ("png", "both"):
            png_path = os.path.join(save_dir, f"flow_graph_{target_idx}.png")
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            saved_paths.append(png_path)
        if output_format in ("svg", "both"):
            mpl.rcParams['svg.fonttype'] = 'none'
            svg_path = os.path.join(save_dir, f"flow_graph_{target_idx}.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            saved_paths.append(svg_path)
        plt.close()

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
            trace_data: DataFrame of token‐trace records
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

        base_dir = os.path.join(save_dir, "base_heatmaps")
        patch_dir = os.path.join(save_dir, "patch_heatmaps")
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(patch_dir, exist_ok=True)

        img = Image.open(image_path)
        preview = img.copy()
        if "resized_dimensions" in feature_mapping:
            w, h = feature_mapping["resized_dimensions"]
            preview = preview.resize((w, h), Image.LANCZOS)

        layers = sorted(trace_data["layer"].unique())
        layer_base_maps = {}
        layer_patch_maps = {}

        for L in layers:
            dfL = trace_data[trace_data["layer"] == L]
            img_toks = dfL[dfL["token_type"] == 2]
            if img_toks.empty:
                layer_base_maps[L] = None
                layer_patch_maps[L] = None
                continue

            mx = img_toks["predicted_top_prob"].max()
            weights = {
                int(r.token_index): r.predicted_top_prob / mx
                for _, r in img_toks.iterrows() if mx > 0
            }

            # Base feature map
            bf = feature_mapping.get("base_feature", {})
            if bf.get("grid") and bf.get("positions"):
                gh, gw = bf["grid"]
                heat = np.zeros((gh, gw), float)
                for tid, w in weights.items():
                    pos = bf["positions"].get(int(tid))
                    if pos:
                        r0, c0 = pos
                        if 0 <= r0 < gh and 0 <= c0 < gw:
                            heat[r0, c0] = w
                layer_base_maps[L] = heat if heat.max() > 0 else None
                if not composite_only and layer_base_maps[L] is not None:
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
            else:
                layer_base_maps[L] = None

            # Patch feature map
            pf = feature_mapping.get("patch_feature", {})
            if pf.get("grid_unpadded") and pf.get("positions"):
                gh, gw = pf["grid_unpadded"]
                heat = np.zeros((gh, gw), float)
                for tid, w in weights.items():
                    pos = pf["positions"].get(int(tid))
                    if pos:
                        r0, c0 = pos
                        if 0 <= r0 < gh and 0 <= c0 < gw:
                            heat[r0, c0] = w
                layer_patch_maps[L] = heat if heat.max() > 0 else None
                if not composite_only and layer_patch_maps[L] is not None:
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
            else:
                layer_patch_maps[L] = None

        # Composite grids (always generate if any valid)
        valid_base = [L for L, hm in layer_base_maps.items() if hm is not None]
        if valid_base:
            out = os.path.join(save_dir, f"composite_base_grid_{target_idx}.png")
            p = self._create_grid_composite(
                heatmap_maps=layer_base_maps,
                layers=layers,
                title=f"Base Influence per Layer for Token {target_idx}",
                save_path=out,
                cmap="hot",
                show_values=show_values
            )
            if p:
                saved_paths.append(p)

        valid_patch = [L for L, hm in layer_patch_maps.items() if hm is not None]
        if valid_patch:
            out = os.path.join(save_dir, f"composite_patch_grid_{target_idx}.png")
            p = self._create_grid_composite(
                heatmap_maps=layer_patch_maps,
                layers=layers,
                title=f"Patch Influence per Layer for Token {target_idx}",
                save_path=out,
                cmap="hot",
                show_values=show_values
            )
            if p:
                saved_paths.append(p)

        return saved_paths



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

        heatmap_maps: dict layer->2D numpy array or None
        layers: full list of layers to include (None entries yield blank panels)
        """
        # count panels
        n = len(layers)
        if n == 0:
            return None

        ncols = math.ceil(math.sqrt(n))
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
        axes = axes.flatten()

        im = None
        for idx, L in enumerate(layers):
            ax = axes[idx]
            hm = heatmap_maps.get(L)
            if hm is None:
                ax.axis("off")
                continue

            im = ax.imshow(hm, vmin=0, vmax=1, cmap=cmap)
            ax.set_title(f"Layer {L}", fontsize=10)
            ax.axis("off")

            if show_values:
                # overlay each cell value
                H,W = hm.shape
                for i in range(H):
                    for j in range(W):
                        val = hm[i,j]
                        if val > 0:
                            ax.text(j, i, f"{val:.2f}",
                                    ha='center', va='center', fontsize=6,
                                    color='white' if val>0.5 else 'black')

        # hide extra axes
        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(title, fontsize=14)
        if im is not None:
            cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.01)
            cbar.set_label("Normalized Influence", fontsize=10)

        plt.tight_layout(rect=[0,0,1,0.96])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path

    
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
            resized_background = original_image.resize(target_size, Image.LANCZOS)
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


# Helper function to run the visualizer from a CSV file
def visualize_semantic_tracing_csv(
    csv_path: str,
    metadata_path: Optional[str] = None,
    image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    create_flow_graph: bool = True,
    create_heatmaps: bool = True,
    create_data_vis: bool = True,
    flow_graph_params: Optional[Dict[str, Any]] = None,
    heatmap_params: Optional[Dict[str, Any]] = None
) -> Dict[str, List[str]]:
    """
    Convenience function to run the visualizer on a CSV file.
    
    Args:
        csv_path: Path to the CSV file with trace data
        metadata_path: Optional path to JSON metadata file
        image_path: Path or URL to original image used in the trace
        output_dir: Directory to save visualizations
        create_flow_graph: Whether to create flow graph visualization
        create_heatmaps: Whether to create heatmap visualizations
        create_data_vis: Whether to create trace data visualizations
        flow_graph_params: Parameters for flow graph visualization
        heatmap_params: Parameters for heatmap visualization
        
    Returns:
        Dictionary mapping visualization types to lists of generated file paths
    """
    # Set default output directory if not provided
    if output_dir is None:
        csv_dir = os.path.dirname(csv_path)
        # Check if we're already in a visualizations directory
        if "visualizations" in csv_dir:
            output_dir = csv_dir
        else:
            output_dir = os.path.join(csv_dir, "visualizations")
    
    # Create visualizer and run it
    visualizer = EnhancedSemanticTracingVisualizer(output_dir)
    
    # Set default flow graph parameters if not provided
    if flow_graph_params is None:
        flow_graph_params = {
            "output_format": "both",
            "align_tokens_by_layer": True,
            "show_orphaned_nodes": True,  # Show all nodes by default
            "min_edge_weight": 0.0,  # Show all edges by default
            "use_variable_node_size": True,
            "debug_mode": False
        }
    
    # Run visualization
    return visualizer.visualize_from_csv(
        csv_path=csv_path,
        metadata_path=metadata_path,
        image_path=image_path,
        create_flow_graph=create_flow_graph,
        create_heatmaps=create_heatmaps,
        create_data_vis=create_data_vis,
        flow_graph_params=flow_graph_params,
        heatmap_params=heatmap_params
    )