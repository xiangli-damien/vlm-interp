# analyzer/token_analyzer.py
"""
Token analyzer for identifying and measuring the roles of different tokens in VLM reasoning.

This module provides analysis of token roles in the semantic tracing process, identifying:
- Information reception nodes: Tokens that gather information from many other tokens
- Reasoning nodes: Tokens that transform received information
- Emission nodes: Tokens that propagate information to many other tokens
- Preservation nodes: Tokens that maintain consistent information across layers

Also supports ablation studies to measure token importance by blocking specific tokens.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional, Union

class TokenAnalyzer:
    """
    Analyzes token roles and importance in semantic tracing across model layers.
    
    This class provides methods to:
    1. Identify roles of tokens in information flow (receptors, processors, emitters)
    2. Measure token importance through network analysis
    3. Perform ablation studies by blocking specific tokens
    4. Visualize token roles and metrics
    """
    
    def __init__(
        self,
        output_dir: str = "token_analysis_results",
        receptivity_threshold: float = 0.7,   # Threshold for identifying reception nodes
        emission_threshold: float = 0.7,      # Threshold for identifying emission nodes
        preservation_threshold: float = 0.5,  # Threshold for identifying preservation nodes
        importance_window: int = 3,           # Number of layers to consider for importance
        debug_mode: bool = False
    ):
        """
        Initialize the token analyzer.
        
        Args:
            output_dir: Directory to save analysis results
            receptivity_threshold: Threshold for identifying reception nodes (higher = stricter)
            emission_threshold: Threshold for identifying emission nodes (higher = stricter)
            preservation_threshold: Threshold for identifying preservation nodes
            importance_window: Number of layers to consider for rolling importance
            debug_mode: Whether to print additional debug information
        """
        self.output_dir = output_dir
        self.receptivity_threshold = receptivity_threshold
        self.emission_threshold = emission_threshold
        self.preservation_threshold = preservation_threshold
        self.importance_window = importance_window
        self.debug_mode = debug_mode
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Storage for token analysis results
        self.token_metrics = {}
        self.token_roles = {}
        self.network_metrics = {}
        self.ablation_results = {}
        
        # Token type descriptions
        self.token_type_names = {
            0: "Generated",
            1: "Text",
            2: "Image"
        }
    
    def analyze_trace_data(
        self, 
        csv_path: str, 
        metadata_path: Optional[str] = None,
        importance_column: str = "importance_weight"
    ) -> Dict[str, Any]:
        """
        Analyze token roles and metrics from traced data CSV file.
        
        Args:
            csv_path: Path to the trace data CSV
            metadata_path: Optional path to metadata JSON
            importance_column: Column to use for importance values
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing trace data from: {csv_path}")
        
        # Load trace data
        try:
            trace_df = pd.read_csv(csv_path)
            if trace_df.empty:
                print("Error: Trace data is empty")
                return {"error": "Empty trace data"}
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return {"error": f"Failed to load CSV: {e}"}
        
        # Load metadata if provided
        metadata = {}
        if metadata_path:
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load metadata: {e}")
        
        # Extract target token information
        target_tokens = []
        if "target_tokens" in metadata:
            target_tokens = metadata["target_tokens"]
        elif "target_token" in metadata:
            target_tokens = [metadata["target_token"]]
        
        # Analyze token metrics and roles
        token_metrics = self._calculate_token_metrics(trace_df, importance_column)
        self.token_metrics = token_metrics
        
        # Identify token roles
        token_roles = self._identify_token_roles(trace_df, token_metrics)
        self.token_roles = token_roles
        
        # Build token flow graph and calculate network metrics
        network_metrics = self._calculate_network_metrics(trace_df)
        self.network_metrics = network_metrics
        
        # Combine all metrics
        combined_metrics = self._combine_metrics(token_metrics, token_roles, network_metrics)
        
        # Generate summary statistics
        summary = self._generate_analysis_summary(combined_metrics, trace_df, target_tokens)
        
        # Save results
        self._save_analysis_results(combined_metrics, summary)
        
        return {
            "token_metrics": token_metrics,
            "token_roles": token_roles,
            "network_metrics": network_metrics,
            "combined_metrics": combined_metrics,
            "summary": summary
        }
    
    def _calculate_token_metrics(
        self, 
        trace_df: pd.DataFrame, 
        importance_column: str = "importance_weight"
    ) -> Dict[int, Dict[str, Any]]:
        """
        Calculate various metrics for each token across layers.
        
        Args:
            trace_df: DataFrame with trace data
            importance_column: Column containing importance values
            
        Returns:
            Dictionary mapping token indices to metrics
        """
        # Get all unique tokens and layers
        tokens = trace_df["token_index"].unique()
        layers = sorted(trace_df["layer"].unique())
        
        # Initialize metrics dictionary
        token_metrics = {}
        
        # Process each token
        for token_idx in tokens:
            # Get token data across layers
            token_df = trace_df[trace_df["token_index"] == token_idx]
            
            # Skip if token only appears in one layer
            if len(token_df) <= 1:
                continue
            
            # Get token type and text
            token_type = token_df["token_type"].iloc[0]
            token_text = token_df["token_text"].iloc[0]
            
            # Initialize metrics for this token
            metrics = {
                "token_index": token_idx,
                "token_text": token_text,
                "token_type": token_type,
                "token_type_name": self.token_type_names.get(token_type, "Unknown"),
                "layers_present": sorted(token_df["layer"].unique()),
                "layer_metrics": {},
                "global_metrics": {}
            }
            
            # Calculate layer-specific metrics
            for layer in sorted(token_df["layer"].unique()):
                layer_row = token_df[token_df["layer"] == layer].iloc[0]
                
                # Get importance weight
                importance = layer_row[importance_column] if importance_column in layer_row else 0
                
                # Get source tokens (tokens flowing into this token)
                sources_indices = []
                sources_weights = []
                
                if "sources_indices" in layer_row and pd.notna(layer_row["sources_indices"]):
                    sources_indices = [int(idx) for idx in layer_row["sources_indices"].split(",") if idx]
                    
                if "sources_weights" in layer_row and pd.notna(layer_row["sources_weights"]):
                    sources_weights = [float(w) for w in layer_row["sources_weights"].split(",") if w]
                
                # Count incoming and outgoing connections
                incoming_count = len(sources_indices)
                
                # Find tokens in this layer that have this token as a source
                if "sources_indices" in trace_df.columns:
                    outgoing_indices = []
                    outgoing_weights = []
                    
                    # Check other tokens in the same layer that have this token as source
                    # Check other tokens in the same layer that have this token as source
                    for _, row in trace_df[(trace_df["layer"] == layer) & (trace_df["token_index"] != token_idx)].iterrows():
                        if "sources_indices" in row and pd.notna(row["sources_indices"]):
                            source_indices = [int(idx) for idx in str(row["sources_indices"]).split(",") if idx]
                            source_weights = []
                            
                            if "sources_weights" in row and pd.notna(row["sources_weights"]):
                                source_weights = [float(w) for w in str(row["sources_weights"]).split(",") if w]
                            
                            # Check if this token is a source
                            if token_idx in source_indices:
                                idx = source_indices.index(token_idx)
                                weight = source_weights[idx] if idx < len(source_weights) else 0
                                outgoing_indices.append(row["token_index"])
                                outgoing_weights.append(weight)
                    
                    outgoing_count = len(outgoing_indices)
                else:
                    outgoing_indices = []
                    outgoing_weights = []
                    outgoing_count = 0
                
                # Calculate connectivity metrics for this layer
                source_diversity = len(set(sources_indices))
                target_diversity = len(set(outgoing_indices))
                
                # Calculate receptivity (amount of information received)
                receptivity = sum(sources_weights) if sources_weights else 0
                
                # Calculate emission (amount of information emitted)
                emission = sum(outgoing_weights) if outgoing_weights else 0
                
                # Calculate throughput (information passing through)
                throughput = min(receptivity, emission)
                
                # Calculate transformation (difference between input and output information)
                transformation = abs(emission - receptivity) / (max(emission, receptivity) + 1e-8)
                
                # Store layer metrics
                metrics["layer_metrics"][layer] = {
                    "importance": importance,
                    "sources_indices": sources_indices,
                    "sources_weights": sources_weights,
                    "targets_indices": outgoing_indices,
                    "targets_weights": outgoing_weights,
                    "incoming_count": incoming_count,
                    "outgoing_count": outgoing_count,
                    "source_diversity": source_diversity,
                    "target_diversity": target_diversity,
                    "receptivity": receptivity,
                    "emission": emission,
                    "throughput": throughput,
                    "transformation": transformation
                }
            
            # Calculate global metrics across all layers
            # Number of layers where token appears
            metrics["global_metrics"]["layer_presence"] = len(metrics["layers_present"])
            
            # Average metrics across layers
            avg_metrics = {
                "avg_importance": 0,
                "avg_incoming": 0,
                "avg_outgoing": 0,
                "avg_receptivity": 0,
                "avg_emission": 0,
                "avg_throughput": 0,
                "avg_transformation": 0
            }
            
            for layer, layer_metrics in metrics["layer_metrics"].items():
                avg_metrics["avg_importance"] += layer_metrics["importance"]
                avg_metrics["avg_incoming"] += layer_metrics["incoming_count"]
                avg_metrics["avg_outgoing"] += layer_metrics["outgoing_count"]
                avg_metrics["avg_receptivity"] += layer_metrics["receptivity"]
                avg_metrics["avg_emission"] += layer_metrics["emission"]
                avg_metrics["avg_throughput"] += layer_metrics["throughput"]
                avg_metrics["avg_transformation"] += layer_metrics["transformation"]
            
            num_layers = len(metrics["layer_metrics"])
            for key in avg_metrics:
                avg_metrics[key] /= max(1, num_layers)
                metrics["global_metrics"][key] = avg_metrics[key]
            
            # Calculate preservation score (consistency across layers)
            # High score means token maintains similar importance and connectivity across layers
            importance_variation = np.std([m["importance"] for m in metrics["layer_metrics"].values()])
            connectivity_variation = np.std([m["incoming_count"] + m["outgoing_count"] for m in metrics["layer_metrics"].values()])
            
            # Normalize by average values to get relative variation
            avg_importance = avg_metrics["avg_importance"]
            avg_connectivity = avg_metrics["avg_incoming"] + avg_metrics["avg_outgoing"]
            
            rel_importance_var = importance_variation / (avg_importance + 1e-8)
            rel_connectivity_var = connectivity_variation / (avg_connectivity + 1e-8)
            
            # Lower variation means higher preservation
            preservation_score = 1.0 - 0.5 * (rel_importance_var + rel_connectivity_var)
            preservation_score = max(0, min(1, preservation_score))  # Clamp to [0, 1]
            
            metrics["global_metrics"]["preservation_score"] = preservation_score
            metrics["global_metrics"]["importance_variation"] = importance_variation
            metrics["global_metrics"]["connectivity_variation"] = connectivity_variation
            
            # Calculate rolling importance scores (changes over consecutive layers)
            # This helps identify patterns like steadily increasing importance
            layer_importance = {layer: metrics["layer_metrics"][layer]["importance"] 
                               for layer in metrics["layer_metrics"]}
            
            sorted_layers = sorted(layer_importance.keys())
            if len(sorted_layers) >= 2:
                importance_deltas = []
                for i in range(1, len(sorted_layers)):
                    prev_layer = sorted_layers[i-1]
                    curr_layer = sorted_layers[i]
                    delta = layer_importance[curr_layer] - layer_importance[prev_layer]
                    importance_deltas.append(delta)
                
                metrics["global_metrics"]["importance_trend"] = sum(importance_deltas) / len(importance_deltas)
                metrics["global_metrics"]["importance_deltas"] = importance_deltas
            else:
                metrics["global_metrics"]["importance_trend"] = 0
                metrics["global_metrics"]["importance_deltas"] = []
            
            # Store in token_metrics dictionary
            token_metrics[token_idx] = metrics
        
        return token_metrics
    
    def _identify_token_roles(
        self, 
        trace_df: pd.DataFrame, 
        token_metrics: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Identify functional roles for each token based on its metrics.
        
        Args:
            trace_df: DataFrame with trace data
            token_metrics: Dictionary with token metrics
            
        Returns:
            Dictionary mapping tokens to their roles
        """
        token_roles = {}
        
        # Get all layers
        all_layers = sorted(trace_df["layer"].unique())
        
        # Process each token
        for token_idx, metrics in token_metrics.items():
            # Initialize roles dictionary
            roles = {
                "token_index": token_idx,
                "token_text": metrics["token_text"],
                "token_type": metrics["token_type"],
                "layer_roles": {},
                "global_roles": []
            }
            
            # Determine roles for each layer
            for layer in metrics["layers_present"]:
                layer_metrics = metrics["layer_metrics"].get(layer, {})
                
                # Skip if no metrics for this layer
                if not layer_metrics:
                    continue
                
                # Initialize layer roles
                layer_roles = []
                
                # Check if token is a receptor (receives significant information)
                if (layer_metrics["incoming_count"] > 0 and 
                    layer_metrics["source_diversity"] >= 2 and
                    layer_metrics["receptivity"] > self.receptivity_threshold):
                    layer_roles.append("receptor")
                
                # Check if token is an emitter (emits significant information)
                if (layer_metrics["outgoing_count"] > 0 and
                    layer_metrics["target_diversity"] >= 2 and
                    layer_metrics["emission"] > self.emission_threshold):
                    layer_roles.append("emitter")
                
                # Check if token is a transformer (changes information significantly)
                # High transformation means different incoming vs outgoing patterns
                if layer_metrics["transformation"] > 0.5:
                    layer_roles.append("transformer")
                
                # Check if token is a connector (high throughput, connects information)
                if (layer_metrics["incoming_count"] > 0 and
                    layer_metrics["outgoing_count"] > 0 and
                    layer_metrics["throughput"] > 0.5):
                    layer_roles.append("connector")
                
                # Check if token is an attractor (high importance, many incoming)
                if (layer_metrics["importance"] > 0.7 and
                    layer_metrics["incoming_count"] > 3):
                    layer_roles.append("attractor")
                
                # Check if token is a broadcaster (many outgoing connections)
                if layer_metrics["outgoing_count"] > 5:
                    layer_roles.append("broadcaster")
                
                # Check if token is a sentinel (important but few connections)
                if (layer_metrics["importance"] > 0.7 and
                    layer_metrics["incoming_count"] <= 2 and
                    layer_metrics["outgoing_count"] <= 2):
                    layer_roles.append("sentinel")
                
                # Store layer roles
                roles["layer_roles"][layer] = layer_roles
            
            # Determine global roles
            global_roles = []
            
            # Check if token is a preservation node (consistent across layers)
            if metrics["global_metrics"]["preservation_score"] > self.preservation_threshold:
                global_roles.append("preservation")
            
            # Check if token is a gateway (connects earlier to later layers)
            if (len(metrics["layers_present"]) >= 3 and
                metrics["global_metrics"]["avg_throughput"] > 0.6):
                global_roles.append("gateway")
            
            # Check if token is a growth node (importance increases over layers)
            if metrics["global_metrics"]["importance_trend"] > 0.2:
                global_roles.append("growth")
            
            # Check if token is a decay node (importance decreases over layers)
            if metrics["global_metrics"]["importance_trend"] < -0.2:
                global_roles.append("decay")
            
            # Check if token is a key node (consistently high importance)
            if (metrics["global_metrics"]["avg_importance"] > 0.7 and
                metrics["global_metrics"]["importance_variation"] < 0.2):
                global_roles.append("key")
            
            # Check if token is a hub (high connectivity across layers)
            if (metrics["global_metrics"]["avg_incoming"] + 
                metrics["global_metrics"]["avg_outgoing"] > 8):
                global_roles.append("hub")
            
            # Store global roles
            roles["global_roles"] = global_roles
            
            # Store in token_roles dictionary
            token_roles[token_idx] = roles
        
        return token_roles
    
    def _calculate_network_metrics(self, trace_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Calculate network centrality metrics for tokens using graph analysis.
        
        Args:
            trace_df: DataFrame with trace data
            
        Returns:
            Dictionary with network metrics for each token
        """
        import networkx as nx
        
        # Get all layers
        layers = sorted(trace_df["layer"].unique())
        
        # Initialize network metrics
        network_metrics = {}
        
        # Create layer-specific graphs
        layer_graphs = {}
        for layer in layers:
            # Get layer data
            layer_df = trace_df[trace_df["layer"] == layer]
            
            # Create directed graph for this layer
            G = nx.DiGraph()
            
            # Add all tokens as nodes
            for _, row in layer_df.iterrows():
                token_idx = row["token_index"]
                token_text = row["token_text"]
                token_type = row["token_type"]
                G.add_node(token_idx, text=token_text, type=token_type)
            
            # Add edges based on source relationships
            for _, row in layer_df.iterrows():
                if "sources_indices" in row and pd.notna(row["sources_indices"]):
                    # Get target token
                    target_idx = row["token_index"]
                    
                    # Get source tokens and weights
                    sources_indices = [int(idx) for idx in str(row["sources_indices"]).split(",") if idx]
                    sources_weights = []
                    
                    if "sources_weights" in row and pd.notna(row["sources_weights"]):
                        sources_weights = [float(w) for w in str(row["sources_weights"]).split(",") if w]
                    else:
                        sources_weights = [1.0] * len(sources_indices)
                    
                    # Add edges with weights
                    for i, source_idx in enumerate(sources_indices):
                        if i < len(sources_weights):
                            weight = sources_weights[i]
                            if source_idx in G and target_idx in G:  # Ensure nodes exist
                                G.add_edge(source_idx, target_idx, weight=weight)
            
            # Store graph
            layer_graphs[layer] = G
        
        # Calculate metrics for each token in each layer
        for layer, G in layer_graphs.items():
            # Skip if graph is empty
            if len(G) == 0:
                continue
            
            # Calculate centrality metrics
            try:
                # Degree centrality
                in_degree = dict(G.in_degree(weight="weight"))
                out_degree = dict(G.out_degree(weight="weight"))
                
                # Betweenness centrality (how often a node lies on shortest paths)
                betweenness = nx.betweenness_centrality(G, weight="weight")
                
                # Eigenvector centrality (influence of a node)
                try:
                    eigenvector = nx.eigenvector_centrality(G, weight="weight", max_iter=1000)
                except:
                    # Fall back to unweighted if it fails
                    try:
                        eigenvector = nx.eigenvector_centrality(G, weight=None, max_iter=1000)
                    except:
                        eigenvector = {node: 0.0 for node in G.nodes()}
                
                # PageRank (importance of node based on incoming links)
                pagerank = nx.pagerank(G, weight="weight")
                
                # Store metrics for each token
                for token_idx in G.nodes():
                    if token_idx not in network_metrics:
                        network_metrics[token_idx] = {
                            "token_index": token_idx,
                            "layer_metrics": {},
                            "global_metrics": {
                                "avg_in_degree": 0,
                                "avg_out_degree": 0,
                                "avg_betweenness": 0,
                                "avg_eigenvector": 0,
                                "avg_pagerank": 0,
                                "layer_count": 0
                            }
                        }
                    
                    # Store layer-specific metrics
                    network_metrics[token_idx]["layer_metrics"][layer] = {
                        "in_degree": in_degree.get(token_idx, 0),
                        "out_degree": out_degree.get(token_idx, 0),
                        "betweenness": betweenness.get(token_idx, 0),
                        "eigenvector": eigenvector.get(token_idx, 0),
                        "pagerank": pagerank.get(token_idx, 0)
                    }
                    
                    # Update global metrics
                    global_metrics = network_metrics[token_idx]["global_metrics"]
                    global_metrics["avg_in_degree"] += in_degree.get(token_idx, 0)
                    global_metrics["avg_out_degree"] += out_degree.get(token_idx, 0)
                    global_metrics["avg_betweenness"] += betweenness.get(token_idx, 0)
                    global_metrics["avg_eigenvector"] += eigenvector.get(token_idx, 0)
                    global_metrics["avg_pagerank"] += pagerank.get(token_idx, 0)
                    global_metrics["layer_count"] += 1
            
            except Exception as e:
                print(f"Error calculating network metrics for layer {layer}: {e}")
                continue
        
        # Calculate average metrics
        for token_idx, metrics in network_metrics.items():
            global_metrics = metrics["global_metrics"]
            layer_count = global_metrics["layer_count"]
            
            if layer_count > 0:
                global_metrics["avg_in_degree"] /= layer_count
                global_metrics["avg_out_degree"] /= layer_count
                global_metrics["avg_betweenness"] /= layer_count
                global_metrics["avg_eigenvector"] /= layer_count
                global_metrics["avg_pagerank"] /= layer_count
        
        return network_metrics
    
    def _combine_metrics(
        self,
        token_metrics: Dict[int, Dict[str, Any]],
        token_roles: Dict[int, Dict[str, Any]],
        network_metrics: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Combine all metrics into a unified representation for each token.
        
        Args:
            token_metrics: Dictionary with token metrics
            token_roles: Dictionary with token roles
            network_metrics: Dictionary with network metrics
            
        Returns:
            Combined metrics dictionary
        """
        combined_metrics = {}
        
        # Get all token indices
        all_tokens = set(token_metrics.keys()) | set(token_roles.keys()) | set(network_metrics.keys())
        
        # Combine metrics for each token
        for token_idx in all_tokens:
            combined = {
                "token_index": token_idx,
                "metrics": {},
                "roles": {},
                "network": {}
            }
            
            # Add basic info if available
            if token_idx in token_metrics:
                combined["token_text"] = token_metrics[token_idx]["token_text"]
                combined["token_type"] = token_metrics[token_idx]["token_type"]
                combined["token_type_name"] = token_metrics[token_idx]["token_type_name"]
                combined["metrics"] = token_metrics[token_idx]
            
            # Add roles if available
            if token_idx in token_roles:
                combined["roles"] = token_roles[token_idx]
            
            # Add network metrics if available
            if token_idx in network_metrics:
                combined["network"] = network_metrics[token_idx]
            
            # Calculate overall importance score
            importance_score = 0.0
            score_components = 0
            
            # Component 1: Average importance from token metrics
            if token_idx in token_metrics and "global_metrics" in token_metrics[token_idx]:
                importance_score += token_metrics[token_idx]["global_metrics"].get("avg_importance", 0)
                score_components += 1
            
            # Component 2: PageRank from network metrics
            if token_idx in network_metrics and "global_metrics" in network_metrics[token_idx]:
                importance_score += network_metrics[token_idx]["global_metrics"].get("avg_pagerank", 0) * 5  # Scale up
                score_components += 1
            
            # Component 3: Role-based importance
            if token_idx in token_roles:
                # More critical roles increase importance
                global_roles = token_roles[token_idx].get("global_roles", [])
                critical_roles = ["key", "hub", "preservation", "gateway"]
                role_score = sum(0.15 for role in global_roles if role in critical_roles)
                importance_score += role_score
                if role_score > 0:
                    score_components += 1
            
            # Calculate average and normalize to [0, 1]
            if score_components > 0:
                importance_score /= score_components
                importance_score = min(1.0, max(0.0, importance_score))
            
            combined["overall_importance"] = importance_score
            
            # Store combined metrics
            combined_metrics[token_idx] = combined
        
        return combined_metrics
    
    def _generate_analysis_summary(
        self,
        combined_metrics: Dict[int, Dict[str, Any]],
        trace_df: pd.DataFrame,
        target_tokens: List[Dict[str, Any]] = []
    ) -> Dict[str, Any]:
        """
        Generate a summary of token analysis results.
        
        Args:
            combined_metrics: Dictionary with combined token metrics
            trace_df: DataFrame with trace data
            target_tokens: List of target token information
            
        Returns:
            Summary dictionary
        """
        # Initialize summary
        summary = {
            "token_counts": {
                "total": len(combined_metrics),
                "by_type": {},
                "by_role": {}
            },
            "top_tokens": {
                "by_importance": [],
                "by_preservation": [],
                "by_centrality": []
            },
            "token_roles": {
                "global_roles": {},
                "layer_roles": {}
            },
            "target_token_analysis": []
        }
        
        # Count tokens by type
        token_types = {}
        for token_idx, metrics in combined_metrics.items():
            token_type = metrics.get("token_type", -1)
            token_type_name = metrics.get("token_type_name", "Unknown")
            
            if token_type_name not in token_types:
                token_types[token_type_name] = 0
            token_types[token_type_name] += 1
        
        summary["token_counts"]["by_type"] = token_types
        
        # Count tokens by global role
        global_roles = {}
        for token_idx, metrics in combined_metrics.items():
            if "roles" in metrics and "global_roles" in metrics["roles"]:
                for role in metrics["roles"]["global_roles"]:
                    if role not in global_roles:
                        global_roles[role] = 0
                    global_roles[role] += 1
        
        summary["token_counts"]["by_role"] = global_roles
        
        # Get top tokens by importance
        top_by_importance = sorted(
            combined_metrics.items(),
            key=lambda x: x[1]["overall_importance"],
            reverse=True
        )[:10]
        
        summary["top_tokens"]["by_importance"] = [
            {
                "token_index": token_idx,
                "token_text": metrics.get("token_text", ""),
                "token_type": metrics.get("token_type_name", ""),
                "importance": metrics["overall_importance"],
                "global_roles": metrics.get("roles", {}).get("global_roles", [])
            }
            for token_idx, metrics in top_by_importance
        ]
        
        # Get top tokens by preservation
        top_by_preservation = sorted(
            [
                (token_idx, metrics) for token_idx, metrics in combined_metrics.items()
                if "metrics" in metrics and "global_metrics" in metrics["metrics"]
            ],
            key=lambda x: x[1]["metrics"]["global_metrics"].get("preservation_score", 0),
            reverse=True
        )[:10]
        
        summary["top_tokens"]["by_preservation"] = [
            {
                "token_index": token_idx,
                "token_text": metrics.get("token_text", ""),
                "token_type": metrics.get("token_type_name", ""),
                "preservation_score": metrics["metrics"]["global_metrics"].get("preservation_score", 0),
                "global_roles": metrics.get("roles", {}).get("global_roles", [])
            }
            for token_idx, metrics in top_by_preservation
        ]
        
        # Get top tokens by centrality (PageRank)
        top_by_centrality = sorted(
            [
                (token_idx, metrics) for token_idx, metrics in combined_metrics.items()
                if "network" in metrics and "global_metrics" in metrics["network"]
            ],
            key=lambda x: x[1]["network"]["global_metrics"].get("avg_pagerank", 0),
            reverse=True
        )[:10]
        
        summary["top_tokens"]["by_centrality"] = [
            {
                "token_index": token_idx,
                "token_text": metrics.get("token_text", ""),
                "token_type": metrics.get("token_type_name", ""),
                "pagerank": metrics["network"]["global_metrics"].get("avg_pagerank", 0),
                "global_roles": metrics.get("roles", {}).get("global_roles", [])
            }
            for token_idx, metrics in top_by_centrality
        ]
        
        # Count token global roles
        role_counts = {}
        for token_idx, metrics in combined_metrics.items():
            if "roles" in metrics and "global_roles" in metrics["roles"]:
                for role in metrics["roles"]["global_roles"]:
                    if role not in role_counts:
                        role_counts[role] = 0
                    role_counts[role] += 1
        
        summary["token_roles"]["global_roles"] = role_counts
        
        # Count token layer roles
        layer_role_counts = {}
        for token_idx, metrics in combined_metrics.items():
            if "roles" in metrics and "layer_roles" in metrics["roles"]:
                for layer, roles in metrics["roles"]["layer_roles"].items():
                    layer_str = f"layer_{layer}"
                    if layer_str not in layer_role_counts:
                        layer_role_counts[layer_str] = {}
                    
                    for role in roles:
                        if role not in layer_role_counts[layer_str]:
                            layer_role_counts[layer_str][role] = 0
                        layer_role_counts[layer_str][role] += 1
        
        summary["token_roles"]["layer_roles"] = layer_role_counts
        
        # Analyze target tokens
        for target in target_tokens:
            target_idx = target.get("index")
            if target_idx in combined_metrics:
                target_metrics = combined_metrics[target_idx]
                
                target_analysis = {
                    "token_index": target_idx,
                    "token_text": target.get("text", target_metrics.get("token_text", "")),
                    "token_type": target_metrics.get("token_type_name", ""),
                    "importance": target_metrics["overall_importance"],
                    "global_roles": target_metrics.get("roles", {}).get("global_roles", []),
                    "preservation_score": target_metrics.get("metrics", {}).get("global_metrics", {}).get("preservation_score", 0),
                    "connectivity": {
                        "avg_in_degree": target_metrics.get("network", {}).get("global_metrics", {}).get("avg_in_degree", 0),
                        "avg_out_degree": target_metrics.get("network", {}).get("global_metrics", {}).get("avg_out_degree", 0)
                    }
                }
                
                summary["target_token_analysis"].append(target_analysis)
        
        return summary
    
    def _save_analysis_results(
        self,
        combined_metrics: Dict[int, Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> None:
        """
        Save token analysis results to files.
        
        Args:
            combined_metrics: Combined metrics dictionary
            summary: Analysis summary dictionary
        """
        # Create output paths
        metrics_path = os.path.join(self.output_dir, "token_metrics.json")
        summary_path = os.path.join(self.output_dir, "analysis_summary.json")
        
        # Convert to serializable format and save metrics
        serializable_metrics = {}
        for token_idx, metrics in combined_metrics.items():
            serializable_metrics[str(token_idx)] = self._make_serializable(metrics)
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        # Save summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved token metrics to: {metrics_path}")
        print(f"Saved analysis summary to: {summary_path}")
    
    def _make_serializable(self, obj):
        """Convert a complex object to a JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def simulate_token_blocking(
        self,
        model,
        processor,
        input_data: Dict[str, Any],
        tokens_to_block: List[int],
        method: str = "zero_out",
        reference_output: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate blocking specific tokens to analyze their impact on model reasoning.
        
        Args:
            model: Model to use for simulation
            processor: Processor for the model
            input_data: Prepared input data
            tokens_to_block: List of token indices to block
            method: Blocking method ("zero_out", "average", or "noise")
            reference_output: Optional reference output for comparison
            
        Returns:
            Dictionary with simulation results
        """
        import torch
        
        print(f"Simulating token blocking for {len(tokens_to_block)} tokens using method: {method}")
        
        # Define hook to modify hidden states
        def blocking_hook(module, input_tensors, output_tensors):
            # Apply to only hidden states (not attention masks or position IDs)
            if isinstance(output_tensors, torch.Tensor):
                hidden_states = output_tensors
                
                # Create a copy to modify
                modified = hidden_states.clone()
                
                # Apply blocking based on specified method
                for token_idx in tokens_to_block:
                    if token_idx < modified.shape[1]:
                        if method == "zero_out":
                            # Zero out the token representation
                            modified[:, token_idx, :] = 0
                        elif method == "average":
                            # Replace with average of all tokens
                            avg_repr = torch.mean(hidden_states, dim=1, keepdim=True)
                            modified[:, token_idx, :] = avg_repr
                        elif method == "noise":
                            # Replace with Gaussian noise
                            noise = torch.randn_like(modified[:, token_idx, :])
                            modified[:, token_idx, :] = noise
                
                return modified
            else:
                return output_tensors
        
        # Get base LLM module to attach hook
        llm_module = model.get_decoder() if hasattr(model, "get_decoder") else model
        
        # Run inference with blocking
        try:
            # Register forward hook
            hook_handle = llm_module.register_forward_hook(blocking_hook)
            
            # Run model with blocked tokens
            inputs = input_data["inputs"]
            
            with torch.no_grad():
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                              output_hidden_states=True,
                              output_attentions=True)
            
            # Remove hook
            hook_handle.remove()
            
            # Extract generated text
            logits = outputs.logits
            generated_ids = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_text = processor.tokenizer.decode(generated_ids[0].tolist())
            
            # Compare with reference if provided
            comparison = {}
            if reference_output is not None and "output_ids" in reference_output:
                ref_ids = reference_output["output_ids"]
                
                # Compare output probability distributions
                ref_probs = torch.softmax(reference_output["logits"][:, -1, :], dim=-1)
                sim_probs = torch.softmax(logits[:, -1, :], dim=-1)
                
                # KL Divergence as distribution distance
                kl_div = torch.nn.functional.kl_div(
                    sim_probs.log(), ref_probs, reduction="batchmean"
                ).item()
                
                # Token match - whether prediction is the same
                token_match = (generated_ids == ref_ids).all().item()
                
                # Get probability of reference token in blocked model
                ref_token_id = ref_ids[0].item()
                ref_token_prob = sim_probs[0, ref_token_id].item()
                
                comparison = {
                    "kl_divergence": kl_div,
                    "token_match": token_match,
                    "ref_token_prob": ref_token_prob,
                    "impact_score": 1.0 - ref_token_prob if ref_token_prob > 0 else 1.0
                }
            
            # Return results
            results = {
                "method": method,
                "tokens_blocked": tokens_to_block,
                "generated_text": generated_text,
                "comparison": comparison
            }
            
            print(f"Blocking simulation complete. Token match: {comparison.get('token_match', 'N/A')}")
            
            return results
            
        except Exception as e:
            print(f"Error during token blocking simulation: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e),
                "method": method,
                "tokens_blocked": tokens_to_block
            }
    
    def visualize_token_metrics(
        self,
        combined_metrics: Dict[int, Dict[str, Any]],
        output_path: Optional[str] = None,
        token_types_to_include: Optional[List[int]] = None
    ) -> str:
        """
        Create visualization of token metrics as a bubble chart.
        
        Args:
            combined_metrics: Dictionary with combined token metrics
            output_path: Path to save the visualization (None = auto-generate)
            token_types_to_include: List of token types to include (None = all)
            
        Returns:
            Path to saved visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Use default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, "token_metrics_visualization.png")
        
        # Prepare data for visualization
        token_data = []
        for token_idx, metrics in combined_metrics.items():
            # Skip if no metrics or network data
            if "metrics" not in metrics or "network" not in metrics:
                continue
            
            # Skip if token type filter is applied and token doesn't match
            if (token_types_to_include is not None and 
                metrics.get("token_type") not in token_types_to_include):
                continue
            
            # Get token data
            token_text = metrics.get("token_text", "")
            token_type = metrics.get("token_type", -1)
            token_type_name = metrics.get("token_type_name", "Unknown")
            
            # Get x and y values (connectivity and importance)
            x_val = metrics["network"].get("global_metrics", {}).get("avg_pagerank", 0) * 100  # Scale up
            y_val = metrics["metrics"].get("global_metrics", {}).get("avg_importance", 0)
            
            # Get bubble size (based on preservation score)
            size = metrics["metrics"].get("global_metrics", {}).get("preservation_score", 0.1) * 100
            size = max(20, size)  # Ensure minimum bubble size
            
            # Get color based on token type
            if token_type == 0:  # Generated
                color = 'green'
            elif token_type == 1:  # Text
                color = 'blue'
            elif token_type == 2:  # Image
                color = 'red'
            else:
                color = 'gray'
            
            # Get global roles
            global_roles = metrics.get("roles", {}).get("global_roles", [])
            
            # Create token data
            token_data.append({
                "token_idx": token_idx,
                "token_text": token_text,
                "token_type": token_type,
                "token_type_name": token_type_name,
                "x": x_val,
                "y": y_val,
                "size": size,
                "color": color,
                "global_roles": global_roles,
                "importance": metrics["overall_importance"]
            })
        
        # Skip if no data
        if not token_data:
            print("No data for visualization")
            return ""
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        for token in token_data:
            plt.scatter(
                token["x"], token["y"],
                s=token["size"],
                c=token["color"],
                alpha=0.6,
                edgecolors='black',
                linewidths=1
            )
            
            # Add label for important tokens
            if token["importance"] > 0.5:
                plt.annotate(
                    token["token_text"],
                    (token["x"], token["y"]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
        
        # Set axis labels
        plt.xlabel("Centrality (PageRank × 100)")
        plt.ylabel("Average Importance")
        
        # Add title
        plt.title("Token Metrics Visualization\nBubble Size = Preservation Score")
        
        # Add legend for token types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Generated'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Text'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Image')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits with padding
        x_vals = [t["x"] for t in token_data]
        y_vals = [t["y"] for t in token_data]
        plt.xlim(min(0, min(x_vals) * 0.9), max(x_vals) * 1.1)
        plt.ylim(min(0, min(y_vals) * 0.9), max(y_vals) * 1.1)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved token metrics visualization to: {output_path}")
        
        return output_path
    
    def visualize_token_roles(
        self,
        combined_metrics: Dict[int, Dict[str, Any]],
        output_path: Optional[str] = None
    ) -> str:
        """
        Create visualization of token roles across layers.
        
        Args:
            combined_metrics: Dictionary with combined token metrics
            output_path: Path to save the visualization (None = auto-generate)
            
        Returns:
            Path to saved visualization
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Use default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, "token_roles_visualization.png")
        
        # Get all layers and roles
        all_layers = set()
        all_roles = set()
        
        for token_idx, metrics in combined_metrics.items():
            if "roles" in metrics and "layer_roles" in metrics["roles"]:
                for layer, roles in metrics["roles"]["layer_roles"].items():
                    all_layers.add(int(layer))
                    all_roles.update(roles)
        
        all_layers = sorted(all_layers)
        all_roles = sorted(all_roles)
        
        # Skip if no data
        if not all_layers or not all_roles:
            print("No role data for visualization")
            return ""
        
        # Count roles per layer
        role_counts = {layer: {role: 0 for role in all_roles} for layer in all_layers}
        
        for token_idx, metrics in combined_metrics.items():
            if "roles" in metrics and "layer_roles" in metrics["roles"]:
                for layer_str, roles in metrics["roles"]["layer_roles"].items():
                    layer = int(layer_str)
                    for role in roles:
                        if role in all_roles:
                            role_counts[layer][role] += 1
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Convert to arrays for plotting
        layers_array = np.array(all_layers)
        roles_data = []
        
        for role in all_roles:
            role_data = [role_counts[layer][role] for layer in all_layers]
            roles_data.append(role_data)
        
        # Create stacked bar chart
        bottom = np.zeros(len(all_layers))
        for i, role_data in enumerate(roles_data):
            plt.bar(
                layers_array,
                role_data,
                bottom=bottom,
                label=all_roles[i],
                alpha=0.7
            )
            bottom += np.array(role_data)
        
        # Set axis labels
        plt.xlabel("Layer")
        plt.ylabel("Number of Tokens")
        
        # Add title
        plt.title("Token Roles Across Layers")
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set x-axis ticks
        plt.xticks(layers_array)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved token roles visualization to: {output_path}")
        
        return output_path
    
    def visualize_token_graph(
        self,
        trace_df: pd.DataFrame,
        layer: int,
        output_path: Optional[str] = None,
        min_edge_weight: float = 0.1,
        highlight_tokens: Optional[List[int]] = None
    ) -> str:
        """
        Create visualization of token graph for a specific layer.
        
        Args:
            trace_df: DataFrame with trace data
            layer: Layer to visualize
            output_path: Path to save the visualization (None = auto-generate)
            min_edge_weight: Minimum edge weight to include
            highlight_tokens: List of token indices to highlight
            
        Returns:
            Path to saved visualization
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Use default output path if not provided
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"token_graph_layer_{layer}.png")
        
        # Filter data for this layer
        layer_df = trace_df[trace_df["layer"] == layer]
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all tokens as nodes
        for _, row in layer_df.iterrows():
            token_idx = row["token_index"]
            token_text = row["token_text"]
            token_type = row["token_type"]
            G.add_node(token_idx, text=token_text, type=token_type)
        
        # Add edges based on source relationships
        for _, row in layer_df.iterrows():
            if "sources_indices" in row and pd.notna(row["sources_indices"]):
                # Get target token
                target_idx = row["token_index"]
                
                # Get source tokens and weights
                sources_indices = [int(idx) for idx in str(row["sources_indices"]).split(",") if idx]
                sources_weights = []
                
                if "sources_weights" in row and pd.notna(row["sources_weights"]):
                    sources_weights = [float(w) for w in str(row["sources_weights"]).split(",") if w]
                else:
                    sources_weights = [1.0] * len(sources_indices)
                
                # Add edges with weights, filtering by minimum weight
                for i, source_idx in enumerate(sources_indices):
                    if i < len(sources_weights):
                        weight = sources_weights[i]
                        if weight >= min_edge_weight:
                            if source_idx in G and target_idx in G:  # Ensure nodes exist
                                G.add_edge(source_idx, target_idx, weight=weight)
        
        # Skip if graph is empty
        if len(G) == 0:
            print(f"No graph data for layer {layer}")
            return ""
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Calculate node positions
        try:
            # Try force-directed layout
            pos = nx.spring_layout(G, k=0.3, iterations=50)
        except:
            # Fall back to circular layout
            pos = nx.circular_layout(G)
        
        # Define node colors based on token type
        node_colors = []
        for node in G.nodes():
            token_type = G.nodes[node]["type"]
            if token_type == 0:  # Generated
                color = 'green'
            elif token_type == 1:  # Text
                color = 'blue'
            elif token_type == 2:  # Image
                color = 'red'
            else:
                color = 'gray'
            node_colors.append(color)
        
        # Define node sizes based on degree
        node_sizes = [300 + 50 * G.degree(node) for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            edgecolors='black',
            linewidths=1
        )
        
        # Draw edges with varying width and transparency based on weight
        for u, v, data in G.edges(data=True):
            weight = data["weight"]
            width = max(1, weight * 5)
            alpha = max(0.2, min(0.9, weight))
            
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                alpha=alpha,
                arrowsize=10,
                edge_color='gray'
            )
        
        # Draw node labels
        nx.draw_networkx_labels(
            G, pos,
            labels={node: f"{node}: {G.nodes[node]['text']}" for node in G.nodes()},
            font_size=8,
            font_color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5)
        )
        
        # Highlight specific tokens if requested
        if highlight_tokens:
            highlight_nodes = [node for node in G.nodes() if node in highlight_tokens]
            if highlight_nodes:
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=highlight_nodes,
                    node_color='yellow',
                    node_size=[node_sizes[i] * 1.3 for i, node in enumerate(G.nodes()) if node in highlight_tokens],
                    alpha=0.9,
                    edgecolors='black',
                    linewidths=2
                )
        
        # Add title
        plt.title(f"Token Graph for Layer {layer}")
        
        # Add legend for token types
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Generated'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Text'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Image')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Turn off axis
        plt.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Saved token graph visualization to: {output_path}")
        
        return output_path
    
    def identify_critical_tokens(
        self,
        combined_metrics: Dict[int, Dict[str, Any]],
        top_k: int = 10,
        method: str = "combined"
    ) -> List[Dict[str, Any]]:
        """
        Identify the most critical tokens for the reasoning process.
        
        Args:
            combined_metrics: Dictionary with combined token metrics
            top_k: Number of critical tokens to return
            method: Method to use ("importance", "centrality", "roles", or "combined")
            
        Returns:
            List of critical token information
        """
        # Different scoring methods for critical tokens
        if method == "importance":
            # Sort by overall importance
            sorted_tokens = sorted(
                combined_metrics.items(),
                key=lambda x: x[1]["overall_importance"],
                reverse=True
            )
            
        elif method == "centrality":
            # Sort by network centrality (PageRank)
            sorted_tokens = sorted(
                [
                    (token_idx, metrics) for token_idx, metrics in combined_metrics.items()
                    if "network" in metrics and "global_metrics" in metrics["network"]
                ],
                key=lambda x: x[1]["network"]["global_metrics"].get("avg_pagerank", 0),
                reverse=True
            )
            
        elif method == "roles":
            # Score based on critical global roles
            critical_roles = ["key", "hub", "preservation", "gateway"]
            
            def role_score(metrics):
                if "roles" not in metrics or "global_roles" not in metrics["roles"]:
                    return 0
                
                global_roles = metrics["roles"]["global_roles"]
                return sum(2 if role in critical_roles else 1 for role in global_roles)
            
            sorted_tokens = sorted(
                combined_metrics.items(),
                key=lambda x: role_score(x[1]),
                reverse=True
            )
            
        else:  # combined
            # Combined scoring with multiple factors
            def combined_score(metrics):
                # Base score is importance
                score = metrics.get("overall_importance", 0) * 0.5
                
                # Add network centrality component
                pagerank = metrics.get("network", {}).get("global_metrics", {}).get("avg_pagerank", 0)
                score += pagerank * 5  # Scale pagerank which is usually small
                
                # Add role-based component
                critical_roles = ["key", "hub", "preservation", "gateway"]
                global_roles = metrics.get("roles", {}).get("global_roles", [])
                role_bonus = sum(0.15 if role in critical_roles else 0.05 for role in global_roles)
                score += role_bonus
                
                # Add preservation component
                preservation = metrics.get("metrics", {}).get("global_metrics", {}).get("preservation_score", 0)
                score += preservation * 0.2
                
                return score
            
            sorted_tokens = sorted(
                combined_metrics.items(),
                key=lambda x: combined_score(x[1]),
                reverse=True
            )
        
        # Get top-k critical tokens
        critical_tokens = [
            {
                "token_index": token_idx,
                "token_text": metrics.get("token_text", ""),
                "token_type": metrics.get("token_type_name", ""),
                "importance": metrics.get("overall_importance", 0),
                "global_roles": metrics.get("roles", {}).get("global_roles", []),
                "layers_present": metrics.get("metrics", {}).get("layers_present", []),
                "pagerank": metrics.get("network", {}).get("global_metrics", {}).get("avg_pagerank", 0)
            }
            for token_idx, metrics in sorted_tokens[:top_k]
        ]
        
        return critical_tokens