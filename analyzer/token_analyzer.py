# analyzer/token_analyzer.py
"""
Token analyzer focusing on role identification without network metrics
and unnecessary visualizations.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union

class TokenAnalyzer:
    """
    Analyzes token roles and importance in semantic tracing across model layers.
    Focused version without network metrics and visualizations.
    
    This class provides methods to:
    1. Analyze threshold statistics to determine appropriate parameters
    2. Identify roles of tokens in information flow
    3. Measure token importance 
    4. Perform ablation studies in a separate workflow
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
        Initialize the optimized token analyzer.
        
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
        self.threshold_stats = {}
        self.ablation_results = {}
        
        # Token type descriptions
        self.token_type_names = {
            0: "Generated",
            1: "Text",
            2: "Image"
        }
    
    def analyze_threshold_statistics(
        self, 
        csv_path: str, 
        metadata_path: Optional[str] = None,
        importance_column: str = "importance_weight",
        fast_mode: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze the trace data to determine appropriate threshold values.
        
        Args:
            csv_path: Path to the trace data CSV
            metadata_path: Optional path to metadata JSON
            importance_column: Column to use for importance values
            
        Returns:
            Dictionary with threshold statistics
        """
        print(f"Analyzing threshold statistics from: {csv_path}")
        
        # Load trace data
        try:
            if fast_mode:
                trace_df = TokenAnalyzer.optimize_trace_data_parsing(
                    csv_path=csv_path,
                    importance_column=importance_column,
                    max_tokens=50,  # Limit to 50 tokens for faster statistics
                    sample_rate=0.3  # Sample 30% of rows
                )
            else:
                trace_df = pd.read_csv(csv_path)
                
            if trace_df.empty:
                print("Error: Trace data is empty")
                return {"error": "Empty trace data"}
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return {"error": f"Failed to load CSV: {e}"}
        
        # Initialize statistics
        stats = {
            "receptivity": {"values": [], "percentiles": {}},
            "emission": {"values": [], "percentiles": {}},
            "transformation": {"values": [], "percentiles": {}},
            "importance": {"values": [], "percentiles": {}},
            "preservation": {"values": [], "percentiles": {}},
            "by_layer": {},
            "by_token_type": {}
        }
        
        # Calculate basic metrics for each token and layer
        tokens = trace_df["token_index"].unique()
        layers = sorted(trace_df["layer"].unique())
        
        # Initialize by-layer statistics
        for layer in layers:
            stats["by_layer"][layer] = {
                "receptivity": [],
                "emission": [],
                "transformation": []
            }
        
        # Initialize by-token-type statistics
        for token_type in [0, 1, 2]:  # Generated, Text, Image
            stats["by_token_type"][token_type] = {
                "receptivity": [],
                "emission": [],
                "transformation": []
            }
        
        # Process each token to calculate metrics
        for token_idx in tokens:
            # Get token data across layers
            token_df = trace_df[trace_df["token_index"] == token_idx]
            
            # Skip if token only appears in one layer
            if len(token_df) <= 1:
                continue
            
            # Get token type
            token_type = token_df["token_type"].iloc[0]
            
            # Calculate layer-specific metrics
            layer_metrics = {}
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
                outgoing_indices = []
                outgoing_weights = []
                if "sources_indices" in trace_df.columns:
                    # Check other tokens in the same layer that have this token as source
                    for _, row in trace_df[(trace_df["layer"] == layer) & 
                                          (trace_df["token_index"] != token_idx)].iterrows():
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
                
                # Calculate metrics
                source_diversity = len(set(sources_indices))
                target_diversity = len(set(outgoing_indices))
                
                receptivity = sum(sources_weights) if sources_weights else 0
                emission = sum(outgoing_weights) if outgoing_weights else 0
                throughput = min(receptivity, emission)
                
                # Calculate transformation (difference between input and output information)
                transformation = abs(emission - receptivity) / (max(emission, receptivity) + 1e-8)
                
                # Store metrics for this layer
                layer_metrics[layer] = {
                    "importance": importance,
                    "receptivity": receptivity,
                    "emission": emission,
                    "transformation": transformation,
                    "throughput": throughput,
                    "source_diversity": source_diversity,
                    "target_diversity": target_diversity,
                    "incoming_count": incoming_count,
                    "outgoing_count": outgoing_count
                }
                
                # Add to layer statistics
                stats["by_layer"][layer]["receptivity"].append(receptivity)
                stats["by_layer"][layer]["emission"].append(emission)
                stats["by_layer"][layer]["transformation"].append(transformation)
                
                # Add to token type statistics
                stats["by_token_type"][token_type]["receptivity"].append(receptivity)
                stats["by_token_type"][token_type]["emission"].append(emission)
                stats["by_token_type"][token_type]["transformation"].append(transformation)
                
                # Add to global statistics
                stats["receptivity"]["values"].append(receptivity)
                stats["emission"]["values"].append(emission)
                stats["transformation"]["values"].append(transformation)
                stats["importance"]["values"].append(importance)
            
            # Calculate preservation score
            importance_values = [m["importance"] for m in layer_metrics.values()]
            connectivity_values = [m["incoming_count"] + m["outgoing_count"] for m in layer_metrics.values()]
            
            importance_variation = np.std(importance_values)
            connectivity_variation = np.std(connectivity_values)
            
            avg_importance = np.mean(importance_values)
            avg_connectivity = np.mean(connectivity_values)
            
            rel_importance_var = importance_variation / (avg_importance + 1e-8)
            rel_connectivity_var = connectivity_variation / (avg_connectivity + 1e-8)
            
            preservation_score = 1.0 - 0.5 * (rel_importance_var + rel_connectivity_var)
            preservation_score = max(0, min(1, preservation_score))
            
            stats["preservation"]["values"].append(preservation_score)
        
        # Calculate percentiles for each metric
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        
        for metric in ["receptivity", "emission", "transformation", "importance", "preservation"]:
            values = np.array(stats[metric]["values"])
            for p in percentiles:
                stats[metric]["percentiles"][f"p{p}"] = float(np.percentile(values, p))
            
            stats[metric]["mean"] = float(np.mean(values))
            stats[metric]["median"] = float(np.median(values))
            stats[metric]["max"] = float(np.max(values))
            stats[metric]["min"] = float(np.min(values))
        
        # Calculate layer-specific percentiles
        for layer in layers:
            for metric in ["receptivity", "emission", "transformation"]:
                values = np.array(stats["by_layer"][layer][metric])
                if len(values) > 0:
                    stats["by_layer"][layer][f"{metric}_p75"] = float(np.percentile(values, 75))
                    stats["by_layer"][layer][f"{metric}_p90"] = float(np.percentile(values, 90))
                    stats["by_layer"][layer][f"{metric}_mean"] = float(np.mean(values))
        
        # Calculate token-type specific percentiles
        for token_type in [0, 1, 2]:
            for metric in ["receptivity", "emission", "transformation"]:
                values = np.array(stats["by_token_type"][token_type][metric])
                if len(values) > 0:
                    stats["by_token_type"][token_type][f"{metric}_p75"] = float(np.percentile(values, 75))
                    stats["by_token_type"][token_type][f"{metric}_p90"] = float(np.percentile(values, 90))
                    stats["by_token_type"][token_type][f"{metric}_mean"] = float(np.mean(values))
        
        # Recommended thresholds based on statistics
        recommendations = {
            "receptivity_threshold": stats["receptivity"]["percentiles"]["p75"],
            "emission_threshold": stats["emission"]["percentiles"]["p75"],
            "preservation_threshold": stats["preservation"]["percentiles"]["p50"],
        }
        
        stats["recommended_thresholds"] = recommendations
        
        # Save statistics
        stats_path = os.path.join(self.output_dir, "threshold_statistics.json")
        try:
            with open(stats_path, 'w') as f:
                # Convert to serializable format
                serializable_stats = self._make_serializable(stats)
                json.dump(serializable_stats, f, indent=2)
            print(f"Saved threshold statistics to: {stats_path}")
        except Exception as e:
            print(f"Error saving threshold statistics: {e}")
        
        # Store statistics for later use
        self.threshold_stats = stats
        
        # Print key statistics and recommendations
        print("\nKey Threshold Statistics:")
        print(f"  Receptivity:   median={stats['receptivity']['median']:.4f}, "
              f"p75={stats['receptivity']['percentiles']['p75']:.4f}, "
              f"p90={stats['receptivity']['percentiles']['p90']:.4f}")
        print(f"  Emission:      median={stats['emission']['median']:.4f}, "
              f"p75={stats['emission']['percentiles']['p75']:.4f}, "
              f"p90={stats['emission']['percentiles']['p90']:.4f}")
        print(f"  Preservation:  median={stats['preservation']['median']:.4f}, "
              f"p75={stats['preservation']['percentiles']['p75']:.4f}")
        print("\nRecommended Thresholds:")
        print(f"  receptivity_threshold = {recommendations['receptivity_threshold']:.4f}")
        print(f"  emission_threshold = {recommendations['emission_threshold']:.4f}")
        print(f"  preservation_threshold = {recommendations['preservation_threshold']:.4f}")
        
        return stats
    
    def analyze_trace_data(
        self, 
        csv_path: str, 
        metadata_path: Optional[str] = None,
        importance_column: str = "importance_weight",
        analyze_thresholds_first: bool = True,
        fast_mode: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze token roles and metrics from traced data CSV file.
        
        Args:
            csv_path: Path to the trace data CSV
            metadata_path: Optional path to metadata JSON
            importance_column: Column to use for importance values
            analyze_thresholds_first: Whether to analyze thresholds before token roles
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing trace data from: {csv_path}")
        
        # Load trace data
        try:
            if fast_mode:
                trace_df = TokenAnalyzer.optimize_trace_data_parsing(
                    csv_path=csv_path,
                    importance_column=importance_column,
                    max_tokens=100,  # Limit to 50 tokens for faster statistics
                    sample_rate=0.5  # Sample 30% of rows
                )
            else:
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
        
        # Analyze thresholds first if requested
        if analyze_thresholds_first and not self.threshold_stats:
            print("Analyzing threshold statistics first...")
            threshold_stats = self.analyze_threshold_statistics(csv_path, metadata_path, importance_column)
            
            # Optionally update thresholds based on statistics
            if threshold_stats and "recommended_thresholds" in threshold_stats:
                rec = threshold_stats["recommended_thresholds"]
                self.receptivity_threshold = rec["receptivity_threshold"]
                self.emission_threshold = rec["emission_threshold"]
                self.preservation_threshold = rec["preservation_threshold"]
                print("Updated thresholds based on statistics.")
        
        # Analyze token metrics and roles
        token_metrics = self._calculate_token_metrics(trace_df, importance_column)
        self.token_metrics = token_metrics
        
        # Identify token roles
        token_roles = self._identify_token_roles(trace_df, token_metrics)
        self.token_roles = token_roles
        
        # Combine metrics (without network metrics)
        combined_metrics = self._combine_metrics(token_metrics, token_roles)
        
        # Generate summary statistics
        summary = self._generate_analysis_summary(combined_metrics, trace_df, target_tokens)
        
        # Save results
        self._save_analysis_results(combined_metrics, summary)
        
        return {
            "token_metrics": token_metrics,
            "token_roles": token_roles,
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
            token_id = token_df["token_id"].iloc[0]
            
            # Get top prediction information
            top_predictions = {}
            if "predicted_top_token" in token_df.columns and "predicted_top_prob" in token_df.columns:
                for _, row in token_df.iterrows():
                    layer = row["layer"]
                    pred_text = row["predicted_top_token"]
                    pred_prob = row["predicted_top_prob"]
                    top_predictions[layer] = {
                        "text": pred_text,
                        "probability": pred_prob
                    }
            
            # Initialize metrics for this token
            metrics = {
                "token_index": token_idx,
                "token_text": token_text,
                "token_id": token_id,
                "token_type": token_type,
                "token_type_name": self.token_type_names.get(token_type, "Unknown"),
                "layers_present": sorted(token_df["layer"].unique()),
                "layer_metrics": {},
                "top_predictions": top_predictions,
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
                    for _, row in trace_df[(trace_df["layer"] == layer) & 
                                          (trace_df["token_index"] != token_idx)].iterrows():
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
    
    def _combine_metrics(
        self,
        token_metrics: Dict[int, Dict[str, Any]],
        token_roles: Dict[int, Dict[str, Any]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Combine metrics into a unified representation for each token.
        Simplified version without network metrics.
        
        Args:
            token_metrics: Dictionary with token metrics
            token_roles: Dictionary with token roles
            
        Returns:
            Combined metrics dictionary
        """
        combined_metrics = {}
        
        # Get all token indices
        all_tokens = set(token_metrics.keys()) | set(token_roles.keys())
        
        # Combine metrics for each token
        for token_idx in all_tokens:
            combined = {
                "token_index": token_idx,
                "metrics": {},
                "roles": {}
            }
            
            # Add basic info if available
            if token_idx in token_metrics:
                combined["token_text"] = token_metrics[token_idx]["token_text"]
                combined["token_id"] = token_metrics[token_idx].get("token_id", 0)
                combined["token_type"] = token_metrics[token_idx]["token_type"]
                combined["token_type_name"] = token_metrics[token_idx]["token_type_name"]
                combined["metrics"] = token_metrics[token_idx]
            
            # Add roles if available
            if token_idx in token_roles:
                combined["roles"] = token_roles[token_idx]
            
            # Calculate overall importance score
            importance_score = 0.0
            score_components = 0
            
            # Component 1: Average importance from token metrics
            if token_idx in token_metrics and "global_metrics" in token_metrics[token_idx]:
                importance_score += token_metrics[token_idx]["global_metrics"].get("avg_importance", 0)
                score_components += 1
            
            # Component 2: Role-based importance
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
            },
            "token_roles": {
                "global_roles": {},
                "layer_roles": {}
            },
            "target_token_analysis": [],
            "layer_top_tokens": {}
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
        
        # Get top tokens per layer with their roles and predictions
        layers = sorted(trace_df["layer"].unique())
        for layer in layers:
            layer_str = f"layer_{layer}"
            summary["layer_top_tokens"][layer_str] = []
            
            # Get all tokens active in this layer
            layer_tokens = []
            for token_idx, metrics in combined_metrics.items():
                if "metrics" in metrics and "layer_metrics" in metrics["metrics"]:
                    if layer in metrics["metrics"]["layer_metrics"]:
                        layer_tokens.append((token_idx, metrics))
            
            # Sort by importance in this layer
            sorted_layer_tokens = sorted(
                layer_tokens,
                key=lambda x: x[1]["metrics"]["layer_metrics"].get(layer, {}).get("importance", 0),
                reverse=True
            )[:10]  # Top 10 tokens
            
            # Add to summary
            for token_idx, metrics in sorted_layer_tokens:
                token_info = {
                    "token_index": token_idx,
                    "token_text": metrics.get("token_text", ""),
                    "token_type": metrics.get("token_type_name", ""),
                    "importance": metrics["metrics"]["layer_metrics"].get(layer, {}).get("importance", 0),
                    "roles": metrics.get("roles", {}).get("layer_roles", {}).get(layer, []),
                }
                
                # Add top prediction if available
                if "metrics" in metrics and "top_predictions" in metrics["metrics"]:
                    if layer in metrics["metrics"]["top_predictions"]:
                        pred = metrics["metrics"]["top_predictions"][layer]
                        token_info["top_prediction"] = {
                            "text": pred.get("text", ""),
                            "probability": pred.get("probability", 0)
                        }
                
                summary["layer_top_tokens"][layer_str].append(token_info)
        
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
                    "layers": {},
                }
                
                # Add per-layer information for the target token
                if "metrics" in target_metrics and "layer_metrics" in target_metrics["metrics"]:
                    for layer, layer_metrics in target_metrics["metrics"]["layer_metrics"].items():
                        layer_str = f"layer_{layer}"
                        target_analysis["layers"][layer_str] = {
                            "importance": layer_metrics.get("importance", 0),
                            "incoming_count": layer_metrics.get("incoming_count", 0),
                            "outgoing_count": layer_metrics.get("outgoing_count", 0),
                            "roles": target_metrics.get("roles", {}).get("layer_roles", {}).get(layer, []),
                        }
                        
                        # Add top prediction if available
                        if "top_predictions" in target_metrics["metrics"]:
                            if layer in target_metrics["metrics"]["top_predictions"]:
                                pred = target_metrics["metrics"]["top_predictions"][layer]
                                target_analysis["layers"][layer_str]["top_prediction"] = {
                                    "text": pred.get("text", ""),
                                    "probability": pred.get("probability", 0)
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
        
        # Convert summary to serializable format and save
        serializable_summary = self._make_serializable(summary)  # Added this line
        with open(summary_path, 'w') as f:
            json.dump(serializable_summary, f, indent=2)
        
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
    
    def identify_critical_tokens(
        self,
        combined_metrics: Dict[int, Dict[str, Any]],
        top_k: int = 10,
        method: str = "combined"
    ) -> List[Dict[str, Any]]:
        """
        Identify the most critical tokens for the reasoning process.
        Simplified version without network metrics.
        
        Args:
            combined_metrics: Dictionary with combined token metrics
            top_k: Number of critical tokens to return
            method: Method to use ("importance", "roles", or "combined")
            
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
                "layer_predictions": self._extract_token_predictions(metrics)
            }
            for token_idx, metrics in sorted_tokens[:top_k]
        ]
        
        return critical_tokens
    
    def _extract_token_predictions(self, metrics):
        """Extract top predictions for a token across layers."""
        predictions = {}
        if "metrics" in metrics and "top_predictions" in metrics["metrics"]:
            for layer, pred in metrics["metrics"]["top_predictions"].items():
                predictions[f"layer_{layer}"] = {
                    "text": pred.get("text", ""),
                    "probability": pred.get("probability", 0)
                }
        return predictions

    def run_ablation_tests(
        self,
        model,
        processor,
        input_data: Dict[str, Any],
        critical_tokens: List[Dict[str, Any]],
        method: str = "zero_out",
        include_individual_tests: bool = True,
        layer_tests: bool = False,
        num_layer_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Run token ablation experiments to measure token importance.
        Standalone function for ablation testing with layer-specific controls.
        
        Args:
            model: Model to use for simulation
            processor: Processor for the model
            input_data: Prepared input data
            critical_tokens: List of critical tokens to test
            method: Blocking method ("zero_out", "average", "noise", or "interpolate")
            include_individual_tests: Whether to test each token individually
            layer_tests: Whether to test layer-specific blocking
            num_layer_samples: Number of layer samples to test if layer_tests is True
            
        Returns:
            Dictionary with ablation test results
        """
        print(f"\nRunning Token Ablation Tests on {len(critical_tokens)} critical tokens")
        
        # Extract token indices
        token_indices = [token["token_index"] for token in critical_tokens]
        
        # Get an estimate of total layers
        total_layers = 0
        for name, _ in model.named_modules():
            if any(pattern in name for pattern in ['layers.', 'layer.', 'h.']):
                total_layers += 1
        
        # Fast estimation of layer count (may not be precise)
        if total_layers == 0:
            # Fallback estimation for common model architectures
            if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
                total_layers = model.config.num_hidden_layers
            else:
                total_layers = 32  # Common default for large LLMs
        
        print(f"Estimated model layers: {total_layers}")
        
        # First run normal inference as reference
        inputs = input_data["inputs"]
        
        try:
            with torch.no_grad():
                reference_outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"})
                
                # Get reference output token
                reference_logits = reference_outputs.logits
                
                # Handle different logits shapes
                if reference_logits.dim() == 2:  # [batch, vocab]
                    pred_logits = reference_logits
                else:  # [batch, seq, vocab]
                    pred_logits = reference_logits[:, -1, :]
                    
                reference_ids = torch.argmax(pred_logits, dim=-1).unsqueeze(0)
                reference_text = processor.tokenizer.decode(reference_ids[0].tolist())
                
                reference_result = {
                    "output_ids": reference_ids,
                    "output_text": reference_text,
                    "logits": pred_logits
                }
        except Exception as e:
            print(f"Error during reference inference: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Reference inference failed: {str(e)}"}
        
        # Initialize results
        ablation_results = {
            "reference_output": reference_text,
            "individual_ablations": {},
            "all_critical_tokens_ablation": {},
            "layer_specific_ablations": {} if layer_tests else None
        }
        
        # Run individual token tests if requested
        if include_individual_tests:
            for token in critical_tokens:
                token_idx = token["token_index"]
                token_text = token["token_text"]
                print(f"Testing ablation of token {token_idx} ('{token_text}')...")
                
                sim_result = self.simulate_token_blocking(
                    model=model,
                    processor=processor,
                    input_data=input_data,
                    tokens_to_block=[token_idx],
                    method=method,
                    reference_output=reference_result
                )
                
                ablation_results["individual_ablations"][f"token_{token_idx}"] = sim_result
        
        # Test all critical tokens together
        print(f"Testing ablation of all {len(token_indices)} critical tokens together...")
        
        all_tokens_result = self.simulate_token_blocking(
            model=model,
            processor=processor,
            input_data=input_data,
            tokens_to_block=token_indices,
            method=method,
            reference_output=reference_result
        )
        
        ablation_results["all_critical_tokens_ablation"] = all_tokens_result
        
        # Run layer-specific tests if requested
        if layer_tests and total_layers > 0:
            print(f"\nPerforming layer-specific ablation tests...")
            
            # Calculate layer sample points
            layer_samples = []
            if num_layer_samples >= total_layers:
                # Test all layers
                layer_samples = list(range(total_layers))
            else:
                # Sample evenly distributed layers
                for i in range(num_layer_samples):
                    layer_idx = int(i * (total_layers / num_layer_samples))
                    layer_samples.append(layer_idx)
            
            # Test blocking from specific layers onwards
            ablation_results["layer_specific_ablations"]["from_layer"] = {}
            for layer_idx in layer_samples:
                print(f"Testing ablation from layer {layer_idx} onwards...")
                
                layer_result = self.simulate_token_blocking(
                    model=model,
                    processor=processor,
                    input_data=input_data,
                    tokens_to_block=token_indices,
                    method=method,
                    reference_output=reference_result,
                    start_block_layer=layer_idx,
                    end_block_layer=None
                )
                
                ablation_results["layer_specific_ablations"]["from_layer"][f"layer_{layer_idx}"] = layer_result
            
            # Test blocking up to specific layers
            ablation_results["layer_specific_ablations"]["up_to_layer"] = {}
            for layer_idx in layer_samples:
                if layer_idx == 0:
                    continue  # Skip layer 0 as it's equivalent to no blocking
                    
                print(f"Testing ablation up to layer {layer_idx}...")
                
                layer_result = self.simulate_token_blocking(
                    model=model,
                    processor=processor,
                    input_data=input_data,
                    tokens_to_block=token_indices,
                    method=method,
                    reference_output=reference_result,
                    start_block_layer=0,
                    end_block_layer=layer_idx
                )
                
                ablation_results["layer_specific_ablations"]["up_to_layer"][f"layer_{layer_idx}"] = layer_result
            
            # Test blocking specific layers individually
            ablation_results["layer_specific_ablations"]["single_layer"] = {}
            for layer_idx in layer_samples:
                print(f"Testing ablation of only layer {layer_idx}...")
                
                layer_result = self.simulate_token_blocking(
                    model=model,
                    processor=processor,
                    input_data=input_data,
                    tokens_to_block=token_indices,
                    method=method,
                    reference_output=reference_result,
                    start_block_layer=layer_idx,
                    end_block_layer=layer_idx
                )
                
                ablation_results["layer_specific_ablations"]["single_layer"][f"layer_{layer_idx}"] = layer_result
        
        # Print summary
        print("\nAblation Test Results:")
        print(f"Reference output: '{reference_text}'")
        
        if include_individual_tests:
            print("\nIndividual Token Ablation:")
            for token in critical_tokens:
                token_idx = token["token_index"]
                result = ablation_results["individual_ablations"].get(f"token_{token_idx}", {})
                impact = result.get("comparison", {}).get("impact_score", 0)
                match = result.get("comparison", {}).get("token_match", False)
                output = result.get("generated_text", "")
                
                print(f"  Token {token_idx} ('{token['token_text']}'): Impact={impact:.3f}, "
                    f"Match={match}, Output='{output}'")
        
        all_result = ablation_results["all_critical_tokens_ablation"]
        all_impact = all_result.get("comparison", {}).get("impact_score", 0)
        all_match = all_result.get("comparison", {}).get("token_match", False)
        all_output = all_result.get("generated_text", "")
        print(f"\nAll tokens ablation: Impact={all_impact:.3f}, Match={all_match}, Output='{all_output}'")
        
        # Print layer-specific results if available
        if layer_tests and "layer_specific_ablations" in ablation_results:
            print("\nLayer-Specific Ablation Results:")
            
            if "from_layer" in ablation_results["layer_specific_ablations"]:
                print("\n  Blocking from specific layers onwards:")
                for layer_key, result in ablation_results["layer_specific_ablations"]["from_layer"].items():
                    impact = result.get("comparison", {}).get("impact_score", 0)
                    match = result.get("comparison", {}).get("token_match", False)
                    print(f"    {layer_key}: Impact={impact:.3f}, Match={match}")
            
            if "up_to_layer" in ablation_results["layer_specific_ablations"]:
                print("\n  Blocking up to specific layers:")
                for layer_key, result in ablation_results["layer_specific_ablations"]["up_to_layer"].items():
                    impact = result.get("comparison", {}).get("impact_score", 0)
                    match = result.get("comparison", {}).get("token_match", False)
                    print(f"    {layer_key}: Impact={impact:.3f}, Match={match}")
            
            if "single_layer" in ablation_results["layer_specific_ablations"]:
                print("\n  Blocking individual layers:")
                for layer_key, result in ablation_results["layer_specific_ablations"]["single_layer"].items():
                    impact = result.get("comparison", {}).get("impact_score", 0)
                    match = result.get("comparison", {}).get("token_match", False)
                    print(f"    {layer_key}: Impact={impact:.3f}, Match={match}")
        
        # Save results to file
        try:
            import json
            ablation_path = os.path.join(self.output_dir, "ablation_results.json")
            with open(ablation_path, 'w') as f:
                # Convert to serializable format
                serializable_results = self._make_serializable(ablation_results)
                json.dump(serializable_results, f, indent=2)
            print(f"Saved ablation results to: {ablation_path}")
        except Exception as e:
            print(f"Error saving ablation results: {e}")
        
        return ablation_results
    
    def simulate_token_blocking(
        self,
        model,
        processor,
        input_data: Dict[str, Any],
        tokens_to_block: List[int],
        method: str = "zero_out",
        reference_output: Optional[Dict[str, Any]] = None,
        start_block_layer: Optional[int] = None,
        end_block_layer: Optional[int] = None,
        attention_mask_strategy: str = "none",  # 'none', 'block', 'reduce'
        cache_hidden_states: bool = True,
        debug_logging: bool = False
    ) -> Dict[str, Any]:
        """
        Simulate blocking specific tokens to analyze their impact on model reasoning.
        Enhanced version with attention masking and improved layer control.
        
        Args:
            model: Model to use for simulation
            processor: Processor for the model
            input_data: Prepared input data
            tokens_to_block: List of token indices to block
            method: Blocking method ("zero_out", "average", "noise", "interpolate")
            reference_output: Optional reference output for comparison
            start_block_layer: First layer to apply blocking (None = all layers)
            end_block_layer: Last layer to apply blocking (None = all layers)
            attention_mask_strategy: How to modify attention ("none", "block", "reduce")
            cache_hidden_states: Whether to cache hidden states for faster computation
            debug_logging: Whether to print detailed debug logs
            
        Returns:
            Dictionary with simulation results
        """
        import torch
        import gc
        
        # Layer info for debugging
        blocking_layers = "all layers"
        if start_block_layer is not None or end_block_layer is not None:
            if start_block_layer is not None and end_block_layer is not None:
                blocking_layers = f"layers {start_block_layer} to {end_block_layer}"
            elif start_block_layer is not None:
                blocking_layers = f"layers {start_block_layer} and above"
            else:
                blocking_layers = f"layers up to {end_block_layer}"
        
        print(f"Simulating token blocking for {len(tokens_to_block)} tokens using method: {method} on {blocking_layers}")
        print(f"Attention mask strategy: {attention_mask_strategy}")
        
        # Initialize hidden states cache if enabled
        hidden_states_cache = {}
        
        # Map to track which hooks have been called (for debugging)
        hook_call_tracking = {}
        
        # Define hook to modify hidden states
        def blocking_hook(name):
            def hook_fn(module, input_tensors, output_tensors):
                # Extract layer index from name
                layer_idx = -1
                
                # Try to extract layer index from name patterns like "layer.10" or "layers.5"
                import re
                match = re.search(r'layer[s]?\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                else:
                    # Try alternative patterns, e.g., for BERT-type models
                    match = re.search(r'h\.(\d+)', name)
                    if match:
                        layer_idx = int(match.group(1))
                    # Try to get index from module name/attributes
                    elif hasattr(module, 'layer_idx'):
                        layer_idx = module.layer_idx
                
                # Track hook call for debugging
                if debug_logging:
                    hook_call_tracking[name] = hook_call_tracking.get(name, 0) + 1
                
                # Check if we should apply blocking to this layer
                apply_blocking = True
                if start_block_layer is not None and layer_idx < start_block_layer:
                    apply_blocking = False
                if end_block_layer is not None and layer_idx > end_block_layer:
                    apply_blocking = False
                
                # Cache hidden states if needed
                if cache_hidden_states and isinstance(output_tensors, torch.Tensor):
                    # Store copy of original hidden states
                    hidden_states_cache[name] = output_tensors.detach().clone()
                
                # Apply to only hidden states (not attention masks or position IDs) if blocking applies
                if isinstance(output_tensors, torch.Tensor) and apply_blocking:
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
                            elif method == "interpolate":
                                # Replace with interpolation of neighboring tokens
                                if token_idx > 0 and token_idx < modified.shape[1] - 1:
                                    prev_token = hidden_states[:, token_idx-1, :]
                                    next_token = hidden_states[:, token_idx+1, :]
                                    modified[:, token_idx, :] = (prev_token + next_token) / 2
                            elif method == "reduce":
                                # Reduce token representation to 10% of original value
                                modified[:, token_idx, :] *= 0.1
                    
                    return modified
                
                return output_tensors
            return hook_fn
        
        # Define hook for attention modules to modify attention weights
        def attention_output_hook(name):
            def hook_fn(module, input_tensors, output_tensors):
                # Don't apply attention masking if it's disabled
                if attention_mask_strategy == "none":
                    return output_tensors
                
                # Extract layer index from name
                layer_idx = -1
                import re
                match = re.search(r'layer[s]?\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                else:
                    match = re.search(r'h\.(\d+)', name)
                    if match:
                        layer_idx = int(match.group(1))
                    elif hasattr(module, 'layer_idx'):
                        layer_idx = module.layer_idx
                
                # Check if we should apply blocking to this layer
                apply_blocking = True
                if start_block_layer is not None and layer_idx < start_block_layer:
                    apply_blocking = False
                if end_block_layer is not None and layer_idx > end_block_layer:
                    apply_blocking = False
                
                if not apply_blocking:
                    return output_tensors
                
                # For attention outputs (typically a tuple with attention probs)
                if isinstance(output_tensors, tuple) and len(output_tensors) > 0:
                    # First element is typically the output, second might be attention weights
                    if len(output_tensors) > 1 and isinstance(output_tensors[1], torch.Tensor):
                        attention_weights = output_tensors[1].clone()
                        
                        # Modify attention weights based on strategy
                        for token_idx in tokens_to_block:
                            if token_idx < attention_weights.shape[2]:  # Check if token idx is valid
                                if attention_mask_strategy == "block":
                                    # Block this token from receiving attention (column)
                                    attention_weights[:, :, token_idx, :] = 0
                                    # Block this token from providing attention (row)
                                    attention_weights[:, :, :, token_idx] = 0
                                elif attention_mask_strategy == "reduce":
                                    # Reduce attention to/from this token by 90%
                                    attention_weights[:, :, token_idx, :] *= 0.1
                                    attention_weights[:, :, :, token_idx] *= 0.1
                        
                        # Create new output tuple with modified attention weights
                        new_outputs = list(output_tensors)
                        new_outputs[1] = attention_weights
                        return tuple(new_outputs)
                
                return output_tensors
            return hook_fn
        
        # Define hook for attention modules to modify attention masks - for pre-hooks
        def attention_input_hook(name):
            def hook_fn(module, input_tensors):  # Note: pre_hook only has two parameters
                # Don't apply attention masking if it's disabled
                if attention_mask_strategy == "none":
                    return input_tensors
                
                # Extract layer index from name
                layer_idx = -1
                import re
                match = re.search(r'layer[s]?\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                else:
                    match = re.search(r'h\.(\d+)', name)
                    if match:
                        layer_idx = int(match.group(1))
                    elif hasattr(module, 'layer_idx'):
                        layer_idx = module.layer_idx
                
                # Check if we should apply blocking to this layer
                apply_blocking = True
                if start_block_layer is not None and layer_idx < start_block_layer:
                    apply_blocking = False
                if end_block_layer is not None and layer_idx > end_block_layer:
                    apply_blocking = False
                
                if not apply_blocking:
                    return input_tensors
                
                # For attention inputs (typically q, k, v tensors and attention mask)
                if isinstance(input_tensors, tuple) and len(input_tensors) >= 2:
                    # Look for attention mask in inputs (typically position 3)
                    for i in range(len(input_tensors)):
                        if isinstance(input_tensors[i], torch.Tensor) and input_tensors[i].dtype == torch.bool:
                            attention_mask = input_tensors[i].clone()
                            
                            # Modify attention mask based on strategy
                            for token_idx in tokens_to_block:
                                if token_idx < attention_mask.shape[1]:  # Check if token idx is valid
                                    if attention_mask_strategy == "block":
                                        # Set attention mask to False (0) for blocked token
                                        attention_mask[:, token_idx] = False
                                    elif attention_mask_strategy == "reduce":
                                        # Can't partially mask with boolean, handled at attention weights
                                        pass
                            
                            # Create new input tuple with modified attention mask
                            new_inputs = list(input_tensors)
                            new_inputs[i] = attention_mask
                            return tuple(new_inputs)
                
                return input_tensors
            return hook_fn
        
        # Get base LLM module to attach hooks
        llm_module = model.get_decoder() if hasattr(model, "get_decoder") else model
        
        # Run inference with blocking
        try:
            # Register hooks for each layer
            hook_handles = []
            attn_hook_handles = []
            
            # Find all transformer layers in the model
            for name, module in llm_module.named_modules():
                # Look for transformer blocks/layers
                if any(pattern in name for pattern in ['layers.', 'layer.', 'h.']):
                    # Register hook for this module
                    hook_handles.append(module.register_forward_hook(blocking_hook(name)))
                
                # Look for attention modules
                if attention_mask_strategy != "none" and any(pattern in name for pattern in ['attention', 'attn']):
                    # Register attention output hook
                    attn_hook_handles.append(module.register_forward_hook(attention_output_hook(name)))
                    # Also try to register attention input hook
                    if hasattr(module, 'register_forward_pre_hook'):
                        attn_hook_handles.append(module.register_forward_pre_hook(attention_input_hook(f"{name}_input")))
            
            print(f"Registered {len(hook_handles)} layer hooks and {len(attn_hook_handles)} attention hooks")
            
            # Run model with blocked tokens
            inputs = input_data["inputs"]
            
            # Check if we should use reference output's logits shape
            ref_logits_dim = 2
            if reference_output is not None and "logits" in reference_output:
                ref_logits_dim = reference_output["logits"].dim()
            
            with torch.no_grad():
                # Run the model with hooks active
                try:
                    outputs = model(
                        **{k: v for k, v in inputs.items() if k != "token_type_ids"},
                        output_hidden_states=True,
                        output_attentions=True  # Get attention matrices
                    )
                except TypeError:
                    # If model doesn't support output_attentions, try without it
                    print("Model does not support output_attentions, trying without it...")
                    outputs = model(
                        **{k: v for k, v in inputs.items() if k != "token_type_ids"},
                        output_hidden_states=True
                    )
            
            # Remove hooks
            for handle in hook_handles:
                handle.remove()
            for handle in attn_hook_handles:
                handle.remove()
            
            # Clean up hidden states cache to save memory
            hidden_states_cache.clear()
            
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Extract generated text
            logits = outputs.logits
            
            # Handle different logits shapes
            if logits.dim() == 2:  # [batch, vocab]
                # Add sequence dimension if needed to match reference
                if ref_logits_dim == 3:
                    logits = logits.unsqueeze(1)  # [batch, seq=1, vocab]
                    last_pos = 0
                else:
                    last_pos = -1  # Use last position for 2D
            else:  # [batch, seq, vocab]
                last_pos = -1  # Use last position for 3D
                
                # If reference is 2D but our result is 3D, extract last token
                if ref_logits_dim == 2:
                    logits = logits[:, -1, :]  # [batch, vocab]
            
            # Extract prediction
            if logits.dim() == 3:
                pred_logits = logits[:, last_pos, :]
            else:
                pred_logits = logits
                
            generated_ids = torch.argmax(pred_logits, dim=-1).unsqueeze(0)
            generated_text = processor.tokenizer.decode(generated_ids[0].tolist())
            
            # Compare with reference if provided
            comparison = {}
            if reference_output is not None and "output_ids" in reference_output:
                ref_ids = reference_output["output_ids"]
                
                # Get reference logits
                ref_logits = reference_output["logits"]
                
                # Ensure dimensions match for comparison
                if ref_logits.dim() != pred_logits.dim():
                    if ref_logits.dim() == 3 and pred_logits.dim() == 2:
                        ref_logits = ref_logits[:, -1, :]
                    elif ref_logits.dim() == 2 and pred_logits.dim() == 3:
                        pred_logits = pred_logits[:, -1, :]
                
                # Compare output probability distributions
                ref_probs = torch.softmax(ref_logits, dim=-1)
                sim_probs = torch.softmax(pred_logits, dim=-1)
                
                # KL Divergence as distribution distance with numerical stability
                try:
                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-8
                    kl_div = torch.nn.functional.kl_div(
                        (sim_probs + epsilon).log(),
                        ref_probs + epsilon,
                        reduction="batchmean"
                    ).item()
                except Exception as e:
                    if debug_logging:
                        print(f"KL divergence calculation failed: {e}")
                    # Fallback to L2 distance if KL fails
                    kl_div = torch.mean((sim_probs - ref_probs) ** 2).item()
                
                # Token match - whether prediction is the same
                token_match = (generated_ids == ref_ids).all().item()
                
                # Get probability of reference token in blocked model
                ref_token_id = ref_ids[0].item()
                ref_token_prob = sim_probs[0, ref_token_id].item() if ref_token_id < sim_probs.shape[1] else 0.0
                
                # Get top-5 tokens from reference and simulation
                top_k = 5
                ref_top_tokens = torch.topk(ref_probs, k=min(top_k, ref_probs.shape[1]), dim=1)
                sim_top_tokens = torch.topk(sim_probs, k=min(top_k, sim_probs.shape[1]), dim=1)
                
                ref_top_ids = ref_top_tokens.indices[0].tolist()
                ref_top_probs = ref_top_tokens.values[0].tolist()
                sim_top_ids = sim_top_tokens.indices[0].tolist()
                sim_top_probs = sim_top_tokens.values[0].tolist()
                
                # Convert token IDs to text for better interpretability
                ref_top_text = [processor.tokenizer.decode([idx]) for idx in ref_top_ids]
                sim_top_text = [processor.tokenizer.decode([idx]) for idx in sim_top_ids]
                
                # Calculate entropy of both distributions as a measure of uncertainty
                ref_entropy = -torch.sum(ref_probs * torch.log(ref_probs + epsilon)).item()
                sim_entropy = -torch.sum(sim_probs * torch.log(sim_probs + epsilon)).item()
                
                # Calculate impact metrics
                if ref_token_prob > 0:
                    # Probability decrease for reference token
                    prob_impact = 1.0 - (ref_token_prob / ref_probs[0, ref_token_id].item())
                else:
                    prob_impact = 1.0  # Maximum impact
                
                # Change in uncertainty (entropy)
                entropy_change = (sim_entropy - ref_entropy) / max(ref_entropy, 1e-8)
                
                # Final impact score (weighted combination)
                impact_score = 0.7 * prob_impact + 0.3 * min(1.0, max(0.0, entropy_change))
                
                comparison = {
                    "kl_divergence": kl_div,
                    "token_match": token_match,
                    "ref_token_prob": ref_token_prob,
                    "impact_score": impact_score,
                    "entropy_change": entropy_change,
                    "ref_top_tokens": [
                        {"id": id, "text": text, "prob": prob} 
                        for id, text, prob in zip(ref_top_ids, ref_top_text, ref_top_probs)
                    ],
                    "sim_top_tokens": [
                        {"id": id, "text": text, "prob": prob} 
                        for id, text, prob in zip(sim_top_ids, sim_top_text, sim_top_probs)
                    ]
                }
            
            # Return results
            results = {
                "method": method,
                "tokens_blocked": tokens_to_block,
                "blocking_layers": blocking_layers,
                "start_block_layer": start_block_layer,
                "end_block_layer": end_block_layer,
                "attention_mask_strategy": attention_mask_strategy,
                "generated_text": generated_text,
                "comparison": comparison
            }
            
            if debug_logging:
                results["debug"] = {
                    "hook_calls": hook_call_tracking,
                }
            
            print(f"Blocking simulation complete.")
            if "comparison" in results:
                token_match = results["comparison"].get("token_match", "N/A")
                impact = results["comparison"].get("impact_score", "N/A")
                print(f"  Token match: {token_match}, Impact score: {impact}")
            
            return results
            
        except Exception as e:
            print(f"Error during token blocking simulation: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "method": method,
                "tokens_blocked": tokens_to_block,
                "blocking_layers": blocking_layers,
                "attention_mask_strategy": attention_mask_strategy
            }

    @staticmethod
    def optimize_trace_data_parsing(
        csv_path: str,
        importance_column: str = "importance_weight",
        max_tokens: Optional[int] = None,
        sample_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Optimize loading and parsing of the trace data CSV file, with options
        to sample a subset for faster processing when obtaining statistics.
        
        Args:
            csv_path: Path to the trace data CSV
            importance_column: Column to use for importance values
            max_tokens: Maximum number of tokens to analyze (sample if larger)
            sample_rate: Optional sampling rate for rows (0.0-1.0)
            
        Returns:
            DataFrame with trace data (potentially sampled)
        """
        try:
            # Load only essential columns to speed up reading
            essential_cols = [
                "layer", "token_index", "token_text", "token_id", "token_type",
                "sources_indices", "sources_weights", importance_column
            ]
            
            # Use low_memory=False for more reliable parsing
            trace_df = pd.read_csv(csv_path, low_memory=False)
            
            # Check if we have the essential columns
            avail_cols = [col for col in essential_cols if col in trace_df.columns]
            if not avail_cols:
                # Fall back to reading all columns
                trace_df = pd.read_csv(csv_path, low_memory=False)
                
            # Apply sampling if needed
            if sample_rate and 0 < sample_rate < 1:
                trace_df = trace_df.sample(frac=sample_rate, random_state=42)
                print(f"Sampled {len(trace_df)} rows at rate {sample_rate}")
                
            # Limit number of tokens
            if max_tokens:
                unique_tokens = trace_df["token_index"].unique()
                if len(unique_tokens) > max_tokens:
                    sample_tokens = np.random.choice(unique_tokens, size=max_tokens, replace=False)
                    trace_df = trace_df[trace_df["token_index"].isin(sample_tokens)]
                    print(f"Limited analysis to {max_tokens} randomly selected tokens")
                    
            # Convert weights to numeric and handle NaNs
            if importance_column in trace_df.columns:
                trace_df[importance_column] = pd.to_numeric(trace_df[importance_column], errors='coerce').fillna(0)
            
            return trace_df
            
        except Exception as e:
            print(f"Error optimizing trace data: {e}")
            # Return the original DataFrame
            return pd.read_csv(csv_path)

    @staticmethod
    def analyze_layer_impact(
        model,
        processor,
        input_data: Dict[str, Any],
        critical_tokens: List[Dict[str, Any]],
        output_dir: str = "token_analysis_results",
        method: str = "zero_out",
        attention_mask_strategy: str = "none",
        layer_count: Optional[int] = None,
        debug_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze the impact of blocking tokens at different layers.
        This function tests blocking tokens starting from each layer up to the last layer,
        and visualizes how the impact changes as blocking progresses through the model.
        
        Args:
            model: Model to use for simulation
            processor: Processor for the model
            input_data: Prepared input data
            critical_tokens: List of critical tokens to block
            output_dir: Directory to save analysis results
            method: Blocking method ("zero_out", "average", "noise", or "interpolate")
            attention_mask_strategy: How to modify attention ("none", "block", "reduce")
            layer_count: Total number of model layers (if None, tries to detect automatically)
            debug_mode: Whether to print detailed debug information
            
        Returns:
            Dictionary with layer impact analysis results and visualization paths
        """
        import os
        import json
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from typing import Dict, List, Any, Optional, Union
        from matplotlib.colors import LinearSegmentedColormap
        
        print("\n--- Analyzing Layer-wise Impact of Token Blocking ---")
        
        # Initialize TokenAnalyzer for token blocking
        from analyzer.token_analyzer import TokenAnalyzer
        analyzer = TokenAnalyzer(output_dir=output_dir, debug_mode=debug_mode)
        
        # Create output directory for visualizations
        viz_dir = os.path.join(output_dir, "layer_impact_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Determine model layer count if not provided
        if layer_count is None:
            layer_count = 0
            # Count layers based on common model layer naming patterns
            for name, _ in model.named_modules():
                if any(pattern in name for pattern in ['layers.', 'layer.', 'h.']):
                    layer_count += 1
            
            # Fallback estimation for common model architectures
            if layer_count == 0:
                if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
                    layer_count = model.config.num_hidden_layers
                else:
                    layer_count = 24  # Common default for large LLMs
        
        print(f"Model has {layer_count} layers")
        
        # First run normal inference as reference
        inputs = input_data["inputs"]
        
        try:
            with torch.no_grad():
                reference_outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"})
                
                # Get reference output token
                reference_logits = reference_outputs.logits
                
                # Handle different logits shapes
                if reference_logits.dim() == 2:  # [batch, vocab]
                    pred_logits = reference_logits
                else:  # [batch, seq, vocab]
                    pred_logits = reference_logits[:, -1, :]
                    
                reference_ids = torch.argmax(pred_logits, dim=-1).unsqueeze(0)
                reference_text = processor.tokenizer.decode(reference_ids[0].tolist())
                
                # Get probabilities
                reference_probs = torch.softmax(pred_logits, dim=-1)
                
                reference_result = {
                    "output_ids": reference_ids,
                    "output_text": reference_text,
                    "logits": pred_logits,
                    "probabilities": reference_probs
                }
        except Exception as e:
            print(f"Error during reference inference: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Reference inference failed: {str(e)}"}
        
        # Extract token indices from critical tokens
        token_indices = [token["token_index"] for token in critical_tokens]
        
        # Test blocking from each layer to the end
        layer_results = []
        
        print(f"Testing blocking from each layer to the end...")
        for start_layer in range(layer_count):
            print(f"  Testing blocking from layer {start_layer} to the end...")
            
            # Run blocking simulation
            result = analyzer.simulate_token_blocking(
                model=model,
                processor=processor,
                input_data=input_data,
                tokens_to_block=token_indices,
                method=method,
                reference_output=reference_result,
                start_block_layer=start_layer,
                end_block_layer=None,
                attention_mask_strategy=attention_mask_strategy
            )
            
            # Extract key metrics
            impact = result.get("comparison", {}).get("impact_score", 0)
            token_match = result.get("comparison", {}).get("token_match", False)
            ref_token_prob = result.get("comparison", {}).get("ref_token_prob", 0)
            
            # Store result
            layer_results.append({
                "start_layer": start_layer,
                "end_layer": layer_count - 1,
                "impact_score": impact,
                "token_match": token_match,
                "ref_token_prob": ref_token_prob,
                "generated_text": result.get("generated_text", "")
            })
        
        # Collect metrics for visualization
        layers = [r["start_layer"] for r in layer_results]
        impact_scores = [r["impact_score"] for r in layer_results]
        ref_token_probs = [r["ref_token_prob"] for r in layer_results]
        
        # Create visualizations
        
        # 1. Impact score vs layer plot
        plt.figure(figsize=(10, 6))
        plt.plot(layers, impact_scores, 'o-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Starting Layer for Blocking', fontsize=12)
        plt.ylabel('Impact Score', fontsize=12)
        plt.title(f'Impact of Blocking Tokens from Different Layers\nMethod: {method}, Attention Mask: {attention_mask_strategy}', fontsize=14)
        plt.xticks(range(0, layer_count, max(1, layer_count // 10)))
        plt.ylim(0, max(1.0, max(impact_scores) * 1.1))
        
        # Add high, medium, low impact regions
        plt.axhspan(0.7, 1.0, alpha=0.2, color='red', label='High Impact')
        plt.axhspan(0.3, 0.7, alpha=0.2, color='yellow', label='Medium Impact')
        plt.axhspan(0.0, 0.3, alpha=0.2, color='green', label='Low Impact')
        
        plt.legend()
        plt.tight_layout()
        impact_plot_path = os.path.join(viz_dir, f'impact_by_layer_{method}_{attention_mask_strategy}.png')
        plt.savefig(impact_plot_path)
        
        # 2. Reference token probability vs layer plot
        plt.figure(figsize=(10, 6))
        plt.plot(layers, ref_token_probs, 'o-', linewidth=2, markersize=8, color='green')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Starting Layer for Blocking', fontsize=12)
        plt.ylabel('Reference Token Probability', fontsize=12)
        plt.title(f'Probability of Reference Token After Blocking\nMethod: {method}, Attention Mask: {attention_mask_strategy}', fontsize=14)
        plt.xticks(range(0, layer_count, max(1, layer_count // 10)))
        
        # Add original reference probability line
        ref_prob = reference_probs[0, reference_ids[0]].item()
        plt.axhline(y=ref_prob, color='r', linestyle='--', label=f'Original Probability ({ref_prob:.3f})')
        
        plt.legend()
        plt.tight_layout()
        prob_plot_path = os.path.join(viz_dir, f'reference_prob_by_layer_{method}_{attention_mask_strategy}.png')
        plt.savefig(prob_plot_path)
        
        # 3. Heatmap of impact for all critical tokens individually
        if len(critical_tokens) <= 10:  # Only for a reasonable number of tokens
            all_token_results = []
            for token in critical_tokens:
                token_idx = token["token_index"]
                token_text = token["token_text"]
                print(f"  Analyzing layer impact for token {token_idx} ('{token_text}')...")
                
                token_layer_impact = []
                for start_layer in range(0, layer_count, max(1, layer_count // 10)):  # Sample fewer layers for speed
                    result = analyzer.simulate_token_blocking(
                        model=model,
                        processor=processor,
                        input_data=input_data,
                        tokens_to_block=[token_idx],
                        method=method,
                        reference_output=reference_result,
                        start_block_layer=start_layer,
                        end_block_layer=None,
                        attention_mask_strategy=attention_mask_strategy
                    )
                    
                    impact = result.get("comparison", {}).get("impact_score", 0)
                    token_layer_impact.append({
                        "start_layer": start_layer,
                        "impact_score": impact
                    })
                
                all_token_results.append({
                    "token_index": token_idx,
                    "token_text": token_text,
                    "layer_impacts": token_layer_impact
                })
            
            # Create heatmap
            if all_token_results:
                # Extract data for heatmap
                token_texts = [t["token_text"] for t in all_token_results]
                sample_layers = [r["start_layer"] for r in all_token_results[0]["layer_impacts"]]
                
                impact_matrix = np.zeros((len(all_token_results), len(sample_layers)))
                for i, token_result in enumerate(all_token_results):
                    for j, layer_impact in enumerate(token_result["layer_impacts"]):
                        impact_matrix[i, j] = layer_impact["impact_score"]
                
                plt.figure(figsize=(12, len(all_token_results) * 0.8 + 2))
                
                # Custom colormap: green (low impact) to yellow to red (high impact)
                cmap = LinearSegmentedColormap.from_list('impact_cmap', 
                                                    [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
                
                # Plot heatmap
                im = plt.imshow(impact_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                plt.colorbar(im, label='Impact Score')
                
                plt.xlabel('Starting Layer for Blocking')
                plt.ylabel('Token')
                plt.title(f'Impact of Blocking Individual Tokens at Different Layers\nMethod: {method}, Attention Mask: {attention_mask_strategy}')
                
                # Set tick labels
                plt.yticks(range(len(token_texts)), token_texts)
                plt.xticks(range(len(sample_layers)), sample_layers)
                
                plt.tight_layout()
                heatmap_path = os.path.join(viz_dir, f'token_impact_heatmap_{method}_{attention_mask_strategy}.png')
                plt.savefig(heatmap_path)
        
        # Save detailed results to file
        results = {
            "reference_output": reference_text,
            "reference_token_id": reference_ids[0].item(),
            "reference_token_probability": ref_prob,
            "method": method,
            "attention_mask_strategy": attention_mask_strategy,
            "layer_count": layer_count,
            "layer_results": layer_results,
            "visualization_paths": {
                "impact_plot": impact_plot_path,
                "probability_plot": prob_plot_path
            }
        }
        
        if "heatmap_path" in locals():
            results["visualization_paths"]["token_heatmap"] = heatmap_path
        
        results_path = os.path.join(output_dir, f"layer_impact_analysis_{method}_{attention_mask_strategy}.json")
        with open(results_path, 'w') as f:
            # Convert to serializable format (handle numpy data types)
            serializable_results = analyzer._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"Layer impact analysis complete. Results saved to: {results_path}")
        print(f"Visualizations saved to: {viz_dir}")
        
        plt.close('all')  # Close all plots
        
        return results