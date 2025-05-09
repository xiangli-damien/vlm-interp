"""
Visualization utilities for multi-hop reasoning experiments.
Provides tools for visualizing layer-wise experiment results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from typing import Dict, List, Any, Optional, Tuple

def plot_layer_statistics(
    results: Dict[str, Any],
    output_path: str,
    title: str = "VLM Multi-hop Reasoning Analysis"
) -> str:
    """
    Plot layer-wise statistics for multi-hop experiment.
    
    Args:
        results: Experiment results dictionary
        output_path: Path to save the plot
        title: Plot title
        
    Returns:
        Path to saved plot
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    fig.suptitle(title, fontsize=16, y=0.98)
    
    # Extract layer indices
    layers = sorted([int(layer) for layer in results["aggregate"]["first_hop"]["entity_sub"].keys()])
    
    # Plot first hop results
    entity_sub_freqs = [results["aggregate"]["first_hop"]["entity_sub"][layer] for layer in layers]
    rel_sub_freqs = [results["aggregate"]["first_hop"]["rel_sub"][layer] for layer in layers]
    
    ax1.plot(layers, entity_sub_freqs, 'b-', marker='o', label='Entity substitution')
    ax1.plot(layers, rel_sub_freqs, 'r--', marker='s', label='Relation substitution')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    ax1.set_ylabel('Relative frequency')
    ax1.set_title('First Hop: Entity Recall Test')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # Add layer markers for peak values
    max_entity_layer = results["aggregate"]["first_hop"]["max_layer_entity"]
    max_rel_layer = results["aggregate"]["first_hop"]["max_layer_rel"]
    
    ax1.plot(max_entity_layer, results["aggregate"]["first_hop"]["entity_sub"][max_entity_layer], 
             'bo', markersize=10, fillstyle='none', markeredgewidth=2)
    ax1.plot(max_rel_layer, results["aggregate"]["first_hop"]["rel_sub"][max_rel_layer], 
             'rs', markersize=10, fillstyle='none', markeredgewidth=2)
    
    # Plot second hop results
    success_freqs = [results["aggregate"]["second_hop"]["success"][layer] for layer in layers]
    
    ax2.plot(layers, success_freqs, 'g-', marker='o', label='Intervention success')
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    ax2.set_ylabel('Relative frequency')
    ax2.set_title('Second Hop: Gradient Intervention Test')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    
    # Add layer marker for peak value
    max_second_layer = results["aggregate"]["second_hop"]["max_layer"]
    ax2.plot(max_second_layer, results["aggregate"]["second_hop"]["success"][max_second_layer], 
             'go', markersize=10, fillstyle='none', markeredgewidth=2)
    
    # Plot combined results
    entity_traversal_freqs = [results["aggregate"]["combined"]["entity_traversal"][layer] for layer in layers]
    rel_traversal_freqs = [results["aggregate"]["combined"]["rel_traversal"][layer] for layer in layers]
    
    ax3.plot(layers, entity_traversal_freqs, 'b-', marker='o', label='Entity traversal')
    ax3.plot(layers, rel_traversal_freqs, 'r--', marker='s', label='Relation traversal')
    ax3.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    ax3.set_xlabel('Layer index')
    ax3.set_ylabel('Relative frequency')
    ax3.set_title('Combined: Full Multi-hop Reasoning')
    ax3.legend(loc='upper left')
    ax3.grid(alpha=0.3)
    
    # Add layer markers for peak values
    max_entity_traversal_layer = results["aggregate"]["combined"]["max_layer_entity"]
    max_rel_traversal_layer = results["aggregate"]["combined"]["max_layer_rel"]
    
    ax3.plot(max_entity_traversal_layer, results["aggregate"]["combined"]["entity_traversal"][max_entity_traversal_layer], 
             'bo', markersize=10, fillstyle='none', markeredgewidth=2)
    ax3.plot(max_rel_traversal_layer, results["aggregate"]["combined"]["rel_traversal"][max_rel_traversal_layer], 
             'rs', markersize=10, fillstyle='none', markeredgewidth=2)
    
    # Add annotations for peak values
    ax1.annotate(f'Max: {results["aggregate"]["first_hop"]["max_value_entity"]:.2f} (Layer {max_entity_layer})',
                xy=(max_entity_layer, results["aggregate"]["first_hop"]["entity_sub"][max_entity_layer]),
                xytext=(max_entity_layer+2, results["aggregate"]["first_hop"]["entity_sub"][max_entity_layer]+0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    ax2.annotate(f'Max: {results["aggregate"]["second_hop"]["max_value"]:.2f} (Layer {max_second_layer})',
                xy=(max_second_layer, results["aggregate"]["second_hop"]["success"][max_second_layer]),
                xytext=(max_second_layer+2, results["aggregate"]["second_hop"]["success"][max_second_layer]+0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    ax3.annotate(f'Max: {results["aggregate"]["combined"]["max_value_entity"]:.2f} (Layer {max_entity_traversal_layer})',
                xy=(max_entity_traversal_layer, results["aggregate"]["combined"]["entity_traversal"][max_entity_traversal_layer]),
                xytext=(max_entity_traversal_layer+2, results["aggregate"]["combined"]["entity_traversal"][max_entity_traversal_layer]+0.05),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save and return path
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def plot_detailed_entrec(
    entrec_values: Dict[str, Dict[int, float]],
    layer_indices: List[int],
    bridge_entity: str,
    output_path: str
) -> str:
    """
    Plot detailed ENTREC values across layers for different prompts.
    
    Args:
        entrec_values: Dictionary of ENTREC values
        layer_indices: List of layer indices to plot
        bridge_entity: Name of bridge entity
        output_path: Path to save the plot
        
    Returns:
        Path to saved plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot ENTREC values
    ax.plot(layer_indices, [entrec_values["original"][l] for l in layer_indices], 'b-', 
            marker='o', label='Original prompt', linewidth=2)
    ax.plot(layer_indices, [entrec_values["entity_sub"][l] for l in layer_indices], 'r--', 
            marker='s', label='Entity substitution', linewidth=2)
    ax.plot(layer_indices, [entrec_values["rel_sub"][l] for l in layer_indices], 'g:', 
            marker='^', label='Relation substitution', linewidth=2)
    
    # Calculate differences for shading
    entrec_diff_entity = np.array([entrec_values["original"][l] - entrec_values["entity_sub"][l] 
                                 for l in layer_indices])
    entrec_diff_rel = np.array([entrec_values["original"][l] - entrec_values["rel_sub"][l] 
                                for l in layer_indices])
    
    # Shade areas where original > substitution
    for i in range(len(layer_indices)-1):
        if entrec_diff_entity[i] > 0 and entrec_diff_entity[i+1] > 0:
            ax.axvspan(layer_indices[i], layer_indices[i+1], alpha=0.1, color='blue')
        if entrec_diff_rel[i] > 0 and entrec_diff_rel[i+1] > 0:
            ax.axvspan(layer_indices[i], layer_indices[i+1], alpha=0.1, color='red')
    
    # Find best layer
    best_layer_idx = np.argmax(entrec_diff_entity)
    best_layer = layer_indices[best_layer_idx]
    
    # Add annotation for best layer
    ax.annotate(f'Strongest evidence\nLayer {best_layer}',
                xy=(best_layer, entrec_values["original"][best_layer]),
                xytext=(best_layer+2, entrec_values["original"][best_layer]+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=7),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('ENTREC Value (log probability)')
    ax.set_title(f'ENTREC Values by Layer for Bridge Entity "{bridge_entity}"')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    
    # Add text explaining the metric
    textstr = (
        f"ENTREC measures the internal recall of the bridge entity '{bridge_entity}'\n"
        f"Higher values indicate stronger entity recall\n"
        f"Blue shading shows where original > entity substitution\n"
        f"Red shading shows where original > relation substitution"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save and return path
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path

def create_layer_heatmap(
    results: Dict[str, Any],
    output_path: str
) -> str:
    """
    Create a layer-by-layer heatmap of multi-hop reasoning performance.
    
    Args:
        results: Experiment results dictionary
        output_path: Path to save the heatmap
        
    Returns:
        Path to saved heatmap
    """
    # Extract layer indices
    layers = sorted([int(layer) for layer in results["aggregate"]["first_hop"]["entity_sub"].keys()])
    
    # Create data arrays
    data = np.zeros((3, len(layers)))
    
    # Fill with data
    for i, layer in enumerate(layers):
        data[0, i] = results["aggregate"]["first_hop"]["entity_sub"][layer]
        data[1, i] = results["aggregate"]["second_hop"]["success"][layer]
        data[2, i] = results["aggregate"]["combined"]["entity_traversal"][layer]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Create custom colormap from blue to red
    cmap = colors.LinearSegmentedColormap.from_list(
        'BlueRed', [(0, 'lightblue'), (0.5, 'blue'), (1, 'darkred')]
    )
    
    # Create heatmap
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Relative frequency', rotation=90, va='bottom')
    
    # Set ticks
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(['First Hop\n(Entity Recall)', 'Second Hop\n(Intervention)', 'Combined\n(Full Traversal)'])
    
    # Rotate x-tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(3):
        for j in range(len(layers)):
            text = ax.text(j, i, f"{data[i, j]:.2f}",
                          ha="center", va="center", color="white" if data[i, j] > 0.5 else "black",
                          fontweight="bold" if data[i, j] > 0.7 else "normal")
    
    # Add threshold lines
    random_thresholds = [0.5, 0.5, 0.25]  # Random chance for each row
    for i, threshold in enumerate(random_thresholds):
        ax.axhline(y=i+0.5, color='gray', linestyle='--', alpha=0.5)
        ax.text(-0.5, i, f"Rand: {threshold}", va='center', ha='right', 
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Formatting
    ax.set_xlabel('Layer Index')
    ax.set_title('Multi-hop Reasoning Performance by Layer')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save and return path
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return output_path