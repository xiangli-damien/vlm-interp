# analysis/saliency_viz.py

import matplotlib.pyplot as plt
from typing import Dict, Optional

def visualize_information_flow(
    metrics: Dict[int, Dict[str, float]],
    title: str = "VLM Information Flow Analysis",
    save_path: Optional[str] = None,
    use_top_k: bool = False
):
    """
    Visualizes information flow metrics across model layers.
    
    Args:
        metrics: Dictionary mapping layer indices to flow metrics dictionaries
        title: Title for the plot
        save_path: Path to save the figure (optional)
        use_top_k: Whether to use top-k image token metrics if available
    """
    if not metrics:
        print("Warning: No metrics to visualize.")
        return
    
    # Define consistent markers and labels for flow types
    flow_styles = {
        # Standard metrics
        "Siq_mean": {"marker": "o", "label": "Image→Target (Mean)"},
        "Stq_mean": {"marker": "^", "label": "Text→Target (Mean)"},
        "Sgq_mean": {"marker": "s", "label": "Gen→Target (Mean)"},
        "Siq_sum":  {"marker": "o", "linestyle": "--", "label": "Image→Target (Sum)"},
        "Stq_sum":  {"marker": "^", "linestyle": "--", "label": "Text→Target (Sum)"},
        "Sgq_sum":  {"marker": "s", "linestyle": "--", "label": "Gen→Target (Sum)"},
        
        # Top-k metrics (will be selected based on use_top_k parameter)
        "Siq_top10_mean": {"marker": "o", "label": "Image→Target (Top-10 Mean)"},
        "Siq_top10_sum":  {"marker": "o", "linestyle": "--", "label": "Image→Target (Top-10 Sum)"},
        "Siq_top20_mean": {"marker": "o", "label": "Image→Target (Top-20 Mean)"},
        "Siq_top20_sum":  {"marker": "o", "linestyle": "--", "label": "Image→Target (Top-20 Sum)"}
    }
    
    # Determine available metrics and layers
    layers = sorted(metrics.keys())
    available_metrics = set()
    for layer in layers:
        available_metrics.update(metrics[layer].keys())
    
    # Select which metrics to plot based on use_top_k parameter
    selected_metrics = []
    
    # Mean metrics
    if use_top_k:
        # Check if top-k metrics exist, with fallbacks
        if "Siq_top10_mean" in available_metrics:
            selected_metrics.append("Siq_top10_mean")
        elif "Siq_top20_mean" in available_metrics:
            selected_metrics.append("Siq_top20_mean")
        else:
            # Fall back to standard if no top-k available
            selected_metrics.append("Siq_mean")
    else:
        selected_metrics.append("Siq_mean")
    
    # Always include text and generated metrics
    selected_metrics.extend(["Stq_mean", "Sgq_mean"])
    
    # Sum metrics
    if use_top_k:
        # Check if top-k metrics exist, with fallbacks
        if "Siq_top10_sum" in available_metrics:
            selected_metrics.append("Siq_top10_sum")
        elif "Siq_top20_sum" in available_metrics:
            selected_metrics.append("Siq_top20_sum")
        else:
            # Fall back to standard if no top-k available
            selected_metrics.append("Siq_sum")
    else:
        selected_metrics.append("Siq_sum")
    
    # Always include text and generated metrics
    selected_metrics.extend(["Stq_sum", "Sgq_sum"])
    
    # Filter selected metrics to those actually available
    selected_metrics = [m for m in selected_metrics if m in available_metrics]
    
    # Build data for plotting
    plot_data = {}
    for key in selected_metrics:
        values = []
        for layer in layers:
            values.append(metrics[layer].get(key, 0.0))
        plot_data[key] = values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    
    # Mean metrics subplot
    ax1.set_title("Mean Information Flow per Layer")
    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Mean Attention / Saliency")
    
    # Plot mean metrics
    mean_metrics = [m for m in selected_metrics if m.endswith("_mean")]
    for key in mean_metrics:
        if key in plot_data:
            style = flow_styles[key]
            ax1.plot(
                layers, plot_data[key], 
                marker=style["marker"], 
                linestyle=style.get("linestyle", "-"),
                label=style["label"], 
                linewidth=2
            )
    
    ax1.legend(loc="best")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Sum metrics subplot
    ax2.set_title("Total Information Flow per Layer")
    ax2.set_xlabel("Layer Index")
    ax2.set_ylabel("Summed Attention / Saliency")
    
    # Plot sum metrics
    sum_metrics = [m for m in selected_metrics if m.endswith("_sum")]
    for key in sum_metrics:
        if key in plot_data:
            style = flow_styles[key]
            ax2.plot(
                layers, plot_data[key], 
                marker=style["marker"], 
                linestyle=style.get("linestyle", "--"),
                label=style["label"], 
                linewidth=2
            )
    
    ax2.legend(loc="best")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    # Set overall title
    prefix = "Top-k " if use_top_k else ""
    fig.suptitle(f"{prefix}{title}", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    # Save figure if path provided
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Information flow visualization saved to: {save_path}")
        except Exception as e:
            print(f"Error saving information flow plot to {save_path}: {e}")
    
    plt.show()
    plt.close(fig)