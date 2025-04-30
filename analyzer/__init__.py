"""
Initialization file for the VLM Analysis analyzer module.

This module contains:
- Core analyzer classes/functions for specific analysis types (logit_lens, saliency).
- Workflow functions that orchestrate common end-to-end analysis procedures.
"""

# Core Analyzer Components
from analyzer.logit_lens import LLaVANextLogitLensAnalyzer
from analyzer.saliency import (
    calculate_saliency_scores,
    analyze_layerwise_saliency_flow,
    compute_flow_metrics_optimized
)

# Include stepwise_logit_lens_workflow as requested
try:
    from analyzer.stepwise_logit_lens_workflow import run_stepwise_logit_lens_workflow
except ImportError:
    print("Warning: analyzer.stepwise_logit_lens_workflow not found or has errors.")
    run_stepwise_logit_lens_workflow = None

# Workflow Functions
from analyzer.workflows import (
    run_logit_lens_workflow,
    run_saliency_workflow,
    # Semantic tracing modular functions
    run_semantic_tracing_analysis,  # Runs only the analysis part
    visualize_semantic_tracing_results,  # Creates visualizations from analysis results
    visualize_semantic_tracing_from_csv,  # Creates visualizations directly from saved CSV
    run_semantic_tracing_experiment  # Full experiment (analysis + visualization)
)

# Semantic Tracing Analyzer and Visualizer
from analyzer.semantic_tracing import EnhancedSemanticTracer
from semantic_tracing_visualizer import SemanticTracingVisualizer

__all__ = [
    # Analyzers & Components
    "LLaVANextLogitLensAnalyzer",
    "calculate_saliency_scores",
    "analyze_layerwise_saliency_flow",
    "compute_flow_metrics_optimized",
    "EnhancedSemanticTracer",
    "SemanticTracingVisualizer",
    
    # Workflows
    "run_logit_lens_workflow",
    "run_saliency_workflow",
    "run_semantic_tracing_analysis",
    "visualize_semantic_tracing_results",
    "visualize_semantic_tracing_from_csv",
    "run_semantic_tracing_experiment",
]

# Add stepwise_logit_lens_workflow if available
if run_stepwise_logit_lens_workflow is not None:
    __all__.append("run_stepwise_logit_lens_workflow")