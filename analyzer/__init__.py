# -*- coding: utf-8 -*-
"""
Initialization file for the VLM Analysis analyzer module.

This module contains:
- Core analyzer classes/functions for specific analysis types (logit_lens, saliency).
- Workflow functions that orchestrate common end-to-end analysis procedures.
"""

# Core Analyzer Components
from analyzer.logit_lens_analyzer import LLaVANextLogitLensAnalyzer
from analyzer.saliency_analyzer import (
    calculate_saliency_scores,
    analyze_layerwise_saliency_flow,
    compute_flow_metrics_optimized
)
from analyzer.stepwise_logit_lens_workflow import run_stepwise_logit_lens_workflow

# Workflow Functions
from analyzer.workflows import (
    run_logit_lens_workflow,
    run_saliency_workflow
)


# Placeholder for the token-by-token logit lens analyzer (as per user request)
# from .token_by_token_logit_lens import LLaVANextTokenLogitLensAnalyzer

__all__ = [
    # Analyzers & Components
    "LLaVANextLogitLensAnalyzer",
    "calculate_saliency_scores",
    "analyze_layerwise_saliency_flow",
    "compute_flow_metrics_optimized",
    # Workflows
    "run_logit_lens_workflow",
    "run_saliency_workflow",
    "run_stepwise_logit_lens_workflow",
    # "LLaVANextTokenLogitLensAnalyzer", # Uncomment when implemented
]