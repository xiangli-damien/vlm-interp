# -*- coding: utf-8 -*-
"""
Initialization file for the VLM Analysis utilities module.

This module provides various helper functions and classes for data handling,
model interaction, visualization, and hooking into model internals.
"""

from utils.data_utils import (
    clean_memory,
    load_image,
    build_conversation,
    find_token_indices,
    find_image_token_spans,
    get_token_indices,
    get_token_masks,
    get_image_token_spans,
)

from utils.hook_utils import (
    ActivationCache,
    GradientAttentionCapture,
)

try:
    from utils.model_utils import (
        load_model,
        get_module_by_name,
        get_llm_attention_layer_names,
        matches_pattern,
        analyze_model_architecture,  # Added based on notebook content
        print_architecture_summary,  # Added based on notebook content
        analyze_image_processing,     # Added based on notebook content
    )
except ImportError:
    print("Warning: utils.model_utils not found or fully populated yet.")
    pass

try:
    from utils.visual_utils import (
        visualize_information_flow,
        visualize_attention_heatmap,
        visualize_processed_image_input,  # Added based on notebook content
    )
except ImportError:
    print("Warning: utils.visual_utils not found or fully populated yet.")
    pass

__all__ = [
    # data_utils
    "clean_memory",
    "load_image",
    "build_conversation",
    "find_token_indices",
    "find_image_token_spans",
    "get_token_indices",
    "get_token_masks",
    "get_image_token_spans",
    # hook_utils
    "ActivationCache",
    "GradientAttentionCapture",
    # model_utils (如果可用)
    "load_model",
    "get_module_by_name",
    "get_llm_attention_layer_names",
    "matches_pattern",
    "analyze_model_architecture",
    "print_architecture_summary",
    "analyze_image_processing",
    # visual_utils (如果可用)
    "visualize_information_flow",
    "visualize_attention_heatmap",
    "visualize_processed_image_input",
]
