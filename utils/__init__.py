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
    build_formatted_prompt,
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
        analyze_model_architecture,
        print_architecture_summary,
        analyze_image_processing,
    )
except ImportError:
    print("Warning: utils.model_utils not found or fully populated yet.")
    pass

try:
    from utils.viz_utils import (
        visualize_information_flow,
        visualize_attention_heatmap,
        visualize_processed_image_input,
        visualize_token_probabilities,
        create_composite_image,
    )
except ImportError:
    print("Warning: utils.viz_utils not found or fully populated yet.")
    pass

__all__ = [
    # data_utils
    "clean_memory",
    "load_image",
    "build_conversation",
    "build_formatted_prompt",
    "get_token_indices",
    "get_token_masks",
    "get_image_token_spans",
    
    # hook_utils
    "ActivationCache",
    "GradientAttentionCapture",
    
    # model_utils (if available)
    "load_model",
    "get_module_by_name",
    "get_llm_attention_layer_names",
    "matches_pattern",
    "analyze_model_architecture",
    "print_architecture_summary",
    "analyze_image_processing",
    
    # viz_utils (if available)
    "visualize_information_flow",
    "visualize_attention_heatmap",
    "visualize_processed_image_input",
    "visualize_token_probabilities",
    "create_composite_image",
]