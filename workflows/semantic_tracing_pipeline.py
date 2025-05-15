# workflows/semantic_tracing_pipeline.py
"""
End-to-end pipeline for running semantic tracing on VLMs.
"""

import torch
import os
import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any

from workflows.semantic_tracing import SemanticTracer
from runtime.model_utils import load_model, get_llm_attention_layer_names
from runtime.selection import SelectionConfig
from preprocess.input_builder import prepare_inputs
from preprocess.image import load_image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_semantic_tracing_pipeline(
    model_id: str,
    image_path: str,
    prompt_text: str,
    output_dir: str = "semantic_tracing_results",
    num_tokens: int = 1,
    target_token_idx: Optional[int] = None,
    analyze_specific_indices: Optional[List[int]] = None,
    analyze_last_token: bool = False,
    load_in_4bit: bool = True,
    tracing_mode: str = "saliency",
    single_forward_pass: bool = False,
    selection_config: Optional[Dict[str, Any]] = None,
    logit_lens_concepts: Optional[List[str]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    cpu_offload: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run a complete semantic tracing pipeline on a VLM.
    
    Args:
        model_id: Hugging Face model ID
        image_path: Path to the image file
        prompt_text: Text prompt
        output_dir: Directory to save results
        num_tokens: Number of tokens to generate and analyze
        target_token_idx: Specific token index to analyze
        analyze_specific_indices: List of token indices to analyze
        analyze_last_token: Whether to analyze last token in prompt
        load_in_4bit: Whether to load the model in 4-bit precision
        tracing_mode: "saliency", "attention", or "both"
        single_forward_pass: Use one forward pass for all layers
        selection_config: Configuration for node selection
        logit_lens_concepts: Concepts to track with logit lens
        image_size: Size to resize the image to
        cpu_offload: Whether to offload tensors to CPU when possible
        debug: Whether to print additional debug information
        
    Returns:
        Dictionary with semantic tracing results
    """
    # 1. Set up directories
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Load model and processor
    logger.info(f"Loading model: {model_id}")
    model, processor = load_model(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        enable_gradients=True  # Required for saliency analysis
    )
    
    # 3. Load image
    logger.info(f"Loading image: {image_path}")
    image = load_image(image_path, resize_to=[336, 336])
    
    # 4. Prepare inputs
    logger.info("Preparing inputs")
    input_data = prepare_inputs(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt_text
    )
    
    # 5. Create selection config if provided
    if selection_config:
        sel_config = SelectionConfig(
            beta_target=selection_config.get("beta_target", 0.8),
            beta_layer=selection_config.get("beta_layer", 1.0),
            min_keep=selection_config.get("min_keep", 4),
            max_keep=selection_config.get("max_keep", 6),
            min_keep_layer=selection_config.get("min_keep_layer", 8),
            max_keep_layer=selection_config.get("max_keep_layer", 400)
        )
    else:
        sel_config = None
    
    # 6. Create tracer
    logger.info("Initializing semantic tracer")
    tracer = SemanticTracer(
        model=model,
        processor=processor,
        output_dir=output_dir,
        selection_config=sel_config,
        logit_lens_concepts=logit_lens_concepts,
        cpu_offload=cpu_offload,
        debug=debug
    )
    
    # 7. Run tracing
    logger.info(f"Running semantic tracing with mode: {tracing_mode}")
    start_time = time.time()
    
    try:
        result = tracer.trace(
            input_data=input_data,
            num_tokens=num_tokens,
            target_token_idx=target_token_idx,
            analyze_specific_indices=analyze_specific_indices,
            analyze_last_token=analyze_last_token,
            tracing_mode=tracing_mode,
            single_forward_pass=single_forward_pass
        )
    except Exception as e:
        logger.error(f"Error during semantic tracing: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    # 8. Calculate and log elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Semantic tracing completed in {elapsed_time:.2f} seconds")
    
    # 9. Add extra metadata to result
    result["elapsed_time"] = elapsed_time
    result["model_id"] = model_id
    result["image_path"] = image_path
    result["prompt_text"] = prompt_text
    
    # 10. Clean up resources
    del model, processor, tracer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result