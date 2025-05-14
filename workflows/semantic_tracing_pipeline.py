"""
End-to-end semantic tracing pipeline for comprehensive analysis.
Combines all workflow components into a convenient interface.
"""

import os
import time
import torch
import gc
from typing import Dict, List, Optional, Any, Tuple, Union
from PIL import Image

from runtime.model_utils import load_model
from runtime.selection import SelectionConfig
from preprocess.input_builder import prepare_inputs
from workflows.semantic_tracing import SemanticTracingWorkflow


def run_semantic_tracing(
    model_id: str,
    image_path: str, 
    prompt: str,
    output_dir: str,
    beta_target: float = 0.8,
    beta_layer: float = 0.7,
    min_keep: int = 3,
    max_keep: int = 10,
    min_keep_layer: int = 8,
    max_keep_layer: int = 400,
    num_tokens: int = 1,
    load_in_4bit: bool = True,
    target_token_idx: Optional[int] = None,
    analyze_specific_indices: Optional[List[int]] = None,
    generate_only: bool = False,
    concepts_to_track: Optional[List[str]] = None,
    tracing_mode: str = "saliency",
    enable_gradients: bool = False,
    use_flash_attn: bool = False,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Run semantic tracing analysis on an image-text pair with advanced options.
    
    Args:
        model_id: HuggingFace model ID
        image_path: Path to the image file
        prompt: Text prompt to use
        output_dir: Directory to save results
        beta_target: Coverage threshold for selecting source nodes per target
        beta_layer: Coverage threshold for pruning at layer level
        min_keep: Minimum nodes to keep per target
        max_keep: Maximum nodes to keep per target
        min_keep_layer: Minimum nodes to keep per layer
        max_keep_layer: Maximum nodes to keep per layer
        num_tokens: Number of tokens to generate and analyze
        load_in_4bit: Whether to load the model in 4-bit precision
        target_token_idx: Index of specific token to analyze (if None, uses first generated token)
        analyze_specific_indices: List of specific token indices to analyze
        generate_only: If True, only generate tokens without analysis
        concepts_to_track: List of concepts to track with logit lens
        tracing_mode: The tracing mode to use ("saliency", "attention", or "both")
        enable_gradients: Whether to enable gradient calculation for tracing
        use_flash_attn: Whether to use flash-attention for faster inference
        debug_mode: Whether to print detailed debug information
    
    Returns:
        Dictionary with paths to saved CSV files and metadata
    """
    start_time = time.time()
    print(f"=== Starting Semantic Tracing Analysis ===")
    print(f"Model: {model_id}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Output Directory: {output_dir}")
    print(f"Tracing Mode: {tracing_mode}")
    
    # Create organized output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Print analysis mode information
    if generate_only:
        mode_str = f"Generating {num_tokens} tokens without analysis"
    elif analyze_specific_indices:
        mode_str = f"Generating {num_tokens} tokens and analyzing specific indices: {analyze_specific_indices}"
    else:
        mode_str = f"Analyzing {num_tokens} token(s)"
    print(f"Mode: {mode_str}")
    
    # Print node selection parameters
    print(f"Coverage Parameters: β_target={beta_target}, β_layer={beta_layer}")
    print(f"Node Limits: min_keep={min_keep}, max_keep={max_keep}, min_keep_layer={min_keep_layer}, max_keep_layer={max_keep_layer}")
    
    # 1. Load model and processor
    print("\nLoading model and processor...")
    model, processor = load_model(
        model_id=model_id,
        use_flash_attn=use_flash_attn,
        load_in_4bit=load_in_4bit,
        enable_gradients=enable_gradients,
        device_map="auto"
    )
    
    # 2. Load image
    print("\nLoading image...")
    from preprocess.image import load_image
    image = load_image(image_path, resize_to=(336, 336))
    
    # 3. Create selection config
    selection_config = SelectionConfig(
        beta_target=beta_target,
        beta_layer=beta_layer,
        min_keep=min_keep,
        max_keep=max_keep,
        min_keep_layer=min_keep_layer,
        max_keep_layer=max_keep_layer
    )
    
    # 4. Create semantic tracing workflow
    workflow = SemanticTracingWorkflow(
        model=model,
        processor=processor,
        output_dir=output_dir,
        selection_config=selection_config,
        logit_lens_concepts=concepts_to_track,
        debug=debug_mode
    )
    
    # 5. Prepare inputs
    print("\nPreparing inputs...")
    input_data = workflow.prepare_inputs(image, prompt)
    
    # 6. Run semantic tracing based on the selected mode
    print("\nRunning semantic tracing...")
    
    if generate_only:
        # Generate tokens without analysis (for faster generation)
        gen_results = workflow.autoregressive_generate(input_data["inputs"], num_tokens)
        
        # Create results without trace data
        trace_results = {
            "full_sequence": gen_results["full_sequence"],
            "generated_tokens": gen_results["generated_tokens"],
            "original_seq_len": gen_results["original_seq_len"],
        }
    elif analyze_specific_indices:
        # Generate all tokens first, then analyze specific indices
        trace_results = workflow.generate_all_then_analyze_specific(
            input_data=input_data,
            num_tokens=num_tokens,
            analyze_indices=analyze_specific_indices,
            tracing_mode=tracing_mode
        )
    elif target_token_idx is not None:
        # Analyze a specific token
        result = workflow.analyze_specific_token(
            input_data=input_data,
            token_idx=target_token_idx,
            mode=tracing_mode
        )
        trace_results = {
            "trace_results": {tracing_mode: result},
            "target_token": result["target_tokens"][0] if "target_tokens" in result else None,
            "metadata_path": result.get("metadata_path"),
        }
    elif num_tokens > 1:
        # Analyze multiple consecutively generated tokens
        trace_results = workflow.generate_and_trace(
            input_data=input_data,
            num_tokens=num_tokens,
            mode=tracing_mode
        )
    else:
        # Generate and analyze a single token
        result = workflow.generate_and_trace(
            input_data=input_data,
            num_tokens=1,
            mode=tracing_mode
        )
        # Simplify results structure for single token
        trace_results = {
            "trace_results": result["trace_results"],
            "target_token": result["target_tokens"][0] if result.get("target_tokens") else None,
            "full_sequence": result.get("full_sequence", {}),
            "metadata_path": result.get("metadata_path"),
        }
    
    # 7. Prepare output information
    output_info = {
        "csv_files": [],
        "metadata_path": trace_results.get("metadata_path"),
        "target_tokens": [],
        "all_generated_tokens": trace_results.get("all_generated_tokens", []),
        "full_sequence": trace_results.get("full_sequence", {}),
        "analysis_time": time.time() - start_time,
        "image_path": image_path,
        "tracing_mode": tracing_mode
    }
    
    # Extract CSV paths from results (skip for generate_only mode)
    if not generate_only:
        if "trace_results" in trace_results:
            # Handle both single token and multiple token results
            if isinstance(trace_results["trace_results"], dict):
                # Process each token's results
                for mode_key, mode_value in trace_results["trace_results"].items():
                    # Check if this is a tracing mode results or a token results
                    if isinstance(mode_value, dict) and "csv" in mode_value:
                        # This is a tracing mode result
                        output_info["csv_files"].append(mode_value["csv"])
                    elif isinstance(mode_value, dict):
                        # This is a token results with potentially multiple tracing modes
                        for subkey, subvalue in mode_value.items():
                            if isinstance(subvalue, dict) and "csv" in subvalue:
                                output_info["csv_files"].append(subvalue["csv"])
    
    # Extract target token information
    if "target_tokens" in trace_results:
        output_info["target_tokens"] = trace_results["target_tokens"]
    elif "target_token" in trace_results and trace_results["target_token"]:
        output_info["target_tokens"] = [trace_results["target_token"]]
    
    # Save analysis summary
    csv_dir = os.path.join(output_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)
    summary_path = os.path.join(csv_dir, "analysis_summary.json")
    try:
        import json
        with open(summary_path, 'w') as f:
            # Create a JSON-serializable version of the output info
            summary_info = {
                "model_id": model_id,
                "image_path": image_path,
                "prompt": prompt,
                "num_tokens": num_tokens,
                "tracing_mode": tracing_mode,
                "beta_target": beta_target,
                "beta_layer": beta_layer,
                "min_keep": min_keep,
                "max_keep": max_keep,
                "min_keep_layer": min_keep_layer,
                "max_keep_layer": max_keep_layer,
                "analyze_specific_indices": analyze_specific_indices,
                "csv_files": output_info["csv_files"],
                "metadata_path": output_info["metadata_path"],
                "target_tokens": output_info["target_tokens"],
                "all_generated_tokens": output_info.get("all_generated_tokens", []),
                "full_sequence": output_info["full_sequence"],
                "analysis_time": output_info["analysis_time"]
            }
            json.dump(summary_info, f, indent=2)
        output_info["summary_path"] = summary_path
    except Exception as e:
        print(f"Error saving analysis summary: {e}")
    
    print(f"\n=== Analysis Complete ===")
    print(f"Analysis time: {output_info['analysis_time']:.2f} seconds")
    print(f"CSV data saved to: {os.path.join(output_dir, 'csv_data')}")
    
    if not generate_only:
        print(f"Generated {len(output_info['csv_files'])} trace data files")
    else:
        print(f"Generated {len(output_info.get('all_generated_tokens', []))} tokens without analysis")
    
    # Clean up memory
    del model, processor, workflow, input_data, trace_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_info