# workflows/semantic_tracing_pipeline.py

import os
import torch
import json
from typing import Dict, Any, Optional, List, Union
from runtime.selection import SelectionConfig
from workflows.semantic_tracing import SemanticTracingWorkflow
from preprocess.image import load_image
from runtime.model_utils import load_model
from analysis.semantic_viz import FlowGraphVisualizer
import logging

def run_semantic_tracing_test(
    model_id: str,
    image_url: str,
    prompt_text: str,
    output_dir: Optional[str] = None,
    num_tokens: int = 1,
    beta_target: float = 0.8,
    beta_layer: float = 0.7,
    min_keep: int = 1,
    max_keep: int = 30,
    min_keep_layer: int = 5,
    max_keep_layer: int = 100,
    concepts_to_track: Optional[List[str]] = None,
    load_in_4bit: bool = True,
    analyze_last_token: bool = False,
    single_forward_pass: bool = False,
    target_token_idx: Optional[int] = None,
    analyze_specific_indices: Optional[List[int]] = None,
    tracing_mode: str = "saliency",
    skip_visualization: bool = False,
    output_format: str = "both",
    show_orphaned_nodes: bool = False,
    min_edge_weight: float = 0.0,
    use_variable_node_size: bool = True,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Run semantic tracing with improved multi-token handling and specific token analysis.
    
    Args:
        model_id: HuggingFace model ID
        image_url: URL or path to the image
        prompt_text: Text prompt to use
        output_dir: Directory to save results
        num_tokens: Number of tokens to generate and analyze
        beta_target: Coverage threshold for selecting source nodes per target
        beta_layer: Coverage threshold for pruning at layer level
        min_keep: Minimum nodes to keep per target
        max_keep: Maximum nodes to keep per target
        min_keep_layer: Minimum nodes to keep per layer
        max_keep_layer: Maximum nodes to keep per layer
        concepts_to_track: List of concepts to track
        load_in_4bit: Whether to use 4-bit quantization
        analyze_last_token: Analyze last token in prompt
        single_forward_pass: Use single forward pass
        target_token_idx: Index of specific token to analyze (None means generate new tokens)
        analyze_specific_indices: List of specific token indices to analyze
        tracing_mode: The tracing mode to use ("saliency" or "attention")
        skip_visualization: Skip visualization step
        output_format: Output format (png, svg, or both)
        show_orphaned_nodes: Show nodes with no connections
        min_edge_weight: Minimum edge weight threshold
        use_variable_node_size: Vary node size based on importance
        debug_mode: Enable debug mode
        
    Returns:
        Dictionary with test results
    """

    
    # Set up logging
    logger = logging.getLogger("semantic_tracing_test")
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "results", f"semantic_tracing_{model_id.split('/')[-1]}")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Step 1: Load model and processor
        logger.info(f"Loading model: {model_id}")
        model, processor = load_model(
            model_id=model_id,
            load_in_4bit=load_in_4bit
        )
        
        # Step 2: Load and process image
        logger.info(f"Loading image from: {image_url}")
        image = load_image(image_url)
        
        # Step 3: Create selection configuration
        selection_config = SelectionConfig(
            beta_target=beta_target,
            beta_layer=beta_layer,
            min_keep=min_keep, 
            max_keep=max_keep,
            min_keep_layer=min_keep_layer,
            max_keep_layer=max_keep_layer
        )
        
        # Step 4: Initialize the workflow
        logger.info("Initializing semantic tracing workflow")
        workflow = SemanticTracingWorkflow(
            model=model,
            processor=processor,
            output_dir=output_dir,
            selection_config=selection_config,
            debug=debug_mode
        )
        
        # Step 5: Prepare inputs
        logger.info(f"Preparing inputs with prompt: '{prompt_text}'")
        input_data = workflow.prepare_inputs(image, prompt_text)
        
        # Step 6: Run the semantic tracing
        logger.info(f"Running semantic tracing with mode: {tracing_mode}")
        
        # Different execution paths based on what we're analyzing
        if analyze_last_token:
            logger.info("Analyzing last token in prompt")
            results = workflow.analyze_last_token(
                input_data=input_data,
                mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        elif target_token_idx is not None:
            logger.info(f"Analyzing specific token at index: {target_token_idx}")
            results = workflow.analyze_specific_token(
                input_data=input_data,
                token_idx=target_token_idx,
                mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        elif analyze_specific_indices:
            logger.info(f"Analyzing specific token indices: {analyze_specific_indices}")
            results = workflow.analyze_multiple_tokens(
                input_data=input_data,
                token_indices=analyze_specific_indices,
                mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        else:
            logger.info(f"Generating and analyzing {num_tokens} tokens")
            results = workflow.generate_and_trace(
                input_data=input_data,
                num_tokens=num_tokens,
                mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        
        # Step 7: Generate visualization if needed
        if not skip_visualization and "trace_results" in results:
            logger.info("Generating visualizations")
            vis_paths = []
            
            # Get all CSV files from results
            csv_paths = []
            trace_results = results["trace_results"]
            
            # Extract CSV paths based on structure
            if isinstance(trace_results, dict):
                for mode_key, mode_data in trace_results.items():
                    if isinstance(mode_data, dict) and "trace_data_path" in mode_data:
                        csv_paths.append(mode_data["trace_data_path"])
            
            # Create visualizations for each CSV file
            for csv_path in csv_paths:
                if os.path.exists(csv_path):
                    visualizer = FlowGraphVisualizer(
                        output_dir=os.path.join(output_dir, "visualizations"), 
                        debug_mode=debug_mode
                    )
                    
                    # Load and preprocess CSV
                    df = visualizer._preprocess_csv(csv_path)
                    
                    # Extract target token info
                    metadata_path = os.path.join(os.path.dirname(csv_path), "trace_metadata.json")
                    metadata = {}
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    target_token = visualizer._extract_target_token_info(df, metadata)
                    target_idx = target_token.get("index", "unknown")
                    target_text = target_token.get("text", "unknown")
                    
                    # Create save directory with clean token name
                    target_text_clean = "".join(c if c.isalnum() else "_" for c in str(target_text)).strip("_")
                    save_dir = os.path.join(os.path.dirname(csv_path), f"viz_token_{target_idx}_{target_text_clean}")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Generate flow graph
                    paths = visualizer.create_cytoscape_flow_graph_from_csv(
                        trace_data=df,
                        target_text=target_text,
                        target_idx=target_idx,
                        save_dir=save_dir,
                        min_edge_weight=min_edge_weight,
                        show_orphaned_nodes=show_orphaned_nodes,
                        use_variable_node_size=use_variable_node_size,
                        output_format=output_format
                    )
                    vis_paths.extend(paths)
            
            # Add visualization paths to results
            results["visualization_paths"] = vis_paths
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic tracing test: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}