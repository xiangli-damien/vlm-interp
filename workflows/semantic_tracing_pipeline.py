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
    analyze_last_token: bool = False,
    target_token_idx: Optional[int] = None,
    analyze_specific_indices: Optional[List[int]] = None,
    tracing_mode: str = "saliency",
    skip_visualization: bool = False,
    output_format: str = "both",
    show_orphaned_nodes: bool = False,
    min_edge_weight: float = 0.0,
    use_variable_node_size: bool = True,
    debug_mode: bool = False,
    low_memory_mode: bool = True,
    fallback_to_attention: bool = True,
    offload_to_cpu: bool = True,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run semantic tracing with improved memory management and OOM prevention strategies.
    """
    # Import torch inside the function to prevent UnboundLocalError
    import torch
    from enum import Enum
    import gc
    
    # Define TraceMode enum inside the function since it's used within the function
    class TraceMode(Enum):
        """Tracing modes for semantic analysis."""
        ATTENTION = "attention"
        SALIENCY = "saliency"
    
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
        # Step 1: Load model with parameters optimized for semantic tracing
        logger.info(f"Loading model: {model_id}")
        
        model, processor = load_model(
            model_id=model_id,  # Use the provided model_id parameter
            use_flash_attn=False,  # Disable Flash Attention to ensure attention matrices can be captured
            load_in_4bit=False,    # Don't use 4-bit quantization to allow gradient flow
            enable_gradients=True, # Enable gradients for all parameters
            device_map="auto"      # Or "cuda:0" for single GPU
        )
        
        # Verify attention implementation
        attn_impl = getattr(model.config, "_attn_implementation", "unknown")
        logger.info(f"Using attention implementation: {attn_impl}")
        
        # Step 2: Load and process image
        logger.info(f"Loading image from: {image_url}")
        image = load_image(image_url, resize_to=(336, 336))
        
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
        
        # Handle top_k parameter if provided
        if top_k is not None and hasattr(workflow, 'logit_backend') and workflow.logit_backend is not None:
            workflow.logit_backend.top_k = top_k
        
        # Add concepts to track if provided
        if concepts_to_track and hasattr(workflow, 'logit_backend') and workflow.logit_backend is not None:
            workflow.logit_backend.concepts = concepts_to_track
            logger.info(f"Tracking concepts: {concepts_to_track}")
        
        # Step 5: Prepare inputs
        logger.info(f"Preparing inputs with prompt: '{prompt_text}'")
        input_data = workflow.prepare_inputs(image, prompt_text)
        
        # Check available memory and adjust settings if needed
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            available_memory = torch.cuda.get_device_properties(current_device).total_memory - torch.cuda.memory_allocated(current_device)
            available_memory_gb = available_memory / (1024**3)
            logger.info(f"Available GPU memory before tracing: {available_memory_gb:.2f} GB")
            
            # Force ultra-low memory mode if memory is very limited
            if available_memory_gb < 2.0:
                logger.warning("Very limited GPU memory available. Forcing ultra-low memory mode.")
                low_memory_mode = True
                fallback_to_attention = True
                offload_to_cpu = True
        
        # Apply memory optimization settings
        if low_memory_mode and hasattr(workflow, 'backends'):
            for backend_mode, backend in workflow.backends.items():
                if hasattr(backend, 'cpu_offload'):
                    backend.cpu_offload = True
                    
                # Configure saliency backend for low memory
                if backend_mode == TraceMode.SALIENCY and hasattr(backend, 'compute_batch_saliency'):
                    original_compute = backend.compute_batch_saliency
                    backend.compute_batch_saliency = lambda target_indices, inputs, layer_batch_size=1: \
                        original_compute(target_indices, inputs, layer_batch_size=1, offload_tensors=True)
        
        # Step 6: Run the semantic tracing
        logger.info(f"Running semantic tracing with mode: {tracing_mode}")
        
        try:
            # Choose analysis approach based on parameters
            compute_ll_projections = not low_memory_mode  # Skip LogitLens in low memory mode
            
            if analyze_last_token:
                logger.info("Analyzing last token in prompt")
                results = workflow.analyze_last_token(
                    input_data=input_data,
                    mode=tracing_mode,
                    single_forward_pass=False,
                    compute_ll_projections=compute_ll_projections
                )
            elif target_token_idx is not None:
                logger.info(f"Analyzing specific token at index: {target_token_idx}")
                results = workflow.analyze_specific_token(
                    input_data=input_data,
                    token_idx=target_token_idx,
                    mode=tracing_mode,
                    single_forward_pass=False,
                    compute_ll_projections=compute_ll_projections
                )
            elif analyze_specific_indices:
                logger.info(f"Analyzing specific token indices: {analyze_specific_indices}")
                results = workflow.analyze_multiple_tokens(
                    input_data=input_data,
                    token_indices=analyze_specific_indices,
                    mode=tracing_mode,
                    single_forward_pass=False,
                    compute_ll_projections=compute_ll_projections
                )
            else:
                logger.info(f"Generating and analyzing {num_tokens} tokens")
                results = workflow.generate_and_trace(
                    input_data=input_data,
                    num_tokens=num_tokens,
                    mode=tracing_mode,
                    single_forward_pass=False,
                    compute_ll_projections=compute_ll_projections
                )
                
            # Check if results contain an error
            if isinstance(results, dict) and "error" in results and fallback_to_attention and tracing_mode == "saliency":
                logger.info("Falling back to attention mode after saliency failure")
                
                # Simplified fallback to attention mode
                if analyze_last_token:
                    results = workflow.analyze_last_token(
                        input_data=input_data, mode="attention", 
                        single_forward_pass=False, compute_ll_projections=compute_ll_projections
                    )
                elif target_token_idx is not None:
                    results = workflow.analyze_specific_token(
                        input_data=input_data, token_idx=target_token_idx, mode="attention",
                        single_forward_pass=False, compute_ll_projections=compute_ll_projections
                    )
                elif analyze_specific_indices:
                    results = workflow.analyze_multiple_tokens(
                        input_data=input_data, token_indices=analyze_specific_indices, mode="attention",
                        single_forward_pass=False, compute_ll_projections=compute_ll_projections
                    )
                else:
                    results = workflow.generate_and_trace(
                        input_data=input_data, num_tokens=num_tokens, mode="attention",
                        single_forward_pass=False, compute_ll_projections=compute_ll_projections
                    )
                
                if "error" not in results:
                    results["fallback_mode"] = "attention"
        
        except Exception as e:
            logger.error(f"Error during tracing: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "traceback": traceback.format_exc()}
        
        # Step 7: Generate visualization if needed
        if not skip_visualization and isinstance(results, dict) and "trace_results" in results:
            logger.info("Generating visualizations")
            vis_paths = []
            
            # Get all CSV files from results
            csv_paths = []
            trace_results = results["trace_results"]
            
            # Extract CSV paths
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
                    
                    try:
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
                    except Exception as viz_e:
                        logger.error(f"Error generating visualization for {csv_path}: {viz_e}")
            
            results["visualization_paths"] = vis_paths
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic tracing test: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}