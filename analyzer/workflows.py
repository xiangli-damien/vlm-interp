"""
End-to-end analysis workflow functions for VLM explainability.

Provides workflow functions for running complete analysis pipelines:
- Logit Lens analysis: projects hidden states through LM head to examine activations
- Saliency analysis: computes gradient-based saliency for attention flow analysis
"""

import torch
import os
import gc
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple
from analyzer.heatmap_visualizer import HeatmapVisualizer

# Import analyzing components
from analyzer.logit_lens import LLaVANextLogitLensAnalyzer
from analyzer.saliency import (
    calculate_saliency_scores,
    analyze_layerwise_saliency_flow
)

# Import utilities
from utils.data_utils import get_token_indices, load_image, build_conversation
from utils.model_utils import get_llm_attention_layer_names, load_model
from utils.hook_utils import GradientAttentionCapture
from utils.viz_utils import visualize_token_probabilities, visualize_information_flow
from analyzer.semantic_tracing import EnhancedSemanticTracer
from analyzer.semantic_tracing_visualizer import EnhancedSemanticTracingVisualizer
from analyzer.token_analyzer import TokenAnalyzer

def run_logit_lens_workflow(
    model: torch.nn.Module,
    processor: Any,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    concepts_to_track: Optional[List[str]] = None,
    selected_layers: Optional[List[int]] = None,
    output_dir: str = "logit_lens_analysis",
    cpu_offload: bool = True
) -> Dict[str, Any]:
    """
    Performs the complete logit lens analysis pipeline using LLaVANextLogitLensAnalyzer.

    Args:
        model: The loaded LLaVA-Next model instance
        processor: The corresponding processor instance
        image_source: PIL image, URL string, or local file path
        prompt_text: The text prompt for the model
        concepts_to_track: List of concept strings (e.g., "cat", "sign")
        selected_layers: Specific layer indices to analyze/visualize
        output_dir: Directory path to save analysis outputs
        cpu_offload: Whether to move intermediate tensors to CPU during extraction

    Returns:
        Dict containing analysis results ('token_probabilities', 'generated_text',
             'feature_mapping', 'concepts_tracked', 'visualization_paths', 'summary_path')
    """
    print("\n--- Starting LLaVA-Next Logit Lens Workflow ---")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    try:
        # 1. Instantiate the analyzer core
        analyzer = LLaVANextLogitLensAnalyzer(model, processor)

        # 2. Prepare Inputs
        input_data = analyzer.prepare_inputs(image_source, prompt_text)
        if not input_data: return {"error": "Input preparation failed."}
        results["feature_mapping"] = input_data["feature_mapping"]

        # 3. Extract Hidden States & Sample Text
        outputs = analyzer.extract_hidden_states(input_data["inputs"])
        if not outputs: return {"error": "Hidden state extraction failed."}
        results["generated_text"] = outputs["generated_text"]

        # 4. Prepare Concepts Dictionary
        concept_token_ids = {}
        concepts_actually_tracked = []
        if concepts_to_track:
            for concept in concepts_to_track:
                try:
                    token_ids = processor.tokenizer.encode(concept, add_special_tokens=False)
                    if token_ids:
                        concept_token_ids[concept] = token_ids
                        concepts_actually_tracked.append(concept)
                except Exception as e: print(f"  Warning: Error encoding concept '{concept}': {e}")
        results["concepts_tracked"] = concept_token_ids

        # 5. Extract Token Probabilities
        if not concept_token_ids:
             print("  No valid concepts to track. Skipping probability extraction and visualization.")
             token_probs = {}
             viz_paths = []
        else:
            token_probs = analyzer.extract_token_probabilities(
                input_data=input_data,
                outputs=outputs,
                concepts_to_track=concept_token_ids,
                cpu_offload=cpu_offload
            )
            results["token_probabilities"] = token_probs

            # 6. Visualize Probabilities
            print("  Calling visualization function...")
            viz_paths = visualize_token_probabilities(
                token_probs=token_probs,
                input_data=input_data,
                selected_layers=selected_layers,
                output_dir=output_dir
            )
            results["visualization_paths"] = viz_paths

        # 7. Save Summary File
        summary_path = os.path.join(output_dir, "analysis_summary.txt")
        print(f"  Saving analysis summary to: {summary_path}")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("LLaVA-Next Logit Lens Analysis Summary\n" + "=" * 35 + "\n\n")
                f.write(f"Model ID: {getattr(model.config, '_name_or_path', 'N/A')}\n")
                img_src_repr = image_source if isinstance(image_source, str) else f"PIL Image ({input_data['original_image'].size} {input_data['original_image'].mode})"
                f.write(f"Image Source: {img_src_repr}\n")
                f.write(f"Prompt: {prompt_text}\n\n")
                f.write(f"Concepts Tracked: {concepts_actually_tracked}\n")
                for concept, ids in concept_token_ids.items(): f.write(f"  - '{concept}': {ids}\n")
                f.write(f"\nGenerated Text Sample:\n{results.get('generated_text', 'N/A')}\n\n")
                f.write("Image Processing Information:\n")
                fm = results.get("feature_mapping", {})
                if fm:
                    orig_w, orig_h = fm.get('original_size', ('N/A', 'N/A')); f.write(f"  Original Size (WxH): ({orig_w}, {orig_h})\n")
                    padded_w, padded_h = fm.get('padded_dimensions', ('N/A', 'N/A')); f.write(f"  Padded Preview Size (WxH): ({padded_w}, {padded_h})\n")
                    f.write(f"  Raw Patch Size: {fm.get('patch_size', 'N/A')}\n")
                    if fm.get("base_feature"): f.write(f"  Base Feature Grid: {fm['base_feature'].get('grid', 'N/A')}\n")
                    if fm.get("patch_feature"): f.write(f"  Patch Feature Unpadded Grid: {fm['patch_feature'].get('grid_unpadded', 'N/A')}\n")
                else: f.write("  Feature mapping information not available.\n")
                f.write(f"\nLayers Analyzed: {'All' if selected_layers is None else selected_layers}\n")
                f.write(f"\nVisualizations saved to subdirectories in: {output_dir}\n")
                f.write(f"Number of visualization files generated: {len(viz_paths)}\n")
            results["summary_path"] = summary_path
        except Exception as e:
            print(f"    Error writing summary file: {e}")
            results["summary_path"] = None

        print(f"--- Logit Lens Workflow Complete ---")

    except Exception as e:
        print(f"Error during Logit Lens workflow: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    # Clean up large intermediate data
    if 'outputs' in locals(): del outputs
    if 'input_data' in locals(): del input_data
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results

def run_saliency_workflow(
    model: torch.nn.Module,
    processor: Any,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    num_tokens: int = 5,
    output_dir: str = "saliency_analysis",
    image_size: Tuple[int, int] = (336, 336),
    layer_batch_size: int = 2,
    save_plots: bool = True,
    top_k_image_tokens: Optional[int] = 10
) -> Dict[str, Any]:
    """
    Performs token-by-token generation with gradient-based saliency analysis workflow.

    Uses batched layer processing for memory efficiency during gradient computation.
    Calls visualization utilities internally if save_plots is True.

    Args:
        model: The VLM model instance (must have gradients enabled)
        processor: The corresponding processor instance
        image_source: URL, path, or PIL Image for analysis
        prompt_text: The text prompt
        num_tokens: Number of tokens to generate and analyze
        output_dir: Directory to save plots
        image_size: Target size for image loading
        layer_batch_size: Number of attention layers to compute gradients for in each batch
        save_plots: Whether to generate and save information flow plots

    Returns:
        Dict containing analysis results per token, generated sequence, timing, etc.
    """
    print(f"\n--- Starting Gradient-Based Saliency Workflow ---")
    print(f" Config: NumTokens={num_tokens}, ImgSize={image_size}, LayerBatch={layer_batch_size}")
    if save_plots: os.makedirs(output_dir, exist_ok=True); print(f" Plots will be saved to: {output_dir}")
    final_results = {"error": "Workflow did not complete."} # Default error state
    start_time = time.time()

    try:
        # --- 1. Initial Setup ---
        if not any(p.requires_grad for p in model.parameters()):
             raise RuntimeError("Gradients are not enabled on the model.")
        image = load_image(image_source, resize_to=image_size, verbose=False)
        conversation = build_conversation(prompt_text)
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        initial_input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        image_sizes = inputs.get("image_sizes")
        image_token_id = getattr(model.config, "image_token_index", 32000)
        text_indices, image_indices = get_token_indices(initial_input_ids, image_token_id)
        all_attn_layer_names = get_llm_attention_layer_names(model)
        if not all_attn_layer_names: raise ValueError("Could not find attention layers.")
        print(f" Found {len(all_attn_layer_names)} attention layers.")

        # --- 2. Helper for Batched Gradient Computation ---
        def generate_next_token_and_compute_saliency(current_input_ids):
            all_saliency_scores = {}; next_token_id = None; loss_val = None
            model.eval()
            with torch.no_grad():
                outputs_pred = model(input_ids=current_input_ids,
                                    attention_mask=torch.ones_like(current_input_ids),
                                    pixel_values=pixel_values,
                                    image_sizes=image_sizes,
                                    use_cache=True)
                logits = outputs_pred.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                loss_val = -log_probs[0, next_token_id].item()
                del outputs_pred, logits, log_probs; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            grad_capture = GradientAttentionCapture(cpu_offload=True)
            num_layers = len(all_attn_layer_names)
            for batch_start in range(0, num_layers, layer_batch_size):
                batch_end = min(batch_start + layer_batch_size, num_layers)
                current_layer_batch_names = all_attn_layer_names[batch_start:batch_end]
                
                model.zero_grad(set_to_none=True)
                grad_capture.register_hooks(model, current_layer_batch_names)
                with torch.enable_grad():
                    outputs_grad = model(input_ids=current_input_ids,
                                        attention_mask=torch.ones_like(current_input_ids),
                                        pixel_values=pixel_values,
                                        image_sizes=image_sizes,
                                        use_cache=False,
                                        output_attentions=True)
                    logits_grad = outputs_grad.logits[:, -1, :]
                    log_probs_grad = torch.log_softmax(logits_grad.float(), dim=-1)
                    loss = -log_probs_grad[0, next_token_id]
                try: loss.backward()
                except Exception as bk_err: print(f" Backward pass error: {bk_err}"); grad_capture.clear_hooks(); raise
                
                batch_saliency = grad_capture.compute_saliency()
                all_saliency_scores.update(batch_saliency)
                grad_capture.clear_hooks()
                
                del outputs_grad, logits_grad, log_probs_grad, loss
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            model.eval()
            return next_token_id, all_saliency_scores, loss_val

        # --- 3. Token-by-Token Generation and Analysis Loop ---
        token_results = {}
        generated_sequence = ""
        current_input_ids = initial_input_ids.clone()
        loop_start_time = time.time()
        
        for step_idx in range(num_tokens):
            print(f"--- Analyzing Saliency Token {step_idx+1}/{num_tokens} ---")
            try:
                next_token_id, current_saliency_scores, loss_val = generate_next_token_and_compute_saliency(current_input_ids)
                if next_token_id is None: print("  Failed to generate next token. Stopping."); break
                
                new_token_text = processor.tokenizer.decode([next_token_id.item()])
                print(f"  Generated token: '{new_token_text}' (ID: {next_token_id.item()}, Loss: {loss_val:.4f})")
                
                target_idx = current_input_ids.shape[1] - 1 # Target is last token of input
                flow_metrics = analyze_layerwise_saliency_flow(
                    current_saliency_scores, text_indices, image_indices, target_idx, 
                    cpu_offload=True, top_k_image_tokens=top_k_image_tokens  # Pass the top-k parameter
                )
                
                if save_plots and flow_metrics:
                    model_name_short = model.config._name_or_path.split('/')[-1]
                    plot_filename = f"token_{(step_idx+1):02d}_{new_token_text.strip().replace(' ', '_')}_saliency_flow.png"
                    plot_path = os.path.join(output_dir, plot_filename)
                    visualize_information_flow(
                        flow_metrics, 
                        f"{model_name_short} - Saliency Flow for Token {step_idx+1} ('{new_token_text}')",
                        plot_path,
                        use_top_k=top_k_image_tokens is not None  # Add parameter to use top-k in visualization
                    )
                
                token_results[f"token_{step_idx+1}"] = {
                    "token_text": new_token_text,
                    "token_id": next_token_id.item(),
                    "loss": loss_val,
                    "metrics": flow_metrics
                }
                
                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)
                generated_sequence += new_token_text
                
                # Cleanup
                del current_saliency_scores, flow_metrics
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error analyzing token {step_idx+1}: {e}")
                import traceback
                traceback.print_exc()
                break
                
        loop_end_time = time.time()

        final_results = {
            "token_results": token_results,
            "sequence_text": generated_sequence,
            "total_time": loop_end_time - loop_start_time,
            "model_name": model.config._name_or_path,
            "config": {
                "num_tokens": num_tokens,
                "image_size": image_size,
                "layer_batch_size": layer_batch_size,
                "top_k_image_tokens": top_k_image_tokens,  # Add to config
                "prompt": prompt_text,
                "image_source": image_source if isinstance(image_source, str) else "PIL Input"
            }
        }
        print(f"\n--- Saliency Workflow Finished ({final_results['total_time']:.2f} seconds) ---")

    except Exception as e:
        print(f"Error during Saliency workflow: {e}")
        import traceback
        traceback.print_exc()
        final_results["error"] = str(e)

    return final_results

def run_semantic_tracing_analysis(
    model_id: str,
    image_path: str, 
    prompt: str,
    output_dir: str,
    beta_target: float = 0.8,
    beta_layer: float = 0.7,
    min_keep: int = 1,
    max_keep: int = 30,
    min_keep_layer: int = 5,
    max_keep_layer: int = 100,
    top_k: Optional[int] = None,
    num_tokens: int = 1,
    load_in_4bit: bool = True,
    target_token_idx: Optional[int] = None,
    analyze_specific_indices: Optional[List[int]] = None,  # New parameter
    generate_only: bool = False,  # New parameter
    image_size: Tuple[int, int] = (336, 336),
    concepts_to_track: Optional[List[str]] = None,
    normalize_weights: bool = True,
    single_forward_pass: bool = False,
    analyze_last_token: bool = False,
    tracing_mode: str = "saliency",
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
        top_k: Deprecated parameter (use beta_target instead)
        num_tokens: Number of tokens to generate and analyze
        load_in_4bit: Whether to load the model in 4-bit precision
        target_token_idx: Index of specific token to analyze (if None, uses first generated token)
        analyze_specific_indices: List of specific token indices to analyze
        generate_only: If True, only generate tokens without analysis
        image_size: Size to resize the image to
        concepts_to_track: List of concepts to track with logit lens
        normalize_weights: Whether to normalize token importance weights between layers
        single_forward_pass: Use one forward pass for all layers (reduces memory usage)
        analyze_last_token: Whether to analyze the last token in the given prompt
        tracing_mode: The tracing mode to use ("saliency", "attention", or "both")
        debug_mode: Whether to print detailed debug information
    
    Returns:
        Dictionary with paths to saved CSV files and metadata
    """
    from utils.data_utils import load_image
    from utils.model_utils import load_model
    
    start_time = time.time()
    print(f"=== Starting Semantic Tracing Analysis ===")
    print(f"Model: {model_id}")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}")
    print(f"Output Directory: {output_dir}")
    print(f"Tracing Mode: {tracing_mode}")
    
    # Create organized output directory structure
    os.makedirs(output_dir, exist_ok=True)
    csv_dir = os.path.join(output_dir, "csv_data")
    os.makedirs(csv_dir, exist_ok=True)
    
    # Print analysis mode information
    if generate_only:
        mode_str = f"Generating {num_tokens} tokens without analysis"
    elif analyze_specific_indices:
        mode_str = f"Generating {num_tokens} tokens and analyzing specific indices: {analyze_specific_indices}"
    elif analyze_last_token:
        mode_str = "Analyzing last token in prompt"
    else:
        mode_str = f"Analyzing {num_tokens} token(s)" + (" with single forward pass" if single_forward_pass else "")
    print(f"Mode: {mode_str}")
    
    # Print node selection parameters
    print(f"Coverage Parameters: β_target={beta_target}, β_layer={beta_layer}")
    print(f"Node Limits: min_keep={min_keep}, max_keep={max_keep}, min_keep_layer={min_keep_layer}, max_keep_layer={max_keep_layer}")
    
    # 1. Load model and processor
    print("\nLoading model and processor...")
    model, processor = load_model(
        model_id=model_id,
        load_in_4bit=load_in_4bit, 
        enable_gradients=True  # Always enable gradients for saliency tracing
    )
    
    # 2. Load image
    print("\nLoading image...")
    image = load_image(image_path, resize_to=image_size)
    
    # 3. Create Enhanced semantic tracer
    tracer = EnhancedSemanticTracer(
        model=model,
        processor=processor,
        top_k=top_k,  # Pass for backward compatibility
        output_dir=csv_dir,  # Use the CSV directory
        logit_lens_concepts=concepts_to_track,
        normalize_weights=normalize_weights,
        beta_target=beta_target,
        beta_layer=beta_layer,
        min_keep=min_keep,
        max_keep=max_keep,
        min_keep_layer=min_keep_layer,
        max_keep_layer=max_keep_layer,
        debug=debug_mode
    )
    
    # 4. Prepare inputs
    print("\nPreparing inputs...")
    input_data = tracer.prepare_inputs(image, prompt)
    
    # 5. Run semantic tracing based on the selected mode
    print("\nRunning semantic tracing...")
    
    if generate_only:
        # Generate tokens without analysis (for faster generation)
        model.eval()
        inputs = input_data["inputs"]
        current_input_ids = inputs["input_ids"].clone()
        original_seq_len = current_input_ids.shape[1]
        
        with torch.no_grad():
            for _ in range(num_tokens):
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"}, use_cache=True)
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
                
                # Update inputs for next iteration
                inputs["input_ids"] = current_input_ids
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.ones_like(current_input_ids)
        
        # Decode the complete sequence
        generated_ids = current_input_ids[0, original_seq_len:].tolist()
        generated_text = processor.tokenizer.decode(generated_ids)
        
        # Create results without trace data
        trace_results = {
            "full_sequence": {
                "ids": current_input_ids[0].tolist(),
                "text": processor.tokenizer.decode(current_input_ids[0].tolist()),
                "generated_text": generated_text,
                "prompt_length": original_seq_len
            },
            "generated_tokens": [
                {
                    "index": original_seq_len + i,
                    "id": token_id,
                    "text": processor.tokenizer.decode([token_id])
                }
                for i, token_id in enumerate(generated_ids)
            ]
        }
    elif analyze_specific_indices:
        # Generate all tokens first, then analyze specific indices
        trace_results = tracer.generate_all_then_analyze_specific(
            input_data=input_data,
            num_tokens=num_tokens,
            analyze_indices=analyze_specific_indices,
            tracing_mode=tracing_mode,
            batch_compute=not single_forward_pass
        )
    elif analyze_last_token:
        # Analyze the last token in the prompt directly
        trace_results = tracer.analyze_last_token(
            input_data=input_data,
            single_forward_pass=single_forward_pass,
            tracing_mode=tracing_mode
        )
    elif num_tokens > 1 and target_token_idx is None:
        # Analyze multiple consecutively generated tokens
        trace_results = tracer.generate_and_analyze_multiple(
            input_data=input_data,
            num_tokens=num_tokens,
            batch_compute=not single_forward_pass,
            tracing_mode=tracing_mode
        )

    else:
        # Standard single token analysis
        trace_results = tracer.generate_and_analyze(
            input_data=input_data,
            target_token_idx=target_token_idx,
            num_tokens=num_tokens if target_token_idx is None else 1,
            batch_compute=not single_forward_pass,
            tracing_mode=tracing_mode
        )
    
    # tracer.plot_attention_vs_saliency(save_path="semantic_tracing_results/att_vs_sal.png")
    
    # Prepare output information
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
                    if isinstance(mode_value, dict) and "trace_data_path" in mode_value:
                        # This is a tracing mode result
                        output_info["csv_files"].append(mode_value["trace_data_path"])
                    elif isinstance(mode_value, dict):
                        # This is a token results with potentially multiple tracing modes
                        for subkey, subvalue in mode_value.items():
                            if isinstance(subvalue, dict) and "trace_data_path" in subvalue:
                                output_info["csv_files"].append(subvalue["trace_data_path"])
    
    # Extract target token information
    if "target_tokens" in trace_results:
        output_info["target_tokens"] = trace_results["target_tokens"]
    elif "target_token" in trace_results:
        output_info["target_tokens"] = [trace_results["target_token"]]
    
    # Save analysis summary
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
                "analyze_last_token": analyze_last_token,
                "single_forward_pass": single_forward_pass,
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
    print(f"CSV data saved to: {csv_dir}")
    
    if not generate_only:
        print(f"Generated {len(output_info['csv_files'])} trace data files")
    else:
        print(f"Generated {len(output_info.get('all_generated_tokens', []))} tokens without analysis")
    
    # Clean up memory
    del model, processor, tracer, input_data, trace_results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_info

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
    min_keep_layer: int = 1,
    max_keep_layer: int = 500,
    concepts_to_track: Optional[List[str]] = None,
    load_in_4bit: bool = True,
    analyze_last_token: bool = False,
    single_forward_pass: bool = False,
    target_token_idx: Optional[int] = None,
    analyze_specific_indices: Optional[List[int]] = None,
    generate_first: bool = False,
    tracing_mode: str = "both",
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Run semantic tracing analysis to generate trace CSV data without visualization.
    
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
        generate_first: Two-step process: first generate all tokens, then analyze specific indices
        tracing_mode: The tracing mode to use ("saliency", "attention", or "both")
        debug_mode: Whether to print detailed debug information
        
    Returns:
        Dictionary with CSV file paths and metadata for later visualization
    """
    # Import necessary functions
    from utils.data_utils import load_image
    from utils.model_utils import load_model
    from analyzer.semantic_tracing import EnhancedSemanticTracer
    
    # Extract short model name for output directory
    model_short_name = model_id.split('/')[-1].replace("-hf", "")
    
    # Set default output directory if not provided
    if output_dir is None:
        base_dir = os.path.join(os.getcwd(), "results")
        output_dir = os.path.join(base_dir, f"semantic_tracing_{model_short_name}")
    
    # Define default concepts if none provided
    if concepts_to_track is None:
        concepts_to_track = ["person", "object", "building", "nature", "sign", "color"]
    
    # Print analysis configuration
    print(f"\n{'='*50}")
    print(f"RUNNING SEMANTIC TRACING TEST WITH {model_id.split('/')[-1].upper()}")
    print(f"{'='*50}")
    print(f"Model: {model_id}")
    print(f"Image: {image_url}")
    print(f"Prompt: {prompt_text}")
    print(f"Output Directory: {output_dir}")
    print(f"Tokens to generate: {num_tokens}")
    
    # Print specific token analysis info if applicable
    if analyze_specific_indices:
        print(f"Analyzing specific token indices: {analyze_specific_indices}")
    elif target_token_idx is not None:
        print(f"Analyzing specific token index: {target_token_idx}")
    elif analyze_last_token:
        print("Analyzing last token in prompt")
    else:
        print(f"Analyzing all generated tokens")
    
    print(f"Tracing mode: {tracing_mode}")
    print(f"Coverage Parameters: β_target={beta_target}, β_layer={beta_layer}")
    print(f"Node Limits: min_keep={min_keep}, max_keep={max_keep}, min_keep_layer={min_keep_layer}, max_keep_layer={max_keep_layer}")
    print(f"Using single forward pass: {single_forward_pass}")
    
    # Create the visualizer for image handling
    visualizer = EnhancedSemanticTracingVisualizer(output_dir=output_dir)
    
    # First, ensure we have a local copy of the image
    image_dir = os.path.join(output_dir, "downloaded_images")
    local_image_path = visualizer.get_local_image_path(image_url, image_dir)
    
    if not local_image_path:
        print(f"Error: Could not obtain local image from {image_url}")
        return {"error": f"Image not available: {image_url}"}
    
    # Clean memory
    try:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Memory cleaned before starting.")
    except ImportError:
        print("Warning: Unable to clean memory.")
    
    start_time = time.time()
    
    try:
        # Two-step process if requested
        if generate_first:
            print("\n=== STEP 1: Generating complete sequence ===")
            # First step: Generate all tokens without analysis
            generation_results = run_semantic_tracing_analysis(
                model_id=model_id,
                image_path=local_image_path,
                prompt=prompt_text,
                output_dir=output_dir,
                num_tokens=num_tokens,
                load_in_4bit=load_in_4bit,
                generate_only=True  # Only generate, don't analyze
            )
            
            # Extract token information from generation
            all_tokens = generation_results.get("all_generated_tokens", [])
            prompt_length = generation_results.get("full_sequence", {}).get("prompt_length", 0)
            
            if all_tokens:
                print(f"\nGenerated {len(all_tokens)} tokens:")
                for i, token in enumerate(all_tokens):
                    if i < 10 or i >= len(all_tokens) - 5:  # Show first 10 and last 5
                        print(f"  {i}: '{token['text']}' at position {token['index']} (ID: {token['id']})")
                    elif i == 10:
                        print(f"  ... ({len(all_tokens) - 15} more tokens) ...")
                
                # If no specific indices provided, prompt the user
                if analyze_specific_indices is None:
                    print("\nNo specific token indices provided for analysis.")
                    print("Please provide specific indices in the 'analyze_specific_indices' parameter.")
                    analyze_specific_indices = [prompt_length]  # Default to analyzing the first generated token
            
            print("\n=== STEP 2: Analyzing specific token indices ===")
            if analyze_specific_indices:
                print(f"Analyzing indices: {analyze_specific_indices}")
            else:
                print("No valid indices to analyze.")
                return generation_results
            
            # Second step: Analyze specific tokens
            analysis_results = run_semantic_tracing_analysis(
                model_id=model_id,
                image_path=local_image_path,
                prompt=prompt_text,
                output_dir=output_dir,
                beta_target=beta_target,
                beta_layer=beta_layer,
                min_keep=min_keep,
                max_keep=max_keep,
                min_keep_layer=min_keep_layer,
                max_keep_layer=max_keep_layer,
                num_tokens=num_tokens,
                load_in_4bit=load_in_4bit,
                concepts_to_track=concepts_to_track,
                normalize_weights=True,
                single_forward_pass=single_forward_pass,
                analyze_specific_indices=analyze_specific_indices,
                tracing_mode=tracing_mode,
                debug_mode=debug_mode
            )
        else:
            # Single-step process: traditional analysis
            analysis_results = run_semantic_tracing_analysis(
                model_id=model_id,
                image_path=local_image_path,
                prompt=prompt_text,
                output_dir=output_dir,
                beta_target=beta_target,
                beta_layer=beta_layer,
                min_keep=min_keep,
                max_keep=max_keep,
                min_keep_layer=min_keep_layer,
                max_keep_layer=max_keep_layer,
                num_tokens=num_tokens,
                load_in_4bit=load_in_4bit,
                concepts_to_track=concepts_to_track,
                normalize_weights=True,
                single_forward_pass=single_forward_pass,
                analyze_last_token=analyze_last_token,
                target_token_idx=target_token_idx,
                analyze_specific_indices=analyze_specific_indices,
                tracing_mode=tracing_mode,
                debug_mode=debug_mode
            )
        
        # Prepare results to return
        test_results = {
            "analysis_results": analysis_results,
            "visualization_paths": [],
            "image_path": local_image_path,
            "tracing_mode": tracing_mode
        }
        
        # Check if we have valid CSV files
        if not analysis_results.get("csv_files"):
            print("\nNo CSV files generated for visualization. Analysis may have failed.")
            test_results["error"] = "No CSV files generated"
            return test_results
        
        # Calculate and add total time
        total_time = time.time() - start_time
        test_results["total_time"] = total_time
        
        # Print summary information
        print(f"\n=== Test Complete ===")
        print(f"Total time: {total_time:.2f} seconds")
        
        # Display results summary
        if "target_tokens" in analysis_results:
            print(f"\nAnalyzed {len(analysis_results['target_tokens'])} tokens:")
            for i, token_info in enumerate(analysis_results['target_tokens']):
                print(f"  {i+1}. '{token_info['text']}' at position {token_info['index']}")
        
        if "full_sequence" in analysis_results and "text" in analysis_results["full_sequence"]:
            print(f"\nGenerated text: {analysis_results['full_sequence']['text']}")
        
        return test_results
        
    except Exception as e:
        print(f"Error in semantic tracing test: {e}")
        import traceback
        traceback.print_exc()
        
        return {"error": str(e), "traceback": traceback.format_exc()}
    
    finally:
        # Clean memory again
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def build_heatmaps_offline(
        trace_csv: str,
        metadata_json: str,
        image_path: str,
        output_dir: str,
        weight_column: str = "importance_weight",
        composite_only: bool = True,
        unified_colorscale: bool = False,
        download_resize_image: bool = False,
        target_image_size: Optional[Tuple[int, int]] = None,
        include_all_token_types: bool = False,
        debug: bool = False
    ):
    """
    Generate comprehensive heatmap visualizations from semantic tracing data.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 调试: 验证图像路径
    print(f"DEBUG: Image path: {image_path}")
    if os.path.exists(image_path):
        print(f"DEBUG: Image exists with size: {os.path.getsize(image_path)} bytes")
        # 尝试打开图像来验证它是否有效
        try:
            from PIL import Image
            img_test = Image.open(image_path)
            print(f"DEBUG: Image opened successfully: size={img_test.size}, mode={img_test.mode}")
            # 显示样本像素值以确认图像有效
            #img_array = np.array(img_test)
            #print(f"DEBUG: Image array shape={img_array.shape}, min/max={img_array.min()}/{img_array.max()}")
        except Exception as e:
            print(f"DEBUG: Failed to open image: {e}")
    else:
        print(f"DEBUG: Image does not exist at path: {image_path}")
        if image_path.startswith(('http://', 'https://')):
            print(f"DEBUG: Image path appears to be a URL")
    
    # Handle image download and resize if needed
    processed_image_path = image_path
    if download_resize_image and image_path.startswith(('http://', 'https://')):
        # Create a temporary directory for downloaded images
        download_dir = os.path.join(output_dir, "downloaded_images")
        os.makedirs(download_dir, exist_ok=True)
        temp_image_path = os.path.join(download_dir, os.path.basename(image_path) or "downloaded_image.jpg")
        
        try:
            # Download the image
            import requests
            from PIL import Image
            import io
            
            print(f"Downloading image from {image_path}...")
            response = requests.get(image_path, stream=True, timeout=10)
            response.raise_for_status()
            
            # Load the image
            img = Image.open(io.BytesIO(response.content))
            print(f"DEBUG: Downloaded image: size={img.size}, mode={img.mode}")
            
            # Resize if target size is provided
            if target_image_size:
                img = img.resize(target_image_size, Image.LANCZOS)
                print(f"Resized image to {target_image_size}")
            
            # Save the processed image
            img.save(temp_image_path)
            print(f"Image saved to {temp_image_path}")
            
            # Update image path to the local file
            processed_image_path = temp_image_path
        except Exception as e:
            print(f"Error downloading or processing image: {e}")
            # Continue with original image path
    
    # DEBUG: 检查元数据文件
    print(f"DEBUG: Checking metadata file: {metadata_json}")
    try:
        import json
        with open(metadata_json, 'r') as f:
            metadata = json.load(f)
        feature_mapping = metadata.get("feature_mapping", {})
        if feature_mapping:
            if "base_feature" in feature_mapping:
                print(f"DEBUG: base_feature grid: {feature_mapping['base_feature'].get('grid')}")
            if "patch_feature" in feature_mapping:
                print(f"DEBUG: patch_feature grid: {feature_mapping['patch_feature'].get('grid_unpadded')}")
            if "resized_dimensions" in feature_mapping:
                print(f"DEBUG: resized_dimensions: {feature_mapping['resized_dimensions']}")
    except Exception as e:
        print(f"DEBUG: Error reading metadata: {e}")
    
    # First attempt: standard approach
    try:
        # IMPROVED: Add debug information before creating visualizer
        print(f"Reading trace CSV: {trace_csv}")
        # Read a few sample rows to inspect data
        if debug:
            import pandas as pd
            sample_df = pd.read_csv(trace_csv, nrows=5)
            print(f"CSV sample columns: {list(sample_df.columns)}")
            if weight_column in sample_df.columns:
                print(f"Sample weight values: {sample_df[weight_column].tolist()}")
            else:
                print(f"Warning: Weight column '{weight_column}' not found in CSV")
        
        # Create visualizer with custom parameters
        hv = HeatmapVisualizer(
            csv_path=trace_csv,
            metadata_path=metadata_json,
            image_path=processed_image_path,
            out_dir=output_dir,
            weight_column=weight_column,
            debug_mode=True  # 启用调试模式
        )
        
        # Set custom visualization parameters
        hv.unified_colorscale = unified_colorscale
        hv.include_all_token_types = include_all_token_types
        
        # 创建一个简单的直接可视化，用于调试确认图像加载正确
        try:
            from PIL import Image
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 创建调试目录
            debug_dir = os.path.join(output_dir, "debug_images")
            os.makedirs(debug_dir, exist_ok=True)
            
            # 加载图像
            test_img = Image.open(processed_image_path).convert('RGB')
            debug_img_path = os.path.join(debug_dir, "debug_direct_image.png")
            
            # 保存一个直接的图像用于检查
            plt.figure(figsize=(8, 8))
            plt.imshow(np.array(test_img), origin='upper')
            plt.title("Direct Image Test")
            plt.axis('off')
            plt.savefig(debug_img_path, dpi=150)
            plt.close()
            
            print(f"DEBUG: Saved direct image test to {debug_img_path}")
            
            # 创建一个简单的热力图叠加测试
            H, W = test_img.size[1], test_img.size[0]
            # 创建一个简单的测试热力图
            test_heat = np.random.rand(24, 24)
            # 上采样热力图到图像大小
            test_heat_up = np.kron(test_heat, np.ones((H//24, W//24)))
            if test_heat_up.shape[0] < H or test_heat_up.shape[1] < W:
                pad_h = max(0, H - test_heat_up.shape[0])
                pad_w = max(0, W - test_heat_up.shape[1])
                test_heat_up = np.pad(test_heat_up, ((0, pad_h), (0, pad_w)), mode='constant')
            test_heat_up = test_heat_up[:H, :W]
            
            # 保存一个直接的热力图叠加用于检查
            debug_overlay_path = os.path.join(debug_dir, "debug_direct_overlay.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(np.array(test_img), origin='upper')
            plt.imshow(test_heat_up, cmap='hot', alpha=0.5, origin='upper')
            plt.title("Direct Overlay Test")
            plt.axis('off')
            plt.savefig(debug_overlay_path, dpi=150)
            plt.close()
            
            print(f"DEBUG: Saved direct overlay test to {debug_overlay_path}")
            
        except Exception as e:
            print(f"DEBUG: Direct visualization test failed: {e}")
        
        # Run visualization and collect output files
        files = hv.run(composite_only=composite_only, show_values=not composite_only)
        
        if files:
            print(f"Successfully generated {len(files)} visualization files")
            return files
        
        # If no files were generated, try fallback approach
        print("No visualization files generated. Trying fallback approach...")
        raise ValueError("Initial approach failed to generate any files")
        
    except Exception as e:
        print(f"Visualization error: {e}")
        
        # ADDED: Fallback approach - try with preprocessed CSV
        try:
            print("\nTrying fallback approach with preprocessed CSV...")
            
            # 创建一个直接的可视化测试
            direct_viz_dir = os.path.join(output_dir, "direct_viz")
            os.makedirs(direct_viz_dir, exist_ok=True)
            
            # 加载图像并创建一个直接的matplotlib可视化
            try:
                from PIL import Image
                import matplotlib.pyplot as plt
                import numpy as np
                
                direct_img = Image.open(processed_image_path).convert('RGB')
                direct_output_path = os.path.join(direct_viz_dir, "direct_combined.png")
                
                # 创建一个简单的热力图模板
                h, w = direct_img.size[1], direct_img.size[0]
                heat_template = np.ones((h//20, w//20)) * 0.5
                for i in range(heat_template.shape[0]):
                    for j in range(heat_template.shape[1]):
                        heat_template[i, j] = max(0.1, np.cos(i/5) * np.sin(j/5) + 0.5)
                
                heat_big = np.kron(heat_template, np.ones((20, 20)))
                heat_big = heat_big[:h, :w]
                
                plt.figure(figsize=(10, 10))
                plt.imshow(np.array(direct_img), origin='upper')
                plt.imshow(heat_big, alpha=0.6, cmap='hot', origin='upper')
                plt.title("Direct Fallback Visualization")
                plt.axis('off')
                plt.savefig(direct_output_path, dpi=150)
                plt.close()
                
                print(f"Created direct fallback visualization at {direct_output_path}")
            except Exception as e_direct:
                print(f"Direct visualization fallback failed: {e_direct}")
            
            # Read and preprocess the CSV
            import pandas as pd
            df = pd.read_csv(trace_csv)
            
            # Fix weight column
            if weight_column in df.columns:
                # Force numeric conversion
                df[weight_column] = pd.to_numeric(df[weight_column], errors="coerce").fillna(0)
                
                # Ensure non-zero values for visibility
                if df[weight_column].max() <= 0.01:
                    print(f"Adding visibility offset to weights (all <= 0.01)")
                    df[weight_column] = df[weight_column] + 0.1
            
            # Filter out problematic rows
            img_tokens = df[df["token_type"] == 2]
            if img_tokens.empty:
                print("Warning: No image tokens found in trace data")
            else:
                print(f"Found {len(img_tokens)} image tokens")
                
                # Log token counts by layer
                layer_counts = img_tokens.groupby('layer').size()
                print(f"Image tokens per layer: {layer_counts.to_dict()}")
                
                # Log weight statistics
                if weight_column in img_tokens.columns:
                    weight_stats = img_tokens[weight_column].describe()
                    print(f"Weight statistics: min={weight_stats['min']:.4f}, max={weight_stats['max']:.4f}, mean={weight_stats['mean']:.4f}")
            
            # Save preprocessed data
            preprocessed_csv = os.path.join(output_dir, "preprocessed_trace.csv")
            df.to_csv(preprocessed_csv, index=False)
            
            # Create visualizer with preprocessed data
            print(f"Creating visualizer with preprocessed data")
            hv_fallback = HeatmapVisualizer(
                csv_path=preprocessed_csv,
                metadata_path=metadata_json,
                image_path=processed_image_path,
                out_dir=os.path.join(output_dir, "fallback"),
                weight_column=weight_column,
                debug_mode=True  # Always enable debug for fallback
            )
            
            # Disable unified colorscale for better visibility
            hv_fallback.unified_colorscale = False
            
            # Run visualization
            fallback_files = hv_fallback.run(composite_only=composite_only, show_values=True)
            
            if fallback_files:
                print(f"Fallback approach generated {len(fallback_files)} visualization files")
                return fallback_files
            
        except Exception as e2:
            print(f"Fallback approach failed: {e2}")
    
    print("All visualization attempts failed. No heatmaps generated.")
    return []

def analyze_token_thresholds(
    model: torch.nn.Module,
    processor: Any,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    trace_data_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    output_dir: str = "token_threshold_analysis",
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Analyzes threshold statistics to determine appropriate parameters for token role identification.
    
    Args:
        model: The loaded VLM model instance
        processor: The corresponding processor instance
        image_source: PIL image, URL string, or local file path
        prompt_text: The text prompt for the model
        trace_data_path: Path to pre-existing trace CSV (if None, new trace will be run)
        metadata_path: Path to trace metadata JSON (if None, metadata from trace will be used)
        output_dir: Directory path to save analysis outputs
        debug_mode: Whether to print detailed debug information
        
    Returns:
        Dict containing threshold statistics and recommendations
    """
    print("\n--- Starting Token Threshold Analysis ---")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    try:
        # 1. Check if we need to run semantic tracing or use provided data
        if trace_data_path is None or not os.path.exists(trace_data_path):
            print("No trace data provided. Running semantic tracing...")
            
            # Import semantic tracing components
            from analyzer.semantic_tracing import EnhancedSemanticTracer
            
            # Run semantic tracing
            tracer = EnhancedSemanticTracer(
                model=model,
                processor=processor,
                output_dir=output_dir,
                debug=debug_mode
            )
            
            # Prepare inputs
            input_data = tracer.prepare_inputs(image_source, prompt_text)
            
            # Run tracing
            trace_results = tracer.generate_and_analyze(
                input_data=input_data,
                num_tokens=1,  # Just analyze the first generated token
                tracing_mode="saliency"  # Use saliency tracing
            )
            
            # Get paths to trace data
            if "trace_results" in trace_results:
                if "saliency" in trace_results["trace_results"]:
                    trace_data_path = trace_results["trace_results"]["saliency"].get("trace_data_path")
                elif "attention" in trace_results["trace_results"]:
                    trace_data_path = trace_results["trace_results"]["attention"].get("trace_data_path")
            
            # Get metadata path
            metadata_path = trace_results.get("metadata_path")
            
            # Store original trace results
            results["trace_results"] = trace_results
            
            # Ensure we have trace data
            if trace_data_path is None or not os.path.exists(trace_data_path):
                print("Error: Trace data not generated correctly")
                return {"error": "Trace data not generated correctly"}
            
            print(f"Generated trace data: {trace_data_path}")
            
        else:
            print(f"Using provided trace data: {trace_data_path}")
        
        # 2. Initialize token analyzer
        analyzer = TokenAnalyzer(
            output_dir=output_dir,
            debug_mode=debug_mode
        )
        
        # 3. Analyze threshold statistics
        threshold_stats = analyzer.analyze_threshold_statistics(
            csv_path=trace_data_path,
            metadata_path=metadata_path
        )
        
        # Store threshold statistics
        results["threshold_stats"] = threshold_stats
        
        # 4. Print recommended thresholds
        if "recommended_thresholds" in threshold_stats:
            rec = threshold_stats["recommended_thresholds"]
            print("\nRecommended Parameter Values:")
            print(f"  receptivity_threshold = {rec['receptivity_threshold']:.4f}")
            print(f"  emission_threshold = {rec['emission_threshold']:.4f}")
            print(f"  preservation_threshold = {rec['preservation_threshold']:.4f}")
        
        print(f"\n--- Token Threshold Analysis Complete ---")
        return results

    except Exception as e:
        print(f"Error during token threshold analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def run_token_analysis(
    model: torch.nn.Module,
    processor: Any,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    trace_data_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    output_dir: str = "token_analysis_results",
    receptivity_threshold: float = 0.7,
    emission_threshold: float = 0.7,
    preservation_threshold: float = 0.5,
    importance_window: int = 3,
    identify_critical_tokens: bool = True,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Performs token role analysis on semantic tracing data to identify critical tokens.
    Optimized version without network metrics and visualizations.
    
    Args:
        model: The loaded VLM model instance
        processor: The corresponding processor instance
        image_source: PIL image, URL string, or local file path
        prompt_text: The text prompt for the model
        trace_data_path: Path to pre-existing trace CSV (if None, new trace will be run)
        metadata_path: Path to trace metadata JSON (if None, metadata from trace will be used)
        output_dir: Directory path to save analysis outputs
        receptivity_threshold: Threshold for identifying reception nodes
        emission_threshold: Threshold for identifying emission nodes
        preservation_threshold: Threshold for identifying preservation nodes
        importance_window: Number of layers to consider for importance
        identify_critical_tokens: Whether to identify and rank critical tokens
        debug_mode: Whether to print detailed debug information
        
    Returns:
        Dict containing analysis results
    """
    print("\n--- Starting Token Analysis ---")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    try:
        # 1. Check if we need to run semantic tracing or use provided data
        if trace_data_path is None or not os.path.exists(trace_data_path):
            print("No trace data provided. Running semantic tracing...")
            
            # Import semantic tracing components
            from analyzer.semantic_tracing import EnhancedSemanticTracer
            
            # Run semantic tracing
            tracer = EnhancedSemanticTracer(
                model=model,
                processor=processor,
                output_dir=output_dir,
                debug=debug_mode
            )
            
            # Prepare inputs
            input_data = tracer.prepare_inputs(image_source, prompt_text)
            
            # Run tracing
            trace_results = tracer.generate_and_analyze(
                input_data=input_data,
                num_tokens=1,  # Just analyze the first generated token
                tracing_mode="saliency"  # Use saliency tracing
            )
            
            # Get paths to trace data
            if "trace_results" in trace_results:
                if "saliency" in trace_results["trace_results"]:
                    trace_data_path = trace_results["trace_results"]["saliency"].get("trace_data_path")
                elif "attention" in trace_results["trace_results"]:
                    trace_data_path = trace_results["trace_results"]["attention"].get("trace_data_path")
            
            # Get metadata path
            metadata_path = trace_results.get("metadata_path")
            
            # Store original trace results
            results["trace_results"] = trace_results
            
            # Ensure we have trace data
            if trace_data_path is None or not os.path.exists(trace_data_path):
                print("Error: Trace data not generated correctly")
                return {"error": "Trace data not generated correctly"}
            
            print(f"Generated trace data: {trace_data_path}")
            
        else:
            print(f"Using provided trace data: {trace_data_path}")
        
        # 2. Initialize token analyzer with specified thresholds
        analyzer = TokenAnalyzer(
            output_dir=output_dir,
            receptivity_threshold=receptivity_threshold,
            emission_threshold=emission_threshold,
            preservation_threshold=preservation_threshold,
            importance_window=importance_window,
            debug_mode=debug_mode
        )
        
        # 3. Analyze trace data
        analysis_results = analyzer.analyze_trace_data(
            csv_path=trace_data_path,
            metadata_path=metadata_path,
            analyze_thresholds_first=False  # Already using provided thresholds
        )
        
        # Store token analysis results
        results["token_analysis"] = analysis_results
        
        # 4. Identify critical tokens if requested
        if identify_critical_tokens and "combined_metrics" in analysis_results:
            print("\nIdentifying critical tokens...")
            critical_tokens = analyzer.identify_critical_tokens(
                combined_metrics=analysis_results["combined_metrics"],
                top_k=10,
                method="combined"
            )
            
            results["critical_tokens"] = critical_tokens
            
            # Print critical tokens
            print("\nTop Critical Tokens:")
            for i, token in enumerate(critical_tokens):
                print(f"  {i+1}. Token {token['token_index']} ('{token['token_text']}'): "
                      f"Importance={token['importance']:.3f}, "
                      f"Roles={token['global_roles']}")
                
                # Print layer predictions
                if "layer_predictions" in token and token["layer_predictions"]:
                    print("    Layer predictions:")
                    for layer, pred in token["layer_predictions"].items():
                        print(f"      {layer}: '{pred['text']}' (p={pred['probability']:.3f})")
                
                # Print layer roles from summary
                if ("summary" in analysis_results and 
                    "target_token_analysis" in analysis_results["summary"]):
                    for target in analysis_results["summary"]["target_token_analysis"]:
                        if target["token_index"] == token["token_index"]:
                            for layer, layer_info in target.get("layers", {}).items():
                                roles = layer_info.get("roles", [])
                                if roles:
                                    print(f"      {layer} roles: {roles}")
        
        print(f"\n--- Token Analysis Complete ---")
        return results

    except Exception as e:
        print(f"Error during token analysis: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
        
def run_enhanced_ablation_experiments(
    model: torch.nn.Module,
    processor: Any,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    critical_tokens: Optional[List[Dict[str, Any]]] = None,
    trace_data_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    output_dir: str = "enhanced_ablation_results",
    methods: List[str] = ["zero_out"],
    attention_strategies: List[str] = ["none", "block"],
    include_individual_tests: bool = True,
    run_layer_impact_analysis: bool = True,
    layer_tests: bool = False,
    num_layer_samples: int = 5,
    debug_mode: bool = False
) -> Dict[str, Any]:
    """
    Enhanced ablation experiments with multiple blocking strategies and visualizations.
    
    This function coordinates comprehensive token blocking experiments to measure 
    how different tokens and blocking strategies affect model predictions.
    
    Args:
        model: The loaded VLM model instance
        processor: The corresponding processor instance
        image_source: PIL image, URL string, or local file path
        prompt_text: The text prompt for the model
        critical_tokens: List of critical tokens to block (if None, they will be automatically identified)
        trace_data_path: Path to pre-existing trace CSV
        metadata_path: Path to trace metadata JSON
        output_dir: Directory path to save analysis outputs
        methods: List of blocking methods to test ("zero_out", "average", "noise", "interpolate", "reduce")
        attention_strategies: List of attention mask strategies ("none", "block", "reduce")
        include_individual_tests: Whether to test each token individually
        run_layer_impact_analysis: Whether to run layer-wise impact analysis
        layer_tests: Whether to run layer-specific ablation tests
        num_layer_samples: Number of layer samples to test if layer_tests is True
        debug_mode: Whether to print detailed debug information
        
    Returns:
        Dict containing ablation experiment results
    """
    import os
    import json
    import torch
    import traceback
    from typing import Dict, List, Any, Optional, Union
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n=== Starting Enhanced Token Ablation Experiments ===")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for visualizations
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    overall_results = {
        "experiment_config": {
            "methods": methods,
            "attention_strategies": attention_strategies,
            "include_individual_tests": include_individual_tests,
            "run_layer_impact_analysis": run_layer_impact_analysis,
            "layer_tests": layer_tests,
            "prompt_text": prompt_text
        },
        "token_analysis": {},
        "ablation_results": {},
        "layer_impact_analysis": {},
        "visualizations": {}
    }

    try:
        # If no critical tokens provided, we need to identify them first
        if critical_tokens is None:
            print("No critical tokens provided. Running token analysis first...")
            
            # Run token analysis to identify critical tokens
            from analyzer.semantic_tracing import run_token_analysis
            
            analysis_results = run_token_analysis(
                model=model,
                processor=processor,
                image_source=image_source,
                prompt_text=prompt_text,
                trace_data_path=trace_data_path,
                metadata_path=metadata_path,
                output_dir=output_dir,
                identify_critical_tokens=True,
                debug_mode=debug_mode
            )
            
            if "critical_tokens" not in analysis_results:
                print("Error: Failed to identify critical tokens")
                return {"error": "Failed to identify critical tokens"}
            
            critical_tokens = analysis_results["critical_tokens"]
            overall_results["token_analysis"] = analysis_results
        
        # Prepare input data
        from analyzer.semantic_tracing import EnhancedSemanticTracer
        
        tracer = EnhancedSemanticTracer(
            model=model,
            processor=processor,
            output_dir=output_dir,
            debug=debug_mode
        )
        
        input_data = tracer.prepare_inputs(image_source, prompt_text)
        
        # Initialize token analyzer for ablation
        from analyzer.token_analyzer import TokenAnalyzer
        
        analyzer = TokenAnalyzer(
            output_dir=output_dir,
            debug_mode=debug_mode
        )
        
        # Run reference inference to get baseline
        print("Running reference inference...")
        with torch.no_grad():
            reference_outputs = model(**{k: v for k, v in input_data["inputs"].items() if k != "token_type_ids"})
            
            # Get reference output token
            reference_logits = reference_outputs.logits
            
            # Handle different logits shapes
            if reference_logits.dim() == 2:  # [batch, vocab]
                pred_logits = reference_logits
            else:  # [batch, seq, vocab]
                pred_logits = reference_logits[:, -1, :]
                
            reference_ids = torch.argmax(pred_logits, dim=-1).unsqueeze(0)
            reference_text = processor.tokenizer.decode(reference_ids[0].tolist())
            
            # Get reference probabilities
            reference_probs = torch.softmax(pred_logits, dim=-1)
            
            reference_result = {
                "output_ids": reference_ids,
                "output_text": reference_text,
                "logits": pred_logits,
                "probabilities": reference_probs
            }
        
        overall_results["reference_output"] = {
            "text": reference_text,
            "token_id": reference_ids[0].item(),
            "probability": reference_probs[0, reference_ids[0]].item()
        }
        
        print(f"Reference output: '{reference_text}'")
        
        # Extract token indices
        token_indices = [token["token_index"] for token in critical_tokens]
        
        # Run ablation tests with all combinations of methods and attention strategies
        all_ablation_results = {}
        
        for method in methods:
            all_ablation_results[method] = {}
            
            for attention_strategy in attention_strategies:
                print(f"\nTesting method: {method}, attention strategy: {attention_strategy}")
                
                strategy_key = f"{method}_{attention_strategy}"
                all_ablation_results[method][attention_strategy] = {}
                
                # Individual token tests if requested
                individual_results = {}
                if include_individual_tests:
                    for token in critical_tokens:
                        token_idx = token["token_index"]
                        token_text = token["token_text"]
                        print(f"  Testing ablation of token {token_idx} ('{token_text}')...")
                        
                        sim_result = analyzer.simulate_token_blocking(
                            model=model,
                            processor=processor,
                            input_data=input_data,
                            tokens_to_block=[token_idx],
                            method=method,
                            reference_output=reference_result,
                            attention_mask_strategy=attention_strategy
                        )
                        
                        individual_results[f"token_{token_idx}"] = sim_result
                
                # All tokens test
                print(f"  Testing ablation of all {len(token_indices)} critical tokens together...")
                
                all_tokens_result = analyzer.simulate_token_blocking(
                    model=model,
                    processor=processor,
                    input_data=input_data,
                    tokens_to_block=token_indices,
                    method=method,
                    reference_output=reference_result,
                    attention_mask_strategy=attention_strategy
                )
                
                # Store results
                all_ablation_results[method][attention_strategy] = {
                    "individual_ablations": individual_results,
                    "all_critical_tokens_ablation": all_tokens_result
                }
        
        overall_results["ablation_results"] = all_ablation_results
        
        # Run layer impact analysis if requested
        if run_layer_impact_analysis:
            layer_impact_results = {}
            for method in methods:
                layer_impact_results[method] = {}
                
                for attention_strategy in attention_strategies:
                    print(f"\nRunning layer impact analysis for method: {method}, attention strategy: {attention_strategy}")
                    
                    layer_result = TokenAnalyzer.analyze_layer_impact(
                        model=model,
                        processor=processor,
                        input_data=input_data,
                        critical_tokens=critical_tokens,
                        output_dir=output_dir,
                        method=method,
                        attention_mask_strategy=attention_strategy,
                        debug_mode=debug_mode
                    )
                    
                    layer_impact_results[method][attention_strategy] = layer_result
            
            overall_results["layer_impact_analysis"] = layer_impact_results
        
        # Layer-specific tests if requested
        if layer_tests:
            layer_specific_results = {}
            
            # Only test one combination for layer tests to save time
            method = methods[0]
            attention_strategy = attention_strategies[0]
            
            print(f"\nRunning layer-specific tests with method: {method}, attention strategy: {attention_strategy}")
            
            ablation_results = analyzer.run_ablation_tests(
                model=model,
                processor=processor,
                input_data=input_data,
                critical_tokens=critical_tokens,
                method=method,
                include_individual_tests=False,  # Skip individual tests to save time
                layer_tests=True,
                num_layer_samples=num_layer_samples
            )
            
            layer_specific_results[f"{method}_{attention_strategy}"] = ablation_results
            overall_results["layer_specific_results"] = layer_specific_results
        
        # Create comparative visualizations of different strategies
        print("\nCreating comparative visualizations...")
        
        # 1. Compare impact scores across methods and attention strategies
        method_impacts = {}
        for method in methods:
            strategy_impacts = {}
            for attention_strategy in attention_strategies:
                result = all_ablation_results[method][attention_strategy]["all_critical_tokens_ablation"]
                impact = result.get("comparison", {}).get("impact_score", 0)
                strategy_impacts[attention_strategy] = impact
            method_impacts[method] = strategy_impacts
        
        # Plot comparison bar chart
        plt.figure(figsize=(12, 6))
        bar_width = 0.8 / len(attention_strategies)
        
        for i, method in enumerate(methods):
            x = np.arange(len(attention_strategies))
            impacts = [method_impacts[method][strategy] for strategy in attention_strategies]
            plt.bar(x + i*bar_width, impacts, width=bar_width, 
                    label=f"{method}", alpha=0.7)
        
        plt.xlabel('Attention Mask Strategy')
        plt.ylabel('Impact Score')
        plt.title('Comparison of Blocking Methods and Attention Strategies')
        plt.xticks(np.arange(len(attention_strategies)) + bar_width*(len(methods)-1)/2, attention_strategies)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        strategies_plot_path = os.path.join(viz_dir, 'strategy_comparison.png')
        plt.savefig(strategies_plot_path)
        
        overall_results["visualizations"]["strategy_comparison"] = strategies_plot_path
        
        # 2. If we have individual token results, compare their impacts
        if include_individual_tests and critical_tokens:
            # Use the first method and strategy
            method = methods[0]
            attention_strategy = attention_strategies[0]
            
            token_impacts = {}
            for token in critical_tokens:
                token_idx = token["token_index"]
                result_key = f"token_{token_idx}"
                
                if result_key in all_ablation_results[method][attention_strategy]["individual_ablations"]:
                    result = all_ablation_results[method][attention_strategy]["individual_ablations"][result_key]
                    impact = result.get("comparison", {}).get("impact_score", 0)
                    token_impacts[token_idx] = {
                        "impact": impact,
                        "text": token["token_text"]
                    }
            
            # Sort tokens by impact
            sorted_tokens = sorted(token_impacts.items(), key=lambda x: x[1]["impact"], reverse=True)
            
            # Plot token impacts
            plt.figure(figsize=(12, 6))
            token_indices = [t[0] for t in sorted_tokens]
            token_impact_values = [t[1]["impact"] for t in sorted_tokens]
            token_texts = [t[1]["text"] for t in sorted_tokens]
            
            plt.bar(range(len(token_indices)), token_impact_values, alpha=0.7)
            plt.xlabel('Token')
            plt.ylabel('Impact Score')
            plt.title(f'Impact of Individual Token Blocking\nMethod: {method}, Strategy: {attention_strategy}')
            plt.xticks(range(len(token_indices)), [f"{idx}\n'{text}'" for idx, text in zip(token_indices, token_texts)], 
                       rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            
            tokens_plot_path = os.path.join(viz_dir, 'token_impacts.png')
            plt.savefig(tokens_plot_path)
            
            overall_results["visualizations"]["token_impacts"] = tokens_plot_path
        
        # Save overall results
        results_path = os.path.join(output_dir, "enhanced_ablation_results.json")
        try:
            with open(results_path, 'w') as f:
                # Convert to serializable format
                serializable_results = analyzer._make_serializable(overall_results)
                json.dump(serializable_results, f, indent=2)
            print(f"\nSaved comprehensive results to: {results_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
        
        print(f"\n=== Enhanced Token Ablation Experiments Complete ===")
        print(f"Visualizations saved to: {viz_dir}")
        
        return overall_results

    except Exception as e:
        print(f"Error during enhanced ablation experiments: {e}")
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}
