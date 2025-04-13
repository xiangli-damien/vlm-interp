# -*- coding: utf-8 -*-
"""
End-to-end analysis workflow functions.

"""

import torch
import os
import gc
import time # For saliency workflow timing
import json # For potential future summary saving
import pickle # For potential future results saving
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple

# --- Imports from this project ---
# Analyzers
from analyzer.logit_lens_analyzer import LLaVANextLogitLensAnalyzer
from analyzer.saliency_analyzer import (
    calculate_saliency_scores,
    analyze_layerwise_saliency_flow
)
# Utils
try:
    from utils.data_utils import load_image, get_token_indices, build_conversation, find_token_indices, find_image_token_spans, get_image_token_spans
    from utils.model_utils import get_llm_attention_layer_names
    from utils.hook_utils import GradientAttentionCapture
    from utils.visual_utils import visualize_token_probabilities, visualize_information_flow
except ImportError as e:
     print(f"Warning: Could not import required components in workflows.py: {e}")
     # Define dummy functions/classes if needed for linting/type checking
     class LLaVANextLogitLensAnalyzer: pass
     class GradientAttentionCapture: pass
     def calculate_saliency_scores(*args, **kwargs): return {}
     def analyze_layerwise_saliency_flow(*args, **kwargs): return {}
     def load_image(*args, **kwargs): raise NotImplementedError()
     def build_conversation(*args, **kwargs): raise NotImplementedError()
     def find_token_indices(*args, **kwargs): raise NotImplementedError()
     def get_llm_attention_layer_names(*args, **kwargs): return []
     def visualize_token_probabilities(*args, **kwargs): return []
     def visualize_information_flow(*args, **kwargs): pass


# --- Workflow 1: Logit Lens ---

def run_logit_lens_workflow(
    model: torch.nn.Module, # Expects LlavaNextForConditionalGeneration
    processor: Any, # Expects LlavaNextProcessor
    image_source: Union[str, Image.Image],
    prompt_text: str,
    concepts_to_track: Optional[List[str]] = None,
    selected_layers: Optional[List[int]] = None,
    output_dir: str = "logit_lens_analysis",
    cpu_offload: bool = True
) -> Dict[str, Any]:
    """
    Performs the complete logit lens analysis pipeline using LLaVANextLogitLensAnalyzer.

    Orchestrates input preparation, hidden state extraction, probability calculation,
    visualization, and saving a summary file.

    Args:
        model: The loaded LLaVA-Next model instance.
        processor: The corresponding processor instance.
        image_source (Union[str, Image.Image]): PIL image, URL string, or local file path.
        prompt_text (str): The text prompt for the model.
        concepts_to_track (Optional[List[str]]): List of concept strings (e.g., "cat", "sign").
        selected_layers (Optional[List[int]]): Specific layer indices to analyze/visualize. None for all.
        output_dir (str): Directory path to save analysis outputs (visualizations, summary).
        cpu_offload (bool): Whether to move intermediate tensors to CPU during probability extraction.

    Returns:
        Dict[str, Any]: Dictionary containing analysis results ('token_probabilities', 'generated_text',
                        'feature_mapping', 'concepts_tracked', 'visualization_paths', 'summary_path').
                        Includes an 'error' key on failure.
    """
    print("\n--- Starting LLaVA-Next Logit Lens Workflow ---")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    try:
        # 1. Instantiate the analyzer core
        analyzer = LLaVANextLogitLensAnalyzer(model, processor)

        # 2. Prepare Inputs using the analyzer's method
        input_data = analyzer.prepare_inputs(image_source, prompt_text)
        if not input_data: return {"error": "Input preparation failed."}
        results["feature_mapping"] = input_data["feature_mapping"]

        # 3. Extract Hidden States & Sample Text using the analyzer's method
        outputs = analyzer.extract_hidden_states(input_data["inputs"])
        if not outputs: return {"error": "Hidden state extraction failed."}
        results["generated_text"] = outputs["generated_text"]

        # 4. Prepare Concepts Dictionary
        concept_token_ids = {}
        concepts_actually_tracked = []
        if concepts_to_track:
            # print(f"  Preparing token IDs for concepts: {concepts_to_track}") # Less verbose
            for concept in concepts_to_track:
                try:
                    token_ids = processor.tokenizer.encode(concept, add_special_tokens=False)
                    if token_ids:
                        concept_token_ids[concept] = token_ids
                        concepts_actually_tracked.append(concept)
                        # print(f"    '{concept}' -> IDs: {token_ids}") # Less verbose
                    # else: print(f"    Warning: Concept '{concept}' produced no token IDs.") # Less verbose
                except Exception as e: print(f"  Warning: Error encoding concept '{concept}': {e}")
        # else: print("  No specific concepts provided to track.") # Less verbose
        results["concepts_tracked"] = concept_token_ids

        # 5. Extract Token Probabilities using the analyzer's method
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

            # 6. Visualize Probabilities (using the utility function)
            print("  Calling visualization function...")
            viz_paths = visualize_token_probabilities(
                token_probs=token_probs,
                input_data=input_data,
                selected_layers=selected_layers,
                output_dir=output_dir
            )
            results["visualization_paths"] = viz_paths

        # 7. Save Summary File (logic moved from the method)
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

    # Clean up large intermediate data explicitly? Optional.
    if 'outputs' in locals(): del outputs
    if 'input_data' in locals(): del input_data
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return results


# --- Workflow 2: Saliency Analysis ---

def run_saliency_workflow(
    model: torch.nn.Module,
    processor: Any,
    image_source: Union[str, Image.Image],
    prompt_text: str,
    num_tokens: int = 5,
    output_dir: str = "saliency_analysis",
    image_size: Tuple[int, int] = (336, 336),
    layer_batch_size: int = 2,
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Performs token-by-token generation with gradient-based saliency analysis workflow.

    Uses batched layer processing for memory efficiency during gradient computation.
    Calls visualization utilities internally if `save_plots` is True.
    Returns a dictionary of results; saving the dictionary should be handled by the caller.

    Args:
        model: The VLM model instance (must have gradients enabled).
        processor: The corresponding processor instance.
        image_source: URL, path, or PIL Image for analysis.
        prompt_text: The text prompt.
        num_tokens: Number of tokens to generate and analyze.
        output_dir: Directory to save plots.
        image_size: Target size for image loading.
        layer_batch_size: Number of attention layers to compute gradients for in each backward pass.
        save_plots: Whether to generate and save information flow plots.

    Returns:
        Dict[str, Any]: Dictionary containing analysis results per token ('token_results'),
                        the full generated sequence ('sequence_text'), timing ('total_time'),
                        model name, and config details. Includes an 'error' key on failure.
    """
    # (Function definition remains the same as provided previously for the notebook)
    # Including setup, helper function, loop, and returning the results dict.
    # **Crucially, removed the pickle.dump / json.dump logic from here.**
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
        initial_input_ids = inputs["input_ids"]; pixel_values = inputs["pixel_values"]; image_sizes = inputs.get("image_sizes")
        image_token_id = getattr(model.config, "image_token_index", 32000)
        text_indices, image_indices = find_token_indices(initial_input_ids, image_token_id)
        all_attn_layer_names = get_llm_attention_layer_names(model)
        if not all_attn_layer_names: raise ValueError("Could not find attention layers.")
        print(f" Found {len(all_attn_layer_names)} attention layers.")

        # --- 2. Helper for Batched Gradient Computation ---
        # (Helper function definition remains the same)
        def generate_next_token_and_compute_saliency(current_input_ids):
            all_saliency_scores = {}; next_token_id = None; loss_val = None
            model.eval()
            with torch.no_grad():
                outputs_pred = model(input_ids=current_input_ids,attention_mask=torch.ones_like(current_input_ids),pixel_values=pixel_values,image_sizes=image_sizes,use_cache=True)
                logits = outputs_pred.logits[:, -1, :]; next_token_id = torch.argmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits.float(), dim=-1); loss_val = -log_probs[0, next_token_id].item()
                del outputs_pred, logits, log_probs; gc.collect(); torch.cuda.empty_cache()
            # model.train(); 
            grad_capture = GradientAttentionCapture(cpu_offload=True)
            num_layers = len(all_attn_layer_names)
            for batch_start in range(0, num_layers, layer_batch_size):
                batch_end = min(batch_start + layer_batch_size, num_layers); current_layer_batch_names = all_attn_layer_names[batch_start:batch_end]
                # print(f"  Processing saliency for layers {batch_start+1}-{batch_end}...") # Less verbose
                model.zero_grad(set_to_none=True); grad_capture.register_hooks(model, current_layer_batch_names)
                with torch.enable_grad():
                    outputs_grad = model(input_ids=current_input_ids,attention_mask=torch.ones_like(current_input_ids),pixel_values=pixel_values,image_sizes=image_sizes,use_cache=False,output_attentions=True)
                    logits_grad = outputs_grad.logits[:, -1, :]; log_probs_grad = torch.log_softmax(logits_grad.float(), dim=-1)
                    loss = -log_probs_grad[0, next_token_id]
                try: loss.backward()
                except Exception as bk_err: print(f" Backward pass error: {bk_err}"); grad_capture.clear_hooks(); raise
                batch_saliency = grad_capture.compute_saliency(); all_saliency_scores.update(batch_saliency)
                grad_capture.clear_hooks() # Clear hooks for the current batch *after* compute_saliency
                del outputs_grad, logits_grad, log_probs_grad, loss; gc.collect(); torch.cuda.empty_cache()
            model.eval()
            return next_token_id, all_saliency_scores, loss_val

        # --- 3. Token-by-Token Generation and Analysis Loop ---
        token_results = {}; generated_sequence = ""; current_input_ids = initial_input_ids.clone()
        loop_start_time = time.time()
        for step_idx in range(num_tokens):
            print(f"--- Analyzing Saliency Token {step_idx+1}/{num_tokens} ---")
            try:
                next_token_id, current_saliency_scores, loss_val = generate_next_token_and_compute_saliency(current_input_ids)
                if next_token_id is None: print("  Failed to generate next token. Stopping."); break
                new_token_text = processor.tokenizer.decode([next_token_id.item()])
                print(f"  Generated token: '{new_token_text}' (ID: {next_token_id.item()}, Loss: {loss_val:.4f})")
                target_idx = current_input_ids.shape[1] - 1 # Target is last token of *input* to prediction step
                flow_metrics = analyze_layerwise_saliency_flow(current_saliency_scores, text_indices, image_indices, target_idx, cpu_offload=True)
                if save_plots and flow_metrics:
                    model_name_short = model.config._name_or_path.split('/')[-1]
                    plot_filename = f"token_{(step_idx+1):02d}_{new_token_text.strip().replace(' ', '_')}_saliency_flow.png"
                    plot_path = os.path.join(output_dir, plot_filename)
                    visualize_information_flow(flow_metrics, f"{model_name_short} - Saliency Flow for Token {step_idx+1} ('{new_token_text}')", plot_path)
                token_results[f"token_{step_idx+1}"] = {"token_text": new_token_text,"token_id": next_token_id.item(),"loss": loss_val,"metrics": flow_metrics}
                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)
                generated_sequence += new_token_text
                del current_saliency_scores, flow_metrics; gc.collect(); torch.cuda.empty_cache()
            except Exception as e: print(f"Error analyzing token {step_idx+1}: {e}"); import traceback; traceback.print_exc(); break
        loop_end_time = time.time()

        final_results = {
            "token_results": token_results,
            "sequence_text": generated_sequence,
            "total_time": loop_end_time - loop_start_time,
            "model_name": model.config._name_or_path,
            "config": {"num_tokens": num_tokens,"image_size": image_size,"layer_batch_size": layer_batch_size,"prompt": prompt_text,"image_source": image_source if isinstance(image_source, str) else "PIL Input"}
        }
        print(f"\n--- Saliency Workflow Finished ({final_results['total_time']:.2f} seconds) ---")

    except Exception as e:
        print(f"Error during Saliency workflow: {e}")
        import traceback; traceback.print_exc()
        final_results["error"] = str(e)

    # No saving logic here, return the dict for the caller to handle
    return final_results