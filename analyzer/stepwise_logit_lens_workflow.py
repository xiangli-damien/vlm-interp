# -*- coding: utf-8 -*-
"""
Implements a step-by-step Logit Lens analysis workflow for LLaVA-Next models.

This workflow generates tokens one by one and performs Logit Lens analysis
(hidden state extraction, projection, probability calculation, visualization)
at each step to observe the evolution of internal representations.
"""

import torch
import os
import gc
import time
import json # For potential future summary saving
import traceback # For printing detailed error stack traces
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple

# --- Imports from existing project components ---
# Analyzers
# Assuming analyzer is in the same directory level or accessible via python path
from analyzer.logit_lens import LLaVANextLogitLensAnalyzer
# Utils
# Assuming utils are in a 'utils' directory relative to this file's parent
# Adjust path if structure is different (e.g., from ..utils import ...)
try:
    from utils.viz_utils import visualize_token_probabilities
    from utils.data_utils import get_token_indices # If needed, though analyzer handles it
except ImportError:
    print("Warning: Could not import required components in stepwise_logit_lens_workflow.py")
    print("Ensure analyzer and utils directories are correctly structured and accessible.")
    # Define dummy functions/classes if needed for linting/type checking
    class LLaVANextLogitLensAnalyzer: pass
    def visualize_token_probabilities(*args, **kwargs): return []

# Ensure transformers classes are available if needed directly, though analyzer wraps most
# from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

def run_stepwise_logit_lens_workflow(
    model: torch.nn.Module, # Expects LlavaNextForConditionalGeneration
    processor: Any, # Expects LlavaNextProcessor
    image_source: Union[str, Image.Image],
    prompt_text: str,
    concepts_to_track: Optional[List[str]] = None,
    num_tokens: int = 10, # Number of generation steps to analyze
    selected_layers: Optional[List[int]] = None,
    output_dir: str = "stepwise_logit_lens_analysis",
    cpu_offload: bool = True,
    save_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Performs a step-by-step logit lens analysis during token generation.

    At each generation step, extracts hidden states for the current sequence,
    computes concept probabilities (logit lens), and optionally visualizes them.

    Args:
        model: The loaded LLaVA-Next model instance.
        processor: The corresponding processor instance.
        image_source (Union[str, Image.Image]): PIL image, URL string, or local file path.
        prompt_text (str): The text prompt for the model.
        concepts_to_track (Optional[List[str]]): List of concept strings (e.g., "cat", "sign").
        num_tokens (int): The number of tokens to generate and analyze step-by-step.
        selected_layers (Optional[List[int]]): Specific layer indices to analyze/visualize. None for all.
        output_dir (str): Base directory path to save analysis outputs. Step-specific subdirs created inside.
        cpu_offload (bool): Whether to move intermediate tensors to CPU during probability extraction.
        save_visualizations (bool): If True, generate and save visualizations at each step.

    Returns:
        Dict[str, Any]: Dictionary containing analysis results per step ('step_results'),
                        the full generated sequence ('full_generated_text'), timing ('total_time'),
                        model name, and config details. Includes an 'error' key on failure.
    """
    print("\n--- Starting Step-by-Step LLaVA-Next Logit Lens Workflow ---")
    print(f"  Analyzing {num_tokens} generation steps.")
    print(f"  Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Any] = {"step_results": {}} # Store results per step
    start_time_workflow = time.time()

    try:
        # 1. Instantiate the analyzer core (reused for its methods)
        analyzer = LLaVANextLogitLensAnalyzer(model, processor)

        # 2. Prepare Initial Inputs and Static Data
        print("Preparing initial inputs and feature mapping...")
        initial_input_data = analyzer.prepare_inputs(image_source, prompt_text)
        if not initial_input_data:
            return {"error": "Initial input preparation failed."}

        # Store data needed repeatedly in the loop
        feature_mapping = initial_input_data["feature_mapping"]
        original_image = initial_input_data["original_image"]
        spatial_preview_image = initial_input_data["spatial_preview_image"]
        # Keep a copy of the initial inputs for the model pass each time if needed
        # Or just manage the input_ids dynamically
        current_input_ids = initial_input_data["inputs"]["input_ids"].clone()
        pixel_values = initial_input_data["inputs"]["pixel_values"] # Static for all steps
        image_sizes = initial_input_data["inputs"].get("image_sizes") # Static for all steps

        # Prepare concepts dictionary
        concept_token_ids: Dict[str, List[int]] = {}
        concepts_actually_tracked: List[str] = []
        if concepts_to_track:
            for concept in concepts_to_track:
                try:
                    # Use processor tokenizer directly
                    token_ids = processor.tokenizer.encode(concept, add_special_tokens=False)
                    if token_ids:
                        concept_token_ids[concept] = token_ids
                        concepts_actually_tracked.append(concept)
                except Exception as e: print(f"  Warning: Error encoding concept '{concept}': {e}")
        print(f"Tracking concepts: {concepts_actually_tracked}")
        if not concept_token_ids and save_visualizations:
            print("Warning: No valid concepts to track, visualizations will be skipped.")
            save_visualizations = False # Disable viz if no concepts

        # Store some config for the final results
        results["model_name"] = getattr(model.config, '_name_or_path', 'N/A')
        results["config"] = {
             "num_tokens": num_tokens,
             "concepts_tracked": concepts_actually_tracked,
             "concept_token_ids": concept_token_ids,
             "selected_layers": selected_layers if selected_layers is not None else "All",
             "cpu_offload": cpu_offload,
             "image_source": image_source if isinstance(image_source, str) else "PIL Input",
             "prompt": prompt_text
        }

        # 3. Step-by-Step Generation and Analysis Loop
        model.eval() # Ensure model is in evaluation mode
        full_generated_sequence = "" # Accumulate generated text

        for step in range(num_tokens):
            step_start_time = time.time()
            print(f"\n--- Analyzing Step {step + 1}/{num_tokens} ---")
            print(f"  Current sequence length: {current_input_ids.shape[1]}")
            step_results_this_token: Dict[str, Any] = {}

            # --- a) Extract Hidden States for Current Sequence ---
            hidden_states_this_step: Optional[Tuple[torch.Tensor, ...]] = None
            logits_this_step: Optional[torch.Tensor] = None
            try:
                print(f"  Running forward pass for hidden states...")
                with torch.no_grad():
                    # Run model forward pass to get hidden states for *all* layers
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=torch.ones_like(current_input_ids), # Assume full attention mask
                        pixel_values=pixel_values,
                        image_sizes=image_sizes, # Pass image sizes if available/needed by model
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False # Ensure no caching interferes with hidden state shapes
                    )
                    hidden_states_this_step = outputs.get("hidden_states")
                    logits_this_step = outputs.get("logits") # Get logits too for prediction

                if hidden_states_this_step is None or logits_this_step is None:
                    raise ValueError("Model output missing 'hidden_states' or 'logits' in step {step+1}.")

            except Exception as e:
                print(f"  Error during hidden state extraction in step {step + 1}: {e}")
                traceback.print_exc()
                results["error"] = f"Error in step {step+1} during hidden state extraction: {e}"
                break # Stop the workflow

            # --- b) Extract Token Probabilities (Logit Lens) ---
            token_probs: Dict[int, Dict[str, Any]] = {}
            if concept_token_ids: # Only run if there are concepts to track
                print(f"  Extracting token probabilities (Logit Lens)...")
                # Prepare data for the analyzer method
                # Note: input_data needs feature_mapping, outputs needs hidden_states
                mock_input_data_for_probs = {"feature_mapping": feature_mapping}
                mock_outputs_for_probs = {"hidden_states": hidden_states_this_step}
                try:
                    token_probs = analyzer.extract_token_probabilities(
                        input_data=mock_input_data_for_probs,
                        outputs=mock_outputs_for_probs,
                        concepts_to_track=concept_token_ids,
                        cpu_offload=cpu_offload
                    )
                    step_results_this_token["token_probabilities_summary"] = {
                        k: f"Data shape: {type(v)}" for k, v in token_probs.items() # Basic summary
                    } if token_probs else "None"
                    
                    # Add debugging information about token probabilities and positions
                    if token_probs:
                        layer_indices = sorted(token_probs.keys())
                        print(f"  Token probabilities extracted for layers: {layer_indices}")
                        
                        # Check the last layer as representative
                        if layer_indices:
                            last_layer = max(layer_indices)
                            print(f"  Checking token coverage in last layer ({last_layer}):")
                            
                            # Check base feature positions
                            base_feature = token_probs[last_layer].get("base_feature", {})
                            if base_feature:
                                positions = base_feature.get("positions", {})
                                pos_min = min(positions.keys()) if positions else "N/A"
                                pos_max = max(positions.keys()) if positions else "N/A"
                                print(f"    Base feature position range: {pos_min} to {pos_max}")
                                
                                # Check if previously generated token is included
                                if step > 0:  # Not the first step
                                    prev_token_pos = new_token_position - 1
                                    included = prev_token_pos in positions
                                    print(f"    Previously generated token at position {prev_token_pos} included: {included}")
                                    step_results_this_token["prev_token_included"] = included
                            
                            # Check patch feature positions
                            patch_feature = token_probs[last_layer].get("patch_feature", {})
                            if patch_feature:
                                positions = patch_feature.get("positions", {})
                                pos_min = min(positions.keys()) if positions else "N/A"
                                pos_max = max(positions.keys()) if positions else "N/A"
                                print(f"    Patch feature position range: {pos_min} to {pos_max}")
                                
                except Exception as e:
                    print(f"  Error during probability extraction in step {step+1}: {e}")
                    traceback.print_exc()
                    # Continue to next step if possible, but log error
                    step_results_this_token["probability_extraction_error"] = str(e)


            # --- c) Visualize Probabilities ---
            step_viz_paths = []
            if save_visualizations and token_probs:
                print(f"  Generating visualizations...")
                # Create a subdirectory for this specific step's visualizations
                # Use token text if available later, otherwise just step number
                step_output_dir = os.path.join(output_dir, f"step_{step + 1:02d}")
                os.makedirs(step_output_dir, exist_ok=True)

                # Prepare input data for visualization function
                viz_input_data = {
                    "feature_mapping": feature_mapping,
                    "original_image": original_image,
                    "spatial_preview_image": spatial_preview_image
                    # Add other required keys if visualize_token_probabilities needs them
                }
                try:
                    # Call the visualization utility function
                    step_viz_paths = visualize_token_probabilities(
                        token_probs=token_probs,
                        input_data=viz_input_data,
                        selected_layers=selected_layers,
                        output_dir=step_output_dir, # Save to step-specific directory
                        generate_composite=True, # Explicitly request composite images
                        only_composite=True # Only keep composite images
                    )
                    print(f"  Saved {len(step_viz_paths)} visualization files to {step_output_dir}")
                except Exception as e:
                    print(f"  Error during visualization in step {step+1}: {e}")
                    traceback.print_exc()
                    step_results_this_token["visualization_error"] = str(e)

            step_results_this_token["visualization_paths"] = step_viz_paths

            # --- d) Predict Next Token ---
            # Use the logits obtained during the hidden state extraction pass
            next_token_logits = logits_this_step[:, -1, :] # Logits for the last token position
            next_token_id = torch.argmax(next_token_logits, dim=-1) # Greedy decoding
            # Calculate loss/perplexity for the predicted token (optional)
            try:
                 log_probs = torch.log_softmax(next_token_logits.float(), dim=-1)
                 loss_val = -log_probs[0, next_token_id].item()
                 step_results_this_token["predicted_token_loss"] = loss_val
            except Exception:
                 step_results_this_token["predicted_token_loss"] = None

            # --- e) Decode and Update ---
            try:
                # Decode the single token ID
                token_text = processor.tokenizer.decode([next_token_id.item()])
                step_results_this_token["generated_token"] = token_text
                step_results_this_token["generated_token_id"] = next_token_id.item()
                print(f"  Predicted next token: '{token_text}' (ID: {next_token_id.item()})")
                print(f"  Current sequence length: {current_input_ids.shape[1]}")
                new_token_position = current_input_ids.shape[1]  # Position of token when added to sequence
                print(f"  New token will be at position {new_token_position} after appending")
            except Exception as e:
                print(f"  Error decoding token ID {next_token_id.item()}: {e}")
                step_results_this_token["generated_token"] = "[Decoding Error]"
                step_results_this_token["generated_token_id"] = next_token_id.item()
                results["error"] = f"Error decoding token in step {step+1}"
                break # Stop if decoding fails

            # Append to the overall sequence
            full_generated_sequence += token_text

            # Update input_ids for the next iteration
            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)

            # Store results for this step
            step_results_this_token["step_time"] = time.time() - step_start_time
            results["step_results"][f"step_{step + 1}"] = step_results_this_token

            # --- f) Memory Cleanup ---
            del hidden_states_this_step
            del logits_this_step
            del token_probs # Delete the potentially large dict
            del outputs # Delete the model output object
            if 'next_token_logits' in locals(): del next_token_logits
            if 'log_probs' in locals(): del log_probs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  Step {step + 1} completed in {step_results_this_token['step_time']:.2f} seconds.")

        # --- 4. Finalize Workflow ---
        results["full_generated_text"] = full_generated_sequence
        workflow_duration = time.time() - start_time_workflow
        results["total_time"] = workflow_duration
        # Remove error key if workflow completed without breaking
        if "error" not in results:
             results.pop("error", None)

        print(f"\n--- Step-by-Step Logit Lens Workflow Finished ({workflow_duration:.2f} seconds) ---")
        print(f"Full generated text: {full_generated_sequence}")

    except Exception as e:
        print(f"\n--- Error during Step-by-Step Logit Lens workflow ---")
        traceback.print_exc()
        results["error"] = f"Workflow failed with error: {e}"

    # Return the results dictionary
    return results