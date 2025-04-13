"""
Logit Lens analysis implementation for LLaVA-Next models.

The LogitLens technique analyzes hidden states at different layers of the model
by projecting them through the language model head to see what tokens are predicted.
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple, Callable

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers.image_processing_utils import select_best_resolution
from transformers.image_transforms import resize as hf_resize, pad as hf_pad
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    to_numpy_array
)
import gc
from utils.data_utils import load_image, get_image_token_spans, get_token_masks

class LLaVANextLogitLensAnalyzer:
    """
    Analyzer for applying the logit lens technique to LLaVA-Next models.
    
    Allows extracting and analyzing hidden states at different layers of the
    transformer by projecting them through the language model head to examine
    which tokens would be predicted at each layer.
    """

    def __init__(self, model: LlavaNextForConditionalGeneration, processor: LlavaNextProcessor):
        """
        Initialize the analyzer with a model and processor.
        
        Args:
            model: A loaded LLaVA-Next model
            processor: The corresponding processor
        """
        if not isinstance(model, LlavaNextForConditionalGeneration):
             print(f"Warning: Model provided is type {type(model)}, expected LlavaNextForConditionalGeneration.")
        if not isinstance(processor, LlavaNextProcessor):
             print(f"Warning: Processor provided is type {type(processor)}, expected LlavaNextProcessor.")

        self.model = model
        self.processor = processor
        self.device = model.device

        # Get image token ID from config or processor
        self.image_token_id = getattr(model.config, "image_token_index", None)
        if self.image_token_id is None:
             try:
                  self.image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
             except Exception:
                  print("Warning: Could not determine image_token_id from config or processor. Using default 32000.")
                  self.image_token_id = 32000

        # Get number of layers from config
        self.num_layers = getattr(model.language_model.config, "num_hidden_layers", 32)

        # Vision tower patch calculation
        vision_config = getattr(model.config, "vision_config", None)
        base_image_size = getattr(vision_config, "image_size", 336) if vision_config else 336
        self.vision_raw_patch_size = getattr(vision_config, "patch_size", 14) if vision_config else 14
        if self.vision_raw_patch_size > 0:
             self.vision_patch_grid_size = base_image_size // self.vision_raw_patch_size
        else:
             print("Warning: Vision config patch_size is invalid. Defaulting grid size to 24.")
             self.vision_patch_grid_size = 24

        # Get vision feature selection strategy
        self.vision_feature_select = getattr(model.config, "vision_feature_select_strategy", "default")
        
        # Get image grid pinpoints for high-res image support
        self.image_grid_pinpoints = getattr(model.config, "image_grid_pinpoints",
                                            [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]])

        # Language model head for projecting hidden states to vocabulary tokens
        self.lm_head = model.language_model.lm_head

        print(f"Initialized LLaVANextLogitLensAnalyzer")
        print(f"  Model: {getattr(model.config, '_name_or_path', 'N/A')}")
        print(f"  Device: {self.device}")
        print(f"  LLM Layers: {self.num_layers}")
        print(f"  Image Token ID: {self.image_token_id}")
        print(f"  Vision Base Grid Size: {self.vision_patch_grid_size}x{self.vision_patch_grid_size}")
        print(f"  Vision Raw Patch Size: {self.vision_raw_patch_size}x{self.vision_raw_patch_size}")
        print(f"  Vision Feature Selection: {self.vision_feature_select}")
        print(f"  Image Grid Pinpoints: {self.image_grid_pinpoints}")

    def create_feature_mapping(self, image_spans: List[Tuple[int, int]], image_size_orig_hw: Tuple[int, int]) -> Dict[str, Any]:
        """
        Creates a mapping between token indices and feature positions for both base and patch features.
        
        Args:
            image_spans: List of (start_idx, end_idx) tuples identifying image token spans
            image_size_orig_hw: Original image size as (height, width)
            
        Returns:
            Dictionary containing mappings for base and patch features
        """
        if not image_spans:
            print("Error: No image spans found, cannot create feature mapping.")
            return {}

        # Handle potentially multiple spans (though usually one for LLaVA-Next)
        span_start = image_spans[0][0]
        span_end = image_spans[-1][1]
        if len(image_spans) > 1:
            print(f"Warning: Multiple image spans detected ({len(image_spans)}). Processing combined range: {span_start}-{span_end}")

        total_image_tokens = span_end - span_start + 1
        if total_image_tokens <= 0:
             print(f"Error: Invalid token span range resulted in non-positive token count ({total_image_tokens}).")
             return {}

        # --- Base Feature Mapping (Fixed Grid) ---
        base_grid_size = self.vision_patch_grid_size
        expected_base_tokens = base_grid_size * base_grid_size
        actual_base_token_count = min(expected_base_tokens, total_image_tokens)
        base_start_idx = span_start
        base_end_idx = span_start + actual_base_token_count - 1

        print(f"Mapping Base Features: Tokens {base_start_idx}-{base_end_idx} (Count: {actual_base_token_count}) -> Grid: {base_grid_size}x{base_grid_size}")
        mapping_base = {}
        for i in range(actual_base_token_count):
            token_idx = base_start_idx + i
            r = i // base_grid_size
            c = i % base_grid_size
            mapping_base[token_idx] = (r, c)

        # --- Spatial (Patch) Feature Mapping ---
        spatial_start_idx_potential = base_end_idx + 1
        num_spatial_tokens_available = max(0, span_end - spatial_start_idx_potential + 1)

        orig_height, orig_width = image_size_orig_hw

        mapping_spatial = {}
        mapping_newline = {}
        unpadded_grid_rows, unpadded_grid_cols = 0, 0
        grid_rows_padded, grid_cols_padded = 0, 0
        target_resolution_wh = (0, 0)
        resized_dimensions_wh = (0, 0)
        padded_dimensions_wh = (0, 0)
        actual_spatial_start_idx = -1
        actual_spatial_end_idx = -1

        if num_spatial_tokens_available > 0:
            print(f"Mapping Spatial Features: Available tokens {spatial_start_idx_potential}-{span_end} (Count: {num_spatial_tokens_available})")

            # 1. Determine target resolution
            target_resolution_hw = select_best_resolution(image_size_orig_hw, self.image_grid_pinpoints)
            target_height, target_width = target_resolution_hw
            target_resolution_wh = (target_width, target_height)

            # 2. Calculate resized dimensions (aspect ratio preserved)
            resized_width, resized_height = self._calculate_resized_dimensions(image_size_orig_hw, target_resolution_hw)
            resized_dimensions_wh = (resized_width, resized_height)

            # 3. Calculate padded dimensions (multiple of raw patch size)
            padded_height = math.ceil(resized_height / self.vision_raw_patch_size) * self.vision_raw_patch_size
            padded_width = math.ceil(resized_width / self.vision_raw_patch_size) * self.vision_raw_patch_size
            padded_dimensions_wh = (padded_width, padded_height)

            # Grid sizes based on padded dimensions (for visualization background)
            grid_rows_padded = padded_height // self.vision_raw_patch_size
            grid_cols_padded = padded_width // self.vision_raw_patch_size

            # Grid sizes based on unpadded dimensions (for mapping features)
            unpadded_grid_rows = math.ceil(resized_height / self.vision_raw_patch_size)
            unpadded_grid_cols = math.ceil(resized_width / self.vision_raw_patch_size)
            num_unpadded_patches = unpadded_grid_rows * unpadded_grid_cols

            # 4. Determine expected token structure (with/without newlines)
            has_newline = unpadded_grid_rows > 1
            expected_spatial_tokens_structure = num_unpadded_patches
            if has_newline:
                expected_spatial_tokens_structure += (unpadded_grid_rows - 1)

            print(f"  Original HxW: {image_size_orig_hw}")
            print(f"  Target Res HxW: {target_resolution_hw}")
            print(f"  Resized HxW: ({resized_height}, {resized_width})")
            print(f"  Padded HxW: ({padded_height}, {padded_width})")
            print(f"  Unpadded Grid HxW: ({unpadded_grid_rows}, {unpadded_grid_cols}) -> {num_unpadded_patches} patches")
            print(f"  Padded Vis Grid HxW: ({grid_rows_padded}, {grid_cols_padded})")
            print(f"  Expect Newlines: {has_newline} (Expected Tokens: {expected_spatial_tokens_structure})")

            if num_spatial_tokens_available != expected_spatial_tokens_structure:
                 print(f"  Warning: Actual available spatial tokens ({num_spatial_tokens_available}) differs from expected structural count ({expected_spatial_tokens_structure}). Mapping will proceed based on available tokens.")

            # 5. Perform mapping based on expected structure
            current_token_idx = spatial_start_idx_potential
            processed_spatial_count = 0
            for r in range(unpadded_grid_rows):
                # Map patch tokens for the current row
                for c in range(unpadded_grid_cols):
                    if current_token_idx <= span_end:
                        if processed_spatial_count == 0: actual_spatial_start_idx = current_token_idx
                        mapping_spatial[current_token_idx] = (r, c)
                        actual_spatial_end_idx = current_token_idx
                        current_token_idx += 1
                        processed_spatial_count += 1
                    else: break

                if current_token_idx > span_end: break

                # Map newline token if expected and available
                if has_newline and r < (unpadded_grid_rows - 1):
                    if current_token_idx <= span_end:
                        if processed_spatial_count == 0: actual_spatial_start_idx = current_token_idx
                        mapping_newline[current_token_idx] = r
                        actual_spatial_end_idx = current_token_idx
                        current_token_idx += 1
                        processed_spatial_count += 1
                    else: break

            if current_token_idx <= span_end:
                 remaining = span_end - current_token_idx + 1
                 print(f"  Warning: After mapping structure, {remaining} tokens remain in span (Indices {current_token_idx}-{span_end}). These are unmapped.")

            print(f"  Mapped {processed_spatial_count} spatial/newline tokens. Actual index range mapped: {actual_spatial_start_idx}-{actual_spatial_end_idx}")

        else:
            print("No spatial tokens available to map.")
            actual_spatial_start_idx = -1
            actual_spatial_end_idx = -1

        # Final check and assembly of results
        total_mapped = actual_base_token_count + len(mapping_spatial) + len(mapping_newline)
        print(f"--- Mapping Summary ---")
        print(f"  Base Mapped: {actual_base_token_count} tokens (Indices {base_start_idx}-{base_end_idx})")
        print(f"  Spatial Patches Mapped: {len(mapping_spatial)} tokens")
        print(f"  Newline Tokens Mapped: {len(mapping_newline)} tokens")
        print(f"  Total Mapped: {total_mapped} tokens")
        print(f"  Total Image Tokens in Span: {total_image_tokens} tokens (Indices {span_start}-{span_end})")
        if total_mapped < total_image_tokens:
            print(f"  Note: {total_image_tokens - total_mapped} tokens within the span were not mapped.")
        print(f"-----------------------")

        return {
            "base_feature": {
                "start_idx": base_start_idx,
                "end_idx": base_end_idx,
                "grid": (base_grid_size, base_grid_size), # Rows (H), Cols (W)
                "positions": mapping_base, # {token_idx: (row, col)}
            },
            "patch_feature": {
                "start_idx": actual_spatial_start_idx if mapping_spatial else -1,
                "end_idx": max(mapping_spatial.keys()) if mapping_spatial else -1,
                "grid_for_visualization": (grid_rows_padded, grid_cols_padded), # Rows (H), Cols (W)
                "grid_unpadded": (unpadded_grid_rows, unpadded_grid_cols), # Rows (H), Cols (W)
                "positions": mapping_spatial, # {token_idx: (row, col)} within unpadded grid
            },
            "newline_feature": {
                "start_idx": min(mapping_newline.keys()) if mapping_newline else -1,
                "end_idx": max(mapping_newline.keys()) if mapping_newline else -1,
                "positions": mapping_newline, # {token_idx: row_idx}
            },
            "combined_spatial_end_idx": actual_spatial_end_idx,
            "patch_size": self.vision_raw_patch_size,
            "original_size": (orig_width, orig_height), # W, H
            "best_resolution": target_resolution_wh, # W, H
            "padded_dimensions": padded_dimensions_wh, # W, H
            "resized_dimensions": resized_dimensions_wh, # W, H
        }

    def _calculate_resized_dimensions(self, orig_size_hw: Tuple[int, int], target_res_hw: Tuple[int, int]) -> Tuple[int, int]:
        """
        Calculates aspect-ratio preserved dimensions within target resolution.
        
        Args:
            orig_size_hw: Original image size as (height, width)
            target_res_hw: Target resolution as (height, width)
            
        Returns:
            (width, height) of the resized image
        """
        orig_h, orig_w = orig_size_hw
        target_h, target_w = target_res_hw
        scale_w = target_w / orig_w
        scale_h = target_h / orig_h
        if scale_w < scale_h:
            new_w = target_w
            new_h = min(math.ceil(orig_h * scale_w), target_h)
        else:
            new_h = target_h
            new_w = min(math.ceil(orig_w * scale_h), target_w)
        return new_w, new_h

    def _resize_for_patching(self, image: np.array, target_resolution_hw: tuple, resample, input_data_format):
        """
        Resizes image using HF's resize, preserving aspect ratio.
        
        Args:
            image: Input image as numpy array
            target_resolution_hw: Target resolution as (height, width)
            resample: Resampling method
            input_data_format: Channel format
            
        Returns:
            Resized image as numpy array
        """
        resized_w, resized_h = self._calculate_resized_dimensions(
             get_image_size(image, channel_dim=input_data_format), target_resolution_hw
        )
        return hf_resize(image, size=(resized_h, resized_w), resample=resample, input_data_format=input_data_format)

    def _pad_for_patching(self, image: np.array, input_data_format):
        """
        Pads image to be divisible by raw patch size using numpy.pad.
        
        Args:
            image: Input image as numpy array
            input_data_format: Channel format
            
        Returns:
            Padded image as numpy array
        """
        resized_height, resized_width = get_image_size(image, channel_dim=input_data_format)
        padded_height = math.ceil(resized_height / self.vision_raw_patch_size) * self.vision_raw_patch_size
        padded_width = math.ceil(resized_width / self.vision_raw_patch_size) * self.vision_raw_patch_size
        pad_height_total = padded_height - resized_height
        pad_width_total = padded_width - resized_width

        if pad_height_total == 0 and pad_width_total == 0: return image

        pad_top = pad_height_total // 2
        pad_bottom = pad_height_total - pad_top
        pad_left = pad_width_total // 2
        pad_right = pad_width_total - pad_left

        if input_data_format == ChannelDimension.FIRST or input_data_format == "channels_first":
            padding_np = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right))
        elif input_data_format == ChannelDimension.LAST or input_data_format == "channels_last":
            padding_np = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else: raise ValueError(f"Unsupported input_data_format for padding: {input_data_format}")

        return np.pad(image, pad_width=padding_np, mode='constant', constant_values=0)

    def compute_spatial_preview_image(self, image: Image.Image) -> Image.Image:
        """
        Computes the spatial preview image (resized + padded) for patch feature visualization.
        
        Args:
            image: Original PIL image
            
        Returns:
            Processed PIL image ready for visualization
        """
        print("Computing spatial preview image (resized + padded)...")
        if image.mode != "RGB": image = image.convert("RGB")
        image_np = to_numpy_array(image)
        input_df = infer_channel_dimension_format(image_np)
        orig_size_hw = (image.height, image.width)

        target_resolution_hw = select_best_resolution(orig_size_hw, self.image_grid_pinpoints)
        print(f"  Original HxW: {orig_size_hw}, Target Res HxW: {target_resolution_hw}")

        # Use BICUBIC resampling
        resized_image_np = self._resize_for_patching(image_np, target_resolution_hw, resample=PILImageResampling.BICUBIC, input_data_format=input_df)
        resized_h, resized_w = get_image_size(resized_image_np, channel_dim=input_df)
        print(f"  Resized HxW: ({resized_h}, {resized_w})")

        padded_image_np = self._pad_for_patching(resized_image_np, input_data_format=input_df)
        padded_h, padded_w = get_image_size(padded_image_np, channel_dim=input_df)
        print(f"  Padded HxW: ({padded_h}, {padded_w})")

        # Convert back to PIL Image
        if padded_image_np.dtype != np.uint8:
             padded_image_np = np.clip(padded_image_np, 0, 255).astype(np.uint8)
        if input_df == ChannelDimension.FIRST:
             padded_image_np = padded_image_np.transpose(1, 2, 0)

        spatial_preview = Image.fromarray(padded_image_np)
        print("  Spatial preview image computed.")
        return spatial_preview

    def prepare_inputs(self, image_source: Union[str, Image.Image], prompt_text: str) -> Dict[str, Any]:
        """
        Prepares input data for logit lens analysis.
        
        Args:
            image_source: PIL image, URL, or local file path
            prompt_text: Text prompt for the model
            
        Returns:
            Dictionary containing all prepared inputs and metadata
        """
        print("Preparing inputs for Logit Lens analysis...")
        try:
            # Load image
            original_image = load_image(image_source, resize_to=None, convert_mode="RGB", verbose=False)
            original_size_hw = (original_image.height, original_image.width)
            print(f"  Original image loaded. Size HxW: {original_size_hw}")

            # Format prompt using chat template
            conversation = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "image"}]}]
            try: formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            except Exception as e:
                print(f"  Warning: Using basic prompt format due to chat template error: {e}")
                image_token = getattr(self.processor, "image_token", "<image>")
                formatted_prompt = f"USER: {image_token}\n{prompt_text} ASSISTANT:"

            # Process inputs
            inputs = self.processor(images=original_image, text=formatted_prompt, return_tensors="pt").to(self.device)
            input_ids = inputs.get("input_ids")
            if input_ids is None: raise ValueError("Processor did not return 'input_ids'.")
            print(f"  Inputs processed by processor. Input IDs shape: {input_ids.shape}")

            # Find image tokens and create mapping
            image_spans = get_image_token_spans(input_ids, self.image_token_id)
            _, image_mask = get_token_masks(input_ids, self.image_token_id)
            feature_mapping = self.create_feature_mapping(image_spans, original_size_hw)
            if not feature_mapping: raise ValueError("Failed to create feature mapping.")

            # Compute the spatial preview image
            spatial_preview_image = self.compute_spatial_preview_image(original_image)

            return {
                "inputs": inputs,
                "image_spans": image_spans,
                "image_mask": image_mask,
                "feature_mapping": feature_mapping,
                "original_size": original_size_hw,
                "original_image": original_image,
                "spatial_preview_image": spatial_preview_image,
                "prompt_text": prompt_text
            }

        except Exception as e:
             print(f"Error during input preparation: {e}")
             import traceback; traceback.print_exc()
             return {}

    def extract_hidden_states(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Extracts hidden states from all model layers.
        
        Args:
            inputs: Dictionary of model inputs from processor
            
        Returns:
            Dictionary containing hidden states and generated text
        """
        print(f"Extracting hidden states from all {self.num_layers + 1} layers (incl. embeddings)...")
        self.model.eval()
        results = {}
        try:
            with torch.no_grad():
                # Get hidden states from a single forward pass
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.get("hidden_states")
                logits = outputs.get("logits")
                if hidden_states is None or logits is None:
                     raise ValueError("Model output missing 'hidden_states' or 'logits'.")

                results["hidden_states"] = hidden_states
                results["logits"] = logits
                print(f"  Extracted {len(hidden_states)} hidden state tensors.")
                print(f"  Shape of final hidden state: {hidden_states[-1].shape}")

                # Perform a short generation for context
                print("Generating sample text...")
                gen_outputs = self.model.generate(**inputs, max_new_tokens=60, num_beams=3, return_dict_in_generate=True,
                                                  eos_token_id=self.processor.tokenizer.eos_token_id,
                                                  pad_token_id=self.processor.tokenizer.pad_token_id)
                generated_text_raw = self.processor.decode(gen_outputs.sequences[0], skip_special_tokens=True)

                # Clean up prompt part
                cleaned_gen_text = generated_text_raw
                separators = ["ASSISTANT:", "[/INST]", "USER:", "\n "]
                for sep in separators:
                    if sep in generated_text_raw:
                        parts = generated_text_raw.split(sep, 1)
                        if len(parts) > 1: cleaned_gen_text = parts[1].strip(); break
                results["generated_text"] = cleaned_gen_text
                print(f"  Sample generated text (cleaned): {cleaned_gen_text[:100]}...")

            return results

        except Exception as e:
             print(f"Error during hidden state extraction or generation: {e}")
             import traceback; traceback.print_exc()
             return {}


    def extract_token_probabilities(
        self,
        input_data: Dict[str, Any],
        outputs: Dict[str, Any],
        concepts_to_track: Optional[Dict[str, List[int]]] = None,
        cpu_offload: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extracts token probabilities by projecting hidden states through the language model head.
        
        Args:
            input_data: Dictionary from prepare_inputs
            outputs: Dictionary from extract_hidden_states
            concepts_to_track: Dictionary mapping concept names to token IDs to track
            cpu_offload: Whether to offload tensors to CPU during computation
            
        Returns:
            Dictionary mapping layer indices to probability maps for each feature type and concept
        """
        print("Extracting token probabilities (Logit Lens)...")
        if not concepts_to_track:
            print("  Warning: No concepts_to_track provided. Cannot extract probabilities.")
            return {}

        hidden_states = outputs.get("hidden_states")
        feature_mapping = input_data.get("feature_mapping")
        if not hidden_states or not feature_mapping:
            print("  Error: Missing 'hidden_states' or 'feature_mapping'. Cannot proceed.")
            return {}

        token_probs_by_layer: Dict[int, Dict[str, Any]] = {}
        num_layers_to_process = len(hidden_states)
        vocab_size = getattr(self.model.config, "vocab_size", -1)
        if vocab_size == -1: print("Warning: Could not determine vocab size from model config.")

        # Validate concept token IDs against vocab size
        valid_concepts = {}
        for concept, ids in concepts_to_track.items():
             valid_ids = [tid for tid in ids if 0 <= tid < vocab_size] if vocab_size > 0 else ids
             if valid_ids: valid_concepts[concept] = valid_ids
             else: print(f"  Warning: Concept '{concept}' has no valid token IDs within vocab size {vocab_size}. Skipping.")
        if not valid_concepts:
             print("  Error: No valid concepts left to track after vocabulary check.")
             return {}
        print(f"  Tracking valid concepts: {list(valid_concepts.keys())}")

        # Determine LM head device
        try: lm_head_device = next(self.lm_head.parameters()).device
        except StopIteration: lm_head_device = self.device

        self.lm_head.eval()

        with torch.no_grad():
            for layer_idx in range(num_layers_to_process):
                layer_hidden = hidden_states[layer_idx]
                orig_hidden_device = layer_hidden.device

                # Move hidden state if needed
                if layer_hidden.device != lm_head_device:
                    try: layer_hidden = layer_hidden.to(lm_head_device)
                    except Exception as e: print(f"Warning: Skip layer {layer_idx}, failed to move hidden state: {e}"); continue

                # Project & Softmax
                try:
                     logits = self.lm_head(layer_hidden)
                     probs = torch.softmax(logits.float(), dim=-1)
                except Exception as e:
                     print(f"Warning: Skip layer {layer_idx}, error during projection/softmax: {e}"); continue
                finally:
                    if layer_hidden.device != orig_hidden_device: del layer_hidden
                    if 'logits' in locals() and logits.device == lm_head_device: del logits

                # Extract probabilities for tracked concepts
                layer_results = {"base_feature": {}, "patch_feature": {}, "newline_feature": {}}
                seq_len = probs.shape[1]

                # --- Base Features ---
                base_info = feature_mapping.get("base_feature", {})
                if base_info.get("positions") and base_info.get("grid"):
                    grid_h, grid_w = base_info["grid"]
                    base_grids = {c: np.zeros((grid_h, grid_w), dtype=np.float32) for c in valid_concepts}
                    for token_idx, (r, c) in base_info["positions"].items():
                        if 0 <= token_idx < seq_len:
                            for concept, token_ids in valid_concepts.items():
                                concept_probs = probs[0, token_idx, token_ids]
                                base_grids[concept][r, c] = torch.max(concept_probs).item()
                    layer_results["base_feature"] = base_grids

                # --- Patch Features ---
                patch_info = feature_mapping.get("patch_feature", {})
                if patch_info.get("positions") and patch_info.get("grid_unpadded"):
                    grid_h, grid_w = patch_info["grid_unpadded"]
                    patch_grids = {c: np.zeros((grid_h, grid_w), dtype=np.float32) for c in valid_concepts}
                    for token_idx, (r, c) in patch_info["positions"].items():
                         if 0 <= token_idx < seq_len and 0 <= r < grid_h and 0 <= c < grid_w:
                             for concept, token_ids in valid_concepts.items():
                                 concept_probs = probs[0, token_idx, token_ids]
                                 patch_grids[concept][r, c] = torch.max(concept_probs).item()
                    layer_results["patch_feature"] = patch_grids

                # --- Newline Features ---
                newline_info = feature_mapping.get("newline_feature", {})
                if newline_info.get("positions"):
                    newline_dict = {c: {} for c in valid_concepts}
                    for token_idx, row_idx in newline_info["positions"].items():
                        if 0 <= token_idx < seq_len:
                            for concept, token_ids in valid_concepts.items():
                                concept_probs = probs[0, token_idx, token_ids]
                                newline_dict[concept][row_idx] = max(newline_dict[concept].get(row_idx, 0.0), torch.max(concept_probs).item())
                    layer_results["newline_feature"] = newline_dict

                # --- Store results for the layer ---
                token_probs_by_layer[layer_idx] = layer_results

                # --- Offload or Cleanup ---
                if cpu_offload:
                     if probs.device != torch.device('cpu'):
                          try: probs_cpu = probs.cpu(); del probs
                          except Exception as e: print(f"Warning: Failed to move/delete probs tensor for layer {layer_idx}: {e}")
                elif probs.device == lm_head_device:
                    del probs

                # Periodic CUDA cache clearing
                if lm_head_device.type == 'cuda' and layer_idx % 5 == 0: torch.cuda.empty_cache()

        print("Token probability extraction complete.")
        return token_probs_by_layer