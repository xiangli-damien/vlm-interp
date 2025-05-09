# preprocess/mapper.py
"""
Utilities for vision processing in VLM interpretability.
"""

import torch
import math
from typing import Dict, List, Tuple, Any, Optional
import logging

# Configure logging
logger = logging.getLogger("vision_utils")
logger.setLevel(logging.INFO)

class VisionMapper:
    """
    Utilities for mapping image tokens to visual features and grid positions.
    """
    
    @staticmethod
    def get_embedding_projection(model, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get the token embedding projections from the model.
        
        Args:
            model: The VLM model
            input_ids: Input token IDs
            
        Returns:
            Token embedding tensor
        """
        try:
            # Access the embedding layer
            if hasattr(model, 'get_input_embeddings'):
                # Use the model's own method if available
                embedding_layer = model.get_input_embeddings()
            elif hasattr(model, 'language_model') and hasattr(model.language_model, 'get_input_embeddings'):
                # Try language model component
                embedding_layer = model.language_model.get_input_embeddings()
            elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embeddings'):
                # Direct access to embeddings
                embedding_layer = model.language_model.embeddings.word_embeddings
            else:
                logger.warning("Could not find embedding layer in model")
                return None
            
            # Get embeddings
            with torch.no_grad():
                embeddings = embedding_layer(input_ids)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embedding projections: {e}")
            return None

    @staticmethod
    def map_tokens_to_grid(model, image_spans: List[Tuple[int, int]], original_image_size_hw: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Maps image tokens to a grid structure based on model configuration.
        Handles both base grid features and high-resolution patch features.
        
        Args:
            model: The VLM model
            image_spans: List of (start_idx, end_idx) tuples for image token spans
            original_image_size_hw: Optional original image size as (height, width)
            
        Returns:
            Dictionary of feature mapping information
        """
        try:
            if not image_spans:
                logger.warning("No image spans provided")
                return {}
                
            # Extract model configuration
            model_config = getattr(model, "config", None)
            vision_config = getattr(model_config, "vision_config", None) if model_config else None
            
            if not vision_config:
                logger.warning("Vision configuration not found in model")
                return {"spans": image_spans}
                
            # Get patch size and image size
            patch_size = getattr(vision_config, "patch_size", 14)
            image_size = getattr(vision_config, "image_size", 336)
            grid_size = image_size // patch_size if patch_size > 0 else 24
            
            logger.info(f"Base grid: {grid_size}x{grid_size}, patch size: {patch_size}")
            
            # --- Base Feature Mapping ---
            span_start = image_spans[0][0]
            span_end = image_spans[-1][1]
            total_image_tokens = span_end - span_start + 1
            
            expected_base_tokens = grid_size * grid_size
            actual_base_token_count = min(expected_base_tokens, total_image_tokens)
            base_end_idx = span_start + actual_base_token_count - 1
            
            mapping = {
                "base_feature": {
                    "start_idx": span_start,
                    "end_idx": base_end_idx,
                    "grid": (grid_size, grid_size),  # (rows, cols)
                    "positions": {}
                },
                "spans": image_spans,
                "patch_size": patch_size,
                "image_size": image_size
            }
            
            # Add original image size if provided
            if original_image_size_hw:
                mapping["original_size"] = original_image_size_hw
                
                # Calculate resized dimensions (aspect ratio preserved)
                if model_config and hasattr(model_config, "image_grid_pinpoints"):
                    grid_pinpoints = getattr(model_config, "image_grid_pinpoints", [])
                    target_res_hw = VisionMapper._select_best_resolution(original_image_size_hw, grid_pinpoints)
                    resized_wh = VisionMapper._calculate_resized_dimensions(original_image_size_hw, target_res_hw)
                    mapping["resized_dimensions"] = resized_wh
                    mapping["best_resolution"] = (target_res_hw[1], target_res_hw[0])  # Convert to WH format
                    
                    # Calculate padded dimensions
                    w, h = resized_wh
                    padded_h = math.ceil(h / patch_size) * patch_size
                    padded_w = math.ceil(w / patch_size) * patch_size
                    mapping["padded_dimensions"] = (padded_w, padded_h)
            
            # Map base feature tokens to grid positions
            for i in range(actual_base_token_count):
                token_idx = span_start + i
                row = i // grid_size
                col = i % grid_size
                mapping["base_feature"]["positions"][token_idx] = (row, col)
            
            # --- Patch Feature Mapping (for high-resolution support) ---
            if len(image_spans) > 1 or total_image_tokens > expected_base_tokens:
                patch_start_idx = base_end_idx + 1
                if patch_start_idx <= span_end:
                    # Model has high-res feature support
                    patch_tokens_available = span_end - patch_start_idx + 1
                    
                    # Get high-res configuration
                    if model_config and hasattr(model_config, "image_grid_pinpoints"):
                        grid_pinpoints = getattr(model_config, "image_grid_pinpoints", [])
                        
                        # Check if we have resized dimensions
                        if "resized_dimensions" in mapping:
                            resized_w, resized_h = mapping["resized_dimensions"]
                            
                            # Calculate unpadded grid
                            unpadded_grid_h = math.ceil(resized_h / patch_size)
                            unpadded_grid_w = math.ceil(resized_w / patch_size)
                            
                            # Create patch feature mapping
                            mapping["patch_feature"] = {
                                "start_idx": patch_start_idx,
                                "end_idx": span_end,
                                "grid_unpadded": (unpadded_grid_h, unpadded_grid_w),
                                "grid_for_visualization": (
                                    mapping["padded_dimensions"][1] // patch_size,
                                    mapping["padded_dimensions"][0] // patch_size
                                ),
                                "positions": {}
                            }
                            
                            # Map positions - this is simplified and may need refinement
                            for i in range(min(patch_tokens_available, unpadded_grid_h * unpadded_grid_w)):
                                token_idx = patch_start_idx + i
                                row = i // unpadded_grid_w
                                col = i % unpadded_grid_w
                                mapping["patch_feature"]["positions"][token_idx] = (row, col)
                                
                            # Check for newline tokens in structured layout
                            has_newlines = unpadded_grid_h > 1
                            if has_newlines:
                                # In some models, newlines are inserted after each row
                                mapping["newline_feature"] = {
                                    "positions": {},
                                    "start_idx": -1,
                                    "end_idx": -1
                                }
                                
                                # Simple heuristic to detect newline tokens
                                tokens_per_row = unpadded_grid_w
                                for i in range(unpadded_grid_h - 1):
                                    newline_pos = patch_start_idx + tokens_per_row * (i + 1) + i
                                    if newline_pos <= span_end:
                                        mapping["newline_feature"]["positions"][newline_pos] = i
                                        if mapping["newline_feature"]["start_idx"] == -1:
                                            mapping["newline_feature"]["start_idx"] = newline_pos
                                        mapping["newline_feature"]["end_idx"] = newline_pos
            
            return mapping
        
        except Exception as e:
            logger.error(f"Error in map_tokens_to_grid: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "spans": image_spans}

    @staticmethod
    def _select_best_resolution(image_size_hw: Tuple[int, int], pinpoints: List[List[int]]) -> Tuple[int, int]:
        """Select best resolution from pinpoints based on image size."""
        # Implementation similar to select_best_resolution from transformers
        orig_h, orig_w = image_size_hw
        
        # Default if no pinpoints
        if not pinpoints:
            return (336, 336)
        
        # Find best fit
        best_fit = pinpoints[0]
        best_area = 0
        
        for point in pinpoints:
            h, w = point
            
            if h >= orig_h and w >= orig_w:
                area = h * w
                if best_area == 0 or area < best_area:
                    best_area = area
                    best_fit = point
        
        # If no good fit found, use the largest
        if best_area == 0:
            best_area = 0
            for point in pinpoints:
                h, w = point
                area = h * w
                if area > best_area:
                    best_area = area
                    best_fit = point
        
        return (best_fit[0], best_fit[1])

    @staticmethod
    def _calculate_resized_dimensions(orig_size_hw: Tuple[int, int], target_res_hw: Tuple[int, int]) -> Tuple[int, int]:
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