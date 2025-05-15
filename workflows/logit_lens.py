"""
Logit Lens workflow for VLM interpretability.
Projects hidden states through the LM head to analyze token predictions at each layer.
"""

import torch
import os
import gc
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image

from runtime.io import TraceIO
from preprocess.input_builder import prepare_inputs
from runtime.generation import GenerationMixin
from backends.logit_backend import LogitLensBackend
from runtime.selection import SelectionConfig
from runtime.cache import ActivationCache
from preprocess.mapper import VisionMapper
from analysis.logit_viz import create_composite_image, visualize_token_probabilities

# Configure logging
logger = logging.getLogger("logit_lens.workflow")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class LogitLensWorkflow(GenerationMixin):
    """
    Workflow for Logit Lens analysis.
    Manages input preparation, hidden state projection, and result processing.
    """
    
    def __init__(self, model: torch.nn.Module, processor: Any, output_dir: str, 
                 selection_config: Optional[SelectionConfig] = None,
                 debug: bool = False):
        """
        Initialize the Logit Lens workflow.
        
        Args:
            model: The model to analyze
            processor: The model's processor
            output_dir: Directory for output files
            selection_config: Configuration for token selection
            debug: Enable debug logging
        """
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.debug = debug
        
        # Set debug level if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize selection config
        self.config = selection_config or SelectionConfig()
        
        # Get model device
        self.device = next(model.parameters()).device
        
        # Initialize I/O handler
        self.io = TraceIO(output_dir)
        
        # Initialize cache with CPU offload to save GPU memory
        self.cache = ActivationCache(cpu_offload=True)
        
        # Get number of layers from config
        self.num_layers = getattr(model.config, "num_hidden_layers", None)
        if self.num_layers is None:
            # Try to infer from model structure
            if hasattr(model, "language_model") and hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
                self.num_layers = len(model.language_model.model.layers)
            else:
                self.num_layers = 32  # Reasonable default for many VLMs
                logger.warning(f"Could not determine number of layers. Using default: {self.num_layers}")
                
        logger.info(f"Detected {self.num_layers} model layers")
        
        # Initialize backend
        self.backend = LogitLensBackend(model, self.cache, self.device)
    
    def prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """
        Prepare inputs for the model with keyword arguments.
        
        Args:
            image: Input image
            prompt: Text prompt
            
        Returns:
            Dictionary of prepared inputs and metadata
        """
        # Use keyword arguments to match input_builder.prepare_inputs signature
        return prepare_inputs(
            model=self.model, 
            processor=self.processor, 
            image=image, 
            prompt=prompt
        )
    
    def generate(self, input_data: Dict[str, Any], num_tokens: int = 1) -> Dict[str, Any]:
        """
        Generate tokens from the model.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate
            
        Returns:
            Dictionary with generation results
        """
        return self.autoregressive_generate(input_data["inputs"], num_tokens)
    
    def run_projection(self, input_data: Dict[str, Any], target_layers: Optional[List[int]] = None, 
                    token_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run Logit Lens projection for specified layers and tokens.
        Uses vectorized batch processing for efficiency.
        
        Args:
            input_data: Prepared input data
            target_layers: List of layer indices to analyze
            token_indices: List of token indices to project
            
        Returns:
            Dictionary of projection results
        """
        try:
            # Extract inputs
            inputs = input_data["inputs"]
            
            # If target_layers not specified, use all layers plus embedding layer (-1)
            if target_layers is None:
                target_layers = [-1] + list(range(self.num_layers))
                
            # If token_indices not specified, use all tokens
            if token_indices is None:
                token_indices = list(range(inputs["input_ids"].shape[1]))
            
            # Filter out duplicate indices
            token_indices = sorted(list(set(token_indices)))
                
            logger.info(f"Running Logit Lens projection for {len(target_layers)} layers and {len(token_indices)} tokens")
            
            # Ensure hidden states are cached for all target layers
            self.backend.ensure_cache(inputs, target_layers)
            
            # Prepare results structure
            results = {
                "layers": {},
                "tokens": {},
                "layer_token_grid": {},
                "token_probs": {}  # Add this to store token probabilities for visualization
            }
            
            # Get mappings from token indices to token text
            token_text_map = {}
            for idx in token_indices:
                if idx < inputs["input_ids"].shape[1]:
                    token_id = inputs["input_ids"][0, idx].item()
                    token_text = self.processor.tokenizer.decode([token_id])
                    token_text_map[idx] = token_text
            
            # Initialize token probability structure for visualization
            for layer_idx in target_layers:
                results["token_probs"][layer_idx] = {
                    "base_feature": {},
                    "patch_feature": {},
                    "newline_feature": {}
                }
            
            # Process each layer with vectorized batch projection for efficiency
            for layer_idx in target_layers:
                logger.info(f"Processing layer {layer_idx}")
                
                # Use vectorized batch projection for all tokens in the layer
                layer_batch_results = self.backend.project_tokens_batch(
                    layer_idx=layer_idx,
                    token_indices=token_indices,
                    tokenizer=self.processor.tokenizer,
                    top_k=5
                )
                
                # Process results for this layer
                layer_results = {}
                
                for token_idx, token_projection in layer_batch_results.items():
                    # Add token text to the projection
                    token_projection["token_text"] = token_text_map.get(token_idx, "")
                    
                    # Store in results
                    layer_results[token_idx] = token_projection
                    
                    # Also organize by token
                    if token_idx not in results["tokens"]:
                        results["tokens"][token_idx] = {}
                    results["tokens"][token_idx][layer_idx] = token_projection
                    
                    # Add to layer-token grid
                    grid_key = f"{layer_idx}_{token_idx}"
                    results["layer_token_grid"][grid_key] = {
                        "layer": layer_idx,
                        "token": token_idx,
                        "token_text": token_text_map.get(token_idx, ""),
                        "top_prediction": token_projection["predictions"][0] if token_projection["predictions"] else None
                    }
                    
                    # Extract and organize token probabilities for visualization
                    if "feature_mapping" in input_data:
                        feature_mapping = input_data["feature_mapping"]
                        
                        # Check if this token is in base feature positions
                        if "base_feature" in feature_mapping and token_idx in feature_mapping["base_feature"].get("positions", {}):
                            # Organize base feature probabilities
                            for concept in self._extract_concepts(token_projection):
                                if concept not in results["token_probs"][layer_idx]["base_feature"]:
                                    # Initialize base feature grid
                                    grid_h, grid_w = feature_mapping["base_feature"]["grid"]
                                    results["token_probs"][layer_idx]["base_feature"][concept] = np.zeros((grid_h, grid_w), dtype=np.float32)
                                
                                # Get position and probability
                                row, col = feature_mapping["base_feature"]["positions"][token_idx]
                                prob = self._get_concept_probability(token_projection, concept)
                                
                                # Store in grid
                                try:
                                    results["token_probs"][layer_idx]["base_feature"][concept][row, col] = prob
                                except IndexError:
                                    logger.warning(f"Index error storing base feature prob at ({row}, {col}) for layer {layer_idx}, concept {concept}")
                        
                        # Check if this token is in patch feature positions
                        elif "patch_feature" in feature_mapping and token_idx in feature_mapping["patch_feature"].get("positions", {}):
                            # Organize patch feature probabilities
                            for concept in self._extract_concepts(token_projection):
                                if concept not in results["token_probs"][layer_idx]["patch_feature"]:
                                    # Initialize patch feature grid
                                    grid_h, grid_w = feature_mapping["patch_feature"]["grid_unpadded"]
                                    results["token_probs"][layer_idx]["patch_feature"][concept] = np.zeros((grid_h, grid_w), dtype=np.float32)
                                
                                # Get position and probability
                                row, col = feature_mapping["patch_feature"]["positions"][token_idx]
                                prob = self._get_concept_probability(token_projection, concept)
                                
                                # Store in grid
                                try:
                                    results["token_probs"][layer_idx]["patch_feature"][concept][row, col] = prob
                                except IndexError:
                                    logger.warning(f"Index error storing patch feature prob at ({row}, {col}) for layer {layer_idx}, concept {concept}")
                        
                        # Check if this token is a newline token
                        elif "newline_feature" in feature_mapping and token_idx in feature_mapping["newline_feature"].get("positions", {}):
                            # Organize newline feature probabilities
                            for concept in self._extract_concepts(token_projection):
                                if concept not in results["token_probs"][layer_idx]["newline_feature"]:
                                    results["token_probs"][layer_idx]["newline_feature"][concept] = {}
                                
                                # Get row index and probability
                                row_idx = feature_mapping["newline_feature"]["positions"][token_idx]
                                prob = self._get_concept_probability(token_projection, concept)
                                
                                # Store in dictionary
                                results["token_probs"][layer_idx]["newline_feature"][concept][row_idx] = prob
                
                # Store layer results
                results["layers"][layer_idx] = layer_results
                
                # Clear CUDA cache for memory efficiency
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Add metadata
            results["metadata"] = {
                "num_layers": self.num_layers,
                "num_tokens": len(token_indices),
                "target_layers": target_layers,
                "token_indices": token_indices
            }
            
            logger.info(f"Projection complete for {len(results['layers'])} layers")

            # Visualize token probabilities if we have the necessary data
            if "token_probs" in results and results["token_probs"] and "feature_mapping" in input_data:
                # Extract feature mapping & images from input_data
                feature_mapping = input_data.get("feature_mapping", {})
                orig_img = input_data.get("original_image")
                spatial_preview = input_data.get("spatial_preview_image")
                
                # If spatial_preview_image is not available but original_image is, 
                # create a preview from the original
                if spatial_preview is None and orig_img is not None:
                    spatial_preview = self._create_spatial_preview(orig_img, feature_mapping)
                
                if orig_img is not None and spatial_preview is not None:
                    # Infer concepts present
                    concepts = self._extract_available_concepts(results["token_probs"])
                    
                    if concepts:
                        # Call our viz helper
                        viz_dir = os.path.join(self.output_dir, "logit_lens_viz")
                        saved_files = visualize_token_probabilities(
                            token_probs=results["token_probs"],
                            feature_mapping=feature_mapping,
                            original_image=orig_img,
                            spatial_preview=spatial_preview,
                            concepts=concepts,
                            output_dir=viz_dir
                        )
                        logger.info(f"Saved {len(saved_files)} logit-lens viz files under {viz_dir}")
                        results["visualization_paths"] = saved_files
            
            return results
            
        except Exception as e:
            logger.error(f"Error during Logit Lens projection: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _extract_concepts(self, token_projection: Dict[str, Any]) -> List[str]:
        """
        Extract concept names from token predictions.
        
        Args:
            token_projection: Dictionary of token projection results
            
        Returns:
            List of concept names
        """
        concepts = []
        
        if "predictions" in token_projection:
            for pred in token_projection["predictions"]:
                if "text" in pred:
                    # Clean up text to use as concept
                    text = pred["text"].strip()
                    if text and len(text) < 20:  # Limit to reasonable concepts
                        concepts.append(text)
        
        return concepts

    def _get_concept_probability(self, token_projection: Dict[str, Any], concept: str) -> float:
        """
        Get probability for a specific concept from token predictions.
        
        Args:
            token_projection: Dictionary of token projection results
            concept: Concept name to look for
            
        Returns:
            Probability value (0.0-1.0)
        """
        if "predictions" in token_projection:
            for pred in token_projection["predictions"]:
                if pred.get("text", "").strip() == concept:
                    return pred.get("prob", 0.0)
        
        return 0.0

    def _extract_available_concepts(self, token_probs: Dict[int, Dict[str, Any]]) -> List[str]:
        """
        Extract available concepts from token probability data.
        
        Args:
            token_probs: Dictionary of token probabilities
            
        Returns:
            List of available concept names
        """
        concepts = set()
        
        # Look through all layers
        for layer_idx, layer_data in token_probs.items():
            # Check base feature concepts
            if "base_feature" in layer_data:
                concepts.update(layer_data["base_feature"].keys())
            
            # Check patch feature concepts
            if "patch_feature" in layer_data:
                concepts.update(layer_data["patch_feature"].keys())
            
            # Check newline feature concepts
            if "newline_feature" in layer_data:
                concepts.update(layer_data["newline_feature"].keys())
        
        return sorted(list(concepts))

    def _create_spatial_preview(self, original_image: Image.Image, feature_mapping: Dict[str, Any]) -> Optional[Image.Image]:
        """
        Create spatial preview image for patch feature visualization.
        
        Args:
            original_image: Original PIL image
            feature_mapping: Feature mapping dictionary
            
        Returns:
            Spatial preview PIL image or None if creation fails
        """
        try:
            # Check if we have the necessary information
            if not feature_mapping or "patch_size" not in feature_mapping:
                logger.warning("Cannot create spatial preview: missing patch_size in feature_mapping")
                return None
            
            if "resized_dimensions" not in feature_mapping or "padded_dimensions" not in feature_mapping:
                logger.warning("Cannot create spatial preview: missing dimensions in feature_mapping")
                return None
            
            # Extract parameters
            patch_size = feature_mapping["patch_size"]
            resized_w, resized_h = feature_mapping["resized_dimensions"]
            padded_w, padded_h = feature_mapping["padded_dimensions"]
            
            # Resize original image while preserving aspect ratio
            if original_image.mode != "RGB":
                original_image = original_image.convert("RGB")
            
            # Resize to target dimensions
            resized_image = original_image.resize((resized_w, resized_h), Image.LANCZOS)
            
            # Create padded image
            padded_image = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))
            
            # Calculate padding
            pad_left = (padded_w - resized_w) // 2
            pad_top = (padded_h - resized_h) // 2
            
            # Paste resized image onto padded canvas
            padded_image.paste(resized_image, (pad_left, pad_top))
            
            return padded_image
            
        except Exception as e:
            logger.error(f"Error creating spatial preview: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_image_tokens(self, input_data: Dict[str, Any], target_layers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze image tokens specifically, including original image dimensions.
        
        Args:
            input_data: Prepared input data
            target_layers: List of layer indices to analyze
            
        Returns:
            Dictionary of analysis results for image tokens
        """
        try:
            # Extract image indices
            image_indices = input_data["image_indices"].tolist()
            
            if not image_indices:
                logger.warning("No image tokens found in input")
                return {"error": "No image tokens found"}
                
            logger.info(f"Analyzing {len(image_indices)} image tokens")
            
            # Get original image dimensions
            original_image = input_data.get("original_image")
            original_image_size_hw = None
            if original_image:
                original_image_size_hw = (original_image.height, original_image.width)
                logger.info(f"Original image dimensions (HxW): {original_image_size_hw}")
            
            # Run projection for image tokens
            results = self.run_projection(
                input_data=input_data,
                target_layers=target_layers,
                token_indices=image_indices
            )
            
            # Add image token mapping with original dimensions
            feature_mapping = VisionMapper.map_tokens_to_grid(
                self.model, 
                [(min(image_indices), max(image_indices))],
                original_image_size_hw=original_image_size_hw
            )
            results["feature_mapping"] = feature_mapping
            
            # Add original image dimensions to results
            if original_image_size_hw:
                results["original_image_size"] = {
                    "height": original_image_size_hw[0],
                    "width": original_image_size_hw[1]
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during image token analysis: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def analyze_sequence(self, input_data: Dict[str, Any], target_layers: Optional[List[int]] = None, 
                       skip_image_tokens: bool = False) -> Dict[str, Any]:
        """
        Analyze the full input sequence.
        
        Args:
            input_data: Prepared input data
            target_layers: List of layer indices to analyze
            skip_image_tokens: Whether to skip image tokens in the analysis
            
        Returns:
            Dictionary of analysis results for the sequence
        """
        try:
            # Determine which tokens to analyze
            token_indices = None
            
            if skip_image_tokens:
                # Use only text tokens
                token_indices = input_data["text_indices"].tolist()
                logger.info(f"Analyzing {len(token_indices)} text tokens (skipping image tokens)")
            else:
                # Use all tokens
                token_indices = list(range(input_data["inputs"]["input_ids"].shape[1]))
                logger.info(f"Analyzing all {len(token_indices)} tokens")
                
            # Run projection
            results = self.run_projection(
                input_data=input_data,
                target_layers=target_layers,
                token_indices=token_indices
            )
            
            # Add token type information
            text_indices_set = set(input_data["text_indices"].tolist())
            image_indices_set = set(input_data["image_indices"].tolist())
            
            for token_idx in results["tokens"]:
                if token_idx in text_indices_set:
                    token_type = "text"
                elif token_idx in image_indices_set:
                    token_type = "image"
                else:
                    token_type = "generated"
                    
                for layer_idx in results["tokens"][token_idx]:
                    results["tokens"][token_idx][layer_idx]["token_type"] = token_type
            
            return results
            
        except Exception as e:
            logger.error(f"Error during sequence analysis: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def analyze_generated_tokens(self, input_data: Dict[str, Any], num_tokens: int = 1,
                               target_layers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Generate tokens and analyze their logit lens projections.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate
            target_layers: List of layer indices to analyze
            
        Returns:
            Dictionary of analysis results for generated tokens
        """
        try:
            # Generate tokens
            logger.info(f"Generating and analyzing {num_tokens} tokens")
            gen_result = self.generate(input_data, num_tokens)
            
            # Get generated token indices
            generated_indices = [token["index"] for token in gen_result["generated_tokens"]]
            
            if not generated_indices:
                logger.warning("No tokens were generated")
                return {"error": "No tokens were generated"}
                
            # Prepare combined input data with generated tokens
            combined_inputs = {
                "inputs": gen_result["inputs"],
                "text_indices": input_data["text_indices"],
                "image_indices": input_data["image_indices"],
                "original_image": input_data.get("original_image")
            }
            
            # Run projection on generated tokens
            results = self.run_projection(
                input_data=combined_inputs,
                target_layers=target_layers,
                token_indices=generated_indices
            )
            
            # Add generation metadata
            results["generation_info"] = {
                "generated_tokens": gen_result["generated_tokens"],
                "original_seq_len": gen_result["original_seq_len"],
                "full_sequence": gen_result["full_sequence"]
            }
            
            # For each token, add token type (should all be "generated")
            for token_idx in generated_indices:
                for layer_idx in results["tokens"].get(token_idx, {}):
                    if layer_idx in results["tokens"][token_idx]:
                        results["tokens"][token_idx][layer_idx]["token_type"] = "generated"
            
            return results
            
        except Exception as e:
            logger.error(f"Error during generated token analysis: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def compare_layers(self, input_data: Dict[str, Any], token_idx: int, 
                      layers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compare projections of a single token across multiple layers.
        
        Args:
            input_data: Prepared input data
            token_idx: Token index to analyze
            layers: List of layer indices to compare
            
        Returns:
            Dictionary of comparison results
        """
        try:
            # If layers not specified, use a subset of layers
            if layers is None:
                # Include embedding, early, middle, and late layers
                layers = [-1, 0, self.num_layers // 4, self.num_layers // 2, 
                         3 * self.num_layers // 4, self.num_layers - 1]
                
            logger.info(f"Comparing token {token_idx} across layers: {layers}")
            
            # Validate token index
            if token_idx >= input_data["inputs"]["input_ids"].shape[1]:
                return {"error": f"Token index {token_idx} out of bounds"}
                
            # Get token info
            token_id = input_data["inputs"]["input_ids"][0, token_idx].item()
            token_text = self.processor.tokenizer.decode([token_id])
            
            # Determine token type
            text_indices_set = set(input_data["text_indices"].tolist())
            image_indices_set = set(input_data["image_indices"].tolist())
            
            if token_idx in text_indices_set:
                token_type = "text"
            elif token_idx in image_indices_set:
                token_type = "image"
            else:
                token_type = "generated"
                
            # Run projection
            results = self.run_projection(
                input_data=input_data,
                target_layers=layers,
                token_indices=[token_idx]
            )
            
            # Add token info
            results["token_info"] = {
                "index": token_idx,
                "id": token_id,
                "text": token_text,
                "type": token_type
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during layer comparison: {e}")
            import traceback
            traceback.print_exc()
            raise