"""
Semantic tracing for VLMs: Combines saliency analysis with logit lens to track 
information flow through model layers, revealing how concepts evolve in the
model's reasoning process. Optimized version with memory efficiency improvements
and fixes for numerical stability and edge cases.
"""

import torch
import math
import numpy as np
import gc
import os
import json
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from tqdm.auto import tqdm
from collections import defaultdict

# Import components from existing modules
from analyzer.saliency import calculate_saliency_scores
from analyzer.logit_lens import LLaVANextLogitLensAnalyzer
from utils.hook_utils import GradientAttentionCapture
from utils.model_utils import get_llm_attention_layer_names
from utils.data_utils import get_token_indices

class EnhancedSemanticTracer:
    """
    Traces information flow through VLM layers by combining saliency scores or attention maps with
    logit lens projections to reveal how concepts evolve and flow from input tokens
    to generated tokens. Supports multiple tracing modes and coverage-based node selection.
    """
    
    def __init__(
        self,
        model,
        processor,
        top_k: Optional[int] = None,  # Deprecated parameter
        device: Optional[str] = None,
        output_dir: str = "semantic_tracing_results",
        cpu_offload: bool = True,
        layer_batch_size: int = 2,
        logit_lens_concepts: Optional[List[str]] = None,
        normalize_weights: bool = True,
        beta_target: float = 0.8,  # Target-level coverage threshold
        beta_layer: float = 1.0,   # Layer-level coverage threshold 
        min_keep: int = 3,         # Minimum nodes to keep per target
        max_keep: int = 8,        # Maximum nodes to keep per target
        min_keep_layer: int = 10,   # Minimum nodes to keep per layer
        max_keep_layer: int = 500, # Maximum nodes to keep per layer
        epsilon: float = 1e-7,     
        debug: bool = False,
    ):
        """
        Initialize the enhanced semantic tracer.
        
        Args:
            model: The VLM model (LLaVA-Next)
            processor: The corresponding processor
            top_k: Deprecated parameter - use beta_target and min/max_keep instead
            device: Device to use (defaults to model's device)
            output_dir: Directory to save results
            cpu_offload: Whether to offload tensors to CPU when possible
            layer_batch_size: Number of layers to process at once for gradient computation
            logit_lens_concepts: List of concepts to track with logit lens
            normalize_weights: Whether to normalize token importance weights between layers
            beta_target: Coverage threshold for selecting source nodes for each target token
            beta_layer: Coverage threshold for pruning all source nodes at a layer level
            min_keep: Minimum number of source nodes to keep per target token
            max_keep: Maximum number of source nodes to keep per target token
            min_keep_layer: Minimum nodes to keep after layer-level pruning
            max_keep_layer: Maximum nodes to keep after layer-level pruning
            epsilon: Small value to inject for numerical stability
            debug: Whether to print additional debug information
        """
        self.model = model
        self.processor = processor
        self.device = device or model.device
        self.output_dir = output_dir
        self.cpu_offload = cpu_offload
        self.layer_batch_size = layer_batch_size
        self.normalize_weights = normalize_weights
        self.debug = debug
        self.epsilon = epsilon  # Increased from 1e-6 to 1e-4 for better numerical stability
        
        # Handle deprecated top_k parameter
        if top_k is not None:
            print(f"Warning: 'top_k' parameter is deprecated and will be removed in a future version. "
                  f"Using coverage-based selection with beta_target={beta_target} instead.")
            self.top_k = top_k  # Keep for backward compatibility
        else:
            self.top_k = 3  # Default value for backward compatibility
        
        # Coverage-based node selection parameters
        self.beta_target = beta_target
        self.beta_layer = beta_layer
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.min_keep_layer = min_keep_layer
        self.max_keep_layer = max_keep_layer
        
        # Create LogitLens analyzer
        self.logit_lens = LLaVANextLogitLensAnalyzer(model, processor)
        
        # Set default concepts if none provided
        self.logit_lens_concepts = logit_lens_concepts or ["cat", "dog", "person", "building", "water", "sky", "car"]
        
        # Get language model attention layer names
        self.attention_layer_names = get_llm_attention_layer_names(model)
        if not self.attention_layer_names:
            raise ValueError("Could not find attention layers in the model")
        
        # Map layer names to indices for easier lookups
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(self.attention_layer_names)}
        self.num_layers = len(self.attention_layer_names)
        
        # Determine image token ID
        self.image_token_id = getattr(model.config, "image_token_index", 32000)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "csv_data"), exist_ok=True)
        
        # Track hidden states cache to avoid repeated forward passes
        self.hidden_states_cache = {}
        self.saliency_cache = {}
        self.attention_cache = {}
        
        # Create a unique trace ID counter for identification
        self.trace_id_counter = 0
        self.layer_att_means: List[float] = []
        self.layer_sal_norms: List[float] = []
        
        # Special tokens mapping for handling empty token display issues
        self.SPECIAL_TEXT = {13: "\\n", 28705: "_"}
        
        print(f"Initialized EnhancedSemanticTracer with {self.num_layers} attention layers")
        print(f"Coverage-based node selection: β_target={beta_target}, β_layer={beta_layer}")
        print(f"Node limits: min_keep={min_keep}, max_keep={max_keep}, min_keep_layer={min_keep_layer}, max_keep_layer={max_keep_layer}")
        print(f"Numerical stability epsilon: {epsilon}")
    
    def _get_layer_idx(self, layer_name_or_idx):
        """Helper to get consistent layer index from name or index"""
        if isinstance(layer_name_or_idx, int):
            return layer_name_or_idx
        return self.layer_name_to_idx.get(layer_name_or_idx, -1)
    
    def _get_layer_name(self, layer_idx):
        """Helper to get layer name from index"""
        if 0 <= layer_idx < len(self.attention_layer_names):
            return self.attention_layer_names[layer_idx]
        return None
    
    def _sanitize_text_for_display(self, text: str) -> str:
        """
        Sanitize text to be displayed in CSV files or visualizations.
        
        Args:
            text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        # Basic sanitization to prevent issues with CSV
        text = text.replace(",", " ").replace("\n", "\\n").replace("\r", "").replace("\t", " ")
        return text
    
    def _get_token_text(self, token_id: int) -> str:
        """
        Get text representation of a token, handling special tokens.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Text representation of the token
        """
        # Handle special tokens that would otherwise display as empty
        if token_id in self.SPECIAL_TEXT:
            return self.SPECIAL_TEXT[token_id]
        
        # Regular token decoding
        token_text = self.processor.tokenizer.decode([token_id])
        
        # Additional check for empty result (could be other special tokens)
        if not token_text.strip():
            # If empty after stripping, use a placeholder with the ID
            return f"<tok_{token_id}>"
            
        return token_text
    
    def prepare_inputs(self, image_source, prompt_text):
        """
        Prepare model inputs for analysis.
        
        Args:
            image_source: PIL image, path, or URL
            prompt_text: Text prompt
            
        Returns:
            Dictionary with prepared inputs and metadata
        """
        print("Preparing inputs...")
        # Use existing logit lens analyzer to prepare inputs
        input_data = self.logit_lens.prepare_inputs(image_source, prompt_text)
        
        # Extract input IDs and identify text vs image tokens
        input_ids = input_data["inputs"]["input_ids"]
        text_indices, image_indices = get_token_indices(input_ids, self.image_token_id)
        
        # Add indices to input data
        input_data["text_indices"] = text_indices
        input_data["image_indices"] = image_indices
        
        return input_data
    
    def renormalize_weights(
        self, 
        weights: Dict[int, float], 
        min_value: float = 1e-4  # Increased from 1e-6 for better numerical stability
    ) -> Dict[int, float]:
        """
        Safely renormalize a dictionary of weights, handling zero-sum cases.
        
        Args:
            weights: Dictionary of {index: weight} pairs
            min_value: Minimum threshold for total sum, below which equal weights are assigned
            
        Returns:
            Dictionary of normalized weights
        """
        if not weights:
            return {}
            
        total = sum(weights.values())

        max_weight = max(abs(w) for w in weights.values()) if weights else 1.0
        eps_rel = min_value * max_weight
        
        # If total is zero or very small, use uniform weights
        if total < eps_rel:
            if self.debug:
                print(f"Warning: Near-zero total weight ({total:.4e}). Using uniform weights.")
            return {idx: 1.0 / len(weights) for idx in weights}
        
        # Otherwise, normalize normally
        return {idx: weight / total for idx, weight in weights.items()}
    
    def generate_and_analyze(
        self,
        input_data: Dict[str, Any],
        target_token_idx: Optional[int] = None,
        num_tokens: int = 1,
        batch_compute: bool = True,
        tracing_mode: str = "saliency",  # Options: "saliency", "attention", "both"
    ) -> Dict[str, Any]:
        """
        Generate text and perform semantic tracing for a specific target token.
        
        Args:
            input_data: Prepared input data from prepare_inputs
            target_token_idx: Index of the token to analyze (if None, uses the first generated token)
            num_tokens: Number of tokens to generate if target_token_idx is None
            batch_compute: Whether to compute saliency in layer batches to save memory
            tracing_mode: The tracing mode to use ("saliency", "attention", or "both")
            
        Returns:
            Dictionary with analysis results
        """
        # Validate tracing mode
        valid_modes = ["saliency", "attention", "both"]
        if tracing_mode not in valid_modes:
            raise ValueError(f"Invalid tracing mode: {tracing_mode}. Must be one of {valid_modes}")
        
        # If analyzing multiple tokens, use the specialized function
        if target_token_idx is None and num_tokens > 1:
            return self.generate_and_analyze_multiple(
                input_data, num_tokens, batch_compute, tracing_mode
            )
        
        model = self.model
        device = self.device
        
        # 1. Generate text
        inputs = input_data["inputs"]
        current_input_ids = inputs["input_ids"].clone()
        original_seq_len = current_input_ids.shape[1]
        
        if target_token_idx is None:
            # Generate tokens
            print(f"Generating {num_tokens} tokens...")
            model.eval()
            with torch.no_grad():
                for _ in range(num_tokens):
                    outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                   use_cache=True)
                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                    current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
                    
                    # Update inputs for next iteration
                    inputs["input_ids"] = current_input_ids
                    if "attention_mask" in inputs:
                        inputs["attention_mask"] = torch.ones_like(current_input_ids)
            
            # Set target to the first generated token
            target_token_idx = original_seq_len
            
            # Decode generated text
            generated_tokens = current_input_ids[0, original_seq_len:].tolist()
            generated_text = self.processor.tokenizer.decode(generated_tokens)
            print(f"Generated text: '{generated_text}'")
        else:
            # Use provided target token index
            if target_token_idx >= current_input_ids.shape[1]:
                raise ValueError(f"Target token index {target_token_idx} exceeds sequence length {current_input_ids.shape[1]}")
            
            # Decode the target token for display
            token_id = current_input_ids[0, target_token_idx].item()
            token_text = self._get_token_text(token_id)
            print(f"Analyzing token at index {target_token_idx}: '{token_text}' (ID: {token_id})")
            
            # Update inputs if needed
            inputs["input_ids"] = current_input_ids
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.ones_like(current_input_ids)
        
        # Save the analyzed token information
        token_id = current_input_ids[0, target_token_idx].item()
        token_text = self._get_token_text(token_id)
        
        # Set up results dictionary
        results = {
            "input_data": input_data,
            "target_token": {
                "index": target_token_idx,
                "id": token_id,
                "text": token_text,
            },
            "full_sequence": {
                "ids": current_input_ids[0].tolist(),
                "text": self.processor.tokenizer.decode(current_input_ids[0].tolist()),
            },
            "trace_results": {},
            "metadata": {"tracing_mode": tracing_mode}  # Store tracing mode in metadata
        }
        
        # Clear the caches
        self.hidden_states_cache = {}
        self.saliency_cache = {}
        self.attention_cache = {}
        
        # Increment trace ID for this trace path
        self.trace_id_counter += 1
        
        # 2. Run appropriate tracing based on the mode
        print(f"\nStarting semantic tracing with mode '{tracing_mode}' for token '{token_text}' at position {target_token_idx}...")
        
        if tracing_mode == "saliency" or tracing_mode == "both":
            # Run saliency-based tracing
            saliency_results = self._recursive_trace(
                inputs=inputs,
                text_indices=input_data["text_indices"],
                image_indices=input_data["image_indices"],
                target_token_idx=target_token_idx,
                batch_compute=batch_compute,
                trace_id=self.trace_id_counter,
                tracing_mode="saliency",
            )
            results["trace_results"]["saliency"] = saliency_results
        
        if tracing_mode == "attention" or tracing_mode == "both":
            # Run attention-based tracing
            attention_results = self.trace_by_attention(
                inputs=inputs,
                text_indices=input_data["text_indices"],
                image_indices=input_data["image_indices"],
                target_token_idx=target_token_idx,
                trace_id=self.trace_id_counter,
            )
            results["trace_results"]["attention"] = attention_results
        
        # Store feature mapping in metadata if available - needed for visualization
        if "feature_mapping" in input_data:
            results["metadata"]["feature_mapping"] = input_data["feature_mapping"]
        
        # Save images in metadata if available - needed for visualization
        if "original_image" in input_data:
            results["metadata"]["image_available"] = True
        
        # Save metadata separately for visualization
        metadata_path = os.path.join(self.output_dir, "csv_data", f"trace_metadata.json")
        self._save_trace_metadata(results, metadata_path)
        results["metadata_path"] = metadata_path
        
        return results
    
    def _save_trace_metadata(self, results, metadata_path):
        """
        Save trace metadata to a JSON file for visualization use with complete
        feature mapping information required for offline visualization.
        
        Args:
            results: Results dictionary containing metadata
            metadata_path: Path to save metadata
        """
        # Extract metadata that can be serialized to JSON
        metadata = {
            "target_tokens": [],
            "feature_mapping": {},
            "image_available": results["metadata"].get("image_available", False),
            "tracing_mode": results["metadata"].get("tracing_mode", "saliency")
        }
        
        # Add target token information
        if "target_tokens" in results:
            # Multiple tokens case
            for token in results["target_tokens"]:
                metadata["target_tokens"].append({
                    "index": token["index"],
                    "text": token["text"],
                    "id": token["id"]
                })
        elif "target_token" in results:
            # Single token case
            metadata["target_tokens"].append({
                "index": results["target_token"]["index"],
                "text": results["target_token"]["text"],
                "id": results["target_token"]["id"]
            })
        
        # Add feature mapping if available (clean of non-serializable objects)
        if "feature_mapping" in results["metadata"]:
            feature_map = results["metadata"]["feature_mapping"]
            
            # Create a serializable version of feature mapping
            serializable_mapping = {}
            
            # Handle base feature
            if "base_feature" in feature_map:
                base_feature = feature_map["base_feature"]
                serializable_base = {
                    "grid": base_feature.get("grid", [0, 0]),
                    "positions": {str(k): v for k, v in base_feature.get("positions", {}).items()}
                }
                serializable_mapping["base_feature"] = serializable_base
            
            # Handle patch feature
            if "patch_feature" in feature_map:
                patch_feature = feature_map["patch_feature"]
                serializable_patch = {
                    "grid_unpadded": patch_feature.get("grid_unpadded", [0, 0]),
                    "positions": {str(k): v for k, v in patch_feature.get("positions", {}).items()}
                }
                
                # CRITICAL FIX: Include grid_for_visualization
                serializable_patch["grid_for_visualization"] = patch_feature.get("grid_for_visualization", [0, 0])
                
                serializable_mapping["patch_feature"] = serializable_patch
            
            # Add other serializable properties
            for key in ["patch_size", "resized_dimensions"]:
                if key in feature_map:
                    serializable_mapping[key] = feature_map[key]
            
            # CRITICAL FIX: Include padded_dimensions
            if "padded_dimensions" in feature_map:
                serializable_mapping["padded_dimensions"] = feature_map["padded_dimensions"]
            else:
                # If missing, try to infer from other dimensions
                if "original_size" in feature_map and "patch_size" in feature_map:
                    # Make best-effort guess about padded dimensions
                    w, h = feature_map.get("original_size", (0, 0))
                    patch_size = feature_map.get("patch_size", 14)
                    # Round up to multiple of patch size
                    padded_h = math.ceil(h / patch_size) * patch_size
                    padded_w = math.ceil(w / patch_size) * patch_size
                    serializable_mapping["padded_dimensions"] = (padded_w, padded_h)
                else:
                    # Use resized dimensions as fallback
                    serializable_mapping["padded_dimensions"] = serializable_mapping.get("resized_dimensions", [0, 0])
            
            metadata["feature_mapping"] = serializable_mapping
        
        # Save to file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def generate_and_analyze_multiple(
        self,
        input_data: Dict[str, Any],
        num_tokens: int = 5,
        batch_compute: bool = True,
        tracing_mode: str = "saliency",  # Options: "saliency", "attention", "both"
    ) -> Dict[str, Any]:
        """
        Generate multiple tokens and perform semantic tracing for each token.
        
        Args:
            input_data: Prepared input data from prepare_inputs
            num_tokens: Number of tokens to generate and analyze
            batch_compute: Whether to compute saliency in layer batches to save memory
            tracing_mode: The tracing mode to use ("saliency", "attention", or "both")
            
        Returns:
            Dictionary with analysis results for all generated tokens
        """
        # Validate tracing mode
        valid_modes = ["saliency", "attention", "both"]
        if tracing_mode not in valid_modes:
            raise ValueError(f"Invalid tracing mode: {tracing_mode}. Must be one of {valid_modes}")
        
        model = self.model
        device = self.device
        
        # 1. Generate text
        inputs = input_data["inputs"]
        current_input_ids = inputs["input_ids"].clone()
        original_seq_len = current_input_ids.shape[1]
        
        # Track all results by token position
        all_results = {
            "input_data": input_data,
            "target_tokens": [],
            "full_sequence": {"ids": [], "text": ""},
            "trace_results": {},
            "metadata": {"tracing_mode": tracing_mode}  # Store tracing mode in metadata
        }
        
        # Generate tokens one by one
        print(f"Generating {num_tokens} tokens...")
        model.eval()
        
        for token_idx in range(num_tokens):
            print(f"\n=== Processing token {token_idx+1}/{num_tokens} ===")
            
            with torch.no_grad():
                # Generate next token
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                               use_cache=True)
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
                
                # Update inputs for next iteration
                inputs["input_ids"] = current_input_ids
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.ones_like(current_input_ids)
            
            # Set target to the newly generated token
            target_token_idx = original_seq_len + token_idx
            
            # Decode generated text so far
            generated_tokens = current_input_ids[0, original_seq_len:].tolist()
            generated_text = self.processor.tokenizer.decode(generated_tokens)
            print(f"Generated text so far: '{generated_text}'")
            
            # Get the specific token info
            token_id = current_input_ids[0, target_token_idx].item()
            token_text = self._get_token_text(token_id)
            print(f"Analyzing token at index {target_token_idx}: '{token_text}' (ID: {token_id})")
            
            # Store token information
            all_results["target_tokens"].append({
                "index": target_token_idx,
                "id": token_id,
                "text": token_text,
            })
            
            # Clear any cached states before tracing (important for memory!)
            self.hidden_states_cache = {}
            self.saliency_cache = {}
            self.attention_cache = {}
            
            # Increment trace ID counter for this new trace path
            self.trace_id_counter += 1
            current_trace_id = self.trace_id_counter
            
            # Create a token-specific results container
            token_key = f"token_{target_token_idx}"
            all_results["trace_results"][token_key] = {}
            
            # Run appropriate tracing based on the mode
            print(f"Starting semantic tracing with mode '{tracing_mode}' for token '{token_text}' at position {target_token_idx}...")
            
            if tracing_mode == "saliency" or tracing_mode == "both":
                # Run saliency-based tracing
                saliency_results = self._recursive_trace(
                    inputs=inputs,
                    text_indices=input_data["text_indices"],
                    image_indices=input_data["image_indices"],
                    target_token_idx=target_token_idx,
                    batch_compute=batch_compute,
                    trace_id=current_trace_id,
                    tracing_mode="saliency",
                )
                all_results["trace_results"][token_key]["saliency"] = saliency_results
            
            if tracing_mode == "attention" or tracing_mode == "both":
                # Run attention-based tracing
                attention_results = self.trace_by_attention(
                    inputs=inputs,
                    text_indices=input_data["text_indices"],
                    image_indices=input_data["image_indices"],
                    target_token_idx=target_token_idx,
                    trace_id=current_trace_id,
                )
                all_results["trace_results"][token_key]["attention"] = attention_results
            
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final sequence information
        all_results["full_sequence"] = {
            "ids": current_input_ids[0].tolist(),
            "text": self.processor.tokenizer.decode(current_input_ids[0].tolist()),
        }
        
        # Save metadata separately for visualization
        metadata_path = os.path.join(self.output_dir, "csv_data", f"trace_metadata.json")
        self._save_trace_metadata(all_results, metadata_path)
        all_results["metadata_path"] = metadata_path
        
        return all_results
    
    def trace_by_attention(
        self,
        inputs: Dict[str, torch.Tensor],
        text_indices: torch.Tensor,
        image_indices: torch.Tensor,
        target_token_idx: int,
        trace_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Trace information flow using attention weights (without gradient computation).
        
        Args:
            inputs: Model inputs
            text_indices: Indices of text tokens
            image_indices: Indices of image tokens
            target_token_idx: Index of the target token
            trace_id: Unique identifier for this trace path
            
        Returns:
            Dictionary with trace results for each layer
        """
        model = self.model
        device = self.device
        
        # Dictionaries to store results
        trace_results = {}
        current_targets = {target_token_idx: 1.0}  # Initial target token with weight 1.0
        
        # Calculate token type counts for stats
        num_text_tokens = len(text_indices)
        num_image_tokens = len(image_indices)
        num_generated_tokens = inputs["input_ids"].shape[1] - (num_text_tokens + num_image_tokens)
        
        # Create token type masks for plotting/visualization
        seq_len = inputs["input_ids"].shape[1]
        token_types = torch.zeros(seq_len, dtype=torch.long, device=device)
        token_types[text_indices] = 1  # Text tokens
        token_types[image_indices] = 2  # Image tokens
        # Generated tokens remain 0
        
        # Process layers from deepest to shallowest
        all_token_ids = inputs["input_ids"][0].tolist()
        all_token_texts = [self._get_token_text(tid) for tid in all_token_ids]
        
        # Create DataFrame to record all traced tokens
        trace_records = []
        
        # If we don't have cached attention weights, get them now with a standard forward pass
        if not self.attention_cache:
            print("\nPerforming one-time forward pass to collect attention weights for all layers...")
            with torch.no_grad():
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                output_hidden_states=True,
                                output_attentions=True,
                                use_cache=False,
                                return_dict=True)
                
                self._cache_hidden_states(outputs)
                
                # Cache all attention weights
                if outputs.attentions:
                    for layer_idx, attn in enumerate(outputs.attentions):
                        self.attention_cache[layer_idx] = attn.detach().cpu() if self.cpu_offload else attn.detach()
            
            print(f"Cached hidden states for {len(self.hidden_states_cache)} layers")
            print(f"Cached attention weights for {len(self.attention_cache)} layers")
        
        # Process layers from deepest to shallowest
        for layer_idx in range(self.num_layers - 1, -1, -1):
            layer_name = self._get_layer_name(layer_idx)
            print(f"\nProcessing layer {layer_idx} ({layer_name})...")
            
            # Debug information for the current layer and targets
            if self.debug:
                print(f"[DBG][layer {layer_idx}] #cur_tgt={len(current_targets)} "
                      f"sum_tgt_weights={sum(current_targets.values()):.4e}")
            
            # 1. Get attention weights for current targets
            current_target_indices = list(current_targets.keys())
            current_target_weights = list(current_targets.values())
            
            layer_results = {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "target_tokens": [],
                "source_tokens": [],
                "logit_lens_projections": {},
            }
            
            # Get attention weights for this layer
            if layer_idx not in self.attention_cache:
                print(f"Warning: No attention weights for layer {layer_idx}. Skipping.")
                continue
            
            # Get attention weights (shape: [batch, head, seq, seq])
            attention_weights = self.attention_cache[layer_idx]
            
            # Average across heads to get a single attention map
            # Shape: [batch, seq, seq]
            attention_map = attention_weights.mean(dim=1)
            
            if attention_map.ndim == 3:
                # If batch dimension is present, take the first batch
                attention_map = attention_map[0]
            
            attn_mean = attention_map.mean().item()
            self.layer_att_means.append(attn_mean)

            # 2. For each target, find the important source tokens using coverage-based selection
            target_to_sources = {}
            
            for target_idx, target_weight in current_targets.items():
                if target_idx >= attention_map.shape[0]:
                    print(f"Warning: Target index {target_idx} exceeds attention map dimensions {attention_map.shape}. Skipping.")
                    continue
                
                # Get attention from the target token to all previous tokens (causal)
                target_vector = attention_map[target_idx, :target_idx]  # Only consider previous tokens
                
                if len(target_vector) == 0:
                    print(f"Warning: Empty target vector for token {target_idx}. Skipping.")
                    continue
                
                # Debug: Check target vector sum
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] Target {target_idx} vector sum: {target_vector.sum().item():.4e}")
                
                # Get indices of important source tokens using coverage-based selection
                # Note: For attention, we don't need to take abs() as attention weights are always non-negative
                selected_indices, selected_values = self.select_sources(
                    target_vector,
                    beta_target=self.beta_target,
                    min_keep=self.min_keep,
                    max_keep=self.max_keep
                )
                
                # Debug: Check selected values sum
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] Selected {len(selected_indices)} sources, "
                          f"sum: {selected_values.sum().item():.4e}")
                
                # Get source info and normalize weights for next iteration
                sources = []
                total_attention = selected_values.sum().item()
                
                for i, (idx, val) in enumerate(zip(selected_indices.tolist(), selected_values.tolist())):
                    # Calculate relative importance for this source
                    if self.normalize_weights and total_attention > self.epsilon:
                        relative_weight = val / total_attention
                        # Scale by the target's weight to get global importance
                        scaled_weight = relative_weight * target_weight
                    else:
                        # If total is zero or very small, use equal weights
                        relative_weight = 1.0 / len(selected_indices)
                        scaled_weight = relative_weight * target_weight
                    
                    # Get token info
                    token_id = all_token_ids[idx]
                    token_text = all_token_texts[idx]
                    token_type = token_types[idx].item()
                    
                    source_info = {
                        "index": idx,
                        "id": token_id,
                        "text": token_text,
                        "type": token_type,  # 0=generated, 1=text, 2=image
                        "attention_score": val,
                        "relative_weight": relative_weight,
                        "scaled_weight": scaled_weight,
                        "trace_id": trace_id,
                    }
                    sources.append(source_info)
                
                # Record target token info
                target_info = {
                    "index": target_idx,
                    "id": all_token_ids[target_idx],
                    "text": all_token_texts[target_idx],
                    "type": token_types[target_idx].item(),
                    "weight": target_weight,
                    "sources": sources,
                    "trace_id": trace_id,
                }
                layer_results["target_tokens"].append(target_info)
                
                # Save source indices for next iteration
                target_to_sources[target_idx] = sources
            
            # Debug: Print total source weights
            if self.debug:
                total_src_weight = sum(s["scaled_weight"] for src_list in target_to_sources.values() for s in src_list)
                print(f"[DBG][layer {layer_idx}] Total source weight before pruning: {total_src_weight:.4e}")
            
            # 3. Compute logit lens projections for all tokens involved
            all_token_indices = set()
            for target_info in layer_results["target_tokens"]:
                all_token_indices.add(target_info["index"])
                for source in target_info["sources"]:
                    all_token_indices.add(source["index"])
            
            all_token_indices = sorted(list(all_token_indices))
            if all_token_indices:
                print(f"Computing logit lens projections for {len(all_token_indices)} tokens...")
                
                # Use cached hidden states
                logit_lens_results = self._compute_logit_lens_projections(
                    inputs=inputs,
                    layer_idx=layer_idx,
                    token_indices=all_token_indices,
                )
                
                if logit_lens_results:
                    layer_results["logit_lens_projections"] = logit_lens_results
                    
                    # Add records to the trace dataframe
                    for token_idx in all_token_indices:
                        # Find if this token is a target
                        is_target = any(t["index"] == token_idx for t in layer_results["target_tokens"])
                        
                        # Find if this token is a source and for which target(s)
                        # Modified to track all targets this token is a source for
                        source_targets = []
                        for t in layer_results["target_tokens"]:
                            for s in t["sources"]:
                                if s["index"] == token_idx:
                                    source_targets.append(t["index"])
                        
                        # Get token projection data
                        token_projection = logit_lens_results.get(token_idx, {})
                        top_predictions = token_projection.get("top_predictions", [])
                        concept_predictions = token_projection.get("concept_predictions", {})
                        
                        # Extract top prediction
                        top_pred_text = ""
                        top_pred_prob = 0.0
                        if top_predictions and len(top_predictions) > 0:
                            top_pred_text = top_predictions[0].get("token_text", "")
                            top_pred_prob = top_predictions[0].get("probability", 0.0)
                        
                        # Extract target concept probabilities
                        concept_probs = {}
                        for concept, data in concept_predictions.items():
                            concept_probs[concept] = data.get("probability", 0.0)
                        
                        # Sanitize token text for CSV
                        sanitized_token_text = self._sanitize_text_for_display(all_token_texts[token_idx])
                        sanitized_pred_text = self._sanitize_text_for_display(top_pred_text)
                        
                        # Create record
                        record = {
                            "layer": layer_idx,
                            "token_index": token_idx,
                            "token_text": sanitized_token_text,
                            "token_id": all_token_ids[token_idx],
                            "token_type": token_types[token_idx].item(),
                            "is_target": is_target,
                            "source_for_targets": ",".join(map(str, source_targets)),  # Store as comma-separated list
                            "predicted_top_token": sanitized_pred_text,
                            "predicted_top_prob": top_pred_prob,
                            "trace_id": trace_id,
                        }

                        # Add importance weight for heatmap visualization
                        # For a source token, we find its scaled_weight in relation to any target
                        # If it's a source for multiple targets, we use the maximum weight
                        importance_weight = 0.0
                        for t in layer_results["target_tokens"]:
                            for s in t["sources"]:
                                if s["index"] == token_idx:
                                    # In attention mode, we use scaled_weight which combines attention_score with target weight
                                    importance_weight = max(importance_weight, s["scaled_weight"])
                        record["importance_weight"] = importance_weight
                        
                        # Add source-target relationship (needed for flow graph visualization)
                        if is_target:
                            sources_indices = []
                            sources_weights = []
                            for target in layer_results["target_tokens"]:
                                if target["index"] == token_idx:
                                    for src in target["sources"]:
                                        sources_indices.append(src["index"])
                                        sources_weights.append(src["scaled_weight"])
                            
                            # Store as comma-separated strings for CSV compatibility
                            record["sources_indices"] = ",".join(map(str, sources_indices))
                            record["sources_weights"] = ",".join(map(str, sources_weights))
                        
                        # Add concept probabilities
                        for concept, prob in concept_probs.items():
                            record[f"concept_{concept}_prob"] = prob
                            
                        trace_records.append(record)
            
            # 4. Layer-level source node pruning - OPTIMIZED to avoid duplicates
            # Aggregate sources by index to avoid duplicates
            aggregated_sources = {}
            for target_idx, sources in target_to_sources.items():
                for source in sources:
                    idx = source["index"]
                    weight = source["scaled_weight"]
                    
                    if idx in aggregated_sources:
                        # Sum weights from different targets
                        aggregated_sources[idx]["scaled_weight"] += weight
                    else:
                        # Create a copy to avoid modifying the original
                        aggregated_sources[idx] = source.copy()
            
            # Convert to list for pruning
            unique_sources_list = list(aggregated_sources.values())
            
            # Debug: Print unique vs total count
            if self.debug:
                print(f"[DBG][layer {layer_idx}] Sources before de-duplication: {sum(len(srcs) for srcs in target_to_sources.values())}")
                print(f"[DBG][layer {layer_idx}] Unique sources after de-duplication: {len(unique_sources_list)}")
            
            if unique_sources_list:
                # Apply layer-level pruning to the unique sources
                pruned_sources = self.layer_post_prune(
                    unique_sources_list,
                    beta_layer=self.beta_layer,
                    min_keep_layer=self.min_keep_layer,
                    max_keep_layer=self.max_keep_layer
                )
                
                # Debug: Print pruning results
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] Pruned {len(unique_sources_list)} -> {len(pruned_sources)} unique sources")
                
                # Create a set of remaining source indices after pruning
                remaining_source_indices = set(s["index"] for s in pruned_sources)
                
                # Update target_to_sources to only include remaining sources
                for target_idx in target_to_sources:
                    target_to_sources[target_idx] = [
                        s for s in target_to_sources[target_idx] 
                        if s["index"] in remaining_source_indices
                    ]
            
            # 5. Update current_targets for next layer
            new_targets = {}
            for target_idx, sources in target_to_sources.items():
                for source in sources:
                    source_idx = source["index"]
                    source_weight = source["scaled_weight"]
                    
                    # Multiple targets might share the same source
                    if source_idx in new_targets:
                        new_targets[source_idx] += source_weight
                    else:
                        new_targets[source_idx] = source_weight
            
            # Debug: Print new targets sum before normalization
            if self.debug and new_targets:
                print(f"[DBG][layer {layer_idx}] New targets sum before norm: {sum(new_targets.values()):.4e}")
            
            # 6. Safely normalize weights for new targets
            if new_targets:
                # Use robust normalization that handles near-zero cases
                current_targets = self.renormalize_weights(new_targets, self.epsilon)
                
                # Debug: Print after normalization
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] New targets after norm: "
                          f"count={len(current_targets)}, sum={sum(current_targets.values()):.4f}")
            else:
                print("Warning: No valid sources found for this layer. Stopping trace.")
                break
            
            # Save layer results
            trace_results[layer_idx] = layer_results
            
            # Clean up to save memory
            if self.cpu_offload and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Save trace records to CSV
        if trace_records:
            df = pd.DataFrame(trace_records)
            csv_path = os.path.join(self.output_dir, "csv_data", f"trace_attn_{trace_id}_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved attention-based trace data to {csv_path}")
            
            # Add to results
            trace_results["trace_data_path"] = csv_path
        
        print("Attention-based semantic tracing complete.")
        return trace_results
    
    def select_sources(
        self,
        target_vector: torch.Tensor,
        beta_target: float = 0.8,
        min_keep: int = 1,
        max_keep: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select source nodes based on cumulative coverage threshold with robust handling of edge cases.
        
        Args:
            target_vector: Vector of importance scores from target to sources
            beta_target: Coverage threshold (cumulative ratio)
            min_keep: Minimum number of sources to keep
            max_keep: Maximum number of sources to keep
            
        Returns:
            Tuple of (selected indices, selected values)
        """
        # For saliency values, take absolute values for importance-based sorting
        # (attention values are already non-negative, so this won't change them)
        importance = target_vector.abs()
        
        if importance.sum() < self.epsilon:
            if self.debug:
                print(f"Warning: Near-zero importance vector detected (sum={importance.sum().item():.4e}). Adding epsilon.")
            importance = importance + self.epsilon
            
        # 1. Sort values and indices by importance (descending)
        vals, idxs = torch.sort(importance, descending=True)
        
        # Get the original values (maintaining signs for saliency case)
        original_vals = target_vector[idxs]
        
        # 2. Calculate cumulative coverage based on importance
        cum = torch.cumsum(vals, 0)
        coverage = cum / cum[-1] + self.epsilon
        
        # 3. Create boolean mask for selected indices
        keep = coverage <= beta_target
        
        # 4. Apply min/max constraints
        keep[:min_keep] = True  # Always keep at least min_keep
        if keep.sum() > max_keep:
            keep[max_keep:] = False  # Keep at most max_keep
            
        if self.debug:
            print(f"Selected {keep.sum().item()} sources from {len(vals)} candidates (coverage {beta_target:.2f})")
            
            # Debug: show importance distribution
            if len(vals) > 0:
                print(f"Top importance values: {vals[:5].tolist()}")
                print(f"Importance values sum: {vals.sum().item():.4e}")
        
        # 5. Return selected indices and their ORIGINAL values (not importance-based values)
        return idxs[keep], original_vals[keep]
    
    def layer_post_prune(
        self,
        src_nodes: List[Dict],
        beta_layer: float = 0.7,
        min_keep_layer: int = 5,
        max_keep_layer: int = 100
    ) -> List[Dict]:
        """
        Second-level pruning at layer level to control overall node count.
        Ensures numerical stability and minimum node preservation.
        
        Args:
            src_nodes: List of source node dictionaries (with "scaled_weight" key)
            beta_layer: Coverage threshold for layer-level pruning
            min_keep_layer: Minimum number of nodes to keep after pruning
            max_keep_layer: Maximum number of nodes to keep after pruning
            
        Returns:
            Pruned list of source nodes
        """
        if not src_nodes:
            if self.debug:
                print("Warning: Empty source node list. Nothing to prune.")
            return []
        
        # Always ensure we keep at least min_keep_layer nodes if available
        if len(src_nodes) <= min_keep_layer:
            if self.debug:
                print(f"Layer pruning: Only {len(src_nodes)} available, less than min_keep_layer={min_keep_layer}. Keeping all.")
            return src_nodes
        
        # Get unique source indices count before pruning
        unique_src_indices = {s["index"] for s in src_nodes}
        if self.debug:
            print(f"Layer pruning: Processing {len(unique_src_indices)} unique source nodes")
        
        # 1. Sort nodes by absolute scaled_weight (decreasing)
        # Use abs() for sorting to ensure consistency with saliency tracing
        sorted_nodes = sorted(src_nodes, key=lambda x: abs(x.get('scaled_weight', 0)), reverse=True)
        
        # Extract weights as tensor for easier calculation
        weights = torch.tensor([abs(n.get('scaled_weight', 0)) for n in sorted_nodes])
        
        
        # 2. Calculate cumulative coverage
        cum_sum = torch.cumsum(weights, 0)
        coverage = cum_sum / (cum_sum[-1] + self.epsilon)
        
        # 3. Create boolean mask for keeping nodes
        keep_mask = coverage <= beta_layer
        
        # 4. Apply min/max constraints
        keep_mask[:min_keep_layer] = True  # Always keep at least min_keep_layer
        if keep_mask.sum() > max_keep_layer:
            keep_mask[max_keep_layer:] = False  # Keep at most max_keep_layer
        
        # 5. Return pruned list
        pruned = [n for k, n in zip(keep_mask, sorted_nodes) if k]
        
        if self.debug:
            print(f"Layer pruning: Kept {len(pruned)} of {len(src_nodes)} nodes (coverage {beta_layer:.2f})")
        
        return pruned
    
    def _recursive_trace(
        self,
        inputs: Dict[str, torch.Tensor],
        text_indices: torch.Tensor,
        image_indices: torch.Tensor,
        target_token_idx: int,
        batch_compute: bool = True,
        trace_id: int = 0,
        single_forward_pass: bool = False,
        tracing_mode: str = "saliency",
    ) -> Dict[str, Any]:
        """
        Recursively trace token influence backward through layers using saliency.
        
        Args:
            inputs: Model inputs
            text_indices: Indices of text tokens
            image_indices: Indices of image tokens
            target_token_idx: Index of the target token
            batch_compute: Whether to compute saliency in batches
            trace_id: Unique identifier for this trace path
            single_forward_pass: Whether to do just one forward pass for all layers
            tracing_mode: Tracing method ("saliency" or "attention")
            
        Returns:
            Dictionary with trace results for each layer
        """
        model = self.model
        device = self.device
        
        # Dictionaries to store results
        trace_results = {}
        current_targets = {target_token_idx: 1.0}  # Initial target token with weight 1.0
        
        # Calculate token type counts for stats
        num_text_tokens = len(text_indices)
        num_image_tokens = len(image_indices)
        num_generated_tokens = inputs["input_ids"].shape[1] - (num_text_tokens + num_image_tokens)
        
        # Create token type masks for plotting/visualization
        seq_len = inputs["input_ids"].shape[1]
        token_types = torch.zeros(seq_len, dtype=torch.long, device=device)
        token_types[text_indices] = 1  # Text tokens
        token_types[image_indices] = 2  # Image tokens
        # Generated tokens remain 0
        
        # Process layers from deepest to shallowest
        all_token_ids = inputs["input_ids"][0].tolist()
        all_token_texts = [self._get_token_text(tid) for tid in all_token_ids]
        
        # Create DataFrame to record all traced tokens
        trace_records = []
        
        # Single forward pass for all layers if requested
        if single_forward_pass and not self.hidden_states_cache:
            print("\nPerforming one-time forward pass to cache hidden states and attention for all layers...")
            # Register hooks for all layers at once if doing single pass
            grad_capture = GradientAttentionCapture(cpu_offload=self.cpu_offload)
            all_layer_names = self.attention_layer_names
            
            try:
                grad_capture.register_hooks(model, all_layer_names)
                
                # Compute forward and backward pass just once
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                    output_hidden_states=True,
                                    output_attentions=True,
                                    use_cache=False,
                                    return_dict=True)
                    
                    # Cache all hidden states
                    self._cache_hidden_states(outputs)
                    
                    # Cache all attention weights
                    if outputs.attentions:
                        for layer_idx, attn in enumerate(outputs.attentions):
                            self.attention_cache[layer_idx] = attn.detach().cpu() if self.cpu_offload else attn.detach()
                    
                    # Compute loss for the target token
                    logits = outputs.logits[0, target_token_idx]
                    target_token_id = inputs["input_ids"][0, target_token_idx].item()
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    loss = -log_probs[target_token_id]
                    loss.backward()
                
                # Store all saliency maps
                all_saliency_scores = grad_capture.compute_saliency()
                for layer_name, saliency_map in all_saliency_scores.items():
                    layer_idx = self._get_layer_idx(layer_name)
                    if layer_idx != -1:
                        self.saliency_cache[layer_idx] = saliency_map
                
                grad_capture.clear_hooks()
                print(f"Cached hidden states for {len(self.hidden_states_cache)} layers")
                print(f"Cached saliency maps for {len(self.saliency_cache)} layers")
                
            except Exception as e:
                print(f"Error during single forward pass: {e}")
                import traceback
                traceback.print_exc()
                if 'grad_capture' in locals():
                    grad_capture.clear_hooks()
                
                # Fall back to per-layer computation
                single_forward_pass = False
                self.hidden_states_cache = {}
                self.saliency_cache = {}
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # If we don't have cached hidden states yet and we're not doing a single forward pass, 
        # cache them now with a standard forward pass
        if not self.hidden_states_cache and not single_forward_pass:
            print("\nPerforming one-time forward pass to cache hidden states for all layers...")
            with torch.no_grad():
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                output_hidden_states=True, 
                                output_attentions=True,
                                use_cache=False,
                                return_dict=True)
                # Cache all hidden states in CPU memory
                self._cache_hidden_states(outputs)
                
                # Cache all attention weights
                if outputs.attentions:
                    for layer_idx, attn in enumerate(outputs.attentions):
                        self.attention_cache[layer_idx] = attn.detach().cpu() if self.cpu_offload else attn.detach()
            
            print(f"Cached hidden states for {len(self.hidden_states_cache)} layers")
            print(f"Cached attention weights for {len(self.attention_cache)} layers")
        
        # Process layers from deepest to shallowest
        for layer_idx in range(self.num_layers - 1, -1, -1):
            layer_name = self._get_layer_name(layer_idx)
            print(f"\nProcessing layer {layer_idx} ({layer_name})...")
            
            # Debug information for the current layer and targets
            if self.debug:
                print(f"[DBG][layer {layer_idx}] #cur_tgt={len(current_targets)} "
                      f"sum_tgt_weights={sum(current_targets.values()):.4e}")
            
            # 1. Compute saliency scores for current targets
            current_target_indices = list(current_targets.keys())
            current_target_weights = list(current_targets.values())
            
            layer_results = {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "target_tokens": [],
                "source_tokens": [],
                "logit_lens_projections": {},
            }
            
            # Use cached saliency map if available (from single forward pass)
            if single_forward_pass and layer_idx in self.saliency_cache:
                saliency_map = self.saliency_cache[layer_idx]
                # Process the cached saliency map into the format needed for the rest of the pipeline
                if saliency_map.ndim == 4:  # [batch, head, seq, seq]
                    saliency_map = saliency_map.mean(dim=(0, 1))
                elif saliency_map.ndim == 3:  # [batch, seq, seq] or [head, seq, seq]
                    saliency_map = saliency_map.mean(dim=0)
                    
                sal_norm = saliency_map.norm().item()
                self.layer_sal_norms.append(sal_norm)

                all_saliency_maps = {target_idx: saliency_map for target_idx in current_target_indices}
            else:
                # OPTIMIZATION: Process all target tokens in one backward pass
                all_saliency_maps = self._compute_saliency_for_multiple_tokens(
                    inputs=inputs,
                    layer_idx=layer_idx,
                    target_indices=current_target_indices,
                    target_weights=current_target_weights,
                    batch_compute=batch_compute
                )
            
            if not all_saliency_maps:
                print(f"Warning: No valid saliency maps computed for layer {layer_idx}. Stopping trace.")
                break
            
            # 2. For each target, find the important source tokens using coverage-based selection
            target_to_sources = {}
            
            for target_idx, target_weight in current_targets.items():
                if target_idx not in all_saliency_maps:
                    print(f"Warning: No saliency map for target {target_idx}. Skipping.")
                    continue
                
                saliency_map = all_saliency_maps[target_idx]
                target_vector = saliency_map[target_idx, :target_idx]  # Only consider previous tokens (causal)
                
                if len(target_vector) == 0:
                    print(f"Warning: Empty target vector for token {target_idx}. Skipping.")
                    continue
                
                # Debug: Check target vector sum
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] Target {target_idx} vector sum: {target_vector.sum().item():.4e}")
                    print(f"[DBG][layer {layer_idx}] Target {target_idx} vector abs sum: {target_vector.abs().sum().item():.4e}")
                
                # Get indices of important source tokens using coverage-based selection
                # select_sources will handle taking abs() for importance-based selection
                selected_indices, selected_values = self.select_sources(
                    target_vector,
                    beta_target=self.beta_target,
                    min_keep=self.min_keep,
                    max_keep=self.max_keep
                )
                
                # Debug: Check selected values sum
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] Selected {len(selected_indices)} sources, "
                          f"sum: {selected_values.sum().item():.4e}")
                    print(f"[DBG][layer {layer_idx}] Selected values abs sum: {selected_values.abs().sum().item():.4e}")
                
                # Get source info and normalize weights for next iteration
                sources = []
                # Use absolute values for importance-based calculation
                total_saliency_abs = selected_values.abs().sum().item()
                
                for i, (idx, val) in enumerate(zip(selected_indices.tolist(), selected_values.tolist())):
                    # Calculate relative importance for this source
                    if self.normalize_weights and total_saliency_abs > self.epsilon:
                        # Use absolute values for relative importance
                        relative_weight = abs(val) / total_saliency_abs
                        # Scale by the target's weight to get global importance
                        scaled_weight = relative_weight * target_weight
                    else:
                        # If total is zero or very small, use equal weights
                        relative_weight = 1.0 / len(selected_indices)
                        scaled_weight = relative_weight * target_weight
                    
                    # Get token info
                    token_id = all_token_ids[idx]
                    token_text = all_token_texts[idx]
                    token_type = token_types[idx].item()
                    
                    source_info = {
                        "index": idx,
                        "id": token_id,
                        "text": token_text,
                        "type": token_type,  # 0=generated, 1=text, 2=image
                        "saliency_score": val,
                        "relative_weight": relative_weight,
                        "scaled_weight": scaled_weight,
                        "trace_id": trace_id,
                    }
                    sources.append(source_info)
                
                # Record target token info
                target_info = {
                    "index": target_idx,
                    "id": all_token_ids[target_idx],
                    "text": all_token_texts[target_idx],
                    "type": token_types[target_idx].item(),
                    "weight": target_weight,
                    "sources": sources,
                    "trace_id": trace_id,
                }
                layer_results["target_tokens"].append(target_info)
                
                # Save source indices for next iteration
                target_to_sources[target_idx] = sources
            
            # Debug: Print total source weights
            if self.debug:
                total_src_weight = sum(s["scaled_weight"] for src_list in target_to_sources.values() for s in src_list)
                print(f"[DBG][layer {layer_idx}] Total source weight before pruning: {total_src_weight:.4e}")
            
            # 3. Compute logit lens projections for all tokens involved
            all_token_indices = set()
            for target_info in layer_results["target_tokens"]:
                all_token_indices.add(target_info["index"])
                for source in target_info["sources"]:
                    all_token_indices.add(source["index"])
            
            all_token_indices = sorted(list(all_token_indices))
            if all_token_indices:
                print(f"Computing logit lens projections for {len(all_token_indices)} tokens...")
                
                # OPTIMIZATION: Use cached hidden states
                logit_lens_results = self._compute_logit_lens_projections(
                    inputs=inputs,
                    layer_idx=layer_idx,
                    token_indices=all_token_indices,
                )
                
                if logit_lens_results:
                    layer_results["logit_lens_projections"] = logit_lens_results
                    
                    # Add records to the trace dataframe
                    for token_idx in all_token_indices:
                        # Find if this token is a target
                        is_target = any(t["index"] == token_idx for t in layer_results["target_tokens"])
                        
                        # Find if this token is a source and for which target(s)
                        # Modified to track all targets this token is a source for
                        source_targets = []
                        for t in layer_results["target_tokens"]:
                            for s in t["sources"]:
                                if s["index"] == token_idx:
                                    source_targets.append(t["index"])
                        
                        # Get token projection data
                        token_projection = logit_lens_results.get(token_idx, {})
                        top_predictions = token_projection.get("top_predictions", [])
                        concept_predictions = token_projection.get("concept_predictions", {})
                        
                        # Extract top prediction
                        top_pred_text = ""
                        top_pred_prob = 0.0
                        if top_predictions and len(top_predictions) > 0:
                            top_pred_text = top_predictions[0].get("token_text", "")
                            top_pred_prob = top_predictions[0].get("probability", 0.0)
                        
                        # Extract target concept probabilities
                        concept_probs = {}
                        for concept, data in concept_predictions.items():
                            concept_probs[concept] = data.get("probability", 0.0)
                        
                        # Sanitize token text for CSV
                        sanitized_token_text = self._sanitize_text_for_display(all_token_texts[token_idx])
                        sanitized_pred_text = self._sanitize_text_for_display(top_pred_text)
                        
                        # Create record
                        record = {
                            "layer": layer_idx,
                            "token_index": token_idx,
                            "token_text": sanitized_token_text,
                            "token_id": all_token_ids[token_idx],
                            "token_type": token_types[token_idx].item(),
                            "is_target": is_target,
                            "source_for_targets": ",".join(map(str, source_targets)),  # Store as comma-separated list
                            "predicted_top_token": sanitized_pred_text,
                            "predicted_top_prob": top_pred_prob,
                            "trace_id": trace_id,
                        }

                        importance_weight = 0.0
                        for t in layer_results["target_tokens"]:
                            for s in t["sources"]:
                                if s["index"] == token_idx:
                                    # Use scaled_weight as it combines local and global importance
                                    importance_weight = max(importance_weight, s["scaled_weight"])
                        
                        # Add importance weight to record
                        record["importance_weight"] = importance_weight
                        
                        # Add source-target relationship (needed for flow graph visualization)
                        if is_target:
                            sources_indices = []
                            sources_weights = []
                            for target in layer_results["target_tokens"]:
                                if target["index"] == token_idx:
                                    for src in target["sources"]:
                                        sources_indices.append(src["index"])
                                        sources_weights.append(src["scaled_weight"])
                            
                            # Store as comma-separated strings for CSV compatibility
                            record["sources_indices"] = ",".join(map(str, sources_indices))
                            record["sources_weights"] = ",".join(map(str, sources_weights))
                        
                        # Add concept probabilities
                        for concept, prob in concept_probs.items():
                            record[f"concept_{concept}_prob"] = prob
                            
                        trace_records.append(record)
            
            # 4. Layer-level source node pruning - OPTIMIZED to avoid duplicates
            # Aggregate sources by index to avoid duplicates
            aggregated_sources = {}
            for target_idx, sources in target_to_sources.items():
                for source in sources:
                    idx = source["index"]
                    weight = source["scaled_weight"]
                    
                    if idx in aggregated_sources:
                        # Sum weights from different targets
                        aggregated_sources[idx]["scaled_weight"] += weight
                    else:
                        # Create a copy to avoid modifying the original
                        aggregated_sources[idx] = source.copy()
            
            # Convert to list for pruning
            unique_sources_list = list(aggregated_sources.values())
            
            # Debug: Print unique vs total count
            if self.debug:
                print(f"[DBG][layer {layer_idx}] Sources before de-duplication: {sum(len(srcs) for srcs in target_to_sources.values())}")
                print(f"[DBG][layer {layer_idx}] Unique sources after de-duplication: {len(unique_sources_list)}")
            
            if unique_sources_list:
                # Apply layer-level pruning to the unique sources
                pruned_sources = self.layer_post_prune(
                    unique_sources_list,
                    beta_layer=self.beta_layer,
                    min_keep_layer=self.min_keep_layer,
                    max_keep_layer=self.max_keep_layer
                )
                
                # Debug: Print pruning results
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] Pruned {len(unique_sources_list)} -> {len(pruned_sources)} unique sources")
                
                # Create a set of remaining source indices after pruning
                remaining_source_indices = set(s["index"] for s in pruned_sources)
                
                # Update target_to_sources to only include remaining sources
                for target_idx in target_to_sources:
                    target_to_sources[target_idx] = [
                        s for s in target_to_sources[target_idx] 
                        if s["index"] in remaining_source_indices
                    ]
            
            # 5. Update current_targets for next layer
            new_targets = {}
            for target_idx, sources in target_to_sources.items():
                for source in sources:
                    source_idx = source["index"]
                    source_weight = source["scaled_weight"]
                    
                    # Multiple targets might share the same source
                    if source_idx in new_targets:
                        new_targets[source_idx] += source_weight
                    else:
                        new_targets[source_idx] = source_weight
            
            # Debug: Print new targets sum before normalization
            if self.debug and new_targets:
                print(f"[DBG][layer {layer_idx}] New targets sum before norm: {sum(new_targets.values()):.4e}")
            
            # 6. Safely normalize weights for new targets
            if new_targets:
                # Use robust normalization that handles near-zero cases
                current_targets = self.renormalize_weights(new_targets, self.epsilon)
                
                # Debug: Print after normalization
                if self.debug:
                    print(f"[DBG][layer {layer_idx}] New targets after norm: "
                          f"count={len(current_targets)}, sum={sum(current_targets.values()):.4f}")
            else:
                print("Warning: No valid sources found for this layer. Stopping trace.")
                break
            
            # Save layer results
            trace_results[layer_idx] = layer_results
            
            # Clean up to save memory
            del all_saliency_maps
            if self.cpu_offload and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # Save trace records to CSV
        if trace_records:
            df = pd.DataFrame(trace_records)
            # Correctly use tracing_mode in the filename
            csv_path = os.path.join(self.output_dir, "csv_data", f"trace_{tracing_mode}_{trace_id}_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved {tracing_mode}-based trace data to {csv_path}")
            
            # Add to results
            trace_results["trace_data_path"] = csv_path
        
        print(f"{tracing_mode.capitalize()}-based semantic tracing complete.")
        return trace_results
    
    def _compute_saliency_for_multiple_tokens(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int,
        target_indices: List[int],
        target_weights: Optional[List[float]] = None,
        batch_compute: bool = True
    ) -> Dict[int, torch.Tensor]:
        """
        Compute saliency maps for multiple target tokens in a single backward pass.
        
        Args:
            inputs: Model inputs
            layer_idx: Index of the layer
            target_indices: List of target token indices
            target_weights: Optional weights for each target token
            batch_compute: Whether to batch computations for memory efficiency
            
        Returns:
            Dictionary mapping target indices to saliency tensor
        """
        model = self.model
        layer_name = self._get_layer_name(layer_idx)
        
        if layer_name is None:
            print(f"Warning: Invalid layer index {layer_idx}.")
            return {}
            
        if not target_indices:
            print("Warning: No target indices provided.")
            return {}
            
        # Default to equal weights if not provided
        if target_weights is None:
            target_weights = [1.0 / len(target_indices)] * len(target_indices)
            
        # For small number of targets or if batch_compute is False, fall back to individual processing
        if not batch_compute or len(target_indices) == 1:
            all_saliency_scores = {}
            
            for idx, (target_idx, target_weight) in enumerate(zip(target_indices, target_weights)):
                try:
                    saliency_map = self._compute_saliency_for_token(
                        inputs=inputs,
                        layer_idx=layer_idx,
                        target_idx=target_idx
                    )
                    if saliency_map is not None:
                        # Scale by weight if needed
                        if target_weight != 1.0:
                            saliency_map = saliency_map * target_weight
                        all_saliency_scores[target_idx] = saliency_map
                except Exception as e:
                    print(f"Error processing target {target_idx}: {e}")
                
                # Clean up GPU memory after each token
                if idx % 5 == 0 and self.cpu_offload and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            return all_saliency_scores
            
        # OPTIMIZATION: Process multiple targets in a single backward pass
        try:
            # Get a fresh grad capture instance
            grad_capture = GradientAttentionCapture(cpu_offload=self.cpu_offload)
            
            # Register hook for the specific layer
            grad_capture.register_hooks(model, [layer_name])
            
            # Compute forward pass with gradients enabled
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                               output_attentions=True,
                               use_cache=False)
                
                # Build combined loss for all target tokens
                combined_loss = 0
                input_ids = inputs["input_ids"][0]
                
                # Calculate total weight sum for normalization
                total_weight = sum(target_weights)
                
                for target_idx, target_weight in zip(target_indices, target_weights):
                    if target_idx >= outputs.logits.shape[1]:
                        print(f"Warning: Target index {target_idx} exceeds logits length {outputs.logits.shape[1]}. Skipping.")
                        continue
                        
                    # Extract the target token logits
                    logits = outputs.logits[0, target_idx]
                    target_token_id = input_ids[target_idx].item()
                    
                    # Compute log probability for this token
                    log_probs = torch.log_softmax(logits.float(), dim=-1)
                    token_loss = -log_probs[target_token_id]
                    
                    # Add to combined loss with weight
                    combined_loss = combined_loss + token_loss * target_weight
                
                # Normalize the combined loss by total weight to avoid gradient scaling issues
                if total_weight > self.epsilon:
                    combined_loss = combined_loss / total_weight
                
                # Single backward pass for all targets
                if combined_loss.requires_grad:
                    combined_loss.backward()
                else:
                    print("Warning: Combined loss doesn't require gradients. Check model setup.")
                    grad_capture.clear_hooks()
                    return {}
            
            # Compute saliency scores from the combined gradients
            combined_saliency_scores = grad_capture.compute_saliency()
            grad_capture.clear_hooks()
            
            # Extract and save individual saliency maps
            all_saliency_scores = {}
            if layer_name in combined_saliency_scores:
                combined_map = combined_saliency_scores[layer_name]
                
                # Average over batch and head dimensions if needed
                if combined_map.ndim == 4:  # [batch, head, seq, seq]
                    combined_map = combined_map.mean(dim=(0, 1))
                elif combined_map.ndim == 3:  # [batch, seq, seq] or [head, seq, seq]
                    combined_map = combined_map.mean(dim=0)
                
                # Since we used a weighted sum of losses, the resulting gradients represent
                # the contribution to all targets combined. Each target token gets a copy.
                for target_idx in target_indices:
                    all_saliency_scores[target_idx] = combined_map.clone() if torch.is_tensor(combined_map) else combined_map
            else:
                print(f"Warning: No saliency scores computed for layer {layer_name}.")
            
            return all_saliency_scores
                
        except Exception as e:
            print(f"Error computing saliency for multiple tokens at layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            # Ensure hooks are cleared
            if 'grad_capture' in locals():
                grad_capture.clear_hooks()
            
            # Clean up
            if self.cpu_offload and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _compute_saliency_for_token(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int,
        target_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Compute saliency map for a specific target token at a specific layer.
        
        Args:
            inputs: Model inputs
            layer_idx: Index of the layer
            target_idx: Index of the target token
            
        Returns:
            2D tensor of saliency scores or None if computation failed
        """
        model = self.model
        layer_name = self._get_layer_name(layer_idx)
        
        if layer_name is None:
            print(f"Warning: Invalid layer index {layer_idx}.")
            return None
        
        # Get a fresh grad capture instance
        grad_capture = GradientAttentionCapture(cpu_offload=self.cpu_offload)
        
        try:
            # Register hook for the specific layer
            grad_capture.register_hooks(model, [layer_name])
            
            # Compute forward pass with gradients enabled
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                               output_attentions=True,
                               use_cache=False)
                
                # Extract the target token logits
                logits = outputs.logits[0, target_idx]
                target_token_id = inputs["input_ids"][0, target_idx].item()
                
                # Compute loss for the target token
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                loss = -log_probs[target_token_id]
                loss.backward()
            
            # Compute saliency scores
            saliency_scores = grad_capture.compute_saliency()
            grad_capture.clear_hooks()
            
            # Extract saliency map for the layer
            if layer_name in saliency_scores:
                saliency_map = saliency_scores[layer_name]
                
                # Average over batch and head dimensions if needed
                if saliency_map.ndim == 4:  # [batch, head, seq, seq]
                    saliency_map = saliency_map.mean(dim=(0, 1))
                elif saliency_map.ndim == 3:  # [batch, seq, seq] or [head, seq, seq]
                    saliency_map = saliency_map.mean(dim=0)
                
                return saliency_map
            else:
                print(f"Warning: No saliency scores computed for layer {layer_name}.")
                return None
                
        except Exception as e:
            print(f"Error computing saliency for token {target_idx} at layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Ensure hooks are cleared
            grad_capture.clear_hooks()
            
            # Clean up
            if self.cpu_offload and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _compute_logit_lens_projections(
        self,
        inputs: Dict[str, torch.Tensor],
        layer_idx: int,
        token_indices: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Compute logit lens projections for specific tokens at a specific layer.
        Uses cached hidden states to avoid repeated forward passes.
        
        Args:
            inputs: Model inputs
            layer_idx: Index of the layer
            token_indices: List of token indices to analyze
            
        Returns:
            Dictionary mapping token indices to their projections
        """
        model = self.model
        
        try:
            # OPTIMIZATION: Use cached hidden states instead of re-running the model
            if layer_idx in self.hidden_states_cache:
                print(f"Using cached hidden states for layer {layer_idx}")
                layer_hidden = self.hidden_states_cache[layer_idx]
                
                # Move to appropriate device if needed
                if layer_hidden.device != self.device:
                    layer_hidden = layer_hidden.to(self.device)
            else:
                # Fallback to computing if not cached (shouldn't happen with optimized flow)
                print(f"Warning: No cached hidden states for layer {layer_idx}. Computing on demand.")
                model.eval()
                with torch.no_grad():
                    outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                                   output_hidden_states=True,
                                   use_cache=False,
                                   return_dict=True)
                    
                    # Get hidden states for the specified layer
                    hidden_states = outputs.hidden_states
                    if layer_idx >= len(hidden_states):
                        print(f"Warning: Layer index {layer_idx} out of range for hidden states (max: {len(hidden_states)-1}).")
                        return {}
                    
                    layer_hidden = hidden_states[layer_idx]
            
            # Check if layer_hidden exists
            if 'layer_hidden' not in locals() or layer_hidden is None:
                print(f"Error: Could not obtain hidden states for layer {layer_idx}")
                return {}
                
            # Create dictionary to store results
            projections = {}
            
            # LM head from model for projecting
            lm_head = model.language_model.lm_head
            
            # Process each token
            for token_idx in token_indices:
                if token_idx >= layer_hidden.shape[1]:
                    print(f"Warning: Token index {token_idx} out of range for sequence length {layer_hidden.shape[1]}.")
                    continue
                
                # Extract hidden state for this token
                token_hidden = layer_hidden[:, token_idx:token_idx+1]
                
                # Project through LM head
                token_logits = lm_head(token_hidden).float()
                token_probs = torch.softmax(token_logits, dim=-1)
                
                # Get top-k predictions for each concept
                concept_predictions = {}
                
                for concept in self.logit_lens_concepts:
                    # Tokenize the concept
                    concept_token_ids = self.processor.tokenizer.encode(concept, add_special_tokens=False)
                    
                    if not concept_token_ids:
                        continue
                    
                    # Get probabilities for this concept's tokens
                    concept_probs = token_probs[0, 0, concept_token_ids]
                    max_prob = concept_probs.max().item()
                    
                    concept_predictions[concept] = {
                        "probability": max_prob,
                        "token_ids": concept_token_ids
                    }
                
                # Get overall top-k predictions (regardless of concepts)
                k = 5  # Number of top predictions to save
                top_probs, top_indices = torch.topk(token_probs[0, 0], k)
                
                top_predictions = []
                for i, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist())):
                    token_text = self.processor.tokenizer.decode([idx])
                    top_predictions.append({
                        "rank": i + 1,
                        "token_id": idx,
                        "token_text": token_text,
                        "probability": prob
                    })
                
                # Save results for this token
                projections[token_idx] = {
                    "top_predictions": top_predictions,
                    "concept_predictions": concept_predictions
                }
            
            return projections
        
        except Exception as e:
            print(f"Error computing logit lens projections for layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    def generate_all_then_analyze_specific(
        self,
        input_data: Dict[str, Any],
        num_tokens: int = 10,
        analyze_indices: Optional[List[int]] = None,
        tracing_mode: str = "saliency",  # Options: "saliency", "attention", "both"
        batch_compute: bool = True
    ) -> Dict[str, Any]:
        """
        First generates a complete sequence of tokens, then analyzes only specific token indices.
        
        Args:
            input_data: Prepared input data from prepare_inputs
            num_tokens: Number of tokens to generate in the complete sequence
            analyze_indices: Specific indices to analyze (None means analyze all generated tokens)
            tracing_mode: The tracing mode to use ("saliency", "attention", or "both")
            batch_compute: Whether to compute saliency in layer batches to save memory
            
        Returns:
            Dictionary with analysis results and full sequence information
        """
        model = self.model
        device = self.device
        
        # 1. Generate the complete sequence first
        print(f"\n=== Generating complete sequence of {num_tokens} tokens ===")
        inputs = input_data["inputs"]
        current_input_ids = inputs["input_ids"].clone()
        original_seq_len = current_input_ids.shape[1]
        
        model.eval()
        generated_tokens = []
        
        with torch.no_grad():
            for token_idx in range(num_tokens):
                outputs = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                            use_cache=True)
                logits = outputs.logits[:, -1, :]
                next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                current_input_ids = torch.cat([current_input_ids, next_token_id], dim=1)
                
                # Store token info
                token_id = next_token_id.item()
                token_text = self._get_token_text(token_id)
                generated_tokens.append({
                    "index": original_seq_len + token_idx,
                    "id": token_id,
                    "text": token_text
                })
                
                # Update inputs for next iteration
                inputs["input_ids"] = current_input_ids
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = torch.ones_like(current_input_ids)
                
                # Print progress
                if token_idx < 5 or token_idx % 10 == 0:
                    print(f"Generated token {token_idx+1}/{num_tokens}: '{token_text}' (ID: {token_id})")
        
        # Decode the complete sequence
        gen_ids = current_input_ids[0, original_seq_len:].tolist()
        complete_text = self.processor.tokenizer.decode(gen_ids)
        print(f"\nComplete generated text: '{complete_text}'")
        
        # 2. Determine which tokens to analyze
        if analyze_indices is None:
            # Analyze all generated tokens (original behavior)
            analyze_indices = [original_seq_len + i for i in range(len(generated_tokens))]
        else:
            # Filter to ensure indices are valid
            analyze_indices = [idx for idx in analyze_indices if idx < current_input_ids.shape[1]]
            
            # IMPORTANT FIX: Check if we're analyzing non-generated tokens (like text tokens)
            # This is critical when analyzing specific indices that might be input tokens
            for idx in analyze_indices:
                if idx < original_seq_len:
                    print(f"Note: Analyzing input token at index {idx} (not a generated token)")
        
        print(f"\n=== Analyzing {len(analyze_indices)} specific tokens ===")
        for i, idx in enumerate(analyze_indices):
            if idx >= original_seq_len and idx - original_seq_len < len(generated_tokens):
                # Generated token
                token = generated_tokens[idx - original_seq_len]
                print(f"Token {i+1}: '{token['text']}' at position {idx} (ID: {token['id']}) [Generated]")
            else:
                # Input token
                token_id = current_input_ids[0, idx].item()
                token_text = self._get_token_text(token_id)
                token_type = "Text" if idx in input_data["text_indices"] else "Image" if idx in input_data["image_indices"] else "Other"
                print(f"Token {i+1}: '{token_text}' at position {idx} (ID: {token_id}) [{token_type}]")
        
        # 3. Set up results dictionary
        all_results = {
            "input_data": input_data,  # Keep the entire input_data for reference
            "target_tokens": [],
            "all_generated_tokens": generated_tokens,
            "full_sequence": {
                "ids": current_input_ids[0].tolist(),
                "text": self.processor.tokenizer.decode(current_input_ids[0].tolist()),
                "generated_text": complete_text
            },
            "trace_results": {},
            "metadata": {
                "tracing_mode": tracing_mode
            }
        }
        
        # CRITICAL FIX: Preserve feature_mapping information from input_data
        if "feature_mapping" in input_data:
            all_results["metadata"]["feature_mapping"] = input_data["feature_mapping"]
            print("Preserved feature mapping information from input data")
        
        # Save images in metadata if available - needed for visualization
        if "original_image" in input_data:
            all_results["metadata"]["image_available"] = True
        
        # 4. Populate target_tokens with all analyzed tokens
        for target_idx in analyze_indices:
            if target_idx >= current_input_ids.shape[1]:
                print(f"Warning: Token index {target_idx} exceeds sequence length {current_input_ids.shape[1]}. Skipping.")
                continue
                
            # Get token info
            token_id = current_input_ids[0, target_idx].item()
            token_text = self._get_token_text(token_id)
            
            # Add to target_tokens
            token_info = {
                "index": target_idx,
                "id": token_id,
                "text": token_text,
            }
            all_results["target_tokens"].append(token_info)
        
        # 5. Analyze each requested token
        for target_idx in analyze_indices:
            if target_idx >= current_input_ids.shape[1]:
                continue
            
            # Get token info
            token_id = current_input_ids[0, target_idx].item()
            token_text = self._get_token_text(token_id)
            print(f"\nAnalyzing token at index {target_idx}: '{token_text}' (ID: {token_id})")
            
            # Clear caches for fresh analysis
            self.hidden_states_cache = {}
            self.saliency_cache = {}
            self.attention_cache = {}
            
            # Increment trace ID
            self.trace_id_counter += 1
            current_trace_id = self.trace_id_counter
            
            # Create token-specific results container
            token_key = f"token_{target_idx}"
            all_results["trace_results"][token_key] = {}
            
            # Run appropriate tracing based on the mode
            if tracing_mode == "saliency" or tracing_mode == "both":
                saliency_results = self._recursive_trace(
                    inputs=inputs,
                    text_indices=input_data["text_indices"],
                    image_indices=input_data["image_indices"],
                    target_token_idx=target_idx,
                    batch_compute=batch_compute,
                    trace_id=current_trace_id,
                    tracing_mode="saliency",
                )
                all_results["trace_results"][token_key]["saliency"] = saliency_results
            
            if tracing_mode == "attention" or tracing_mode == "both":
                attention_results = self.trace_by_attention(
                    inputs=inputs,
                    text_indices=input_data["text_indices"],
                    image_indices=input_data["image_indices"],
                    target_token_idx=target_idx,
                    trace_id=current_trace_id,
                )
                all_results["trace_results"][token_key]["attention"] = attention_results
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 6. Save metadata for visualization
        metadata_path = os.path.join(self.output_dir, "csv_data", f"trace_metadata.json")
        self._save_trace_metadata(all_results, metadata_path)
        all_results["metadata_path"] = metadata_path
        
        return all_results
    
    def _cache_hidden_states(self, outputs):
        for hf_idx, hidden in enumerate(outputs.hidden_states):
            if hf_idx == 0:
                continue
            layer_idx = hf_idx - 1
            self.hidden_states_cache[layer_idx] = hidden.detach().cpu() if self.cpu_offload else hidden.detach()
