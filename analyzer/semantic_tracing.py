"""
Semantic tracing for VLMs: Combines saliency analysis with logit lens to track 
information flow through model layers, revealing how concepts evolve in the
model's reasoning process. Optimized version with memory efficiency improvements.
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
    Traces information flow through VLM layers by combining saliency scores with
    logit lens projections to reveal how concepts evolve and flow from input tokens
    to generated tokens. Optimized for memory efficiency.
    """
    
    def __init__(
        self,
        model,
        processor,
        top_k: int = 3,
        device: Optional[str] = None,
        output_dir: str = "semantic_tracing_results",
        cpu_offload: bool = True,
        layer_batch_size: int = 2,
        logit_lens_concepts: Optional[List[str]] = None,
        normalize_weights: bool = True,  # New parameter for controlling normalization
        debug: bool = False,
    ):
        """
        Initialize the enhanced semantic tracer.
        
        Args:
            model: The VLM model (LLaVA-Next)
            processor: The corresponding processor
            top_k: Number of top contributing tokens to track at each step
            device: Device to use (defaults to model's device)
            output_dir: Directory to save results
            cpu_offload: Whether to offload tensors to CPU when possible
            layer_batch_size: Number of layers to process at once for gradient computation
            logit_lens_concepts: List of concepts to track with logit lens
            normalize_weights: Whether to normalize token importance weights between layers
            debug: Whether to print additional debug information
        """
        self.model = model
        self.processor = processor
        self.device = device or model.device
        self.top_k = top_k
        self.output_dir = output_dir
        self.cpu_offload = cpu_offload
        self.layer_batch_size = layer_batch_size
        self.normalize_weights = normalize_weights
        self.debug = debug
        
        # Create LogitLens analyzer from existing code
        from analyzer.logit_lens import LLaVANextLogitLensAnalyzer
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
        
        # Create a unique trace ID counter for identification
        self.trace_id_counter = 0
        
        print(f"Initialized EnhancedSemanticTracer with {self.num_layers} attention layers")
        print(f"Using top_k={top_k} for tracing")
        print(f"Normalize weights between layers: {self.normalize_weights}")
    
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
    
    def generate_and_analyze_multiple(
        self,
        input_data: Dict[str, Any],
        num_tokens: int = 5,
        batch_compute: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate multiple tokens and perform semantic tracing for each token.
        
        Args:
            input_data: Prepared input data from prepare_inputs
            num_tokens: Number of tokens to generate and analyze
            batch_compute: Whether to compute saliency in layer batches to save memory
            
        Returns:
            Dictionary with analysis results for all generated tokens
        """
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
            "metadata": {}  # Store metadata for visualization
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
            token_text = self.processor.tokenizer.decode([token_id])
            print(f"Analyzing token at index {target_token_idx}: '{token_text}' (ID: {token_id})")
            
            # Store token information
            all_results["target_tokens"].append({
                "index": target_token_idx,
                "id": token_id,
                "text": token_text,
            })
            
            # Clear any cached hidden states before tracing (important for memory!)
            self.hidden_states_cache = {}
            
            # Run semantic tracing for this token
            print(f"Starting semantic tracing for token '{token_text}' at position {target_token_idx}...")
            
            # Increment trace ID counter for this new trace path
            self.trace_id_counter += 1
            current_trace_id = self.trace_id_counter
            
            trace_results = self._recursive_trace(
                inputs=inputs,
                text_indices=input_data["text_indices"],
                image_indices=input_data["image_indices"],
                target_token_idx=target_token_idx,
                batch_compute=batch_compute,
                trace_id=current_trace_id,
            )
            
            # Store trace results for this token
            token_key = f"token_{target_token_idx}"
            all_results["trace_results"][token_key] = trace_results
            
            # Store feature mapping in metadata if available - needed for visualization
            if "feature_mapping" in input_data:
                if "feature_mapping" not in all_results["metadata"]:
                    all_results["metadata"]["feature_mapping"] = input_data["feature_mapping"]
            
            # Save images in metadata if available - needed for visualization
            if "original_image" in input_data and "original_image" not in all_results["metadata"]:
                # Reference to images, not copies (to avoid duplicating large data)
                all_results["metadata"]["image_available"] = True
            
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
    
    def generate_and_analyze(
        self,
        input_data: Dict[str, Any],
        target_token_idx: Optional[int] = None,
        num_tokens: int = 1,
        batch_compute: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate text and perform semantic tracing for a specific target token.
        
        Args:
            input_data: Prepared input data from prepare_inputs
            target_token_idx: Index of the token to analyze (if None, uses the first generated token)
            num_tokens: Number of tokens to generate if target_token_idx is None
            batch_compute: Whether to compute saliency in layer batches to save memory
            
        Returns:
            Dictionary with analysis results
        """
        # If analyzing multiple tokens, use the specialized function
        if target_token_idx is None and num_tokens > 1:
            return self.generate_and_analyze_multiple(input_data, num_tokens, batch_compute)
        
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
            token_text = self.processor.tokenizer.decode([token_id])
            print(f"Analyzing token at index {target_token_idx}: '{token_text}' (ID: {token_id})")
            
            # Update inputs if needed
            inputs["input_ids"] = current_input_ids
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.ones_like(current_input_ids)
        
        # Save the analyzed token information
        token_id = current_input_ids[0, target_token_idx].item()
        token_text = self.processor.tokenizer.decode([token_id])
        
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
            "metadata": {}  # Store metadata for visualization
        }
        
        # Clear the hidden states cache
        self.hidden_states_cache = {}
        
        # Increment trace ID for this trace path
        self.trace_id_counter += 1
        
        # 2. Start recursive tracing
        print(f"\nStarting recursive semantic tracing for token '{token_text}' at position {target_token_idx}...")
        trace_results = self._recursive_trace(
            inputs=inputs,
            text_indices=input_data["text_indices"],
            image_indices=input_data["image_indices"],
            target_token_idx=target_token_idx,
            batch_compute=batch_compute,
            trace_id=self.trace_id_counter,
        )
        
        results["trace_results"] = trace_results
        
        # Store feature mapping in metadata if available - needed for visualization
        if "feature_mapping" in input_data:
            results["metadata"]["feature_mapping"] = input_data["feature_mapping"]
        
        # Save images in metadata if available - needed for visualization
        if "original_image" in input_data:
            # Just mark that image is available, don't duplicate in memory
            results["metadata"]["image_available"] = True
        
        # Save metadata separately for visualization
        metadata_path = os.path.join(self.output_dir, "csv_data", f"trace_metadata.json")
        self._save_trace_metadata(results, metadata_path)
        results["metadata_path"] = metadata_path
        
        return results
    
    def _save_trace_metadata(self, results, metadata_path):
        """
        Save trace metadata to a JSON file for visualization use.
        
        Args:
            results: Results dictionary containing metadata
            metadata_path: Path to save metadata
        """
        # Extract metadata that can be serialized to JSON
        metadata = {
            "target_tokens": [],
            "feature_mapping": {},
            "image_available": results["metadata"].get("image_available", False),
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
                serializable_mapping["patch_feature"] = serializable_patch
            
            # Add other serializable properties
            for key in ["patch_size", "resized_dimensions"]:
                if key in feature_map:
                    serializable_mapping[key] = feature_map[key]
            
            metadata["feature_mapping"] = serializable_mapping
        
        # Save to file
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def analyze_last_token(
        self,
        input_data: Dict[str, Any],
        single_forward_pass: bool = True,
        batch_compute: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform a single forward pass and analyze only the last token in the sequence.
        
        Args:
            input_data: Prepared input data from prepare_inputs
            single_forward_pass: Whether to use one-time forward pass optimization
            batch_compute: Whether to compute saliency in batches for memory efficiency
            
        Returns:
            Dictionary with analysis results for the last token
        """
        model = self.model
        device = self.device
        
        # Get model inputs
        inputs = input_data["inputs"]
        current_input_ids = inputs["input_ids"].clone()
        
        # Target is the last token in the sequence
        target_token_idx = current_input_ids.shape[1] - 1
        
        # Decode the target token for display
        token_id = current_input_ids[0, target_token_idx].item()
        token_text = self.processor.tokenizer.decode([token_id])
        print(f"Analyzing last token at index {target_token_idx}: '{token_text}' (ID: {token_id})")
        
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
            "metadata": {}  # Store metadata for visualization
        }
        
        # Clear the hidden states cache
        self.hidden_states_cache = {}
        
        # Increment trace ID for this trace path
        self.trace_id_counter += 1
        
        # Start recursive tracing with single forward pass option
        print(f"\nStarting recursive semantic tracing for token '{token_text}' at position {target_token_idx}...")
        trace_results = self._recursive_trace(
            inputs=inputs,
            text_indices=input_data["text_indices"],
            image_indices=input_data["image_indices"],
            target_token_idx=target_token_idx,
            batch_compute=batch_compute,
            trace_id=self.trace_id_counter,
            single_forward_pass=single_forward_pass,
        )
        
        results["trace_results"] = trace_results
        
        # Store feature mapping in metadata if available - needed for visualization
        if "feature_mapping" in input_data:
            results["metadata"]["feature_mapping"] = input_data["feature_mapping"]
        
        # Save images in metadata if available - needed for visualization
        if "original_image" in input_data:
            # Just mark that image is available, don't duplicate in memory
            results["metadata"]["image_available"] = True
        
        # Save metadata separately for visualization
        metadata_path = os.path.join(self.output_dir, "csv_data", f"trace_metadata.json")
        self._save_trace_metadata(results, metadata_path)
        results["metadata_path"] = metadata_path
        
        return results
    
    def _recursive_trace(
        self,
        inputs: Dict[str, torch.Tensor],
        text_indices: torch.Tensor,
        image_indices: torch.Tensor,
        target_token_idx: int,
        batch_compute: bool = True,
        trace_id: int = 0,
        single_forward_pass: bool = False,
    ) -> Dict[str, Any]:
        """
        Recursively trace token influence backward through layers.
        
        Args:
            inputs: Model inputs
            text_indices: Indices of text tokens
            image_indices: Indices of image tokens
            target_token_idx: Index of the target token
            batch_compute: Whether to compute saliency in batches
            trace_id: Unique identifier for this trace path
            single_forward_pass: Whether to do just one forward pass for all layers
            
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
        all_token_texts = [self.processor.tokenizer.decode([tid]) for tid in all_token_ids]
        
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
                    for layer_idx, hidden in enumerate(outputs.hidden_states):
                        self.hidden_states_cache[layer_idx] = hidden.detach().cpu() if self.cpu_offload else hidden.detach()
                    
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
                                use_cache=False,
                                return_dict=True)
                # Cache all hidden states in CPU memory
                for layer_idx, hidden in enumerate(outputs.hidden_states):
                    self.hidden_states_cache[layer_idx] = hidden.detach().cpu() if self.cpu_offload else hidden.detach()
            print(f"Cached hidden states for {len(self.hidden_states_cache)} layers")
        
        # Add saliency cache for the single forward pass case
        if not hasattr(self, 'saliency_cache'):
            self.saliency_cache = {}
        
        # Process layers from deepest to shallowest
        for layer_idx in range(self.num_layers - 1, -1, -1):
            layer_name = self._get_layer_name(layer_idx)
            print(f"\nProcessing layer {layer_idx} ({layer_name})...")
            
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
            
            # 2. For each target, find the top-k source tokens
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
                
                # Get indices of top-k sources
                if len(target_vector) <= self.top_k:
                    # If we have fewer tokens than top_k, use all of them
                    topk_indices = torch.arange(len(target_vector), device=target_vector.device)
                    topk_values = target_vector
                else:
                    # Otherwise find top-k
                    topk_values, topk_indices = torch.topk(target_vector, self.top_k)
                
                # Get source info and normalize weights for next iteration
                sources = []
                total_saliency = topk_values.sum().item()
                
                for i, (idx, val) in enumerate(zip(topk_indices.tolist(), topk_values.tolist())):
                    # Calculate relative importance for this source
                    # NOTE: Normalization is done to enable comparing tokens in the next layer,
                    # not for physical interpretation of values
                    if self.normalize_weights and total_saliency > 0:
                        relative_weight = val / total_saliency
                        # Scale by the target's weight to get global importance
                        scaled_weight = relative_weight * target_weight
                    else:
                        # If not normalizing, just use raw values while preserving the sum
                        relative_weight = val
                        scaled_weight = val * target_weight
                    
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
                        
                        # Create record
                        record = {
                            "layer": layer_idx,
                            "token_index": token_idx,
                            "token_text": all_token_texts[token_idx],
                            "token_id": all_token_ids[token_idx],
                            "token_type": token_types[token_idx].item(),
                            "is_target": is_target,
                            "source_for_targets": source_targets,
                            "predicted_top_token": top_pred_text,
                            "predicted_top_prob": top_pred_prob,
                            "trace_id": trace_id,
                        }
                        
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
            
            # 4. Update current_targets for next layer
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
            
            # Normalize weights for new targets if needed
            if new_targets:
                if self.normalize_weights:
                    total_weight = sum(new_targets.values())
                    current_targets = {idx: weight / total_weight for idx, weight in new_targets.items()} if total_weight > 0 else new_targets
                else:
                    current_targets = new_targets
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
            csv_path = os.path.join(self.output_dir, "csv_data", f"trace_{trace_id}_data.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved trace data to {csv_path}")
            
            # Add to results
            trace_results["trace_data_path"] = csv_path
        
        print("Semantic tracing complete.")
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