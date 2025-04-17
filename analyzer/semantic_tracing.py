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
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import ImageFont
from PIL import ImageDraw
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
    to generated tokens. Optimized for memory efficiency and with improved visualization.
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
        os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
        
        # Track hidden states cache to avoid repeated forward passes
        self.hidden_states_cache = {}
        
        # Create a unique trace ID counter for visualization
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
            
            # Force garbage collection to free memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final sequence information
        all_results["full_sequence"] = {
            "ids": current_input_ids[0].tolist(),
            "text": self.processor.tokenizer.decode(current_input_ids[0].tolist()),
        }
        
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
        return results
    
    def _recursive_trace(
        self,
        inputs: Dict[str, torch.Tensor],
        text_indices: torch.Tensor,
        image_indices: torch.Tensor,
        target_token_idx: int,
        batch_compute: bool = True,
        trace_id: int = 0,
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
        
        # Precompute logit lens for all tokens and cache (one-time pass)
        if not self.hidden_states_cache:
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
    
    def visualize_trace(self, results: Dict[str, Any], save_dir: Optional[str] = None) -> List[str]:
        """
        Visualize the semantic tracing results with enhanced outputs.
        
        Args:
            results: Results from generate_and_analyze
            save_dir: Directory to save visualizations (defaults to self.output_dir)
            
        Returns:
            List of saved file paths
        """
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "visualizations")
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        # Check if we have multi-token results
        is_multi_token = "target_tokens" in results and isinstance(results["target_tokens"], list)
        
        if is_multi_token:
            # Process multi-token results
            for i, target_token in enumerate(results["target_tokens"]):
                token_idx = target_token["index"]
                token_text = target_token["text"]
                token_key = f"token_{token_idx}"
                
                if token_key in results["trace_results"]:
                    token_dir = os.path.join(save_dir, f"token_{token_idx}_{token_text}")
                    os.makedirs(token_dir, exist_ok=True)
                    
                    trace_results = results["trace_results"][token_key]
                    print(f"\nVisualizing trace for token {i+1}: '{token_text}' at position {token_idx}")
                    
                    # Visualize this token's trace
                    token_paths = self._visualize_single_trace(trace_results, token_text, token_idx, token_dir)
                    saved_paths.extend(token_paths)
            
            # Create combined visualization across all tokens
            combined_dir = os.path.join(save_dir, "combined")
            os.makedirs(combined_dir, exist_ok=True)
            combined_paths = self._visualize_combined_traces(results, combined_dir)
            saved_paths.extend(combined_paths)
            
        else:
            # Single token results (traditional format)
            trace_results = results.get("trace_results", {})
            if not trace_results:
                print("No trace results to visualize.")
                return saved_paths
            
            # Basic info for labeling
            target_token = results.get("target_token", {})
            target_text = target_token.get("text", "unknown")
            target_idx = target_token.get("index", -1)
            
            # Visualize the single trace
            token_paths = self._visualize_single_trace(trace_results, target_text, target_idx, save_dir)
            saved_paths.extend(token_paths)
        
        print(f"Total visualization files created: {len(saved_paths)}")
        return saved_paths
    
    def _visualize_single_trace(
        self,
        trace_results: Dict[int, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str
    ) -> List[str]:
        """Helper to visualize a single token's trace results"""
        saved_paths = []
        
        # 1. Visualize the trace as a stacked bar chart
        try:
            paths = self._visualize_source_distribution(trace_results, target_text, target_idx, save_dir)
            saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing source distribution: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Visualize the semantic evolution using logit lens
        try:
            paths = self._visualize_semantic_evolution(trace_results, target_text, target_idx, save_dir)
            saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing semantic evolution: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Visualize image token heatmaps if available
        try:
            paths = self._visualize_image_token_heatmaps(trace_results, target_text, target_idx, save_dir)
            saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing image token heatmaps: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Create a text report
        try:
            report_path = self._create_text_report(
                {"trace_results": trace_results, "target_token": {"text": target_text, "index": target_idx}}, 
                os.path.join(save_dir, f"semantic_trace_report_{target_idx}.txt")
            )
            saved_paths.append(report_path)
        except Exception as e:
            print(f"Error creating text report: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. Create trace data visualization from CSV if available
        try:
            if "trace_data_path" in trace_results and os.path.exists(trace_results["trace_data_path"]):
                paths = self._visualize_trace_data(trace_results["trace_data_path"], target_text, target_idx, save_dir)
                saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing trace data: {e}")
            import traceback
            traceback.print_exc()
        
        return saved_paths
    
    def _visualize_combined_traces(self, results: Dict[str, Any], save_dir: str) -> List[str]:
        """Create visualizations combining data from multiple token traces"""
        saved_paths = []
        
        # 1. Create combined concept evolution plot
        all_traces = []
        all_target_texts = []
        
        for target_token in results.get("target_tokens", []):
            token_idx = target_token["index"]
            token_text = target_token["text"]
            token_key = f"token_{token_idx}"
            
            if token_key in results["trace_results"]:
                all_traces.append(results["trace_results"][token_key])
                all_target_texts.append(token_text)
        
        if all_traces:
            try:
                # Combined concept evolution
                paths = self._visualize_multi_token_concept_evolution(all_traces, all_target_texts, save_dir)
                saved_paths.extend(paths)
                
                # Combined token type distribution
                paths = self._visualize_multi_token_type_distribution(all_traces, all_target_texts, save_dir)
                saved_paths.extend(paths)
            except Exception as e:
                print(f"Error visualizing combined traces: {e}")
                import traceback
                traceback.print_exc()
        
        return saved_paths
    
    def _visualize_source_distribution(
        self,
        trace_results: Dict[int, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str,
    ) -> List[str]:
        """Visualize the distribution of source token types (text, image, generated) across layers"""
        saved_paths = []
        
        # Convert keys to integers if they're strings
        if trace_results and all(isinstance(key, str) for key in trace_results.keys()):
            trace_results = {int(k): v for k, v in trace_results.items()}
        
        layers = sorted(trace_results.keys())
        
        if not layers:
            print("Warning: No layers found in trace results.")
            return saved_paths
        
        # Prepare data for plotting
        text_contributions = []
        image_contributions = []
        generated_contributions = []
        
        for layer_idx in layers:
            layer_data = trace_results[layer_idx]
            
            # Combine all sources across all targets for this layer
            all_sources = []
            for target in layer_data["target_tokens"]:
                all_sources.extend(target["sources"])
            
            # Calculate total contribution by type
            text_contrib = sum(src["scaled_weight"] for src in all_sources if src["type"] == 1)
            image_contrib = sum(src["scaled_weight"] for src in all_sources if src["type"] == 2)
            gen_contrib = sum(src["scaled_weight"] for src in all_sources if src["type"] == 0)
            
            # Normalize to sum to 1.0
            total = text_contrib + image_contrib + gen_contrib
            if total > 0:
                text_contrib /= total
                image_contrib /= total
                gen_contrib /= total
            
            text_contributions.append(text_contrib)
            image_contributions.append(image_contrib)
            generated_contributions.append(gen_contrib)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bar_width = 0.8
        x = layers
        
        ax.bar(x, text_contributions, bar_width, label='Text Tokens', color='#3498db')
        ax.bar(x, image_contributions, bar_width, bottom=text_contributions, label='Image Tokens', color='#e74c3c')
        
        bottom = [t + i for t, i in zip(text_contributions, image_contributions)]
        ax.bar(x, generated_contributions, bar_width, bottom=bottom, label='Generated Tokens', color='#2ecc71')
        
        ax.set_ylabel('Contribution Proportion')
        ax.set_xlabel('Layer')
        ax.set_title(f'Source Token Type Distribution for Target "{target_text}" (idx: {target_idx})')
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in x], rotation=90)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f"token_type_distribution_{target_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        saved_paths.append(save_path)
        
        plt.close(fig)
        
        # Create line plot version (sometimes more readable)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(x, text_contributions, marker='o', linewidth=2, label='Text Tokens', color='#3498db')
        ax.plot(x, image_contributions, marker='s', linewidth=2, label='Image Tokens', color='#e74c3c')
        ax.plot(x, generated_contributions, marker='^', linewidth=2, label='Generated Tokens', color='#2ecc71')
        
        ax.set_ylabel('Contribution Proportion')
        ax.set_xlabel('Layer')
        ax.set_title(f'Source Token Type Contribution for Target "{target_text}" (idx: {target_idx})')
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in x], rotation=90)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f"token_type_line_plot_{target_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        saved_paths.append(save_path)
        
        plt.close(fig)
        
        return saved_paths
    
    def _visualize_semantic_evolution(
        self,
        trace_results: Dict[int, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str
    ) -> List[str]:
        """Visualize how concepts evolve across layers using logit lens projections"""
        saved_paths = []
        
        # Convert keys to integers if they're strings
        if trace_results and all(isinstance(key, str) for key in trace_results.keys()):
            trace_results = {int(k): v for k, v in trace_results.items()}
            
        layers = sorted(trace_results.keys())
        
        # Extract concept probabilities for the top sources at each layer
        concept_data = {concept: [] for concept in self.logit_lens_concepts}
        
        for layer_idx in layers:
            layer_data = trace_results[layer_idx]
            projections = layer_data.get("logit_lens_projections", {})
            
            # Average probabilities for each concept across all top sources in this layer
            if projections:
                for concept in self.logit_lens_concepts:
                    concept_probs = []
                    
                    # Check all tokens with projections
                    for token_idx, proj_data in projections.items():
                        # Handle token_idx as string if needed
                        if isinstance(token_idx, str):
                            token_idx = int(token_idx)
                            
                        concept_pred = proj_data.get("concept_predictions", {}).get(concept)
                        if concept_pred:
                            concept_probs.append(concept_pred["probability"])
                    
                    # Average if we have values
                    if concept_probs:
                        avg_prob = sum(concept_probs) / len(concept_probs)
                    else:
                        avg_prob = 0.0
                        
                    concept_data[concept].append(avg_prob)
            else:
                # No projections for this layer, use zeros
                for concept in self.logit_lens_concepts:
                    concept_data[concept].append(0.0)
        
        # Create line chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for concept, probs in concept_data.items():
            ax.plot(layers, probs, marker='o', linewidth=2, label=concept)
        
        ax.set_ylabel('Average Concept Probability')
        ax.set_xlabel('Layer')
        ax.set_title(f'Concept Evolution Across Layers for Target "{target_text}" (idx: {target_idx})')
        ax.set_xticks(layers)
        ax.set_xticklabels([str(l) for l in layers], rotation=90)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(save_dir, f"concept_evolution_{target_idx}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        saved_paths.append(save_path)
        
        plt.close(fig)
        
        return saved_paths
    
    def _visualize_image_token_heatmaps(
        self,
        trace_results: Dict[int, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str
    ) -> List[str]:
        """
        Visualize heatmaps of image token influence across layers by mapping tokens
        back to their spatial positions in the original image.
        """
        saved_paths = []
        
        # Convert keys to integers if they're strings
        if trace_results and all(isinstance(key, str) for key in trace_results.keys()):
            trace_results = {int(k): v for k, v in trace_results.items()}
            
        layers = sorted(trace_results.keys())
        
        # Get the input data with feature mapping information
        input_data = trace_results.get("input_data", {})
        if not input_data:
            print("Error: No input_data found in trace_results. Cannot visualize image token heatmaps.")
            return saved_paths
        
        # Extract required data for visualization
        feature_mapping = input_data.get("feature_mapping", {})
        if not feature_mapping:
            print("Error: No feature_mapping found in input_data. Cannot visualize image token heatmaps.")
            return saved_paths
        
        # Get original image and preview image for overlays
        original_image = input_data.get("original_image")
        spatial_preview_image = input_data.get("spatial_preview_image")
        
        if not original_image or not spatial_preview_image:
            print("Error: Missing original_image or spatial_preview_image. Cannot visualize.")
            return saved_paths
        
        # Extract base and patch feature information
        base_feature_info = feature_mapping.get("base_feature", {})
        patch_feature_info = feature_mapping.get("patch_feature", {})
        
        if not base_feature_info or not patch_feature_info:
            print("Warning: Missing feature mapping information. Visualizations may be incomplete.")
        
        # Prepare output directories
        base_heatmap_dir = os.path.join(save_dir, "base_feature_heatmaps")
        patch_heatmap_dir = os.path.join(save_dir, "patch_feature_heatmaps")
        os.makedirs(base_heatmap_dir, exist_ok=True)
        os.makedirs(patch_heatmap_dir, exist_ok=True)
        
        # Process each layer
        for layer_idx in layers:
            layer_data = trace_results[layer_idx]
            
            # Process all targets in this layer
            targets = layer_data.get("target_tokens", [])
            if not targets:
                continue
            
            # Collect all image token sources across all targets for this layer
            image_token_weights = {}  # {token_idx: weight}
            
            for target in targets:
                for source in target.get("sources", []):
                    if source.get("type") == 2:  # Image token
                        source_idx = source.get("index")
                        weight = source.get("scaled_weight", 0.0)
                        
                        # Combine weights if multiple targets use the same source
                        if source_idx in image_token_weights:
                            image_token_weights[source_idx] += weight
                        else:
                            image_token_weights[source_idx] = weight
            
            if not image_token_weights:
                continue  # No image tokens in this layer
            
            # Normalize weights for visualization
            max_weight = max(image_token_weights.values())
            if max_weight > 0:
                normalized_weights = {idx: weight / max_weight for idx, weight in image_token_weights.items()}
            else:
                normalized_weights = image_token_weights
            
            # 1. Visualize base features
            if base_feature_info.get("positions") and base_feature_info.get("grid"):
                base_grid_h, base_grid_w = base_feature_info["grid"]
                
                # Initialize empty heatmap
                base_heatmap = np.zeros((base_grid_h, base_grid_w), dtype=np.float32)
                
                # Fill the heatmap with normalized weights
                for token_idx, weight in normalized_weights.items():
                    # Convert token_idx to int if it's a string
                    if isinstance(token_idx, str):
                        token_idx = int(token_idx)
                        
                    position = base_feature_info["positions"].get(token_idx)
                    if position:
                        r, c = position
                        if 0 <= r < base_grid_h and 0 <= c < base_grid_w:
                            base_heatmap[r, c] = weight
                
                # Create visualization if we have data
                if np.max(base_heatmap) > 0:
                    base_path = self._create_base_feature_overlay(
                        heatmap=base_heatmap,
                        original_image=original_image,
                        grid_size=(base_grid_h, base_grid_w),
                        layer_idx=layer_idx,
                        target_idx=target_idx,
                        title=f"Base Image Token Influence - Layer {layer_idx}",
                        save_path=os.path.join(base_heatmap_dir, f"base_influence_layer_{layer_idx}_{target_idx}.png")
                    )
                    if base_path:
                        saved_paths.append(base_path)
            
            # 2. Visualize patch features
            if patch_feature_info.get("positions") and patch_feature_info.get("grid_unpadded"):
                prob_grid_h, prob_grid_w = patch_feature_info["grid_unpadded"]
                
                # Initialize empty heatmap
                patch_heatmap = np.zeros((prob_grid_h, prob_grid_w), dtype=np.float32)
                
                # Fill the heatmap with normalized weights
                for token_idx, weight in normalized_weights.items():
                    # Convert token_idx to int if it's a string
                    if isinstance(token_idx, str):
                        token_idx = int(token_idx)
                        
                    position = patch_feature_info["positions"].get(token_idx)
                    if position:
                        r, c = position
                        if 0 <= r < prob_grid_h and 0 <= c < prob_grid_w:
                            patch_heatmap[r, c] = weight
                
                # Create visualization if we have data
                if np.max(patch_heatmap) > 0:
                    # Get required dimensions
                    resized_dims_wh = feature_mapping.get("resized_dimensions", (0, 0))
                    patch_size = feature_mapping.get("patch_size", 14)  # Default to 14 if not specified
                    
                    patch_path = self._create_patch_feature_overlay(
                        heatmap=patch_heatmap,
                        spatial_preview_image=spatial_preview_image,
                        feature_mapping=feature_mapping,
                        patch_size=patch_size,
                        layer_idx=layer_idx,
                        target_idx=target_idx,
                        title=f"Patch Image Token Influence - Layer {layer_idx}",
                        save_path=os.path.join(patch_heatmap_dir, f"patch_influence_layer_{layer_idx}_{target_idx}.png")
                    )
                    if patch_path:
                        saved_paths.append(patch_path)
        
        # Create composite visualizations if we have multiple layers
        if len(saved_paths) > 0:
            # Create composite for base feature heatmaps
            base_paths = [p for p in saved_paths if "base_influence" in p]
            if base_paths:
                try:
                    base_layers = [int(os.path.basename(p).split('_')[2]) for p in base_paths]
                    base_composite_path = os.path.join(save_dir, f"composite_base_influence_{target_idx}.png")
                    composite_path = self._create_composite_image(
                        image_paths=base_paths,
                        layers=base_layers,
                        output_filename=base_composite_path,
                        title=f"Base Image Token Influence Across Layers for Target '{target_text}'"
                    )
                    if composite_path:
                        saved_paths.append(composite_path)
                except Exception as e:
                    print(f"Error creating base composite: {e}")
            
            # Create composite for patch feature heatmaps
            patch_paths = [p for p in saved_paths if "patch_influence" in p]
            if patch_paths:
                try:
                    patch_layers = [int(os.path.basename(p).split('_')[2]) for p in patch_paths]
                    patch_composite_path = os.path.join(save_dir, f"composite_patch_influence_{target_idx}.png")
                    composite_path = self._create_composite_image(
                        image_paths=patch_paths,
                        layers=patch_layers,
                        output_filename=patch_composite_path,
                        title=f"Patch Image Token Influence Across Layers for Target '{target_text}'"
                    )
                    if composite_path:
                        saved_paths.append(composite_path)
                except Exception as e:
                    print(f"Error creating patch composite: {e}")
        
        return saved_paths

    def _create_base_feature_overlay(
        self,
        heatmap: np.ndarray,
        original_image: Image.Image,
        grid_size: Tuple[int, int],
        layer_idx: int,
        target_idx: int,
        title: str,
        save_path: str,
        target_size: Tuple[int, int] = (336, 336),
        colormap: str = "hot",
        alpha: float = 0.7,
        add_gridlines: bool = True
    ) -> Optional[str]:
        """
        Create a heatmap overlay visualization for base image features.
        
        Args:
            heatmap: 2D numpy array with heatmap values
            original_image: Original PIL image
            grid_size: Tuple of (height, width) for the grid
            layer_idx: Index of the layer
            target_idx: Index of the target token
            title: Title for the plot
            save_path: Path to save the visualization
            target_size: Target size for the visualization
            colormap: Matplotlib colormap name
            alpha: Alpha blending value for overlay
            add_gridlines: Whether to add grid lines
            
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            # Resize original image for overlay
            resized_background = original_image.resize(target_size, Image.Resampling.LANCZOS)
            background_np = np.array(resized_background)
            
            # Upscale heatmap to match image dimensions
            grid_h, grid_w = grid_size
            
            # Use scikit-image if available, otherwise fall back to simple method
            try:
                from skimage.transform import resize as skimage_resize
                upscaled_heatmap = skimage_resize(
                    heatmap, target_size, order=1, mode='constant', 
                    cval=0, anti_aliasing=True, preserve_range=True
                )
            except ImportError:
                # Simple upscaling method
                repeat_y = target_size[1] // grid_h
                repeat_x = target_size[0] // grid_w
                upscaled_heatmap = np.kron(heatmap, np.ones((repeat_y, repeat_x)))
                upscaled_heatmap = upscaled_heatmap[:target_size[1], :target_size[0]]
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot background image
            ax.imshow(background_np, extent=(0, target_size[0], target_size[1], 0))
            
            # Plot heatmap overlay
            im = ax.imshow(
                upscaled_heatmap, 
                alpha=alpha,
                cmap=colormap,
                vmin=0,
                vmax=1,
                extent=(0, target_size[0], target_size[1], 0),
                interpolation="nearest"
            )
            
            # Add grid lines if requested
            if add_gridlines:
                cell_height = target_size[1] / grid_h
                cell_width = target_size[0] / grid_w
                
                # Horizontal grid lines
                for i in range(1, grid_h):
                    y = i * cell_height
                    ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Vertical grid lines
                for i in range(1, grid_w):
                    x = i * cell_width
                    ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Influence Weight")
            
            # Set title and remove axes
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            return save_path
        
        except Exception as e:
            print(f"Error creating base feature overlay: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_patch_feature_overlay(
        self,
        heatmap: np.ndarray,
        spatial_preview_image: Image.Image,
        feature_mapping: Dict[str, Any],
        patch_size: int,
        layer_idx: int,
        target_idx: int,
        title: str,
        save_path: str,
        colormap: str = "hot",
        alpha: float = 0.7,
        add_gridlines: bool = True
    ) -> Optional[str]:
        """
        Create a heatmap overlay visualization for patch image features.
        
        Args:
            heatmap: 2D numpy array with heatmap values
            spatial_preview_image: Preprocessed spatial image
            feature_mapping: Feature mapping dictionary
            patch_size: Raw patch size for the vision model
            layer_idx: Index of the layer
            target_idx: Index of the target token
            title: Title for the plot
            save_path: Path to save the visualization
            colormap: Matplotlib colormap name
            alpha: Alpha blending value for overlay
            add_gridlines: Whether to add grid lines
            
        Returns:
            Path to saved visualization or None if failed
        """
        try:
            # Get patch feature information
            patch_feature_info = feature_mapping.get("patch_feature", {})
            if not patch_feature_info:
                print("Error: No patch feature information available.")
                return None
            
            prob_grid_h, prob_grid_w = patch_feature_info.get("grid_unpadded", (0, 0))
            if prob_grid_h == 0 or prob_grid_w == 0:
                print("Error: Invalid grid dimensions.")
                return None
            
            # Get dimensions for the visualization
            preview_w, preview_h = spatial_preview_image.size
            background_np = np.array(spatial_preview_image)
            
            # Get the actual content dimensions and padding
            resized_dims_wh = feature_mapping.get("resized_dimensions", (0, 0))
            if resized_dims_wh == (0, 0):
                print("Error: Missing resized dimensions in feature mapping.")
                return None
            
            resized_w_actual, resized_h_actual = resized_dims_wh
            
            # Calculate padding
            pad_h_total = preview_h - resized_h_actual
            pad_w_total = preview_w - resized_w_actual
            pad_top = max(0, pad_h_total // 2)
            pad_left = max(0, pad_w_total // 2)
            
            # Upscale heatmap by patch size first
            try:
                # Upscale probability map by raw patch size first
                heatmap_unpadded = np.repeat(np.repeat(heatmap, patch_size, axis=0), patch_size, axis=1)
                heatmap_h_unpadded, heatmap_w_unpadded = heatmap_unpadded.shape
                
                # Resize to match the actual content area dimensions
                target_heatmap_size = (resized_h_actual, resized_w_actual)
                
                # Use scikit-image if available
                try:
                    from skimage.transform import resize as skimage_resize
                    resized_heatmap = skimage_resize(
                        heatmap_unpadded, target_heatmap_size, order=1,
                        mode='constant', cval=0, anti_aliasing=True,
                        preserve_range=True
                    )
                except ImportError:
                    # Fallback to simpler method
                    if heatmap_h_unpadded > 0 and heatmap_w_unpadded > 0:
                        scale_y = target_heatmap_size[0] / heatmap_h_unpadded
                        scale_x = target_heatmap_size[1] / heatmap_w_unpadded
                        y_indices = np.clip((np.arange(target_heatmap_size[0]) / scale_y), 0, heatmap_h_unpadded - 1).astype(int)
                        x_indices = np.clip((np.arange(target_heatmap_size[1]) / scale_x), 0, heatmap_w_unpadded - 1).astype(int)
                        resized_heatmap = heatmap_unpadded[y_indices[:, None], x_indices]
                    else:
                        # Not enough data for resizing
                        print("Warning: Cannot resize heatmap with zero dimensions.")
                        return None
            except Exception as e:
                print(f"Error upscaling/resizing patch heatmap: {e}")
                return None
            
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(8, 8 * preview_h / max(1, preview_w)))
            
            # Plot background image
            ax.imshow(background_np, extent=(0, preview_w, preview_h, 0))
            
            # Calculate extent for overlay using padding and dimensions
            extent = (pad_left, pad_left + resized_w_actual, pad_top + resized_h_actual, pad_top)
            
            # Plot heatmap overlay
            im = ax.imshow(
                resized_heatmap, 
                alpha=alpha,
                cmap=colormap,
                vmin=0,
                vmax=1,
                extent=extent,
                interpolation="nearest"
            )
            
            # Add grid lines if requested
            if add_gridlines:
                # Calculate cell sizes
                cell_height = resized_h_actual / prob_grid_h
                cell_width = resized_w_actual / prob_grid_w
                
                # Horizontal grid lines
                for i in range(1, prob_grid_h):
                    y = pad_top + i * cell_height
                    ax.axhline(y=y, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                # Vertical grid lines
                for i in range(1, prob_grid_w):
                    x = pad_left + i * cell_width
                    ax.axvline(x=x, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Normalized Influence Weight")
            
            # Set title and remove axes
            ax.set_title(title, fontsize=12)
            ax.axis("off")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            return save_path
        
        except Exception as e:
            print(f"Error creating patch feature overlay: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _create_composite_image(
        self,
        image_paths: List[str],
        layers: List[int],
        output_filename: str,
        title: str,
        background_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Optional[str]:
        """
        Creates a composite image grid from individual heatmap visualizations.
        
        Args:
            image_paths: List of file paths to the individual images
            layers: List of layer indices corresponding to each image
            output_filename: Path to save the composite image
            title: Title for the composite image
            background_color: RGB background color
            
        Returns:
            Path to saved composite image or None if failed
        """
        try:
            # Internal padding constants
            padding = 10
            label_padding = 25
            
            if not image_paths:
                print(f"Error: No image paths provided for composite.")
                return None
            
            if len(image_paths) != len(layers):
                print(f"Warning: Mismatch between image paths ({len(image_paths)}) and layers ({len(layers)}).")
                # Match paths and layers based on filename
                path_layer_map = {}
                for p in image_paths:
                    try:
                        layer_num_str = os.path.basename(p).split('_')[2]  # Assumes format like "base_influence_layer_32_..."
                        path_layer_map[int(layer_num_str)] = p
                    except (IndexError, ValueError):
                        print(f"Could not extract layer number from filename: {os.path.basename(p)}")
                
                # Rebuild lists based on layers
                matched_paths = [path_layer_map.get(l) for l in layers]
                filtered_layers = [l for l, p in zip(layers, matched_paths) if p is not None]
                filtered_paths = [p for p in matched_paths if p is not None]
                
                if not filtered_paths:
                    print(f"Error: No images could be matched to layers.")
                    return None
                
                image_paths = filtered_paths
                layers = filtered_layers
            
            # Load first image to get dimensions
            with Image.open(image_paths[0]) as img:
                img_w, img_h = img.size
                img_mode = img.mode
            
            # Calculate grid dimensions
            num_images = len(image_paths)
            cols = math.ceil(math.sqrt(num_images))
            rows = math.ceil(num_images / cols)
            
            # Calculate canvas dimensions
            cell_w = img_w + padding
            cell_h = img_h + padding + label_padding
            title_height = 50
            canvas_w = cols * cell_w + padding
            canvas_h = rows * cell_h + padding + title_height
            
            # Create canvas and draw object
            canvas = Image.new(img_mode, (canvas_w, canvas_h), background_color)
            draw = ImageDraw.Draw(canvas)
            
            # Define fonts
            try:
                DEFAULT_FONT = ImageFont.truetype("arial.ttf", 18)
                DEFAULT_FONT_SMALL = ImageFont.truetype("arial.ttf", 12)
            except IOError:
                DEFAULT_FONT = ImageFont.load_default()
                DEFAULT_FONT_SMALL = ImageFont.load_default()
            
            # Add title
            try:
                title_bbox = draw.textbbox((0, 0), title, font=DEFAULT_FONT)
                title_w = title_bbox[2] - title_bbox[0]
                title_h = title_bbox[3] - title_bbox[1]
            except AttributeError:
                title_w, title_h = draw.textlength(title, font=DEFAULT_FONT), 20
            
            title_x = (canvas_w - title_w) // 2
            title_y = padding
            draw.text((title_x, title_y), title, fill=(0, 0, 0), font=DEFAULT_FONT)
            
            # Paste images and add labels
            current_col = 0
            current_row = 0
            
            for i, (img_path, layer_idx) in enumerate(zip(image_paths, layers)):
                try:
                    with Image.open(img_path) as img:
                        if img.mode != canvas.mode:
                            img = img.convert(canvas.mode)
                        
                        paste_x = padding + current_col * cell_w
                        paste_y = padding + current_row * cell_h + title_height
                        
                        canvas.paste(img, (paste_x, paste_y))
                        
                        label_text = f"Layer {layer_idx}"
                        try:
                            label_bbox = draw.textbbox((0, 0), label_text, font=DEFAULT_FONT_SMALL)
                            label_w = label_bbox[2] - label_bbox[0]
                        except AttributeError:
                            label_w = draw.textlength(label_text, font=DEFAULT_FONT_SMALL)
                        
                        label_x = paste_x + (img_w - label_w) // 2
                        label_y = paste_y + img_h + (padding // 2)
                        
                        draw.text((label_x, label_y), label_text, fill=(50, 50, 50), font=DEFAULT_FONT_SMALL)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
                
                current_col += 1
                if current_col >= cols:
                    current_col = 0
                    current_row += 1
            
            # Save the composite image
            canvas.save(output_filename)
            print(f"Saved composite image to: {output_filename}")
            return output_filename
        
        except Exception as e:
            print(f"Error creating composite image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_text_report(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Creates a detailed text report of semantic tracing results.
        
        Args:
            results: Results dictionary from semantic tracing
            output_path: Path to save the report
            
        Returns:
            Path to saved report
        """
        trace_results = results.get("trace_results", {})
        target_token = results.get("target_token", {})
        target_text = target_token.get("text", "unknown")
        target_idx = target_token.get("index", -1)
        
        # Convert keys to integers if they're strings
        if trace_results and all(isinstance(key, str) for key in trace_results.keys()):
            trace_results = {int(k): v for k, v in trace_results.items()}
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Semantic Tracing Report\n")
            f.write(f"======================\n\n")
            f.write(f"Target Token: '{target_text}' (Index: {target_idx})\n\n")
            
            layers = sorted(trace_results.keys())
            f.write(f"Analysis of {len(layers)} layers:\n")
            
            for layer_idx in layers:
                layer_data = trace_results[layer_idx]
                layer_name = layer_data.get("layer_name", f"Layer {layer_idx}")
                
                f.write(f"\n{'='*40}\n")
                f.write(f"Layer {layer_idx} ({layer_name}):\n")
                f.write(f"{'-'*40}\n")
                
                # Targets in this layer
                targets = layer_data.get("target_tokens", [])
                f.write(f"Targets in this layer: {len(targets)}\n")
                
                for i, target in enumerate(targets):
                    f.write(f"\n  Target {i+1}: '{target.get('text', '')}' (Index: {target.get('index', -1)})\n")
                    f.write(f"  Importance Weight: {target.get('weight', 0.0):.4f}\n")
                    
                    # Sources for this target
                    sources = target.get("sources", [])
                    f.write(f"  Sources: {len(sources)}\n")
                    
                    for j, source in enumerate(sources):
                        source_type = source.get("type", -1)
                        type_name = "Text" if source_type == 1 else "Image" if source_type == 2 else "Generated"
                        
                        f.write(f"    Source {j+1}: '{source.get('text', '')}' (Index: {source.get('index', -1)}, Type: {type_name})\n")
                        f.write(f"      Raw Saliency: {source.get('saliency_score', 0.0):.4f}\n")
                        f.write(f"      Relative Weight: {source.get('relative_weight', 0.0):.4f}\n")
                        f.write(f"      Global Weight: {source.get('scaled_weight', 0.0):.4f}\n")
                        
                        # Add logit lens info if available
                        logit_lens = layer_data.get("logit_lens_projections", {}).get(source.get("index", -1), {})
                        if not logit_lens and isinstance(source.get("index"), int):
                            # Try looking up as string key
                            logit_lens = layer_data.get("logit_lens_projections", {}).get(str(source.get("index", -1)), {})
                        
                        if logit_lens:
                            top_predictions = logit_lens.get("top_predictions", [])
                            if top_predictions:
                                f.write("      Top predictions:\n")
                                for pred in top_predictions[:3]:  # Show top 3
                                    f.write(f"        {pred.get('rank', 0)}: '{pred.get('token_text', '')}' ({pred.get('probability', 0.0):.4f})\n")
                            
                            concept_predictions = logit_lens.get("concept_predictions", {})
                            if concept_predictions:
                                f.write("      Concept probabilities:\n")
                                for concept, data in concept_predictions.items():
                                    f.write(f"        '{concept}': {data.get('probability', 0.0):.4f}\n")
        
        return output_path
    
    def _visualize_trace_data(
        self,
        csv_path: str,
        target_text: str,
        target_idx: int,
        save_dir: str
    ) -> List[str]:
        """Visualize trace data from the CSV file"""
        saved_paths = []
        
        if not os.path.exists(csv_path):
            print(f"Trace data CSV not found at {csv_path}")
            return saved_paths
            
        try:
            df = pd.read_csv(csv_path)
            
            # Create directory for trace visualizations
            trace_vis_dir = os.path.join(save_dir, "trace_data_plots")
            os.makedirs(trace_vis_dir, exist_ok=True)
            
            # 1. Top predicted tokens by layer
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Focus on unique token positions
            token_positions = df["token_index"].unique()
            
            # Create color map for token positions
            cmap = plt.cm.get_cmap('tab20', len(token_positions))
            color_map = {pos: cmap(i) for i, pos in enumerate(token_positions)}
            
            # Group by layer and get top predicted tokens
            layers = sorted(df["layer"].unique())
            
            # For each unique token position, plot its top prediction probability across layers
            for pos in token_positions:
                token_df = df[df["token_index"] == pos]
                if len(token_df) >= len(layers) * 0.5:  # Only include if present in at least half the layers
                    token_text = token_df["token_text"].iloc[0]
                    layer_probs = []
                    
                    for layer in layers:
                        layer_row = token_df[token_df["layer"] == layer]
                        if not layer_row.empty:
                            prob = layer_row["predicted_top_prob"].iloc[0]
                            layer_probs.append(prob)
                        else:
                            layer_probs.append(None)
                    
                    # Plot with positions that have data
                    valid_indices = [i for i, p in enumerate(layer_probs) if p is not None]
                    valid_layers = [layers[i] for i in valid_indices]
                    valid_probs = [layer_probs[i] for i in valid_indices]
                    
                    if valid_probs:
                        ax.plot(valid_layers, valid_probs, marker='o', label=f"{pos}: '{token_text}'", 
                                color=color_map[pos], linewidth=2, alpha=0.8)
            
            ax.set_xlabel("Layer")
            ax.set_ylabel("Top Prediction Probability")
            ax.set_title(f"Top Token Prediction Confidence by Layer for Target '{target_text}'")
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xticks(layers)
            
            # Add legend with reasonable size
            if len(token_positions) > 15:
                # Too many tokens, use a compact legend
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
            else:
                ax.legend(loc='best')
                
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(trace_vis_dir, f"top_predictions_by_layer_{target_idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_paths.append(save_path)
            
            plt.close(fig)
            
            # 2. Concept probabilities by layer for selected tokens
            # Identify concepts in the data
            concept_cols = [col for col in df.columns if col.startswith("concept_") and col.endswith("_prob")]
            concepts = [col.replace("concept_", "").replace("_prob", "") for col in concept_cols]
            
            if concepts:
                # For each concept, create a plot
                for concept, concept_col in zip(concepts, concept_cols):
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Again plot by token position
                    for pos in token_positions:
                        token_df = df[df["token_index"] == pos]
                        if len(token_df) >= len(layers) * 0.5:  # Only include if present in at least half the layers
                            token_text = token_df["token_text"].iloc[0]
                            layer_probs = []
                            
                            for layer in layers:
                                layer_row = token_df[token_df["layer"] == layer]
                                if not layer_row.empty and concept_col in layer_row.columns:
                                    prob = layer_row[concept_col].iloc[0]
                                    layer_probs.append(prob)
                                else:
                                    layer_probs.append(None)
                            
                            # Plot with positions that have data
                            valid_indices = [i for i, p in enumerate(layer_probs) if p is not None]
                            valid_layers = [layers[i] for i in valid_indices]
                            valid_probs = [layer_probs[i] for i in valid_indices]
                            
                            if valid_probs and max(valid_probs) > 0.01:  # Only show tokens with some probability
                                ax.plot(valid_layers, valid_probs, marker='o', label=f"{pos}: '{token_text}'", 
                                        color=color_map[pos], linewidth=2, alpha=0.8)
                    
                    ax.set_xlabel("Layer")
                    ax.set_ylabel(f"'{concept}' Probability")
                    ax.set_title(f"Concept '{concept}' Probability by Layer for Target '{target_text}'")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_xticks(layers)
                    
                    # Add legend with reasonable size
                    if ax.get_legend_handles_labels()[0]:  # Only add legend if there are items
                        if len(token_positions) > 15:
                            # Too many tokens, use a compact legend
                            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
                        else:
                            ax.legend(loc='best')
                    
                    plt.tight_layout()
                    
                    # Save figure
                    save_path = os.path.join(trace_vis_dir, f"concept_{concept}_by_layer_{target_idx}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    saved_paths.append(save_path)
                    
                    plt.close(fig)
                
                # 3. Create heatmap of all concept probabilities for key tokens
                # Group by token_index and get max probability for each concept
                token_probs = df.groupby('token_index').agg({
                    'token_text': 'first',
                    'token_type': 'first',
                    **{col: 'max' for col in concept_cols}
                }).reset_index()
                
                # Select tokens with at least one significant concept probability
                token_probs['max_concept_prob'] = token_probs[concept_cols].max(axis=1)
                significant_tokens = token_probs[token_probs['max_concept_prob'] > 0.05]
                
                if len(significant_tokens) > 0:
                    # Create a heatmap
                    plt.figure(figsize=(12, max(6, len(significant_tokens) * 0.4)))
                    
                    # Prepare data for heatmap
                    heatmap_data = significant_tokens.set_index('token_text')[concept_cols]
                    # Rename columns to just concept names
                    heatmap_data.columns = concepts
                    
                    # Create heatmap
                    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Probability'})
                    plt.title(f'Maximum Concept Probabilities for Key Tokens - Target {target_idx}: "{target_text}"')
                    plt.tight_layout()
                    
                    # Save figure
                    save_path = os.path.join(trace_vis_dir, f"concept_heatmap_{target_idx}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    saved_paths.append(save_path)
                    
                    plt.close()
        
        except Exception as e:
            print(f"Error visualizing trace data: {e}")
            import traceback
            traceback.print_exc()
        
        return saved_paths
    
    def _visualize_multi_token_concept_evolution(
        self,
        all_traces: List[Dict[int, Dict[str, Any]]],
        target_texts: List[str],
        save_dir: str
    ) -> List[str]:
        """Create visualization of concept evolution across multiple target tokens"""
        saved_paths = []
        
        if not all_traces or not target_texts:
            return saved_paths
            
        # Check that we have traces with consistent layers
        all_layers = [sorted(trace.keys()) for trace in all_traces]
        if not all(layers == all_layers[0] for layers in all_layers):
            print("Warning: Inconsistent layer indices across traces. Using intersection.")
            common_layers = sorted(set.intersection(*[set(layers) for layers in all_layers]))
            if not common_layers:
                print("No common layers found across traces. Cannot create combined visualization.")
                return saved_paths
        else:
            common_layers = all_layers[0]
        
        # For each concept, create a plot showing evolution across all tokens
        for concept in self.logit_lens_concepts:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for trace_idx, (trace, target_text) in enumerate(zip(all_traces, target_texts)):
                # Extract concept probabilities for this trace
                concept_probs = []
                
                for layer_idx in common_layers:
                    layer_data = trace.get(layer_idx, {})
                    projections = layer_data.get("logit_lens_projections", {})
                    
                    if projections:
                        token_probs = []
                        for token_idx, proj_data in projections.items():
                            concept_pred = proj_data.get("concept_predictions", {}).get(concept)
                            if concept_pred:
                                token_probs.append(concept_pred["probability"])
                        
                        if token_probs:
                            avg_prob = sum(token_probs) / len(token_probs)
                        else:
                            avg_prob = 0.0
                    else:
                        avg_prob = 0.0
                        
                    concept_probs.append(avg_prob)
                
                # Plot this token's concept evolution
                ax.plot(common_layers, concept_probs, marker='o', linewidth=2, 
                       label=f"'{target_text}'")
            
            ax.set_ylabel(f"'{concept}' Probability")
            ax.set_xlabel("Layer")
            ax.set_title(f"Concept '{concept}' Evolution Across Layers for Multiple Tokens")
            ax.set_xticks(common_layers)
            ax.set_xticklabels([str(l) for l in common_layers], rotation=90)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, f"multi_token_concept_{concept}_evolution.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_paths.append(save_path)
            
            plt.close(fig)
        
        return saved_paths
    
    def _visualize_multi_token_type_distribution(
        self,
        all_traces: List[Dict[int, Dict[str, Any]]],
        target_texts: List[str],
        save_dir: str
    ) -> List[str]:
        """Create visualization of token type distributions across multiple target tokens"""
        saved_paths = []
        
        if not all_traces or not target_texts:
            return saved_paths
            
        # Check that we have traces with consistent layers
        all_layers = [sorted(trace.keys()) for trace in all_traces]
        if not all(layers == all_layers[0] for layers in all_layers):
            print("Warning: Inconsistent layer indices across traces. Using intersection.")
            common_layers = sorted(set.intersection(*[set(layers) for layers in all_layers]))
            if not common_layers:
                print("No common layers found across traces. Cannot create combined visualization.")
                return saved_paths
        else:
            common_layers = all_layers[0]
        
        # Create a plot for each token type
        for token_type, type_name in [(1, "Text"), (2, "Image"), (0, "Generated")]:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for trace_idx, (trace, target_text) in enumerate(zip(all_traces, target_texts)):
                # Extract type contributions for this trace
                type_contribs = []
                
                for layer_idx in common_layers:
                    layer_data = trace.get(layer_idx, {})
                    
                    # Combine all sources across all targets for this layer
                    all_sources = []
                    for target in layer_data.get("target_tokens", []):
                        all_sources.extend(target.get("sources", []))
                    
                    # Calculate contribution for this type
                    type_contrib = sum(src.get("scaled_weight", 0.0) for src in all_sources if src.get("type") == token_type)
                    
                    # Normalize relative to all contributions
                    total = sum(src.get("scaled_weight", 0.0) for src in all_sources)
                    if total > 0:
                        type_contrib /= total
                        
                    type_contribs.append(type_contrib)
                
                # Plot this token's type contribution
                ax.plot(common_layers, type_contribs, marker='o', linewidth=2, 
                       label=f"'{target_text}'")
            
            ax.set_ylabel(f"{type_name} Token Contribution")
            ax.set_xlabel("Layer")
            ax.set_title(f"{type_name} Token Influence Across Layers for Multiple Tokens")
            ax.set_xticks(common_layers)
            ax.set_xticklabels([str(l) for l in common_layers], rotation=90)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(save_dir, f"multi_token_{type_name.lower()}_contribution.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            saved_paths.append(save_path)
            
            plt.close(fig)
        
        return saved_paths