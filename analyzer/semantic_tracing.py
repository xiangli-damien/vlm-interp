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
import networkx as nx

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
        self.saliency_cache = {}
        
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
            
            # Rest of the method remains unchanged...
            # [existing code for processing saliency maps, updating target tokens, etc.]
            
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
            
            # [existing code for the rest of the method...]
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
        trace_results: Dict[str, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str,
    ) -> List[str]:
        """Visualize the distribution of source token types (text, image, generated) across layers"""
        saved_paths = []
        
        # Handle mixed key types (string and integer)
        # First convert to a consistent format - all integers
        int_keyed_results = {}
        for k, v in trace_results.items():
            try:
                # Try to convert to int if it's a string
                key = int(k) if isinstance(k, str) else k
                int_keyed_results[key] = v
            except (ValueError, TypeError):
                # If conversion fails, skip this item
                print(f"Warning: Could not convert key {k} to integer, skipping.")
                continue
        
        # Now replace the original dict with our sanitized version
        trace_results = int_keyed_results
        
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
        trace_results: Dict[str, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str
    ) -> List[str]:
        """Visualize how concepts evolve across layers using logit lens projections"""
        saved_paths = []
        
        # Handle mixed key types (string and integer)
        int_keyed_results = {}
        for k, v in trace_results.items():
            try:
                key = int(k) if isinstance(k, str) else k
                int_keyed_results[key] = v
            except (ValueError, TypeError):
                print(f"Warning: Could not convert key {k} to integer, skipping.")
                continue
        
        trace_results = int_keyed_results
        
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
        trace_results: Dict[str, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str,
        input_data: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Visualize heatmaps of image token influence across layers by mapping tokens
        back to their spatial positions in the original image.
        """
        saved_paths = []
        
        # Extract special keys before integer conversion
        special_keys = {}
        for k, v in trace_results.items():
            if k == "input_data" or k == "trace_data_path":
                special_keys[k] = v
        
        # Handle mixed key types (string and integer)
        int_keyed_results = {}
        for k, v in trace_results.items():
            # Skip special keys
            if k == "input_data" or k == "trace_data_path":
                continue
                
            try:
                key = int(k) if isinstance(k, str) else k
                int_keyed_results[key] = v
            except (ValueError, TypeError):
                print(f"Warning: Could not convert key {k} to integer, skipping.")
                continue
        
        # Now replace the original dict with our sanitized version
        trace_results = int_keyed_results
        
        # Determine the source of input_data (parameter or trace_results)
        if input_data is None:
            input_data = special_keys.get("input_data", {})
        
        # Get feature mapping and image information
        if not input_data:
            print("Error: No input_data found in trace_results or parameters. Cannot visualize image token heatmaps.")
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
        
        # Get numeric layers
        layers = sorted(trace_results.keys())
        if not layers:
            print("Warning: No layers found in trace_results. Cannot create heatmaps.")
            return saved_paths
        
        # Extract base and patch feature information
        base_feature_info = feature_mapping.get("base_feature", {})
        patch_feature_info = feature_mapping.get("patch_feature", {})
        
        if not base_feature_info and not patch_feature_info:
            print("Error: No valid feature mapping information found. Cannot create visualizations.")
            return saved_paths
        
        if not base_feature_info:
            print("Warning: Missing base feature mapping information. Base feature visualizations will be skipped.")
        
        if not patch_feature_info:
            print("Warning: Missing patch feature mapping information. Patch feature visualizations will be skipped.")
        
        # Prepare output directories
        base_heatmap_dir = os.path.join(save_dir, "base_feature_heatmaps")
        patch_heatmap_dir = os.path.join(save_dir, "patch_feature_heatmaps")
        os.makedirs(base_heatmap_dir, exist_ok=True)
        os.makedirs(patch_heatmap_dir, exist_ok=True)
        
        # Keep track of which layers have valid visualizations
        base_valid_layers = []
        patch_valid_layers = []
        
        # Process each layer
        for layer_idx in layers:
            layer_data = trace_results[layer_idx]
            
            # Process all targets in this layer
            targets = layer_data.get("target_tokens", [])
            if not targets:
                print(f"Layer {layer_idx}: No target tokens found. Skipping.")
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
                print(f"Layer {layer_idx}: No image tokens with influence found. Skipping.")
                continue
            
            # Normalize weights for visualization
            max_weight = max(image_token_weights.values())
            if max_weight <= 0:
                print(f"Layer {layer_idx}: All image token weights are zero or negative. Skipping.")
                continue
                
            normalized_weights = {idx: weight / max_weight for idx, weight in image_token_weights.items()}
            
            # 1. Visualize base features
            if base_feature_info.get("positions") and base_feature_info.get("grid"):
                base_grid_h, base_grid_w = base_feature_info["grid"]
                
                # Initialize empty heatmap
                base_heatmap = np.zeros((base_grid_h, base_grid_w), dtype=np.float32)
                
                # Fill the heatmap with normalized weights
                mapped_tokens = 0
                for token_idx, weight in normalized_weights.items():
                    # Convert token_idx to int if it's a string
                    if isinstance(token_idx, str):
                        token_idx = int(token_idx)
                        
                    position = base_feature_info["positions"].get(token_idx)
                    if position:
                        r, c = position
                        if 0 <= r < base_grid_h and 0 <= c < base_grid_w:
                            base_heatmap[r, c] = weight
                            mapped_tokens += 1
                
                # Create visualization if we have data
                if mapped_tokens > 0 and np.max(base_heatmap) > 0:
                    print(f"Layer {layer_idx}: Creating base feature heatmap with {mapped_tokens} mapped tokens.")
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
                        base_valid_layers.append(layer_idx)
                else:
                    print(f"Layer {layer_idx}: No valid base feature mapping. Max value: {np.max(base_heatmap)}")
            
            # 2. Visualize patch features
            if patch_feature_info.get("positions") and patch_feature_info.get("grid_unpadded"):
                prob_grid_h, prob_grid_w = patch_feature_info["grid_unpadded"]
                
                # Initialize empty heatmap
                patch_heatmap = np.zeros((prob_grid_h, prob_grid_w), dtype=np.float32)
                
                # Fill the heatmap with normalized weights
                mapped_tokens = 0
                for token_idx, weight in normalized_weights.items():
                    # Convert token_idx to int if it's a string
                    if isinstance(token_idx, str):
                        token_idx = int(token_idx)
                        
                    position = patch_feature_info["positions"].get(token_idx)
                    if position:
                        r, c = position
                        if 0 <= r < prob_grid_h and 0 <= c < prob_grid_w:
                            patch_heatmap[r, c] = weight
                            mapped_tokens += 1
                
                # Create visualization if we have data
                if mapped_tokens > 0 and np.max(patch_heatmap) > 0:
                    print(f"Layer {layer_idx}: Creating patch feature heatmap with {mapped_tokens} mapped tokens.")
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
                        patch_valid_layers.append(layer_idx)
                else:
                    print(f"Layer {layer_idx}: No valid patch feature mapping. Max value: {np.max(patch_heatmap)}")
        
        # Create composite visualizations if we have multiple layers
        if len(saved_paths) > 0:
            # Create composite for base feature heatmaps
            base_paths = [p for p in saved_paths if "base_influence" in p]
            if base_paths:
                try:
                    print(f"Creating composite base image with {len(base_paths)} heatmaps...")
                    base_composite_path = os.path.join(save_dir, f"composite_base_influence_{target_idx}.png")
                    composite_path = self._create_composite_image(
                        image_paths=base_paths,
                        layers=base_valid_layers,
                        output_filename=base_composite_path,
                        title=f"Base Image Token Influence Across Layers for Target '{target_text}'"
                    )
                    if composite_path:
                        saved_paths.append(composite_path)
                        print(f"Successfully created base composite image: {composite_path}")
                except Exception as e:
                    print(f"Error creating base composite: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Create composite for patch feature heatmaps
            patch_paths = [p for p in saved_paths if "patch_influence" in p]
            if patch_paths:
                try:
                    print(f"Creating composite patch image with {len(patch_paths)} heatmaps...")
                    patch_composite_path = os.path.join(save_dir, f"composite_patch_influence_{target_idx}.png")
                    composite_path = self._create_composite_image(
                        image_paths=patch_paths,
                        layers=patch_valid_layers,
                        output_filename=patch_composite_path,
                        title=f"Patch Image Token Influence Across Layers for Target '{target_text}'"
                    )
                    if composite_path:
                        saved_paths.append(composite_path)
                        print(f"Successfully created patch composite image: {composite_path}")
                except Exception as e:
                    print(f"Error creating patch composite: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("No valid heatmaps were created. Check if image tokens have significant influence in any layer.")
        
        return saved_paths

    def _visualize_single_trace(
        self,
        trace_results: Dict[int, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str,
        input_data: Optional[Dict[str, Any]] = None,
        flow_graph_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Helper to visualize a single token's trace results"""
        saved_paths = []
        
        # Extract special keys before processing numeric keys
        special_keys = {}
        for k, v in trace_results.items():
            if not isinstance(k, int) and not k.isdigit():
                special_keys[k] = v
        
        # Try to get input_data from trace_results if not provided
        if input_data is None and "input_data" in special_keys:
            input_data = special_keys["input_data"]
        
        # Set default flow graph parameters if not provided
        if flow_graph_params is None:
            flow_graph_params = {
                "output_format": "both",
                "align_tokens_by_layer": True,
                "show_orphaned_nodes": False,
                "use_variable_node_size": True
            }
        
        # 1. Create new token flow graph visualization (NEW)
        try:
            print("Creating token flow graph visualization...")
            paths = self._visualize_token_flow_graph(
                trace_results, 
                target_text, 
                target_idx, 
                save_dir,
                **flow_graph_params  # Pass the flow graph parameters
            )
            saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing token flow graph: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Visualize the trace as a stacked bar chart
        try:
            paths = self._visualize_source_distribution(trace_results, target_text, target_idx, save_dir)
            saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing source distribution: {e}")
            import traceback
            traceback.print_exc()
        
        # 3. Visualize the semantic evolution using logit lens
        try:
            paths = self._visualize_semantic_evolution(trace_results, target_text, target_idx, save_dir)
            saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing semantic evolution: {e}")
            import traceback
            traceback.print_exc()
        
        # 4. Visualize image token heatmaps if available - pass input_data
        try:
            paths = self._visualize_image_token_heatmaps(
                trace_results, 
                target_text, 
                target_idx, 
                save_dir,
                input_data=input_data
            )
            saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing image token heatmaps: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. Create a text report
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
        
        # 6. Create trace data visualization from CSV if available
        try:
            trace_data_path = special_keys.get("trace_data_path")
            if trace_data_path and os.path.exists(trace_data_path):
                paths = self._visualize_trace_data(trace_data_path, target_text, target_idx, save_dir)
                saved_paths.extend(paths)
        except Exception as e:
            print(f"Error visualizing trace data: {e}")
            import traceback
            traceback.print_exc()
        
        return saved_paths

    def visualize_trace(
        self, 
        results: Dict[str, Any], 
        save_dir: Optional[str] = None,
        flow_graph_params: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Visualize the semantic tracing results with enhanced outputs.
        
        Args:
            results: Results from generate_and_analyze
            save_dir: Directory to save visualizations (defaults to self.output_dir)
            flow_graph_params: Optional parameters for flow graph visualization
            
        Returns:
            List of saved file paths
        """
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, "visualizations")
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        # Set default flow graph params if not provided
        if flow_graph_params is None:
            flow_graph_params = {
                "output_format": "both",
                "align_tokens_by_layer": True,
                "show_orphaned_nodes": False,
                "use_variable_node_size": True
            }
        
        # Extract top-level input_data from results
        input_data = results.get("input_data")
        
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
                    
                    # Pass input_data and flow_graph_params to visualization
                    token_paths = self._visualize_single_trace(
                        trace_results, 
                        token_text, 
                        token_idx, 
                        token_dir,
                        input_data=input_data,
                        flow_graph_params=flow_graph_params
                    )
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
            
            # Visualize the single trace - pass input_data and flow_graph_params
            token_paths = self._visualize_single_trace(
                trace_results, 
                target_text, 
                target_idx, 
                save_dir,
                input_data=input_data,
                flow_graph_params=flow_graph_params
            )
            saved_paths.extend(token_paths)
        
        print(f"Total visualization files created: {len(saved_paths)}")
        return saved_paths

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
        
        # Handle mixed key types (string and integer)
        int_keyed_results = {}
        for k, v in trace_results.items():
            try:
                key = int(k) if isinstance(k, str) else k
                int_keyed_results[key] = v
            except (ValueError, TypeError):
                print(f"Warning: Could not convert key {k} to integer, skipping.")
                continue
        
        trace_results = int_keyed_results
        
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
            
            # 2. Create heatmap of all concept probabilities for key tokens
            concept_cols = [col for col in df.columns if col.startswith("concept_") and col.endswith("_prob")]
            if concept_cols:
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
                    heatmap_data.columns = [col.replace("concept_", "").replace("_prob", "") for col in concept_cols]
                    
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
        all_traces: List[Dict[str, Dict[str, Any]]],
        target_texts: List[str],
        save_dir: str
    ) -> List[str]:
        """Create visualization of concept evolution across multiple target tokens"""
        saved_paths = []
        
        if not all_traces or not target_texts:
            return saved_paths
        
        # Process each trace to handle mixed key types
        processed_traces = []
        for trace in all_traces:
            int_keyed_trace = {}
            for k, v in trace.items():
                try:
                    key = int(k) if isinstance(k, str) else k
                    int_keyed_trace[key] = v
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert key {k} to integer, skipping.")
                    continue
            processed_traces.append(int_keyed_trace)
        
        all_traces = processed_traces
        
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
                            # Handle token_idx as string if needed
                            if isinstance(token_idx, str):
                                token_idx = int(token_idx)
                                
                            concept_pred = proj_data.get("concept_predictions", {}).get(concept)
                            if concept_pred:
                                token_probs.append(concept_pred["probability"])
                        
                        # Average if we have values
                        if token_probs:
                            avg_prob = sum(token_probs) / len(token_probs)
                        else:
                            avg_prob = 0.0
                    else:
                        # No projections for this layer, use zeros
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
        all_traces: List[Dict[str, Dict[str, Any]]],
        target_texts: List[str],
        save_dir: str
    ) -> List[str]:
        """Create visualization of token type distributions across multiple target tokens"""
        saved_paths = []
        
        if not all_traces or not target_texts:
            return saved_paths
        
        # Process each trace to handle mixed key types
        processed_traces = []
        for trace in all_traces:
            int_keyed_trace = {}
            for k, v in trace.items():
                try:
                    key = int(k) if isinstance(k, str) else k
                    int_keyed_trace[key] = v
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert key {k} to integer, skipping.")
                    continue
            processed_traces.append(int_keyed_trace)
        
        all_traces = processed_traces
        
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
                        # More robust layer extraction from filename
                        filename = os.path.basename(p)
                        parts = filename.split('_')
                        layer_num = None
                        
                        # Look for "layer_X" pattern in filename
                        for i, part in enumerate(parts):
                            if part == "layer" and i+1 < len(parts) and parts[i+1].isdigit():
                                layer_num = int(parts[i+1])
                                break
                        
                        # Fallback: Try other patterns commonly found in filenames
                        if layer_num is None:
                            for part in parts:
                                if part.isdigit() and 0 <= int(part) < 100:  # Reasonable layer range
                                    layer_num = int(part)
                                    break
                        
                        if layer_num is not None:
                            path_layer_map[layer_num] = p
                        else:
                            print(f"Could not extract layer number from filename: {filename}")
                    except (IndexError, ValueError) as e:
                        print(f"Could not extract layer number from filename: {os.path.basename(p)}: {e}")
                    
                # Rebuild lists based on layers
                matched_paths = [path_layer_map.get(l) for l in layers]
                filtered_layers = [l for l, p in zip(layers, matched_paths) if p is not None]
                filtered_paths = [p for p in matched_paths if p is not None]
                
                if not filtered_paths:
                    print(f"Error: No images could be matched to layers.")
                    return None
                
                image_paths = filtered_paths
                layers = filtered_layers
            
            # Rest of the function remains the same...
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

    
    def _visualize_token_flow_graph(
        self,
        trace_results: Dict[str, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str,
        title: str = None,
        max_layers: int = None,
        output_format: str = "both",  # "png", "svg", or "both"
        align_tokens_by_layer: bool = True,
        show_orphaned_nodes: bool = False,  
        min_edge_weight: float = 0.05,
        use_variable_node_size: bool = True,
        min_node_size: int = 800,
        max_node_size: int = 2000,
        debug_mode: bool = False,
        dpi: int = 150,
        show_continuation_edges: bool = False,  # New parameter to control continuation edges
        use_exponential_scaling: bool = True  # New parameter for exponential weight scaling
    ) -> List[str]:
        """
        Visualizes the semantic trace as a flow graph showing token connections across layers.
        
        Creates a layered visualization with:
        - One column per model layer
        - Tokens as nodes colored by type (image=red, text=blue, generated=green)
        - Connections between source and target tokens across layers
        - Line thickness representing saliency score contributions
        - Variable node size representing token importance (exponential scaling optional)
        - Each token showing its text and highest predicted token
        
        Args:
            trace_results: Dictionary with trace results for each layer
            target_text: Text of the target token
            target_idx: Index of the target token
            save_dir: Directory to save visualizations
            title: Custom title for the visualization
            max_layers: Maximum number of layers to include (None = all)
            output_format: Output format, one of "png", "svg", or "both"
            align_tokens_by_layer: Whether to align tokens in strict columns by layer
            show_orphaned_nodes: Whether to show nodes with no connections
            min_edge_weight: Minimum edge weight to display (filters weak connections)
            use_variable_node_size: Whether to vary node size based on weight
            min_node_size: Minimum node size for visualization
            max_node_size: Maximum node size for visualization
            debug_mode: Whether to print debug information
            dpi: DPI for PNG output
            show_continuation_edges: Whether to show dashed continuation edges between layers
            use_exponential_scaling: Whether to use exponential scaling for node sizes and edge widths
            
        Returns:
            List of saved file paths
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        import matplotlib as mpl
        import math
        import numpy as np
        
        saved_paths = []
        
        # Extract special keys before integer conversion
        special_keys = {}
        for k, v in trace_results.items():
            if not isinstance(k, int) and not k.isdigit():
                special_keys[k] = v
        
        # Handle mixed key types (string and integer)
        int_keyed_results = {}
        for k, v in trace_results.items():
            # Skip special keys
            if k == "input_data" or k == "trace_data_path":
                continue
                    
            try:
                key = int(k) if isinstance(k, str) else k
                int_keyed_results[key] = v
            except (ValueError, TypeError):
                if debug_mode:
                    print(f"Warning: Could not convert key {k} to integer, skipping.")
                continue
        
        # Now replace the original dict with our sanitized version
        trace_results = int_keyed_results
        
        # Sort layers (assume sequential layers)
        layers = sorted(trace_results.keys())
        if not layers:
            print("Warning: No layers found in trace results.")
            return saved_paths
        
        if max_layers is not None and max_layers < len(layers):
            # Sample layers evenly if we need to reduce
            step = len(layers) // max_layers
            layers = layers[::step]
            if len(layers) > max_layers:
                layers = layers[:max_layers]
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Dictionary to map token indices to node IDs
        token_to_node = {}  # {token_idx: {layer_idx: node_id}}
        
        # Dictionary to store node metadata
        node_metadata = {}  # {node_id: {metadata}}
        
        # Dictionary to track tokens by layer
        layer_tokens = {layer_idx: [] for layer_idx in layers}  # {layer_idx: [token_indices]}
        
        # Track all unique token indices
        all_token_indices = set()
        
        # Special handling for start token and other special tokens
        special_token_indices = set()  # Collect indices of special tokens to ensure they show up
        
        if debug_mode:
            print("DEBUG MODE: Processing trace results to create flow graph")
            print(f"Found {len(layers)} layers: {layers}")
        
        # PHASE 1: First pass to collect all tokens and create unified node IDs
        for layer_idx in layers:
            layer_data = trace_results[layer_idx]
            
            if debug_mode:
                print(f"\nDEBUG: Processing layer {layer_idx}")
                num_targets = len(layer_data.get("target_tokens", []))
                print(f"Found {num_targets} target tokens in layer {layer_idx}")
            
            # Process target tokens in this layer
            for target in layer_data.get("target_tokens", []):
                target_token_idx = target.get("index")
                if target_token_idx is None:
                    continue
                
                # Add to tracking collections
                all_token_indices.add(target_token_idx)
                
                # Check for special tokens like <s> or <image>
                target_text = target.get("text", "")
                if target_text in ["<s>", "<pad>", "<bos>", "<eos>", "<image>"]:
                    special_token_indices.add(target_token_idx)
                
                if target_token_idx not in layer_tokens[layer_idx]:
                    layer_tokens[layer_idx].append(target_token_idx)
                
                # Create a unified node ID format for both targets and sources
                node_id = f"L{layer_idx}_T{target_token_idx}"
                
                # Get top prediction if available
                top_pred = ""
                logit_lens = layer_data.get("logit_lens_projections", {}).get(target_token_idx, {})
                if not logit_lens and isinstance(target_token_idx, int):
                    logit_lens = layer_data.get("logit_lens_projections", {}).get(str(target_token_idx), {})
                
                if logit_lens:
                    top_predictions = logit_lens.get("top_predictions", [])
                    if top_predictions and len(top_predictions) > 0:
                        pred_text = top_predictions[0].get('token_text', '')
                        # Escape special characters that might cause font rendering issues
                        pred_text = self._sanitize_text_for_display(pred_text)
                        top_pred = f"→{pred_text}"
                
                # Improved handling of special token text
                token_display_text = target.get("text", "")
                if token_display_text:
                    token_display_text = self._sanitize_text_for_display(token_display_text)
                    # Ensure special tokens display properly
                    if token_display_text in ["<s>", "<image>", "<pad>", "<bos>", "<eos>"]:
                        token_display_text = f"'{token_display_text}'"
                
                # Store node metadata
                node_metadata[node_id] = {
                    "type": target.get("type", 0),
                    "text": token_display_text,
                    "idx": target_token_idx,
                    "layer": layer_idx,
                    "weight": target.get("weight", 1.0),
                    "top_pred": top_pred,
                    "is_target": True
                }
                
                # Register this node in token_to_node lookup
                if target_token_idx not in token_to_node:
                    token_to_node[target_token_idx] = {}
                token_to_node[target_token_idx][layer_idx] = node_id
                
                # Process sources for this target, but only collect their info
                # Don't create nodes for them in the current layer yet
                for source in target.get("sources", []):
                    source_idx = source.get("index")
                    if source_idx is None:
                        continue
                    
                    # Just add to tracking collections for now
                    all_token_indices.add(source_idx)
                    
                    # Also track special tokens in sources
                    source_text = source.get("text", "")
                    if source_text in ["<s>", "<pad>", "<bos>", "<eos>", "<image>"]:
                        special_token_indices.add(source_idx)
        
        # PHASE 2: Now process source nodes, ensuring they come from previous layers
        for layer_idx in layers:
            layer_data = trace_results[layer_idx]
            prev_layers = [l for l in layers if l < layer_idx]
            
            # Process all target nodes in this layer again
            for target in layer_data.get("target_tokens", []):
                target_token_idx = target.get("index")
                if target_token_idx is None:
                    continue
                    
                # Process sources for this target
                for source in target.get("sources", []):
                    source_idx = source.get("index")
                    source_weight = source.get("scaled_weight", 0.0)
                    
                    if source_idx is None or source_weight < min_edge_weight:
                        continue
                        
                    # First look for this source token in previous layers
                    source_node_id = None
                    source_layer = None
                    
                    for prev_layer in reversed(prev_layers):  # Start from nearest previous layer
                        if prev_layer in token_to_node.get(source_idx, {}):
                            source_node_id = token_to_node[source_idx][prev_layer]
                            source_layer = prev_layer
                            break
                    
                    # If not found in previous layers and it's in the current layer's tokens,
                    # we need to make a special case to handle intra-layer references
                    if source_node_id is None and source_idx in layer_tokens[layer_idx]:
                        if debug_mode:
                            print(f"Warning: Source token {source_idx} only found in current layer {layer_idx}.")
                            
                        # We'll skip this connection since we want to avoid same-layer connections
                        continue
                    
                    # If still not found, this is a source without a previous appearance
                    # Skip it to avoid same-layer connections
                    if source_node_id is None:
                        if debug_mode:
                            print(f"Skipping orphaned source token {source_idx} with no previous appearance")
                        continue
                    
                    # Get source metadata from the previous layer where we found it
                    if source_node_id not in node_metadata:
                        source_text = "Unknown"  # Fallback
                        source_type = 0  # Default type
                        
                        # Try to get more info from the source layer's data
                        source_layer_data = trace_results.get(source_layer, {})
                        
                        # Look in targets from that layer
                        for old_target in source_layer_data.get("target_tokens", []):
                            if old_target.get("index") == source_idx:
                                source_text = old_target.get("text", "Unknown")
                                source_type = old_target.get("type", 0)
                                break
                        
                        # Improved handling of special token text
                        if source_text in ["<s>", "<image>", "<pad>", "<bos>", "<eos>"]:
                            source_text = f"'{source_text}'"
                        
                        # Get top prediction if available for the source
                        source_top_pred = ""
                        source_logit_lens = source_layer_data.get("logit_lens_projections", {}).get(source_idx, {})
                        if not source_logit_lens and isinstance(source_idx, int):
                            source_logit_lens = source_layer_data.get("logit_lens_projections", {}).get(str(source_idx), {})
                        
                        if source_logit_lens:
                            source_top_predictions = source_logit_lens.get("top_predictions", [])
                            if source_top_predictions and len(source_top_predictions) > 0:
                                pred_text = source_top_predictions[0].get('token_text', '')
                                # Sanitize the prediction text
                                pred_text = self._sanitize_text_for_display(pred_text)
                                source_top_pred = f"→{pred_text}"
                        
                        node_metadata[source_node_id] = {
                            "type": source_type,
                            "text": self._sanitize_text_for_display(source_text),
                            "idx": source_idx,
                            "layer": source_layer,
                            "weight": source.get("scaled_weight", 0.0),
                            "top_pred": source_top_pred
                        }
        
        # PHASE 3: Add nodes and edges to the graph
        for layer_idx in layers:
            current_layer = layer_idx
            prev_layers = [l for l in layers if l < current_layer]
            
            layer_data = trace_results[layer_idx]
            
            if debug_mode:
                print(f"\nDEBUG: Creating nodes and edges for layer {layer_idx}")
                print(f"Previous layers available: {prev_layers}")
            
            # Add nodes for all targets in this layer
            for target in layer_data.get("target_tokens", []):
                target_token_idx = target.get("index")
                if target_token_idx is None:
                    continue
                
                # Get target node ID
                target_node_id = token_to_node[target_token_idx].get(layer_idx)
                if not target_node_id:
                    continue
                
                # Add target node to graph
                G.add_node(target_node_id)
                
                if debug_mode:
                    print(f"Added target node: {target_node_id} for token {target_token_idx}")
                
                # Process each source for this target
                for source in target.get("sources", []):
                    source_idx = source.get("index")
                    source_weight = source.get("scaled_weight", 0.0)
                    
                    if source_idx is None or source_weight < min_edge_weight:
                        continue
                    
                    # Look for this source in previous layers only
                    source_node_id = None
                    for prev_layer in reversed(prev_layers):  # Start from nearest previous layer
                        if prev_layer in token_to_node.get(source_idx, {}):
                            source_node_id = token_to_node[source_idx][prev_layer]
                            break
                    
                    # Skip if no valid source node found in previous layers
                    if not source_node_id:
                        if debug_mode:
                            print(f"  No previous-layer node found for source token {source_idx}. Skipping.")
                        continue
                    
                    # Add source node to graph if not already present
                    if source_node_id not in G.nodes():
                        G.add_node(source_node_id)
                        if debug_mode:
                            print(f"  Added source node: {source_node_id} for token {source_idx}")
                    
                    # Add edge from source to target with weight
                    G.add_edge(
                        source_node_id,
                        target_node_id,
                        weight=source_weight,
                        saliency=source.get("saliency_score", 0.0)
                    )
                    
                    if debug_mode:
                        print(f"  Added edge: {source_node_id} -> {target_node_id} with weight {source_weight:.3f}")
        
        # Add special edges between the same token in adjacent layers - only if requested
        if show_continuation_edges:
            for token_idx in all_token_indices:
                token_layers = sorted(token_to_node.get(token_idx, {}).keys())
                
                for i in range(len(token_layers) - 1):
                    current_layer = token_layers[i]
                    next_layer = token_layers[i+1]
                    
                    # Only connect adjacent layers in our selected layers list
                    if current_layer in layers and next_layer in layers and layers.index(next_layer) == layers.index(current_layer) + 1:
                        src_node_id = token_to_node[token_idx][current_layer]
                        dst_node_id = token_to_node[token_idx][next_layer]
                        
                        # Only add if both nodes are in the graph
                        if src_node_id in G.nodes() and dst_node_id in G.nodes():
                            # Add a continuation edge with special weight
                            G.add_edge(
                                src_node_id,
                                dst_node_id,
                                weight=0.3,  # Reduced from 0.5 to make them less prominent
                                is_continuation=True
                            )
                            
                            if debug_mode:
                                print(f"Added continuation edge: {src_node_id} -> {dst_node_id}")
        
        if debug_mode:
            print(f"\nDEBUG: Graph construction complete")
            print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        
        # Remove orphaned nodes if requested
        if not show_orphaned_nodes:
            orphaned_nodes = [n for n in G.nodes() if G.degree(n) == 0]
            if orphaned_nodes:
                if debug_mode:
                    print(f"Removing {len(orphaned_nodes)} orphaned nodes with no connections")
                G.remove_nodes_from(orphaned_nodes)
        
        # Calculate positions for nodes
        pos = {}
        
        # Set spacing based on graph size
        # Calculate the number of nodes per layer for spacing adjustments
        nodes_per_layer = {}
        for node in G.nodes():
            layer = node_metadata[node]["layer"]
            if layer not in nodes_per_layer:
                nodes_per_layer[layer] = 0
            nodes_per_layer[layer] += 1
        
        max_nodes_in_layer = max(nodes_per_layer.values()) if nodes_per_layer else 1
        
        # Increased spacing to avoid node overlap
        x_spacing = 6.0  # Horizontal spacing between layers
        y_spacing = 3.0  # Vertical spacing between nodes within a layer
        
        # Get all layers present in the graph (some might have been removed)
        graph_layers = sorted(set(node_metadata[n]["layer"] for n in G.nodes()))
        
        # Organize nodes in strict columns by layer with enhanced spacing
        layer_nodes = {}
        
        # Group nodes by layer
        for node in G.nodes():
            layer = node_metadata[node]["layer"]
            if layer not in layer_nodes:
                layer_nodes[layer] = []
            layer_nodes[layer].append(node)
        
        # Position nodes with fixed x by layer, distributed y with increased spacing
        for layer_idx, layer_node_list in layer_nodes.items():
            # Layer-specific x-coordinate
            layer_x = graph_layers.index(layer_idx) * x_spacing
            
            layer_node_list.sort(key=lambda n: node_metadata[n]["idx"])
            # Distribute nodes vertically
            num_nodes = len(layer_node_list)
            if num_nodes > 0:    
                
                # Calculate vertical spacing with much more room between nodes
                total_height = (num_nodes - 1) * y_spacing
                start_y = -total_height / 2
                
                for i, node_id in enumerate(layer_node_list):
                    # Position with significantly increased vertical spacing
                    pos[node_id] = (layer_x, start_y + i * y_spacing)
        
        # Calculate figure size based on graph dimensions
        # Make figure much larger for better readability
        fig_width = len(graph_layers) * 3 + 3  # Width scales with number of layers
        fig_height = max(12, max_nodes_in_layer * 1.2)  # Height scales with max nodes in any layer
        
        # Create figure
        plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Set some better fonts that support special characters
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
        # Define colors for different token types with better contrast
        token_colors = {
            0: "#2ecc71",  # Generated = green
            1: "#3498db",  # Text = blue
            2: "#e74c3c"   # Image = red
        }
        
        # Calculate node sizes based on weights with exponential scaling if requested
        if use_variable_node_size:
            # Get all weights
            all_weights = [
                node_metadata.get(node, {}).get("weight", 1.0) for node in G.nodes()
            ]
            
            if all_weights:
                max_weight = max(all_weights)
                min_weight = min(all_weights)
                weight_range = max_weight - min_weight
                
                # Calculate size for each node
                node_sizes = []
                for node in G.nodes():
                    weight = node_metadata.get(node, {}).get("weight", 1.0)
                    
                    # Scale node size based on weight
                    if weight_range > 0:
                        # Normalized weight between 0 and 1
                        norm_weight = (weight - min_weight) / weight_range
                        
                        if use_exponential_scaling:
                            # Exponential scaling (stronger emphasis on higher weights)
                            # Using a power function with exponent 0.5 (square root is milder than quadratic)
                            # This makes small differences more visible
                            scaled_weight = math.pow(norm_weight, 0.5)
                            size = min_node_size + (max_node_size - min_node_size) * scaled_weight
                        else:
                            # Standard linear scaling
                            size = min_node_size + (max_node_size - min_node_size) * norm_weight
                    else:
                        size = (min_node_size + max_node_size) / 2
                    
                    node_sizes.append(size)
            else:
                node_sizes = [min_node_size] * len(G.nodes())
        else:
            # Use fixed size based on node type
            node_sizes = []
            for node in G.nodes():
                is_target = node_metadata.get(node, {}).get("is_target", False)
                size = max_node_size if is_target else min_node_size
                node_sizes.append(size)
        
        # Get node colors based on token type
        node_colors = [
            token_colors.get(node_metadata.get(node, {}).get("type", 0), "#7f8c8d") 
            for node in G.nodes()
        ]
        
        # Draw nodes with variable size
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Draw edges with width based on saliency/weight
        edge_width_multiplier = 5.0  # Controls overall edge thickness
        
        # Draw edges in two passes for better visualization
        # First a white "halo" for contrast
        for u, v, data in G.edges(data=True):
            if data.get('is_continuation', False):
                continue  # Skip continuation edges in this pass
                    
            weight = data.get("weight", 0.0)
            
            # Apply exponential scaling to edge widths if requested
            if use_exponential_scaling:
                # Using square root scaling for better visualization
                width = math.sqrt(weight) * edge_width_multiplier
            else:
                width = weight * edge_width_multiplier
                
            if width >= min_edge_weight * edge_width_multiplier:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=width + 1.5,  # Slightly wider for background
                    alpha=0.3,
                    edge_color='white'
                )
        
        # Then the actual edges with proper width and alpha
        for u, v, data in G.edges(data=True):
            if data.get('is_continuation', False):
                continue  # Skip continuation edges in this pass
                    
            weight = data.get("weight", 0.0)
            
            # Apply exponential scaling to edge widths if requested
            if use_exponential_scaling:
                # Using square root scaling for better visualization
                width = math.sqrt(weight) * edge_width_multiplier
            else:
                width = weight * edge_width_multiplier
                
            if width >= min_edge_weight * edge_width_multiplier:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[(u, v)],
                    width=max(width, 0.8),  # Set minimum width
                    alpha=min(0.8, max(0.2, weight * 1.5)),  # Scale alpha by weight
                    arrows=True,
                    arrowsize=10,
                    connectionstyle="arc3,rad=0.1"  # Slightly curved edges
                )
        
        # Draw continuation edges with dashed style (only if requested)
        if show_continuation_edges:
            continuation_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('is_continuation', False)]
            if continuation_edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=continuation_edges,
                    width=1.0,
                    alpha=0.5,
                    edge_color='gray',
                    style='dashed',
                    arrows=True,
                    arrowsize=8
                )
        
        # Create single-line labels: token_text and top_pred joined on one line
        labels = {}
        for node in G.nodes():
            meta = node_metadata.get(node, {})
            token_text = meta.get('text', '')
            top_pred   = meta.get('top_pred', '')
            if top_pred:
                # e.g. "Eiffel →tower"
                labels[node] = f"{token_text} {top_pred}"
            else:
                labels[node] = token_text
        
        # Calculate appropriate font sizes
        base_font_size = 10
        font_sizes = []
        for i, node in enumerate(G.nodes()):
            is_target = node_metadata.get(node, {}).get("is_target", False)
            if use_variable_node_size:
                font_size = base_font_size * (node_sizes[i] / min_node_size) ** 0.25
            else:
                font_size = base_font_size * 1.2 if is_target else base_font_size
            font_sizes.append(min(font_size, 12))
        
        # Draw each label as a single centered line
        for i, node in enumerate(G.nodes()):
            x, y   = pos[node]
            label  = labels.get(node, "")
            plt.text(
                x, y,
                label,
                fontsize=font_sizes[i],
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                zorder=100
            )
        
        # Add layer labels at the top
        for i, layer_idx in enumerate(graph_layers):
            layer_x = i * x_spacing
            max_y = max([y for _, y in pos.values()]) if pos else 0
            plt.text(layer_x, max_y + 1.5, f"Layer {layer_idx}", 
                    ha='center', va='center', fontsize=14, fontweight='bold')
        
        # Set title
        if title is None:
            title = f"Semantic Trace Flow Graph for Token '{target_text}' (idx: {target_idx})"
        plt.title(title, fontsize=16)
        
        # Add legend for token types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#3498db", markersize=10, label='Text Token'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#e74c3c", markersize=10, label='Image Token'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor="#2ecc71", markersize=10, label='Generated Token')
        ]
        
        if use_variable_node_size:
            # Add size legend
            legend_elements.extend([
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Low Influence'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='High Influence')
            ])
        
        # Add edge type legend (only if we're showing continuation edges)
        if show_continuation_edges:
            legend_elements.extend([
                plt.Line2D([0], [0], linestyle='-', color='black', linewidth=2, label='Token Influence'),
                plt.Line2D([0], [0], linestyle='--', color='gray', linewidth=1, label='Token Continuation')
            ])
        else:
            legend_elements.extend([
                plt.Line2D([0], [0], linestyle='-', color='black', linewidth=2, label='Token Influence')
            ])
        
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
        
        # Remove axis and ensure layout has proper padding
        plt.axis('off')
        plt.tight_layout(pad=3.0)  # Increased padding for better margins
        
        # Save in desired formats
        if output_format in ["png", "both"]:
            png_path = os.path.join(save_dir, f"flow_graph_{target_idx}.png")
            plt.savefig(png_path, dpi=dpi, bbox_inches='tight')
            saved_paths.append(png_path)
        
        if output_format in ["svg", "both"]:
            # Use non-font embedding for SVG for better compatibility
            mpl.rcParams['svg.fonttype'] = 'none'
            
            svg_path = os.path.join(save_dir, f"flow_graph_{target_idx}.svg")
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            saved_paths.append(svg_path)
        
        plt.close()
        
        print(f"Flow graph visualization saved to {', '.join(saved_paths)}")
        return saved_paths


    def _sanitize_text_for_display(self, text):
        """
        Sanitize text to avoid font rendering issues with special characters.
        
        Args:
            text: Input text that may contain special characters
            
        Returns:
            Sanitized text that should render properly in matplotlib
        """
        if not text:
            return ""
            
        # Special handling for common tokens
        if text in ["<s>", "<pad>", "<bos>", "<eos>", "<image>"]:
            return text
            
        # Replace common problematic characters
        replacements = {
            # Replace various quotes
            '"': '"', '"': '"', ''': "'", ''': "'",
            # Replace emoji and special symbols with simple alternatives
            '→': '->',
            '←': '<-',
            '⟨': '<',
            '⟩': '>',
            '⇒': '=>',
            '⇐': '<=',
            '≤': '<=',
            '≥': '>=',
            '…': '...',
        }
        
        # Apply replacements
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Filter out other potentially problematic characters
        result = ""
        for char in text:
            # Keep ASCII characters and common symbols
            if ord(char) < 128 or char in '°±²³½¼¾×÷':
                result += char
            else:
                # Replace other non-ASCII characters with a placeholder
                result += '·'  # Middle dot as placeholder
        
        return result
    
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
        return results