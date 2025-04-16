"""
Semantic tracing for VLMs: Combines saliency analysis with logit lens to track 
information flow through model layers, revealing how concepts evolve in the
model's reasoning process.
"""

import torch
import numpy as np
import gc
import os
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from tqdm.auto import tqdm
from collections import defaultdict

# Import components from existing modules
from analyzer.saliency import calculate_saliency_scores
from analyzer.logit_lens import LLaVANextLogitLensAnalyzer
from utils.hook_utils import GradientAttentionCapture
from utils.model_utils import get_llm_attention_layer_names
from utils.data_utils import find_token_indices


class SemanticTracer:
    """
    Traces information flow through VLM layers by combining saliency scores with
    logit lens projections to reveal how concepts evolve and flow from input tokens
    to generated tokens.
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
    ):
        """
        Initialize the semantic tracer.
        
        Args:
            model: The VLM model (LLaVA-Next)
            processor: The corresponding processor
            top_k: Number of top contributing tokens to track at each step
            device: Device to use (defaults to model's device)
            output_dir: Directory to save results
            cpu_offload: Whether to offload tensors to CPU when possible
            layer_batch_size: Number of layers to process at once for gradient computation
            logit_lens_concepts: List of concepts to track with logit lens
        """
        self.model = model
        self.processor = processor
        self.device = device or model.device
        self.top_k = top_k
        self.output_dir = output_dir
        self.cpu_offload = cpu_offload
        self.layer_batch_size = layer_batch_size
        
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
        
        print(f"Initialized SemanticTracer with {self.num_layers} attention layers")
        print(f"Using top_k={top_k} for tracing")
    
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
        text_indices, image_indices = find_token_indices(input_ids, self.image_token_id)
        
        # Add indices to input data
        input_data["text_indices"] = text_indices
        input_data["image_indices"] = image_indices
        
        return input_data
    
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
        
        # 2. Start recursive tracing
        print(f"\nStarting recursive semantic tracing for token '{token_text}' at position {target_token_idx}...")
        trace_results = self._recursive_trace(
            inputs=inputs,
            text_indices=input_data["text_indices"],
            image_indices=input_data["image_indices"],
            target_token_idx=target_token_idx,
            batch_compute=batch_compute,
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
    ) -> Dict[str, Any]:
        """
        Recursively trace token influence backward through layers.
        
        Args:
            inputs: Model inputs
            text_indices: Indices of text tokens
            image_indices: Indices of image tokens
            target_token_idx: Index of the target token
            batch_compute: Whether to compute saliency in batches
            
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
        
        for layer_idx in range(self.num_layers - 1, -1, -1):
            layer_name = self._get_layer_name(layer_idx)
            print(f"\nProcessing layer {layer_idx} ({layer_name})...")
            
            # 1. Compute saliency scores for current targets
            current_target_indices = list(current_targets.keys())
            layer_results = {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "target_tokens": [],
                "source_tokens": [],
                "logit_lens_projections": {},
            }
            
            all_saliency_maps = {}
            total_batches = len(current_target_indices)
            
            if batch_compute and total_batches > 1:
                batch_size = 1  # Process one target token at a time for memory efficiency
                num_batches = (total_batches + batch_size - 1) // batch_size
                
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, total_batches)
                    batch_target_indices = current_target_indices[start_idx:end_idx]
                    
                    for target_idx in tqdm(batch_target_indices, desc=f"Computing saliency for {len(batch_target_indices)} targets"):
                        # Process one target at a time
                        saliency_map = self._compute_saliency_for_token(
                            inputs=inputs,
                            layer_idx=layer_idx,
                            target_idx=target_idx
                        )
                        if saliency_map is not None:
                            all_saliency_maps[target_idx] = saliency_map
                        
                        # Clear cache after each target
                        if self.cpu_offload and torch.cuda.is_available():
                            torch.cuda.empty_cache()
            else:
                # Compute saliency for each target token individually 
                for target_idx in tqdm(current_target_indices, desc=f"Computing saliency"):
                    saliency_map = self._compute_saliency_for_token(
                        inputs=inputs,
                        layer_idx=layer_idx,
                        target_idx=target_idx
                    )
                    if saliency_map is not None:
                        all_saliency_maps[target_idx] = saliency_map
            
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
                    relative_weight = val / total_saliency if total_saliency > 0 else 1.0 / len(topk_indices)
                    # Scale by the target's weight to get global importance
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
                        "scaled_weight": scaled_weight
                    }
                    sources.append(source_info)
                
                # Record target token info
                target_info = {
                    "index": target_idx,
                    "id": all_token_ids[target_idx],
                    "text": all_token_texts[target_idx],
                    "type": token_types[target_idx].item(),
                    "weight": target_weight,
                    "sources": sources
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
                
                # Use the logit lens analyzer to project hidden states
                logit_lens_results = self._compute_logit_lens_projections(
                    inputs=inputs,
                    layer_idx=layer_idx,
                    token_indices=all_token_indices,
                )
                
                if logit_lens_results:
                    layer_results["logit_lens_projections"] = logit_lens_results
            
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
            
            # Normalize weights for new targets
            if new_targets:
                total_weight = sum(new_targets.values())
                current_targets = {idx: weight / total_weight for idx, weight in new_targets.items()}
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
        
        print("Semantic tracing complete.")
        return trace_results
    
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
        
        Args:
            inputs: Model inputs
            layer_idx: Index of the layer
            token_indices: List of token indices to analyze
            
        Returns:
            Dictionary mapping token indices to their projections
        """
        model = self.model
        
        try:
            # First, get hidden states for the layer
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
        Visualize the semantic tracing results.
        
        Args:
            results: Results from generate_and_analyze
            save_dir: Directory to save visualizations (defaults to self.output_dir)
            
        Returns:
            List of saved file paths
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        trace_results = results.get("trace_results", {})
        if not trace_results:
            print("No trace results to visualize.")
            return saved_paths
        
        # Basic info for labeling
        target_token = results.get("target_token", {})
        target_text = target_token.get("text", "unknown")
        target_idx = target_token.get("index", -1)
        
        # 1. Visualize the trace as a stacked bar chart
        try:
            self._visualize_source_distribution(trace_results, target_text, target_idx, save_dir, saved_paths)
        except Exception as e:
            print(f"Error visualizing source distribution: {e}")
        
        # 2. Visualize the semantic evolution using logit lens
        try:
            self._visualize_semantic_evolution(trace_results, target_text, target_idx, save_dir, saved_paths)
        except Exception as e:
            print(f"Error visualizing semantic evolution: {e}")
        
        # 3. Create a text report
        try:
            report_path = self._create_text_report(results, os.path.join(save_dir, "semantic_trace_report.txt"))
            saved_paths.append(report_path)
        except Exception as e:
            print(f"Error creating text report: {e}")
        
        return saved_paths
    
    def _visualize_source_distribution(
        self,
        trace_results: Dict[int, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str,
        saved_paths: List[str]
    ):
        """Visualize the distribution of source token types (text, image, generated) across layers"""
        layers = sorted(trace_results.keys())
        
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
        save_path = os.path.join(save_dir, f"token_type_distribution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        saved_paths.append(save_path)
        
        plt.close(fig)
    
    def _visualize_semantic_evolution(
        self,
        trace_results: Dict[int, Dict[str, Any]],
        target_text: str,
        target_idx: int,
        save_dir: str,
        saved_paths: List[str]
    ):
        """Visualize how concepts evolve across layers using logit lens projections"""
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
        save_path = os.path.join(save_dir, f"concept_evolution.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        saved_paths.append(save_path)
        
        plt.close(fig)