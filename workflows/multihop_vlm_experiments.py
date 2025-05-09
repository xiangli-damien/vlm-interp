"""
Multi-hop Reasoning Experiment for Vision-Language Models
=========================================================
Implements methodology adapted from Yang et al.'s paper "Do Large Language 
Models Latently Perform Multi-Hop Reasoning?" for the VLM domain.

This module investigates whether VLMs use multi-hop reasoning paths when 
answering questions that require intermediary reasoning steps like:
"Name one of the famous bridges on the river next to the landmark in the picture."

Key metrics:
1. Visual-ENTREC (V-ENTREC): Measures internal recall of bridge entities
2. Cross-modal Consistency: Tests if intervention along ∇V-ENTREC improves consistency
"""

import os
import json
import logging
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from tqdm import tqdm
from PIL import Image

from preprocess.input_builder import prepare_inputs
from runtime.hooks import TraceHookManager
from runtime.cache import TracingCache
from backends.logit_backend import LogitBackend
from runtime.model_utils import get_llm_attention_layer_names, load_model, get_module_by_name
from runtime.io import TraceIO

# Configure logging
logger = logging.getLogger("multihop_vlm")
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Data Structures
# ----------------------------------------------------------------------------

@dataclass
class TwoHopSample:
    """Single sample for visual multi-hop reasoning analysis."""
    image: str                      # Path or URL to image
    prompt_two_hop: str             # Prompt with indirect mention (e.g., "the river next to the landmark")
    prompt_one_hop: str             # Prompt with explicit mention (e.g., "the Seine River")
    prompt_two_hop_entity_sub: str  # Modified prompt with wrong entity
    prompt_two_hop_rel_sub: str     # Modified prompt with wrong relation
    bridge_entity: str              # Text of bridge entity (e.g., "Seine")
    bridge_entity_tokens: List[int] = field(default_factory=list)  # IDs of bridge entity tokens

    def __post_init__(self):
        """Validate sample fields."""
        if not self.image or not os.path.exists(self.image) and not self.image.startswith(("http://", "https://")):
            logger.warning(f"Image may not be accessible: {self.image}")


@dataclass
class ExperimentConfig:
    """Configuration for multi-hop experiment."""
    model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf"
    output_dir: str = "results/multihop_experiment"
    use_flash_attn: bool = False
    load_in_4bit: bool = False
    intervention_alpha: float = 1e-2  # Step size for gradient intervention
    cache_dir: Optional[str] = None
    device: str = "auto"  # "auto", "cuda", "cpu"


# ----------------------------------------------------------------------------
# Metrics Implementation
# ----------------------------------------------------------------------------

class MultihopMetrics:
    """Metrics for multi-hop reasoning analysis with corrected implementations."""

    @staticmethod
    def compute_entrec(vector: torch.Tensor, lm_head, token_ids: Union[int, List[int]]) -> torch.Tensor:
        """
        Calculate ENTREC for a hidden vector with support for multi-token entities.
        
        Args:
            vector: Hidden vector from intermediate layer
            lm_head: Language model head for projection
            token_ids: Token ID or list of token IDs for the bridge entity
            
        Returns:
            Log probability (average across multiple tokens if provided)
        """
        # Project hidden vector through LM head
        logits = lm_head(vector)  # [vocab_size]
        
        # Convert to log probabilities
        logp = F.log_softmax(logits, dim=-1)
        
        # Handle both single token ID and list of token IDs
        if isinstance(token_ids, list):
            if len(token_ids) > 1:
                # For multiple tokens, average the log probabilities
                return torch.mean(torch.tensor([logp[token_id] for token_id in token_ids]))
            elif len(token_ids) == 1:
                # For a list with one token, use that token
                return logp[token_ids[0]]
            else:
                # Handle empty list case
                logger.warning("Empty token_ids list provided, using UNK token")
                return logp[0]  # Use ID 0 (typically UNK token) as fallback
        else:
            # Handle case where token_ids is a single integer
            return logp[token_ids]

    @staticmethod
    def consistency_score(p_two_hop: torch.Tensor, p_one_hop: torch.Tensor) -> torch.Tensor:
        """
        Calculate correctly implemented symmetric cross-entropy between distributions.
        
        Args:
            p_two_hop: Output distribution from two-hop prompt (log-probs or probs)
            p_one_hop: Output distribution from one-hop prompt (log-probs or probs)
            
        Returns:
            Consistency score using symmetric cross-entropy
        """
        # Ensure inputs are probabilities, not logits
        if p_two_hop.dim() > 1 and p_two_hop.size(0) == 1:
            p_two_hop = p_two_hop.squeeze(0)
        if p_one_hop.dim() > 1 and p_one_hop.size(0) == 1:
            p_one_hop = p_one_hop.squeeze(0)
            
        # If inputs are logits, convert to probabilities
        if not torch.all((p_two_hop >= 0) & (p_two_hop <= 1)):
            p_two_hop = F.softmax(p_two_hop, dim=-1)
        if not torch.all((p_one_hop >= 0) & (p_one_hop <= 1)):
            p_one_hop = F.softmax(p_one_hop, dim=-1)
            
        # Compute symmetric KL divergence / cross-entropy
        # Explicitly using definition to avoid type errors with F.cross_entropy
        p_two_hop_log = torch.log(p_two_hop + 1e-10)  # Add epsilon to avoid log(0)
        p_one_hop_log = torch.log(p_one_hop + 1e-10)
        
        # KL(p_two || p_one) + KL(p_one || p_two)
        kl_two_one = -torch.sum(p_two_hop * p_one_hop_log)
        kl_one_two = -torch.sum(p_one_hop * p_two_hop_log)
        
        # Return negative average (higher is more consistent)
        return -(kl_two_one + kl_one_two) / 2.0


# ----------------------------------------------------------------------------
# Core Experiment Runner
# ----------------------------------------------------------------------------

class MultihopVLMExperiment:
    """Main class for running multi-hop VLM experiments."""
    
    def __init__(
        self, 
        model, 
        processor, 
        config: ExperimentConfig,
        device: torch.device
    ):
        """
        Initialize the experiment.
        
        Args:
            model: VLM model (e.g., LLaVA)
            processor: Model processor/tokenizer
            config: Experiment configuration
            device: Computation device
        """
        self.model = model
        self.processor = processor
        self.config = config
        self.device = device
        self.io = TraceIO(config.output_dir)
        
        # Extract number of layers
        if hasattr(model.config, "num_hidden_layers"):
            self.num_layers = model.config.num_hidden_layers
        else:
            # Estimate from model structure
            self.num_layers = self._estimate_num_layers()
        
        # Layers to analyze (all of them)
        self.layers = list(range(self.num_layers))
        logger.info(f"Will analyze {len(self.layers)} layers (0-{self.num_layers-1})")
        
        # Initialize hook manager
        self.hooks = TraceHookManager(model, cpu_offload=True)
        
        # Register hooks for all layers
        self._register_hooks()
        
        # Find LM head
        self.lm_head = self._find_lm_head()
    
    def _estimate_num_layers(self) -> int:
        """Estimate number of layers from model structure."""
        # Check various common model architectures
        if hasattr(self.model, "language_model"):
            if hasattr(self.model.language_model, "model") and hasattr(self.model.language_model.model, "layers"):
                return len(self.model.language_model.model.layers)
            elif hasattr(self.model.language_model, "layers"):
                return len(self.model.language_model.layers)
        
        # Default estimate
        logger.warning("Could not determine number of layers, using default of 32")
        return 32
    
    def _find_lm_head(self) -> torch.nn.Module:
        """Find language model head in the model."""
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "lm_head"):
            logger.info("Found LM head at model.language_model.lm_head")
            return self.model.language_model.lm_head
        elif hasattr(self.model, "lm_head"):
            logger.info("Found LM head at model.lm_head")
            return self.model.lm_head
        else:
            logger.error("Could not find LM head, experiment will fail")
            raise ValueError("LM head not found in model")
    
    def _register_hooks(self):
        """Register hooks for all attention layers."""
        attn_layer_names = get_llm_attention_layer_names(self.model)
        logger.info(f"Found {len(attn_layer_names)} attention layer names")
        
        # Register each attention layer
        for idx, name in enumerate(attn_layer_names):
            if idx in self.layers:
                # Capture both hidden states and gradients for intervention tests
                self.hooks.add_layer(name, capture=["hidden", "attention", "grad"], layer_idx=idx)
        
        # Install hooks
        num_installed = self.hooks.install()
        logger.info(f"Installed {num_installed} hooks")
    
    def _tokenize_bridge_entity(self, sample: TwoHopSample) -> List[int]:
        """
        Tokenize bridge entity and handle multi-token entities properly.
        
        Args:
            sample: Two-hop sample to update
            
        Returns:
            List of token IDs for bridge entity
        """
        # First try with space prefix for better subword handling
        space_prefix = " " + sample.bridge_entity if not sample.bridge_entity.startswith(" ") else sample.bridge_entity
        bridge_tokens = self.processor.tokenizer.encode(space_prefix, add_special_tokens=False)
        
        # If that returned nothing, try without space
        if not bridge_tokens:
            bridge_tokens = self.processor.tokenizer.encode(sample.bridge_entity, add_special_tokens=False)
        
        # If still nothing, try various casing
        if not bridge_tokens:
            variants = [
                sample.bridge_entity.lower(),
                sample.bridge_entity.upper(),
                sample.bridge_entity.title()
            ]
            for variant in variants:
                bridge_tokens = self.processor.tokenizer.encode(variant, add_special_tokens=False)
                if bridge_tokens:
                    break
        
        # Ensure we always have at least one token ID
        if not bridge_tokens:
            logger.warning(f"Could not tokenize '{sample.bridge_entity}', using default token")
            bridge_tokens = [0]  # Use UNK token as fallback
        
        # Update sample - always store as a list
        sample.bridge_entity_tokens = bridge_tokens
        
        # Log tokenization details
        token_texts = [self.processor.tokenizer.decode([token]) for token in bridge_tokens]
        logger.debug(f"Bridge entity '{sample.bridge_entity}' tokenized as {bridge_tokens}")
        logger.debug(f"Token texts: {token_texts}")
        
        return bridge_tokens
    
    def _locate_descriptor_token(self, inputs: Dict[str, Any]) -> int:
        """
        Locate the token index for the end of the descriptor phrase.
        
        Args:
            inputs: Model inputs with tokenized prompt
            
        Returns:
            Token index of the last token in the descriptor phrase
        """
        # For now, we use a simple heuristic: the last text token
        # before the first image token or the end of the sequence
        text_indices = inputs.get("text_indices", [])
        if len(text_indices) == 0:
            return 0
        
        # Use the last text token
        return text_indices[-1].item()
    
    def _forward(self, image, prompt: str) -> Tuple[Dict[str, Any], Any]:
        """
        Run forward pass through the model.
        
        Args:
            image: Image to process
            prompt: Text prompt
            
        Returns:
            Tuple of (prepared inputs, model outputs)
        """
        # Prepare inputs
        inputs = prepare_inputs(
            model=self.model,
            processor=self.processor,
            image=image,
            prompt=prompt
        )
        
        # Run model without gradients
        with torch.no_grad():
            self.hooks.cache.clear()
            outputs = self.hooks.run(inputs["inputs"])
            
        return inputs, outputs
    
    def run_first_hop_test(self, sample: TwoHopSample) -> Dict[int, Dict[str, bool]]:
        """
        Run first hop test using entity and relation substitution.
        
        Args:
            sample: Two-hop sample to test
            
        Returns:
            Dictionary of {layer_idx: {"entity_sub": bool, "rel_sub": bool}}
        """
        # Ensure bridge entity is tokenized
        if not sample.bridge_entity_tokens:
            self._tokenize_bridge_entity(sample)
        
        # Get the first bridge token ID, ensuring it's handled correctly whether 
        # bridge_entity_tokens is a list or single integer
        bridge_token_id = sample.bridge_entity_tokens[0] if isinstance(sample.bridge_entity_tokens, list) else sample.bridge_entity_tokens
        
        results = {layer: {"entity_sub": False, "rel_sub": False} for layer in self.layers}
        
        # Capture ENTREC for original and substituted prompts
        def get_entrec_values(prompt: str) -> Dict[int, float]:
            """Get ENTREC values for all layers for a prompt."""
            inputs, _ = self._forward(sample.image, prompt)
            descriptor_idx = self._locate_descriptor_token(inputs)
            entrec_values = {}
            
            for layer in self.layers:
                # Get hidden state for this layer
                hidden = self.hooks.cache.get(layer, "hidden", self.device)
                if hidden is None:
                    logger.warning(f"No hidden state for layer {layer}")
                    entrec_values[layer] = float('-inf')
                    continue
                
                # Get vector for descriptor token
                vector = hidden[:, descriptor_idx, :].squeeze(0)
                
                # Compute ENTREC - pass bridge_token_id directly for consistent handling
                entrec = MultihopMetrics.compute_entrec(vector, self.lm_head, bridge_token_id)
                entrec_values[layer] = entrec.item()
            
            return entrec_values
        
        # Get ENTREC values for all three prompts
        entrec_orig = get_entrec_values(sample.prompt_two_hop)
        entrec_entity_sub = get_entrec_values(sample.prompt_two_hop_entity_sub)
        entrec_rel_sub = get_entrec_values(sample.prompt_two_hop_rel_sub)
        
        # Compare values
        for layer in self.layers:
            # Entity substitution test: original should have higher ENTREC than entity-sub
            results[layer]["entity_sub"] = entrec_orig[layer] > entrec_entity_sub[layer]
            
            # Relation substitution test: original should have higher ENTREC than rel-sub
            results[layer]["rel_sub"] = entrec_orig[layer] > entrec_rel_sub[layer]
        
        return results
    
    def run_second_hop_test(self, sample: TwoHopSample) -> Dict[int, bool]:
        """
        Run correctly implemented second hop test using gradient intervention and consistency.
        
        Args:
            sample: Two-hop sample to test
            
        Returns:
            Dictionary of {layer_idx: intervention_successful}
        """
        # Ensure bridge entity is tokenized
        if not sample.bridge_entity_tokens:
            self._tokenize_bridge_entity(sample)
        
        # Fix: Properly handle bridge_token_id whether it's a list or integer
        bridge_token_id = sample.bridge_entity_tokens[0] if isinstance(sample.bridge_entity_tokens, list) else sample.bridge_entity_tokens
        results = {layer: False for layer in self.layers}
        
        # 1. First get the one-hop results as reference
        one_hop_inputs = prepare_inputs(
            model=self.model,
            processor=self.processor,
            image=sample.image,
            prompt=sample.prompt_one_hop
        )
        
        with torch.no_grad():
            one_hop_outputs = self.model(**one_hop_inputs["inputs"], return_dict=True)
            one_hop_probs = F.softmax(one_hop_outputs.logits[:, -1, :], dim=-1)
        
        # 2. Prepare inputs with two-hop prompt
        two_hop_inputs = prepare_inputs(
            model=self.model,
            processor=self.processor,
            image=sample.image,
            prompt=sample.prompt_two_hop
        )
        
        # Locate descriptor token (more robust method)
        descriptor_idx = self._locate_descriptor_token(two_hop_inputs)
        
        # Run model with gradients enabled
        self.model.zero_grad(set_to_none=True)
        self.hooks.cache.clear()
        
        # Important: Set detach_after_forward=False when creating hooks
        self.hooks._detach_after_forward = False
        
        for layer in self.layers:
            try:
                with torch.set_grad_enabled(True):
                    # 3. Forward pass with original inputs
                    two_hop_outputs = self.hooks.run(two_hop_inputs["inputs"])
                    two_hop_probs = F.softmax(two_hop_outputs.logits[:, -1, :], dim=-1)
                    
                    # Calculate baseline consistency
                    baseline_consistency = MultihopMetrics.consistency_score(
                        two_hop_probs, 
                        one_hop_probs
                    )
                    
                    # Get hidden state for this layer
                    hidden = self.hooks.cache.get(layer, "hidden", self.device)
                    if hidden is None:
                        logger.warning(f"No hidden state for layer {layer}")
                        continue
                    
                    # Get vector for descriptor token and ensure it has gradients
                    vector = hidden[:, descriptor_idx, :].clone()
                    
                    # Must use original tensor that's connected to graph
                    # Cannot use .clone() or .detach() here
                    if not vector.requires_grad:
                        logger.warning(f"Layer {layer} hidden states don't have requires_grad=True")
                        vector.requires_grad_(True)
                    
                    # 4. Compute ENTREC - Fix: Use bridge_token_id (singular) here
                    entrec = MultihopMetrics.compute_entrec(
                        vector.squeeze(0), 
                        self.lm_head, 
                        bridge_token_id  # Fixed: was bridge_token_ids (plural)
                    )
                    
                    # 5. Compute gradient
                    self.model.zero_grad(set_to_none=True)
                    entrec.backward(retain_graph=True)
                    
                    # Check if we got gradients
                    if vector.grad is None:
                        logger.warning(f"No gradient for layer {layer}, skipping")
                        continue
                        
                    grad = vector.grad.clone()
                    
                    # 6. Perform intervention
                    alpha = self.config.intervention_alpha
                    perturbed_vector = vector + alpha * grad
                    
                    # 7. Re-run model with perturbed hidden state
                    # This is tricky - we need to patch the model to use our perturbed hidden state
                    # at the specific layer and position
                    def get_modified_forward(orig_module, layer_idx, pos_idx, new_vector):
                        """Create a modified forward function that injects our vector."""
                        def modified_forward(*args, **kwargs):
                            outputs = orig_module(*args, **kwargs)
                            # Only modify the target position
                            if isinstance(outputs, torch.Tensor):
                                modified = outputs.clone()
                                modified[:, pos_idx, :] = new_vector
                                return modified
                            else:
                                # Handle tuple outputs (common in attention blocks)
                                if len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                                    result_list = list(outputs)
                                    modified = result_list[0].clone()
                                    modified[:, pos_idx, :] = new_vector
                                    result_list[0] = modified
                                    return tuple(result_list)
                            return outputs
                        return modified_forward
                    
                    # Locate the module for this layer
                    layer_name = self._idx_to_name.get(layer)
                    if not layer_name:
                        logger.warning(f"No layer name found for layer {layer}")
                        continue
                        
                    module = get_module_by_name(self.model, layer_name)
                    
                    # Temporarily replace forward method
                    orig_forward = module.forward
                    module.forward = get_modified_forward(
                        orig_forward, layer, descriptor_idx, perturbed_vector
                    )
                    
                    # 8. Run model with intervention
                    with torch.no_grad():  # No need for gradients in intervention test
                        perturbed_outputs = self.model(**two_hop_inputs["inputs"], return_dict=True)
                        perturbed_probs = F.softmax(perturbed_outputs.logits[:, -1, :], dim=-1)
                    
                    # 9. Restore original forward method
                    module.forward = orig_forward
                    
                    # 10. Calculate new consistency
                    perturbed_consistency = MultihopMetrics.consistency_score(
                        perturbed_probs, 
                        one_hop_probs
                    )
                    
                    # 11. Check if intervention improves consistency
                    # This is the key test for second hop reasoning
                    results[layer] = (perturbed_consistency > baseline_consistency).item()
                    
                    # Also log the consistency improvement
                    logger.debug(f"Layer {layer}: Consistency {baseline_consistency.item():.4f} → "
                            f"{perturbed_consistency.item():.4f}, "
                            f"Success: {results[layer]}")
                    
            except Exception as e:
                logger.error(f"Error in layer {layer} intervention test: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return results
    
    def run_combined_test(self, sample: TwoHopSample) -> Dict[str, Dict[int, Any]]:
        """
        Run both first and second hop tests, plus get stats for full traversal.
        
        Args:
            sample: Two-hop sample to test
            
        Returns:
            Dictionary of combined results
        """
        # Run first hop test
        first_hop_results = self.run_first_hop_test(sample)
        
        # Run second hop test
        second_hop_results = self.run_second_hop_test(sample)
        
        # Calculate combined results for full traversal
        combined_results = {}
        for layer in self.layers:
            # Full traversal: entity_sub test AND second hop successful
            entity_traversal = (
                first_hop_results[layer]["entity_sub"] and 
                second_hop_results[layer]
            )
            
            # Full traversal: rel_sub test AND second hop successful
            rel_traversal = (
                first_hop_results[layer]["rel_sub"] and 
                second_hop_results[layer]
            )
            
            combined_results[layer] = {
                "entity_traversal": entity_traversal,
                "rel_traversal": rel_traversal
            }
        
        return {
            "first_hop": first_hop_results,
            "second_hop": second_hop_results,
            "combined": combined_results
        }
    
    def run_experiment_batch(self, samples: List[TwoHopSample]) -> Dict[str, Any]:
        """
        Run experiment on a batch of samples and collect statistics.
        
        Args:
            samples: List of two-hop samples
            
        Returns:
            Dictionary of experiment results
        """
        # Initialize counters
        results = {
            "first_hop": {
                layer: {"entity_sub": 0, "rel_sub": 0, "total": 0} 
                for layer in self.layers
            },
            "second_hop": {
                layer: {"success": 0, "total": 0} 
                for layer in self.layers
            },
            "combined": {
                layer: {"entity_traversal": 0, "rel_traversal": 0, "total": 0} 
                for layer in self.layers
            },
            "samples": []
        }
        
        # Process each sample
        for idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
            try:
                # Run all tests
                sample_results = self.run_combined_test(sample)
                
                # Store per-sample results
                results["samples"].append({
                    "index": idx,
                    "bridge_entity": sample.bridge_entity,
                    "results": sample_results
                })
                
                # Update statistics
                for layer in self.layers:
                    # First hop stats
                    if sample_results["first_hop"][layer]["entity_sub"]:
                        results["first_hop"][layer]["entity_sub"] += 1
                    if sample_results["first_hop"][layer]["rel_sub"]:
                        results["first_hop"][layer]["rel_sub"] += 1
                    results["first_hop"][layer]["total"] += 1
                    
                    # Second hop stats
                    if sample_results["second_hop"][layer]:
                        results["second_hop"][layer]["success"] += 1
                    results["second_hop"][layer]["total"] += 1
                    
                    # Combined stats
                    if sample_results["combined"][layer]["entity_traversal"]:
                        results["combined"][layer]["entity_traversal"] += 1
                    if sample_results["combined"][layer]["rel_traversal"]:
                        results["combined"][layer]["rel_traversal"] += 1
                    results["combined"][layer]["total"] += 1
            
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                continue
        
        # Calculate aggregate statistics
        results["aggregate"] = self._calculate_aggregate_stats(results)
        
        return results
    
    def _calculate_aggregate_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate aggregate statistics from raw results.
        
        Args:
            results: Raw experiment results
            
        Returns:
            Dictionary of aggregate statistics
        """
        aggregate = {
            "first_hop": {
                "entity_sub": {layer: 0.0 for layer in self.layers},
                "rel_sub": {layer: 0.0 for layer in self.layers},
                "max_layer_entity": 0,
                "max_layer_rel": 0,
                "max_value_entity": 0.0,
                "max_value_rel": 0.0
            },
            "second_hop": {
                "success": {layer: 0.0 for layer in self.layers},
                "max_layer": 0,
                "max_value": 0.0
            },
            "combined": {
                "entity_traversal": {layer: 0.0 for layer in self.layers},
                "rel_traversal": {layer: 0.0 for layer in self.layers},
                "max_layer_entity": 0,
                "max_layer_rel": 0,
                "max_value_entity": 0.0,
                "max_value_rel": 0.0
            }
        }
        
        # Calculate percentages for first hop
        for layer in self.layers:
            # Entity substitution
            if results["first_hop"][layer]["total"] > 0:
                aggregate["first_hop"]["entity_sub"][layer] = (
                    results["first_hop"][layer]["entity_sub"] / 
                    results["first_hop"][layer]["total"]
                )
            
            # Relation substitution
            if results["first_hop"][layer]["total"] > 0:
                aggregate["first_hop"]["rel_sub"][layer] = (
                    results["first_hop"][layer]["rel_sub"] / 
                    results["first_hop"][layer]["total"]
                )
        
        # Find max layers for first hop
        max_entity_layer = max(
            self.layers, 
            key=lambda l: aggregate["first_hop"]["entity_sub"][l]
        )
        max_rel_layer = max(
            self.layers, 
            key=lambda l: aggregate["first_hop"]["rel_sub"][l]
        )
        
        aggregate["first_hop"]["max_layer_entity"] = max_entity_layer
        aggregate["first_hop"]["max_value_entity"] = aggregate["first_hop"]["entity_sub"][max_entity_layer]
        aggregate["first_hop"]["max_layer_rel"] = max_rel_layer
        aggregate["first_hop"]["max_value_rel"] = aggregate["first_hop"]["rel_sub"][max_rel_layer]
        
        # Calculate percentages for second hop
        for layer in self.layers:
            if results["second_hop"][layer]["total"] > 0:
                aggregate["second_hop"]["success"][layer] = (
                    results["second_hop"][layer]["success"] / 
                    results["second_hop"][layer]["total"]
                )
        
        # Find max layer for second hop
        max_second_layer = max(
            self.layers, 
            key=lambda l: aggregate["second_hop"]["success"][l]
        )
        
        aggregate["second_hop"]["max_layer"] = max_second_layer
        aggregate["second_hop"]["max_value"] = aggregate["second_hop"]["success"][max_second_layer]
        
        # Calculate percentages for combined
        for layer in self.layers:
            if results["combined"][layer]["total"] > 0:
                aggregate["combined"]["entity_traversal"][layer] = (
                    results["combined"][layer]["entity_traversal"] / 
                    results["combined"][layer]["total"]
                )
                aggregate["combined"]["rel_traversal"][layer] = (
                    results["combined"][layer]["rel_traversal"] / 
                    results["combined"][layer]["total"]
                )
        
        # Find max layers for combined
        max_entity_traversal_layer = max(
            self.layers, 
            key=lambda l: aggregate["combined"]["entity_traversal"][l]
        )
        max_rel_traversal_layer = max(
            self.layers, 
            key=lambda l: aggregate["combined"]["rel_traversal"][l]
        )
        
        aggregate["combined"]["max_layer_entity"] = max_entity_traversal_layer
        aggregate["combined"]["max_value_entity"] = aggregate["combined"]["entity_traversal"][max_entity_traversal_layer]
        aggregate["combined"]["max_layer_rel"] = max_rel_traversal_layer
        aggregate["combined"]["max_value_rel"] = aggregate["combined"]["rel_traversal"][max_rel_traversal_layer]
        
        return aggregate
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Generate visualizations from experiment results.
        
        Args:
            results: Experiment results
            output_dir: Directory to save visualizations
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot first hop results
        plt.figure(figsize=(12, 7))
        plt.plot(
            self.layers,
            [results["aggregate"]["first_hop"]["entity_sub"][l] for l in self.layers],
            'b-', label='Entity Substitution'
        )
        plt.plot(
            self.layers,
            [results["aggregate"]["first_hop"]["rel_sub"][l] for l in self.layers],
            'r-', label='Relation Substitution'
        )
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Layer Index')
        plt.ylabel('Relative Frequency of ENTREC Increase')
        plt.title('First Hop: Entity Recall Test')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'first_hop_results.png'), dpi=300)
        
        # Plot second hop results
        plt.figure(figsize=(12, 7))
        plt.plot(
            self.layers,
            [results["aggregate"]["second_hop"]["success"][l] for l in self.layers],
            'g-', label='Gradient Intervention Success'
        )
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Layer Index')
        plt.ylabel('Relative Frequency of ENTREC Increase')
        plt.title('Second Hop: Consistency Test')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'second_hop_results.png'), dpi=300)
        
        # Plot combined results
        plt.figure(figsize=(12, 7))
        plt.plot(
            self.layers,
            [results["aggregate"]["combined"]["entity_traversal"][l] for l in self.layers],
            'b-', label='Entity Traversal'
        )
        plt.plot(
            self.layers,
            [results["aggregate"]["combined"]["rel_traversal"][l] for l in self.layers],
            'r-', label='Relation Traversal'
        )
        plt.axhline(y=0.25, color='gray', linestyle='--', alpha=0.7, label='Random Chance (25%)')
        plt.xlabel('Layer Index')
        plt.ylabel('Relative Frequency of Full Traversal')
        plt.title('Combined: Multi-hop Reasoning Test')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'combined_results.png'), dpi=300)
        
        # Save results as JSON
        with open(os.path.join(output_dir, 'experiment_results.json'), 'w') as f:
            # Filter out tensors and non-serializable objects
            filtered_results = self._make_serializable(results)
            json.dump(filtered_results, f, indent=2)
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Make an object JSON-serializable by recursively converting non-serializable elements.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            Serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
        
    def analyze_case_study(self, sample: TwoHopSample) -> Dict[str, Any]:
        """
        Run a detailed analysis on a single sample to explore the multi-hop reasoning pathway.
        
        Args:
            sample: Two-hop sample to analyze
            
        Returns:
            Dictionary of detailed analysis results
        """
        # Ensure bridge entity is tokenized
        if not sample.bridge_entity_tokens:
            self._tokenize_bridge_entity(sample)
        
        # Token ID of bridge entity
        bridge_token_id = sample.bridge_entity_tokens[0]
        
        # Run first and second hop tests
        first_hop_results = self.run_first_hop_test(sample)
        second_hop_results = self.run_second_hop_test(sample)
        
        # Get layer-by-layer ENTREC values for all three prompts
        def get_detailed_entrec(prompt: str) -> Dict[int, float]:
            inputs, _ = self._forward(sample.image, prompt)
            descriptor_idx = self._locate_descriptor_token(inputs)
            
            # Collect ENTREC values for all layers
            entrec_values = {}
            for layer in self.layers:
                hidden = self.hooks.cache.get(layer, "hidden", self.device)
                if hidden is None:
                    entrec_values[layer] = float('-inf')
                    continue
                    
                vector = hidden[:, descriptor_idx, :].squeeze(0)
                entrec = MultihopMetrics.compute_entrec(vector, self.lm_head, bridge_token_id)
                entrec_values[layer] = entrec.item()
                
            return entrec_values
        
        # Get ENTREC values for all three prompts
        entrec_orig = get_detailed_entrec(sample.prompt_two_hop)
        entrec_entity_sub = get_detailed_entrec(sample.prompt_two_hop_entity_sub)
        entrec_rel_sub = get_detailed_entrec(sample.prompt_two_hop_rel_sub)
        
        # Get token decoder for readable output
        from runtime.decode import TokenDecoder
        decoder = TokenDecoder(self.processor.tokenizer)
        
        # Get top K predictions at descriptor token for most interesting layers
        def get_top_predictions(prompt: str, layer_idx: int, k: int = 5) -> List[Dict[str, Any]]:
            inputs, _ = self._forward(sample.image, prompt)
            descriptor_idx = self._locate_descriptor_token(inputs)
            
            # Get hidden state
            hidden = self.hooks.cache.get(layer_idx, "hidden", self.device)
            if hidden is None:
                return []
                
            vector = hidden[:, descriptor_idx, :].squeeze(0)
            
            # Project through LM head
            with torch.no_grad():
                logits = self.lm_head(vector)
                probs = F.softmax(logits, dim=-1)
                
                # Get top K
                topk_probs, topk_indices = torch.topk(probs, k=k)
                
                # Create readable predictions
                predictions = []
                for i in range(k):
                    token_id = topk_indices[i].item()
                    token_text = decoder.decode_token(token_id)
                    predictions.append({
                        "token_id": token_id,
                        "token_text": token_text,
                        "probability": topk_probs[i].item()
                    })
                
                return predictions
        
        # Get best and worst layers for first hop
        best_layer = max(self.layers, key=lambda l: entrec_orig[l] - entrec_entity_sub[l])
        worst_layer = min(self.layers, key=lambda l: entrec_orig[l] - entrec_entity_sub[l])
        
        # Get predictions for best and worst layers
        best_layer_preds = get_top_predictions(sample.prompt_two_hop, best_layer)
        worst_layer_preds = get_top_predictions(sample.prompt_two_hop, worst_layer)
        
        # Combine everything into detailed analysis
        detailed_analysis = {
            "bridge_entity": sample.bridge_entity,
            "bridge_token_id": bridge_token_id,
            "first_hop": first_hop_results,
            "second_hop": second_hop_results,
            "entrec_values": {
                "original": entrec_orig,
                "entity_sub": entrec_entity_sub,
                "rel_sub": entrec_rel_sub
            },
            "layer_analysis": {
                "best_layer": {
                    "index": best_layer,
                    "entrec_diff": entrec_orig[best_layer] - entrec_entity_sub[best_layer],
                    "predictions": best_layer_preds
                },
                "worst_layer": {
                    "index": worst_layer,
                    "entrec_diff": entrec_orig[worst_layer] - entrec_entity_sub[worst_layer],
                    "predictions": worst_layer_preds
                }
            }
        }
        
        return detailed_analysis