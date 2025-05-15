# workflows/semantic_tracing.py
"""
Semantic Tracing implementation for Vision-Language Models.

Combines saliency analysis with logit lens to track information flow through model layers,
revealing how concepts evolve in the model's reasoning process.
"""

import torch
import pandas as pd
import os
import gc
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Set

from backends.saliency_backend import SaliencyBackend
from backends.attention_backend import AttentionBackend
from backends.logit_backend import LogitLensBackend
from runtime.selection import SelectionConfig, SelectionStrategy
from runtime.io import TraceIO
from runtime.decode import TokenDecoder
from runtime.generation import GenerationMixin
from preprocess.mapper import VisionMapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SemanticTracer(GenerationMixin):
    """
    Traces information flow through VLM layers by combining saliency scores or attention maps with
    logit lens projections to reveal how concepts evolve from input tokens to generated tokens.
    
    Key features:
    - Multiple tracing modes (saliency, attention, or both)
    - Coverage-based node selection for efficient information flow visualization
    - Token-by-token tracing with logit lens projections for concept tracking
    - Memory-optimized implementation with batch processing options
    """
    
    def __init__(
        self,
        model,
        processor,
        output_dir: str = "semantic_tracing_results",
        selection_config: Optional[SelectionConfig] = None,
        logit_lens_concepts: Optional[List[str]] = None,
        device: Optional[str] = None,
        cpu_offload: bool = True,
        layer_batch_size: int = 2,
        normalize_weights: bool = True,
        debug: bool = False,
        epsilon: float = 1e-7
    ):
        """
        Initialize the semantic tracer with model and configuration.
        
        Args:
            model: The VLM model (e.g., LLaVA-Next)
            processor: The corresponding processor
            output_dir: Directory to save results
            selection_config: Configuration for node selection
            logit_lens_concepts: Concepts to track with logit lens
            device: Device to use (defaults to model's device)
            cpu_offload: Whether to offload tensors to CPU when possible
            layer_batch_size: Layers to process at once for gradient computation
            normalize_weights: Whether to normalize token importance weights
            debug: Whether to print additional debug information
            epsilon: Small value for numerical stability
        """
        self.model = model
        self.processor = processor
        self.device = device or next(model.parameters()).device
        self.output_dir = output_dir
        self.cpu_offload = cpu_offload
        self.layer_batch_size = layer_batch_size
        self.normalize_weights = normalize_weights
        self.debug = debug
        self.epsilon = epsilon
        
        # Set up concepts to track with logit lens
        self.logit_lens_concepts = logit_lens_concepts or [
            "person", "building", "water", "sky", "car", "sign", "object"
        ]
        
        # Set up selection configuration
        self.selection_config = selection_config or SelectionConfig(
            beta_target=0.8,  # Coverage threshold for selecting source nodes
            beta_layer=0.7,   # Coverage threshold for pruning at layer level
            min_keep=1,       # Minimum nodes to keep per target
            max_keep=30,      # Maximum nodes to keep per target
            min_keep_layer=5, # Minimum nodes to keep per layer
            max_keep_layer=100 # Maximum nodes to keep per layer
        )
        
        # Initialize token decoder
        self.token_decoder = TokenDecoder(processor.tokenizer)
        self._activation_cache = {}
        
        # Initialize backends
        self.saliency_backend = SaliencyBackend(
            model=model,
            device=self.device,
            cpu_offload=cpu_offload,
            batch_size=layer_batch_size
        )
        
        self.attention_backend = AttentionBackend(
            model=model,
            device=self.device,
            cpu_offload=cpu_offload
        )
        
        # workflows/semantic_tracing.py (continued)
        self.logit_backend = LogitLensBackend(
            model=model,
            device=self.device,
            cpu_offload=cpu_offload
        )
        
        # Create I/O handler
        self.io = TraceIO(output_dir=output_dir)
        
        # Get attention layers
        self.attention_layer_names = self._get_attention_layer_names()
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(self.attention_layer_names)}
        self.num_layers = len(self.attention_layer_names)
        
        # Get image token ID
        self.image_token_id = getattr(model.config, "image_token_index", 32000)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a unique trace ID counter
        self.trace_id_counter = 0
        
        logger.info(f"Initialized SemanticTracer with {self.num_layers} attention layers")
        logger.info(f"Coverage parameters: beta_target={self.selection_config.beta_target}, "
                    f"beta_layer={self.selection_config.beta_layer}")
        logger.info(f"Node limits: min_keep={self.selection_config.min_keep}, "
                    f"max_keep={self.selection_config.max_keep}, "
                    f"min_keep_layer={self.selection_config.min_keep_layer}, "
                    f"max_keep_layer={self.selection_config.max_keep_layer}")
    
    def _get_attention_layer_names(self) -> List[str]:
        """Get attention layer names from the model."""
        from runtime.model_utils import get_llm_attention_layer_names
        
        layer_names = get_llm_attention_layer_names(self.model)
        
        if not layer_names:
            logger.warning("No attention layers found in model. Using default layer patterns.")
            # Fallback to common patterns based on model architecture
            for name, module in self.model.named_modules():
                if any(attn_name in name.lower() for attn_name in ["attention", "attn"]):
                    if any(key in name.lower() for key in ["self", "cross"]):
                        layer_names.append(name)
            
            # Sort by natural depth in model
            layer_names = sorted(layer_names)
            
        return layer_names
    
    def _get_layer_idx(self, layer_name_or_idx):
        """Get layer index from name or index."""
        if isinstance(layer_name_or_idx, int):
            return layer_name_or_idx
        return self.layer_name_to_idx.get(layer_name_or_idx, -1)
    
    def _get_layer_name(self, layer_idx):
        """Get layer name from index."""
        if 0 <= layer_idx < len(self.attention_layer_names):
            return self.attention_layer_names[layer_idx]
        return None
    
    def trace(
        self,
        input_data: Dict[str, Any],
        num_tokens: int = 1,
        target_token_idx: Optional[int] = None,
        analyze_specific_indices: Optional[List[int]] = None,
        analyze_last_token: bool = False,
        tracing_mode: str = "saliency",
        single_forward_pass: bool = False,
    ) -> Dict[str, Any]:
        """
        Main entry point for semantic tracing analysis.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate and analyze
            target_token_idx: Specific token index to analyze
            analyze_specific_indices: List of token indices to analyze
            analyze_last_token: Whether to analyze last token in prompt
            tracing_mode: "saliency", "attention", or "both"
            single_forward_pass: Use one forward pass for all layers
            
        Returns:
            Dictionary with trace results
        """
        # Validate tracing mode
        valid_modes = ["saliency", "attention", "both"]
        if tracing_mode not in valid_modes:
            raise ValueError(f"Invalid tracing mode: {tracing_mode}. Must be one of {valid_modes}")
        
        # Choose appropriate tracing method based on parameters
        if analyze_specific_indices:
            # Analyze specific token indices
            return self.generate_all_then_analyze_specific(
                input_data=input_data,
                num_tokens=num_tokens,
                analyze_indices=analyze_specific_indices,
                tracing_mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        elif analyze_last_token:
            # Analyze last token in prompt
            return self.analyze_last_token(
                input_data=input_data,
                tracing_mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        elif target_token_idx is not None:
            # Analyze specific target token
            return self.analyze_specific_token(
                input_data=input_data,
                target_token_idx=target_token_idx,
                tracing_mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        elif num_tokens > 1:
            # Analyze multiple generated tokens
            return self.generate_and_analyze_multiple(
                input_data=input_data,
                num_tokens=num_tokens,
                tracing_mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
        else:
            # Default: generate and analyze a single token
            return self.generate_and_analyze_single(
                input_data=input_data,
                tracing_mode=tracing_mode,
                single_forward_pass=single_forward_pass
            )
    
    def generate_and_analyze_single(
        self,
        input_data: Dict[str, Any],
        tracing_mode: str = "saliency",
        single_forward_pass: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a single token and perform semantic tracing.
        
        Args:
            input_data: Prepared input data
            tracing_mode: "saliency", "attention", or "both"
            single_forward_pass: Use one forward pass for all layers
            
        Returns:
            Dictionary with analysis results
        """
        # 1. Generate a token
        generation_result = self.autoregressive_generate(input_data["inputs"], num_tokens=1)
        
        if not generation_result.get("generated_tokens"):
            logger.error("Failed to generate any tokens")
            return {"error": "Token generation failed"}
        
        # 2. Get the generated token info
        target_token_idx = generation_result["generated_tokens"][0]["index"]
        token_id = generation_result["generated_tokens"][0]["id"]
        token_text = self.token_decoder.decode_token(token_id)
        
        logger.info(f"Generated token: '{token_text}' (ID: {token_id}) at position {target_token_idx}")
        
        # 3. Analyze the generated token
        analysis_result = self._analyze_token(
            inputs=generation_result["inputs"],
            text_indices=input_data["text_indices"],
            image_indices=input_data["image_indices"],
            target_token_idx=target_token_idx,
            tracing_mode=tracing_mode,
            single_forward_pass=single_forward_pass
        )
        
        # 4. Combine results
        result = {
            "input_data": input_data,
            "target_token": {
                "index": target_token_idx,
                "id": token_id,
                "text": token_text,
            },
            "full_sequence": generation_result["full_sequence"],
            "trace_results": analysis_result,
            "metadata": {
                "tracing_mode": tracing_mode,
                "logit_lens_concepts": self.logit_lens_concepts,
            }
        }
        
        # 5. Add image feature mapping if available
        if "original_image" in input_data and "image_spans" in input_data:
            feature_mapping = VisionMapper.map_tokens_to_grid(
                self.model,
                input_data["image_spans"],
                input_data.get("original_image_size_hw")
            )
            result["metadata"]["feature_mapping"] = feature_mapping
        
        # 6. Save results to disk
        trace_path = self._save_trace_results(result)
        result["trace_path"] = trace_path
        
        return result
    
    def generate_and_analyze_multiple(
        self,
        input_data: Dict[str, Any],
        num_tokens: int = 5,
        tracing_mode: str = "saliency",
        single_forward_pass: bool = False
    ) -> Dict[str, Any]:
        """
        Generate multiple tokens and perform semantic tracing for each.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate and analyze
            tracing_mode: "saliency", "attention", or "both"
            single_forward_pass: Use one forward pass for all layers
            
        Returns:
            Dictionary with analysis results for all tokens
        """
        # 1. Generate all tokens
        generation_result = self.autoregressive_generate(input_data["inputs"], num_tokens=num_tokens)
        
        if not generation_result.get("generated_tokens"):
            logger.error("Failed to generate any tokens")
            return {"error": "Token generation failed"}
        
        # 2. Set up results structure
        result = {
            "input_data": input_data,
            "target_tokens": [],
            "full_sequence": generation_result["full_sequence"],
            "trace_results": {},
            "metadata": {
                "tracing_mode": tracing_mode,
                "logit_lens_concepts": self.logit_lens_concepts,
            }
        }
        
        # 3. Add image feature mapping if available
        if "original_image" in input_data and "image_spans" in input_data:
            feature_mapping = VisionMapper.map_tokens_to_grid(
                self.model,
                input_data["image_spans"],
                input_data.get("original_image_size_hw")
            )
            result["metadata"]["feature_mapping"] = feature_mapping
        
        # 4. Analyze each token
        for i, token_info in enumerate(generation_result["generated_tokens"]):
            logger.info(f"Analyzing token {i+1}/{num_tokens}: '{token_info['text']}' at position {token_info['index']}")
            
            # Store token info
            result["target_tokens"].append({
                "index": token_info["index"],
                "id": token_info["id"],
                "text": token_info["text"],
            })
            
            # Analyze token
            token_result = self._analyze_token(
                inputs=generation_result["inputs"],
                text_indices=input_data["text_indices"],
                image_indices=input_data["image_indices"],
                target_token_idx=token_info["index"],
                tracing_mode=tracing_mode,
                single_forward_pass=single_forward_pass,
                trace_id=f"token_{token_info['index']}"
            )
            
            # Add to results
            result["trace_results"][f"token_{token_info['index']}"] = token_result
        
        # 5. Save combined results
        trace_path = self._save_trace_results(result)
        result["trace_path"] = trace_path
        
        return result
    
    def analyze_specific_token(
        self,
        input_data: Dict[str, Any],
        target_token_idx: int,
        tracing_mode: str = "saliency",
        single_forward_pass: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a specific token position in the input sequence.
        
        Args:
            input_data: Prepared input data
            target_token_idx: Index of token to analyze
            tracing_mode: "saliency", "attention", or "both"
            single_forward_pass: Use one forward pass for all layers
            
        Returns:
            Dictionary with analysis results
        """
        inputs = input_data["inputs"]
        input_ids = inputs["input_ids"]
        
        # Validate target token index
        if target_token_idx >= input_ids.shape[1]:
            raise ValueError(f"Target token index {target_token_idx} exceeds sequence length {input_ids.shape[1]}")
        
        # Get token info
        token_id = input_ids[0, target_token_idx].item()
        token_text = self.token_decoder.decode_token(token_id)
        
        logger.info(f"Analyzing token at index {target_token_idx}: '{token_text}' (ID: {token_id})")
        
        # Analyze the token
        analysis_result = self._analyze_token(
            inputs=inputs,
            text_indices=input_data["text_indices"],
            image_indices=input_data["image_indices"],
            target_token_idx=target_token_idx,
            tracing_mode=tracing_mode,
            single_forward_pass=single_forward_pass
        )
        
        # Combine results
        result = {
            "input_data": input_data,
            "target_token": {
                "index": target_token_idx,
                "id": token_id,
                "text": token_text,
            },
            "full_sequence": {
                "ids": input_ids[0].tolist(),
                "text": self.processor.tokenizer.decode(input_ids[0].tolist()),
            },
            "trace_results": analysis_result,
            "metadata": {
                "tracing_mode": tracing_mode,
                "logit_lens_concepts": self.logit_lens_concepts,
            }
        }
        
        # Add image feature mapping if available
        if "original_image" in input_data and "image_spans" in input_data:
            feature_mapping = VisionMapper.map_tokens_to_grid(
                self.model,
                input_data["image_spans"],
                input_data.get("original_image_size_hw")
            )
            result["metadata"]["feature_mapping"] = feature_mapping
        
        # Save results to disk
        trace_path = self._save_trace_results(result)
        result["trace_path"] = trace_path
        
        return result
    
    def analyze_last_token(
        self,
        input_data: Dict[str, Any],
        tracing_mode: str = "saliency",
        single_forward_pass: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze the last token in the input sequence.
        
        Args:
            input_data: Prepared input data
            tracing_mode: "saliency", "attention", or "both"
            single_forward_pass: Use one forward pass for all layers
            
        Returns:
            Dictionary with analysis results
        """
        inputs = input_data["inputs"]
        input_ids = inputs["input_ids"]
        
        # Get the last token index
        target_token_idx = input_ids.shape[1] - 1
        
        # Analyze using the specific token method
        return self.analyze_specific_token(
            input_data=input_data,
            target_token_idx=target_token_idx,
            tracing_mode=tracing_mode,
            single_forward_pass=single_forward_pass
        )
    
    def generate_all_then_analyze_specific(
        self,
        input_data: Dict[str, Any],
        num_tokens: int = 10,
        analyze_indices: List[int] = None,
        tracing_mode: str = "saliency",
        single_forward_pass: bool = False
    ) -> Dict[str, Any]:
        """
        First generates a complete sequence of tokens, then analyzes specific token indices.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate
            analyze_indices: Specific token indices to analyze
            tracing_mode: "saliency", "attention", or "both"
            single_forward_pass: Use one forward pass for all layers
            
        Returns:
            Dictionary with analysis results
        """
        # 1. Generate all tokens first
        logger.info(f"Generating complete sequence of {num_tokens} tokens")
        generation_result = self.autoregressive_generate(input_data["inputs"], num_tokens=num_tokens)
        
        if not generation_result.get("generated_tokens"):
            logger.error("Failed to generate any tokens")
            return {"error": "Token generation failed"}
        
        # 2. Set up results structure
        result = {
            "input_data": input_data,
            "target_tokens": [],
            "all_generated_tokens": generation_result["generated_tokens"],
            "full_sequence": generation_result["full_sequence"],
            "trace_results": {},
            "metadata": {
                "tracing_mode": tracing_mode,
                "logit_lens_concepts": self.logit_lens_concepts,
            }
        }
        
        # 3. Add image feature mapping if available
        if "original_image" in input_data and "image_spans" in input_data:
            feature_mapping = VisionMapper.map_tokens_to_grid(
                self.model,
                input_data["image_spans"],
                input_data.get("original_image_size_hw")
            )
            result["metadata"]["feature_mapping"] = feature_mapping
        
        # 4. Filter indices to analyze
        original_seq_len = generation_result["original_seq_len"]
        max_seq_len = generation_result["inputs"]["input_ids"].shape[1]
        
        # Filter for valid indices
        valid_indices = [idx for idx in analyze_indices 
                          if original_seq_len <= idx < max_seq_len]
        
        if not valid_indices:
            logger.warning("No valid token indices to analyze")
            return result
        
        logger.info(f"Analyzing {len(valid_indices)} specific tokens from the generated sequence")
        
        # 5. Analyze each specified token
        for i, token_idx in enumerate(valid_indices):
            # Find token info
            token_info = None
            for token in generation_result["generated_tokens"]:
                if token["index"] == token_idx:
                    token_info = token
                    break
            
            if token_info is None:
                logger.warning(f"Token at index {token_idx} not found in generated tokens")
                continue
            
            logger.info(f"Analyzing token {i+1}/{len(valid_indices)}: '{token_info['text']}' at position {token_idx}")
            
            # Store token info
            result["target_tokens"].append({
                "index": token_idx,
                "id": token_info["id"],
                "text": token_info["text"],
            })
            
            # Analyze token
            token_result = self._analyze_token(
                inputs=generation_result["inputs"],
                text_indices=input_data["text_indices"],
                image_indices=input_data["image_indices"],
                target_token_idx=token_idx,
                tracing_mode=tracing_mode,
                single_forward_pass=single_forward_pass,
                trace_id=f"token_{token_idx}"
            )
            
            # Add to results
            result["trace_results"][f"token_{token_idx}"] = token_result
        
        # 6. Save combined results
        trace_path = self._save_trace_results(result)
        result["trace_path"] = trace_path
        
        return result
    
    def _analyze_token(
        self,
        inputs: Dict[str, torch.Tensor],
        text_indices: torch.Tensor,
        image_indices: torch.Tensor,
        target_token_idx: int,
        tracing_mode: str = "saliency",
        single_forward_pass: bool = False,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Core token analysis function that implements the semantic tracing process.
        
        This function handles the complete tracing workflow:
        1. Sets up appropriate backends based on the tracing mode
        2. Processes token information flow layer by layer
        3. Enriches the trace with logit lens projections
        4. Optimizes memory usage through batching and caching
        
        Args:
            inputs: Model inputs dictionary
            text_indices: Indices of text tokens in the sequence
            image_indices: Indices of image tokens in the sequence
            target_token_idx: Index of token to analyze
            tracing_mode: "saliency", "attention", or "both"
            single_forward_pass: Whether to use a single forward/backward pass for all layers
                            to optimize memory usage at the cost of more GPU memory
            trace_id: Optional identifier for the trace
            
        Returns:
            Dictionary with trace results organized by analysis mode and layer
        """
        analysis_result = {}
        
        # Increment trace ID counter for unique identification
        if trace_id is None:
            self.trace_id_counter += 1
            trace_id = str(self.trace_id_counter)
        
        # FIX: Check if we have cached results for these inputs
        cache_key = self._generate_cache_key(inputs)
        cache_hit = cache_key in self._activation_cache
        
        # Enable single_forward_pass by adjusting batch size
        if single_forward_pass and tracing_mode in ["saliency", "both"]:
            # Increase batch size to include all layers in one pass
            self.saliency_backend.batch_size = self.num_layers
            logger.info(f"Using single forward pass mode with batch size {self.num_layers}")
        
        # Pass single_forward_pass to attention backend
        if single_forward_pass and tracing_mode in ["attention", "both"]:
            logger.info("Using single forward pass for attention analysis")
            self.attention_backend.use_single_pass = True
        
        # Setup analysis backends if needed
        if tracing_mode in ["saliency", "both"]:
            self.saliency_backend.setup(self.attention_layer_names)
        
        if tracing_mode in ["attention", "both"]:
            self.attention_backend.setup(self.attention_layer_names)
        
        # Logit lens backend is always used for concept tracking
        self.logit_backend.setup(self.attention_layer_names)
        
        # Calculate token types for all tokens in the sequence
        seq_len = inputs["input_ids"].shape[1]
        token_types = torch.zeros(seq_len, dtype=torch.long, device=self.device)
        token_types[text_indices] = 1  # Text tokens
        token_types[image_indices] = 2  # Image tokens
        # Generated tokens remain 0
        
        # Convert token_types to a dictionary for easy reference
        token_type_map = {idx: int(token_types[idx].item()) for idx in range(seq_len)}
        
        # Get all token IDs and texts for the whole sequence
        all_token_ids = inputs["input_ids"][0].tolist()
        all_token_texts = [self.token_decoder.decode_token(tid) for tid in all_token_ids]
        
        # FIX: Use cached results if available to avoid recomputation
        if cache_hit:
            logger.info(f"Using cached computation results for input sequence")
            cached_data = self._activation_cache[cache_key]
            
            # Run appropriate analyses based on the tracing mode, using cached data where possible
            if tracing_mode in ["saliency", "both"] and "saliency_scores" in cached_data:
                logger.info(f"Using cached saliency scores for token at position {target_token_idx}")
                saliency_result = {"saliency_scores": cached_data["saliency_scores"]}
                analysis_result["saliency"] = self._process_saliency_results(
                    saliency_result,
                    target_token_idx,
                    all_token_ids,
                    all_token_texts,
                    token_type_map,
                    trace_id
                )
            elif tracing_mode in ["saliency", "both"]:
                # Compute if not in cache
                logger.info(f"Computing saliency scores for token at position {target_token_idx}")
                saliency_result = self.saliency_backend.compute(inputs, [target_token_idx])
                analysis_result["saliency"] = self._process_saliency_results(
                    saliency_result,
                    target_token_idx,
                    all_token_ids,
                    all_token_texts,
                    token_type_map,
                    trace_id
                )
                # Cache the results
                if "saliency_scores" in saliency_result:
                    if cache_key not in self._activation_cache:
                        self._activation_cache[cache_key] = {}
                    self._activation_cache[cache_key]["saliency_scores"] = saliency_result["saliency_scores"]
            
            if tracing_mode in ["attention", "both"] and "attention_maps" in cached_data:
                logger.info(f"Using cached attention maps for token at position {target_token_idx}")
                attention_result = {"attention_maps": cached_data["attention_maps"]}
                analysis_result["attention"] = self._process_attention_results(
                    attention_result,
                    target_token_idx,
                    all_token_ids,
                    all_token_texts,
                    token_type_map,
                    trace_id
                )
            elif tracing_mode in ["attention", "both"]:
                # Compute if not in cache
                logger.info(f"Computing attention maps for token at position {target_token_idx}")
                attention_result = self.attention_backend.compute(inputs, [target_token_idx])
                analysis_result["attention"] = self._process_attention_results(
                    attention_result,
                    target_token_idx,
                    all_token_ids,
                    all_token_texts,
                    token_type_map,
                    trace_id
                )
                # Cache the results
                if "attention_maps" in attention_result:
                    if cache_key not in self._activation_cache:
                        self._activation_cache[cache_key] = {}
                    self._activation_cache[cache_key]["attention_maps"] = attention_result["attention_maps"]
        else:
            # No cache hit, compute everything
            # Run appropriate analyses based on the tracing mode
            if tracing_mode in ["saliency", "both"]:
                logger.info(f"Computing saliency scores for token at position {target_token_idx}")
                saliency_result = self.saliency_backend.compute(inputs, [target_token_idx])
                analysis_result["saliency"] = self._process_saliency_results(
                    saliency_result,
                    target_token_idx,
                    all_token_ids,
                    all_token_texts,
                    token_type_map,
                    trace_id
                )
                # Cache the results
                if "saliency_scores" in saliency_result:
                    if cache_key not in self._activation_cache:
                        self._activation_cache[cache_key] = {}
                    self._activation_cache[cache_key]["saliency_scores"] = saliency_result["saliency_scores"]
            
            if tracing_mode in ["attention", "both"]:
                logger.info(f"Computing attention maps for token at position {target_token_idx}")
                attention_result = self.attention_backend.compute(inputs, [target_token_idx])
                analysis_result["attention"] = self._process_attention_results(
                    attention_result,
                    target_token_idx,
                    all_token_ids,
                    all_token_texts,
                    token_type_map,
                    trace_id
                )
                # Cache the results
                if "attention_maps" in attention_result:
                    if cache_key not in self._activation_cache:
                        self._activation_cache[cache_key] = {}
                    self._activation_cache[cache_key]["attention_maps"] = attention_result["attention_maps"]
        
        # Run logit lens analysis for all tokens involved in the trace
        # We always need to run this part since it's token-specific
        logger.info("Running logit lens analysis for all tokens involved in the trace")
        token_indices_to_analyze = self._get_all_traced_tokens(analysis_result)
        
        if token_indices_to_analyze:
            # Check for cached projections
            use_cached_projections = False
            if cache_hit and "projections" in self._activation_cache[cache_key]:
                # Only use cached projections if they cover ALL needed tokens
                cached_tokens = set(
                    token_idx for layer_proj in self._activation_cache[cache_key]["projections"].values() 
                    for token_idx in layer_proj.keys()
                )
                if all(idx in cached_tokens for idx in token_indices_to_analyze):
                    logger.info(f"Using cached logit lens projections for {len(token_indices_to_analyze)} tokens")
                    logit_result = {"projections": self._activation_cache[cache_key]["projections"]}
                    use_cached_projections = True
            
            if not use_cached_projections:
                # Compute projections for tokens not in cache
                logger.info(f"Computing logit lens projections for {len(token_indices_to_analyze)} tokens")
                logit_result = self.logit_backend.compute(
                    inputs, 
                    token_indices_to_analyze,
                    # Add concept tracking
                    concept_ids={concept: self.processor.tokenizer.encode(concept, add_special_tokens=False) 
                                for concept in self.logit_lens_concepts}
                )
                # Cache the results
                if "projections" in logit_result:
                    if cache_key not in self._activation_cache:
                        self._activation_cache[cache_key] = {}
                    self._activation_cache[cache_key]["projections"] = logit_result["projections"]
            
            self._enrich_trace_with_logit_lens(
                analysis_result,
                logit_result,
                all_token_ids,
                all_token_texts
            )
        
        # Clean up backends
        self.saliency_backend.cleanup()
        self.attention_backend.cleanup()
        self.logit_backend.cleanup()
        
        # Reset batch size after analysis
        if single_forward_pass and tracing_mode in ["saliency", "both"]:
            self.saliency_backend.batch_size = self.layer_batch_size
        
        # Reset attention backend use_single_pass
        if single_forward_pass and tracing_mode in ["attention", "both"]:
            self.attention_backend.use_single_pass = False
        
        return analysis_result
    
    def _process_saliency_results(
        self,
        saliency_result: Dict[str, Any],
        target_token_idx: int,
        all_token_ids: List[int],
        all_token_texts: List[str],
        token_type_map: Dict[int, int],
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Process saliency results into an information flow graph.
        
        Combines gradient-based attention saliency scores with token metadata to create
        a complete information flow diagram between layers. Uses coverage-based selection
        to identify the most important source tokens at each layer.
        
        Args:
            saliency_result: Raw saliency backend results with saliency scores
            target_token_idx: The target token position to analyze
            all_token_ids: List of all token IDs in the sequence
            all_token_texts: List of all token texts in the sequence
            token_type_map: Mapping from token index to token type (0=generated, 1=text, 2=image)
            trace_id: Unique identifier for the trace
            
        Returns:
            Dictionary with processed saliency results organized by layer
        """
        if not saliency_result.get("saliency_scores"):
            logger.warning("No saliency scores found in result")
            return {"error": "No saliency scores found"}
        
        # Results to return
        trace_results = {}
        trace_records = []
        
        # Track information flow
        current_targets = {target_token_idx: 1.0}  # Initial target with weight 1.0
        
        # Process layers from deepest to shallowest
        for layer_idx in range(self.num_layers - 1, -1, -1):
            layer_name = self._get_layer_name(layer_idx)
            
            if not layer_name or layer_name not in saliency_result["saliency_scores"]:
                continue
            
            logger.info(f"Processing saliency for layer {layer_idx} ({layer_name})")
            
            # Get the saliency map for this layer
            saliency_map = saliency_result["saliency_scores"][layer_name]
            
            # Average over batch and head dimensions if needed
            if saliency_map.ndim == 4:  # [batch, head, seq, seq]
                saliency_map = saliency_map.mean(dim=(0, 1))
            elif saliency_map.ndim == 3:  # [head, seq, seq] or [batch, seq, seq]
                saliency_map = saliency_map.mean(dim=0)
            
            # Create layer results dictionary
            layer_results = {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "target_tokens": [],
                "source_tokens": []
            }
            
            # Process each current target token
            target_to_sources = {}
            for target_idx, target_weight in current_targets.items():
                if target_idx >= saliency_map.shape[0]:
                    logger.warning(f"Target index {target_idx} exceeds saliency map dimensions. Skipping.")
                    continue
                
                # Get saliency from target to all previous tokens (causal)
                target_vector = saliency_map[target_idx, :target_idx].cpu().numpy()
                
                if len(target_vector) == 0:
                    logger.warning(f"Empty target vector for token {target_idx}. Skipping.")
                    continue
                
                # Select important source tokens using coverage-based selection
                selected_sources = SelectionStrategy.select_sources(
                    target_vector, 
                    self.selection_config
                )
                
                # Get source info
                sources = []
                for idx, weight in selected_sources:
                    # Scale by the target's weight to get global importance
                    scaled_weight = weight * target_weight
                    
                    # Get token info
                    token_id = all_token_ids[idx]
                    token_text = all_token_texts[idx]
                    token_type = token_type_map.get(idx, 0)
                    
                    source_info = {
                        "index": idx,
                        "id": token_id,
                        "text": token_text,
                        "type": token_type,
                        "saliency_score": float(target_vector[idx]),
                        "relative_weight": float(weight),
                        "scaled_weight": float(scaled_weight),
                        "trace_id": trace_id
                    }
                    sources.append(source_info)
                
                # Record target token info
                target_info = {
                    "index": target_idx,
                    "id": all_token_ids[target_idx],
                    "text": all_token_texts[target_idx],
                    "type": token_type_map.get(target_idx, 0),
                    "weight": float(target_weight),
                    "sources": sources,
                    "trace_id": trace_id
                }
                layer_results["target_tokens"].append(target_info)
                
                # Store sources for next iteration
                target_to_sources[target_idx] = sources
            
            # Layer-level source pruning to control overall node count
            # First, aggregate sources to avoid duplicates
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
            
            # Convert to list for pruning - FIX: Convert to expected tuple format
            unique_sources = list(aggregated_sources.values())
            tuple_sources = [(s["index"], s["scaled_weight"]) for s in unique_sources]
            
            # Apply layer-level pruning
            if tuple_sources:
                pruned = SelectionStrategy.prune_layer(
                    tuple_sources,
                    self.selection_config
                )
                
                # FIX: Create a set of remaining indices after pruning
                remaining_indices = set(idx for idx, _ in pruned)
                
                # Update target_to_sources to only include remaining sources
                for target_idx in target_to_sources:
                    target_to_sources[target_idx] = [
                        s for s in target_to_sources[target_idx]
                        if s["index"] in remaining_indices
                    ]
            
            # Generate new targets for next layer
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
                # FIX: Ensure we normalize weights after pruning
                current_targets = SelectionStrategy.renormalize(new_targets)
            else:
                logger.warning(f"No valid sources found for layer {layer_idx}.")
                break
            
            # Add layer results
            trace_results[layer_idx] = layer_results
            
            # Add to trace records for CSV output
            this_layer_records = self._create_trace_records(
                layer_idx, 
                layer_results,
                trace_id,
                tracing_mode="saliency",
                all_token_ids=all_token_ids,  # FIX: Pass through token info
                all_token_texts=all_token_texts
            )
            trace_records.extend(this_layer_records)
        
        # Save trace records to CSV
        if trace_records:
            csv_path = self.io.write_trace_data(
                trace_id=f"saliency_{trace_id}",
                records=trace_records,
                metadata={"mode": "saliency"}
            )
            trace_results["trace_data_path"] = csv_path
        
        return trace_results
    
    def _process_attention_results(
        self,
        attention_result: Dict[str, Any],
        target_token_idx: int,
        all_token_ids: List[int],
        all_token_texts: List[str],
        token_type_map: Dict[int, int],
        trace_id: str
    ) -> Dict[str, Any]:
        """
        Process attention results into an information flow graph.
        
        Transforms raw attention weights into a structured information flow that tracks
        how attention propagates from generated tokens back through the model layers.
        Uses coverage-based selection to identify the most important connections.
        
        Args:
            attention_result: Raw attention backend results containing attention maps
            target_token_idx: The target token position to analyze
            all_token_ids: List of all token IDs in the sequence
            all_token_texts: List of all token texts in the sequence
            token_type_map: Mapping from token index to token type (0=generated, 1=text, 2=image)
            trace_id: Unique identifier for the trace
            
        Returns:
            Dictionary with processed attention results organized by layer
        """
        if not attention_result.get("attention_maps"):
            logger.warning("No attention maps found in result")
            return {"error": "No attention maps found"}
        
        # Results to return
        trace_results = {}
        trace_records = []
        
        # Track information flow
        current_targets = {target_token_idx: 1.0}  # Initial target with weight 1.0
        
        # Process layers from deepest to shallowest
        for layer_idx in range(self.num_layers - 1, -1, -1):
            layer_name = self._get_layer_name(layer_idx)
            
            if not layer_name or layer_name not in attention_result["attention_maps"]:
                continue
            
            logger.info(f"Processing attention for layer {layer_idx} ({layer_name})")
            
            # Get the attention map for this layer
            attention_map = attention_result["attention_maps"][layer_name]
            
            # Average over batch and head dimensions if needed
            if attention_map.ndim == 4:  # [batch, head, seq, seq]
                attention_map = attention_map.mean(dim=1)  # Average over heads
            
            if attention_map.ndim == 3:
                # Check if batch dimension is present
                if attention_map.shape[0] > 1:
                    attention_map = attention_map[0]  # Take the first batch
            
            # Create layer results dictionary
            layer_results = {
                "layer_idx": layer_idx,
                "layer_name": layer_name,
                "target_tokens": [],
                "source_tokens": []
            }
            
            # Process each current target token
            target_to_sources = {}
            for target_idx, target_weight in current_targets.items():
                if target_idx >= attention_map.shape[0]:
                    logger.warning(f"Target index {target_idx} exceeds attention map dimensions. Skipping.")
                    continue
                
                # Get attention from target to all previous tokens (causal)
                target_vector = attention_map[target_idx, :target_idx].cpu().numpy()
                
                if len(target_vector) == 0:
                    logger.warning(f"Empty target vector for token {target_idx}. Skipping.")
                    continue
                
                # Select important source tokens using coverage-based selection
                selected_sources = SelectionStrategy.select_sources(
                    target_vector, 
                    self.selection_config
                )
                
                # Get source info
                sources = []
                for idx, weight in selected_sources:
                    # Scale by the target's weight to get global importance
                    scaled_weight = weight * target_weight
                    
                    # Get token info
                    token_id = all_token_ids[idx]
                    token_text = all_token_texts[idx]
                    token_type = token_type_map.get(idx, 0)
                    
                    source_info = {
                        "index": idx,
                        "id": token_id,
                        "text": token_text,
                        "type": token_type,
                        "attention_score": float(target_vector[idx]),
                        "relative_weight": float(weight),
                        "scaled_weight": float(scaled_weight),
                        "trace_id": trace_id
                    }
                    sources.append(source_info)
                
                # Record target token info
                target_info = {
                    "index": target_idx,
                    "id": all_token_ids[target_idx],
                    "text": all_token_texts[target_idx],
                    "type": token_type_map.get(target_idx, 0),
                    "weight": float(target_weight),
                    "sources": sources,
                    "trace_id": trace_id
                }
                layer_results["target_tokens"].append(target_info)
                
                # Store sources for next iteration
                target_to_sources[target_idx] = sources
            
            # Layer-level source pruning to control overall node count
            # First, aggregate sources to avoid duplicates
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
            
            # Convert to list for pruning - FIX: Convert to expected tuple format
            unique_sources = list(aggregated_sources.values())
            tuple_sources = [(s["index"], s["scaled_weight"]) for s in unique_sources]
            
            # Apply layer-level pruning
            if tuple_sources:
                pruned = SelectionStrategy.prune_layer(
                    tuple_sources,
                    self.selection_config
                )
                
                # FIX: Create a set of remaining indices after pruning
                remaining_indices = set(idx for idx, _ in pruned)
                
                # Update target_to_sources to only include remaining sources
                for target_idx in target_to_sources:
                    target_to_sources[target_idx] = [
                        s for s in target_to_sources[target_idx]
                        if s["index"] in remaining_indices
                    ]
            
            # Generate new targets for next layer
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
                # FIX: Ensure we normalize weights after pruning
                current_targets = SelectionStrategy.renormalize(new_targets)
            else:
                logger.warning(f"No valid sources found for layer {layer_idx}.")
                break
            
            # Add layer results
            trace_results[layer_idx] = layer_results
            
            # Add to trace records for CSV output
            this_layer_records = self._create_trace_records(
                layer_idx, 
                layer_results,
                trace_id,
                tracing_mode="attention",
                all_token_ids=all_token_ids,  # FIX: Pass through token info
                all_token_texts=all_token_texts
            )
            trace_records.extend(this_layer_records)
        
        # Save trace records to CSV
        if trace_records:
            csv_path = self.io.write_trace_data(
                trace_id=f"attention_{trace_id}",
                records=trace_records,
                metadata={"mode": "attention"}
            )
            trace_results["trace_data_path"] = csv_path
        
        return trace_results
        
    def _get_all_traced_tokens(self, analysis_result: Dict[str, Any]) -> List[int]:
        """
        Extract all token indices involved in the trace for logit lens analysis.
        
        Args:
            analysis_result: Results from saliency or attention analysis
            
        Returns:
            List of all token indices to analyze with logit lens
        """
        token_indices = set()
        
        # Process each tracing mode
        for mode_name, mode_results in analysis_result.items():
            # Process each layer
            for layer_idx, layer_result in mode_results.items():
                if isinstance(layer_idx, int) and isinstance(layer_result, dict):
                    # Add target tokens
                    for target in layer_result.get("target_tokens", []):
                        token_indices.add(target["index"])
                        
                        # Add source tokens
                        for source in target.get("sources", []):
                            token_indices.add(source["index"])
        
        return sorted(list(token_indices))
    
    def _enrich_trace_with_logit_lens(
        self,
        analysis_result: Dict[str, Any],
        logit_result: Dict[str, Any],
        all_token_ids: List[int],
        all_token_texts: List[str]
    ) -> None:
        """
        Add logit lens projections to the trace results.
        
        Enriches the trace with token prediction information at each layer,
        helping to visualize how token representations evolve through the model.
        Also adds concept probability tracking for specific concepts of interest.
        
        Args:
            analysis_result: Results from saliency or attention analysis
            logit_result: Results from logit lens analysis
            all_token_ids: List of all token IDs
            all_token_texts: List of all token texts
        """
        if "projections" not in logit_result:
            logger.warning("No projections found in logit result")
            return
        
        projections = logit_result["projections"]
        
        # Process each tracing mode
        for mode_name, mode_results in analysis_result.items():
            # Process each layer
            for layer_idx, layer_result in mode_results.items():
                if isinstance(layer_idx, int) and isinstance(layer_result, dict):
                    # Get layer name
                    layer_name = layer_result.get("layer_name")
                    
                    if not layer_name or layer_name not in projections:
                        continue
                    
                    # Get projections for this layer
                    layer_projections = projections[layer_name]
                    
                    # Create a container for logit lens projections
                    layer_result["logit_lens_projections"] = {}
                    
                    # Process target tokens
                    for target in layer_result.get("target_tokens", []):
                        token_idx = target["index"]
                        
                        if token_idx in layer_projections:
                            # Extract projection info
                            token_projections = layer_projections[token_idx]
                            
                            # Get top predicted tokens
                            top_indices = token_projections["top_k"]["indices"]
                            top_probs = token_projections["top_k"]["probs"]
                            
                            # FIX: Extract concept predictions if available
                            concept_predictions = token_projections.get("concept_predictions", {})
                            
                            # Format projections for better readability
                            top_predictions = []
                            for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
                                top_predictions.append({
                                    "rank": i + 1,
                                    "token_id": idx,
                                    "token_text": self.token_decoder.decode_token(idx),
                                    "probability": prob
                                })
                            
                            # Add to result with concept predictions
                            layer_result["logit_lens_projections"][token_idx] = {
                                "top_predictions": top_predictions,
                                "concept_predictions": concept_predictions  # FIX: Include concept predictions
                            }
                            
                            # Add logit lens info to target
                            if top_predictions:
                                target["predicted_top_token"] = top_predictions[0]["token_text"]
                                target["predicted_top_prob"] = top_predictions[0]["probability"]
                                
                            # FIX: Add concept probabilities to target
                            for concept, data in concept_predictions.items():
                                target[f"concept_{concept}_prob"] = data.get("probability", 0.0)
                    
                    # Process source tokens
                    for target in layer_result.get("target_tokens", []):
                        for source in target.get("sources", []):
                            token_idx = source["index"]
                            
                            if token_idx in layer_projections:
                                # Extract projection info
                                token_projections = layer_projections[token_idx]
                                
                                # Get top predicted tokens
                                top_indices = token_projections["top_k"]["indices"]
                                top_probs = token_projections["top_k"]["probs"]
                                
                                # FIX: Extract concept predictions
                                concept_predictions = token_projections.get("concept_predictions", {})
                                
                                if top_indices and top_probs:
                                    # Add logit lens info to source
                                    source["predicted_top_token"] = self.token_decoder.decode_token(top_indices[0])
                                    source["predicted_top_prob"] = top_probs[0]
                                    
                                    # FIX: Add concept probabilities to source
                                    for concept, data in concept_predictions.items():
                                        source[f"concept_{concept}_prob"] = data.get("probability", 0.0)
    
    def _create_trace_records(
        self,
        layer_idx: int,
        layer_results: Dict[str, Any],
        trace_id: str,
        tracing_mode: str = "saliency",
        all_token_ids: List[int] = None,
        all_token_texts: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create trace records for CSV output and visualization.
        
        Transforms the layer-specific analysis results into a flat format suitable for
        CSV output and visualization. Each token involved in the trace at this layer
        gets a separate record with all its attributes including concept probabilities.
        
        Args:
            layer_idx: Layer index
            layer_results: Results for this layer
            trace_id: Unique identifier for the trace
            tracing_mode: "saliency" or "attention"
            all_token_ids: List of all token IDs in the sequence
            all_token_texts: List of all token texts in the sequence
            
        Returns:
            List of record dictionaries for this layer
        """
        records = []
        
        # Process all tokens involved in this layer
        all_token_indices = set()
        
        # First, collect all token indices
        for target_info in layer_results.get("target_tokens", []):
            all_token_indices.add(target_info["index"])
            for source in target_info.get("sources", []):
                all_token_indices.add(source["index"])
        
        # Then create records for each token
        for token_idx in all_token_indices:
            # Find if this token is a target
            is_target = any(t["index"] == token_idx for t in layer_results.get("target_tokens", []))
            
            # Find all targets this token is a source for
            source_targets = []
            for t in layer_results.get("target_tokens", []):
                for s in t.get("sources", []):
                    if s["index"] == token_idx:
                        source_targets.append(t["index"])
            
            # Get target token info if target
            target_info = None
            if is_target:
                for t in layer_results.get("target_tokens", []):
                    if t["index"] == token_idx:
                        target_info = t
                        break
            
            # Get token projection data if available
            top_pred_text = ""
            top_pred_prob = 0.0
            
            # FIX: Get concept probabilities if available
            concept_probs = {}
            
            if "logit_lens_projections" in layer_results and token_idx in layer_results["logit_lens_projections"]:
                projections = layer_results["logit_lens_projections"][token_idx]
                if projections.get("top_predictions"):
                    top_pred = projections["top_predictions"][0]
                    top_pred_text = top_pred.get("token_text", "")
                    top_pred_prob = top_pred.get("probability", 0.0)
                    
                # FIX: Extract concept predictions
                if "concept_predictions" in projections:
                    for concept, data in projections["concept_predictions"].items():
                        concept_probs[concept] = data.get("probability", 0.0)
            
            # FIX: Use all_token_texts and all_token_ids for proper token information
            token_text = all_token_texts[token_idx] if all_token_texts and token_idx < len(all_token_texts) else "UNKNOWN"
            token_id = all_token_ids[token_idx] if all_token_ids and token_idx < len(all_token_ids) else -1
            
            # Create base record
            record = {
                "layer": layer_idx,
                "token_index": token_idx,
                "token_text": token_text,
                "token_id": token_id,
                "token_type": target_info["type"] if target_info else (
                    token_type_map.get(token_idx, 0) if "token_type_map" in locals() else 0
                ),
                "is_target": is_target,
                "source_for_targets": ",".join(map(str, source_targets)),
                "predicted_top_token": top_pred_text,
                "predicted_top_prob": top_pred_prob,
                "trace_id": trace_id,
                "mode": tracing_mode
            }
            
            # FIX: Add concept probabilities to the record
            for concept, prob in concept_probs.items():
                record[f"concept_{concept}_prob"] = prob
            
            # Add source-target relationships if target
            if is_target and target_info:
                sources_indices = []
                sources_weights = []
                
                for source in target_info.get("sources", []):
                    sources_indices.append(source["index"])
                    sources_weights.append(source["scaled_weight"])
                
                # Store as comma-separated lists
                record["sources_indices"] = ",".join(map(str, sources_indices))
                record["sources_weights"] = ",".join(map(str, sources_weights))
            
            records.append(record)
        
        return records
    
    def _save_trace_results(self, result: Dict[str, Any]) -> str:
        """
        Save trace results to disk and return the path.
        
        Args:
            result: The trace results
            
        Returns:
            Path to the saved results
        """
        # Create a unique ID for this trace
        self.trace_id_counter += 1
        trace_id = f"trace_{self.trace_id_counter}"
        
        # Create metadata
        metadata = {
            "tracing_mode": result.get("metadata", {}).get("tracing_mode", "saliency"),
            "logit_lens_concepts": self.logit_lens_concepts,
            "target_tokens": result.get("target_tokens", [result.get("target_token", {})]),
            "selection_config": {
                "beta_target": self.selection_config.beta_target,
                "beta_layer": self.selection_config.beta_layer,
                "min_keep": self.selection_config.min_keep,
                "max_keep": self.selection_config.max_keep,
                "min_keep_layer": self.selection_config.min_keep_layer,
                "max_keep_layer": self.selection_config.max_keep_layer
            }
        }
        
        # Add image availability info
        if "original_image" in result.get("input_data", {}):
            metadata["image_available"] = True
        
        # Add feature mapping if available
        if "feature_mapping" in result.get("metadata", {}):
            metadata["feature_mapping"] = result["metadata"]["feature_mapping"]
        
        # Save metadata to a JSON file
        metadata_path = os.path.join(self.output_dir, f"{trace_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def _generate_cache_key(self, inputs: Dict[str, torch.Tensor]) -> str:
        """
        Generate a unique key for identifying inputs for caching.
        
        This allows reusing computation results across multiple token analyses
        with the same input sequence.
        
        Args:
            inputs: Model input tensors
            
        Returns:
            A unique string key for the inputs
        """
        # Use input IDs shape and a hash of the content
        input_ids = inputs["input_ids"]
        # Convert to numpy for hashing
        as_numpy = input_ids.detach().cpu().numpy()
        # Simple hash based on sequence length and content
        content_hash = hash(as_numpy.tobytes())
        return f"seq_len_{input_ids.shape[1]}_{content_hash}"