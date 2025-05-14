# workflow/semantic_tracing.py
"""
Semantic tracing workflow for analyzing information flow in VLMs.
Coordinates the analysis process from input preparation to result generation.
"""

import os
import time
import torch
from typing import Dict, List, Optional, Any, Tuple, Union
from PIL import Image

from runtime.cache import TracingCache
from runtime.selection import SelectionConfig, SelectionStrategy
from runtime.io import TraceIO
from runtime.model_utils import get_llm_attention_layer_names
from backends import SaliencyBackend, AttentionBackend, LogitBackend
from preprocess.input_builder import prepare_inputs
from runtime.generation import GenerationMixin


class SemanticTracingWorkflow(GenerationMixin):
    """
    Workflow for semantic tracing of information flow in VLMs.
    Analyzes how information propagates from input to output tokens.
    """
    
    def __init__(self, model, processor, output_dir: str,
                 selection_config: Optional[SelectionConfig] = None,
                 debug: bool = False):
        """
        Initialize the semantic tracing workflow.
        
        Args:
            model: The model to analyze
            processor: The model's processor
            output_dir: Directory for saving results
            selection_config: Configuration for token selection/pruning
            debug: Whether to print additional debug information
        """
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.device = next(model.parameters()).device
        self.layer_names = get_llm_attention_layer_names(model)
        self.cache = TracingCache(cpu_offload=True)
        self.backends = {}  # Lazy initialized backends
        self.sel_cfg = selection_config or SelectionConfig()
        self.debug = debug
        self.trace_io = TraceIO(output_dir)
        
        # Debug info
        if debug:
            print(f"Initialized SemanticTracingWorkflow with {len(self.layer_names)} layers")
            print(f"Device: {self.device}")
            print(f"Selection config: β_target={self.sel_cfg.beta_target}, β_layer={self.sel_cfg.beta_layer}")
            print(f"Output directory: {output_dir}")
    
    def prepare_inputs(self, image: Union[str, Image.Image], prompt: str) -> Dict[str, Any]:
        """
        Prepare model inputs from image and prompt.
        
        Args:
            image: PIL image, path, or URL
            prompt: Text prompt
            
        Returns:
            Dictionary with prepared inputs and metadata
        """
        return prepare_inputs(
            model=self.model,
            processor=self.processor,
            image=image,
            prompt=prompt
        )
    
    def _get_backend(self, mode: str):
        """
        Get the appropriate backend for the requested analysis mode.
        Backends are created lazily to save memory.
        
        Args:
            mode: Analysis mode ('attention', 'saliency', or 'logit')
            
        Returns:
            The requested backend instance
        """
        if mode not in self.backends:
            if mode == "attention":
                self.backends[mode] = AttentionBackend(
                    self.model, self.layer_names, self.cache, self.device
                )
            elif mode == "saliency":
                self.backends[mode] = SaliencyBackend(
                    self.model, self.layer_names, self.cache, self.device
                )
            elif mode == "logit":
                self.backends[mode] = LogitBackend(
                    self.model, self.layer_names, self.cache, self.device
                )
            else:
                raise ValueError(f"Unknown backend mode: {mode}")
        
        return self.backends[mode]
    
    def trace(self, input_data: Dict[str, Any],
              target_tokens: Dict[int, float],
              mode: str = "saliency",
              trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Trace information flow for specific target tokens.
        
        Args:
            input_data: Prepared input data
            target_tokens: Dictionary mapping target token indices to weights
            mode: Analysis mode ('saliency' or 'attention')
            trace_id: Optional unique identifier for this trace
            
        Returns:
            Dictionary with analysis results
        """
        if not trace_id:
            trace_id = f"{mode}_{int(time.time())}"
        
        # Get appropriate backend
        backend = self._get_backend(mode)
        
        # Ensure necessary data is cached
        backend.ensure_cache(input_data["inputs"], list(target_tokens.keys()))
        
        # Process layers from deepest to shallowest
        records = []
        current = target_tokens
        
        for layer in reversed(range(len(self.layer_names))):
            if not current:
                if self.debug:
                    print(f"No target tokens for layer {layer}, stopping trace")
                break
            
            if self.debug:
                print(f"Processing layer {layer} with {len(current)} targets")
                print(f"Target token weights: {current}")
            
            # Trace this layer
            srcs = backend.trace_layer(layer, current, self.sel_cfg)
            
            # Record source information
            for s in srcs:
                # Add layer information
                records.append({"layer": layer, **s})
            
            # Prepare targets for next layer
            next_dict = {s["index"]: s["weight"] for s in srcs}
            
            # Renormalize and prune
            current = SelectionStrategy.renormalize(
                next_dict, self.sel_cfg, apply_layer_prune=True
            )
            
            # Debug info
            if self.debug and current:
                print(f"Next layer targets: {len(current)} tokens")
                print(f"Total weight: {sum(current.values()):.4f}")
        
        # Add token type information from input data
        token_types = input_data.get("token_types", {})
        for record in records:
            idx = record["index"]
            if idx in token_types:
                record["type"] = token_types[idx]
        
        # Add logit lens projections if using saliency mode
        if mode == "saliency":
            # Get all token indices used in records
            all_indices = {r["index"] for r in records}.union(
                          {r["target"] for r in records})
            
            # Get logit lens backend and ensure hidden states
            logit_backend = self._get_backend("logit")
            logit_backend.ensure_hidden(input_data["inputs"])
            
            # Project tokens through LM head
            proj = logit_backend.project_tokens(
                0, list(all_indices), self.processor.tokenizer
            )
            
            # Add projection info to records
            for r in records:
                if r["index"] in proj:
                    pred = proj[r["index"]].get("predictions", [])
                    if pred:
                        r["pred_top"] = pred[0]["token"]
                        r["pred_prob"] = pred[0]["prob"]
        
        # Prepare metadata
        metadata = {
            "trace_id": trace_id,
            "mode": mode,
            "tracing_mode": mode,  # For compatibility with old format
            "token_types": input_data.get("token_types", {}),
            "target_tokens": [
                {
                    "index": idx,
                    "id": input_data["inputs"]["input_ids"][0][idx].item(),
                    "text": self.processor.tokenizer.decode([input_data["inputs"]["input_ids"][0][idx].item()]),
                    "weight": weight
                }
                for idx, weight in target_tokens.items()
            ],
            "input_length": input_data["inputs"]["input_ids"].shape[1],
            "image_available": "original_image" in input_data
        }
        
        # Save trace data to CSV
        csv_path = self.trace_io.write_trace_data(
            trace_id,
            records,
            metadata=metadata
        )
        
        return {
            "csv": csv_path,
            "records": len(records),
            "trace_id": trace_id,
            "mode": mode,
            "target_tokens": metadata["target_tokens"]
        }
    
    def analyze_specific_token(self, input_data: Dict[str, Any],
                             token_idx: int,
                             mode: str = "saliency") -> Dict[str, Any]:
        """
        Analyze a specific token without generation.
        
        Args:
            input_data: Prepared input data
            token_idx: Index of the token to analyze
            mode: Analysis mode ('saliency' or 'attention')
            
        Returns:
            Dictionary with analysis results
        """
        target_tokens = {token_idx: 1.0}
        
        # Get token information
        token_id = input_data["inputs"]["input_ids"][0, token_idx].item()
        token_text = self.processor.tokenizer.decode([token_id])
        
        print(f"Analyzing token at index {token_idx}: {token_text} (ID: {token_id})")
        
        # Perform tracing
        return self.trace(input_data, target_tokens, mode)
    
    def generate_and_trace(self, input_data: Dict[str, Any],
                         num_tokens: int = 1,
                         mode: str = "saliency") -> Dict[str, Any]:
        """
        Generate tokens and trace the information flow for each.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate
            mode: Analysis mode ('saliency' or 'attention')
            
        Returns:
            Dictionary with generation and analysis results
        """
        # Generate tokens
        print(f"Generating {num_tokens} tokens...")
        gen_results = self.autoregressive_generate(input_data["inputs"], num_tokens)
        
        # Update input data with generation results
        input_data["inputs"] = gen_results["inputs"]
        
        # Get starting position for generated tokens
        original_seq_len = gen_results["original_seq_len"]
        
        # Process each generated token
        all_results = []
        
        for i, token_info in enumerate(gen_results["generated_tokens"]):
            token_idx = token_info["index"]
            print(f"Tracing token {i+1}/{num_tokens}: {token_info['text']} at position {token_idx}")
            
            # Trace this token
            trace_id = f"{mode}_{token_idx}"
            result = self.trace(input_data, {token_idx: 1.0}, mode, trace_id)
            all_results.append(result)
            
            # Clear cache between tokens to save memory
            self.cache.clear()
        
        # Combine results
        combined = {
            "results": all_results,
            "generated_tokens": gen_results["generated_tokens"],
            "full_sequence": gen_results["full_sequence"],
            "original_seq_len": original_seq_len,
            "mode": mode
        }
        
        return combined