"""
Semantic tracing workflow for analyzing information flow in VLMs.
Coordinates the analysis process from input preparation to result generation.
"""

import os
import time
import torch
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from PIL import Image

from runtime.cache import TracingCache
from runtime.selection import SelectionConfig, SelectionStrategy
from runtime.io import TraceIO
from runtime.model_utils import get_llm_attention_layer_names
from backends.saliency_backend import SaliencyBackend
from backends.attention_backend import AttentionBackend
from backends.logit_backend import LogitBackend
from preprocess.input_builder import prepare_inputs
from runtime.generation import GenerationMixin


class SemanticTracingWorkflow(GenerationMixin):
    """
    Workflow for semantic tracing of information flow in VLMs.
    Analyzes how information propagates from input to output tokens.
    """
    
    def __init__(self, model, processor, output_dir: str,
                 selection_config: Optional[SelectionConfig] = None,
                 logit_lens_concepts: Optional[List[str]] = None,
                 debug: bool = False):
        """
        Initialize the semantic tracing workflow.
        
        Args:
            model: The model to analyze
            processor: The model's processor
            output_dir: Directory for saving results
            selection_config: Configuration for token selection/pruning
            logit_lens_concepts: List of concepts to track in logit lens
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
        self.logit_lens_concepts = logit_lens_concepts or ["cat", "dog", "person", "building", "water", "sky", "car"]
        
        # Image token ID for distinguishing between different token types
        self.image_token_id = getattr(model.config, "image_token_index", 32000)
        
        # Trace counter for unique IDs
        self.trace_id_counter = 0
        
        # Debug info
        if debug:
            print(f"Initialized SemanticTracingWorkflow with {len(self.layer_names)} layers")
            print(f"Device: {self.device}")
            print(f"Selection config: β_target={self.sel_cfg.beta_target}, β_layer={self.sel_cfg.beta_layer}")
            print(f"Output directory: {output_dir}")
            print(f"Logit lens concepts: {self.logit_lens_concepts}")
    
    def prepare_inputs(self, image: Union[str, Image.Image], prompt: str) -> Dict[str, Any]:
        """
        Prepare model inputs from image and prompt.
        
        Args:
            image: PIL image, path, or URL
            prompt: Text prompt
            
        Returns:
            Dictionary with prepared inputs and metadata
        """
        input_data = prepare_inputs(
            model=self.model,
            processor=self.processor,
            image=image,
            prompt=prompt
        )
        
        # Check if the token type mapping is created
        if "token_types" not in input_data:
            # Manually create token types mapping based on text and image indices
            if "text_indices" in input_data and "image_indices" in input_data:
                token_types = {}
                for idx in input_data["text_indices"]:
                    token_types[int(idx.item())] = "text"
                for idx in input_data["image_indices"]:
                    token_types[int(idx.item())] = "image"
                    
                input_data["token_types"] = token_types
                
        return input_data
    
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
                    self.model, self.layer_names, self.cache, self.device,
                    concepts=self.logit_lens_concepts
                )
            else:
                raise ValueError(f"Unknown backend mode: {mode}")
        
        return self.backends[mode]
    
    def _get_token_text(self, token_id: int) -> str:
        """
        Get text representation of a token, handling special tokens.
        
        Args:
            token_id: ID of the token
            
        Returns:
            Text representation of the token
        """
        # Handle special tokens 
        special_tokens = {13: "\\n", 28705: "_"}
        if token_id in special_tokens:
            return special_tokens[token_id]
        
        # Regular token decoding
        token_text = self.processor.tokenizer.decode([token_id])
        
        # Additional check for empty result (could be other special tokens)
        if not token_text.strip():
            return f"<tok_{token_id}>"
            
        return token_text
    
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
            self.trace_id_counter += 1
            trace_id = f"{mode}_{self.trace_id_counter}"
        
        # Get appropriate backend
        backend = self._get_backend(mode)
        
        # Ensure necessary data is cached
        backend.ensure_cache(input_data["inputs"], list(target_tokens.keys()))
        
        # Process layers from deepest to shallowest
        records = []
        current = target_tokens.copy()  # Make a copy to avoid modifying the original
        
        # Get token info for reporting
        all_token_ids = input_data["inputs"]["input_ids"][0].tolist()
        all_token_texts = [self._get_token_text(tid) for tid in all_token_ids]
        
        # Create token type masks for plotting/visualization
        seq_len = len(all_token_ids)
        token_types = [0] * seq_len  # Default: generated tokens
        text_indices = input_data.get("text_indices", [])
        image_indices = input_data.get("image_indices", [])
        
        # Set token types
        for idx in text_indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if 0 <= idx < seq_len:
                token_types[idx] = 1  # Text tokens
        
        for idx in image_indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if 0 <= idx < seq_len:
                token_types[idx] = 2  # Image tokens
        
        # Process layers from deepest to shallowest
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
            
            # If next_dict is empty but we found sources, add a fallback mechanism
            # This ensures tracing continues even if sources are sparse
            if not next_dict and current:
                print(f"[WARNING] Layer {layer} found no sources. Adding fallback targets.")
                # Use previous layer targets as fallback with reduced weights
                next_dict = {k: v * 0.5 for k, v in current.items()}
            
            # Renormalize and prune for next layer
            current = SelectionStrategy.renormalize(
                next_dict, self.sel_cfg, apply_layer_prune=True
            )
            
            # Debug info
            if self.debug and current:
                print(f"Next layer targets: {len(current)} tokens")
                print(f"Total weight: {sum(current.values()):.4f}")
    
        
        # Add token type information
        for record in records:
            idx = record["index"]
            if 0 <= idx < len(token_types):
                record["token_type"] = token_types[idx]
                
                # Add token text and ID if not present
                if "text" not in record and idx < len(all_token_texts):
                    record["text"] = all_token_texts[idx]
                    record["token_text"] = all_token_texts[idx]
                if "id" not in record and idx < len(all_token_ids):
                    record["token_id"] = all_token_ids[idx]
        
        # Enrich with logit lens projections (to add pred_top and pred_prob)
        all_unique_indices = set()
        for r in records:
            all_unique_indices.add(r["index"])
            if "target" in r:
                all_unique_indices.add(r["target"])
        
        # Get logit lens backend and ensure hidden states are available
        logit_backend = self._get_backend("logit")
        logit_backend.ensure_hidden(input_data["inputs"])
        
        # Get projections for all relevant tokens
        if all_unique_indices:
            try:
                projections = logit_backend.project_tokens(
                    0, list(all_unique_indices), self.processor.tokenizer
                )
                
                # Add projection info to records
                for r in records:
                    if r["index"] in projections:
                        predictions = projections[r["index"]].get("predictions", [])
                        if predictions:
                            r["predicted_top_token"] = self._sanitize_text_for_display(predictions[0]["token"])
                            r["predicted_top_prob"] = predictions[0]["prob"]
                            
                            # Also add concept probabilities if available
                            concepts = projections[r["index"]].get("concept_predictions", {})
                            for concept, data in concepts.items():
                                r[f"concept_{concept}_prob"] = data.get("probability", 0.0)
            except Exception as e:
                print(f"Warning: Failed to compute logit lens projections: {e}")
        
        # Calculate additional CSV metadata needed for compatibility
        for r in records:
            # Find if each token is a target
            r["is_target"] = r["index"] in target_tokens
            
            # Find which targets this token is a source for
            source_targets = []
            for other_record in records:
                if "target" in other_record and other_record["target"] == r["index"]:
                    source_targets.append(other_record["index"])
            
            r["source_for_targets"] = ",".join(map(str, source_targets))
            
            # Add source relationship for visualization
            if r["is_target"]:
                sources_indices = []
                sources_weights = []
                for other_record in records:
                    if "target" in other_record and other_record["index"] == r["target"]:
                        sources_indices.append(other_record["index"])
                        sources_weights.append(other_record["weight"])
                
                r["sources_indices"] = ",".join(map(str, sources_indices))
                r["sources_weights"] = ",".join(map(str, sources_weights))
                
            # Add trace_id
            r["trace_id"] = trace_id
            
            # Sanitize any text fields
            for key, value in r.items():
                if key.endswith("_text") and isinstance(value, str):
                    r[key] = self._sanitize_text_for_display(value)
        
        # 确保列名匹配老版本
        for r in records:
            # 修正列名以匹配老版本
            if "token_text" not in r and "text" in r:
                r["token_text"] = r["text"]
            if "token_id" not in r and "id" in r:
                r["token_id"] = r["id"]
            if "predicted_top_token" not in r:
                r["predicted_top_token"] = ""
            if "predicted_top_prob" not in r:
                r["predicted_top_prob"] = 0.0

            if mode == "saliency" and "trace_id" in r:
                if not str(r["trace_id"]).startswith("saliency_"):
                    r["trace_id"] = f"saliency_{r['trace_id']}"
        
        # Prepare metadata for saving
        metadata = {
            "trace_id": trace_id,
            "mode": mode,
            "tracing_mode": mode,  # For compatibility with old format
            "token_types": input_data.get("token_types", {}),
            "target_tokens": [
                {
                    "index": idx,
                    "id": input_data["inputs"]["input_ids"][0][idx].item(),
                    "text": self._get_token_text(input_data["inputs"]["input_ids"][0][idx].item()),
                    "weight": weight
                }
                for idx, weight in target_tokens.items()
            ],
            "input_length": input_data["inputs"]["input_ids"].shape[1],
            "image_available": "original_image" in input_data,
            "logit_lens_concepts": self.logit_lens_concepts,
        }
        
        # Add feature mapping for visualization if available
        if "feature_mapping" in input_data:
            metadata["feature_mapping"] = input_data["feature_mapping"]
        
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
            "target_tokens": metadata["target_tokens"],
            "metadata_path": os.path.join(self.output_dir, "csv_data", "trace_metadata.json")
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
        token_text = self._get_token_text(token_id)
        
        print(f"Analyzing token at index {token_idx}: '{token_text}' (ID: {token_id})")
        
        # Perform tracing
        return self.trace(input_data, target_tokens, mode)
    
    def analyze_last_token(self, input_data: Dict[str, Any],
                          mode: str = "saliency",
                          single_forward_pass: bool = False) -> Dict[str, Any]:
        """
        Analyze the last token in the prompt.
        
        Args:
            input_data: Prepared input data
            mode: Analysis mode ('saliency' or 'attention')
            single_forward_pass: Whether to use single forward pass optimization
            
        Returns:
            Dictionary with analysis results
        """
        # Get last token index
        seq_len = input_data["inputs"]["input_ids"].shape[1]
        target_idx = seq_len - 1
        
        # Get token information
        token_id = input_data["inputs"]["input_ids"][0, target_idx].item()
        token_text = self._get_token_text(token_id)
        
        print(f"Analyzing last token in prompt at index {target_idx}: '{token_text}' (ID: {token_id})")
        
        # Perform tracing
        return self.trace(input_data, {target_idx: 1.0}, mode)
    
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
        all_results = {
            "trace_results": {},
            "target_tokens": [],
            "full_sequence": gen_results["full_sequence"],
            "original_seq_len": original_seq_len,
            "metadata": {"tracing_mode": mode},
        }
        
        for i, token_info in enumerate(gen_results["generated_tokens"]):
            token_idx = token_info["index"]
            print(f"Tracing token {i+1}/{num_tokens}: {token_info['text']} at position {token_idx}")
            
            # Add to target tokens
            all_results["target_tokens"].append({
                "index": token_idx,
                "id": token_info["id"],
                "text": token_info["text"],
            })
            
            # Trace this token
            token_key = f"token_{token_idx}"
            all_results["trace_results"][token_key] = {}
            
            result = self.trace(input_data, {token_idx: 1.0}, mode)
            all_results["trace_results"][token_key][mode] = result
            
            # Clear cache between tokens to save memory
            self.cache.clear()
        
        # Save metadata for visualization
        metadata_path = os.path.join(self.output_dir, "csv_data", "trace_metadata.json")
        self.trace_io._write_unified_metadata({
            "tracing_mode": mode,
            "target_tokens": all_results["target_tokens"],
            "image_available": "original_image" in input_data,
            "logit_lens_concepts": self.logit_lens_concepts,
            "feature_mapping": input_data.get("feature_mapping", {})
        })
        all_results["metadata_path"] = metadata_path
        
        return all_results
    
    def generate_all_then_analyze_specific(
        self,
        input_data: Dict[str, Any],
        num_tokens: int = 10,
        analyze_indices: Optional[List[int]] = None,
        tracing_mode: str = "saliency",
        batch_compute: bool = True
    ) -> Dict[str, Any]:
        """
        First generates a complete sequence of tokens, then analyzes only specific token indices.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate in the complete sequence
            analyze_indices: Specific indices to analyze (None means analyze all generated tokens)
            tracing_mode: Analysis mode ('saliency', 'attention', or 'both')
            batch_compute: Whether to compute saliency in batches
            
        Returns:
            Dictionary with analysis results and full sequence information
        """
        # 1. Generate the complete sequence first
        print(f"\n=== Generating complete sequence of {num_tokens} tokens ===")
        inputs = input_data["inputs"]
        current_input_ids = inputs["input_ids"].clone()
        original_seq_len = current_input_ids.shape[1]
        
        # Generate tokens
        model = self.model
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
            analyze_indices = [idx for idx in analyze_indices if original_seq_len <= idx < current_input_ids.shape[1]]
        
        print(f"\n=== Analyzing {len(analyze_indices)} specific tokens ===")
        for i, idx in enumerate(analyze_indices):
            relative_pos = idx - original_seq_len
            if 0 <= relative_pos < len(generated_tokens):
                token = generated_tokens[relative_pos]
                print(f"Token {i+1}: '{token['text']}' at position {idx} (ID: {token['id']})")
        
        # 3. Set up results dictionary
        all_results = {
            "input_data": input_data,
            "target_tokens": [generated_tokens[idx - original_seq_len] for idx in analyze_indices if original_seq_len <= idx < current_input_ids.shape[1]],
            "all_generated_tokens": generated_tokens,
            "full_sequence": {
                "ids": current_input_ids[0].tolist(),
                "text": self.processor.tokenizer.decode(current_input_ids[0].tolist()),
                "generated_text": complete_text
            },
            "trace_results": {},
            "metadata": {"tracing_mode": tracing_mode}
        }
        
        # Update input_data with generation results
        input_data["inputs"] = inputs
        
        # 4. Analyze each requested token
        modes = [tracing_mode]
        if tracing_mode == "both":
            modes = ["saliency", "attention"]
        
        for target_idx in analyze_indices:
            if target_idx < original_seq_len or target_idx >= current_input_ids.shape[1]:
                print(f"Warning: Token index {target_idx} is outside the valid range. Skipping.")
                continue
            
            # Get token info
            token_id = current_input_ids[0, target_idx].item()
            token_text = self._get_token_text(token_id)
            print(f"\nAnalyzing token at index {target_idx}: '{token_text}' (ID: {token_id})")
            
            # Clear caches for fresh analysis
            self.cache.clear()
            
            # Create token-specific results container
            token_key = f"token_{target_idx}"
            all_results["trace_results"][token_key] = {}
            
            # Run each tracing mode
            for mode in modes:
                result = self.trace(input_data, {target_idx: 1.0}, mode)
                all_results["trace_results"][token_key][mode] = result
            
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 5. Save metadata for visualization
        metadata_path = os.path.join(self.output_dir, "csv_data", "trace_metadata.json")
        self.trace_io._write_unified_metadata({
            "tracing_mode": tracing_mode,
            "target_tokens": all_results["target_tokens"],
            "image_available": "original_image" in input_data,
            "logit_lens_concepts": self.logit_lens_concepts,
            "feature_mapping": input_data.get("feature_mapping", {})
        })
        all_results["metadata_path"] = metadata_path
        
        return all_results