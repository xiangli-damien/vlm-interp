"""
Semantic tracing workflow for VLM interpretability.
Traces token contributions through model layers.
"""

import torch
import os
import time
import gc
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from PIL import Image
from enum import Enum

from runtime.io import TraceIO
from preprocess.input_builder import prepare_inputs
from runtime.generation import GenerationMixin
from runtime.cache import TracingCache
from runtime.selection import SelectionStrategy
from preprocess.mapper import VisionMapper
from backends.attention_backend import AttentionBackend
from runtime.selection import SelectionConfig
from backends.logit_backend import LogitBackend
from backends.saliency_backend import SaliencyBackend
from runtime.decode import TokenDecoder
from analysis.semantic_viz import FlowGraphVisualizer

# Configure logging
logger = logging.getLogger("semantic_tracing.workflow")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TraceMode(Enum):
    """Tracing modes for semantic analysis."""
    ATTENTION = "attention"
    SALIENCY = "saliency"

class SourceMerger:
    """Utility class for merging sources from different backends."""
    
    @staticmethod
    def merge_sources(attention_sources: List[Dict[str, Any]], 
                     saliency_sources: List[Dict[str, Any]],
                     weight_factor: float = 0.5) -> List[Dict[str, Any]]:
        """
        Merge sources from attention and saliency backends.
        
        Args:
            attention_sources: List of source dictionaries from attention backend
            saliency_sources: List of source dictionaries from saliency backend
            weight_factor: Weight factor for combining scores (0.5 means equal weight)
            
        Returns:
            List of merged source dictionaries
        """
        # Create index map for quick lookup
        attention_map = {src["index"]: src for src in attention_sources}
        saliency_map = {src["index"]: src for src in saliency_sources}
        
        # Find all unique indices
        all_indices = set(attention_map.keys()) | set(saliency_map.keys())
        
        # Merge sources
        merged_sources = []
        
        for idx in all_indices:
            attn_src = attention_map.get(idx)
            saliency_src = saliency_map.get(idx)
            
            if attn_src and saliency_src:
                # Both backends have this source - use the provided weight factor
                merge_weight = weight_factor
            elif attn_src:
                # Only attention backend has this source
                merge_weight = 0.0
            else:
                # Only saliency backend has this source
                merge_weight = 1.0
                
            # Create merged source
            if attn_src and saliency_src:
                # Combine weights with the specified factor
                weight = (1 - merge_weight) * attn_src["weight"] + merge_weight * saliency_src["weight"]
                raw_score = (1 - merge_weight) * attn_src["raw_score"] + merge_weight * saliency_src["raw_score"]
                
                merged_src = {
                    "index": idx,
                    "weight": weight,
                    "raw_score": raw_score,
                    "target": attn_src.get("target", saliency_src.get("target")),
                    "type": attn_src.get("type", saliency_src.get("type", "unknown")),
                    "attn_weight": attn_src["weight"],
                    "saliency_weight": saliency_src["weight"],
                    "attn_score": attn_src["raw_score"],
                    "saliency_score": saliency_src["raw_score"],
                    "merge_weight": merge_weight
                }
            elif attn_src:
                # Only attention backend has this source
                merged_src = dict(attn_src)
                merged_src["merge_weight"] = merge_weight
                merged_src["saliency_weight"] = 0.0
                merged_src["saliency_score"] = 0.0
            else:
                # Only saliency backend has this source
                merged_src = dict(saliency_src)
                merged_src["merge_weight"] = merge_weight
                merged_src["attn_weight"] = 0.0
                merged_src["attn_score"] = 0.0
                
            merged_sources.append(merged_src)
            
        return merged_sources

class SemanticTracingWorkflow(GenerationMixin):
    """
    Workflow for semantic tracing analysis.
    Coordinates input preparation, tracing, and output processing.
    """

    def __init__(self, model: torch.nn.Module, processor: Any, output_dir: str, 
                 selection_config: Optional[SelectionConfig] = None,
                 debug: bool = False):
        """
        Initialize the semantic tracing workflow.

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

        if debug:
            logger.setLevel(logging.DEBUG)

        os.makedirs(output_dir, exist_ok=True)

        self.config = selection_config or SelectionConfig()

        self.device = next(model.parameters()).device

        try:
            from runtime.model_utils import get_llm_attention_layer_names
            self.layer_names = get_llm_attention_layer_names(model)
        except ImportError:
            logger.warning("Could not import get_llm_attention_layer_names. Using empty layer list.")
            self.layer_names = []

        if not self.layer_names:
            logger.warning("No attention layers found. Semantic tracing may not work.")
        else:
            logger.info(f"Found {len(self.layer_names)} attention layers.")

        self.io = TraceIO(output_dir)

        # Enable memory-safe CPU cache
        self.cache = TracingCache(cpu_offload=True)

        # Lazy backend registry (Attention/Saliency)
        self.backends = {}

        # Optional Logit Lens backend
        try:
            self.logit_backend = LogitBackend(
                model=self.model,
                cache=self.cache,
                device=self.device
            )
            logger.info("Logit Lens backend initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize Logit Lens backend: {e}")
            self.logit_backend = None

        self.token_decoder = TokenDecoder(self.processor)

        self.records_by_mode = {
            TraceMode.ATTENTION: [],
            TraceMode.SALIENCY: []
        }

        self.token_types = {}

    def _get_backend(self, mode: TraceMode):
        """
        Lazy initialization of attention or saliency backend.

        Args:
            mode: The trace mode to get backend for

        Returns:
            The appropriate backend instance
        """
        if mode not in self.backends:
            if mode == TraceMode.ATTENTION:
                from backends.attention_backend import AttentionBackend
                self.backends[mode] = AttentionBackend(
                    model=self.model,
                    layer_names=self.layer_names,
                    cache=self.cache,
                    device=self.device
                )
            elif mode == TraceMode.SALIENCY:
                from backends.saliency_backend import SaliencyBackend
                self.backends[mode] = SaliencyBackend(
                    model=self.model,
                    layer_names=self.layer_names,
                    cache=self.cache,
                    device=self.device
                )
            else:
                raise ValueError(f"Unknown tracing mode: {mode}")
        return self.backends[mode]


    def _get_token_text(self, token_id: int) -> str:
        """
        Get readable text for a token, handling special tokens properly.
        """
        return self.token_decoder.decode_token(token_id)
        
    def _initialize_backends(self) -> Dict[TraceMode, Any]:
        """
        Initialize backends using dynamic registry pattern.
        
        Returns:
            Dictionary mapping trace modes to backend instances
        """
        backends = {}
        
        # Initialize attention backend
        backends[TraceMode.ATTENTION] = AttentionBackend(
            model=self.model,
            layer_names=self.layer_names,
            cache=self.cache,
            device=self.device
        )
        
        # Initialize saliency backend
        backends[TraceMode.SALIENCY] = SaliencyBackend(
            model=self.model,
            layer_names=self.layer_names,
            cache=self.cache,
            device=self.device
        )
        
        # Try to initialize logit lens backend
        try:
            logit_backend = LogitBackend(
                model=self.model,
                cache=self.cache,
                device=self.device
            )
            self.logit_backend = logit_backend
            logger.info("Logit Lens backend initialized.")
        except Exception as e:
            logger.warning(f"Could not initialize Logit Lens backend: {e}")
            self.logit_backend = None
        
        return backends
        
    def prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """
        Prepare inputs for the model using keyword arguments.
        
        Args:
            image: Input image
            prompt: Text prompt
            
        Returns:
            Dictionary of prepared inputs and metadata
        """
        # Use keyword arguments to match prepare_inputs signature
        input_data = prepare_inputs(
            model=self.model,
            processor=self.processor,
            image=image,
            prompt=prompt
        )
        
        # Store token types for later use
        self.token_types = input_data["token_types"]
        
        return input_data
            
    def generate(self, input_data: Dict[str, Any], num_tokens: int = 1) -> Dict[str, Any]:
        """
        Generate tokens from the model.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate
            
        Returns:
            Dictionary with generation results
        """
        gen_result = self.autoregressive_generate(input_data["inputs"], num_tokens)
        
        # Update token types with generated tokens
        for token in gen_result["generated_tokens"]:
            self.token_types[token["index"]] = "generated"
            
        # Create a result that includes original input_data elements plus generation
        result = {
            "input_data": {
                "inputs": gen_result["inputs"],
                "text_indices": input_data["text_indices"],
                "image_indices": input_data["image_indices"],
                "token_types": self.token_types,
                "formatted_prompt": input_data["formatted_prompt"],
                "original_image": input_data["original_image"],
                "original_image_size_hw": (input_data["original_image"].height, 
                                          input_data["original_image"].width)
            },
            "full_sequence": gen_result["full_sequence"],
            "generated_tokens": gen_result["generated_tokens"],
            "original_seq_len": gen_result["original_seq_len"]
        }
        
        return result
        
    def trace(self, input_data: Dict[str, Any], target_tokens: Dict[int, float], 
         mode: Union[str, TraceMode] = TraceMode.SALIENCY, 
         trace_id: str = None,
         single_forward_pass: bool = False,
         compute_ll_projections: bool = True) -> Dict[str, Any]:
        """
        Trace token contributions through model layers.
        
        Args:
            input_data: Prepared input data
            target_tokens: Dictionary mapping target token indices to weights
            mode: Tracing mode (attention or saliency)
            trace_id: Unique identifier for this trace
            single_forward_pass: Whether to use a single forward pass
            compute_ll_projections: Whether to compute Logit Lens projections
            
        Returns:
            Dictionary containing trace results
        """
        # Create trace ID if not provided
        if trace_id is None:
            trace_id = f"trace_{int(time.time())}"
        
        # Clear previous records for the selected mode
        for mode_key in self.records_by_mode:
            self.records_by_mode[mode_key] = []
        
        # Convert mode string to enum if needed
        if isinstance(mode, str):
            try:
                mode = TraceMode(mode.lower())
            except ValueError:
                # Default to saliency if invalid mode provided
                logger.warning(f"Invalid tracing mode: {mode}. Using saliency mode instead.")
                mode = TraceMode.SALIENCY

        # Initialize the backend if it doesn't exist yet
        if mode not in self.backends:
            logger.info(f"Initializing {mode.value} backend on first use")
            if mode == TraceMode.ATTENTION:
                self.backends[mode] = AttentionBackend(
                    model=self.model,
                    layer_names=self.layer_names,
                    cache=self.cache,
                    device=self.device
                )
            elif mode == TraceMode.SALIENCY:
                self.backends[mode] = SaliencyBackend(
                    model=self.model,
                    layer_names=self.layer_names,
                    cache=self.cache,
                    device=self.device
                )
        
        # Get the appropriate backend based on the mode
        backend = self.backends[mode]
        layer_traces = []
        
        # Prepare token information for records - needed for token_text and token_id fields
        input_ids = input_data["inputs"]["input_ids"][0].tolist()
        all_token_texts = [self._get_token_text(tid) for tid in input_ids]
        
        # If using single forward pass, ensure all activations are cached once at the beginning
        if single_forward_pass:
            logger.info(f"Using single forward pass for {mode.value} backend")
            backend.ensure_cache(
                input_data["inputs"],
                target_indices=list(target_tokens.keys()),
                single_pass=True
            )
            
        # Get the backend's layer map to ensure correct indexing
        layer_name_map = getattr(backend, "layer_name_map", {})
        
        # Trace through layers from last to first
        current_targets = dict(target_tokens)  # Copy to avoid modifying the original
        
        for layer_idx in reversed(range(len(self.layer_names))):
            # Skip if no targets left
            if not current_targets:
                break
                
            # Ensure necessary activations are cached for this specific layer
            # but only if not already cached by single_forward_pass
            if not single_forward_pass:
                backend.ensure_cache(
                    input_data["inputs"],
                    target_indices=list(target_tokens.keys())
                )
                    
            # Get the corresponding layer name if layer map is available
            layer_name = layer_name_map.get(layer_idx, None)
            if layer_name is None:
                # Fall back to direct index if no map available
                layer_name = str(layer_idx)
                    
            # Trace this layer
            sources = backend.trace_layer(layer_idx, current_targets, self.config)
            
            # Skip if no sources found
            if not sources:
                continue
                
            # Add token types if available
            for source in sources:
                idx = source["index"]
                source["type"] = self.token_types.get(idx, "unknown")
            
            # Compute Logit Lens projections if requested
            ll_projections = None
            if compute_ll_projections and self.logit_backend is not None:
                # Get unique token IDs from current targets and sources
                token_indices = set(current_targets.keys()) | {s["index"] for s in sources}
                if token_indices:
                    ll_projections = {}
                    
                    # Optimize: use batch projection if available
                    if hasattr(self.logit_backend, "project_tokens_batch"):
                        # Use optimized batch projection
                        batch_results = self.logit_backend.project_tokens_batch(
                            layer_idx=layer_idx,
                            token_indices=list(token_indices),
                            tokenizer=self.processor.tokenizer,
                            top_k=3
                        )
                        
                        # Process batch results
                        for token_idx, projection in batch_results.items():
                            if projection and projection.get("predictions"):
                                ll_projections[token_idx] = {
                                    "top1_token": projection["predictions"][0].get("text", ""),
                                    "top1_prob": projection["predictions"][0].get("prob", 0.0),
                                    "top1_logit": projection["predictions"][0].get("logit", 0.0),
                                    "top2_token": projection["predictions"][1].get("text", "") if len(projection["predictions"]) > 1 else "",
                                    "top2_prob": projection["predictions"][1].get("prob", 0.0) if len(projection["predictions"]) > 1 else 0.0
                                }
                    else:
                        # Fallback to individual token projection
                        for token_idx in token_indices:
                            projection = self.logit_backend.project_token(
                                layer_idx=layer_idx,
                                token_idx=token_idx,
                                tokenizer=self.processor.tokenizer,
                                top_k=3
                            )
                            if projection and projection.get("predictions"):
                                ll_projections[token_idx] = {
                                    "top1_token": projection["predictions"][0].get("text", ""),
                                    "top1_prob": projection["predictions"][0].get("prob", 0.0),
                                    "top1_logit": projection["predictions"][0].get("logit", 0.0),
                                    "top2_token": projection["predictions"][1].get("text", "") if len(projection["predictions"]) > 1 else "",
                                    "top2_prob": projection["predictions"][1].get("prob", 0.0) if len(projection["predictions"]) > 1 else 0.0
                                }
            
            # Create layer trace and append to records
            layer_trace = {
                "layer_idx": layer_idx,
                "targets": current_targets,
                "sources": sources,
                "ll_projections": ll_projections,
                "mode": mode.value
            }
            
            layer_traces.append(layer_trace)
            
            # Group sources by target for proper record creation
            sources_by_target = {}
            for source in sources:
                target_idx = source.get("target", -1)
                if target_idx not in sources_by_target:
                    sources_by_target[target_idx] = []
                sources_by_target[target_idx].append(source)
            
            # Create records for all tokens involved in this layer (targets AND sources)
            all_token_indices = set()
            for source in sources:
                all_token_indices.add(source["index"])
                if "target" in source:
                    all_token_indices.add(source["target"])
                    
            # Also add current targets as tokens
            for target_idx in current_targets:
                all_token_indices.add(target_idx)
                
            # Make sure all indices are valid
            all_token_indices = [idx for idx in all_token_indices if 0 <= idx < len(input_ids)]
            
            # Create enhanced records for all tokens involved
            for token_idx in all_token_indices:
                # Determine if this token is a target
                is_target = token_idx in current_targets
                
                # Get token info
                token_id = input_ids[token_idx] if token_idx < len(input_ids) else -1
                token_text = all_token_texts[token_idx] if token_idx < len(all_token_texts) else self._get_token_text(token_id)
                token_type = self.token_types.get(token_idx, "unknown") 
                
                # Collect source targets - i.e., which targets does this token act as a source for?
                source_for_targets = []
                for target in sources_by_target:
                    if any(s["index"] == token_idx for s in sources_by_target[target]):
                        source_for_targets.append(target)
                
                # Create source indices/weights list, but only for target tokens
                sources_indices = []
                sources_weights = []
                if is_target and token_idx in sources_by_target:
                    sources_indices = [s["index"] for s in sources_by_target[token_idx]]
                    sources_weights = [s["weight"] for s in sources_by_target[token_idx]]
                
                # Create record with ALL fields from old version for compatibility
                record = {
                    "layer": layer_idx,
                    "token_index": token_idx,
                    "token_text": token_text,
                    "token_id": token_id,
                    "token_type": token_type,
                    "is_target": is_target,
                    "source_idx": token_idx,  # For compatibility with new fields
                    "target_idx": token_idx if is_target else -1,  # For compatibility with new fields
                    "source_for_targets": ",".join(map(str, source_for_targets)),
                    "sources_indices": ",".join(map(str, sources_indices)),
                    "sources_weights": ",".join(map(str, sources_weights)),
                    "weight": current_targets.get(token_idx, 0.0) if is_target else 0.0,
                    "raw_score": 0.0,  # Default for consistency
                    "mode": mode.value,
                    "type": token_type
                }
                
                # Add source-specific data if this token is a source
                if source_for_targets:
                    # Find any source record for this token
                    for target_idx in source_for_targets:
                        source_record = next((s for s in sources_by_target[target_idx] if s["index"] == token_idx), None)
                        if source_record:
                            record["weight"] = source_record["weight"]
                            record["raw_score"] = source_record["raw_score"]
                            break
                
                # Add Logit Lens projection data if available
                if ll_projections and token_idx in ll_projections:
                    ll_data = ll_projections[token_idx]
                    record.update({
                        "predicted_top_token": ll_data.get("top1_token", ""),
                        "predicted_top_prob": ll_data.get("top1_prob", 0.0),
                        "ll_top1_token": ll_data.get("top1_token", ""),
                        "ll_top1_prob": ll_data.get("top1_prob", 0.0),
                        "ll_top1_logit": ll_data.get("top1_logit", 0.0),
                        "ll_top2_token": ll_data.get("top2_token", ""),
                        "ll_top2_prob": ll_data.get("top2_prob", 0.0)
                    })
                else:
                    # Add empty prediction fields for consistency
                    record.update({
                        "predicted_top_token": "",
                        "predicted_top_prob": 0.0,
                        "ll_top1_token": "",
                        "ll_top1_prob": 0.0,
                        "ll_top1_logit": 0.0,
                        "ll_top2_token": "",
                        "ll_top2_prob": 0.0
                    })
                
                # Append to the appropriate records list
                self.records_by_mode[mode].append(record)
            
            # Update targets for next layer
            current_targets = {s["index"]: s["weight"] for s in sources}
            current_targets = SelectionStrategy.renormalize(current_targets, self.config, apply_layer_prune=True)

            # Clear the cache
            logger.debug(f"Completed layer {layer_idx}. Forcing GC and CUDA cache clear.")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save results
        results = {}
        if self.records_by_mode[mode]:
            csv_path = self.io.write_trace_data(
                trace_id=trace_id,
                records=self.records_by_mode[mode],
                metadata={
                    "mode": mode.value, 
                    "tracing_mode": mode.value,
                    "target_tokens": target_tokens,
                    "logit_lens_concepts": getattr(self.logit_backend, "concepts", []),
                    "image_available": "original_image" in input_data,
                    "feature_mapping": input_data.get("feature_mapping", {})
                }
            )
            results = {
                "trace_data_path": csv_path,
                "csv_path": csv_path,  # Add explicit CSV path for UI
                "num_records": len(self.records_by_mode[mode]),
                "layer_traces": layer_traces
            }
            
            # Clear records after saving
            self.records_by_mode[mode] = []

        # Add metadata
        results["metadata"] = {
            "trace_id": trace_id,
            "mode": mode.value,
            "tracing_mode": mode.value,
            "target_tokens": target_tokens,
            "num_layers": len(self.layer_names),
            "logit_lens_concepts": getattr(self.logit_backend, "concepts", []),
            "image_available": "original_image" in input_data,
            "feature_mapping": input_data.get("feature_mapping", {})
        }
                
        return results
    
    
    def generate_and_trace(self, input_data: Dict[str, Any], num_tokens: int = 1,
                     mode: Union[str, TraceMode] = TraceMode.SALIENCY, 
                     single_forward_pass: bool = False,
                     compute_ll_projections: bool = True) -> Dict[str, Any]:
        """
        Generate tokens and trace their semantic contributions.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate
            mode: Tracing mode (attention or saliency)
            single_forward_pass: Whether to use a single forward pass
            compute_ll_projections: Whether to compute Logit Lens projections
            
        Returns:
            Dictionary containing generation and trace results
        """
        try:
            # First generate tokens
            logger.info(f"Starting generate-and-trace workflow: {num_tokens} tokens, {mode} mode")
            gen_results = self.generate(input_data, num_tokens)
            
            # Get indices of generated tokens
            target_indices = [t["index"] for t in gen_results["generated_tokens"]]
            
            # Create target tokens dictionary
            target_tokens = {idx: 1.0 for idx in target_indices}
            
            # Then trace them
            trace_results = self.trace(
                input_data=gen_results["input_data"],
                target_tokens=target_tokens,
                mode=mode,
                single_forward_pass=single_forward_pass,
                compute_ll_projections=compute_ll_projections
            )
            
            # Combine results
            result = {
                "generation_results": gen_results,
                "trace_results": trace_results,
                "target_tokens": target_tokens,
                "target_indices": target_indices
            }
            
            logger.info("Generate-and-trace workflow complete")
            return result
            
        except Exception as e:
            logger.error(f"Error during generate-and-trace: {e}")
            raise
        
    def analyze_specific_token(self, input_data: Dict[str, Any], token_idx: int,
                              mode: Union[str, TraceMode] = TraceMode.SALIENCY,
                              single_forward_pass: bool = False,
                              compute_ll_projections: bool = True) -> Dict[str, Any]:
        """
        Analyze a specific token by index.
        
        Args:
            input_data: Prepared input data
            token_idx: Index of the token to analyze
            mode: Tracing mode (attention, saliency, or both)
            single_forward_pass: Whether to use a single forward pass
            compute_ll_projections: Whether to compute Logit Lens projections
            
        Returns:
            Dictionary containing trace results
        """
        try:
            # Validate token index
            if token_idx >= input_data["inputs"]["input_ids"].shape[1]:
                raise ValueError(f"Token index {token_idx} out of bounds for sequence length {input_data['inputs']['input_ids'].shape[1]}")
            
            # Create token info
            token_id = input_data["inputs"]["input_ids"][0, token_idx].item()
            token_text = self.processor.tokenizer.decode([token_id])
            
            logger.info(f"Analyzing specific token index {token_idx} ('{token_text}') using {mode} mode")
            
            token_info = {
                "index": token_idx,
                "id": token_id,
                "text": token_text,
                "type": self.token_types.get(token_idx, "unknown")
            }
            
            # Set up target tokens dictionary
            target_tokens = {token_idx: 1.0}
            
            # Trace the token
            trace_results = self.trace(
                input_data=input_data,
                target_tokens=target_tokens,
                mode=mode,
                single_forward_pass=single_forward_pass,
                compute_ll_projections=compute_ll_projections
            )
            
            # Combine results
            return {
                "trace_results": trace_results,
                "target_token": token_info
            }
            
        except Exception as e:
            logger.error(f"Error analyzing specific token: {e}")
            raise
        
    def analyze_last_token(self, input_data: Dict[str, Any], 
                          mode: Union[str, TraceMode] = TraceMode.SALIENCY,
                          single_forward_pass: bool = False,
                          compute_ll_projections: bool = True) -> Dict[str, Any]:
        """
        Analyze the last token in the prompt.
        
        Args:
            input_data: Prepared input data
            mode: Tracing mode (attention, saliency, or both)
            single_forward_pass: Whether to use a single forward pass
            compute_ll_projections: Whether to compute Logit Lens projections
            
        Returns:
            Dictionary containing trace results
        """
        # Get index of last token
        last_idx = input_data["inputs"]["input_ids"].shape[1] - 1
        logger.info(f"Analyzing last token at index {last_idx}")
        
        # Analyze it
        return self.analyze_specific_token(
            input_data=input_data,
            token_idx=last_idx,
            mode=mode,
            single_forward_pass=single_forward_pass,
            compute_ll_projections=compute_ll_projections
        )
        
    def analyze_multiple_tokens(self, input_data: Dict[str, Any], 
                              token_indices: List[int],
                              mode: Union[str, TraceMode] = TraceMode.SALIENCY,
                              weights: Optional[Dict[int, float]] = None,
                              single_forward_pass: bool = False,
                              compute_ll_projections: bool = True) -> Dict[str, Any]:
        """
        Analyze multiple tokens together.
        
        Args:
            input_data: Prepared input data
            token_indices: List of token indices to analyze
            mode: Tracing mode (attention, saliency, or both)
            weights: Optional dictionary mapping token indices to weights
            single_forward_pass: Whether to use a single forward pass
            compute_ll_projections: Whether to compute Logit Lens projections
            
        Returns:
            Dictionary containing trace results
        """
        try:
            # Validate token indices
            max_idx = input_data["inputs"]["input_ids"].shape[1] - 1
            valid_indices = [idx for idx in token_indices if 0 <= idx <= max_idx]
            
            if len(valid_indices) < len(token_indices):
                invalid = set(token_indices) - set(valid_indices)
                logger.warning(f"Ignoring {len(invalid)} out-of-bounds token indices: {invalid}")
                
            if not valid_indices:
                logger.error("No valid token indices provided")
                return {"error": "No valid token indices provided"}
                
            # Create token info list
            token_info_list = []
            for idx in valid_indices:
                token_id = input_data["inputs"]["input_ids"][0, idx].item()
                token_text = self.processor.tokenizer.decode([token_id])
                token_info_list.append({
                    "index": idx,
                    "id": token_id,
                    "text": token_text,
                    "type": self.token_types.get(idx, "unknown")
                })
                
            logger.info(f"Analyzing {len(valid_indices)} tokens using {mode} mode")
            
            # Set up target tokens dictionary with weights
            if weights is None:
                # Equal weights if not provided
                target_tokens = {idx: 1.0 / len(valid_indices) for idx in valid_indices}
            else:
                # Use provided weights and normalize
                target_tokens = {idx: weights.get(idx, 1.0) for idx in valid_indices}
                total = sum(target_tokens.values())
                if total > 0:
                    target_tokens = {idx: weight / total for idx, weight in target_tokens.items()}
                    
            # Trace the tokens
            trace_results = self.trace(
                input_data=input_data,
                target_tokens=target_tokens,
                mode=mode,
                single_forward_pass=single_forward_pass,
                compute_ll_projections=compute_ll_projections
            )
            
            # Combine results
            return {
                "trace_results": trace_results,
                "target_tokens": target_tokens,
                "target_token_info": token_info_list
            }
            
        except Exception as e:
            logger.error(f"Error analyzing multiple tokens: {e}")
            raise

    def visualize_trace_results(self, csv_path: str, output_dir: Optional[str] = None,
                          min_edge_weight: float = 0.0005,
                          show_orphaned_nodes: bool = False,
                          max_nodes_per_layer: int = 1000) -> List[str]:
        """
        Generate visualization from a trace CSV file.
        
        Args:
            csv_path: Path to the CSV file containing trace data
            output_dir: Optional output directory (defaults to csv directory with '_viz' suffix)
            min_edge_weight: Minimum edge weight to display
            show_orphaned_nodes: Whether to show nodes with no connections
            max_nodes_per_layer: Maximum number of nodes per layer
            
        Returns:
            List of paths to generated visualization files
        """
        # Set default output directory if not provided
        if output_dir is None:
            csv_dir = os.path.dirname(csv_path)
            output_dir = os.path.join(csv_dir, "visualizations")
        
        # Create visualizer instance
        visualizer = FlowGraphVisualizer(
            output_dir=output_dir, 
            debug_mode=self.debug
        )
        
        # Check if CSV exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return []
        
        # Process CSV and generate visualization
        try:
            # Load and preprocess CSV
            df = visualizer._preprocess_csv(csv_path)
            
            # Try to load metadata from default location
            metadata_path = os.path.join(os.path.dirname(csv_path), "trace_metadata.json")
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.info(f"Loaded metadata from {metadata_path}")
                except Exception as e:
                    logger.warning(f"Error loading metadata: {e}")
            
            # Extract target token info
            target_token = visualizer._extract_target_token_info(df, metadata)
            target_idx = target_token.get("index", "unknown")
            target_text = target_token.get("text", "unknown")
            
            # Create save directory with clean token name
            target_text_clean = "".join(c if c.isalnum() else "_" for c in str(target_text)).strip("_")
            save_dir = os.path.join(output_dir, f"token_{target_idx}_{target_text_clean}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Generate the visualization
            vis_paths = visualizer.create_cytoscape_flow_graph_from_csv(
                trace_data=df,
                target_text=target_text,
                target_idx=target_idx,
                save_dir=save_dir,
                min_edge_weight=min_edge_weight,
                max_nodes_per_layer=max_nodes_per_layer,
                layout_name="grid",
                show_orphaned_nodes=show_orphaned_nodes
            )
            
            logger.info(f"Generated {len(vis_paths)} visualization files in {save_dir}")
            return vis_paths
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            import traceback
            traceback.print_exc()
            return []