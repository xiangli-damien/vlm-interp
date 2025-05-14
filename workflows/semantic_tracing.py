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
    def merge(self, other_cache: 'TracingCache') -> None:
        """
        Merge another cache into this one, keeping only the highest priority data.
        
        Args:
            other_cache: Another TracingCache instance to merge from
        """
        # Merge all cache types in priority order: saliency > grad > attention > hidden
        # Process each layer index
        all_layer_indices = set(
            list(self.saliency.keys()) + 
            list(self.grad.keys()) + 
            list(self.attention.keys()) + 
            list(self.hidden_states.keys()) +
            list(other_cache.saliency.keys()) + 
            list(other_cache.grad.keys()) + 
            list(other_cache.attention.keys()) + 
            list(other_cache.hidden_states.keys())
        )
        
        for layer_idx in all_layer_indices:
            # Priority 1: Saliency (pre-computed result)
            if other_cache.has(layer_idx, "saliency"):
                self.set(layer_idx, "saliency", other_cache.get(layer_idx, "saliency"), detach=False)
                # Since we have saliency, we don't need attention or grad tensors
                self.clear_single(layer_idx, "attention")
                self.clear_single(layer_idx, "grad")
            else:
                # Priority 2 & 3: Grad and Attention (needed for computing saliency)
                if other_cache.has(layer_idx, "grad"):
                    self.set(layer_idx, "grad", other_cache.get(layer_idx, "grad"), detach=False)
                if other_cache.has(layer_idx, "attention"):
                    self.set(layer_idx, "attention", other_cache.get(layer_idx, "attention"), detach=False)
                
                # Priority 4: Hidden states (only keep if not already present)
                if not self.has(layer_idx, "hidden") and other_cache.has(layer_idx, "hidden"):
                    self.set(layer_idx, "hidden", other_cache.get(layer_idx, "hidden"), detach=False)
        
        # Merge custom objects
        for tag, obj in other_cache.custom.items():
            if tag not in self.custom:
                self.custom[tag] = obj
                
        # Merge gradient missing flags
        for idx, missing in other_cache.grad_missing.items():
            if idx not in self.grad_missing:
                self.grad_missing[idx] = missing

class SemanticTracingWorkflow(GenerationMixin):
    """
    Workflow for semantic tracing analysis.
    Coordinates input preparation, tracing, and output processing.
    """

    def __init__(self, model: torch.nn.Module, processor: Any, output_dir: str, 
             selection_config: Optional[SelectionConfig] = None,
             debug: bool = False, 
             logit_lens_concepts: Optional[List[str]] = None):
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
        self.logit_lens_concepts = logit_lens_concepts or [
            "eiffel", "tower", "paris"
        ]

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
                device=self.device,
                concepts=self.logit_lens_concepts
            )
            logger.info(f"Logit Lens backend initialized with {len(self.logit_lens_concepts)} concepts")
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
        Memory-optimized tracing that minimizes GPU memory usage.
        
        Args:
            input_data: Prepared input data
            target_tokens: Dictionary mapping target token indices to weights
            mode: Tracing mode (attention or saliency)
            trace_id: Unique identifier for this trace
            single_forward_pass: Whether to use a single forward pass (always overridden to False for memory efficiency)
            compute_ll_projections: Whether to compute Logit Lens projections
            
        Returns:
            Dictionary containing trace results
        """
        # MEMORY OPTIMIZATION: Always override single_forward_pass to be False
        # to avoid memory issues with large backpropagation through all layers
        single_forward_pass = False
        
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
                logger.warning(f"Invalid tracing mode: {mode}. Using saliency mode instead.")
                mode = TraceMode.SALIENCY

        # Initialize the backend if it doesn't exist yet
        if mode not in self.backends:
            logger.info(f"Initializing {mode.value} backend on first use")
            self._get_backend(mode)
        
        # Get the appropriate backend based on the mode
        backend = self.backends[mode]
        layer_traces = []
        
        # Prepare token information for records
        input_ids = input_data["inputs"]["input_ids"][0].tolist()
        all_token_texts = [self._get_token_text(tid) for tid in input_ids]
        
        # Force memory cleanup before starting layer processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # MEMORY OPTIMIZATION: Process individual layers in reverse order
        current_targets = dict(target_tokens)  # Copy to avoid modifying the original
        
        # Get the backend's layer map to ensure correct indexing
        layer_name_map = getattr(backend, "layer_name_map", {})
        
        # Clear any previous cache data completely to prevent memory leaks
        self.cache.clear()
        
        # OPTIMIZATION: Pre-cache important inputs with minimal necessary data
        minimal_inputs = {}
        for k, v in input_data["inputs"].items():
            if isinstance(v, torch.Tensor):
                if v.dtype in [torch.float32, torch.float64] and v.numel() > 1000:
                    # Use half precision for large float tensors
                    minimal_inputs[k] = v.to(torch.float16)
                else:
                    # Keep original dtype for small tensors and integer types
                    minimal_inputs[k] = v
            else:
                minimal_inputs[k] = v
        
        # MEMORY OPTIMIZATION: Setup progressive layer processing
        total_layers = len(self.layer_names)
        
        # Ensure cache is setup for the mode we're using
        backend.ensure_cache(minimal_inputs, list(current_targets.keys()), single_pass=False)
        
        # Process each layer individually from last to first (important for causality)
        for layer_idx in reversed(range(total_layers)):
            # Skip if no targets left
            if not current_targets:
                logger.warning("No target tokens remaining, stopping trace early")
                break
            
            logger.info(f"Processing layer {layer_idx} with {len(current_targets)} target tokens")
                
            # OPTIMIZATION: Trace this layer
            try:
                sources = backend.trace_layer(layer_idx, current_targets, self.config)
                
                # Skip if no sources found
                if not sources:
                    logger.warning(f"No sources found for layer {layer_idx}, continuing with next layer")
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
                        # Create TokenDecoder for better token decoding
                        from runtime.decode import TokenDecoder
                        token_decoder = TokenDecoder(self.processor.tokenizer)
                        
                        # Process LogitLens in small batches
                        ll_projections = {}
                        token_indices_list = sorted(list(token_indices))
                        
                        # Use batch processing to reduce memory usage
                        batch_size = 8
                        for i in range(0, len(token_indices_list), batch_size):
                            batch_indices = token_indices_list[i:i+batch_size]
                            
                            try:
                                # Try using batch projection method
                                if hasattr(self.logit_backend, "project_tokens_batch"):
                                    batch_results = self.logit_backend.project_tokens_batch(
                                        layer_idx=layer_idx,
                                        token_indices=batch_indices,
                                        tokenizer=self.processor.tokenizer,
                                        top_k=5
                                    )
                                    
                                    # Process batch results
                                    for token_idx, projection in batch_results.items():
                                        if projection and projection.get("predictions"):
                                            # Get top prediction
                                            top_pred = projection["predictions"][0]
                                            
                                            # Use TokenDecoder for better token text
                                            top_text = token_decoder.decode_token(top_pred.get("token_id", 0))
                                            
                                            # Store with proper type conversion
                                            ll_projections[token_idx] = {
                                                "top1_token": top_text,
                                                "top1_prob": float(top_pred.get("prob", 0.0)),
                                                "top1_logit": float(top_pred.get("logit", 0.0)),
                                                "top2_token": token_decoder.decode_token(projection["predictions"][1].get("token_id", 0)) if len(projection["predictions"]) > 1 else "",
                                                "top2_prob": float(projection["predictions"][1].get("prob", 0.0)) if len(projection["predictions"]) > 1 else 0.0
                                            }
                                            
                                            # Log for debugging
                                            print(f"Token {token_idx} → '{top_text}' (prob: {top_pred.get('prob', 0.0):.4f})")
                                else:
                                    # Fall back to individual token projection
                                    for token_idx in batch_indices:
                                        projection = self.logit_backend.project_token(
                                            layer_idx=layer_idx,
                                            token_idx=token_idx,
                                            tokenizer=self.processor.tokenizer,
                                            top_k=5
                                        )
                                        
                                        if projection and projection.get("predictions"):
                                            # Get top prediction
                                            top_pred = projection["predictions"][0]
                                            
                                            # Use TokenDecoder for better token text
                                            top_text = token_decoder.decode_token(top_pred.get("token_id", 0))
                                            
                                            ll_projections[token_idx] = {
                                                "top1_token": top_text,
                                                "top1_prob": float(top_pred.get("prob", 0.0)),
                                                "top1_logit": float(top_pred.get("logit", 0.0)),
                                                "top2_token": token_decoder.decode_token(projection["predictions"][1].get("token_id", 0)) if len(projection["predictions"]) > 1 else "",
                                                "top2_prob": float(projection["predictions"][1].get("prob", 0.0)) if len(projection["predictions"]) > 1 else 0.0
                                            }
                            except Exception as e:
                                print(f"Error during LogitLens projection: {e}")
                                import traceback
                                traceback.print_exc()
                            
                            # Force cleanup after each batch
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                # Create layer trace and append to records
                layer_trace = {
                    "layer_idx": layer_idx,
                    "targets": current_targets,
                    "sources": sources,
                    "ll_projections": ll_projections,
                    "mode": mode.value
                }
                
                layer_traces.append(layer_trace)
                
                # Group sources by target for record creation
                sources_by_target = {}
                for source in sources:
                    target_idx = source.get("target", -1)
                    if target_idx not in sources_by_target:
                        sources_by_target[target_idx] = []
                    sources_by_target[target_idx].append(source)
                
                # Create records for all tokens involved in this layer
                all_token_indices = set()
                for source in sources:
                    all_token_indices.add(source["index"])
                    if "target" in source:
                        all_token_indices.add(source["target"])
                        
                # Also add current targets
                for target_idx in current_targets:
                    all_token_indices.add(target_idx)
                    
                # Filter valid indices
                all_token_indices = [idx for idx in all_token_indices if 0 <= idx < len(input_ids)]
                
                # Create records in small batches
                for token_idx in all_token_indices:
                    # Determine if this token is a target
                    is_target = token_idx in current_targets
                    
                    # Get token info
                    token_id = input_ids[token_idx] if token_idx < len(input_ids) else -1
                    token_text = all_token_texts[token_idx] if token_idx < len(all_token_texts) else self._get_token_text(token_id)
                    token_type = self.token_types.get(token_idx, "unknown") 
                    
                    # Process source relationships
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
                    
                    # Create record (with all fields for compatibility)
                    record = {
                        "layer": layer_idx,
                        "token_index": token_idx,
                        "token_text": token_text,
                        "token_id": token_id,
                        "token_type": token_type,
                        "is_target": is_target,
                        "source_idx": token_idx,
                        "target_idx": token_idx if is_target else -1,
                        "source_for_targets": ",".join(map(str, source_for_targets)),
                        "sources_indices": ",".join(map(str, sources_indices)),
                        "sources_weights": ",".join(map(str, sources_weights)),
                        "weight": current_targets.get(token_idx, 0.0) if is_target else 0.0,
                        "raw_score": 0.0,
                        "mode": mode.value,
                        "type": token_type
                    }

                    self._process_logit_lens_projections(token_idx, record, ll_projections)

                    self.records_by_mode[mode].append(record)
                                    
                # Update targets for next layer - use selection strategy to control growth
                current_targets = {s["index"]: s["weight"] for s in sources}
                current_targets = SelectionStrategy.renormalize(current_targets, self.config, apply_layer_prune=True)

                # Cleanup layer-specific caches to save memory
                if hasattr(backend, 'cache'):
                    backend.cache.clear_single(layer_idx, "hidden")
                    backend.cache.clear_single(layer_idx, "attention")
                    backend.cache.clear_single(layer_idx, "grad")
                
                # Force garbage collection after each layer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error processing layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # Try to continue with next layer
                continue
        
        # Save results
        results = {}
        if self.records_by_mode[mode]:
            try:
                csv_path = self.io.write_trace_data(
                    trace_id=trace_id,
                    records=self.records_by_mode[mode],
                    metadata={
                        "mode": mode.value, 
                        "tracing_mode": mode.value,
                        "target_tokens": target_tokens,
                        "logit_lens_concepts": self.logit_lens_concepts,
                        "image_available": "original_image" in input_data,
                        "feature_mapping": input_data.get("feature_mapping", {})
                    }
                )
                results = {
                    "trace_data_path": csv_path,
                    "csv_path": csv_path,
                    "num_records": len(self.records_by_mode[mode]),
                    "layer_traces": layer_traces
                }
            except Exception as e:
                logger.error(f"Error writing trace data: {e}")
                results = {
                    "error": f"Failed to write trace data: {e}",
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
            "logit_lens_concepts": self.logit_lens_concepts,
            "image_available": "original_image" in input_data,
            "feature_mapping": input_data.get("feature_mapping", {})
        }
        
        # Final memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
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

    def _process_logit_lens_projections(self, token_idx: int, record: Dict[str, Any], ll_projections: Dict[int, Dict[str, Any]]) -> None:
        """
        Process and add LogitLens projection data to a record with dynamic concept handling.
        
        This function adds LogitLens projection data to a trace record, ensuring compatibility
        with both old and new field naming conventions. It also dynamically identifies concepts
        from configured sources rather than hardcoding them.
        
        Args:
            token_idx: The token index for which to add projections
            record: The record to update with projection data
            ll_projections: Dictionary mapping token indices to projection data
        """
        if ll_projections and token_idx in ll_projections:
            ll_data = ll_projections[token_idx]
            logger.debug(f"Processing LogitLens data for token {token_idx}: '{ll_data.get('top1_token', '')}'")
            
            # Add to record - ensuring we include BOTH old and new field names for compatibility
            record.update({
                # Original field names from old version
                "predicted_top_token": ll_data.get("top1_token", ""),
                "predicted_top_prob": float(ll_data.get("top1_prob", 0.0)),
                
                # New field names in new version
                "ll_top1_token": ll_data.get("top1_token", ""),
                "ll_top1_prob": float(ll_data.get("top1_prob", 0.0)),
                "ll_top1_logit": float(ll_data.get("top1_logit", 0.0)),
                "ll_top2_token": ll_data.get("top2_token", ""),
                "ll_top2_prob": float(ll_data.get("top2_prob", 0.0))
            })
            
            # Extract token texts for concept matching
            top_token = ll_data.get("top1_token", "").lower()
            second_token = ll_data.get("top2_token", "").lower()
            top_prob = float(ll_data.get("top1_prob", 0.0))
            second_prob = float(ll_data.get("top2_prob", 0.0))
            
            # Get tracked concepts from available sources (in order of priority)
            tracked_concepts = []
            
            # 1. Try to get concepts from the logit lens backend
            if hasattr(self, "logit_backend") and self.logit_backend is not None:
                if hasattr(self.logit_backend, "concepts"):
                    tracked_concepts = self.logit_backend.concepts
                    logger.debug(f"Using {len(tracked_concepts)} concepts from logit_backend")
            
            # 2. Fall back to class-level concepts if available
            if not tracked_concepts and hasattr(self, "logit_lens_concepts"):
                tracked_concepts = self.logit_lens_concepts
                logger.debug(f"Using {len(tracked_concepts)} class-level concepts")
            
            # 3. If we have metadata with concepts, use those
            if not tracked_concepts:
                try:
                    metadata_concepts = next((meta.get("logit_lens_concepts", []) 
                                            for meta in getattr(self, "metadata", {}).values()
                                            if isinstance(meta, dict) and "logit_lens_concepts" in meta), [])
                    if metadata_concepts:
                        tracked_concepts = metadata_concepts
                        logger.debug(f"Using {len(tracked_concepts)} concepts from metadata")
                except Exception as e:
                    logger.debug(f"Error getting concepts from metadata: {e}")
            
            # If no concepts defined, use default set for backward compatibility
            if not tracked_concepts:
                default_concepts = ["cat", "dog", "person", "building", "water", "sky", "car", 
                                "eiffel", "tower", "paris", "france", "seine", "river", "landmark", "bridge"]
                tracked_concepts = default_concepts
                logger.debug(f"Using {len(tracked_concepts)} default concepts")
            
            # Process each concept and add to record
            for concept in tracked_concepts:
                concept_lower = concept.lower()
                field_name = f"concept_{concept}_prob"
                
                # Some concepts might have uppercase first letter in field names (for backward compatibility)
                if concept in ["eiffel", "tower", "paris", "france", "seine"]:
                    field_name = f"concept_{concept.capitalize()}_prob"
                
                # Check if this concept appears in either top token
                if concept_lower in top_token:
                    record[field_name] = top_prob
                    logger.debug(f"Found concept '{concept}' in top token")
                elif concept_lower in second_token:
                    record[field_name] = second_prob
                    logger.debug(f"Found concept '{concept}' in second token")
        else:
            # Add empty fields for consistency
            record.update({
                "predicted_top_token": "",
                "predicted_top_prob": 0.0,
                "ll_top1_token": "",
                "ll_top1_prob": 0.0,
                "ll_top1_logit": 0.0,
                "ll_top2_token": "",
                "ll_top2_prob": 0.0
            })
            
            if ll_projections and token_idx not in ll_projections:
                logger.debug(f"Token {token_idx} not found in LogitLens projections")
            elif not ll_projections:
                logger.debug(f"No LogitLens projections available for token {token_idx}")

    def set_logit_lens_concepts(self, concepts: List[str]) -> None:
        """
        Set the concepts to track in LogitLens projections.
        
        This method configures which concepts should be looked for in token projections.
        These concepts will be used to create fields like 'concept_X_prob' in trace records.
        
        Args:
            concepts: List of concept strings to track
        """
        # Store at the class level for fallback
        self.logit_lens_concepts = concepts
        
        # Also set in the LogitLens backend if available
        if hasattr(self, "logit_backend") and self.logit_backend is not None:
            self.logit_backend.concepts = concepts
            logger.info(f"Set {len(concepts)} concepts to track in LogitLens backend")
        else:
            logger.warning("LogitLens backend not available, concepts stored at workflow level only")