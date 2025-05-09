# workflows/saliency_analysis.py
"""
Saliency analysis workflow for VLM interpretability.
"""

import torch
import os
import gc
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image

from runtime.io import TraceIO
from preprocess.input_builder import prepare_inputs
from runtime.generation import GenerationMixin
from runtime.selection import SelectionConfig
from runtime.cache import TracingCache
from backends.saliency_backend import SaliencyBackend
from analysis.saliency_viz import visualize_information_flow

# Configure logging
logger = logging.getLogger("saliency.workflow")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class SaliencyWorkflow(GenerationMixin):
    """
    Workflow for saliency-based VLM interpretability analysis.
    """
    
    def __init__(self, model: torch.nn.Module, processor: Any, output_dir: str, 
                 selection_config: Optional[SelectionConfig] = None,
                 debug: bool = False):
        """
        Initialize the saliency workflow.
        
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
        
        # Set debug level if requested
        if debug:
            logger.setLevel(logging.DEBUG)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize selection config
        self.config = selection_config or SelectionConfig()
        
        # Get model device
        self.device = next(model.parameters()).device
        
        # Initialize I/O handler
        self.io = TraceIO(output_dir)
        
        # Initialize cache with CPU offload to save GPU memory
        self.cache = TracingCache(cpu_offload=True)
        
        # Get attention layer names
        try:
            from runtime.model_utils import get_llm_attention_layer_names
            self.layer_names = get_llm_attention_layer_names(model)
        except ImportError:
            logger.warning("Could not import get_llm_attention_layer_names. Using empty layer list.")
            self.layer_names = []
            
        if not self.layer_names:
            logger.warning("No attention layers found. Saliency analysis may not work properly.")
        else:
            logger.info(f"Found {len(self.layer_names)} attention layers.")
        
        # Initialize backend
        self.backend = SaliencyBackend(model, self.layer_names, self.cache, self.device)
        
    def prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """
        Prepare inputs for the model.
        
        Args:
            image: Input image
            prompt: Text prompt
            
        Returns:
            Dictionary of prepared inputs and metadata
        """
        return prepare_inputs(model=self.model, processor=self.processor, image=image, prompt=prompt)
    
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
        token_types = dict(input_data["token_types"])
        for token in gen_result["generated_tokens"]:
            token_types[token["index"]] = "generated"
            
        # Create a result that includes original input_data elements plus generation
        result = {
            "inputs": gen_result["inputs"],
            "text_indices": input_data["text_indices"],
            "image_indices": input_data["image_indices"],
            "token_types": token_types,
            "formatted_prompt": input_data["formatted_prompt"],
            "original_image": input_data["original_image"],
            "token_lengths": input_data["token_lengths"],
            "full_sequence": gen_result["full_sequence"],
            "generated_tokens": gen_result["generated_tokens"],
            "original_seq_len": gen_result["original_seq_len"]
        }
        
        return result
    
    def analyze_flow(self, input_data: Dict[str, Any], target_idx: int, 
                     top_k_image_tokens: Optional[int] = 10) -> Dict[str, Any]:
        """
        Analyze saliency flow for a specific target token.
        
        Args:
            input_data: Prepared input data
            target_idx: Index of the target token
            top_k_image_tokens: Number of top image tokens to consider
            
        Returns:
            Dictionary of flow metrics for each layer
        """
        try:
            # Extract inputs
            inputs = input_data["inputs"]
            text_indices = input_data["text_indices"]
            image_indices = input_data["image_indices"]
            
            # Ensure inputs are on the correct device
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.device != self.device:
                    inputs[k] = v.to(self.device)
            
            # Ensure we have gradients enabled
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
            
            logger.info(f"Analyzing saliency flow for target token at index {target_idx}")
            
            # Make sure we have all target indices
            target_indices = [target_idx]
            
            # Compute batch saliency
            self.backend.ensure_cache(inputs, target_indices)
            
            # Compute flow metrics for each layer
            flow_metrics = {}
            
            for layer_idx in range(len(self.layer_names)):
                if not self.cache.has(layer_idx, "saliency"):
                    logger.warning(f"No saliency data for layer {layer_idx}")
                    continue
                
                # Get saliency matrix
                saliency_tensor = self.cache.get(layer_idx, "saliency", self.device)
                
                # Prepare saliency matrix (average over batch and heads)
                saliency_matrix_2d = self.backend.prepare_matrix(saliency_tensor)
                
                # Compute masks
                text_mask, image_mask, generated_mask, causal_mask = self.backend.mask_sources(
                    saliency_matrix_2d, text_indices, image_indices, target_idx
                )
                
                # Compute flow metrics
                metrics = self.backend.aggregate_flow(
                    saliency_matrix_2d, target_idx, text_mask, image_mask, 
                    generated_mask, causal_mask, top_k_image_tokens
                )
                
                # Normalize flow metrics
                normalized_metrics = self.backend.normalize_flow(metrics)
                
                # Store metrics for this layer
                flow_metrics[layer_idx] = normalized_metrics
                
                # Move tensors to CPU to save memory
                if saliency_matrix_2d.device != torch.device("cpu"):
                    del saliency_matrix_2d
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Add metadata
            result = {
                "flow_metrics": flow_metrics,
                "target_idx": target_idx,
                "target_token": {
                    "id": inputs["input_ids"][0, target_idx].item() if target_idx < inputs["input_ids"].shape[1] else None,
                    "text": self.processor.tokenizer.decode([inputs["input_ids"][0, target_idx].item()]) if target_idx < inputs["input_ids"].shape[1] else None,
                    "type": input_data["token_types"].get(target_idx, "unknown")
                },
                "num_layers": len(self.layer_names)
            }
            
            logger.info(f"Flow analysis complete for target token at index {target_idx}")

            viz_path = os.path.join(
                self.output_dir,
                f"saliency_flow_target{target_idx}.png"
            )
            visualize_information_flow(
                metrics=flow_metrics,
                title=f"Saliency Flow for Token {target_idx}",
                save_path=viz_path,
                use_top_k=(top_k_image_tokens is not None)
            )
            logger.info(f"Visualization saved: {viz_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during flow analysis: {e}")
            raise
        finally:
            # Reset model to eval mode
            self.model.eval()
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def analyze_layerwise_flow(self, input_data: Dict[str, Any], target_indices: List[int],
                              top_k_image_tokens: Optional[int] = 10) -> Dict[str, Any]:
        """
        Analyze saliency flow for multiple target tokens.
        
        Args:
            input_data: Prepared input data
            target_indices: List of target token indices
            top_k_image_tokens: Number of top image tokens to consider
            
        Returns:
            Dictionary of flow metrics for each target and layer
        """
        try:
            # Analyze flow for each target token
            target_results = {}
            
            for target_idx in target_indices:
                logger.info(f"Analyzing target token {target_idx}")
                
                # Skip invalid indices
                if target_idx >= input_data["inputs"]["input_ids"].shape[1]:
                    logger.warning(f"Target index {target_idx} out of bounds")
                    continue
                    
                # Analyze flow for this target
                result = self.analyze_flow(
                    input_data=input_data,
                    target_idx=target_idx,
                    top_k_image_tokens=top_k_image_tokens
                )
                
                # Store result
                target_results[target_idx] = result
                
                # Clean up memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Aggregate results
            result = {
                "targets": target_results,
                "num_targets": len(target_results)
            }
            
            logger.info(f"Layerwise flow analysis complete for {len(target_results)} targets")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during layerwise flow analysis: {e}")
            raise

    def analyze_specific_token(self, input_data: Dict[str, Any], token_idx: int,
                              top_k_image_tokens: Optional[int] = 10) -> Dict[str, Any]:
        """
        Analyze a specific token by index.
        
        Args:
            input_data: Prepared input data
            token_idx: Index of the token to analyze
            top_k_image_tokens: Number of top image tokens to consider
            
        Returns:
            Dictionary of analysis results
        """
        try:
            # Validate token index
            if token_idx >= input_data["inputs"]["input_ids"].shape[1]:
                error_msg = f"Token index {token_idx} out of bounds for sequence length {input_data['inputs']['input_ids'].shape[1]}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Create token info
            token_id = input_data["inputs"]["input_ids"][0, token_idx].item()
            token_text = self.processor.tokenizer.decode([token_id])
            token_type = input_data["token_types"].get(token_idx, "unknown")
            
            logger.info(f"Analyzing specific token index {token_idx} ('{token_text}', type: {token_type})")
            
            # Analyze flow for this token
            flow_result = self.analyze_flow(
                input_data=input_data,
                target_idx=token_idx,
                top_k_image_tokens=top_k_image_tokens
            )
            
            # Create result
            result = {
                "flow_result": flow_result,
                "token_info": {
                    "index": token_idx,
                    "id": token_id,
                    "text": token_text,
                    "type": token_type
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing specific token: {e}")
            raise

    def analyze_generated_tokens(self, input_data: Dict[str, Any], num_tokens: int = 1,
                                top_k_image_tokens: Optional[int] = 10) -> Dict[str, Any]:
        """
        Generate tokens and analyze their saliency.
        
        Args:
            input_data: Prepared input data
            num_tokens: Number of tokens to generate
            top_k_image_tokens: Number of top image tokens to consider
            
        Returns:
            Dictionary of analysis results
        """
        try:
            # Generate tokens
            logger.info(f"Generating and analyzing {num_tokens} tokens")
            gen_result = self.generate(input_data, num_tokens)
            
            # Get generated token indices
            generated_indices = [token["index"] for token in gen_result["generated_tokens"]]
            
            if not generated_indices:
                logger.warning("No tokens were generated")
                return {"error": "No tokens were generated"}
                
            # Analyze each generated token
            analysis_results = {}
            
            for token_idx in generated_indices:
                logger.info(f"Analyzing generated token at index {token_idx}")
                
                # Analyze token
                result = self.analyze_specific_token(
                    input_data=gen_result,
                    token_idx=token_idx,
                    top_k_image_tokens=top_k_image_tokens
                )
                
                # Store result
                analysis_results[token_idx] = result
            
            # Create final result
            result = {
                "generation_result": gen_result,
                "analysis_results": analysis_results,
                "generated_indices": generated_indices
            }
            
            logger.info(f"Generation and analysis complete for {len(generated_indices)} tokens")
            
            return result
            
        except Exception as e:
            logger.error(f"Error during generation and analysis: {e}")
            raise

    def export_flow_data(self, flow_result: Dict[str, Any], trace_id: str = None) -> str:
        """
        Export flow metrics to CSV.
        
        Args:
            flow_result: Flow analysis results
            trace_id: Optional trace identifier
            
        Returns:
            Path to the created CSV file
        """
        try:
            if trace_id is None:
                import time
                trace_id = f"flow_{int(time.time())}"
                
            # Extract flow metrics
            flow_metrics = flow_result.get("flow_metrics", {})
            target_idx = flow_result.get("target_idx")
            target_token = flow_result.get("target_token", {})
            
            if not flow_metrics:
                logger.warning("No flow metrics to export")
                return ""
                
            # Create records for CSV
            records = []
            
            for layer_idx, metrics in flow_metrics.items():
                record = {
                    "layer_idx": layer_idx,
                    "target_idx": target_idx,
                    "target_token_id": target_token.get("id"),
                    "target_token_text": target_token.get("text"),
                    "target_token_type": target_token.get("type")
                }
                
                # Add flow metrics
                for k, v in metrics.items():
                    if isinstance(v, (int, float, str, bool)):
                        record[k] = v
                    elif isinstance(v, dict):
                        for sk, sv in v.items():
                            record[f"{k}_{sk}"] = sv
                
                records.append(record)
                
            # Write to CSV
            csv_path = self.io.write_trace_data(
                trace_id=trace_id,
                records=records,
                metadata={
                    "target_idx": target_idx,
                    "target_token": target_token,
                    "num_layers": flow_result.get("num_layers")
                }
            )
            
            logger.info(f"Exported flow data to {csv_path}")
            
            return csv_path
            
        except Exception as e:
            logger.error(f"Error exporting flow data: {e}")
            raise