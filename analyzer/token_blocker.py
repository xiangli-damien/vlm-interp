import torch
import os
import gc
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from PIL import Image

class TokenBlockingExperiment:
    """
    Conducts experiments by blocking specific tokens in a model's hidden states
    to analyze their impact on model prediction.
    
    Memory-optimized version with CPU offloading and enhanced memory management.
    """
    
    def __init__(
        self,
        model,
        processor,
        output_dir: str = "blocking_experiments",
        debug_mode: bool = False,
        cpu_offloading: bool = True,  # New parameter to enable CPU offloading
        batch_size: int = 4          # New parameter for batched token generation
    ):
        """
        Initialize the experiment framework.
        
        Args:
            model: Model to use for experiments
            processor: Model processor/tokenizer
            output_dir: Directory to save results
            debug_mode: Enable detailed debug logging
            cpu_offloading: Whether to offload tensors to CPU to reduce GPU memory
            batch_size: Number of tokens to generate in each batch
        """
        self.model = model
        self.processor = processor
        self.output_dir = output_dir
        self.debug_mode = debug_mode
        self.cpu_offloading = cpu_offloading
        self.batch_size = batch_size
        os.makedirs(output_dir, exist_ok=True)
        
        # For tracking hook calls
        self.hook_call_tracking = {}
        
        # Pre-calculate layer info
        self.all_layers = self._identify_model_layers()
        self.layer_ids = list(range(len(self.all_layers)))
        
        if debug_mode:
            print(f"Found {len(self.all_layers)} layers in model")
            for i, (name, _) in enumerate(self.all_layers):
                print(f"  Layer {i}: {name}")
    
    def _identify_model_layers(self):
        """
        Identifies transformer layers in the model based on common naming patterns.
        Returns a list of (name, module) tuples for each layer in sequence.
        """
        # Common layer patterns in HF models
        layer_patterns = [
            r'language_model\.model\.layers\.(\d+)',  # Llama, Mistral, Vicuna style
            r'language_model\.transformer\.h\.(\d+)',  # GPT-2 style
            r'language_model\.encoder\.layer\.(\d+)',  # BERT style
            r'layers\.(\d+)',                         # Generic pattern
            r'layer\.(\d+)',                          # Generic pattern
            r'h\.(\d+)'                               # Generic pattern
        ]
        
        import re
        
        # Find all modules matching our patterns
        all_matching_modules = []
        for name, module in self.model.named_modules():
            # Check each pattern
            layer_idx = -1
            for pattern in layer_patterns:
                match = re.search(pattern, name)
                if match:
                    layer_idx = int(match.group(1))
                    all_matching_modules.append((name, module, layer_idx))
                    break
        
        # Sort by layer index
        all_matching_modules.sort(key=lambda x: x[2])
        
        # Group by layer index (to handle cases where multiple modules match per layer)
        layers_by_idx = {}
        for name, module, idx in all_matching_modules:
            if idx not in layers_by_idx:
                layers_by_idx[idx] = []
            layers_by_idx[idx].append((name, module))
        
        # Select the "main" module for each layer (typically the outer block)
        result = []
        for idx in sorted(layers_by_idx.keys()):
            modules = layers_by_idx[idx]
            # Prefer modules with shorter names (usually the outer block)
            modules.sort(key=lambda x: len(x[0]))
            result.append(modules[0])
        
        return result
    
    def run_blocking_experiment(
        self,
        inputs,
        tokens_to_block,
        method: str = "zero_out",
        attention_strategy: str = "none",
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        num_tokens_to_generate: int = 1,
        max_memory_usage: float = 0.95,  # New parameter to limit memory usage (fraction of available)
        save_hidden_states: bool = False  # New parameter to optionally disable saving hidden states
    ):
        """
        Run a token blocking experiment, generating multiple tokens.
        
        Args:
            inputs: Model inputs (output from processor)
            tokens_to_block: List of token indices to block
            method: Blocking method ("zero_out", "average", "noise", "interpolate", "reduce")
            attention_strategy: Attention masking strategy ("none", "block", "reduce")
            start_layer: First layer to apply blocking (None = from beginning)
            end_layer: Last layer to apply blocking (None = to end)
            num_tokens_to_generate: Number of tokens to generate
            max_memory_usage: Maximum fraction of GPU memory to use before moving to CPU
            save_hidden_states: Whether to save hidden states (memory intensive)
            
        Returns:
            Dictionary with experiment results
        """
        # Validate parameters
        if not self.all_layers:
            return {"error": "No layers found in model"}
        
        valid_methods = ["zero_out", "average", "noise", "interpolate", "reduce"]
        if method not in valid_methods:
            return {"error": f"Invalid method: {method}. Must be one of: {valid_methods}"}
        
        valid_attention_strategies = ["none", "block", "reduce"]
        if attention_strategy not in valid_attention_strategies:
            return {"error": f"Invalid attention strategy: {attention_strategy}. Must be one of: {valid_attention_strategies}"}
        
        # Set layer range
        start_layer = 0 if start_layer is None else start_layer
        end_layer = len(self.all_layers) - 1 if end_layer is None else end_layer
        
        if start_layer < 0 or start_layer >= len(self.all_layers):
            return {"error": f"Invalid start_layer: {start_layer}. Must be between 0 and {len(self.all_layers)-1}"}
        
        if end_layer < start_layer or end_layer >= len(self.all_layers):
            return {"error": f"Invalid end_layer: {end_layer}. Must be between {start_layer} and {len(self.all_layers)-1}"}
        
        # Clean up memory before starting experiment
        self._clean_memory()
        
        # Run baseline inference first (no blocking)
        print(f"Running baseline inference without blocking...")
        reference_results = self._run_model_inference(
            inputs, 
            num_tokens=num_tokens_to_generate,
            include_hidden_states=save_hidden_states
        )
        
        # Format baseline results
        reference_tokens = reference_results["generated_tokens"]
        reference_text = reference_results["generated_text"]
        
        print(f"Baseline generated: '{reference_text}'")
        
        # Clean up memory again before running the blocking experiment
        self._clean_memory()
        
        # Run blocking experiment
        blocking_range = f"layers {start_layer} to {end_layer}"
        tokens_str = ", ".join(str(t) for t in tokens_to_block)
        print(f"Running blocking experiment: tokens [{tokens_str}], method={method}, attention={attention_strategy}, {blocking_range}")
        
        # Set up hooks for blocking
        hooks = self._register_blocking_hooks(
            tokens_to_block,
            method,
            attention_strategy,
            start_layer,
            end_layer
        )
        
        # Run model with blocking
        try:
            with torch.no_grad():
                blocked_results = self._run_model_inference(
                    inputs, 
                    num_tokens=num_tokens_to_generate,
                    include_hidden_states=save_hidden_states,
                    max_memory_usage=max_memory_usage
                )
            
            # Format results
            blocked_tokens = blocked_results["generated_tokens"]
            blocked_text = blocked_results["generated_text"]
            
            print(f"Blocked generated: '{blocked_text}'")
            
            # Compare results
            token_comparison = self._compare_token_sequences(reference_tokens, blocked_tokens)
            
            # Format experiment results
            results = {
                "config": {
                    "tokens_blocked": tokens_to_block,
                    "method": method,
                    "attention_strategy": attention_strategy,
                    "start_layer": start_layer,
                    "end_layer": end_layer,
                    "num_layers": len(self.all_layers),
                    "num_tokens_generated": num_tokens_to_generate
                },
                "reference": {
                    "tokens": reference_tokens,
                    "text": reference_text
                },
                "blocked": {
                    "tokens": blocked_tokens,
                    "text": blocked_text
                },
                "comparison": token_comparison,
                "hook_calls": dict(self.hook_call_tracking) if self.debug_mode else None
            }
            
            # Add diagnostic info in debug mode
            if self.debug_mode:
                results["diagnostics"] = {
                    "layers_used": [(i, name) for i, (name, _) in enumerate(self.all_layers)
                                     if i >= start_layer and i <= end_layer],
                }
            
            return results
            
        finally:
            # Always clean up hooks
            for hook in hooks:
                hook.remove()
            
            # Reset call tracking
            self.hook_call_tracking = {}
            
            # Clean up memory after experiment
            self._clean_memory()
    
    def _clean_memory(self):
        """
        Clean up memory by releasing CUDA cache and running garbage collector.
        """
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if self.debug_mode:
                # Get current memory stats for debugging
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                max_reserved = torch.cuda.max_memory_reserved() / (1024 ** 3)
                print(f"  CUDA Memory - Allocated: {allocated:.2f}GB (Max: {max_allocated:.2f}GB), "
                      f"Reserved: {reserved:.2f}GB (Max: {max_reserved:.2f}GB)")
    
    def _check_memory_usage(self, max_usage_fraction=0.95):
        """
        Check if GPU memory usage exceeds the threshold.
        Returns True if memory should be offloaded to CPU.
        """
        if not torch.cuda.is_available() or not self.cpu_offloading:
            return False
        
        # Get current memory usage
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        usage_fraction = allocated / total
        
        if self.debug_mode and usage_fraction > 0.7:  # Alert if over 70%
            print(f"  Memory usage: {usage_fraction:.2%} of GPU memory")
            
        return usage_fraction > max_usage_fraction
    
    def _register_blocking_hooks(
        self,
        tokens_to_block,
        method,
        attention_strategy,
        start_layer,
        end_layer
    ):
        """
        Register hooks for blocking tokens in specified layers.
        Returns list of hook handles for later removal.
        """
        hook_handles = []
        
        # 1. Register hooks for hidden state blocking
        for layer_idx, (layer_name, layer_module) in enumerate(self.all_layers):
            # Only register hooks for layers in the specified range
            if layer_idx < start_layer or layer_idx > end_layer:
                continue
                
            # Create forward hook with layer info captured in closure
            hook_fn = self._create_hidden_state_hook(
                layer_idx, 
                layer_name,
                tokens_to_block, 
                method
            )
            
            # Register the hook
            hook_handles.append(layer_module.register_forward_hook(hook_fn))
            
            if self.debug_mode:
                print(f"  Registered hidden state hook for layer {layer_idx}: {layer_name}")
        
        # 2. Register attention hooks if needed
        if attention_strategy != "none":
            attention_hooks = self._register_attention_hooks(
                tokens_to_block,
                attention_strategy,
                start_layer,
                end_layer
            )
            hook_handles.extend(attention_hooks)
        
        return hook_handles
    
    def _create_hidden_state_hook(self, layer_idx, layer_name, tokens_to_block, method):
        """Create a hook function for blocking tokens in hidden states."""
        def hook_fn(module, inputs, output):
            # Track that this hook was called
            key = f"hidden_state_layer_{layer_idx}"
            self.hook_call_tracking[key] = self.hook_call_tracking.get(key, 0) + 1
            
            # Get hidden states from output (handle different output formats)
            if isinstance(output, tuple):
                hidden_states = output[0]
                other_outputs = output[1:]
            else:
                hidden_states = output
                other_outputs = ()
            
            # Clone to avoid modifying the original
            modified = hidden_states.clone()
            
            # Apply blocking based on method
            for token_idx in tokens_to_block:
                if token_idx < modified.shape[1]:  # Ensure token is in sequence
                    if method == "zero_out":
                        # Zero out the token representation
                        modified[:, token_idx, :] = 0
                    elif method == "average":
                        # Replace with average of all tokens
                        avg_repr = torch.mean(hidden_states, dim=1, keepdim=True)
                        modified[:, token_idx, :] = avg_repr
                    elif method == "noise":
                        # Replace with Gaussian noise
                        noise = torch.randn_like(modified[:, token_idx, :])
                        modified[:, token_idx, :] = noise
                    elif method == "interpolate":
                        # Replace with interpolation of neighboring tokens
                        if token_idx > 0 and token_idx < modified.shape[1] - 1:
                            prev_token = hidden_states[:, token_idx-1, :]
                            next_token = hidden_states[:, token_idx+1, :]
                            modified[:, token_idx, :] = (prev_token + next_token) / 2
                    elif method == "reduce":
                        # Reduce token representation to 10% of original value
                        modified[:, token_idx, :] *= 0.1
            
            # Return the modified hidden states
            if isinstance(output, tuple):
                return (modified,) + other_outputs
            else:
                return modified
        
        return hook_fn
    
    def _register_attention_hooks(self, tokens_to_block, attention_strategy, start_layer, end_layer):
        """Register hooks for modifying attention mechanisms."""
        hooks = []
        
        # Find attention modules within transformer blocks
        attention_modules = []
        for layer_idx, (layer_name, layer_module) in enumerate(self.all_layers):
            # Only look at layers in the specified range
            if layer_idx < start_layer or layer_idx > end_layer:
                continue
            
            # Find attention modules within this layer
            for name, module in layer_module.named_modules():
                # Look for attention-related modules
                if any(attn_name in name.lower() for attn_name in ['attention', 'attn']):
                    attention_modules.append((f"{layer_idx}_{name}", module, layer_idx))
                    if self.debug_mode:
                        print(f"  Found attention module in layer {layer_idx}: {name}")
        
        for module_name, module, layer_idx in attention_modules:
            # 1. Pre-hook to modify inputs to attention (query, key, value)
            pre_hook_fn = self._create_attention_input_hook(
                layer_idx, 
                module_name,
                tokens_to_block, 
                attention_strategy
            )
            hooks.append(module.register_forward_pre_hook(pre_hook_fn))
            
            if self.debug_mode:
                print(f"  Registered attention pre-hook for module: {module_name}")
            
            # 2. Post-hook to modify attention weights
            post_hook_fn = self._create_attention_output_hook(
                layer_idx, 
                module_name,
                tokens_to_block, 
                attention_strategy
            )
            hooks.append(module.register_forward_hook(post_hook_fn))
            
            if self.debug_mode:
                print(f"  Registered attention post-hook for module: {module_name}")
        
        return hooks
    
    def _create_attention_input_hook(self, layer_idx, module_name, tokens_to_block, strategy):
        """Create a pre-hook function for modifying attention inputs."""
        def hook_fn(module, inputs):
            # Track that this hook was called
            key = f"attn_input_layer_{layer_idx}_{module_name}"
            self.hook_call_tracking[key] = self.hook_call_tracking.get(key, 0) + 1
            
            # Process different input formats
            if not inputs or not isinstance(inputs, tuple):
                return inputs
                
            # Attempt to modify query/key/value tensors
            # This works for many but not all attention implementations
            if len(inputs) >= 1 and isinstance(inputs[0], torch.Tensor):
                # inputs[0] is often the query input
                modified_inputs = list(inputs)
                
                # Create a clone to modify
                query_input = inputs[0].clone()
                
                # Apply blocking directly to query input
                for token_idx in tokens_to_block:
                    if token_idx < query_input.shape[1]:  # Ensure token is in sequence
                        if strategy == "block":
                            # Zero out the token
                            query_input[:, token_idx, :] = 0
                        elif strategy == "reduce":
                            # Reduce the token's contribution
                            query_input[:, token_idx, :] *= 0.1
                
                # Replace the query input
                modified_inputs[0] = query_input
                
                return tuple(modified_inputs)
            
            # Look for attention mask in inputs (for some implementations)
            mask_idx = None
            for i, tensor in enumerate(inputs):
                if isinstance(tensor, torch.Tensor):
                    if tensor.dtype == torch.bool:
                        mask_idx = i
                        break
                    elif len(tensor.shape) == 4 and tensor.shape[2] == tensor.shape[3]:
                        # Might be an attention mask
                        mask_idx = i
                        break
            
            if mask_idx is not None:
                modified_inputs = list(inputs)
                mask = inputs[mask_idx].clone()
                
                # Modify the mask based on strategy
                for token_idx in tokens_to_block:
                    if token_idx < mask.shape[1] if len(mask.shape) == 2 else token_idx < mask.shape[2]:
                        if strategy == "block":
                            # Set mask to zero (don't attend to/from this token)
                            if len(mask.shape) == 2:
                                # For boolean masks
                                mask[:, token_idx] = False
                            elif len(mask.shape) == 4:
                                # For additive attention masks
                                mask[:, :, token_idx, :] = float('-inf')
                                mask[:, :, :, token_idx] = float('-inf')
                        elif strategy == "reduce":
                            # For float masks, reduce attention weight
                            if len(mask.shape) == 4 and mask.dtype != torch.bool:
                                mask[:, :, token_idx, :] -= 2.0  # Substantially reduce but not eliminate
                                mask[:, :, :, token_idx] -= 2.0
                
                modified_inputs[mask_idx] = mask
                return tuple(modified_inputs)
            
            return inputs
        
        return hook_fn
    
    def _create_attention_output_hook(self, layer_idx, module_name, tokens_to_block, strategy):
        """Create a post-hook function for modifying attention outputs/weights."""
        def hook_fn(module, inputs, outputs):
            # Track that this hook was called
            key = f"attn_output_layer_{layer_idx}_{module_name}"
            self.hook_call_tracking[key] = self.hook_call_tracking.get(key, 0) + 1
            
            # Check if outputs contains attention weights
            if not isinstance(outputs, tuple) or len(outputs) < 2:
                return outputs
            
            # outputs[0] is typically the attention output, outputs[1] might be attention weights
            attn_output = outputs[0]
            
            # If second element looks like attention weights (batch, heads, seq, seq)
            if (isinstance(outputs[1], torch.Tensor) and 
                outputs[1].dim() == 4 and 
                outputs[1].shape[2] == outputs[1].shape[3]):
                
                attn_weights = outputs[1].clone()
                
                # Modify attention weights based on strategy
                for token_idx in tokens_to_block:
                    if token_idx < attn_weights.shape[2]:  # Check if token idx is valid
                        if strategy == "block":
                            # Block this token from receiving/providing attention
                            attn_weights[:, :, token_idx, :] = 0
                            attn_weights[:, :, :, token_idx] = 0
                        elif strategy == "reduce":
                            # Reduce attention to/from this token
                            attn_weights[:, :, token_idx, :] *= 0.1
                            attn_weights[:, :, :, token_idx] *= 0.1
                
                # Return modified outputs
                modified_outputs = list(outputs)
                modified_outputs[1] = attn_weights
                return tuple(modified_outputs)
            
            # If we can't identify attention weights, try to modify the output directly
            if isinstance(attn_output, torch.Tensor):
                modified_output = attn_output.clone()
                
                # Apply blocking directly to attention output
                for token_idx in tokens_to_block:
                    if token_idx < modified_output.shape[1]:  # Ensure token is in sequence
                        if strategy == "block":
                            # Zero out the token output
                            modified_output[:, token_idx, :] = 0
                        elif strategy == "reduce":
                            # Reduce the token's contribution
                            modified_output[:, token_idx, :] *= 0.1
                
                # Return modified outputs
                if isinstance(outputs, tuple):
                    modified_outputs = list(outputs)
                    modified_outputs[0] = modified_output
                    return tuple(modified_outputs)
                else:
                    return modified_output
            
            return outputs
        
        return hook_fn
    
    def _run_model_inference(self, inputs, num_tokens=1, include_hidden_states=False, max_memory_usage=0.95):
        """
        Run model inference, potentially generating multiple tokens.
        
        Memory-optimized version with CPU offloading and batched generation.
        
        Args:
            inputs: Model inputs dictionary
            num_tokens: Number of tokens to generate
            include_hidden_states: Whether to include hidden states in output (memory intensive)
            max_memory_usage: Maximum fraction of GPU memory to use before offloading
            
        Returns:
            Dictionary with generation results
        """
        model_kwargs = {
            "output_hidden_states": include_hidden_states,
            "output_attentions": include_hidden_states
        }
        
        # Make sure model is in eval mode
        self.model.eval()
        
        # For generation with multiple tokens
        if num_tokens > 1:
            # Clone inputs to avoid modifying original
            current_inputs = {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()}
            
            # Store original sequence length
            original_length = current_inputs["input_ids"].shape[1]
            
            # Track generated tokens
            all_tokens = []
            
            # Determine batch size for generation (can be adjusted based on memory)
            batch_size = min(self.batch_size, num_tokens)
            num_batches = (num_tokens + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, num_tokens)
                batch_size_actual = batch_end - batch_start
                
                if self.debug_mode:
                    print(f"  Processing token batch {batch_idx+1}/{num_batches} (tokens {batch_start+1}-{batch_end})")
                
                # Process tokens in this batch
                for i in range(batch_size_actual):
                    token_idx = batch_start + i
                    
                    # Check if we need to offload to CPU to save memory
                    if self._check_memory_usage(max_memory_usage):
                        if self.debug_mode:
                            print(f"  Memory usage high - offloading tensors to CPU")
                        # Move tensors to CPU
                        for k, v in current_inputs.items():
                            if torch.is_tensor(v) and v.device.type == 'cuda':
                                current_inputs[k] = v.cpu()
                        # Clean memory
                        self._clean_memory()
                        # Move back to GPU for inference
                        for k, v in current_inputs.items():
                            if torch.is_tensor(v) and v.device.type == 'cpu':
                                current_inputs[k] = v.to(self.model.device)
                    
                    # Run model with no gradients
                    with torch.no_grad():
                        # Handle potential OOM by reducing batch size or retrying
                        try:
                            # Remove 'token_type_ids' if present (not needed by some models)
                            model_inputs = {k: v for k, v in current_inputs.items() if k != "token_type_ids"}
                            outputs = self.model(**model_inputs, **model_kwargs)
                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                print(f"  CUDA OOM detected - attempting recovery")
                                # Clean up memory
                                self._clean_memory()
                                # Retry with reduced features
                                reduced_kwargs = {k: v for k, v in model_kwargs.items()}
                                reduced_kwargs["output_hidden_states"] = False
                                reduced_kwargs["output_attentions"] = False
                                outputs = self.model(**model_inputs, **reduced_kwargs)
                            else:
                                raise
                    
                    # Get prediction for next token
                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Extract token text
                    token_text = self.processor.tokenizer.decode([next_token_id.item()])
                    
                    # Check for EOS token to potentially stop early
                    is_eos = next_token_id.item() == self.processor.tokenizer.eos_token_id
                    
                    # Store minimal token info to save memory
                    token_info = {
                        "id": next_token_id.item(),
                        "text": token_text,
                        "position": original_length + token_idx,
                    }
                    
                    # Only store logits in debug mode (save memory)
                    if self.debug_mode:
                        # Store only top-k logits to save memory
                        top_k = 5
                        top_vals, top_idx = torch.topk(logits, top_k)
                        token_info["top_logits"] = {
                            "values": top_vals.detach().cpu().tolist()[0],
                            "indices": top_idx.detach().cpu().tolist()[0]
                        }
                    
                    all_tokens.append(token_info)
                    
                    # Update inputs for next iteration
                    current_inputs["input_ids"] = torch.cat([current_inputs["input_ids"], next_token_id], dim=1)
                    if "attention_mask" in current_inputs:
                        current_inputs["attention_mask"] = torch.cat([
                            current_inputs["attention_mask"], 
                            torch.ones_like(next_token_id)
                        ], dim=1)
                    
                    if self.debug_mode:
                        print(f"  Generated token {token_idx+1}/{num_tokens}: '{token_text}'")
                    
                    # If generated EOS token, stop generation
                    if is_eos and token_idx < num_tokens - 1:
                        if self.debug_mode:
                            print(f"  Stopping early at token {token_idx+1} - EOS token generated")
                        break
                    
                    # Clean up for next token (if not last in batch)
                    if i < batch_size_actual - 1:
                        del outputs, logits, next_token_id
                        gc.collect()
                
                # Clean up between batches
                self._clean_memory()
            
            # Get full text
            all_ids = current_inputs["input_ids"][0, original_length:].tolist()
            generated_text = self.processor.tokenizer.decode(all_ids)
            
            # Prepare return data (minimizing memory usage)
            result = {
                "generated_tokens": all_tokens,
                "generated_text": generated_text,
                "token_ids": all_ids,
            }
            
            # Clean up before returning
            del current_inputs
            gc.collect()
            
            return result
        
        # For single token generation (simplified path)
        else:
            # Run model with no gradients
            with torch.no_grad():
                model_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
                outputs = self.model(**model_inputs, **model_kwargs)
            
            # Get prediction for next token
            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1).item()
            token_text = self.processor.tokenizer.decode([next_token_id])
            
            result = {
                "generated_tokens": [{
                    "id": next_token_id,
                    "text": token_text,
                    "position": inputs["input_ids"].shape[1],
                }],
                "generated_text": token_text,
                "token_ids": [next_token_id],
            }
            
            # Clean up
            del outputs, logits
            gc.collect()
            
            return result
    
    def _compare_token_sequences(self, reference_tokens, blocked_tokens):
        """Compare token sequences between reference and blocked runs."""
        comparison = {
            "exact_match": len(reference_tokens) == len(blocked_tokens) and all(
                r["id"] == b["id"] for r, b in zip(reference_tokens, blocked_tokens)
            ),
            "tokens": []
        }
        
        # Track metrics
        num_matching = 0
        
        # Compare each position
        for i in range(min(len(reference_tokens), len(blocked_tokens))):
            ref_token = reference_tokens[i]
            blocked_token = blocked_tokens[i]
            
            # Check if tokens match
            is_match = ref_token["id"] == blocked_token["id"]
            if is_match:
                num_matching += 1
            
            comparison["tokens"].append({
                "position": i,
                "reference": {
                    "id": ref_token["id"],
                    "text": ref_token["text"]
                },
                "blocked": {
                    "id": blocked_token["id"],
                    "text": blocked_token["text"]
                },
                "match": is_match
            })
        
        # Add summary metrics
        comparison["summary"] = {
            "total_tokens": min(len(reference_tokens), len(blocked_tokens)),
            "num_matching": num_matching,
            "match_rate": num_matching / min(len(reference_tokens), len(blocked_tokens)) if min(len(reference_tokens), len(blocked_tokens)) > 0 else 0,
            "length_match": len(reference_tokens) == len(blocked_tokens)
        }
        
        # Calculate impact score: 0 = no impact, 1 = complete change
        # Based on token difference rate
        comparison["impact_score"] = 1.0 - comparison["summary"]["match_rate"]
        
        return comparison
    
    def visualize_results(self, results, save_path=None):
        """Create a visualization of blocking experiment results."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create the figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Blocking configuration visualization
        ax1 = axes[0]
        config = results["config"]
        num_layers = config["num_layers"]
        start_layer = config["start_layer"]
        end_layer = config["end_layer"]
        
        # Create a bar for the blocking range
        blocking_range = np.zeros(num_layers)
        blocking_range[start_layer:end_layer+1] = 1
        
        ax1.bar(range(num_layers), blocking_range, color='red', alpha=0.6)
        ax1.set_xlabel('Model Layers')
        ax1.set_ylabel('Blocking Applied')
        ax1.set_title(f'Blocking Range: Layers {start_layer} to {end_layer}')
        ax1.set_xticks(range(0, num_layers, max(1, num_layers // 10)))
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text annotation
        tokens_str = ", ".join(str(t) for t in config["tokens_blocked"])
        ax1.text(0.5, -0.15, 
                f"Method: {config['method']}, Attn Strategy: {config['attention_strategy']}\nTokens Blocked: [{tokens_str}]",
                ha='center', va='center', transform=ax1.transAxes)
        
        # 2. Token match visualization
        ax2 = axes[1]
        
        if "comparison" in results and "tokens" in results["comparison"]:
            tokens = results["comparison"]["tokens"]
            positions = [t["position"] for t in tokens]
            matches = [1 if t["match"] else 0 for t in tokens]
            
            # Create a bar chart of matches
            bars = ax2.bar(positions, matches, color='green', alpha=0.7)
            
            # Color non-matching tokens red
            for i, match in enumerate(matches):
                if match == 0:
                    bars[i].set_color('red')
            
            # Add token text labels
            for i, token in enumerate(tokens):
                label_y = 1.05 if matches[i] else -0.15
                label = f"{token['reference']['text']}" if matches[i] else f"{token['reference']['text']}→{token['blocked']['text']}"
                ax2.text(positions[i], label_y, label, ha='center', rotation=45 if not matches[i] else 0)
            
            ax2.set_ylim(-0.3, 1.3)
            ax2.set_xlabel('Token Position')
            ax2.set_ylabel('Matches Reference')
            ax2.set_title(f"Token Matches: {results['comparison']['summary']['num_matching']}/{results['comparison']['summary']['total_tokens']} ({results['comparison']['summary']['match_rate']*100:.1f}%)")
            ax2.set_xticks(positions)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add result annotation
            impact = results["comparison"]["impact_score"]
            impact_color = 'green' if impact < 0.3 else ('orange' if impact < 0.7 else 'red')
            ax2.text(0.5, -0.3, 
                    f"Impact Score: {impact:.2f} ({'Low' if impact < 0.3 else ('Medium' if impact < 0.7 else 'High')})",
                    ha='center', va='center', transform=ax2.transAxes,
                    color=impact_color, fontweight='bold')
        
        # Set overall title
        plt.suptitle(f"Token Blocking Experiment Results\nReference: '{results['reference']['text']}' → Blocked: '{results['blocked']['text']}'", y=1.05)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Visualization saved to: {save_path}")
            return save_path
        else:
            plt.show()
            return None