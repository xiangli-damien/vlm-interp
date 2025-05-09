# workflows/generation.py
"""
Mixin class for autoregressive token generation.
"""

import torch
import logging
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger("generation_mixin")
logger.setLevel(logging.INFO)

class GenerationMixin:
    """
    Mixin class providing autoregressive generation functionality.
    """
    
    def autoregressive_generate(self, inputs: Dict[str, torch.Tensor], num_tokens: int = 1) -> Dict[str, Any]:
        """
        Generate tokens from a model autoregressively.
        
        Args:
            inputs: Model input tensors
            num_tokens: Number of tokens to generate
            
        Returns:
            Dictionary with generation results
        """
        try:
            # Make a deep copy of inputs to avoid modifying the original
            inputs_copy = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs_copy[k] = v.clone()
                else:
                    inputs_copy[k] = v
            
            # Get model and processor from the instance
            if not hasattr(self, 'model') or not hasattr(self, 'processor'):
                raise AttributeError("GenerationMixin requires 'model' and 'processor' attributes")
            
            model = self.model
            processor = self.processor
            
            # Start with initial input IDs and attention mask
            current_input_ids = inputs_copy["input_ids"].clone()
            original_seq_len = current_input_ids.shape[1]
            
            # Store original attention mask if it exists
            original_attention_mask = inputs_copy.get("attention_mask", None)
            if original_attention_mask is not None:
                original_attention_mask = original_attention_mask.clone()
            
            # Generate tokens one by one
            model.eval()
            generated_tokens = []
            
            logger.info(f"Generating {num_tokens} tokens...")
            
            with torch.no_grad():
                for i in range(num_tokens):
                    # Forward pass
                    outputs = model(
                        **inputs_copy,
                        use_cache=True
                    )
                    
                    # Get next token
                    logits = outputs.logits[:, -1, :]
                    next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
                    token_id = next_token_id.item()
                    token_text = processor.tokenizer.decode([token_id])
                    
                    # Add to generated tokens
                    generated_tokens.append({
                        "index": original_seq_len + i,
                        "id": token_id,
                        "text": token_text,
                        "type": "generated"
                    })
                    
                    # Update input IDs for next iteration
                    current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0)], dim=1)
                    
                    # Update inputs for next iteration
                    inputs_copy["input_ids"] = current_input_ids
                    
                    # Handle attention mask correctly - preserving any image token padding
                    if "attention_mask" in inputs_copy:
                        if original_attention_mask is not None:
                            # Extend the mask by padding with a 1 (attend to the new token)
                            # This preserves any 0s in the original mask (e.g., for image token padding)
                            inputs_copy["attention_mask"] = F.pad(
                                original_attention_mask, 
                                (0, i+1), 
                                value=1
                            )
                        else:
                            inputs_copy["attention_mask"] = torch.ones_like(current_input_ids)
                        
                    logger.debug(f"Generated token {i+1}: '{token_text}' (ID: {token_id})")
                    
            # Create full sequence info
            full_sequence = {
                "ids": current_input_ids[0].tolist(),
                "text": processor.tokenizer.decode(current_input_ids[0].tolist()),
                "prompt_length": original_seq_len
            }
            
            logger.info(f"Generation complete: {len(generated_tokens)} tokens generated")
            
            return {
                "inputs": inputs_copy,
                "full_sequence": full_sequence,
                "generated_tokens": generated_tokens,
                "original_seq_len": original_seq_len
            }
            
        except Exception as e:
            logger.error(f"Error during token generation: {e}")
            raise