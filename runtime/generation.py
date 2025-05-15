# runtime/generation.py
"""
Utilities for autoregressive token generation.
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class GenerationMixin:
    """
    Mixin class providing autoregressive generation functionality.
    
    Classes that inherit from this mixin should have the following attributes:
    - model: The model to generate from
    - processor: The model's processor
    """
    
    def autoregressive_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        num_tokens: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0
    ) -> Dict[str, Any]:
        """
        Generate tokens autoregressively from the model.
        
        Args:
            inputs: Model inputs
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 for greedy decoding)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            
        Returns:
            Dictionary with generation results
        """
        if not hasattr(self, 'model') or not hasattr(self, 'processor'):
            raise AttributeError("GenerationMixin requires 'model' and 'processor' attributes")
        
        # Make copies of inputs to avoid modifying the originals
        inputs_copy = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs_copy[k] = v.clone()
            else:
                inputs_copy[k] = v
        
        # Extract current input IDs
        current_input_ids = inputs_copy["input_ids"].clone()
        original_seq_len = current_input_ids.shape[1]
        
        # Store original attention mask
        original_attention_mask = None
        if "attention_mask" in inputs_copy:
            original_attention_mask = inputs_copy["attention_mask"].clone()
        
        # Put model in eval mode for generation
        self.model.eval()
        
        # Track generated tokens
        generated_tokens = []
        
        logger.info(f"Generating {num_tokens} tokens...")
        
        with torch.no_grad():
            for i in range(num_tokens):
                # Forward pass
                outputs = self.model(
                    **inputs_copy,
                    use_cache=True
                )
                
                # Get logits for the last token
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply sampling
                next_token_id = self._sample_token(logits, top_p, top_k)
                next_token_id_scalar = next_token_id.item()
                
                # Get the token text
                token_text = self.processor.tokenizer.decode([next_token_id_scalar])
                
                # Add to generated tokens list
                generated_tokens.append({
                    "index": original_seq_len + i,
                    "id": next_token_id_scalar,
                    "text": token_text
                })
                
                # Update input IDs for next iteration
                current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                inputs_copy["input_ids"] = current_input_ids
                
                # Update attention mask to ensure newly generated token is attended to
                if "attention_mask" in inputs_copy:
                    if original_attention_mask is not None:
                        # Extend the mask with a 1 (attend to the new token)
                        inputs_copy["attention_mask"] = F.pad(
                            original_attention_mask,
                            (0, i + 1),
                            value=1
                        )
                    else:
                        inputs_copy["attention_mask"] = torch.ones_like(current_input_ids)
                
                logger.debug(f"Generated token {i+1}/{num_tokens}: '{token_text}' (ID: {next_token_id_scalar})")
        
        # Decode the entire generated sequence
        generated_ids = current_input_ids[0, original_seq_len:].tolist()
        generated_text = self.processor.tokenizer.decode(generated_ids)
        
        # Construct result
        result = {
            "inputs": inputs_copy,  # Updated inputs with the generated tokens
            "original_seq_len": original_seq_len,
            "generated_tokens": generated_tokens,
            "full_sequence": {
                "ids": current_input_ids[0].tolist(),
                "text": self.processor.tokenizer.decode(current_input_ids[0].tolist()),
                "prompt_length": original_seq_len,
                "generated_text": generated_text
            }
        }
        
        logger.info(f"Generated {len(generated_tokens)} tokens")
        
        return result
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        top_p: float = 1.0,
        top_k: int = 0
    ) -> torch.Tensor:
        """
        Sample a token from the logits using top-p and top-k sampling.
        
        Args:
            logits: Logits tensor from the model
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            
        Returns:
            Sampled token ID
        """
        # Default to greedy sampling if top_p=1.0 and top_k=0
        if top_p >= 1.0 and top_k <= 0:
            return torch.argmax(logits, dim=-1)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(probs, top_k, dim=-1)
            mask = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
            probs = probs * mask
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Keep at least top-1 token
            sorted_indices_to_remove[..., 0] = 0
            
            # Shift the indices to the right to keep the first token above threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # Sample from the filtered distribution
        sample = torch.multinomial(probs, 1).squeeze(-1)
        
        return sample