# backends/logit_backend.py
"""
Logit lens backend for analyzing hidden states by projecting through the LM head.
"""

import torch
import gc
from typing import Dict, List, Any, Optional, Tuple
import logging
from runtime.hooks import register_hooks, remove_hooks
from backends.base_backend import BaseBackend

logger = logging.getLogger(__name__)

class LogitLensBackend(BaseBackend):
    """
    Backend for logit lens analysis, projecting hidden states through the LM head 
    to examine token predictions at different layers.
    """
    
    def __init__(self, model, device=None, cpu_offload=True):
        """
        Initialize the logit lens backend.
        
        Args:
            model: The model to analyze
            device: Device to run computations on
            cpu_offload: Whether to offload tensors to CPU when possible
        """
        super().__init__(model, device, cpu_offload)
        self.hooks = []
        self.hidden_states = {}
        self.hooked_layers = set()
        self.projections = {}
        
        # Get the LM head for projections
        if hasattr(model, 'lm_head'):
            self.lm_head = model.lm_head
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
            self.lm_head = model.language_model.lm_head
        else:
            raise ValueError("Could not find LM head in model. Required for logit lens.")
        
    def setup(self, layer_names: List[str]) -> None:
        """
        Set up hooks for the specified layers.
        
        Args:
            layer_names: List of layer names to analyze
        """
        self.cleanup()  # Clear any existing hooks
        self.hooked_layers = set(layer_names)
        
        # Register forward hooks for all specified layers
        hooks = register_hooks(
            self.model,
            layer_names,
            forward_hook=self._create_forward_hook
        )
        
        self.hooks.extend(hooks)
        logger.info(f"Registered hidden state hooks for {len(self.hooks)} layers")
        
    def _create_forward_hook(self, layer_name: str):
        """Create a forward hook for capturing hidden states."""
        def hook(module, inputs, outputs):
            hidden_state = None
            
            # Identify hidden state in outputs
            if isinstance(outputs, torch.Tensor):
                # Simplest case: output is the hidden state tensor
                hidden_state = outputs
            elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                # Common case: output is a tuple, first element is the hidden state
                hidden_state = outputs[0]
            
            if hidden_state is not None:
                # Store the hidden state, detached to avoid memory issues
                processed_hs = hidden_state.detach()
                if self.cpu_offload:
                    processed_hs = processed_hs.cpu()
                self.hidden_states[layer_name] = processed_hs
            else:
                logger.warning(f"Could not identify hidden state in output of layer '{layer_name}'")
        
        return hook
        
    def compute(self, inputs: Dict[str, torch.Tensor], token_indices: List[int], concept_ids: Dict[str, List[int]] = None) -> Dict[str, Any]:
        """
        Compute logit lens projections for the given inputs and target indices.
        
        Projects hidden states through the LM head to examine what tokens would
        be predicted at each layer, which helps understand how representations
        evolve throughout the model.
        
        Args:
            inputs: Model inputs
            token_indices: Indices of token positions to analyze
            concept_ids: Optional dict mapping concept names to token IDs to track
            
        Returns:
            Dictionary with logit lens projections per layer and token
        """
        self.clear_results()
        
        if not token_indices:
            logger.warning("No target indices provided. Cannot compute logit lens projections.")
            return {}
        
        # Ensure model in eval mode
        self.model.eval()
        
        # Forward pass to capture hidden states for all layers
        with torch.no_grad():
            _ = self.model(**{k: v for k, v in inputs.items() if k != "token_type_ids"},
                        output_hidden_states=True)
        
        # Compute projections for each token position in each layer
        self.projections = self._compute_projections(token_indices, concept_ids)
        
        return {"projections": self.projections}

    def _compute_projections(self, token_indices: List[int], concept_ids: Dict[str, List[int]] = None) -> Dict[str, Dict[int, Dict[str, Any]]]:
        """
        Project hidden states through LM head for specific token positions.
        
        This computes what tokens would be predicted at each layer, allowing us to
        see how token representations evolve through the model.
        
        Args:
            token_indices: Indices of token positions to analyze
            concept_ids: Optional dict mapping concept names to token IDs to track
            
        Returns:
            Nested dict: {layer_name: {token_idx: {projection_data}}}
        """
        projections = {}
        
        # Process each layer
        for layer_name, hidden_states in self.hidden_states.items():
            layer_projections = {}
            
            # Move hidden states to LM head device if needed
            hidden_states_device = hidden_states.device
            lm_head_device = next(self.lm_head.parameters()).device
            
            if hidden_states_device != lm_head_device:
                hidden_states = hidden_states.to(lm_head_device)
            
            # Process each token index
            try:
                for token_idx in token_indices:
                    if token_idx >= hidden_states.shape[1]:
                        logger.warning(f"Token index {token_idx} exceeds sequence length. Skipping.")
                        continue
                    
                    # Extract hidden state for this token position
                    token_hidden = hidden_states[:, token_idx:token_idx+1]
                    
                    # Project through LM head
                    token_logits = self.lm_head(token_hidden).float()
                    token_probs = torch.softmax(token_logits, dim=-1)
                    
                    # Get top-k predictions
                    k = 5  # Number of top predictions to track
                    top_probs, top_indices = torch.topk(token_probs[0, 0], k)
                    
                    # FIX: Process concept tracking if provided
                    concept_predictions = {}
                    if concept_ids:
                        for concept_name, ids in concept_ids.items():
                            if not ids:
                                continue
                                
                            # Convert IDs to tensor if needed
                            if not isinstance(ids, torch.Tensor):
                                ids_tensor = torch.tensor(ids, device=token_probs.device)
                            else:
                                ids_tensor = ids.to(token_probs.device)
                                
                            # Get probabilities for this concept's tokens
                            concept_probs = token_probs[0, 0, ids_tensor]
                            max_prob_idx = torch.argmax(concept_probs).item()
                            max_prob = concept_probs[max_prob_idx].item()
                            max_id = ids[max_prob_idx] if max_prob_idx < len(ids) else ids[0]
                            
                            concept_predictions[concept_name] = {
                                "probability": max_prob,
                                "token_id": max_id
                            }
                    
                    # Store results for this token
                    token_projections = {
                        "logits": token_logits[0, 0].detach().cpu() if self.cpu_offload else token_logits[0, 0].detach(),
                        "top_k": {
                            "indices": top_indices.cpu().tolist() if self.cpu_offload else top_indices.tolist(),
                            "probs": top_probs.cpu().tolist() if self.cpu_offload else top_probs.tolist()
                        },
                        "concept_predictions": concept_predictions
                    }
                    
                    layer_projections[token_idx] = token_projections
            
            finally:
                # Clean up if moved to different device
                if hidden_states_device != lm_head_device:
                    del hidden_states
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Store projections for this layer
            projections[layer_name] = layer_projections
        
        return projections
        
    def clear_results(self) -> None:
        """Clear previous computation results."""
        self.hidden_states.clear()
        self.projections.clear()
        
    def cleanup(self) -> None:
        """Remove all hooks and free resources."""
        remove_hooks(self.hooks)
        self.hooks.clear()
        
        self.clear_results()
        self.clear_cache()
        
        self.hooked_layers.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()