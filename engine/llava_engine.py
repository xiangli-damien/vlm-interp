"""
Defines the LLaVANextEngine class for interacting with LLaVA-Next models.
"""

import torch
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from preprocessing.data_utils import load_image, build_conversation
from core.model_utils import load_model, get_llm_attention_layer_names


class LLaVANextEngine:
    """
    A VLM engine specifically designed for LLaVA-Next models.
    
    Handles model loading, input preparation, generation, and provides
    access to lower-level model components for analysis.
    """
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        use_flash_attn: bool = False,
        enable_gradients: bool = False
    ):
        """
        Initialize the LLaVA-Next engine.
        
        Args:
            model_id: HuggingFace model ID
            device: Device to use ('cuda', 'cpu', etc.)
            load_in_4bit: Load model in 4-bit quantization
            use_flash_attn: Use Flash Attention 2 for faster inference
            enable_gradients: Enable gradients for the model
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[LlavaNextProcessor] = None
        self.model: Optional[LlavaNextForConditionalGeneration] = None
        self.load_in_4bit = load_in_4bit if torch.cuda.is_available() else False
        self.use_flash_attn = use_flash_attn if torch.cuda.is_available() else False
        self.enable_gradients = enable_gradients

        # Decide if 4bit loading overrides gradient request
        _effective_enable_gradients = self.enable_gradients
        if self.load_in_4bit and self.enable_gradients:
            print("Warning: Requesting gradients with 4-bit loading. Gradient support is limited or unavailable with 4-bit quantization. Proceeding but expect potential issues.")

        # Load model and processor upon initialization
        self._load_model(_effective_enable_gradients)

    def _load_model(self, enable_gradients: bool):
        """
        Internal method to load the model and processor.
        
        Args:
            enable_gradients: Whether to enable gradients
        """
        print(f"Engine attempting to load model '{self.model_id}'...")
        try:
            # Use the utility function from model_utils
            self.model, self.processor = load_model(
                model_id=self.model_id,
                use_flash_attn=self.use_flash_attn,
                load_in_4bit=self.load_in_4bit,
                enable_gradients=enable_gradients
            )
            # Ensure the model is on the correct device
            if self.model is not None and hasattr(self.model, 'device') and self.model.device != torch.device(self.device):
                 if 'device_map' not in self.model.config.to_dict():
                      try:
                          print(f"Manually moving model to configured device: {self.device}")
                          self.model.to(self.device)
                      except Exception as e:
                          print(f"Warning: Failed to manually move model to {self.device}. Error: {e}")
            elif self.model is not None:
                 print(f"Model loaded with device_map='auto'. Main device may vary.")
                 self.device = str(self.model.device)

        except ImportError as e:
             print(f"ImportError during model loading: {e}. Make sure necessary libraries (like bitsandbytes, flash-attn) are installed if using quantization or flash attention.")
             raise
        except Exception as e:
            print(f"Error loading model '{self.model_id}' in engine: {e}")
            self.model = None
            self.processor = None
            raise

        if self.model is None or self.processor is None:
             raise RuntimeError(f"Failed to initialize model or processor for {self.model_id}")

        print(f"Engine successfully loaded model and processor. Model device: {self.model.device}")


    def build_inputs(
        self,
        image: Union[str, Image.Image],
        prompt: str = "Describe this image in detail.",
        conversation_format: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Build model inputs from an image and a text prompt.
        
        Args:
            image: PIL Image object or path/URL to an image
            prompt: Text prompt to accompany the image
            conversation_format: Use conversation format for the prompt
            
        Returns:
            Dictionary of model inputs
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before building inputs.")

        # Load image if a string path/URL is provided
        loaded_image = load_image(image, resize_to=None, verbose=False)

        # Build conversation structure
        conversation = build_conversation(prompt, conversation_format=conversation_format)

        # Format prompt using the processor's chat template
        try:
             if hasattr(self.processor, "apply_chat_template") and self.processor.chat_template:
                 formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
             else:
                 # Basic fallback
                 image_token = getattr(self.processor, "image_token", "<image>")
                 if conversation_format and isinstance(conversation, list):
                     text_content = next((item['text'] for item in conversation[0]['content'] if item['type'] == 'text'), "")
                     formatted_prompt = f"USER: {image_token}\n{text_content} ASSISTANT:"
                 else:
                      formatted_prompt = f"{prompt} {image_token}"
                 print("Warning: Using basic prompt formatting; chat template may not be applied.")

        except Exception as e:
             print(f"Warning: Error applying chat template: {e}. Using basic prompt format.")
             image_token = getattr(self.processor, "image_token", "<image>")
             formatted_prompt = f"USER: {image_token}\n{prompt} ASSISTANT:"

        # Prepare inputs using the processor
        inputs = self.processor(
            images=loaded_image,
            text=formatted_prompt,
            return_tensors="pt"
        )

        # Move inputs to the model's primary device
        try:
             model_device = self.model.device
             inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception as e:
             print(f"Warning: Failed to move inputs to model device {self.model.device}. Error: {e}")
             # Fallback to engine's configured device
             try:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
             except Exception as inner_e:
                  print(f"Warning: Also failed to move inputs to engine device {self.device}. Error: {inner_e}")

        return inputs

    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str = "Describe this image in detail.",
        max_new_tokens: int = 256,
        num_beams: int = 3,
        temperature: float = 1.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        return_dict_in_generate: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **generation_kwargs: Any
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Generate a response based on an image and prompt.

        Args:
            image: PIL Image object or path/URL to an image
            prompt: Text prompt to accompany the image
            max_new_tokens: Maximum number of new tokens to generate
            num_beams: Number of beams for beam search
            temperature: Temperature for sampling
            top_p: Top-p (nucleus) sampling threshold
            top_k: Top-k sampling threshold
            return_dict_in_generate: Return a dictionary of generation outputs
            output_attentions: Include attention weights in the output
            output_hidden_states: Include hidden states in the output
            **generation_kwargs: Additional generation parameters

        Returns:
            Tuple of (generated_text, full_output_dict)
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before generation.")

        inputs = self.build_inputs(image, prompt)

        print(f"Generating response with max_new_tokens={max_new_tokens}, num_beams={num_beams}...")

        # Set model to evaluation mode
        self.model.eval()

        # Determine if sampling should be enabled based on parameters
        do_sample = False
        if temperature != 1.0 or top_p is not None or top_k is not None:
             if num_beams == 1:
                  do_sample = True
                  print(f"  Sampling enabled (temp={temperature}, top_p={top_p}, top_k={top_k})")
             else:
                  print(f"  Warning: Sampling parameters provided but num_beams > 1. Beam search will be used.")

        with torch.inference_mode():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    return_dict_in_generate=return_dict_in_generate,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    **generation_kwargs
                )
            except Exception as e:
                 print(f"Error during model.generate: {e}")
                 if 'CUDA out of memory' in str(e):
                      print("CUDA OOM Error detected. Try reducing batch size, sequence length, or using quantization/offloading.")
                 elif 'Input type' in str(e) and 'CPU' in str(e) and 'CUDA' in str(e):
                      print("Device mismatch error detected. Ensure model and inputs are on the same device.")
                 raise

        # Decode generated sequence
        raw_generated_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)

        # Clean the generated text to remove the prompt/input part
        cleaned_text = raw_generated_text
        separators = ["ASSISTANT:", "[/INST]", "GPT:", "\n "]
        
        # Try splitting based on known separators
        found_separator = False
        for sep in separators:
             if sep in cleaned_text:
                  parts = cleaned_text.split(sep, 1)
                  if len(parts) > 1:
                      cleaned_text = parts[1].strip()
                      found_separator = True
                      break

        if not found_separator:
             print("Warning: Could not reliably separate prompt from response using known separators. Returning full decoded text.")
             if cleaned_text.startswith(prompt):
                  cleaned_text = cleaned_text[len(prompt):].strip()

        # Return the cleaned text and the full output dictionary if requested
        output_dict = outputs if return_dict_in_generate else None

        print("Generation complete.")
        return cleaned_text, output_dict

    def get_attention_layer_names(self) -> List[str]:
        """
        Returns the names of attention layers in the model.
        
        Returns:
            List of layer names
        """
        if self.model is None:
            raise ValueError("Model must be loaded before getting attention layer names.")

        return get_llm_attention_layer_names(self.model)

    def get_model(self) -> Optional[LlavaNextForConditionalGeneration]:
        """Returns the loaded model instance."""
        return self.model

    def get_processor(self) -> Optional[LlavaNextProcessor]:
        """Returns the loaded processor instance."""
        return self.processor