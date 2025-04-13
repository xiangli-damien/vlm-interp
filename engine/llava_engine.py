# -*- coding: utf-8 -*-
"""
Defines the LLaVANextEngine class for interacting with LLaVA-Next models.
"""

import torch
from PIL import Image
from typing import Dict, Any, Optional, Union, List, Tuple

# Assuming utils are structured as requested in the parent directory
from utils.data_utils import load_image, build_conversation
from utils.model_utils import load_model, get_llm_attention_layer_names

# Import necessary classes from transformers
# (Ensure these specific versions are handled by your environment setup)
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

class LLaVANextEngine:
    """
    A VLM engine specifically designed for LLaVA-Next models.
    """
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        use_flash_attn: bool = False,
        enable_gradients: bool = False
    ):
        
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor: Optional[LlavaNextProcessor] = None
        self.model: Optional[LlavaNextForConditionalGeneration] = None
        self.load_in_4bit = load_in_4bit if torch.cuda.is_available() else False # 4bit only works on CUDA
        self.use_flash_attn = use_flash_attn if torch.cuda.is_available() else False # Flash Attn typically needs CUDA
        self.enable_gradients = enable_gradients

        # Decide if 4bit loading overrides gradient request (common case)
        _effective_enable_gradients = self.enable_gradients
        if self.load_in_4bit and self.enable_gradients:
            print("Warning: Requesting gradients with 4-bit loading. Gradient support is limited or unavailable with 4-bit quantization. Proceeding but expect potential issues.")
            # We pass enable_gradients=True to load_model, but it might internally disable them if quantizing
            # Or raise an error depending on the transformers version and setup.

        # Load model and processor upon initialization
        self._load_model(_effective_enable_gradients)

    def _load_model(self, enable_gradients: bool):
        """
        Internal method to load the model and processor using the utility function.
        """
        print(f"Engine attempting to load model '{self.model_id}'...")
        try:
            # Use the utility function from model_utils
            self.model, self.processor = load_model(
                model_id=self.model_id,
                use_flash_attn=self.use_flash_attn,
                load_in_4bit=self.load_in_4bit,
                enable_gradients=enable_gradients # Pass the potentially adjusted flag
            )
            # Ensure the model is on the correct device, especially if device_map wasn't used or effective
            if self.model is not None and hasattr(self.model, 'device') and self.model.device != torch.device(self.device):
                 if 'device_map' not in self.model.config.to_dict(): # Only move if device_map wasn't used
                      try:
                          print(f"Manually moving model to configured device: {self.device}")
                          self.model.to(self.device)
                      except Exception as e:
                          print(f"Warning: Failed to manually move model to {self.device}. Error: {e}")
            elif self.model is not None:
                 # If device_map was used, model parts might be on different devices.
                 # The engine's self.device is less relevant in that case.
                 print(f"Model loaded with device_map='auto'. Main device may vary.")
                 self.device = str(self.model.device) # Update engine device based on model's primary device

        except ImportError as e:
             print(f"ImportError during model loading: {e}. Make sure necessary libraries (like bitsandbytes, flash-attn) are installed if using quantization or flash attention.")
             raise
        except Exception as e:
            print(f"Error loading model '{self.model_id}' in engine: {e}")
            self.model = None
            self.processor = None
            raise # Re-raise the exception to indicate failure

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
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model and processor must be loaded before building inputs.")

        # Load image if a string path/URL is provided
        # Use verbose=False to reduce console clutter from the utility function
        loaded_image = load_image(image, resize_to=None, verbose=False) # Let processor handle resizing

        # Build conversation structure using the utility function
        conversation = build_conversation(prompt, conversation_format=conversation_format)

        # Format prompt using the processor's chat template
        # The processor handles the image token placement correctly during __call__
        try:
             if hasattr(self.processor, "apply_chat_template") and self.processor.chat_template:
                 formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
             else:
                 # Basic fallback if chat template isn't setup or applicable
                 image_token = getattr(self.processor, "image_token", "<image>")
                 if conversation_format and isinstance(conversation, list): # Use first user message
                     text_content = next((item['text'] for item in conversation[0]['content'] if item['type'] == 'text'), "")
                     formatted_prompt = f"USER: {image_token}\n{text_content} ASSISTANT:"
                 else: # Simple string format
                      formatted_prompt = f"{prompt} {image_token}"
                 print("Warning: Using basic prompt formatting; chat template may not be applied.")

        except Exception as e:
             print(f"Warning: Error applying chat template: {e}. Using basic prompt format.")
             image_token = getattr(self.processor, "image_token", "<image>")
             formatted_prompt = f"USER: {image_token}\n{prompt} ASSISTANT:"

        # Prepare inputs using the processor
        # The processor handles image resizing, normalization, and tokenization
        inputs = self.processor(
            images=loaded_image,
            text=formatted_prompt,
            return_tensors="pt"
        )

        # Move inputs to the model's primary device
        # This handles cases where device_map="auto" might place parts differently.
        # We send inputs to the device where the computation likely starts (usually the embedding layer's device).
        try:
             model_device = self.model.device
             inputs = {k: v.to(model_device) for k, v in inputs.items()}
        except Exception as e:
             print(f"Warning: Failed to move inputs to model device {self.model.device}. Error: {e}")
             # Fallback to engine's configured device if moving to model.device fails
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
        num_beams: int = 3, # Adjusted default based on notebook usage
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
            image (Union[str, Image.Image]): PIL Image object or path/URL to an image.
            prompt (str): Text prompt to accompany the image.
            max_new_tokens (int): Maximum number of new tokens to generate.
            num_beams (int): Number of beams for beam search. Set to 1 for greedy decoding.
            temperature (float): Temperature for sampling (used if num_beams=1 and do_sample=True).
            top_p (Optional[float]): Top-p (nucleus) sampling threshold.
            top_k (Optional[int]): Top-k sampling threshold.
            return_dict_in_generate (bool): Whether the underlying `model.generate` should return a dictionary output. Defaults to True.
            output_attentions (bool): Whether to include attention weights in the output dictionary (requires return_dict_in_generate=True).
            output_hidden_states (bool): Whether to include hidden states in the output dictionary (requires return_dict_in_generate=True).
            **generation_kwargs: Additional keyword arguments passed directly to `model.generate`.

        Returns:
            Tuple[str, Optional[Dict[str, Any]]]:
                - The generated text response (cleaned).
                - The full output dictionary from `model.generate` if return_dict_in_generate is True, otherwise None.

        Raises:
            ValueError: If the model or processor is not loaded.
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
             if num_beams == 1: # Sampling is typically used with greedy decoding (num_beams=1)
                  do_sample = True
                  print(f"  Sampling enabled (temp={temperature}, top_p={top_p}, top_k={top_k})")
             else:
                  print(f"  Warning: Sampling parameters (temp/top_p/top_k) provided but num_beams > 1. Beam search will be used.")


        with torch.inference_mode(): # Use inference_mode for potentially better performance than no_grad
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
                    eos_token_id=self.processor.tokenizer.eos_token_id, # Help generation stop cleanly
                    pad_token_id=self.processor.tokenizer.pad_token_id, # Important for batching if used, good practice otherwise
                    **generation_kwargs
                )
            except Exception as e:
                 print(f"Error during model.generate: {e}")
                 # Attempt to get more info if possible
                 if 'CUDA out of memory' in str(e):
                      print("CUDA OOM Error detected. Try reducing batch size, sequence length, or using quantization/offloading.")
                 elif 'Input type' in str(e) and 'CPU' in str(e) and 'CUDA' in str(e):
                      print("Device mismatch error detected. Ensure model and inputs are on the same device.")
                 raise # Re-raise the exception


        # Decode generated sequence(s)
        # `outputs.sequences` contains the full sequence (input + generated)
        # We typically decode the first sequence in the batch (index 0)
        raw_generated_text = self.processor.decode(outputs.sequences[0], skip_special_tokens=True)

        # Clean the generated text to remove the prompt/input part
        cleaned_text = raw_generated_text
        # Use common separators related to chat templates to find the start of the response
        separators = ["ASSISTANT:", "[/INST]", "GPT:", "\n "] # Add more as needed
        original_prompt_text = prompt # Use the original prompt text for a more robust split if possible

        # Try splitting based on the original prompt first (might be less reliable with complex templates)
        # if original_prompt_text in cleaned_text:
        #      parts = cleaned_text.split(original_prompt_text, 1)
        #      if len(parts) > 1:
        #           cleaned_text = parts[1].strip()

        # More robust approach using known separators
        found_separator = False
        for sep in separators:
             if sep in cleaned_text:
                  parts = cleaned_text.split(sep, 1)
                  if len(parts) > 1:
                      cleaned_text = parts[1].strip()
                      found_separator = True
                      # print(f"Cleaned text using separator '{sep}'") # Debugging print
                      break # Use the first separator found

        if not found_separator:
             print("Warning: Could not reliably separate prompt from response using known separators. Returning full decoded text.")
             # As a last resort, maybe try removing the input prompt if it appears exactly at the beginning
             if cleaned_text.startswith(original_prompt_text):
                  cleaned_text = cleaned_text[len(original_prompt_text):].strip()


        # Return the cleaned text and the full output dictionary if requested
        output_dict = outputs if return_dict_in_generate else None

        print("Generation complete.")
        return cleaned_text, output_dict

    def get_attention_layer_names(self) -> List[str]:
        """
        Returns the names of attention layers in the model.
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