# runtime/decode.py
class TokenDecoder:
    """
    Utility class for robust token decoding with special token handling.
    Ensures special tokens and invisible characters are properly displayed.
    """
    
    def __init__(self, tokenizer, special_tokens_map=None):
        """
        Initialize the token decoder.
        
        Args:
            tokenizer: The tokenizer to use for decoding
            special_tokens_map: Optional dictionary mapping token IDs to display text
        """
        self.tokenizer = tokenizer
        
        # Default special token mappings with common tokenizer-specific mappings
        self.default_special_tokens = {
            0: "<s>",      # Common BOS token (often 0 or 1 depending on the model)
            1: "</s>",     # Common EOS token
            2: "<unk>",    # Common UNK token
            3: "<pad>",    # Common PAD token
            13: "\\n",     # Newline
            28705: "_",    # Underscore placeholder
            32: "␣",       # Space (using visible space symbol)
            9: "\\t",      # Tab
            10: "\\n",     # Alternative newline
            160: "\\xa0",  # Non-breaking space
        }
        
        # Update with user-provided mappings
        self.special_tokens = dict(self.default_special_tokens)
        if special_tokens_map:
            self.special_tokens.update(special_tokens_map)
            
        # Aggressively try to detect model-specific special tokens
        self._detect_model_special_tokens()
    
    def _detect_model_special_tokens(self):
        """Detect model-specific special tokens using multiple methods."""
        try:
            # Method 1: Check for special_tokens_map attribute
            if hasattr(self.tokenizer, "special_tokens_map"):
                for token_name, token_value in self.tokenizer.special_tokens_map.items():
                    try:
                        # Single token case
                        if isinstance(token_value, str):
                            token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                            if token_id != self.tokenizer.unk_token_id:  # Avoid unknown token mismatches
                                self.special_tokens[token_id] = token_value
                        # List of tokens case
                        elif isinstance(token_value, list):
                            for token in token_value:
                                token_id = self.tokenizer.convert_tokens_to_ids(token)
                                if token_id != self.tokenizer.unk_token_id:
                                    self.special_tokens[token_id] = token
                    except Exception as e:
                        print(f"Error processing token {token_name}: {e}")
            
            # Method 2: Try common token attributes directly
            token_attrs = [
                ('bos_token', 'bos_token_id', '<s>'),
                ('eos_token', 'eos_token_id', '</s>'),
                ('unk_token', 'unk_token_id', '<unk>'),
                ('pad_token', 'pad_token_id', '<pad>'),
                ('mask_token', 'mask_token_id', '<mask>'),
                ('cls_token', 'cls_token_id', '<cls>'),
                ('sep_token', 'sep_token_id', '<sep>')
            ]
            
            for token_attr, id_attr, fallback in token_attrs:
                try:
                    if hasattr(self.tokenizer, token_attr) and hasattr(self.tokenizer, id_attr):
                        token = getattr(self.tokenizer, token_attr)
                        token_id = getattr(self.tokenizer, id_attr)
                        
                        if token and token_id is not None:
                            # If the token is a string, use it directly
                            if isinstance(token, str):
                                self.special_tokens[token_id] = token
                            # If it's an object with a __str__ method (like AddedToken), use that
                            else:
                                self.special_tokens[token_id] = str(token)
                except Exception as e:
                    print(f"Error processing {token_attr}: {e}")
            
            # For debug - print all detected special tokens
            print("Detected special tokens:")
            for token_id, token_text in self.special_tokens.items():
                print(f"  ID {token_id}: '{token_text}'")
                
        except Exception as e:
            print(f"Error detecting special tokens: {e}")
    
    def decode_token(self, token_id: int) -> str:
        """
        Decode a single token ID with special token handling.
        
        Args:
            token_id: The token ID to decode
            
        Returns:
            Human-readable representation of the token
        """
        # Check if this is a special token we know about
        if token_id in self.special_tokens:
            return self.special_tokens[token_id]
        
        # Try to get the token before decoding (some tokenizers can do this directly)
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            try:
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                # If this gives us something meaningful, use it
                if token and token != self.tokenizer.unk_token:
                    return token
            except:
                pass
            
        # Try normal decoding
        try:
            decoded = self.tokenizer.decode([token_id])
            
            # If decoding produces empty or whitespace-only result, use a placeholder
            if not decoded.strip():
                return f"<tok_{token_id}>"
            
            return decoded
        except:
            # Last resort
            return f"<tok_{token_id}>"