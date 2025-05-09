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
        
        # Default special token mappings (based on common invisible tokens)
        self.default_special_tokens = {
            13: "\\n",     # Newline
            28705: "_",    # Underscore placeholder (from old version)
            32: "␣",       # Space (using visible space symbol)
            9: "\\t",      # Tab
            10: "\\n",     # Alternative newline
            160: "\\xa0",  # Non-breaking space
            0: "[PAD]",    # Common padding token
            1: "[UNK]",    # Common unknown token
            2: "[CLS]",    # Common classification token
            3: "[SEP]",    # Common separator token
            4: "[MASK]"    # Common mask token
        }
        
        # Update with user-provided mappings
        self.special_tokens = dict(self.default_special_tokens)
        if special_tokens_map:
            self.special_tokens.update(special_tokens_map)
            
        # Try to automatically detect common special tokens from tokenizer
        self._detect_tokenizer_special_tokens()
    
    def _detect_tokenizer_special_tokens(self):
        """Attempt to detect and add special tokens from tokenizer."""
        try:
            # Try to get special tokens from tokenizer if available
            if hasattr(self.tokenizer, "special_tokens_map"):
                for token_name, token_value in self.tokenizer.special_tokens_map.items():
                    try:
                        # Get ID for this special token
                        token_id = self.tokenizer.convert_tokens_to_ids(token_value)
                        if token_id:
                            # Add to our mapping using [TOKEN_NAME] format
                            self.special_tokens[token_id] = f"[{token_name.upper()}]"
                    except:
                        pass
        except:
            # Ignore any errors in special token detection
            pass
    
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
        
        # Try normal decoding
        decoded = self.tokenizer.decode([token_id])
        
        # If decoding produces empty or whitespace-only result, use a placeholder
        if not decoded.strip():
            # Try to get token representation
            try:
                # Some tokenizers have convert_ids_to_tokens method
                if hasattr(self.tokenizer, "convert_ids_to_tokens"):
                    token = self.tokenizer.convert_ids_to_tokens(token_id)
                    return f"<{token}>" if token else f"<tok_{token_id}>"
                else:
                    # Fall back to showing token ID
                    return f"<tok_{token_id}>"
            except:
                # Last resort
                return f"<tok_{token_id}>"
        
        return decoded