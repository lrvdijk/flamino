"""
This module contains helpers to define the model vocabulary.
"""

from collections.abc import Sequence


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

START_OF_SEQ = "<start>"
END_OF_SEQ = "<end>"
PADDING = "<pad>"
UNKNOWN = "<unk>"
MASK = "<mask>"


class Alphabet:
    """
    Represents the vocabulary of a language model.
    
    This class focuses on tokens that can be represented by a single ASCII character, such as amino acids or nucleotides.
    Special tokens are required to start and end with < and >.
    """
    
    def __init__(
        self, 
        tokens: str, 
        special_tokens_prepend: Sequence[str] | None = None,  
        special_tokens_append: Sequence[str] | None = None,
    ):
        tok_prepend = list(special_tokens_prepend) if special_tokens_prepend else [START_OF_SEQ, END_OF_SEQ, PADDING]
         
        # Ensure UNKNOWN is always included as token
        tok_prepend.append(UNKNOWN)
        tok_append = list(special_tokens_append) if special_tokens_append else [MASK]
        
        self.tokens: list[str] = [
            *tok_prepend,
            *tokens,
            *tok_append
        ]
        
        self.longest_special_token: int = max(len(token) for token in tok_prepend + tok_append)
        self._token_to_id: dict[str, int] = {token: i for i, token in enumerate(self.tokens)}
        
    @classmethod
    def amino_acids(cls):
        return cls(AMINO_ACIDS)
        
    @classmethod
    def nucleotides(cls):
        return cls("ACGT")
        
    def token_ix(self, token: str) -> int:
        return self._token_to_id.get(token, self._token_to_id[UNKNOWN])
        
    def tokenize(self, text: str) -> list[int]:
        tokens: list[int] = []
        
        i = 0
        while i < len(text):
            if text[i] == "<":
                # Start of a special token
                match = None
                for j in range(i + self.longest_special_token, i, -1):
                    if text[i:j] in self._token_to_id:
                        match = text[i:j]
                        break
                        
                if match:
                    tokens.append(self._token_to_id[match])
                    i += len(match)
                else:
                    tokens.append(self._token_to_id[UNKNOWN])
                    i += 1
            else:
                tokens.append(self.token_ix(text[i]))
                i += 1
            
        return tokens
        
    def tok_to_str(self, token_ix: int) -> str:
        return self.tokens[token_ix]