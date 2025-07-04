"""
This module contains helpers to define the model vocabulary.
"""

from collections.abc import Iterable

import numpy as np


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
        tokens: list[str], 
    ):
        self.tokens: list[str] = tokens
        self._token_to_id: dict[str, int] = {token: i for i, token in enumerate(self.tokens)}
        
        required_tokens = {START_OF_SEQ, END_OF_SEQ, PADDING, UNKNOWN}
        missing = required_tokens - set(self._token_to_id.keys())
        assert not missing, f"Alphabet requires the presence of tokens {required_tokens}, missing tokens {missing}"
        
        self.longest_special_token: int = max(len(token) for token in self.tokens if token.startswith("<") and token.endswith(">"))
        
    @classmethod
    def from_str(cls, string: str) -> "Alphabet":
        return cls([
            START_OF_SEQ,
            END_OF_SEQ,
            PADDING,
            UNKNOWN,
            *list(string)
        ])
        
    def __getattr__(self, name: str) -> int:
        token_name = f"<{name}>"
        
        if token_name in self._token_to_id:
            return self._token_to_id[token_name]
        
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def token_ix(self, token: str) -> int:
        """
        Get the index of a token. If the token is not found, return the unknown token index.
        
        Arguments
        ---------
        token: str
            The token to get the index of.
            
        Returns
        -------
        int
            The index of the token.
        """
        return self._token_to_id.get(token, self.unk)
        
    def tokenize(self, text: str) -> Iterable[int]:
        """
        Tokenize a string into a sequence of token indices.
        
        Arguments
        ---------
        text: str
            The text to tokenize.
            
        Yields
        -------
        Iterable[int]
            The token indices.
        """
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
                    yield self._token_to_id[match]
                    i += len(match)
                else:
                    # If the single character '<' is a valid token, yield its index
                    # else yield the index of the unknown token
                    yield self.token_ix(text[i])
                    i += 1
            else:
                yield self.token_ix(text[i])
                i += 1
            
    def tokenize_to_arr(self, texts: list[str] | str) -> np.ndarray:
        """
        Tokenize a list of texts into a NumPy array of token indices.
        
        Arguments
        ---------
        texts: list[str] | str
            The text(s) to tokenize. 
            
        Returns
        -------
        np.ndarray
            A (num_texts, max_len) array of token indices for each text.
        """
        
        if isinstance(texts, str):
            texts = [texts]
            
        tokens = [
            np.array([self.start, *self.tokenize(text), self.end])
            for text in texts
        ]
        
        max_len = max(len(tokens) for tokens in tokens)
        out_arr = np.full((len(texts), max_len), self.pad, dtype=np.int32)
        
        for i, text_tokens in enumerate(tokens):
            out_arr[i, :len(text_tokens)] = text_tokens
            
        return out_arr
        
    def tok_to_str(self, token_ix: int) -> str:
        """
        Convert a token index to its string representation.
        
        Arguments
        ---------
        token_ix: int
            The token index to convert.
            
        Returns
        -------
        str
            The string representation of the token.
        """
        return self.tokens[token_ix]
        
        
        
# Ensure token indices match those of the original ESM-2
ESM2 = Alphabet([
    START_OF_SEQ,
    PADDING,
    END_OF_SEQ,
    UNKNOWN,
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
    ".",  # Unsure what these are
    "-",
    MASK,
])