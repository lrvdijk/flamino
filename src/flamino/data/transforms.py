"""
Jax Grain Transforms to prepare data for training and inference.
"""

import numpy as np
from grain import transforms
from typing_extensions import override

from flamino.vocab import Vocabulary


class TokenizeTransform(transforms.Map):
    """
    Transform an input string into a sequence of token indices.
    """

    def __init__(self, vocab: Vocabulary, append_start_stop: bool = True):
        self.vocab: Vocabulary = vocab
        self.append_start_stop: bool = append_start_stop

    @override
    def map(self, element: str) -> np.ndarray:
        if self.append_start_stop:
            return np.array([self.vocab.start, *self.vocab.tokenize(element), self.vocab.end])
        else:
            return np.array(list(self.vocab.tokenize(element)))


class PadTransform(transforms.Map):
    """
    Transform an input sequence into a padded sequence.
    """

    def __init__(self, vocab: Vocabulary, max_len: int):
        self.vocab: Vocabulary = vocab
        self.max_len: int = max_len

    @override
    def map(self, element: np.ndarray | list[int]) -> dict[str, np.ndarray]:
        if isinstance(element, list):
            element = np.array(element)

        padded = np.full(self.max_len, self.vocab.pad, dtype=np.int32)
        padded[: len(element)] = element

        padding_mask = np.zeros(self.max_len, dtype=np.bool_)
        padding_mask[len(element) :] = True

        return {"padded": padded, "padding_mask": padding_mask}


class MaskTransform(transforms.RandomMap):
    """
    Transform an input sequence into a sequence with masked tokens.
    """

    def __init__(self, vocab: Vocabulary, mask_prob: float = 0.4):
        self.vocab: Vocabulary = vocab
        self.mask_prob: float = mask_prob

        try:
            _ = self.vocab.mask
        except AttributeError:
            raise ValueError("Vocabulary must have a mask token")

    @override
    def random_map(self, element: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
        # Exclude start and end tokens from masking
        excl = (element == self.vocab.start) | (element == self.vocab.end)

        possible_ix = np.arange(len(element))[~excl]
        num_masked = int(len(possible_ix) * self.mask_prob)
        mask_ix = rng.choice(possible_ix, num_masked, replace=False)

        masked = np.copy(element)
        masked[mask_ix] = self.vocab.mask

        return {"orig": element, "masked": masked}
