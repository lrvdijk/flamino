"""
Flax NNX-based implementation of the RoPE (Rotary Positional Embeddings) technique.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Dtype


rope_sin_cos_table_cache: dict[tuple[int, Dtype], tuple[jax.Array, jax.Array]] = {}


def apply_rope(x: jax.Array, sin: jax.Array, cos: jax.Array) -> jax.Array:
    assert x.ndim == 2, "No batch dimension allowed for x, expected dimensions (seq_len, embedding_dim)"
    seq_len = x.shape[0]

    sin = sin[:seq_len, :]
    cos = cos[:seq_len, :]

    return (x * cos) + (rotate_half(x) * sin)


def rotate_half(x: jax.Array) -> jax.Array:
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


class Frequencies(nnx.Variable[jax.Array]):
    pass


class RoPE(nnx.Module):
    """
    Implementation of the RoPE (Rotary Positional Encodings) technique.

    Computes caches the sin and cos matrices on-the-fly, depending on the given sequence length.
    """

    def __init__(self, d_embed: int, theta: float = 10000.0):
        if d_embed % 2 != 0:
            raise ValueError(f"`d_embed` must be even, got {d_embed}.")

        self.d_embed: int = d_embed
        self.theta: float = theta

        self._freqs: Frequencies = Frequencies(
            1.0 / (self.theta ** (jnp.arange(0, self.d_embed, 2) / self.d_embed))
        )

    def _compute_sin_cos_table(self, seq_len: int, dtype: Dtype) -> tuple[jax.Array, jax.Array]:
        freqs = self._freqs.value
        freqs = jnp.concatenate([freqs, freqs], axis=-1)
        positions = jnp.arange(seq_len)
        angles = jnp.outer(positions, freqs)

        sin_table = jnp.sin(angles).astype(dtype)
        cos_table = jnp.cos(angles).astype(dtype)

        return sin_table, cos_table

    def __call__(self, x: jax.Array):
        assert x.ndim == 2
        seq_len, embed_size = x.shape
        assert embed_size == self.d_embed, "Sequence embedding dimension mismatch"

        with jax.ensure_compile_time_eval():
            cache_key = (embed_size, x.dtype)

            # Check global cache for the given embedding size and dtype
            if cache_key not in rope_sin_cos_table_cache:
                sin_table, cos_table = self._compute_sin_cos_table(seq_len, x.dtype)
                rope_sin_cos_table_cache[cache_key] = (sin_table, cos_table)
            else:
                sin_table, cos_table = rope_sin_cos_table_cache[cache_key]

            # Re-compute sin/cos tables if length of the current sequence is greater
            freq_seq_len = sin_table.shape[0]
            if freq_seq_len < seq_len:
                sin_table, cos_table = self._compute_sin_cos_table(seq_len, x.dtype)
                rope_sin_cos_table_cache[cache_key] = (sin_table, cos_table)

        return apply_rope(x, sin_table, cos_table)