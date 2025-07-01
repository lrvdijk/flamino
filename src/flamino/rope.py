"""
Flax NNX-based implementation of the RoPE (Rotary Positional Embeddings) technique.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.attention import dot_product_attention
from flax.nnx.nn import dtypes
from flax.typing import (
    Dtype,
    PrecisionLike,
    PromoteDtypeFn,
)


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


class SinCosTable(nnx.Variable[jax.Array]):
    pass


# No batching of frequency vector and sin/cos tables
state_axes = nnx.StateAxes({
    Frequencies: None,
    SinCosTable: None,
})


class RoPE(nnx.Module):
    """
    Implementation of the RoPE (Rotary Positional Encodings) technique.

    Computes caches the sin and cos matrices on-the-fly, depending on the given sequence length.
    """

    def __init__(self, d_embed: int, theta: float = 10000.0, *, rngs: nnx.Rngs):
        if d_embed % 2 != 0:
            raise ValueError(f"`d_embed` must be even, got {d_embed}.")

        self.d_embed: int = d_embed
        self.theta: float = theta

        self._freqs: Frequencies = Frequencies(1.0 / (self.theta ** (jnp.arange(0, self.d_embed, 2) / self.d_embed)))
        self._cached_sin: SinCosTable = SinCosTable(jnp.zeros(0))
        self._cached_cos: SinCosTable = SinCosTable(jnp.zeros(0))

        _, _ = self._update_cache(1024)

    def _update_cache(self, seq_len: int) -> tuple[jax.Array, jax.Array]:
        seq_len_cached = self._cached_sin.value.shape[0] if self._cached_sin.value is not None else 0

        if (
            self._cached_sin.value is None
            or self._cached_cos.value is None
            or seq_len > seq_len_cached
        ):
            freqs = self._freqs.value
            freqs = jnp.concatenate([freqs, freqs], axis=-1)
            positions = jnp.arange(seq_len)
            angles = jnp.outer(positions, freqs)

            self._cached_sin = SinCosTable(jnp.sin(angles))
            self._cached_cos = SinCosTable(jnp.cos(angles))

        return self._cached_sin.value, self._cached_cos.value

    @nnx.vmap(in_axes=(state_axes, 0), out_axes=0)
    def __call__(self, x: jax.Array):
        sin_table, cos_table = self._update_cache(x.shape[0])

        return apply_rope(x, sin_table, cos_table)
        
    def dot_product_attention(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        bias: jax.Array | None = None,
        mask: jax.Array | None = None,
        broadcast_dropout: bool = True,
        dropout_rng: jax.Array | None = None,
        dropout_rate: float = 0.0,
        deterministic: bool = False,
        dtype: Dtype | None = None,
        precision: PrecisionLike = None,
        module: nnx.Module | None = None,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
    ):
        """Drop-in replacement for Flax's dot_product_attention, but with RoPE applied to the query and key."""
        
        query = self(query)
        key = self(key)
        
        return dot_product_attention(
            query, 
            key, 
            value, 
            bias, 
            mask, 
            broadcast_dropout, 
            dropout_rng, 
            dropout_rate, 
            deterministic, 
            dtype, 
            precision, 
            module, 
            promote_dtype
        )
