"""
Transformer implementation with rotary position encoding (RoPE).
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

from flamino.rope import RoPE


class rope_attention_func:
    """
    Encapsulates a `RoPE` module and applies RoPE to the given query, key, and value arrays
    when called, before passing them to Flax' builtin `dot_product_attention` function.
    """
    
    def __init__(self, rope_module: RoPE):
        self.rope_module: RoPE = rope_module

    def __call__(
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
        query = self.rope_module(query)
        key = self.rope_module(key)
        
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


class TransformerEncoder(nnx.Module):
    """
    A single transformer encoder block with rotary position encoding (RoPE).
    """
    
    def __init__(self, d_embed: int, hidden_size: int, num_heads: int, *, rngs: nnx.Rngs) -> None:
        assert d_embed % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.layer_norm1: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)
        self.layer_norm2: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)
        
        self.rope: RoPE = RoPE(d_embed // num_heads, rngs=rngs)
        self.rope_attention_func: rope_attention_func = rope_attention_func(self.rope)
        
        self.attention: nnx.MultiHeadAttention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_embed,
            attention_fn=self.rope_attention_func,
            decode=False,
            rngs=rngs
        )
        
        self.linear1: nnx.Linear = nnx.Linear(d_embed, hidden_size, rngs=rngs)
        self.linear2: nnx.Linear = nnx.Linear(hidden_size, d_embed, rngs=rngs)
        
    def __call__(self, x: jax.Array, pad_mask: jax.Array | None = None):
        seq_len = x.shape[0]
        
        mask = None
        if pad_mask:
            pad_mask_len = pad_mask.shape[0]
            assert pad_mask_len == seq_len, "Padding mask length must match sequence length"
            
            mask = jnp.broadcast_to(jnp.logical_not(pad_mask), (seq_len, seq_len))
            
        y = self.layer_norm1(x)
        y = self.attention(y, y, y, mask=mask, sow_weights=True)
        
        # Residual connection 1
        x = x + y
        
        # Feed forward
        y = self.layer_norm2(x)
        y = self.linear1(y)
        y = jax.nn.gelu(y, approximate=False)
        y = self.linear2(y)
        
        # Residual connection 2
        x = x + y
        
        return x