"""
Transformer implementation with rotary position encoding (RoPE).
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from flamino.rope import RoPE, rope_dot_product_attention


class TransformerEncoder(nnx.Module):
    """
    A single transformer encoder block with rotary position encoding (RoPE).
    """
    
    def __init__(self, d_embed: int, hidden_size: int, num_heads: int, rope: RoPE | None = None, *, rngs: nnx.Rngs) -> None:
        assert d_embed % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.layer_norm1: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)
        self.layer_norm2: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)
        
        if rope:
            attention_fn = partial(rope_dot_product_attention, rope)
        else:
            attention_fn = nnx.dot_product_attention
        
        self.attention: nnx.MultiHeadAttention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_embed,
            attention_fn=attention_fn,
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