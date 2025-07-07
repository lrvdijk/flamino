"""
Transformer implementation with rotary position encoding (RoPE).
"""

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax
from flax import nnx
from flax.nnx.module import first_from

from flamino.rope import RoPE


class MultiHeadAttention(nnx.MultiHeadAttention):
    """
    Modified version of nnx.MultiHeadAttention to allow post processing of queries/keys/values.
    
    For example, you can use this class to apply RoPE-based position embeddings to the queries and keys.
    """
    
    def __call__(
        self,
        inputs_q: jax.Array,
        inputs_k: jax.Array | None = None,
        inputs_v: jax.Array | None = None,
        *,
        mask: jax.Array | None = None,
        deterministic: bool | None = None,
        rngs: nnx.rnglib.Rngs | None = None,
        sow_weights: bool = False,
        decode: bool | None = None,
        process_queries: Callable[[jax.Array], jax.Array] | None = None,
        process_keys: Callable[[jax.Array], jax.Array] | None = None,
        process_values: Callable[[jax.Array], jax.Array] | None = None,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
            inputs_q: input queries of shape `[batch_sizes..., length, features]`.
            inputs_k: key of shape `[batch_sizes..., length, features]`. If None,
                inputs_k will copy the value of inputs_q.
            inputs_v: values of shape `[batch_sizes..., length, features]`. If None,
                inputs_v will copy the value of inputs_k.
            mask: attention mask of shape `[batch_sizes..., num_heads, query_length,
                key/value_length]`. Attention weights are masked out if their
                corresponding mask value is `False`.
            deterministic: if false, the attention weight is masked randomly using
                dropout, whereas if true, the attention weights are deterministic. The
                ``deterministic`` flag passed into the call method will take precedence
                over the ``deterministic`` flag passed into the constructor.
            rngs: rng key. The rng key passed into the call method will take
                precedence over the rng key passed into the constructor.
            sow_weights: if ``True``, the attention weights are sowed into the
                'intermediates' collection.
            decode: whether to prepare and use an autoregressive cache. The ``decode``
                flag passed into the call method will take precedence over the ``decode``
                flag passed into the constructor.
            process_queries: Function to post-process the query vector after projection.
            process_keys: Function to post-process the key vector after projection.
            process_values: Function to post-process the value vector after projection.

        Returns:
            output of shape `[batch_sizes..., length, features]`.
        """
        if rngs is None:
            rngs = self.rngs

        if inputs_k is None:
            if inputs_v is not None:
                raise ValueError(
                    '`inputs_k` cannot be None if `inputs_v` is not None. '
                    'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
                    'value to `inputs_k` and leave `inputs_v` as None.'
                )
            inputs_k = inputs_q
        if inputs_v is None:
            inputs_v = inputs_k

        if inputs_q.shape[-1] != self.in_features:
            raise ValueError(
                f'Incompatible input dimension, got {inputs_q.shape[-1]} '
                f'but module expects {self.in_features}.'
            )

        query = self.query(inputs_q)
        key = self.key(inputs_k)
        value = self.value(inputs_v)

        if self.normalize_qk:
            assert self.query_ln is not None and self.key_ln is not None
            # Normalizing query and key projections stabilizes training with higher
            # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
            query = self.query_ln(query)
            key = self.key_ln(key)
            
        if process_queries:
            query = process_queries(query)
        if process_keys:
            key = process_keys(key)
        if process_values:
            value = process_values(value)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        decode = first_from(
            decode,
            self.decode,
            error_msg="""No `decode` argument was provided to MultiHeadAttention
                as either a __call__ argument, class attribute, or nnx.flag.""",
        )

        if decode:
            if (
                self.cached_key is None
                or self.cached_value is None
                or self.cache_index is None
            ):
                raise ValueError(
                    'Autoregressive cache not initialized, call ``init_cache`` first.'
                )
            (
                *batch_dims,
                max_length,
                num_heads,
                depth_per_head,
            ) = self.cached_key.value.shape
            # shape check of cached keys against query input
            expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
            if expected_shape != query.shape:
                raise ValueError(
                    'Autoregressive cache shape error, '
                    'expected query shape %s instead got %s.'
                    % (expected_shape, query.shape)
                )
            # update key, value caches with our new 1d spatial slices
            cur_index = self.cache_index.value
            zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
            indices = (zero,) * len(batch_dims) + (cur_index, zero, zero)
            key = lax.dynamic_update_slice(self.cached_key.value, key, indices)
            value = lax.dynamic_update_slice(self.cached_value.value, value, indices)
            self.cached_key.value = key
            self.cached_value.value = value
            self.cache_index.value += 1
            # causal mask for cached decoder self-attention:
            # our single query position should only attend to those key
            # positions that have already been generated and cached,
            # not the remaining zero elements.
            mask = nnx.combine_masks(
                mask,
                jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, 1, max_length),
                ),
            )

        if (
            self.dropout_rate > 0.0
        ):    # Require `deterministic` only if using dropout.
            deterministic = first_from(
                deterministic,
                self.deterministic,
                error_msg="""No `deterministic` argument was provided to MultiHeadAttention
                    as either a __call__ argument, class attribute, or nnx.flag.""",
            )
            if not deterministic:
                if rngs is None:
                    raise ValueError(
                        "'rngs' must be provided if 'dropout_rng' is not given."
                    )
                dropout_rng = rngs.dropout()
            else:
                dropout_rng = None
        else:
            deterministic = True
            dropout_rng = None

        # apply attention
        x = self.attention_fn(
            query,
            key,
            value,
            mask=mask,
            dropout_rng=dropout_rng,
            dropout_rate=self.dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=self.precision,
            module=self if sow_weights else None,
        )
        # back to the original inputs dimensions
        out = self.out(x)
        return out


class TransformerEncoder(nnx.Module):
    """
    A single transformer encoder block with rotary position encoding (RoPE).
    """

    def __init__(
        self,
        d_embed: int,
        hidden_size: int,
        num_heads: int,
        rope: RoPE | None = None,
        *,
        rngs: nnx.Rngs
    ) -> None:
        assert d_embed % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.layer_norm1: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)
        self.layer_norm2: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)
        self.rope: RoPE | None = rope

        self.attention: MultiHeadAttention = MultiHeadAttention(
            num_heads=num_heads,
            in_features=d_embed,
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
        
        @nnx.vmap(in_axes=(None, 0), out_axes=0)
        def apply_rope(rope: RoPE, x: jax.Array):
            return rope(x)

        y = self.attention(
            y, 
            y, 
            y, 
            mask=mask, 
            sow_weights=True,
            process_queries=partial(apply_rope, self.rope) if self.rope is not None else None,
            process_keys=partial(apply_rope, self.rope) if self.rope is not None else None
        )

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
