import jax
import jax.numpy as jnp
from flax import nnx

from flamino.transformer import TransformerEncoder


def test_transformer_init(mock_embeddings: dict[str, jax.Array]):
    rngs = nnx.Rngs(42)
    seq = jnp.array([mock_embeddings[token] for token in "AC"])
    d_embed = seq.shape[-1]

    transformer = TransformerEncoder(
        d_embed,
        d_embed * 4,
        4,
        rngs=rngs,
    )

    transformed = transformer(seq)

    assert transformed.shape == (len(seq), d_embed)
    assert not jnp.all(transformed == seq)
