import jax
import jax.numpy as jnp
from flax import nnx

from flamino.rope import RoPE


def test_relative_encoding(mock_embeddings: dict[str, jax.Array]):
    seqs = ["ACAC", "FFGG"]
    seqs = jnp.concatenate([
        jnp.array([mock_embeddings[token] for token in seq])[None, :]  # Add a new axis for batch dimension
        for seq in seqs
    ])
    d_embed = seqs.shape[-1]
    
    rope = RoPE(d_embed)
    batched_rope = nnx.vmap(rope)
    rope_encoding = batched_rope(seqs)
    
    assert rope_encoding.shape == (2, seqs.shape[1], d_embed)
    
    ac1 = rope_encoding[0, :2, :]
    ac2 = rope_encoding[0, 2:, :]
    
    # We expect the embeddings to be different, now with position encoded
    assert not jnp.all(ac1[1] == seqs[0, 1, :])
    
    # Embedding of the same token but at a different position should be different
    assert not jnp.all(ac1[0] == ac2[0])
    
    # Inner product between two identical tokens should be the same regardless of 
    # their absolute position
    inner1 = jnp.inner(ac1[0], ac1[1])
    inner2 = jnp.inner(ac2[0], ac2[1])
    
    assert jnp.allclose(inner1, inner2)
    
    f1 = rope_encoding[1, 0, :]
    f2 = rope_encoding[1, 1, :]
    g1 = rope_encoding[1, 2, :]
    g2 = rope_encoding[1, 3, :]
    
    inner1 = jnp.inner(f1, g1)
    inner2 = jnp.inner(f2, g2)
    
    assert jnp.allclose(inner1, inner2)