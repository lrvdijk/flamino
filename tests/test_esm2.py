import jax.numpy as jnp
from flax import nnx

from flamino.models.esm2 import ESM2
from flamino.vocab import Alphabet

def test_esm2(esm2_alphabet: Alphabet):
    rngs = nnx.Rngs(0)
    model = ESM2(esm2_alphabet, 8, 64, 8, rngs=rngs)
    
    batched_model = nnx.vmap(model)
    
    seq = jnp.array(esm2_alphabet.tokenize_to_arr("AAFGG"))
    embeddings = batched_model(seq)
    
