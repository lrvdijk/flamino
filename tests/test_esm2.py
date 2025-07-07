import jax
import jax.numpy as jnp
from flax import nnx

from flamino.models.esm2 import ESM2
from flamino.vocab import Alphabet


def test_esm2(esm2_alphabet: Alphabet):
    rngs = nnx.Rngs(0)
    model = ESM2(esm2_alphabet, 8, 128, 4, rngs=rngs)
    
    
    # NNX's MultiHeadAttention stores the attention weights as an `nnx.Intermediate` variable,
    # which we want to keep per input sequence. Use nnx.StateAxes to batch those along axis 0.
    state_axes = nnx.StateAxes({nnx.Intermediate: 0, ...: None})
    @nnx.vmap(in_axes=(state_axes, 0), out_axes=0)
    def forward(model: ESM2, seq: jax.Array):
        return model(seq)
    
    
    seq = jnp.array(esm2_alphabet.tokenize_to_arr("AAFGG"))  # (batch, sequence)
    logits = forward(model, seq)
    
    print(logits)
    print(logits.shape)
    
    assert logits.shape == (1, 7, 32)  # (1 seq, seq len + 2, alphabet size)
    
