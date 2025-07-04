import jax
from flax import nnx

from flamino.transformer import TransformerEncoder
from flamino.vocab import Alphabet


class LogitHead(nnx.Module):
    """
    Returns log probabilities for each token in the alphabet given the input embedding.
    """

    def __init__(self, d_embed: int, alphabet_size: int, *, rngs: nnx.Rngs):
        self.linear1: nnx.Linear = nnx.Linear(d_embed, d_embed, rngs=rngs)
        self.linear2: nnx.Linear = nnx.Linear(d_embed, alphabet_size, rngs=rngs)
        self.layer_norm: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = self.linear1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.layer_norm(x)
        x = self.linear2(x)

        return x


class ESM2(nnx.Module):
    def __init__(
        self,
        alphabet: Alphabet,
        num_layers: int = 33,
        d_embed: int = 1024,
        num_heads: int = 16,
        *,
        rngs: nnx.Rngs
    ):
        # Token -> initial embedding
        self.embed: nnx.Embed = nnx.Embed(len(alphabet.tokens), d_embed, rngs=rngs)

        # BERT-style transformer layers with rotary positional encoding embeddings
        # Split RNG generator #layers, and use vmap to generate each layer
        @nnx.split_rngs(splits=num_layers)
        @nnx.vmap(axis_size=num_layers)
        def create_layers(rngs: nnx.Rngs):
            return TransformerEncoder(d_embed, d_embed * 4, num_heads, rngs=rngs)

        self.transformer_layers = create_layers(rngs)
        self.layer_norm_after: nnx.LayerNorm = nnx.LayerNorm(d_embed, rngs=rngs)

        self.logit_head: LogitHead = LogitHead(d_embed, len(alphabet.tokens), rngs=rngs)

    def __call__(self, tokens: jax.Array):
        # Token -> initial embedding
        x = self.embed(tokens)

        # Use nnx.scan to repeatedly apply each transformer layer
        @nnx.scan
        def apply_transformer_layer(x: jax.Array, layer: TransformerEncoder):
            return layer(x), None

        x, _ = apply_transformer_layer(x, self.transformer_layers)

        x = self.layer_norm_after(x)
        logits = self.logit_head(x)

        return logits
