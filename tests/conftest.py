import jax
import pytest

from flamino.vocab import Alphabet


@pytest.fixture
def amino_acid_alphabet() -> Alphabet:
    return Alphabet.amino_acids()


@pytest.fixture
def mock_embeddings(amino_acid_alphabet: Alphabet) -> dict[str, jax.Array]:
    d_embed = 64
    key = jax.random.key(42)

    mock_embeddings: dict[str, jax.Array] = {}
    for token in amino_acid_alphabet.tokens:
        new_key, subkey = jax.random.split(key)
        del key
        
        mock_embeddings[token] = jax.random.normal(subkey, (d_embed,))
        key = new_key
        
    return mock_embeddings
    