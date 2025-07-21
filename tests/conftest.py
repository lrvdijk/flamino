import jax
import pytest

from flamino import vocab


@pytest.fixture
def esm2_alphabet() -> vocab.Vocabulary:
    return vocab.ESM2


@pytest.fixture
def mock_embeddings(esm2_alphabet: vocab.Vocabulary) -> dict[str, jax.Array]:
    d_embed = 64
    key = jax.random.key(42)

    mock_embeddings: dict[str, jax.Array] = {}
    for token in esm2_alphabet.tokens:
        new_key, subkey = jax.random.split(key)
        del key
        
        mock_embeddings[token] = jax.random.normal(subkey, (d_embed,))
        key = new_key
        
    return mock_embeddings
    
