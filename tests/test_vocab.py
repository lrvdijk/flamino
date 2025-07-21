from flamino import vocab


def test_tokenize(esm2_alphabet: vocab.Vocabulary):
    token_strings = list(map(esm2_alphabet.tok_to_str, esm2_alphabet.tokenize("ACDEFGHIKLMNPQRSTVWY")))
    assert token_strings == [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
    ]
    
    token_strings = list(map(esm2_alphabet.tok_to_str, esm2_alphabet.tokenize("<start>ACDF<mask>DGDG<end>")))
    assert token_strings == [
        vocab.START_OF_SEQ, "A", "C", "D", "F", vocab.MASK, "D", "G", "D", "G", vocab.END_OF_SEQ
    ]
    
    token_strings = list(map(esm2_alphabet.tok_to_str, esm2_alphabet.tokenize("J")))
    assert token_strings == [vocab.UNKNOWN]
    
    assert esm2_alphabet.tok_to_str(esm2_alphabet.mask) == vocab.MASK
    
    
def test_tokenize_to_arr(esm2_alphabet: vocab.Vocabulary):
    texts = [
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDF<mask>DGDG",
    ]
    
    token_arr = esm2_alphabet.tokenize_to_arr(texts)
    
    assert token_arr.shape == (2, 22)
    
    for i in range(len(token_arr)):
        assert "".join(
            esm2_alphabet.tok_to_str(tok) for tok in token_arr[i] if tok != esm2_alphabet.pad
        ) == f"{vocab.START_OF_SEQ}{texts[i]}{vocab.END_OF_SEQ}"
