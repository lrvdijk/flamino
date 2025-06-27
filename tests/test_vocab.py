from jxesm2.vocab import Alphabet, MASK, UNKNOWN


def test_tokenize():
    vocab = Alphabet.amino_acids()
    
    token_strings = list(map(vocab.tok_to_str, vocab.tokenize("ACDEFGHIKLMNPQRSTVWY")))
    assert token_strings == [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"
    ]
    
    token_strings = list(map(vocab.tok_to_str, vocab.tokenize("ACDF<mask>DGDG")))
    assert token_strings == [
        "A", "C", "D", "F", MASK, "D", "G", "D", "G"
    ]
    
    token_strings = list(map(vocab.tok_to_str, vocab.tokenize("Z")))
    assert token_strings == [UNKNOWN]