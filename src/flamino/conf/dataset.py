from pydantic import BaseModel


class Dataset(BaseModel):
    """
    Dataset loading configuration
    
    Attributes
    ----------
    files : list[str]
        List of file paths to load.
    vocab : str
        Name of the token vocabulary.
    max_len : int
        Maximum length of the sequences.
    mode : Literal["pad"] | Literal["pack"]
        How to deal with sequences shorter or longer than max_len.
        Setting mode to "pad" will pad sequences shorter than max_len with a special token. The
        mask array will indicate with the value 1 which sequence indices are padding tokens.
        Setting mode to "segment" will concatenate sequences to fill up to max_len. The mask
        array will indicate which tokens belong to which sequence with unique sequence indices.
    """
    
    files: list[str]
    vocab: str
