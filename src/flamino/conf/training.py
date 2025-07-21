from typing import Literal, Any
from pydantic import BaseModel


class TrainingConf(BaseModel):
    """
    Model training configuration

    Attributes
    ----------
    seq_max_len : int
        Maximum length of input sequences to be passed to the model.
    seq_packing_mode: str
        How to deal with variable length strings. Set to "pad"
        to pad sequences shorter than `seq_max_len` with additional
        padding tokens. An additional mask array will indicate padding
        tokens. Set to "pack" to concatenate sequences and split them
        into exactly `seq_max_len` sequences. An additional mask array
        will indicate which positions belong to separate sequences.
    epochs : int
        Number of training epochs
    optimizer : str
        Which optimizer to use. Should be one available in `optax.optimizers`.
    optimizer_conf : dict[str, Any]
        Any optimizer configuration. Should correspond to the keyword arguments of
        the respective optimizer in `optax.optimizer`.
    
    """

    seq_max_len: int
    seq_packing_mode: Literal["pad"] | Literal["pack"]

    epochs: int
    optimizer: str
    optimizer_conf: dict[str, Any] = {}
    random_seed: int | None = None
