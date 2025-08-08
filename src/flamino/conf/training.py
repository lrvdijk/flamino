from collections.abc import Mapping, Sequence
from typing import Literal, Any
from pydantic import BaseModel, ConfigDict, Field


class Schedule(BaseModel):
    """
    Represents a schedule for a given hyper parameter

    Attributes
    ----------
    schedule : str
        Name of the schedule kind. Should match one of Optax's built-in schedules.
    **kwargs : Any
        Any other key-value pairs will be passed to the constructor of schedule.
    """

    model_config: ConfigDict = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, int | float | str | Sequence[Any] | Mapping[Any, Any]] = Field(init=False)  # type: ignore

    schedule: str


class Optimizer(BaseModel):
    """
    Optimizer configuration

    Attributes
    ----------
    name : str
        Name of the optimizer to use. Should match one of Optax's built-in optimizers.
    **kwargs : int | float | Schedule
        Any other configuration keys will be passed as parameters to the optimizer constructor. Hyper-parameters
        can be configured to use a schedule during training by passing a `Schedule`

    Notes
    -----
    A list of optimizers can be found in the Optax documentation:
    https://optax.readthedocs.io/en/latest/api/optimizers.html#

    See Also
    --------
    Schedule
    """

    model_config: ConfigDict = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, int | float | Schedule] = Field(init=False)  # type: ignore

    name: str


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
    batch_size : int
        The training batch size.
    epochs : int
        Number of training epochs
    random_seed : int | None
        Optionally specify a specific random seed.
    optimizer : Optimizer
        Which optimizer to use. Should be one available in `optax.optimizers`.
    """

    seq_max_len: int
    seq_packing_mode: Literal["pad"] | Literal["pack"]

    batch_size: int
    epochs: int
    random_seed: int | None = None

    optimizer: Optimizer
