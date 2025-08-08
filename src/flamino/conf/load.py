"""
Utilities to instantiate optimizers and schedules for a given configuration
"""

import inspect
from typing import Any

from flax import nnx
import optax

from flamino import models
from flamino.conf.model import Model
from flamino.conf.training import Optimizer, Schedule


all_models = {m: model for m in models.__all__ if issubclass((model := getattr(models, m)), nnx.Module)}


def instantiate_model(conf: Model, **kwargs: Any) -> nnx.Module:
    if conf.name not in all_models:
        raise ValueError(f"Model '{conf.name}' not found.")

    model_cls = all_models[conf.name]
    signature = inspect.signature(model_cls.__init__)

    print(signature)

    extra_kwargs = {k: v for k, v in kwargs.items() if k in signature.parameters}

    model_kwargs = dict(conf.__pydantic_extra__)
    model_kwargs.update(extra_kwargs)
    print(model_kwargs)
    return model_cls(**model_kwargs)


def instantiate_schedule(conf: Schedule) -> optax.Schedule:
    optax_func = getattr(optax, conf.schedule)

    return optax_func(**conf.__pydantic_extra__)


def instantiate_optimizer(conf: Optimizer):
    optax_cls = getattr(optax, conf.name)

    if not issubclass(optax_cls, optax.GradientTransformation):
        raise ValueError(f"Optimizer '{conf.name}' is not a valid Optax optimizer.")

    kwargs = {k: instantiate_schedule(v) if isinstance(v, Schedule) else v for k, v in conf.__pydantic_extra__.items()}

    return optax_cls(**kwargs)
