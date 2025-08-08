"""
Command line interface for training Flamino models
"""

import random
import tomllib
from typing import BinaryIO

import click
from flax import nnx

from flamino.conf import Dataset, TrainingConf, load
from flamino.conf.model import Model
from flamino.training import FlaminoTrainer
from flamino.vocab import get_vocab_from_str


@click.command()
@click.argument("dataset", type=click.File("rb"))
@click.option("-c", "--config", type=click.File("rb"), help="Training configuration JSON file.")
@click.option("-m", "--model", type=click.File("rb"), help="Model specification.")
def train(dataset: BinaryIO, config: BinaryIO, model: BinaryIO):
    """
    Train a model using a data specified in DATASET.

    DATASET is a JSON file configuring which FASTA files to read and what token vocabulary to use.
    """

    dataset_conf = Dataset.model_validate(tomllib.load(dataset))
    training_conf = TrainingConf.model_validate(tomllib.load(config))
    model_conf = Model.model_validate(tomllib.load(model))

    vocab = get_vocab_from_str(dataset_conf.vocab)
    rngs = nnx.Rngs(training_conf.random_seed if training_conf.random_seed else random.randint(0, 2**32 - 1))
    model_obj = load.instantiate_model(model_conf, vocabulary=vocab, rngs=rngs)

    trainer = FlaminoTrainer(model_obj)

    trainer.train(dataset_conf, training_conf)
