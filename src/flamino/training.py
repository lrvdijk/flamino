"""
The Flamino training loop.
"""

import random

from flax import nnx
import grain
import pyarrow.parquet as pq
import numpy as np

from flamino import conf
from flamino.data.transforms import PadTransform
from flamino.vocab import get_vocab_from_str
from flamino.data import TokenizeTransform, MaskTransform


class FlaminoTrainer:
    """
    Train any Flamino model
    """

    def __init__(self, model: nnx.Module):
        self.model: nnx.Module = model

    def train(self, dataset_conf: conf.Dataset, training_conf: conf.TrainingConf):
        seed = training_conf.random_seed if training_conf.random_seed else random.randint(0, 2**32 - 1)
        vocab = get_vocab_from_str(dataset_conf.vocab)

        weights = np.array([pq.ParquetFile(f).metadata.num_rows for f in dataset_conf.files], dtype=np.float32)
        weights /= np.sum(weights)

        datasets = [
            (
                grain.experimental.ParquetIterDataset(fname)
                .seed(seed + i + 1)
                .map(lambda e: e["seq"].decode("utf-8"))
                .map(TokenizeTransform(vocab))
                .random_map(MaskTransform(vocab))
                .pipe(grain.experimental.WindowShuffleIterDataset, window_size=512, seed=seed + i + 1)
            )
            for i, fname in enumerate(dataset_conf.files)
        ]

        dataset = grain.IterDataset.mix(datasets, weights)

        if training_conf.seq_packing_mode == "pad":
            dataset = dataset.map(PadTransform(vocab, training_conf.seq_max_len))

        else:
            dataset = dataset.pipe(
                grain.experimental.ConcatThenSplitIterDataset,
                length_struct={"orig": training_conf.seq_max_len, "masked": training_conf.seq_max_len},
            )

        # dataset = dataset.batch(training_conf.batch_size)

        for batch in dataset:
            print(batch)
