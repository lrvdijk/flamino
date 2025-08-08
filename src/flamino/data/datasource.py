"""
A Jax Grain Data Source implementation to read records
from a collection of FASTA files, which need to be bgzipped
and indexed with `samtools faidx`.
"""

import os
from pathlib import Path
from collections.abc import Sequence
from typing import SupportsIndex

from typing_extensions import override
import numpy as np

import pysam
import grain


class FastaDataSource(grain.sources.RandomAccessDataSource[str]):
    """
    Reads records from one or more FASTA files.

    To enable efficient random access, FASTA files need to be
    bgzipped and indexed with `samtools faidx`.
    """

    def __init__(self, fasta_files: Sequence[str | Path]) -> None:
        self.fasta_files: list[str | Path] = []
        self.open_files: list[pysam.FastaFile | None] = []
        self.num_seq_per_file: list[int] = []

        for fname in fasta_files:
            fname = os.path.expanduser(fname)

            with pysam.FastaFile(fname) as f:
                if f.nreferences:
                    self.fasta_files.append(fname)
                    self.open_files.append(None)
                    self.num_seq_per_file.append(f.nreferences)

        self.cum_num_seq: np.ndarray = np.cumsum(np.array(self.num_seq_per_file))
        self.num_sequences: int = self.cum_num_seq[-1]

        print(self.num_seq_per_file)
        print(self.cum_num_seq)

    @override
    def __len__(self) -> int:
        return self.num_sequences

    def __enter__(self) -> "FastaDataSource":
        for ix, fname in enumerate(self.fasta_files):
            self.open_files[ix] = pysam.FastaFile(fname)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for fasta in self.open_files:
            if fasta is not None and not fasta.closed:
                fasta.close()

    def seq_ix_to_file(self, ix: int) -> tuple[int, int]:
        file_ix = np.searchsorted(self.cum_num_seq, ix, side="right").astype(int)

        file_start_ix = 0 if file_ix == 0 else int(self.cum_num_seq[file_ix - 1])
        return file_ix, ix - file_start_ix

    @override
    def __getitem__(self, ix: SupportsIndex) -> str:
        ix = ix.__index__()
        if not 0 <= ix < self.num_sequences:
            raise IndexError(f"Index {ix} out of range")

        file_ix, within_file_ix = self.seq_ix_to_file(ix)
        if self.open_files[file_ix] is None:
            self.open_files[file_ix] = pysam.FastaFile(str(self.fasta_files[file_ix]))

        assert within_file_ix < self.open_files[file_ix].nreferences
        seq_name = self.open_files[file_ix].references[within_file_ix]
        print(
            "got ix",
            ix,
            "file",
            file_ix,
            "within file",
            within_file_ix,
            self.open_files[file_ix].references[within_file_ix],
        )

        return self.open_files[file_ix].fetch(seq_name)
