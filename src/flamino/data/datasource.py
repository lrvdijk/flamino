"""
A Jax Grain Data Source implementation to read records
from a collection of FASTA files, which need to be bgzipped 
and indexed with `samtools faidx`.
"""

from pathlib import Path
from collections.abc import Sequence
from typing import SupportsIndex

from typing_extensions import override
from intervaltree import IntervalTree

import pysam
import grain


class FastaDataSource(grain.sources.RandomAccessDataSource[str]):
    """
    Reads records from one or more FASTA files.
    
    To enable efficient random access, FASTA files need to be 
    bgzipped and indexed with `samtools faidx`.
    """
    
    def __init__(self, fasta_files: Sequence[str | Path]) -> None:
        self.fasta_files: list[str] = [
            str(f) for f in fasta_files
        ]
        
        self.open_files: list[pysam.FastaFile | None] = [None] * len(self.fasta_files)
        
        self.num_seq_per_file: list[int] = []
        for fname in self.fasta_files:
            with pysam.FastaFile(fname) as f:
                self.num_seq_per_file.append(f.nreferences or 0)
        
        prev = 0
        self.seq_ix_to_file: IntervalTree = IntervalTree.from_tuples(
            (prev, (prev := prev + num_seqs), file_ix) for file_ix, num_seqs in enumerate(self.num_seq_per_file)
        )
        
        self.num_sequences: int = sum(self.num_seq_per_file)
        
    @override
    def __len__(self) -> int:
        return self.num_sequences
        
    def __enter__(self) -> 'FastaDataSource':
        for ix, fname in enumerate(self.fasta_files):
            self.open_files[ix] = pysam.FastaFile(fname)
            
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for fasta in self.open_files:
            if fasta is not None and not fasta.closed:
                fasta.close()
    
    @override
    def __getitem__(self, ix: SupportsIndex) -> str:
        ix = ix.__index__()
        if not 0 <= ix < self.num_sequences:
            raise IndexError(f"Index {ix} out of range")
        
        ival_start, _, file_ix = next(iter(self.seq_ix_to_file[ix]))
        
        if not self.open_files[file_ix]:
            self.open_files[file_ix] = pysam.FastaFile(self.fasta_files[file_ix])
            
        within_file_ix = ix - ival_start
        seq_name = self.open_files[file_ix].references[within_file_ix]
        
        return self.open_files[file_ix].fetch(seq_name)
        
        
