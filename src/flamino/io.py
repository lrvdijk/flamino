"""
Utilities for file I/O
"""

import bz2
import gzip
import lzma
import os
from pathlib import Path
from typing import BinaryIO


def open_compressed(fname: str | Path, *args, **kwargs):
    """
    Open a compressed file, inferring the compression method from the file extension.
    """

    _, ext = os.path.splitext(fname)

    funcs = {".gz": gzip.open, ".bz2": bz2.open, ".lz": lzma.open}
    open_func = funcs.get(ext, open)

    return open_func(fname, *args, **kwargs)


def readfq(fp: BinaryIO):  # this is a generator function
    """
    Heng Li's fast FASTA/FASTQ reader.

    Adjusted to use the `bytes` arrays instead of unicode strings.
    """

    last = None  # This is a buffer keeping the last unprocessed line
    while True:  # Mimic closure; is it a bad idea?
        if not last:  # The first record or a record following a fastq
            for line in fp:  # Search for the start of the next record
                if line[0] in b">@":  # fasta/q header line
                    last = line[:-1]  # save this line
                    break
        if not last:
            break
        name, seqs, last = last[1:].partition(b" ")[0], [], None
        for line in fp:  # read the sequence
            if line[0] in b"@+>":
                last = line[:-1]
                break
            seqs.append(line[:-1])
        if not last or last[0] != b"+"[0]:  # this is a fasta record
            yield name, b"".join(seqs), None  # yield a fasta record
            if not last:
                break
        else:  # this is a fastq record
            seq, leng, seqs = b"".join(seqs), 0, []
            for line in fp:  # read the quality
                seqs.append(line[:-1])
                leng += len(line) - 1
                if leng >= len(seq):  # have read enough quality
                    last = None
                    yield name, seq, b"".join(seqs)  # yield a fastq record
                    break
            if last:  # Reach EOF before reading enough quality
                yield name, seq, None  # yield a fasta record instead
                break
