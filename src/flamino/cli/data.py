import click
import pyarrow as pa
import pyarrow.parquet as pq

from flamino.io import readfq, open_compressed


@click.group()
def data():
    """Utilities for managing datasets"""


@data.command()
@click.argument("infile", nargs=-1)
@click.option(
    "-o",
    "--output-prefix",
    required=True,
    help="The output filename prefix. The extension '.parquet' will be automatically added.",
)
@click.option(
    "-c",
    "--chunk-size",
    type=int,
    default=None,
    help="Specify a chunk size to create multiple, smaller datasets, each containing up to CHUNKSIZE records.",
)
def convert(infile: tuple[str, ...], output_prefix: str, chunk_size: int | None = None):
    """
    Convert the FASTA/FASTQ files specified by INFILE to a Parquet files, suitable for
    training Flamino models.
    """

    curr_chunk_ids = []
    curr_chunk_seq = []
    curr_chunk = 0
    for file in infile:
        with open_compressed(file, "rb") as fp:
            for seq_name, seq, _ in readfq(fp):
                curr_chunk_ids.append(seq_name)
                curr_chunk_seq.append(seq)

                if chunk_size and len(curr_chunk_ids) >= chunk_size:
                    fname = f"{output_prefix}.{curr_chunk}.parquet"
                    write_chunk(fname, curr_chunk_ids, curr_chunk_seq)

                    curr_chunk_ids.clear()
                    curr_chunk_seq.clear()
                    curr_chunk += 1

    if curr_chunk_ids:
        if chunk_size:
            fname = f"{output_prefix}.{curr_chunk}.parquet"
        else:
            fname = f"{output_prefix}.parquet"

        write_chunk(fname, curr_chunk_ids, curr_chunk_seq)


def write_chunk(fname: str, seq_ids: list[bytes], seqs: list[bytes]):
    table = pa.table(
        [pa.array(seq_ids, type=pa.binary(length=-1)), pa.array(seqs, type=pa.binary(length=-1))],
        names=["seq_id", "seq"],
    )

    pq.write_table(table, fname)
