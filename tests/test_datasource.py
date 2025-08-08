from flamino.data.datasource import FastaDataSource


def test_fasta_data_source():
    fasta_files = ["tests/test_data/test_fasta1.fa.gz", "tests/test_data/test_fasta2.fa.gz"]

    fasta_source = FastaDataSource(fasta_files)
    assert len(fasta_source) == 5
    assert fasta_source.num_seq_per_file == [3, 2]

    assert fasta_source[0] == "AAAGGTT"
    assert fasta_source[1] == "CCCGGTT"
    assert fasta_source[4] == "CCCCCCCCC"
