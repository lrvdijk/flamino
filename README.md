<h1 align="center">🦩 Flamino</h1>
<p align="center"><strong>A Flax NNX-based reimplementation of ESM-2</strong></p>

**⚠️ Work in progress**

## Currently implemented

- [x] Tokenization
- [x] Rotary Position Encodings (RoPE)
- [x] RoPE-based multihead attention
- [x] Transformer layers
- [x] ESM-2 model
- [ ] Ability to load ESM-2 pretrained weights
- [ ] Training loop
- [ ] Perplexity evaluation
- [ ] Residue-residue contact prediction head
- [ ] Residue-residue contact prediction validation


## Installation

### From source 

This project uses [`uv`](https://docs.astral.sh/uv/) to manage dependencies. Clone the repository
and use `uv` within the repository directory:

```bash
git clone https://github.com/lrvdijk/flamino
cd flamino
uv run python  # Or any other command
```

### Running the Jupyter notebooks locally

To run the notebooks locally, it is recommended to install a project-specific IPython kernel and keep Jupyter installed
in a separate uv environment, as per the [uv documentation](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-within-a-project):

```bash
uv sync
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=flamino
uv run --with jupyter jupyter lab
```

## Usage

### Dataset management

Flamino expects data to be stored in Parquet files. We provide a CLI tool to convert one or more
FASTA/FASTQ files to (chunked) Parquet files:

```bash
flamino data convert -c 1000000 -o output_dataset f1.fasta f2.fasta f3.fasta
```

The `-c` enables dataset chunking, storing at most the specified number of records per file. The output dataset thus
comprises multiple Parquet files, prefixed by the given output prefix with `-o`:

```
output_dataset.1.parquet
output_dataset.2.parquet
output_dataset.3.parquet
...
```

### Training a model

To train a model from scratch, Flamino provides a CLI tool to train any of the included models:

```bash
flamino train -c train_conf.toml -m model_conf.toml data_conf.toml 
```

The training configuration file configures the optimizer, the number of epochs, batch size, and more. The model
configuration file configures the model to train, for example, number of layers, embedding dimensionality, etc. The
data configuration file species the list of files to read and the vocabulary for tokenizing sequences. Each
configuration file is further explained below.


## Training and model configuration

TODO



