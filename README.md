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
