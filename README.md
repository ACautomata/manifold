# manifold

Pretraining and medical-imaging experiments built on top of
[stable-pretraining](https://github.com/galilai-group/stable-pretraining) and
[MONAI](https://monai.io/).

## Setup

This project uses a `src/` layout and is managed with `pyproject.toml`.
Dependencies are installed into a conda environment named after the project.

```bash
# Create the conda environment
conda create -n manifold python=3.12 -y
conda activate manifold

# Install the package (editable) with all dependencies
pip install -e .
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Layout

```
manifold/
├── pyproject.toml
├── README.md
└── src/
    └── manifold/
        └── __init__.py
```
# manifold
