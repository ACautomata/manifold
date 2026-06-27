# manifold

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

Pretraining and medical-imaging experiments built on top of
[stable-pretraining](https://github.com/galilai-group/stable-pretraining) and
[MONAI](https://monai.io/). Follows the
[diffusers](https://github.com/huggingface/diffusers) architecture: pipeline,
scheduler, and models.

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
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## Layout

```
manifold/
├── pyproject.toml
├── README.md
└── src/
    └── manifold/
        ├── __init__.py
        └── py.typed
```

## License

MIT — see [LICENSE](./LICENSE).
