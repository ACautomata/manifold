# Manifold OpenWiki

Manifold is a Python research package for latent-flow generation and translation of 3D medical volumes. It combines MONAI MAISI model implementations, stable-pretraining training modules, Lightning orchestration, and a diffusers-style separation between models, schedulers, training modules, and inference pipelines. The project uses its own component bases; it does not subclass `diffusers` (`README.md`, `CONTEXT.md`, `src/manifold/`).

## Start here

- [Architecture and source map](architecture.md) — component boundaries, data/config layers, domain vocabulary, and where to look in source.
- [Key workflows](workflows.md) — JiT and Paired JiT training, reward/GRPO stages, inference, checkpoints, and export.
- [Operations and testing](operations-and-testing.md) — setup, validation behavior, distributed metrics, runbook cautions, and focused test commands.
- Repository decisions live in `docs/adr/`; ADR-0025 is authoritative for the current distributed-validation policy.

## Setup

Python 3.10–3.13 is supported (`pyproject.toml`). A typical editable development install is:

```bash
conda create -n manifold python=3.12 -y
conda activate manifold
pip install -e ".[dev]"
pytest
ruff check .
```

Machine-specific paths are supplied through an environment config. Start from `configs/env/environment.yaml`: required values use `???` and fail fast until set, while `null` means optional/off. Experiment configuration composes environment, training recipe, and network YAML through OmegaConf; command-line dotlist overrides can replace values such as `model_dir=/tmp/run`.

## Main commands

The console entries are declared in `pyproject.toml`:

```bash
# Noise-to-volume JiT
manifold-train -e configs/env/environment.yaml \
  -c configs/train/config_rflow_jit.yaml \
  -t configs/network/config_network.yaml

# Paired source-to-target translation
manifold-train-paired -e configs/env/environment.yaml \
  -c configs/train/config_paired_jit.yaml \
  -t configs/network/config_network.yaml

# Reward and policy post-training
manifold-train-reward ...
manifold-train-grpo ...
manifold-train-paired-reward ...
manifold-train-paired-grpo ...
```

Training writes full-state Lightning `.ckpt` files. Inference consumes native per-component directories, created through `scripts/export_checkpoint.py`; the current export contract always publishes raw optimizer UNet weights, not EMA weights. See [Key workflows](workflows.md#checkpoint-and-export-contract).

## Core concepts

- **JiT** means an `x0`-predicting latent-flow denoiser, not `torch.jit`. It learns the transport `z = t*x + (1-t)*e`, where `t=0` is noise and `t=1` is clean data.
- **Paired JiT** reuses that transport and Heun integration but starts at a source data latent and ends at a target latent. BraTS any-to-any pairing yields 12 ordered non-self directions for a complete four-contrast subject.
- **Conditioning** is medical metadata—voxel spacing and modality/contrast labels—not text.
- **`scaling_factor`** is VAE-owned latent normalization (`1/std(z)`). Caches remain unscaled; datasets scale reads and VAE decode reverses the scaling.
- **Granular-GRPO** assigns terminal reward to one forked stochastic transition at a time, making policy post-training tractable for 3D data. Paired GRPO uses a training-only Brownian bridge; deployed paired inference remains deterministic Heun.

## Current engineering state

The latest source policy makes FID, paired PSNR/SSIM, and GRPO `val/mean_reward` fully distributed: every rank evaluates its shard and contributes to a global metric. Best-by-metric checkpoint monitors remain enabled under DDP. This supersedes the rank-0-only workaround in ADR-0016, but all-rank full-volume VAE decode is still awaiting an 8-DCU sugon probe; do not treat that vendor-runtime risk as resolved. See [Distributed validation runbook](operations-and-testing.md#distributed-validation-runbook).

EMA training and EMA export were removed. Metric callbacks evaluate the live/raw model, and native export bakes raw `state_dict` UNet weights (`src/manifold/training/export.py`).

## Backlog

- **Reward-data preparation and probes** — `src/manifold/data/paired_reward_pairs.py`, `scripts/generate_reward_pairs.py`, `scripts/diag_*`: deferred because current scripts and the online reward path need a dedicated legacy-versus-current workflow audit.
- **Detailed persistence format** — `src/manifold/configuration.py`, `src/manifold/pipelines/`: native component layout is summarized, but serialization compatibility and schema details are deferred.
- **Dataset/cache implementation guide** — `src/manifold/data/`: high-level warming and pairing contracts are covered; file discovery, manifests, and cache internals are deferred.
