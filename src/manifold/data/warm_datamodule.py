"""A Lightning ``DataModule`` whose ``setup()`` runs the VAE-encode warm post-PG.

ADR-0017 (issue #84): the VAE-encode latent warm must run AFTER the process
group is initialized so the per-rank sharding machinery in
:meth:`~manifold.data.LatentDataset.warm_cache` (``i % world == rank``) activates
- one writer per cache file instead of every rank re-encoding the full set. The
warm was previously in ``main()`` before ``trainer.fit``, when
``dist.is_initialized()`` is ``False`` and ``world=1`` made the sharded branch
unreachable (dead code in production + the ~2.7h cold-start cost).

Lightning initializes the process group inside ``trainer.fit`` BEFORE calling
``setup()``, so moving the warm here turns the sharding on with no manual PG
juggling. The atomic warm unit is the FULL
:func:`~manifold.data.warm_latent_pipeline` (``warm_cache`` -> ``free_encoder`` ->
``estimate_scale_factor``) - never just the encode, or ``estimate_scale_factor``
runs before warm and yields a wrong/empty scale.

The val-real-reference latents (the FIDCallback's ``real_latents``) are computed
inside ``setup()`` post-warm and exposed via :attr:`val_latents` so the FID
callback can pull them lazily at the first validation epoch (F5 - they do not
exist at ``run_training`` construction time).

Two construction modes:
- **warmed** (the test smoke): ``latent_ds`` already materialized; ``setup()`` is a
  no-op (single-GPU parity preserved - the warm already ran).
- **cold** (production): ``vol_ds`` + ``encode_fn`` + ``vae`` + warm kwargs; ``setup()``
  runs :func:`warm_latent_pipeline` post-PG and computes ``val_latents``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .latent_pipeline import LatentPipeline, warm_latent_pipeline


class LatentWarmDataModule(LightningDataModule):
    """A ``DataModule`` whose ``setup()`` runs the VAE-encode warm post-PG.

    Args:
        latent_ds: the ALREADY-warmed :class:`LatentDataset` (the test path), or a
            placeholder ``None`` when ``warm_fn`` is set (the production cold path).
        vae: the frozen VAE (carries ``scaling_factor`` post-warm).
        batch_size / num_workers: the train/val DataLoader batch size + workers.
        val_latents: the FIXED real-reference latents (the test path passes them
            pre-computed; the cold path leaves this ``None`` and ``setup()`` fills it).
        warm_fn: a callable ``() -> LatentPipeline`` run inside ``setup()`` (the
            cold path - wraps :func:`warm_latent_pipeline` with closed-over args).
            When ``None``, the bundle is assumed warmed (``latent_ds`` + ``val_latents``
            already set) and ``setup()`` is a no-op.
        val_subset_size: the FID real-reference subset size (the cold path's seeded
            ``randperm`` prefix). Unused on the warmed path.
    """

    def __init__(
        self,
        *,
        latent_ds,
        vae,
        batch_size: int,
        num_workers: int = 0,
        val_latents: torch.Tensor | None = None,
        warm_fn=None,
        val_subset_size: int = 32,
    ):
        super().__init__()
        self._latent_ds = latent_ds
        self._vae = vae
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._val_latents = val_latents
        self._warm_fn = warm_fn
        self._val_subset_size = val_subset_size

    @property
    def latent_ds(self):
        """The (post-``setup``) warmed latent dataset; ``None`` before ``setup()``."""
        return self._latent_ds

    @property
    def vae(self):
        """The VAE carrying the post-warm ``scaling_factor``."""
        return self._vae

    @property
    def val_latents(self) -> torch.Tensor | None:
        """The fixed real-reference latents; ``None`` until ``setup()`` runs (F5)."""
        return self._val_latents

    def setup(self, stage: str | None = None) -> None:
        """Run the atomic warm post-PG (cold path) or no-op (warmed path).

        On the cold path ``warm_fn`` (a closure over :func:`warm_latent_pipeline`)
        runs here - after Lightning has initialized the process group inside
        ``trainer.fit`` - so the ``i % world == rank`` sharded branch activates.
        ``val_latents`` is then computed (seeded ``randperm`` over the rank-constant
        ``len(latent_ds)``) so the FID real reference is not rank-dependent.
        """
        if self._warm_fn is None:
            return  # warmed path (test smoke): the cache is already materialized.
        pipeline: LatentPipeline = self._warm_fn()
        self._latent_ds = pipeline.latent_ds
        self._vae = pipeline.autoencoder
        # Compute the fixed real-reference subset (seed-0 randperm over the full
        # warmed cache, which every rank loaded identically -> rank-constant).
        self._val_latents = _val_reference_subset(
            pipeline.latent_ds, pipeline.autoencoder, self._val_subset_size
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._latent_ds, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )

    def val_dataloader(self) -> DataLoader:
        # spt.DataModule wraps the val loader in a DistributedSampler under DDP
        # (Lightning auto-wraps); reuse train as val when no separate split is given.
        return DataLoader(
            self._latent_ds, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers
        )


def _val_reference_subset(latent_ds, vae, val_subset_size: int) -> torch.Tensor:
    """The seeded-shuffle prefix of ``val_subset_size`` scaled real latents.

    Identical across ranks (every rank loaded the full cache after warm + the
    ``randperm`` uses seed 0 over the rank-constant ``len(latent_ds)``), so the FID
    real reference is not rank-dependent (F1 gate: ``torch.equal`` on rank 0/1).
    Mirrors the original ``cli._warm_data`` computation.
    """
    n = min(val_subset_size, len(latent_ds))
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(latent_ds), generator=g)[:n].tolist()
    return torch.stack([latent_ds.raw_latent(i) * latent_ds.scaling_factor for i in idx])


__all__ = ["LatentWarmDataModule", "PairedWarmDataModule"]


class PairedWarmDataModule(LightningDataModule):
    """A ``DataModule`` whose ``setup()`` warms the paired train + val caches post-PG.

    F2/F4 (ADR-0017, issue #84): both paired warm calls (train ``latent_ds`` + the
    held-out ``val_latent_ds``) move from ``main()`` (pre-PG) to ``setup()`` (post-PG)
    so the ``PairedLatentDataset.warm_cache`` sharded branch activates. The scale is
    estimated over TRAIN unique latents only and propagated to the val dataset.

    Two modes (mirror :class:`LatentWarmDataModule`):
    - **warmed** (test smoke): ``latent_ds`` + ``val_latent_ds`` already set;
      ``setup()`` is a no-op.
    - **cold** (production): ``warm_fn`` set -> ``setup()`` runs it (a closure over
      the two ``warm_cache`` + ``estimate_paired_scale_factor``) post-PG.
    """

    def __init__(
        self,
        *,
        latent_ds,
        vae,
        batch_size: int,
        num_workers: int = 0,
        val_latent_ds=None,
        warm_fn=None,
    ):
        super().__init__()
        self._latent_ds = latent_ds
        self._vae = vae
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._val_latent_ds = val_latent_ds
        self._warm_fn = warm_fn

    @property
    def latent_ds(self):
        return self._latent_ds

    @property
    def vae(self):
        return self._vae

    @property
    def val_latent_ds(self):
        return self._val_latent_ds

    def setup(self, stage: str | None = None) -> None:
        if self._warm_fn is None:
            return  # warmed (test) path.
        self._latent_ds, self._val_latent_ds, self._vae = self._warm_fn()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._latent_ds, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )

    def val_dataloader(self) -> DataLoader:
        ds = self._val_latent_ds if self._val_latent_ds is not None else self._latent_ds
        return DataLoader(ds, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)
