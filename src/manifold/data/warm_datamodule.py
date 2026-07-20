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

import math
import os
from typing import Any

import torch
import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


from .latent_pipeline import warm_latent_pipeline


class _ValidationDataset(Dataset):
    """Validation-only wrapper that adds a reserved ``_is_padding`` sample flag."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        is_padding = False
        if isinstance(index, tuple):
            index, is_padding = index
        item = self.dataset[index]
        if not isinstance(item, dict):
            raise TypeError("validation datasets must yield mappings for padding masks")
        if "_is_padding" in item:
            raise KeyError("validation sample uses reserved key '_is_padding'")
        return {**item, "_is_padding": bool(is_padding)}


class _TaggedDistributedSampler(DistributedSampler):
    """Default equal-length DDP padding, with padded rows tagged for metric masking."""

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            real = torch.randperm(n, generator=g).tolist()
        else:
            real = list(range(n))
        tagged = [(index, False) for index in real]
        padding_size = self.total_size - n
        if padding_size:
            if n == 0:
                return iter([])
            repeated = (real * math.ceil(padding_size / n))[:padding_size]
            tagged.extend((index, True) for index in repeated)
        tagged = tagged[self.rank : self.total_size : self.num_replicas]
        if len(tagged) != self.num_samples:
            raise AssertionError("tagged sampler produced the wrong per-rank length")
        return iter(tagged)


def _validation_dataset_and_sampler(dataset):
    """Wrap validation samples and attach an equal-length tagged sampler post-PG."""
    wrapped = _ValidationDataset(dataset)
    if not (dist.is_available() and dist.is_initialized()):
        return wrapped, None
    return wrapped, _TaggedDistributedSampler(wrapped, shuffle=False, drop_last=False)




def _validation_loader(dataset, *, batch_size: int, num_workers: int) -> DataLoader:
    wrapped, sampler = _validation_dataset_and_sampler(dataset)
    return DataLoader(
        wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        sampler=sampler,
    )


class _EmptyDataset(Dataset):
    """A zero-length dataset so a no-val ``val_dataloader`` yields 0 batches.

    Returned (never leaked as train) when no held-out val is configured and the
    caller did not opt into ``allow_train_as_val``. Validation is also disabled
    at the Trainer (``limit_val_batches=0`` + ``check_val_every_n_epoch=None``),
    so this is a defensive fallback - it must never expose training data.
    """

    def __len__(self) -> int:
        return 0

    def __getitem__(self, index):
        raise IndexError("empty dataset")


class LatentWarmDataModule(LightningDataModule):
    """A ``DataModule`` whose ``setup()`` runs the VAE-encode warm post-PG.

    Args:
        latent_ds: the ALREADY-warmed :class:`LatentDataset` (the test path), or a
            placeholder ``None`` when ``warm_fn`` is set (the production cold path).
        vae: the frozen VAE (carries ``scaling_factor`` post-warm).
        batch_size / num_workers: the train/val DataLoader batch size + workers.
        val_latents: the FIXED real-reference latents (the test path passes them
            pre-computed; the cold path leaves this ``None`` and ``setup()`` fills it).
        val_latent_ds: the ALREADY-warmed held-out validation :class:`LatentDataset`
            (the test path); the cold path leaves this ``None`` and ``setup()`` fills
            it from the ``warm_fn`` triple.
        warm_fn: a callable ``() -> (latent_ds, autoencoder, val_latent_ds)`` run
            inside ``setup()`` (the cold path — wraps :func:`warm_latent_pipeline`
            plus an optional held-out val warm). When ``None``, the bundle is assumed
            warmed (``latent_ds`` + ``val_latents`` already set) and ``setup()`` is a
            no-op.
        val_subset_size: the FID real-reference subset size (the cold path's seeded
            ``randperm`` prefix). Unused on the warmed path.
        allow_train_as_val: smoke-only opt-in to reuse the train latent set as the
            val loader + derive ``val_latents`` from it (the pre-cleanup behavior).
            Defaults ``False``: with no held-out val, ``val_dataloader`` returns an
            empty loader and ``setup()`` does NOT build a train-derived ``val_latents``
            (production disables validation instead - never silently leaks train).
    """

    def __init__(
        self,
        *,
        latent_ds,
        vae,
        batch_size: int,
        num_workers: int = 0,
        val_latents: torch.Tensor | None = None,
        val_latent_ds=None,
        warm_fn=None,
        val_subset_size: int = 32,
        allow_train_as_val: bool = False,
    ):
        super().__init__()
        self._latent_ds = latent_ds
        self._vae = vae
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._val_latents = val_latents
        self._val_latent_ds = val_latent_ds
        self._warm_fn = warm_fn
        self._val_subset_size = val_subset_size
        self._allow_train_as_val = allow_train_as_val

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

    @property
    def val_latent_ds(self):
        """The held-out validation latent dataset; ``None`` until ``setup()`` runs."""
        return self._val_latent_ds

    def setup(self, stage: str | None = None) -> None:
        """Run the atomic warm post-PG (cold path) or no-op (warmed path).

        On the cold path ``warm_fn`` (a closure over :func:`warm_latent_pipeline`)
        runs here - after Lightning has initialized the process group inside
        ``trainer.fit`` - so the ``i % world == rank`` sharded branch activates.
        It returns ``(latent_ds, autoencoder, val_latent_ds)`` where ``val_latent_ds``
        is the held-out validation latent dataset (warmed from ``val_data_base_dir``)
        or ``None`` when no held-out val is configured.

        ``val_latents`` (the FID real reference) is a seeded prefix of the held-out
        ``val_latent_ds`` when present; otherwise it is left ``None`` except under
        the ``allow_train_as_val`` smoke opt-in (a prefix of the TRAIN cache, which
        tests wiring — not held-out generalization).
        """
        if self._warm_fn is None:
            return  # warmed path (test smoke): the cache is already materialized.
        self._latent_ds, self._vae, self._val_latent_ds = self._warm_fn()
        # The FID real reference: the held-out val prefix when present; else the
        # smoke-only train prefix under allow_train_as_val; else None (validation
        # disabled at the Trainer). Never derived unconditionally from the TRAIN
        # cache in production — that would leak train metrics into validation.
        ref_ds = self._val_latent_ds if self._val_latent_ds is not None else (
            self._latent_ds if self._allow_train_as_val else None
        )
        if ref_ds is not None:
            self._val_latents = _val_reference_subset(
                ref_ds, self._vae, self._val_subset_size
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._latent_ds, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers
        )

    def val_dataloader(self) -> DataLoader:
        # Held-out val (from val_data_base_dir) when present; else the smoke opt-in
        # (allow_train_as_val) reuses train with a loud warning; otherwise an empty
        # loader yields 0 batches (validation is also disabled at the Trainer).
        if self._val_latent_ds is not None:
            return _validation_loader(
                self._val_latent_ds, batch_size=self._batch_size, num_workers=self._num_workers
            )
        if self._allow_train_as_val:
            rank_zero_info(
                "LatentWarmDataModule: no held-out val; reusing TRAIN as val "
                "(allow_train_as_val=True, smoke only) - val/* metrics are NOT held-out."
            )
            return _validation_loader(
                self._latent_ds, batch_size=self._batch_size, num_workers=self._num_workers
            )
        return DataLoader(_EmptyDataset(), batch_size=self._batch_size, num_workers=0)


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
        allow_train_as_val: bool = False,
    ):
        super().__init__()
        self._latent_ds = latent_ds
        self._vae = vae
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._val_latent_ds = val_latent_ds
        self._warm_fn = warm_fn
        self._allow_train_as_val = allow_train_as_val

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
        # Held-out val when available; never silently fall back to train. The smoke
        # opt-in (allow_train_as_val) reuses train with a loud warning; otherwise an
        # empty loader yields 0 batches (validation is also disabled at the Trainer
        # when no held-out val split is configured).
        if self._val_latent_ds is not None:
            return _validation_loader(
                self._val_latent_ds, batch_size=self._batch_size, num_workers=self._num_workers
            )
        if self._allow_train_as_val:
            rank_zero_info(
                "PairedWarmDataModule: no held-out val_latent_ds; reusing TRAIN as val "
                "(allow_train_as_val=True, smoke only) - val/* metrics are NOT held-out."
            )
            return _validation_loader(
                self._latent_ds, batch_size=self._batch_size, num_workers=self._num_workers
            )
        return DataLoader(_EmptyDataset(), batch_size=self._batch_size, num_workers=0)
