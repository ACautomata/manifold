"""Paired latent dataset: encodes each unique volume once, shares it across pairs.

A :class:`~manifold.data.PairedNiftiVolumeDataset` + a frozen-VAE encoder, with a
two-tier disk + RAM cache keyed by **unique volume ``sample_id``** (not by pair).
Warming a 12-pair / 4-contrast subject encodes each of the 4 unique volumes
exactly once — the encode cost is the unique-volume count, not the pair count
(ADR-0014 — shared cache, no 12× duplication).

Emits ``{"src_latent","tgt_latent","src_label","tgt_label","spacing"}`` where both
latents are **scaled** — :class:`~manifold.modules.PairedLatentFlowModule` and
:class:`~manifold.pipelines.PairedLatentFlowPipeline` never reference
``scale_factor`` (ADR-0003 addendum). The data stack is the *only* place that does:

1. :meth:`warm_cache` materializes **unscaled** latents for every unique volume
   (the injected ``encode_fn`` wraps :meth:`~manifold.AutoencoderKL.encode_raw`)
   into a disk + RAM cache keyed by ``sample_id``;
2. :func:`estimate_paired_scale_factor` estimates ``1/std(z)`` over the warmed
   **unique** latents (which is the union of src and tgt volumes, so src+tgt are
   pooled by construction) and sets ``vae.scaling_factor``;
3. :meth:`__getitem__` returns each latent multiplied by that scale (scale-on-read).

Sibling of :class:`~manifold.data.LatentDataset`; the cache helpers
(:func:`~manifold.data.latent_dataset._cache_path` /
``_load_cache`` / ``_save_cache`) are imported and reused so the on-disk format is
identical and a future mixed cache stays consistent. The disk cache stores
**unscaled** latents, so it is reusable across runs and independent of the
estimated scale.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor
from tqdm import tqdm

from .base import MedicalDataset
from .latent_dataset import EncodeFn, _load_cache, _save_cache
from .paired_volume_dataset import PairedNiftiVolumeDataset


class PairedLatentDataset(MedicalDataset):
    """Paired latent dataset (shared unique-volume cache, scale-on-read).

    Args:
        paired_vol_ds: a :class:`PairedNiftiVolumeDataset` (the image-emitting
            pair source). Its :meth:`~PairedNiftiVolumeDataset.unique_sample_ids`
            drives the warm — each unique volume is encoded once.
        encode_fn: callable ``images[B,1,D0,D1,D2] -> unscaled latents[B,C,d0,d1,d2]``
            (no grad). ``None`` is allowed only when the disk cache is fully warm.
        cache_dir: optional directory for per-unique-volume ``.pt`` latent cache.
        cache_tag: tag mixed into the cache key (encode ``target_dim``/``divisor``
            here so changing them invalidates stale cache entries).
    """

    def __init__(
        self,
        paired_vol_ds: PairedNiftiVolumeDataset,
        encode_fn: EncodeFn | None = None,
        cache_dir: str | None = None,
        cache_tag: str = "paired_v1",
    ) -> None:
        self.source = paired_vol_ds
        self.encode_fn = encode_fn
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag
        #: Scale-on-read multiplier (mirrors ``vae.scaling_factor`` once
        #: :func:`estimate_paired_scale_factor` runs; 1.0 = no scaling before that).
        self.scaling_factor: float = 1.0
        #: ``{sample_id: latent_item}`` — one entry per UNIQUE volume, shared by
        #: every pair that references it. ``None`` until :meth:`warm_cache` runs.
        self._ram: dict[str, dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self.source)

    def warm_cache(
        self,
        device: torch.device,
        show_progress: bool = True,
        rank: int | None = None,
        world: int | None = None,
    ) -> None:
        """Materialize an **unscaled** latent for every unique volume (disk hits skip encode).

        Iterates :meth:`PairedNiftiVolumeDataset.unique_sample_ids` — the deduped
        volume set — so each volume is encoded exactly once however many pairs
        reference it. Under DDP every rank reaches this with the full unique set;
        when ``world > 1`` and a ``cache_dir`` is set, each rank encodes only its
        ``i % world == rank`` shard, all ranks barrier, then every rank loads the
        full set from disk (mirrors :meth:`LatentDataset.warm_cache` — one writer
        per file, identical RAM caches).
        """
        # F3 (ADR-0017): derive rank/world from dist when the PG is initialized
        # (fallback 0/1) so a post-PG DataModule.setup() caller need not thread them.
        # Explicit kwargs are honored only when the PG is NOT initialized.
        if dist.is_initialized():
            rank = dist.get_rank()
            world = dist.get_world_size()
        else:
            rank = 0 if rank is None else rank
            world = 1 if world is None else world
        if self.cache_dir is not None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        sample_ids = self.source.unique_sample_ids()
        n = len(sample_ids)
        sharded = world > 1 and self.cache_dir is not None
        if sharded:
            owned = list(range(rank, n, world))
            progress = tqdm(
                owned,
                desc=f"Pre-encoding paired VAE latents (rank {rank}/{world})",
                disable=not show_progress,
            )
            for i in progress:
                self._materialize(sample_ids[i], device)  # writes through _save_cache
            if dist.is_initialized():
                dist.barrier()
            self._ram = {sid: self._materialize(sid, device) for sid in sample_ids}
        else:
            progress = tqdm(
                sample_ids,
                desc="Pre-encoding paired VAE latents",
                disable=not show_progress,
            )
            self._ram = {sid: self._materialize(sid, device) for sid in progress}
        hits = (
            sum(
                1
                for sid in sample_ids
                if _load_cache(self.cache_dir, sid, self.cache_tag) is not None
            )
            if self.cache_dir is not None
            else 0
        )
        rank_zero_info(
            f"PairedLatentDataset: materialized {len(self._ram)} unique latents "
            f"over {len(self.source)} pairs (disk cache {hits}/{n} hits, "
            f"cache_dir={self.cache_dir}, sharded={sharded})."
        )

    def free_encoder(self) -> None:
        """Drop the encoder reference so the VAE can leave GPU before UNet training."""
        self.encode_fn = None

    def raw_latent(self, sample_id: str) -> Tensor:
        """The **unscaled** cached latent for *sample_id* (a scale-estimation input)."""
        if self._ram is None:
            raise RuntimeError("raw_latent requires warm_cache() first.")
        return self._ram[sample_id]["latent"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._ram is None:
            raise RuntimeError(
                "PairedLatentDataset.__getitem__ requires warm_cache() first "
                "(the VAE encodes each unique volume once, then __getitem__ serves "
                "from the shared RAM cache)."
            )
        # pair_meta skips the volume dataset's __getitem__ (which loads both NIfTIs)
        # — a training-batch fetch is pure RAM lookup once the cache is warm.
        meta = self.source.pair_meta(index)
        src_item = self._ram[meta["src_id"]]
        tgt_item = self._ram[meta["tgt_id"]]
        # Scale-on-read (ADR-0003 addendum): the Module receives SCALED latents,
        # applied to BOTH endpoints (one scale_factor pooled over src∪tgt). Spacing
        # comes from the cached src volume (BraTS src/tgt are co-registered).
        return {
            "src_latent": src_item["latent"] * self.scaling_factor,
            "tgt_latent": tgt_item["latent"] * self.scaling_factor,
            "src_label": meta["src_label"],
            "tgt_label": meta["tgt_label"],
            "spacing": src_item["spacing"],
        }

    # -- internals -----------------------------------------------------------

    def _materialize(self, sample_id: str, device: torch.device) -> dict[str, Any]:
        """Disk-cache hit → return; else encode the unique volume once and save.

        Stores the volume's ``spacing`` alongside the latent so a training
        ``__getitem__`` never has to re-open the NIfTI (the affine lives only in
        the file) — the spacing is captured here at warm time and read from RAM
        thereafter.
        """
        if self.cache_dir is not None:
            hit = _load_cache(self.cache_dir, sample_id, self.cache_tag)
            if hit is not None:
                return hit

        if self.encode_fn is None:
            raise RuntimeError(
                f"Paired latent cache miss for {sample_id} and no encoder is set; "
                "provide encode_fn or warm the disk cache first."
            )

        sample = self.source._load_volume(sample_id)
        image = sample["image"].unsqueeze(0).to(device)  # [1, 1, D0, D1, D2]
        latent = self._encode(image, device).squeeze(0).cpu()  # [C, d0, d1, d2] UNSCALED
        item: dict[str, Any] = {
            "latent": latent,
            "sample_id": sample_id,
            "spacing": sample["spacing"],  # captured once → no NIfTI read at train time
        }
        if self.cache_dir is not None:
            _save_cache(self.cache_dir, sample_id, self.cache_tag, item)
        return item

    def _encode(self, image: Tensor, device: torch.device) -> Tensor:
        enabled = device.type == "cuda"
        with torch.no_grad(), torch.autocast(device.type, enabled=enabled):
            return self.encode_fn(image).detach().float()  # type: ignore[misc]


def estimate_paired_scale_factor(
    dataset: PairedLatentDataset,
    vae: Any,
    sample_size: int = 64,
) -> Tensor:
    """Estimate ``scale_factor = 1 / std(z)`` over the warmed **unique** latents.

    The unique set is the union of src and tgt volumes, so this pools src+tgt by
    construction (ADR-0014 — one scale over the paired transport's two data
    endpoints). Computes ``1/std(z)`` over the first ``sample_size`` unique
    unscaled latents, sets it on ``vae.scaling_factor`` (the source of truth) and
    on ``dataset.scaling_factor`` (the scale-on-read multiplier applied to BOTH
    src and tgt at :meth:`PairedLatentDataset.__getitem__`). Each rank loads the
    FULL unique set after warm, so the estimate is identical across ranks and
    needs no all-reduce (mirrors :func:`estimate_scale_factor`). The dataset must
    be warmed first. Only the data stack touches ``scaling_factor`` (ADR-0003
    addendum).
    """
    if dataset._ram is None:
        raise RuntimeError("estimate_paired_scale_factor requires warm_cache() first.")
    sample_ids = list(dataset._ram.keys())
    n = min(sample_size, len(sample_ids))
    latents = torch.stack([dataset.raw_latent(sid) for sid in sample_ids[:n]]).float()
    scale = (1.0 / torch.std(latents)).detach()
    with torch.no_grad():
        vae.scaling_factor.fill_(float(scale))
    dataset.scaling_factor = float(scale)
    rank_zero_info(f"scale_factor -> {float(scale):.6f} (over {n} unique paired latents).")
    return scale


__all__ = ["PairedLatentDataset", "estimate_paired_scale_factor"]
