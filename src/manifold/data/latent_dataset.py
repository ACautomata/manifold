"""Latent dataset: a volume dataset + frozen-VAE encoder, with a disk+RAM cache.

Emits ``{"latent","spacing","label","sample_id"}`` where ``latent`` is the
**scaled** VAE latent — the diffusion Module and Pipeline never reference
``scale_factor`` (ADR-0003 addendum). The data stack is the *only* place that
does:

1. :meth:`warm_cache` materializes **unscaled** latents (the injected
   ``encode_fn`` wraps :meth:`~manifold.AutoencoderKL.encode_raw`) into a
   two-tier disk + RAM cache;
2. :func:`estimate_scale_factor` estimates ``1/std(z)`` over the warmed cache
   and sets ``vae.scaling_factor``;
3. :meth:`__getitem__` returns the latent multiplied by that scale (scale-on-read).

The disk cache stores **unscaled** latents, so it is reusable across runs and
independent of the estimated scale. ``encode_fn`` is injected (and freed after
the warm) so the dataset is encoder-agnostic and testable with a mock.

Scale placement is moved from the training module (applied at train time) to
the data stack (scale-on-read at ``__getitem__``).
"""

from __future__ import annotations

import hashlib
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

#: A callable mapping a batch of images ``[B, 1, D0, D1, D2]`` to **unscaled**
#: latents ``[B, C, d0, d1, d2]`` (no grad). Built by the latent-prep
#: orchestration from the frozen VAE's ``encode_raw`` + a sliding-window inferer,
#: so the dataset stays encoder-agnostic (and testable with a mock).
EncodeFn = Callable[[Tensor], Tensor]


def _cache_path(cache_dir: str, sample_id: str, cache_tag: str) -> Path:
    """Stable per-sample latent path: ``<stem>__<cache_tag>__<hash>.pt``."""
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:12]
    stem = Path(sample_id).name.replace(".nii.gz", "").replace(".nii", "")
    return Path(cache_dir) / f"{stem}__{cache_tag}__{digest}.pt"


def _load_cache(cache_dir: str, sample_id: str, cache_tag: str) -> dict[str, Any] | None:
    """Return a cached item dict or ``None`` (miss / corrupt half-write)."""
    path = _cache_path(cache_dir, sample_id, cache_tag)
    if not path.is_file():
        return None
    try:
        item = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 — treat any read failure as a miss
        return None
    if not isinstance(item, dict) or "latent" not in item:
        return None
    return item


def _save_cache(cache_dir: str, sample_id: str, cache_tag: str, item: dict[str, Any]) -> None:
    """Atomic write: rank-unique ``.tmp`` then ``os.replace`` (DDP-safe)."""
    path = _cache_path(cache_dir, sample_id, cache_tag)
    rank = dist.get_rank() if dist.is_initialized() else 0
    tmp = path.with_suffix(path.suffix + f".tmp.r{rank}.p{os.getpid()}")
    torch.save(item, tmp)
    os.replace(tmp, path)


class LatentDataset(MedicalDataset):
    """Dataset of frozen-VAE latents + per-sample conditioning (scale-on-read).

    Args:
        source: an image-emitting :class:`MedicalDataset` (e.g.
            :class:`NiftiVolumeDataset`).
        encode_fn: callable ``images[B,1,D0,D1,D2] -> unscaled latents[B,C,d0,d1,d2]``
            (no grad). ``None`` is allowed only when the disk cache is fully warm.
        cache_dir: optional directory for per-sample ``.pt`` latent cache.
        cache_tag: tag mixed into the cache key (encode ``target_dim``/``divisor``
            here so changing them invalidates stale cache entries).
    """

    def __init__(
        self,
        source: MedicalDataset,
        encode_fn: EncodeFn | None = None,
        cache_dir: str | None = None,
        cache_tag: str = "v1",
    ) -> None:
        self.source = source
        self.encode_fn = encode_fn
        self.cache_dir = cache_dir
        self.cache_tag = cache_tag
        #: Scale-on-read multiplier (mirrors ``vae.scaling_factor`` once
        #: :func:`estimate_scale_factor` runs; 1.0 = no scaling before that).
        self.scaling_factor: float = 1.0
        self._ram: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self.source)

    def warm_cache(
        self,
        device: torch.device,
        show_progress: bool = True,
        rank: int | None = None,
        world: int | None = None,
    ) -> None:
        """Materialize every **unscaled** latent into RAM (disk-cache hits skip encode).

        Under DDP every rank reaches this with the full source dataset; if all
        ranks encode every sample they race-write the same ``.pt`` files. When
        ``world > 1`` and a ``cache_dir`` is set, each rank encodes only its
        ``i % world == rank`` shard, all ranks barrier, then every rank loads the
        full set from disk — one writer per file, identical RAM caches.
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
        n = len(self.source)
        sharded = world > 1 and self.cache_dir is not None
        if sharded:
            owned = list(range(rank, n, world))
            progress = tqdm(
                owned,
                desc=f"Pre-encoding VAE latents (rank {rank}/{world})",
                disable=not show_progress,
            )
            for i in progress:
                self._materialize(i, device)  # writes through _save_cache
            if dist.is_initialized():
                dist.barrier()
            self._ram = [self._materialize(i, device) for i in range(n)]
        else:
            progress = tqdm(range(n), desc="Pre-encoding VAE latents", disable=not show_progress)
            self._ram = [self._materialize(i, device) for i in progress]
        sample_ids_fn = getattr(self.source, "sample_ids", None)
        if callable(sample_ids_fn):
            hits = sum(
                1
                for sid in sample_ids_fn()
                if self.cache_dir is not None
                and _load_cache(self.cache_dir, sid, self.cache_tag) is not None
            )
        else:
            hits = sum(1 for i in range(n) if self._disk_hit(i))
        rank_zero_info(
            f"LatentDataset: materialized {len(self._ram)} latents "
            f"(disk cache {hits}/{n} hits, cache_dir={self.cache_dir}, "
            f"sharded={sharded})."
        )

    def free_encoder(self) -> None:
        """Drop the encoder reference so the VAE can leave GPU before UNet training."""
        self.encode_fn = None

    def raw_latent(self, index: int) -> Tensor:
        """The **unscaled** cached latent at *index* (the estimation input)."""
        if self._ram is None:
            raise RuntimeError("raw_latent requires warm_cache() first.")
        return self._ram[index]["latent"]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._ram is None:
            raise RuntimeError(
                "LatentDataset.__getitem__ requires warm_cache() first (the VAE "
                "encodes once, then __getitem__ serves from RAM)."
            )
        item = self._ram[index]
        # Scale-on-read (ADR-0003 addendum): the Module receives SCALED latents.
        latent = item["latent"] * self.scaling_factor
        label = item["label"]
        if not torch.is_tensor(label):
            label = torch.tensor(int(label), dtype=torch.long)
        out: dict[str, Any] = {
            "latent": latent,
            "spacing": item["spacing"],
            "label": label,
        }
        if "sample_id" in item:
            out["sample_id"] = item["sample_id"]
        return out

    def label_counts(self) -> dict[int, int]:
        return self.source.label_counts()

    # -- internals -----------------------------------------------------------

    def _disk_hit(self, index: int) -> bool:
        if self.cache_dir is None:
            return False
        sid = self.source[index]["sample_id"]
        return _load_cache(self.cache_dir, sid, self.cache_tag) is not None

    def _materialize(self, index: int, device: torch.device) -> dict[str, Any]:
        # Fast path: if the disk cache already has this sample, look it up
        # without loading the underlying NIfTI. The volume dataset exposes a
        # cheap ``sample_ids()`` so we can hit cache directly.
        if self.cache_dir is not None:
            sample_ids_fn = getattr(self.source, "sample_ids", None)
            if callable(sample_ids_fn):
                try:
                    sid = sample_ids_fn()[index]
                except (IndexError, TypeError):
                    sid = None
                if sid is not None:
                    fast_hit = _load_cache(self.cache_dir, sid, self.cache_tag)
                    if fast_hit is not None:
                        return fast_hit

        sample = self.source[index]
        sample_id = sample["sample_id"]

        if self.cache_dir is not None:
            hit = _load_cache(self.cache_dir, sample_id, self.cache_tag)
            if hit is not None:
                return hit

        if self.encode_fn is None:
            raise RuntimeError(
                f"Latent cache miss for {sample_id} and no encoder is set; "
                "provide encode_fn or warm the disk cache first."
            )

        image = sample["image"].unsqueeze(0).to(device)  # [1, 1, D0, D1, D2]
        latent = self._encode(image, device).squeeze(0).cpu()  # [C, d0, d1, d2] UNSCALED
        item: dict[str, Any] = {
            "latent": latent,
            "spacing": sample["spacing"],
            "label": sample["label"],
            "sample_id": sample_id,
        }
        if self.cache_dir is not None:
            _save_cache(self.cache_dir, sample_id, self.cache_tag, item)
        return item

    def _encode(self, image: Tensor, device: torch.device) -> Tensor:
        enabled = device.type == "cuda"
        with torch.no_grad(), torch.autocast(device.type, enabled=enabled):
            return self.encode_fn(image).detach().float()  # type: ignore[misc]


def estimate_scale_factor(
    dataset: LatentDataset,
    vae: Any,
    sample_size: int = 64,
) -> Tensor:
    """Estimate ``scale_factor = 1 / std(z)`` over the warmed **unscaled** cache.

    Computes ``1/std(z)`` over the first ``sample_size`` unscaled latents, sets it
    on ``vae.scaling_factor`` (the source of truth — at inference it comes from
    the converted checkpoint) and on ``dataset.scaling_factor`` (the scale-on-read
    multiplier). Each rank loads the FULL latent set after warm, so the estimate
    is identical across ranks and needs no all-reduce. The dataset must be warmed
    first. Only the data stack touches ``scaling_factor`` (ADR-0003 addendum).
    """
    if dataset._ram is None:
        raise RuntimeError("estimate_scale_factor requires warm_cache() first.")
    n = min(sample_size, len(dataset))
    latents = torch.stack([dataset.raw_latent(i) for i in range(n)]).float()
    scale = (1.0 / torch.std(latents)).detach()
    with torch.no_grad():
        vae.scaling_factor.fill_(float(scale))
    dataset.scaling_factor = float(scale)
    rank_zero_info(f"scale_factor -> {float(scale):.6f} (over {n} latents).")
    return scale
