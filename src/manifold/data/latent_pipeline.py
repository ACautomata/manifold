"""Latent-prep orchestration: frozen-VAE → unscaled latent cache → scale_factor.

The three build-helpers that turn a composed experiment config into the warmed
bundle the (deferred) trainer consumes:

1. :func:`build_volume_dataset` — the BraTS volume dataset + its label provider;
2. :func:`build_encode_pipeline` — load + freeze the pretrained VAE and build the
   sliding-window ``encode_fn`` over :meth:`~manifold.AutoencoderKL.encode_raw`
   (unscaled latents);
3. :func:`warm_latent_pipeline` — warm the cache, free the encoder, estimate
   ``scale_factor = 1/std(z)`` and set it on the VAE (ADR-0003 addendum).

The warm step accepts the encoder as an injected dependency so the smoke test can
feed a fake ``encode_fn`` + dummy VAE (it never calls :func:`load_vae`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from monai.inferers import sliding_window_inference
from omegaconf import DictConfig
from torch import Tensor

from ..config import autoencoder_divisor, build_vae
from .base import LabelProvider, MedicalDataset
from .labels import label_provider_from_config
from .latent_dataset import EncodeFn, LatentDataset, estimate_scale_factor
from .volume_dataset import NiftiVolumeDataset

#: Default sliding-window ROI for the VAE encode pass (matches the JiT inference config).
_DEFAULT_ENCODE_ROI = [320, 320, 160]
#: Default sliding-window overlap for the VAE encode pass.
_DEFAULT_ENCODE_OVERLAP = 0.4


@dataclass(frozen=True)
class LatentPipeline:
    """The warmed-byproduct bundle of the latent-prep step.

    Attributes:
        vol_ds: the warmed :class:`NiftiVolumeDataset` (kept so the trainer can
            reuse its provider for a held-out val set).
        latent_ds: the **scaled** latent cache — warmed, encoder freed, served
            from RAM (scale-on-read; ADR-0003 addendum).
        autoencoder: the frozen VAE, moved to CPU after the warm (its
            ``scaling_factor`` now carries the estimate).
        scale_factor: ``1/std(z)`` over the warmed cache. Domain property:
            identical across ranks (each holds the full cache after warm).
    """

    vol_ds: MedicalDataset
    latent_ds: LatentDataset
    autoencoder: nn.Module
    scale_factor: Tensor


def _load_checkpoint(path: str, map_location) -> dict:
    """Load a VAE checkpoint, preferring ``weights_only=True`` for safety.

    Falls back to ``weights_only=False`` only if the safe path rejects the file,
    logging a warning. Only point this at trusted checkpoint paths.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:  # noqa: BLE001 — fall back for exotic nested types
        logging.getLogger(__name__).warning(
            f"weights_only=True failed for {path} ({exc!r}); retrying with "
            "weights_only=False. Only load checkpoints from trusted sources."
        )
        return torch.load(path, map_location=map_location, weights_only=False)


def load_vae(
    cfg: DictConfig, device: torch.device, logger: logging.Logger | None = None
) -> nn.Module:
    """Build the VAE from the network config and load trained weights.

    Constructs :class:`~manifold.AutoencoderKL` from ``cfg.autoencoder``, then
    loads ``cfg.trained_autoencoder_path`` into ``vae.autoencoder`` — unwrapping
    the MAISI ``{unet_state_dict}`` wrapper if present (bare autoencoder keys).
    Returned frozen + in eval mode on *device*.
    """
    vae = build_vae(cfg).to(device)
    ckpt = _load_checkpoint(cfg.trained_autoencoder_path, map_location=device)
    if isinstance(ckpt, dict) and "unet_state_dict" in ckpt:
        ckpt = ckpt["unet_state_dict"]
    vae.autoencoder.load_state_dict(ckpt)
    vae.eval()
    if logger is not None:
        logger.info(f"VAE loaded from {cfg.trained_autoencoder_path}")
    return vae


def build_volume_dataset(
    cfg: DictConfig,
    *,
    target_dim: tuple[int, int, int],
    include_modality: bool,
    default_modality: int,
    logger: logging.Logger,
) -> tuple[NiftiVolumeDataset, LabelProvider]:
    """Build the training volume dataset + its label provider (decoupled seam).

    Resolves the data source (a manifest JSON when ``json_data_list`` points at
    one, else ``data_base_dir``), fails fast on an empty training set, and logs
    the label distribution. Returns ``(vol_ds, provider)``.
    """
    provider = label_provider_from_config(
        cfg, include_modality=include_modality, default_modality=default_modality
    )
    json_list = getattr(cfg, "json_data_list", None)
    source = json_list if json_list and os.path.exists(str(json_list)) else cfg.data_base_dir
    divisor = autoencoder_divisor(cfg)
    vol_ds = NiftiVolumeDataset(
        source, provider, target_dim, divisor, data_base_dir=cfg.data_base_dir
    )
    if not len(vol_ds):
        raise FileNotFoundError(
            f"No training NIfTI found under data_base_dir={cfg.data_base_dir} "
            f"or json_data_list={json_list}."
        )
    logger.info(f"num_files_train: {len(vol_ds)}; label_counts={vol_ds.label_counts()}")
    return vol_ds, provider


def build_encode_pipeline(
    cfg: DictConfig,
    *,
    device: torch.device,
    logger: logging.Logger | None = None,
) -> tuple[nn.Module, EncodeFn]:
    """Load + freeze the pretrained VAE and build the sliding-window ``encode_fn``.

    CLI-only path: the smoke test injects a fake ``encode_fn`` + dummy VAE
    instead, bypassing this helper (it does not call :func:`load_vae`).

    Returns ``(autoencoder, encode_fn)`` where ``encode_fn`` maps images to
    **unscaled** latents via :meth:`~manifold.AutoencoderKL.encode_raw` under
    sliding-window inference; ``autoencoder`` is still on *device* (the caller
    moves it to CPU after the cache is warm).
    """
    inf_cfg = cfg.diffusion_unet_inference
    autoencoder = load_vae(cfg, device, logger)
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    roi = list(inf_cfg.get("autoencoder_encode_sliding_window_infer_size", _DEFAULT_ENCODE_ROI))
    overlap = float(
        inf_cfg.get("autoencoder_encode_sliding_window_infer_overlap", _DEFAULT_ENCODE_OVERLAP)
    )

    def encode_fn(images: Tensor) -> Tensor:
        # encode_raw returns the UNSCALED latent; sliding-window stitches patches.
        return sliding_window_inference(
            images,
            roi_size=roi,
            sw_batch_size=1,
            predictor=autoencoder.encode_raw,
            overlap=overlap,
            sw_device=device,
            device=device,
        )

    return autoencoder, encode_fn


def warm_latent_pipeline(
    vol_ds: MedicalDataset,
    encode_fn: EncodeFn,
    autoencoder: nn.Module,
    *,
    cache_dir: str,
    cache_tag: str,
    device: torch.device,
    logger: logging.Logger | None,
    rank: int = 0,
    world: int = 1,
    scale_factor_sample_size: int,
) -> LatentPipeline:
    """Warm the unscaled latent cache once, estimate scale, return the bundle.

    Sequence:

    1. ``rank == 0`` ``mkdir``s the cache dir, then a DDP ``barrier()`` so no rank
       reads before the dir exists.
    2. Build the :class:`LatentDataset` and :meth:`~LatentDataset.warm_cache` it:
       under DDP each rank encodes only its ``i % world == rank`` shard, barriers,
       then every rank loads the full set from disk → identical RAM caches.
    3. :meth:`~LatentDataset.free_encoder` drops the VAE from the dataset.
    4. Move the ``autoencoder`` to CPU and clear the CUDA cache.
    5. :func:`estimate_scale_factor` → ``1/std(z)`` over the warmed cache, set on
       the VAE (and the dataset's scale-on-read multiplier).
    """
    if rank == 0:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    latent_ds = LatentDataset(vol_ds, encode_fn=encode_fn, cache_dir=cache_dir, cache_tag=cache_tag)
    latent_ds.warm_cache(device, logger, show_progress=rank == 0, rank=rank, world=world)
    latent_ds.free_encoder()

    autoencoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if logger is not None:
        logger.info("VAE moved to CPU; training the UNet on cached latents.")

    scale_factor = estimate_scale_factor(
        latent_ds, autoencoder, sample_size=scale_factor_sample_size, logger=logger
    )
    return LatentPipeline(
        vol_ds=vol_ds,
        latent_ds=latent_ds,
        autoencoder=autoencoder,
        scale_factor=scale_factor,
    )
