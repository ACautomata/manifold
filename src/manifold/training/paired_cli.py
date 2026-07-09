"""``manifold-train-paired`` console entry + the testable paired-training core.

The console entry (issue #69) composes the OmegaConf experiment config, warms the
paired latent cache (Slice 2), builds the :class:`~manifold.modules.PairedLatentFlowModule`
+ the fixed validation subset + the callbacks (train metrics, double EMA,
PSNR/SSIM, ``ModelCheckpoint``), and calls ``Trainer.fit``. From scratch
(ADR-0014 — no warm-start). Sibling of ``manifold.training.cli``; the heavy
data-warming lives in :func:`main`, the integration core :func:`run_paired_training`
(Module + datamodule + callbacks + ``ModelCheckpoint`` + ``build_trainer`` +
``fit``) is split out so a tiny CPU smoke can drive it with a fake latent cache
(the issue's testing seam) instead of BraTS + a real VAE.

Best-checkpoint selection monitors ``val/psnr`` (``mode="max"``) on single-GPU;
under DDP (the PSNR/SSIM callback is rank-0-only) it falls back to ``save_last``
+ ``every_n_epochs`` (mirrors the noise→data DDP path).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import opt
from ..data.datamodule import build_datamodule
from ..metrics import PairedPSNRSSIMCallback
from ..modules.paired_latent_flow import PairedLatentFlowModule
from ..pipelines.paired_latent_flow import PairedLatentFlowPipeline
from .ema import DoubleEMACallback
from .metrics import LatentX0MAE, TrainLossLogger
from .trainer import build_trainer

_log = logging.getLogger(__name__)


@dataclass
class _DataBundle:
    """The warmed paired-data bundle ``main`` passes into :func:`run_paired_training`.

    (Injection seam for the CPU smoke test, which feeds a fake paired latent cache
    + a tiny VAE instead of warming BraTS through a real VAE encode.)

    ``val_latent_ds`` is an optional subject-level held-out validation dataset
    (``None`` → ``run_paired_training`` falls back to val=train, the legacy path).
    """

    latent_ds: Any
    vae: Any
    val_latent_ds: Any = None


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_psnr: bool,
    monitor_metric: str = "val/psnr",
    every_n_epochs: int = 1,
    save_top_k: int = 3,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` (ADR-0006).

    Single-GPU + PSNR: monitor ``val/psnr`` (``mode="max"``, top-k, last, full
    state) — the raw-optimizer metric is sufficient on a short from-scratch run.
    Under DDP (the PSNR/SSIM callback is rank-0-only, so the metric is not global)
    fall back to ``save_last`` + ``every_n_epochs`` with no monitor.
    ``auto_insert_metric_name = False`` because the metric key contains a ``/``.
    """
    if monitor_psnr:
        return ModelCheckpoint(
            dirpath=model_dir,
            filename=f"paired-{{epoch:03d}}-{{step}}-{{{monitor_metric}:.3f}}",
            monitor=monitor_metric,
            mode="max",
            save_top_k=save_top_k,
            save_last=True,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
    return ModelCheckpoint(
        dirpath=model_dir,
        filename="paired-{epoch:03d}-{step}",
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=True,
        every_n_epochs=max(1, every_n_epochs),
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


def run_paired_training(
    *,
    module: PairedLatentFlowModule,
    bundle: _DataBundle,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    limit_val_batches: int | float = 4,
    save_top_k: int = 3,
    seed: int = 0,
    num_inference_steps: int = 4,
    every_n_epochs: int = 1,
    monitor_metric: str = "val/psnr",
    ckpt_path: str | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the paired module (the core seam).

    Builds the train-metrics / double-EMA / PSNR-SSIM callbacks + a stock
    ``ModelCheckpoint`` and runs ``Trainer.fit``. Returns ``(trainer, ckpt)`` so
    callers can find the written ``.ckpt``.

    Args:
        bundle: the warmed paired latent dataset + held VAE.
        num_inference_steps: Heun integration steps for the validation rollout.
        monitor_metric: ``"val/psnr"`` or ``"val/ssim"`` (both ``mode="max"``).
        ckpt_path: optional resume checkpoint passed to ``fit``.
    """
    ema = DoubleEMACallback(module)
    # The PSNR/SSIM pipeline carries the LIVE module UNet by reference, so
    # optimizer updates + the EMA swap-in are visible at validation.
    pipeline = PairedLatentFlowPipeline(module.unet, bundle.vae, module.scheduler)
    psnr = PairedPSNRSSIMCallback(
        pipeline=pipeline,
        num_inference_steps=num_inference_steps,
        every_n_epochs=every_n_epochs,
        ema_callback=ema,  # report on the slow-EMA arm (criterion 2)
    )
    callbacks: list = [TrainLossLogger(), LatentX0MAE(), ema, psnr]

    multi_gpu = isinstance(devices, int) and devices > 1
    ckpt = _build_checkpoint(
        model_dir,
        monitor_psnr=not multi_gpu,
        monitor_metric=monitor_metric,
        every_n_epochs=every_n_epochs,
        save_top_k=save_top_k,
    )
    callbacks.append(ckpt)

    datamodule = build_datamodule(
        bundle.latent_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        val_dataset=bundle.val_latent_ds,
    )
    trainer = build_trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    return trainer, ckpt


# -- console entry -----------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manifold-train-paired", description=__doc__.splitlines()[0]
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c", "--train", default="configs/train/config_paired_jit.yaml", help="paired train recipe YAML."
    )
    parser.add_argument("-t", "--network", required=True, help="network construction YAML.")
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs (1 = single-GPU).")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument(
        "--monitor", default="val/psnr", choices=("val/psnr", "val/ssim"), help="checkpoint selection metric."
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist (e.g. diffusion_unet_train.lr=1e-4).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config → warm paired cache → build → ``run_paired_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> _DataBundle`` returning a fake paired latent cache + tiny
    VAE so the full ``main`` path (argparse → compose → build → fit → checkpoint)
    runs without BraTS data or a real VAE encode. The real path warms the paired
    cache via the latent-prep orchestration.
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths
    from ..config.builder import build_scheduler, build_unet

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    require_paths(cfg)
    OmegaConf.resolve(cfg)

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = data_provider(cfg, device) if data_provider is not None else _warm_data(cfg, device)

    train_cfg = cfg.diffusion_unet_train
    num_examples = len(bundle.latent_ds)
    module = PairedLatentFlowModule(
        build_unet(cfg),
        build_scheduler(cfg),
        p_mean=float(opt(cfg.formulation, "p_mean", -0.8)),
        p_std=float(opt(cfg.formulation, "p_std", 0.8)),
        t_eps=float(opt(cfg.formulation, "t_eps", 0.05)),
        loss_weight=str(opt(cfg.formulation, "loss_weight", "1mt_sq")),
        lr=float(train_cfg.lr),
        lr_warmup_steps=int(train_cfg.lr_warmup_steps),
        num_train_examples=num_examples,
        train_batch_size=int(train_cfg.batch_size),
        n_epochs=int(args.max_epochs or train_cfg.n_epochs),
    )

    paired_eval = opt(cfg, "paired_eval", {})
    max_epochs = int(args.max_epochs or train_cfg.n_epochs)
    run_paired_training(
        module=module,
        bundle=bundle,
        model_dir=str(cfg.model_dir),
        max_epochs=max_epochs,
        devices=args.num_gpus if args.num_gpus > 1 else "auto",
        batch_size=int(train_cfg.batch_size),
        seed=seed,
        ckpt_path=args.resume,
        num_inference_steps=int(paired_eval.get("num_inference_steps", 4)),
        every_n_epochs=int(paired_eval.get("every_n_epochs", 1)),
        monitor_metric=args.monitor,
        save_top_k=int(paired_eval.get("save_top_k", 3)),
        limit_val_batches=int(opt(cfg, "val_subset_size", 4)),
    )
    print(f"[manifold-train-paired] done; checkpoints under {cfg.model_dir}")
    return 0


def _train_val_manifests(cfg, manifest):
    """Resolve the (train, val) paired manifests from the configured split mode.

    Two mutually-exclusive modes (mirrors the ``val_data_base_dir`` /
    ``val_fraction`` env-config contract):

    - ``cfg.val_data_base_dir`` set → the **native held-out split**: ``manifest``
      (built from ``data_base_dir``) is the full train set, and val is built from
      ``val_data_base_dir`` — a BraTS directory in the same form as
      ``data_base_dir`` (NOT a manifest JSON; the paired path is BraTS-dir-based
      via :func:`build_brats_pair_manifest`). Use this when the dataset ships its
      own disjoint train/val (e.g. BraTS-2024-GLI's 1621 train / 188 val) — the
      organizer-split subjects are disjoint, so there is no train/val leakage.
    - otherwise → ``cfg.val_fraction`` subject-level split of ``manifest`` (the
      PR #77 path; ``0`` → val=train fallback). A ``null``/``???``/absent
      ``val_data_base_dir`` reads as unset via :func:`~manifold.config.opt`.
    """
    from ..data.paired_brats import build_brats_pair_manifest, split_brats_pair_manifest

    val_dir = opt(cfg, "val_data_base_dir", None)
    if val_dir:
        val_manifest = build_brats_pair_manifest(str(val_dir))
        if not val_manifest:
            raise FileNotFoundError(
                f"No paired BraTS volumes found under val_data_base_dir={val_dir} "
                f"(need ≥1 subject with all 4 contrasts)."
            )
        return manifest, val_manifest
    val_fraction = float(opt(cfg, "val_fraction", 0.0))
    return split_brats_pair_manifest(manifest, val_fraction)


def _warm_data(cfg, device) -> _DataBundle:
    """Warm the real paired latent cache + held VAE (the production data path).

    Resolves the train/val split via :func:`_train_val_manifests`: either the
    native BraTS-2024 train↔val (``val_data_base_dir`` set) or the PR #77
    subject-level held-out split (``val_fraction``). Either way the held-out val
    subjects are disjoint from train, so the per-epoch PSNR/SSIM is measured on
    volumes the model never trains on.
    """
    from ..config import autoencoder_divisor
    from ..data.latent_pipeline import build_encode_pipeline
    from ..data.paired_brats import build_brats_pair_manifest
    from ..data.paired_latent_dataset import (
        PairedLatentDataset,
        estimate_paired_scale_factor,
    )
    from ..data.paired_volume_dataset import PairedNiftiVolumeDataset

    logger = logging.getLogger("manifold.train_paired")
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    divisor = autoencoder_divisor(cfg)

    # The BraTS builder emits the (src, tgt, src_label, tgt_label) manifest; a
    # caller may instead point cfg at a generic paired manifest JSON (future).
    brats_dir = str(cfg.data_base_dir)
    manifest = build_brats_pair_manifest(brats_dir)
    if not manifest:
        raise FileNotFoundError(
            f"No paired BraTS volumes found under data_base_dir={brats_dir} "
            f"(need ≥1 subject with all 4 contrasts)."
        )
    train_manifest, val_manifest = _train_val_manifests(cfg, manifest)
    vol_ds = PairedNiftiVolumeDataset(train_manifest, target_dim=target_dim, divisor=divisor)
    val_dir = opt(cfg, "val_data_base_dir", None)
    split_note = (
        f"val_data_base_dir={val_dir}" if val_dir
        else f"val_fraction={float(opt(cfg, 'val_fraction', 0.0)):.3f}"
    )
    logger.info(
        f"paired manifest: {len(train_manifest)} train / {len(val_manifest)} val "
        f"pairs ({split_note}; {len(vol_ds.unique_sample_ids())} train unique volumes)."
    )

    autoencoder, encode_fn = build_encode_pipeline(cfg, device=device, logger=logger)
    cache_dir = str(opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "paired_latent_cache")))
    latent_ds = PairedLatentDataset(vol_ds, encode_fn=encode_fn, cache_dir=cache_dir, cache_tag="paired_train")
    latent_ds.warm_cache(device, logger=logger, show_progress=True)
    # The held-out val latent dataset. Disjoint subjects → disjoint volumes, so it
    # shares ``cache_dir`` with zero duplicate encoding (each val volume is a cache
    # miss encoded exactly once). ``None`` when ``val_fraction <= 0`` (val=train).
    val_latent_ds: PairedLatentDataset | None = None
    if val_manifest:
        val_vol_ds = PairedNiftiVolumeDataset(val_manifest, target_dim=target_dim, divisor=divisor)
        val_latent_ds = PairedLatentDataset(
            val_vol_ds, encode_fn=encode_fn, cache_dir=cache_dir, cache_tag="paired_train"
        )
        val_latent_ds.warm_cache(device, logger=logger, show_progress=True)
        val_latent_ds.free_encoder()
    latent_ds.free_encoder()

    autoencoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Estimate the scale over TRAIN unique latents only (no val leakage); propagate
    # the same scale-on-read multiplier to the val dataset so its decode is consistent.
    estimate_paired_scale_factor(
        latent_ds, autoencoder, sample_size=int(opt(cfg, "val_subset_size", 64)), logger=logger
    )
    if val_latent_ds is not None:
        val_latent_ds.scaling_factor = latent_ds.scaling_factor
    return _DataBundle(latent_ds=latent_ds, vae=autoencoder, val_latent_ds=val_latent_ds)
