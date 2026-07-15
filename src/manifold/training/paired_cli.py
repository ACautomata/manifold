"""``manifold-train-paired`` console entry + the testable paired-training core.

The console entry (issue #69) composes the OmegaConf experiment config, warms the
paired latent cache (Slice 2), builds the :class:`~manifold.modules.PairedLatentFlowModule`
+ the fixed validation subset + the callbacks (train metrics,
PSNR/SSIM, ``ModelCheckpoint``), and calls ``Trainer.fit``. From scratch
(ADR-0014 — no warm-start). Sibling of ``manifold.training.cli``; the heavy
data-warming lives in :func:`main`, the integration core :func:`run_paired_training`
(Module + datamodule + callbacks + ``ModelCheckpoint`` + ``build_trainer`` +
``fit``) is split out so a tiny CPU smoke can drive it with a fake latent cache
(the issue's testing seam) instead of BraTS + a real VAE.

Best-checkpoint selection monitors ``val/psnr`` (``mode="max"``); under DDP the metric is GLOBAL (all ranks decode + all_reduce the per-volume sums, ADR-0025), so the monitor stays on under multi-GPU too
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
from .metrics import LatentX0MAE, TrainLossLogger
from .trainer import build_trainer

_log = logging.getLogger(__name__)


@dataclass
class _DataBundle:
    """The warmed paired-data bundle ``main`` passes into :func:`run_paired_training`.

    (Injection seam for the CPU smoke test, which feeds a fake paired latent cache
    + a tiny VAE instead of warming BraTS through a real VAE encode.)

    ``val_latent_ds`` is an optional subject-level held-out validation dataset.
    When absent (and not the smoke opt-in) :func:`run_paired_training` DISABLES
    validation rather than reuse train (val/train leakage); ``has_val`` carries
    the cold-path split decision, ``allow_train_as_val`` the smoke opt-in.
    """

    latent_ds: Any = None
    vae: Any = None
    val_latent_ds: Any = None
    warm_fn: Any = None
    has_val: bool | None = None
    allow_train_as_val: bool = False


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/psnr",
    save_top_k: int = 3,
    monitor: bool = True,
    every_n_epochs: int = 1,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` (ADR-0006), monitoring ``val/psnr``.

    ``val/psnr`` is GLOBAL under DDP (all ranks decode + all_reduce the per-volume sums, ADR-0025), so the monitor stays on under multi-GPU. ``mode="max"``, top-k, last, full state; the raw-optimizer metric is sufficient on a short from-scratch run.
    ``auto_insert_metric_name = False`` because the metric key contains a ``/``.

    ``monitor=False`` (no held-out val -> validation disabled): no metric to
    monitor, so keep ``save_last`` + a periodic ``every_n_epochs`` checkpoint.
    """
    if not monitor:
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
    check_val_every_n_epoch: int | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the paired module (the core seam).

    Builds the train-metrics / PSNR-SSIM callbacks + a stock
    ``ModelCheckpoint`` and runs ``Trainer.fit``. Returns ``(trainer, ckpt)`` so
    callers can find the written ``.ckpt``.

    Args:
        bundle: the warmed paired latent dataset + held VAE.
        num_inference_steps: Heun integration steps for the validation rollout.
        monitor_metric: ``"val/psnr"`` or ``"val/ssim"`` (both ``mode="max"``).
        ckpt_path: optional resume checkpoint passed to ``fit``.
        check_val_every_n_epoch: when set (and validation is enabled), forward
            ``check_val_every_n_epoch`` + ``num_sanity_val_steps=0`` to the Trainer so
            validation runs only every N epochs — e.g. ``=max_epochs`` yields a single
            end-of-training val pass (the autoresearch "train fully, then val once" mode).
            ``None`` (default) keeps Lightning's per-epoch validation cadence.
    """
    # Validation requires a held-out val split. When none is configured
    # (val_fraction=0 / no val_data_base_dir AND not the smoke opt-in), validation
    # is DISABLED rather than silently reuse train - val/psnr would otherwise be a
    # train metric (val/train leakage). ``has_val`` is the resolved split decision
    # (cold path: from the val manifest; warmed path: inferred from val_latent_ds).
    has_val = bundle.has_val if bundle.has_val is not None else (bundle.val_latent_ds is not None)
    val_enabled = has_val or bundle.allow_train_as_val
    callbacks: list = [TrainLossLogger(), LatentX0MAE()]
    if val_enabled:
        # The PSNR/SSIM pipeline carries the LIVE module UNet by reference, so
        # optimizer updates are visible at validation.
        pipeline = PairedLatentFlowPipeline(module.unet, bundle.vae, module.scheduler)
        # When last-epoch-only val is active (check_val_every_n_epoch set), Lightning
        # already gates WHEN validation runs. The callback's own ``every_n_epochs`` is
        # then redundant + harmful: its 0-based ``epoch % every_n_epochs`` check could
        # SKIP the single final-epoch pass when every_n_epochs>1 (e.g. epoch 19 % 5),
        # leaving no val/psnr — the run finishes with no ranking metric. Force it to 1
        # so the decode always runs whenever Lightning validates (codex #91).
        if check_val_every_n_epoch is not None and every_n_epochs > 1:
            _log.warning(
                "paired last-epoch-only val (check_val_every_n_epoch=%s) ignores "
                "paired_eval.every_n_epochs=%d — forcing the PSNR/SSIM cadence to 1 so "
                "the callback does not skip the single final-epoch validation pass.",
                check_val_every_n_epoch,
                every_n_epochs,
            )
        psnr = PairedPSNRSSIMCallback(
            pipeline=pipeline,
            num_inference_steps=num_inference_steps,
            every_n_epochs=1 if check_val_every_n_epoch is not None else every_n_epochs,
        )
        callbacks.append(psnr)
        ckpt = _build_checkpoint(
            model_dir,
            monitor_metric=monitor_metric,
            save_top_k=save_top_k,
            # val/psnr is now a GLOBAL metric (all ranks decode + all_reduce the
            # per-volume sums; ADR-0025), so best-by-PSNR selection stays on under
            # multi-GPU too (no more rank-0-shard workaround).
            monitor=True,
            every_n_epochs=every_n_epochs,
        )
    else:
        _log.warning(
            "manifold-train-paired: no held-out validation split configured "
            "(val_fraction=0 / no val_data_base_dir). Validation is DISABLED - "
            "val/psnr will not be logged. Reusing the train set as val would leak "
            "train metrics into validation, which is refused; hold out subjects "
            "(val_fraction>0) or set val_data_base_dir to enable."
        )
        ckpt = _build_checkpoint(
            model_dir, save_top_k=save_top_k, monitor=False, every_n_epochs=every_n_epochs
        )
    callbacks.append(ckpt)

    # F2/F4 (ADR-0017): the paired warm runs in DataModule.setup() (post-PG,
    # per-rank sharded) when bundle.warm_fn is set; the warmed test path makes
    # setup() a no-op.
    from ..data.warm_datamodule import PairedWarmDataModule

    datamodule = PairedWarmDataModule(
        latent_ds=bundle.latent_ds,
        vae=bundle.vae,
        batch_size=batch_size,
        num_workers=num_workers,
        val_latent_ds=bundle.val_latent_ds,
        warm_fn=bundle.warm_fn,
        allow_train_as_val=bundle.allow_train_as_val,
    )
    # When validation is disabled, ``limit_val_batches=0`` makes every validation
    # epoch a 0-batch no-op (the empty val_dataloader yields nothing) and
    # ``num_sanity_val_steps=0`` skips the fit-start sanity probes; the PSNR
    # callback is not attached, so no ``val/*`` metric is logged. (Do NOT also pass
    # ``check_val_every_n_epoch=None`` - Lightning's contract then requires an
    # integer ``val_check_interval``, which the float default violates.)
    extra_kwargs: dict | None = None
    if not val_enabled:
        extra_kwargs = {"num_sanity_val_steps": 0}
    elif check_val_every_n_epoch is not None:
        # Last-epoch-only val (autoresearch): validate ONLY every N epochs instead
        # of every epoch. ``check_val_every_n_epoch=max_epochs`` makes Lightning run
        # a single validation pass at the final epoch; paired with
        # ``num_sanity_val_steps=0`` (skip the 2-batch pre-training probe), the
        # PSNR/SSIM callback decodes exactly once — after all training. The PSNR
        # callback's own ``every_n_epochs`` is left at 1 so it runs whenever
        # Lightning validation runs.
        extra_kwargs = {
            "check_val_every_n_epoch": int(check_val_every_n_epoch),
            "num_sanity_val_steps": 0,
        }
    trainer = build_trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches if val_enabled else 0,
        extra_kwargs=extra_kwargs,
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
        "-c",
        "--train",
        default="configs/train/config_paired_jit.yaml",
        help="paired train recipe YAML.",
    )
    parser.add_argument("-t", "--network", required=True, help="network construction YAML.")
    parser.add_argument(
        "-g", "--num-gpus", type=int, default=1, help="number of GPUs (1 = single-GPU)."
    )
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument(
        "--monitor",
        default="val/psnr",
        choices=("val/psnr", "val/ssim"),
        help="checkpoint selection metric.",
    )
    parser.add_argument(
        "overrides", nargs="*", help="Hydra-style dotlist (e.g. diffusion_unet_train.lr=1e-4)."
    )
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

    if data_provider is not None:
        # Test smoke: the provider returns a WARMED bundle (latent_ds already set).
        bundle = data_provider(cfg, device)
        num_examples = len(bundle.latent_ds) if bundle.latent_ds is not None else 0
    else:
        # Production cold path (ADR-0017): warm deferred to DataModule.setup();
        # num_examples = len(vol_ds) (pre-warm).
        bundle, num_examples = _warm_data(cfg, device)

    train_cfg = cfg.diffusion_unet_train
    module = PairedLatentFlowModule(
        build_unet(cfg),
        build_scheduler(cfg),
        p_mean=float(opt(cfg.formulation, "p_mean", -0.8)),
        p_std=float(opt(cfg.formulation, "p_std", 0.8)),
        t_eps=float(opt(cfg.formulation, "t_eps", 0.05)),
        loss_weight=str(opt(cfg.formulation, "loss_weight", "1mt_sq")),
        lr=float(train_cfg.lr),
        lr_warmup_steps=int(train_cfg.lr_warmup_steps),
        lr_ref_batch_size=int(opt(train_cfg, "lr_ref_batch_size", 8)),
        lr_scale_rule=str(opt(train_cfg, "lr_scale_rule", "sqrt")),
        lr_warmup_ratio=opt(train_cfg, "lr_warmup_ratio", None),
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
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(train_cfg.batch_size),
        seed=seed,
        ckpt_path=args.resume,
        num_inference_steps=int(paired_eval.get("num_inference_steps", 4)),
        every_n_epochs=int(paired_eval.get("every_n_epochs", 1)),
        monitor_metric=args.monitor,
        save_top_k=int(paired_eval.get("save_top_k", 3)),
        limit_val_batches=int(opt(cfg, "val_subset_size", 4)),
        check_val_every_n_epoch=paired_eval.get("check_val_every_n_epoch", None),
    )
    print(f"[manifold-train-paired] done; checkpoints under {cfg.model_dir}")
    return 0


def _train_val_manifests(cfg, manifest):
    """Resolve the (train, val) paired manifests from the configured split mode.

    Two mutually-exclusive modes (mirrors the ``val_data_base_dir`` /
    ``val_fraction`` env-config contract):

    - ``cfg.val_data_base_dir`` set AND an existing directory → the **native
      held-out split**: ``manifest`` (built from ``data_base_dir``) is the full
      train set, and val is built from ``val_data_base_dir`` — a BraTS directory
      in the same form as ``data_base_dir`` (NOT a manifest JSON; the paired path
      is BraTS-dir-based via :func:`build_brats_pair_manifest`). Use this when the
      dataset ships its own disjoint train/val (e.g. BraTS-2024-GLI's 1621 train /
      188 val) — the organizer-split subjects are disjoint, so there is no
      train/val leakage. A non-directory ``val_data_base_dir`` (e.g. the manifest
      JSON the BraTS2023 profile sets) is ignored with a warning and falls back to
      ``val_fraction`` (the pre-native-split behavior).
    - otherwise → ``cfg.val_fraction`` subject-level split of ``manifest`` (the
      PR #77 path; ``0`` → val=train fallback). A ``null``/``???``/absent
      ``val_data_base_dir`` reads as unset via :func:`~manifold.config.opt`.
    """
    from ..data.paired_brats import build_brats_pair_manifest, split_brats_pair_manifest

    val_dir = opt(cfg, "val_data_base_dir", None)
    # The native-split path needs a BraTS *directory* (build_brats_pair_manifest
    # scans NIfTIs); a manifest JSON (e.g. the BraTS2023 profile's
    # brats_all_val.json) or a missing path is not usable here. Fall back to
    # val_fraction (the pre-native-split behavior) instead of building an empty
    # val set and crashing (codex #78, P1).
    if val_dir and os.path.isdir(str(val_dir)):
        val_manifest = build_brats_pair_manifest(str(val_dir))
        if not val_manifest:
            raise FileNotFoundError(
                f"No paired BraTS volumes found under val_data_base_dir={val_dir} "
                f"(need ≥1 subject with all 4 contrasts)."
            )
        return manifest, val_manifest
    if val_dir:
        _log.warning(
            "paired val_data_base_dir=%s is not a directory; the native train/val "
            "split needs a BraTS directory (not a manifest JSON). Falling back to "
            "the val_fraction subject split.",
            val_dir,
        )
    val_fraction = float(opt(cfg, "val_fraction", 0.0))
    return split_brats_pair_manifest(manifest, val_fraction)


def _warm_data(cfg, device) -> tuple[_DataBundle, int]:
    """Warm the real paired latent cache + held VAE (the production data path).

    Resolves the train/val split via :func:`_train_val_manifests`: either the
    native BraTS-2024 train↔val (``val_data_base_dir`` set) or the PR #77
    subject-level held-out split (``val_fraction``). Either way the held-out val
    subjects are disjoint from train, so the per-epoch PSNR/SSIM is measured on
    volumes the model never trains on.
    """
    from ..config import autoencoder_divisor
    from ..data.latent_pipeline import build_encode_pipeline, make_encode_fn, resolve_warm_device
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
    has_val = bool(val_manifest)
    vol_ds = PairedNiftiVolumeDataset(train_manifest, target_dim=target_dim, divisor=divisor)
    val_dir = opt(cfg, "val_data_base_dir", None)
    split_note = (
        f"val_data_base_dir={val_dir}"
        if val_dir
        else f"val_fraction={float(opt(cfg, 'val_fraction', 0.0)):.3f}"
    )
    logger.info(
        f"paired manifest: {len(train_manifest)} train / {len(val_manifest)} val "
        f"pairs ({split_note}; {len(vol_ds.unique_sample_ids())} train unique volumes)."
    )
    if not has_val:
        logger.warning(
            "paired: no held-out val split resolved (%s) - validation will be "
            "DISABLED (val/psnr not logged). Hold out subjects (val_fraction>0) or "
            "set val_data_base_dir to enable; train data is never reused as val.",
            split_note,
        )

    # Build the VAE on CPU pre-PG (P1/ADR-0017): the launch-time ``device`` is the
    # default cuda:0 before LOCAL_RANK is known, so loading on it under DDP would
    # place every rank's encoder on GPU 0. The warm re-stages it onto the per-rank
    # local GPU inside setup(); PSNR's decode stages it to the UNet device at eval.
    # The CPU encode_fn built here is unused on the cold path (rebuilt in warm_fn).
    autoencoder, _cpu_encode_fn = build_encode_pipeline(
        cfg, device=torch.device("cpu"), logger=logger
    )
    cache_dir = str(
        opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "paired_latent_cache"))
    )
    val_subset_size = int(opt(cfg, "val_subset_size", 64))

    def warm_fn():
        # F2/F3 (ADR-0017): both warm calls run here (post-PG, inside setup()) so
        # PairedLatentDataset.warm_cache derives rank/world from dist -> the
        # sharded branch activates (one writer per unique-volume cache file).
        # P1: warm on the per-rank local CUDA device (resolve_warm_device), rebuild
        # the encode_fn bound to it so the sliding-window predictor runs on the
        # right GPU (the launch-time ``device`` is cuda:0 before LOCAL_RANK).
        nonlocal autoencoder
        warm_device = resolve_warm_device(device)
        autoencoder.to(warm_device)
        encode_fn = make_encode_fn(autoencoder, warm_device, cfg)
        latent_ds = PairedLatentDataset(
            vol_ds, encode_fn=encode_fn, cache_dir=cache_dir, cache_tag="paired_train"
        )
        latent_ds.warm_cache(warm_device, logger=logger, show_progress=True)
        val_latent_ds = None
        if val_manifest:
            val_vol_ds = PairedNiftiVolumeDataset(
                val_manifest, target_dim=target_dim, divisor=divisor
            )
            val_latent_ds = PairedLatentDataset(
                val_vol_ds, encode_fn=encode_fn, cache_dir=cache_dir, cache_tag="paired_train"
            )
            val_latent_ds.warm_cache(warm_device, logger=logger, show_progress=True)
            val_latent_ds.free_encoder()
        latent_ds.free_encoder()
        autoencoder.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        estimate_paired_scale_factor(
            latent_ds, autoencoder, sample_size=val_subset_size, logger=logger
        )
        if val_latent_ds is not None:
            val_latent_ds.scaling_factor = latent_ds.scaling_factor
        # Persist the estimated scale so the paired-reward export can bake it into
        # the frozen generator's VAE: the reward pairs scale src latents by this
        # value, and the paired module holds no VAE (so the scale is otherwise lost
        # with the process). scripts/export_checkpoint.py reads it (via
        # torch.load(..., weights_only=True)) for --pipeline paired (codex #98 P1).
        # Rank-0 only: warm_fn runs on every DDP rank, and unlike the latent-cache
        # writes (one-per-file, sharded) this is a single shared path -> concurrent
        # writes would corrupt it (codex #100 P2).
        import torch.distributed as dist

        is_main = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_main:
            os.makedirs(str(cfg.model_dir), exist_ok=True)
            scale_path = os.path.join(str(cfg.model_dir), "paired_scaling_factor.pt")
            torch.save(torch.tensor(float(autoencoder.scaling_factor)), scale_path)
            logger.info(
                f"Persisted paired scaling_factor={float(autoencoder.scaling_factor):.6f}"
                f" -> {scale_path}"
            )
        if dist.is_initialized():
            dist.barrier()
        return latent_ds, val_latent_ds, autoencoder

    return _DataBundle(vae=autoencoder, warm_fn=warm_fn, has_val=has_val), len(vol_ds)
