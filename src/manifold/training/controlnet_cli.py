"""``manifold-train-controlnet`` console entry + the testable supervised-ControlNet core.

The console entry (issue #141 / ADR-0027 stage 1) composes the OmegaConf experiment
config, builds the :class:`ControlNetLatentFlowModule` (a **frozen** noise→data JiT
base UNet + a **trainable** ControlNet) + the paired inputs (real BraTS paired
latent cache warmed through the JiT export's VAE, or a fake via the
``data_provider`` injection seam for the CPU smoke), and calls ``Trainer.fit``. The
supervised ControlNet job is the first stage of the two-stage ControlNet pipeline:
train the ControlNet on the frozen base to translate ``x_src`` → ``x_tgt`` before
any GRPO (stage 2 runs via ``manifold-train-grpo --grpo-mode 2``).

The integration core :func:`run_controlnet_training` (Module + datamodule +
callbacks + ``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a
tiny CPU smoke can drive it with a fake base + ControlNet + toy paired batches (the
issue's testing seam) instead of the real JiT checkpoint + BraTS data. The
real-data launch path (loading the frozen JiT base from ``--native-dir``, warming
the paired cache from ``--latents-dir``) is gated on those artifacts existing on
the cluster; the ``data_provider`` seam ships here for the CPU smoke.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ImportError:  # pragma: no cover
    from pytorch_lightning.utilities.rank_zero import rank_zero_info  # type: ignore

from ..config import opt
from ..data.datamodule import build_datamodule
from ..modules.controlnet_latent_flow import ControlNetLatentFlowModule
from .metrics import LatentX0MAE, TrainLossLogger
from .trainer import build_trainer


@dataclass
class ControlNetInputs:
    """Module-construction + data inputs for one supervised ControlNet run.

    ``unet`` / ``controlnet`` / ``scheduler`` go to the
    :class:`ControlNetLatentFlowModule` ctor; ``train_ds`` / ``val_ds`` emit paired
    batches (``{src_latent, tgt_latent, src_label, tgt_label, spacing}`` — both
    latents scaled). The frozen base is held unregistered by the Module (off the
    optimizer/checkpoint); only the ControlNet is optimized. ``vae`` carries the
    ``scaling_factor`` the export pipeline round-trips; the ``data_provider`` seam
    injects a fake base + ControlNet + toy pairs for the CPU smoke.
    """

    unet: Any
    controlnet: Any
    scheduler: Any
    train_ds: Any
    val_ds: Any
    vae: Any = None
    #: Cold-path warm closure (issue #145 / ADR-0017): when set, ``train_ds`` /
    #: ``val_ds`` are UN-warmed and the paired latent-cache warm runs inside
    #: ``DataModule.setup()`` (post-DDP-spawn, per-rank sharded) instead of in the
    #: CLI before ``fit``. ``None`` ⇒ the datasets are already warmed (the
    #: ``data_provider`` smoke / single-GPU parity path).
    warm_fn: Any = None


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/x0_mae",
    save_top_k: int = 3,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` for the supervised ControlNet (ADR-0006).

    Monitors ``val/x0_mae`` (the latent-space target-prediction MAE, mode ``min`` —
    the raw optimizer arm, no EMA). The metric is the same callback the JiT cli uses
    (the ControlNet's validation forward returns ``pred`` / ``target`` just like
    JiT's), and it is **globally reduced** under DDP: :class:`LatentX0MAE` accumulates
    into a sample-weighted ``torchmetrics.MeanMetric`` and logs the Metric object, so
    Lightning fires the cross-rank reduction to the true global mean (issue #146). A
    rank-local fallback (``save_top_k=1`` + ``save_last``) would throw away
    best-checkpoint selection for no correctness gain, so the monitor stays ON under
    DDP. ``auto_insert_metric_name = False`` because the key contains a ``/``.
    """
    return ModelCheckpoint(
        dirpath=model_dir,
        filename=f"controlnet-{{epoch:03d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=save_top_k,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


def run_controlnet_training(
    *,
    module: ControlNetLatentFlowModule,
    inputs: ControlNetInputs,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    save_top_k: int = 3,
    seed: int = 0,
    ckpt_path: str | None = None,
    limit_val_batches: int | float = 1.0,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the ControlNet module (the core seam).

    Builds the train-metrics + latent-x0-MAE callbacks + a stock
    ``ModelCheckpoint`` and runs ``Trainer.fit`` on the paired datamodule. Returns
    ``(trainer, ckpt)`` so callers can find the written ``.ckpt``.

    Args:
        inputs: the frozen base + ControlNet + scheduler + the paired train/val datasets.
            When ``inputs.warm_fn`` is set the datasets are UN-warmed and the paired
            latent-cache warm runs inside ``DataModule.setup()`` (issue #145 — post-PG,
            per-rank sharded); otherwise they are pre-warmed (the smoke/parity path).
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    pl.seed_everything(seed, workers=True)
    callbacks: list[pl.Callback] = [TrainLossLogger(), LatentX0MAE()]
    # val/x0_mae is globally reduced under DDP (LatentX0MAE logs a sample-weighted
    # MeanMetric), so the monitored checkpoint stays on under multi-GPU (issue #146)
    # — no save_top_k=1 degradation.
    ckpt = _build_checkpoint(
        model_dir,
        monitor_metric="val/x0_mae",
        save_top_k=save_top_k,
    )
    callbacks.append(ckpt)
    if inputs.warm_fn is not None:
        # Cold path (issue #145 / ADR-0017): defer the paired latent-cache warm into
        # DataModule.setup() so it runs per-rank after DDP spawn (the warm_cache
        # sharded branch activates), instead of every rank re-warming pre-fit. The
        # datasets in ``inputs`` are un-warmed placeholders; ``warm_fn`` returns the
        # warmed (train_ds, val_ds, vae) triple (the PairedWarmDataModule contract).
        from ..data.warm_datamodule import PairedWarmDataModule

        def _warm_triple():
            train_ds, val_ds = inputs.warm_fn()
            return train_ds, val_ds, inputs.vae

        datamodule = PairedWarmDataModule(
            latent_ds=inputs.train_ds,
            vae=inputs.vae,
            batch_size=batch_size,
            num_workers=num_workers,
            val_latent_ds=inputs.val_ds,
            warm_fn=_warm_triple,
        )
    else:
        datamodule = build_datamodule(
            inputs.train_ds,
            batch_size=batch_size,
            val_dataset=inputs.val_ds,
            num_workers=num_workers,
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
        prog="manifold-train-controlnet",
        description=(
            "Supervised ControlNet on the frozen JiT base (ADR-0027 stage 1): "
            "paired MRI x_src -> x_tgt translation."
        ),
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c",
        "--train",
        default="configs/train/config_controlnet_supervised.yaml",
        help="ControlNet supervised recipe YAML.",
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (the base UNet architecture)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--native-dir",
        default=None,
        help="JiT native export dir (the frozen base UNet + VAE scale); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--latents-dir",
        default=None,
        help="paired latent cache dir (warmed through the JiT VAE); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config -> build -> ``run_controlnet_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> ControlNetInputs`` returning a fake base + ControlNet + toy
    paired batches so the full ``main`` path runs without the real JiT checkpoint
    or BraTS data. The real path loads the frozen base from ``--native-dir`` and
    warms the paired cache from ``--latents-dir``.
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, opt, require_paths

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # The base UNet comes from --native-dir; only the paired data_dir + output
    # model_dir are required paths here.
    require_paths(cfg, keys=("data_base_dir", "model_dir"))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "diffusion_unet_train", None) is None:
        raise ValueError(
            "Config has no `diffusion_unet_train` block — use the ControlNet supervised "
            "recipe (-c configs/train/config_controlnet_supervised.yaml), not a JiT/GRPO config."
        )

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        inputs = data_provider(cfg, device)
    else:
        # --native-dir / --latents-dir are NOT argparse-required: that would break
        # the data_provider injection seam (the CPU smoke). Validate them here, only
        # on the real path.
        if not args.native_dir or not args.latents_dir:
            raise ValueError(
                "ControlNet supervised training needs --native-dir <JiT native export "
                "(frozen base UNet + VAE)> and --latents-dir <paired latent cache> "
                "(or inject a data_provider for the smoke)."
            )
        inputs = _real_inputs(cfg, args.native_dir, args.latents_dir, device)

    train_cfg = cfg.diffusion_unet_train
    module = ControlNetLatentFlowModule(
        inputs.unet,
        inputs.controlnet,
        inputs.scheduler,
        p_mean=float(opt(cfg.formulation, "p_mean", -0.8)),
        p_std=float(opt(cfg.formulation, "p_std", 0.8)),
        t_eps=float(opt(cfg.formulation, "t_eps", 0.05)),
        l1_weight=float(opt(cfg.formulation, "l1_weight", 0.0)),
        lr=float(train_cfg.lr),
        lr_warmup_steps=int(train_cfg.lr_warmup_steps),
        lr_ref_batch_size=int(opt(train_cfg, "lr_ref_batch_size", 8)),
        lr_scale_rule=str(opt(train_cfg, "lr_scale_rule", "sqrt")),
        lr_warmup_ratio=opt(train_cfg, "lr_warmup_ratio", None),
        num_train_examples=len(inputs.train_ds),
        train_batch_size=int(train_cfg.batch_size),
        n_epochs=int(args.max_epochs or train_cfg.n_epochs),
    )

    run_controlnet_training(
        module=module,
        inputs=inputs,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or train_cfg.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(train_cfg.batch_size),
        save_top_k=int(opt(cfg, "checkpoint.save_top_k", 3)),
        seed=seed,
        ckpt_path=args.resume,
    )
    print(f"[manifold-train-controlnet] done; checkpoints under {cfg.model_dir}")
    return 0


def _real_inputs(
    cfg, native_dir: str, latents_dir: str, device: torch.device
) -> ControlNetInputs:
    """Build the real supervised-ControlNet inputs from the JiT export + paired cache.

    Loads the **frozen base** noise→data JiT UNet from ``--native-dir`` (the raw-arm
    native export; its VAE carries ``scaling_factor``), builds a **fresh** ControlNet
    that clones the base encoder (zero-conv init ⇒ initial residuals are zero ⇒ the
    model starts as the pretrained base unchanged, ADR-0026), resolves the **paired**
    train/val split (``_train_val_manifests`` / ``val_data_base_dir`` /
    ``val_fraction``), and DEFERS the paired latent-cache warm into
    ``DataModule.setup()`` (issue #145 / ADR-0017 — post-DDP-spawn, per-rank sharded).
    Scale-on-read uses the BASE EXPORT's ``scaling_factor`` verbatim (ADR-0021).
    """
    from ..config import autoencoder_divisor
    from ..config.builder import build_controlnet, build_scheduler
    from ..data.latent_pipeline import make_encode_fn, resolve_warm_device
    from ..data.paired_brats import build_brats_pair_manifest
    from ..data.paired_latent_dataset import PairedLatentDataset, paired_cache_tag
    from ..data.paired_volume_dataset import PairedNiftiVolumeDataset
    from ..pipelines.latent_flow import LatentFlowPipeline
    from .paired_reward_cli import _train_val_manifests

    # 1. Frozen base UNet (the raw-arm JiT policy) + VAE (carries scaling_factor).
    base_pipe = LatentFlowPipeline.from_pretrained(str(native_dir))
    base = base_pipe.unet.to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    vae = base_pipe.vae

    # 2. Fresh ControlNet — a zero-conv clone of the base encoder (the only trained arm).
    controlnet = build_controlnet(cfg).to(device)
    controlnet.load_base_encoder_weights(base)
    scheduler = build_scheduler(cfg)

    # 3. Paired BraTS manifest + the train/val split (reuses the paired-reward split
    # logic: native held-out dir, else a subject-level val_fraction).
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    divisor = autoencoder_divisor(cfg)
    brats_dir = str(cfg.data_base_dir)
    manifest = build_brats_pair_manifest(brats_dir)
    if not manifest:
        raise FileNotFoundError(
            f"No paired BraTS volumes found under data_base_dir={brats_dir} "
            f"(need >=1 subject with all 4 contrasts)."
        )
    # Mirror the recipe's controlnet.val_fraction to the root val_fraction read by
    # _train_val_manifests, ONLY when the native-split DIRECTORY path is not taken
    # and the root key is unset (an explicit root override wins) — mirrors
    # paired_reward_cli._real_inputs (codex #99/#100).
    val_dir = opt(cfg, "val_data_base_dir", None)
    if not (val_dir and os.path.isdir(str(val_dir))):
        from omegaconf import OmegaConf

        if OmegaConf.select(cfg, "val_fraction", default=None) is None:
            cn_val_fraction = float(opt(cfg, "controlnet.val_fraction", 0.0))
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"val_fraction": cn_val_fraction}))
    train_manifest, val_manifest = _train_val_manifests(cfg, manifest)
    if not val_manifest:
        raise ValueError(
            "ControlNet supervised training needs a held-out val split "
            "(val_data_base_dir set, or val_fraction > 0); train data is never "
            "reused as val."
        )

    # 4. Defer the paired latent-cache warm into DataModule.setup() (issue #145 /
    # ADR-0017): build the UN-warmed datasets + a warm closure; ``setup()`` runs it
    # post-DDP-spawn so the warm_cache per-rank sharded branch activates (one writer
    # per cache file) instead of every rank re-warming pre-fit. Scale-on-read uses
    # the BASE EXPORT's scaling_factor VERBATIM (ADR-0021): the frozen base was
    # trained on latents scaled by that factor, so re-estimating it from this paired
    # subset would move both src+tgt into a normalization space the base never saw.
    cache_dir = str(
        latents_dir
        or opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "paired_latent_cache"))
    )
    # Fold the encode geometry (target_dim / divisor) into the cache tag (issue #147):
    # the disk filename is keyed by sample_id (path-derived, geometry-free), so a plain
    # tag would silently reuse a stale cache encoded at a different shape/divisor.
    # The geometry suffix makes a config change produce a disjoint cache entry.
    cache_tag = paired_cache_tag(str(opt(cfg, "controlnet.cache_tag", "paired_train")), target_dim, divisor)
    scaling_factor = float(vae.scaling_factor)  # the base export's factor (verbatim)

    def _ds(manifest_split):
        vol_ds = PairedNiftiVolumeDataset(manifest_split, target_dim=target_dim, divisor=divisor)
        return PairedLatentDataset(vol_ds, encode_fn=None, cache_dir=cache_dir, cache_tag=cache_tag)

    train_ds, val_ds = _ds(train_manifest), _ds(val_manifest)

    def warm_fn():
        # Runs INSIDE DataModule.setup() (post-PG). Warm on the per-rank local device,
        # not the launch-time ``device`` (cuda:0 before LOCAL_RANK is known under DDP);
        # rebuild the encode_fn bound to it. Each rank encodes its ``i % world == rank``
        # shard, barriers, then loads the full set (one encode per unique volume,
        # ADR-0014 — disjoint sample_ids ⇒ free disk hits across splits).
        warm_device = resolve_warm_device(device)
        vae.to(warm_device)
        encode_fn = make_encode_fn(vae, warm_device, cfg)
        for ds in (train_ds, val_ds):
            ds.encode_fn = encode_fn
            ds.scaling_factor = scaling_factor  # the base export's factor (ADR-0021)
            ds.warm_cache(warm_device, show_progress=False)
            # Free the encoder closure and move the VAE off the training GPU: supervised
            # validation is latent-only (no decode), so the VAE is only needed for the
            # warm; a lingering GPU VAE can turn a feasible 3D run into an OOM.
            ds.free_encoder()
        vae.to("cpu")
        rank_zero_info(
            "ControlNet supervised: %d train / %d val paired pairs; base frozen, "
            "ControlNet trainable (fresh zero-conv clone); scale_factor=%.6f (base export).",
            len(train_ds), len(val_ds), scaling_factor,
        )
        return train_ds, val_ds

    return ControlNetInputs(
        unet=base,
        controlnet=controlnet,
        scheduler=scheduler,
        train_ds=train_ds,
        val_ds=val_ds,
        vae=vae,
        warm_fn=warm_fn,
    )
