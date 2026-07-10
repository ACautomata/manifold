"""``manifold-train`` console entry + the testable training-orchestration core.

The console entry (issue #28) composes the OmegaConf experiment config, warms
the latent cache, builds the Module + the fixed validation subset + the
callbacks (train metrics, double EMA, FID, ``ModelCheckpoint``), and calls
``Trainer.fit``. From-scratch by default; an optional warm-start path.

The heavy data-warming lives in :func:`main`; the integration core
:func:`run_training` (Module + datamodule + callbacks + ``ModelCheckpoint`` +
``build_trainer`` + ``fit``) is split out so a tiny CPU smoke can drive it with a
fake latent cache (the issue's testing seam) instead of BraTS + a real VAE.
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
from ..metrics import FIDCallback
from ..modules.latent_flow import LatentFlowModule
from .ema import DoubleEMACallback
from .metrics import LatentX0MAE, TrainLossLogger
from .trainer import build_trainer, is_multi_gpu

_log = logging.getLogger(__name__)


@dataclass
class _DataBundle:
    """The warmed-data bundle ``main`` passes into :func:`run_training`.

    (Injection seam for the CPU smoke test, which feeds a fake latent cache + a
    tiny VAE instead of warming BraTS through a real VAE encode.)

    ADR-0017 / F4 (issue #84): the VAE-encode warm now runs in
    ``DataModule.setup()`` (post-PG). Two modes:
    - **warmed** (the test smoke): ``latent_ds`` + ``val_latents`` already set,
      ``warm_fn=None`` -> ``setup()`` is a no-op (single-GPU parity preserved).
    - **cold** (production): ``latent_ds=None`` + ``warm_fn`` set (a closure over
      :func:`warm_latent_pipeline`) -> ``setup()`` warms post-PG and computes
      ``val_latents``. The Module is sized from ``num_examples`` (passed separately,
      ``= len(vol_ds)``) so it does not need ``len(latent_ds)`` pre-warm.
    """

    latent_ds: Any = None
    vae: Any = None
    val_latents: torch.Tensor | None = None
    warm_fn: Any = None


def _dict_subset(d: dict | None, keys: tuple[str, ...]) -> dict:
    """Extract the non-``None`` values from *d* for the given *keys*."""
    if not d:
        return {}
    return {k: d[k] for k in keys if d.get(k) is not None}


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_fid: bool,
    monitor_metric: str = "val/fid_raw",
    every_n_epochs: int = 1,
    save_top_k: int = 3,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` (ADR-0006).

    Single-GPU + FID: monitor ``monitor_metric`` (top-k, last, full state). The
    default ``val/fid_raw`` (raw-optimizer arm) tracks whether the model is
    actually learning — the slow-EMA ``val/fid_avg`` lags on short from-scratch
    runs (a 0.9999 EMA is still mostly init well before epoch 50). Callers that
    disable the raw arm (``log_raw_fid=False``) must pass ``val/fid_avg`` here so
    the monitor matches a metric that is actually logged. Under DDP (FID is
    rank-0-only, so the metric is not global) fall back to ``save_last`` +
    ``every_n_epochs`` with no monitor. ``auto_insert_metric_name = False``
    because the metric key contains a ``/``.
    """
    if monitor_fid:
        return ModelCheckpoint(
            dirpath=model_dir,
            filename=f"unet3d-{{epoch:03d}}-{{step}}-{{{monitor_metric}:.3f}}",
            monitor=monitor_metric,
            mode="min",
            save_top_k=save_top_k,
            save_last=True,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
    return ModelCheckpoint(
        dirpath=model_dir,
        filename="unet3d-{epoch:03d}-{step}",
        save_last=True,
        save_top_k=1,
        save_on_train_epoch_end=True,
        every_n_epochs=max(1, every_n_epochs),
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


def run_training(
    *,
    module: LatentFlowModule,
    bundle: _DataBundle,
    feature_net: Any = None,
    feature_net_factory: Any = None,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    enable_fid: bool = True,
    limit_val_batches: int | float = 4,
    save_top_k: int = 3,
    seed: int = 0,
    ckpt_path: str | None = None,
    inference_recipe: dict | None = None,
    # fid_eval knobs (from the config block / dotlist overrides):
    num_synth: int = 16,
    every_n_epochs: int = 1,
    center_slices_ratio: float = 0.5,
    cov_ridge: float = 1e-6,
    log_raw_fid: bool = True,
    val_subset_size: int = 32,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the module (the core seam).

    Builds the train-metrics / EMA / (optional) FID callbacks + a stock
    ``ModelCheckpoint`` and runs ``Trainer.fit``. Returns ``(trainer, ckpt)`` so
    callers (and the Export) can find the written ``.ckpt``.

    Args:
        bundle: the warmed latent dataset + held VAE + fixed real-subset latents.
        feature_net: the FID feature network (RadImageNet, or a test fake).
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    ema = DoubleEMACallback(module)
    callbacks: list = [TrainLossLogger(), LatentX0MAE(), ema]

    multi_gpu = is_multi_gpu(devices)
    if enable_fid:
        # F5: latent_shape derives from val_latents when present (warmed path);
        # the cold path passes it via inference_recipe["latent_shape"] (it is known
        # from the VAE stride + vol_ds target_dim, not from the not-yet-warmed
        # val_latents). ``_inference_recipe`` honors an explicit latent_shape kwarg.
        # On the cold path (val_latents=None) the caller MUST pass inference_recipe
        # with latent_shape set (main() derives it via _derive_latent_shape); a
        # bare None here is a wiring bug - fail with a clear message, not an
        # AttributeError on ``None.shape`` deep in _inference_recipe.
        if inference_recipe is None and bundle.val_latents is None:
            raise ValueError(
                "enable_fid=True with a cold bundle (val_latents=None) requires an "
                "inference_recipe carrying 'latent_shape' (derive it from the VAE "
                "stride + target_dim). main() does this via _derive_latent_shape."
            )
        inf = inference_recipe or _inference_recipe(module, cfg=None, val_latents=bundle.val_latents)
        fid = FIDCallback(
            module=module,
            vae=bundle.vae,
            ema_callback=ema,
            real_latents=bundle.val_latents,  # F5: None on the cold path -> lazy
            real_latents_source=None,  # set below once the datamodule exists
            feature_net=feature_net,
            feature_net_factory=feature_net_factory,
            latent_shape=inf["latent_shape"],
            spacing=inf["spacing"],
            modality=inf["modality"],
            num_inference_steps=inf["num_inference_steps"],
            guidance_scale=inf["guidance_scale"],
            cfg_interval=inf["cfg_interval"],
            num_synth=num_synth,
            every_n_epochs=every_n_epochs,
            center_slices_ratio=center_slices_ratio,
            cov_ridge=cov_ridge,
            seed=seed,
            log_raw_fid=log_raw_fid,
        )
        callbacks.append(fid)

    ckpt = _build_checkpoint(
        model_dir,
        monitor_fid=enable_fid and not multi_gpu,
        # monitor what's logged: raw arm if present, else the slow-EMA avg.
        monitor_metric="val/fid_raw" if log_raw_fid else "val/fid_avg",
        every_n_epochs=every_n_epochs,
        save_top_k=save_top_k,
    )
    callbacks.append(ckpt)

    # F4/F1 (ADR-0017): the warm runs in DataModule.setup() (post-PG, per-rank
    # sharded) when bundle.warm_fn is set (the production cold path); the warmed
    # test path (warm_fn=None) makes setup() a no-op. The FID callback pulls
    # real_latents LAZILY from the datamodule (F5) so the first validation epoch
    # (post-setup) finds them populated.
    from ..data.warm_datamodule import LatentWarmDataModule

    datamodule = LatentWarmDataModule(
        latent_ds=bundle.latent_ds,
        vae=bundle.vae,
        batch_size=batch_size,
        num_workers=num_workers,
        val_latents=bundle.val_latents,
        warm_fn=bundle.warm_fn,
        val_subset_size=val_subset_size,
    )
    if enable_fid and fid.real_latents is None:
        fid._real_latents_source = datamodule  # F5: lazy pull at first _real_features
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


def _derive_latent_shape(cfg) -> tuple:
    """The single-sample latent shape ``(1, C, D, H, W)`` from the config.

    F5 (ADR-0017): the FIDCallback's ``latent_shape`` must be known at construction,
    but on the cold path ``val_latents`` is ``None`` (the warm is deferred to
    ``setup()``). Derive it from the VAE stride (``autoencoder_divisor``) and the
    inference ``dim`` (the volume target_dim, padded to the divisor) so it matches
    what ``val_latents.shape[1:]`` will be post-warm.
    """
    from ..config import autoencoder_divisor

    inf_cfg = cfg.diffusion_unet_inference
    div = autoencoder_divisor(cfg)
    dim = tuple(int(d) for d in inf_cfg.dim)
    # Volumes are padded to a multiple of the divisor (issue #16); the latent is
    # the padded-vol // divisor.
    latent_spatial = tuple((d + div - 1) // div * div // div for d in dim)
    c = int(cfg.latent_channels) if hasattr(cfg, "latent_channels") else 4
    return (1, c, *latent_spatial)


def _plain_list(value):
    """Return OmegaConf/list/tuple values as a plain Python list."""
    return list(value) if value is not None else None


def _inference_recipe(module: LatentFlowModule, *, cfg=None, val_latents: torch.Tensor | None = None, latent_shape=None) -> dict:
    """Generation recipe for the FID callback (mirrors configured inference).

    ``latent_shape`` is the single-sample template of the **real** validation
    latents — ``(1,) + val_latents.shape[1:]`` — not a hardcoded constant. The FID
    compares features decoded from generated vs. real latents in the same image
    space, so the synthetic latent must carry the real latents' spatial shape
    (e.g. ``(1, 4, 64, 64, 32)``); deriving it from warmed latents tracks any
    ``diffusion_unet_inference.dim`` / VAE-stride change.

    The sampling knobs come from the composed experiment config when available;
    direct unit-test callers without a config retain the old tiny defaults.
    """
    inf_cfg = getattr(cfg, "diffusion_unet_inference", None) if cfg is not None else None
    form_cfg = getattr(cfg, "formulation", None) if cfg is not None else None
    spacing = _plain_list(opt(inf_cfg, "spacing", [1.0, 1.0, 1.0])) if inf_cfg is not None else [1.0, 1.0, 1.0]
    cfg_interval = _plain_list(opt(form_cfg, "cfg_interval", None)) if form_cfg is not None else None
    return {
        "latent_shape": latent_shape if latent_shape is not None else (1,) + tuple(val_latents.shape[1:]),
        "spacing": spacing,
        "modality": int(opt(inf_cfg, "modality", 1)) if inf_cfg is not None else 1,
        "num_inference_steps": int(opt(inf_cfg, "num_inference_steps", 4)) if inf_cfg is not None else 4,
        "guidance_scale": float(opt(inf_cfg, "cfg_guidance_scale", 1.0)) if inf_cfg is not None else 1.0,
        "cfg_interval": cfg_interval,
    }


# -- console entry -----------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="manifold-train", description=__doc__.splitlines()[0])
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument("-c", "--train", default="configs/train/config_rflow_jit.yaml", help="train recipe YAML.")
    parser.add_argument("-t", "--network", required=True, help="network construction YAML.")
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs (1 = single-GPU).")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--warm-start", default=None, help="warm-start UNet checkpoint (None = from scratch)."
    )
    parser.add_argument(
        "--no-fid", action="store_true", help="disable the generative FID callback (latent x0-MAE only)."
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist (e.g. diffusion_unet_train.lr=1e-4).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config → warm cache → build → ``run_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> _DataBundle`` returning a fake latent cache + tiny VAE so
    the full ``main`` path (argparse → compose → build → fit → checkpoint) runs
    without BraTS data or a real VAE encode. The real path warms the cache via
    the latent-prep orchestration.
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, opt, require_paths
    from ..config.builder import build_scheduler, build_unet

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    require_paths(cfg)
    OmegaConf.resolve(cfg)

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        # Test smoke: the provider returns a WARMED bundle (latent_ds + val_latents
        # already set, no warm_fn); num_examples comes from the warmed dataset.
        bundle = data_provider(cfg, device)
        num_examples = len(bundle.latent_ds) if bundle.latent_ds is not None else 0
    else:
        # Production cold path (ADR-0017): the warm is deferred to
        # DataModule.setup() (post-PG); num_examples = len(vol_ds) (pre-warm).
        bundle, num_examples = _warm_data(cfg, device)

    train_cfg = cfg.diffusion_unet_train
    module = LatentFlowModule(
        build_unet(cfg),
        build_scheduler(cfg),
        p_mean=float(opt(cfg.formulation, "p_mean", -0.8)),
        p_std=float(opt(cfg.formulation, "p_std", 0.8)),
        t_eps=float(opt(cfg.formulation, "t_eps", 0.05)),
        lr=float(train_cfg.lr),
        lr_warmup_steps=int(train_cfg.lr_warmup_steps),
        lr_ref_batch_size=int(opt(train_cfg, "lr_ref_batch_size", 8)),
        lr_scale_rule=str(opt(train_cfg, "lr_scale_rule", "sqrt")),
        lr_warmup_ratio=opt(train_cfg, "lr_warmup_ratio", None),
        num_train_examples=num_examples,
        train_batch_size=int(train_cfg.batch_size),
        n_epochs=int(args.max_epochs or train_cfg.n_epochs),
    )
    if args.warm_start:
        _load_warm_start(module.unet, args.warm_start)

    feature_net_factory = None
    if not args.no_fid:
        from ..metrics import make_feature_network

        # L3: build the backbone LAZILY inside the rank-0-gated FID stage path
        # (no eager make_feature_network on every rank pre-PG -> no ~100 MB
        # torch.hub load wasted on N-1 ranks). Probe availability CHEAPLY here
        # (cached-checkpoint check, no model instantiation - P2: building the full
        # backbone to probe would itself defeat the lazy factory + load it twice on
        # rank 0). When the cache is absent the factory attempts the online load at
        # eval (rank 0) and the FID callback skips on failure.
        from ..metrics import feature_network_available

        if not feature_network_available("resnet50"):
            _log.warning(
                "RadImageNet backbone cache not found (%s); disabling FID at launch. "
                "Pre-cache the checkpoint or the factory will retry online at eval.",
                "resnet50",
            )
            args.no_fid = True
        else:
            feature_net_factory = lambda: make_feature_network("resnet50")  # noqa: E731

    max_epochs = int(args.max_epochs or train_cfg.n_epochs)

    # Thread the optional config blocks (from the train recipe) as overrides so
    # fid_eval.* / checkpoint.* knobs are actually reachable via dotlist/YAML.
    fid_cfg = opt(cfg, "fid_eval", {})
    ckpt_cfg = opt(cfg, "checkpoint", {})
    fid_kwargs = _dict_subset(
        fid_cfg, ("num_synth", "every_n_epochs", "center_slices_ratio", "cov_ridge", "log_raw_fid")
    )
    ckpt_save_top_k = fid_cfg.get("save_top_k", ckpt_cfg.get("save_top_k", 3)) if fid_cfg or ckpt_cfg else 3

    run_training(
        module=module,
        bundle=bundle,
        feature_net_factory=feature_net_factory,
        model_dir=str(cfg.model_dir),
        max_epochs=max_epochs,
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(train_cfg.batch_size),
        enable_fid=not args.no_fid,
        seed=seed,
        ckpt_path=args.resume,
        inference_recipe=_inference_recipe(
            module, cfg=cfg, val_latents=bundle.val_latents,
            latent_shape=_derive_latent_shape(cfg) if bundle.val_latents is None else None,
        ),
        save_top_k=ckpt_save_top_k,
        limit_val_batches=int(opt(cfg, "val_subset_size", 4)),
        val_subset_size=int(opt(cfg, "val_subset_size", 32)),
        **fid_kwargs,
    )
    print(f"[manifold-train] done; checkpoints under {cfg.model_dir}")
    return 0


def _warm_data(cfg, device) -> tuple[_DataBundle, int]:
    """Build the cold-start warm bundle + the source volume count (production path).

    ADR-0017 / F1+F4 (issue #84): the warm is DEFERRED to ``DataModule.setup()``
    (post-PG) so the per-rank ``i % world == rank`` sharding activates. Returns a
    ``_DataBundle`` carrying ``warm_fn`` (a closure over
    :func:`warm_latent_pipeline` - the atomic ``warm_cache`` -> ``free_encoder`` ->
    ``estimate_scale_factor`` unit) with NO pre-warmed ``latent_ds`` / ``val_latents``
    (those are populated inside ``setup()``). ``num_examples`` is the source volume
    count ``len(vol_ds)`` (known pre-warm) so the Module's LR horizon is set without
    needing ``len(latent_ds)``.
    """
    from ..data.latent_pipeline import (
        build_encode_pipeline,
        build_volume_dataset,
        make_encode_fn,
        resolve_warm_device,
        warm_latent_pipeline,
    )

    logger = logging.getLogger("manifold.train")
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    vol_ds, _ = build_volume_dataset(
        cfg, target_dim=target_dim, include_modality=True, default_modality=int(inf_cfg.modality), logger=logger
    )
    # Build the VAE on CPU pre-PG (the launch-time ``device`` is the default cuda:0,
    # which under DDP would load on GPU 0 before LOCAL_RANK is known). The warm
    # re-stages it onto the per-rank local GPU inside setup() (P1/ADR-0017); FID's
    # decode stages it to the UNet device at eval (rank 0). The CPU encode_fn built
    # here is unused on the cold path (rebuilt on the local device in warm_fn).
    autoencoder, _cpu_encode_fn = build_encode_pipeline(cfg, device=torch.device("cpu"), logger=logger)
    cache_dir = str(opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "latent_cache")))
    scale_sample = int(opt(cfg, "val_subset_size", 32))

    def warm_fn():
        # P1: warm on the per-rank local CUDA device, not the launch-time ``device``
        # (which is cuda:0 before Lightning assigns LOCAL_RANK). Rebuild the encode_fn
        # bound to that device so the sliding-window predictor runs on the right GPU.
        warm_device = resolve_warm_device(device)
        autoencoder.to(warm_device)
        encode_fn = make_encode_fn(autoencoder, warm_device, cfg)
        # F3: rank/world derived from dist inside warm_latent_pipeline (post-PG).
        return warm_latent_pipeline(
            vol_ds, encode_fn, autoencoder,
            cache_dir=cache_dir, cache_tag="train",
            device=warm_device, logger=logger,
            scale_factor_sample_size=scale_sample,
        )

    return (
        _DataBundle(vae=autoencoder, warm_fn=warm_fn),
        len(vol_ds),
    )


def _load_warm_start(unet, ckpt_path: str) -> None:
    """Load a warm-start UNet checkpoint (a Lightning ``.ckpt`` or a state dict)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # A Lightning .ckpt: take the wrapper's UNet params ("unet.unet.*").
        sd = {
            k[len("unet.unet."):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("unet.unet.")
        }
        unet.unet.load_state_dict(sd, strict=True)
    else:
        unet.unet.load_state_dict(ckpt, strict=True)
