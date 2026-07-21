"""``manifold-train`` console entry + the testable training-orchestration core.

The console entry (issue #28) composes the OmegaConf experiment config, warms
the latent cache, builds the Module + the fixed validation subset + the
callbacks (train metrics, FID, ``ModelCheckpoint``), and calls
``Trainer.fit``. From-scratch by default; an optional warm-start path.

The heavy data-warming lives in :func:`main`; the integration core
:func:`run_training` (Module + datamodule + callbacks + ``ModelCheckpoint`` +
``build_trainer`` + ``fit``) is split out so a tiny CPU smoke can drive it with a
fake latent cache (the issue's testing seam) instead of BraTS + a real VAE.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.utilities.rank_zero import rank_zero_info  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import opt
from ..modules.latent_flow import LatentFlowModule
from manifold.training.callbacks import (
    CallbackContext,
    CheckpointSpec,
    FIDSpec,
    TrainLossSpec,
)
from manifold.training.core import TrainingSpine
from .metrics import LatentX0MAE


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

    ``allow_train_as_val``: smoke-only opt-in to reuse the train cache as the val
    loader + derive ``val_latents`` from it. Production leaves it ``False``.
    ``has_val``: production flag — a held-out val split (``val_data_base_dir``) is
    configured and ``warm_fn`` will materialize ``val_latent_ds`` inside
    ``setup()``; :func:`run_training` then ENABLES validation (FID + ``val/x0_mae``).
    When both ``allow_train_as_val`` and ``has_val`` are ``False`` (no held-out val
    source), validation is DISABLED rather than leak train metrics as val.
    """

    latent_ds: Any = None
    vae: Any = None
    val_latents: torch.Tensor | None = None
    warm_fn: Any = None
    allow_train_as_val: bool = False
    has_val: bool = False


def _dict_subset(d: dict | None, keys: tuple[str, ...]) -> dict:
    """Extract the non-``None`` values from *d* for the given *keys*."""
    if not d:
        return {}
    return {k: d[k] for k in keys if d.get(k) is not None}


def _resolve_fid_overrides(cfg, raw_overrides):
    """Resolve ``fid`` / legacy ``fid_eval`` overrides honoring dotlist precedence.

    ``merge_overrides`` already merged YAML + dotlist into ``cfg``. OmegaConf
    keeps the two namespaces distinct, so the last dotlist token among
    ``fid.*`` / ``fid_eval.*`` for each key must be picked explicitly by looking
    at ``raw_overrides`` order.
    """
    fid = dict(opt(cfg, "fid", {}))
    legacy = dict(opt(cfg, "fid_eval", {}))
    last_ns: dict[str, str] = {}
    for token in raw_overrides or []:
        if "=" not in token:
            continue
        key_path, _ = token.split("=", 1)
        if key_path.startswith("fid_eval."):
            last_ns[key_path.split(".", 1)[1]] = "fid_eval"
        elif key_path.startswith("fid."):
            last_ns[key_path.split(".", 1)[1]] = "fid"
    merged: dict[str, Any] = {}
    for k in set(fid) | set(legacy):
        ns = last_ns.get(k)
        if ns == "fid_eval":
            merged[k] = legacy[k]
        elif ns == "fid":
            merged[k] = fid[k]
        else:
            merged[k] = fid.get(k, legacy.get(k))
    return merged


def _resolve_save_top_k(ckpt_cfg, fid_cfg, raw_overrides):
    """Resolve ``save_top_k`` with dotlist precedence (checkpoint > fid_eval > recipe).

    When the recipe's ``checkpoint`` block carries a default ``save_top_k``
    (e.g. 3 in ``config_rflow_jit.yaml``), ``ckpt_cfg.get("save_top_k")``
    returns that default even when the user explicitly set a legacy override
    via ``fid_eval.save_top_k=2``. This helper checks the raw dotlist first
    so the explicit override wins.
    """
    _ckpt = None
    for token in raw_overrides or []:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        if k == "checkpoint.save_top_k":
            _ckpt = int(v)
        elif k in ("fid_eval.save_top_k", "fid.save_top_k") and _ckpt is None:
            _ckpt = int(v)
    if _ckpt is not None:
        return _ckpt
    return ckpt_cfg.get("save_top_k", fid_cfg.get("save_top_k", 3))


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
    # fid knobs (from the ``fid`` config block / dotlist overrides):
    num_synth: int = 16,
    every_n_epochs: int = 1,
    center_slices_ratio: float = 0.5,
    cov_ridge: float = 1e-6,
    val_subset_size: int = 32,
    # checkpoint knobs (ADR-0029): pass the full ``checkpoint`` block through so
    # monitor_metric / mode / filename / save_last are not silently ignored.
    checkpoint_cfg: dict | None = None,
    # ADR-0032: a CLI ``--callbacks`` override (full name-list replacement of
    # the derived defaults). ``None`` keeps the derived default set (unchanged
    # behaviour); the spine applies the ADR-0029 merge order either way.
    callback_names: list[str] | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the module (the core seam).

    Builds the train-metrics / (optional) FID callbacks + a stock
    ``ModelCheckpoint`` and runs ``Trainer.fit``. Returns ``(trainer, ckpt)`` so
    callers (and the Export) can find the written ``.ckpt``.

    The shell (ADR-0032): seed, build module + datamodule, derive default
    callback-name set, and delegate to :class:`TrainingSpine.run`. The
    ``_real_inputs`` data-assembly path stays here — it is JiT-specific and
    the five data paths are genuinely different.

    Args:
        bundle: the warmed latent dataset + held VAE + fixed real-subset latents.
        feature_net: the FID feature network (RadImageNet, or a test fake).
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    val_enabled = bundle.allow_train_as_val or bundle.has_val
    # F4/F1 (ADR-0017): the warm runs in DataModule.setup() (post-PG, per-rank
    # sharded) when bundle.warm_fn is set (the production cold path); the warmed
    # test path (warm_fn=None) makes setup() a no-op. The FID callback pulls
    # real_latents LAZILY from the datamodule (F5) so the first validation epoch
    # (post-setup) finds them populated. Built before the callbacks so the
    # CallbackContext (ADR-0029) carries the real datamodule, not a placeholder.
    from ..data.warm_datamodule import LatentWarmDataModule

    datamodule = LatentWarmDataModule(
        latent_ds=bundle.latent_ds,
        vae=bundle.vae,
        batch_size=batch_size,
        num_workers=num_workers,
        val_latents=bundle.val_latents,
        warm_fn=bundle.warm_fn,
        val_subset_size=val_subset_size,
        allow_train_as_val=bundle.allow_train_as_val,
    )

    fid_attached = enable_fid and val_enabled
    if fid_attached and inference_recipe is None and bundle.val_latents is None:
        raise ValueError(
            "enable_fid=True with a cold bundle (val_latents=None) requires an "
            "inference_recipe carrying 'latent_shape' (derive it from the VAE "
            "stride + target_dim). main() does this via _derive_latent_shape."
        )
    inf = (
        inference_recipe or _inference_recipe(module, cfg=None, val_latents=bundle.val_latents)
        if fid_attached
        else inference_recipe
    )

    # ADR-0032: default names → callback_cfg knobs → TrainingSpine.run.
    spine = TrainingSpine()
    spine.registry.register("train_loss", TrainLossSpec)
    spine.registry.register("fid", FIDSpec)
    spine.registry.register("checkpoint", CheckpointSpec)

    names = ["train_loss"]
    callback_cfg_built: dict[str, dict] = {}
    if fid_attached:
        names.append("fid")
        callback_cfg_built["fid"] = {
            "num_synth": num_synth,
            "every_n_epochs": every_n_epochs,
            "center_slices_ratio": center_slices_ratio,
            "cov_ridge": cov_ridge,
        }
    names.append("checkpoint")
    callback_cfg_built["checkpoint"] = {
        "monitor_metric": "val/fid" if fid_attached else None,
        "save_top_k": save_top_k,
        "every_n_epochs": every_n_epochs,
    }
    if checkpoint_cfg:
        callback_cfg_built["checkpoint"].update(checkpoint_cfg)
        callback_cfg_built["checkpoint"]["save_top_k"] = save_top_k

    ctx = CallbackContext(
        module=module,
        vae=bundle.vae,
        datamodule=datamodule,
        inference_recipe=inf,
        model_dir=model_dir,
        seed=seed,
        feature_net=feature_net,
        feature_net_factory=feature_net_factory,
    )

    extra_callbacks: list = []
    extra_trainer_kwargs: dict | None = None
    if val_enabled:
        # LatentX0MAE declares logged_metrics={"val/x0_mae"} so the registry's
        # validate_monitor accepts that monitor without mutating module.logged_metrics.
        extra_callbacks.append(LatentX0MAE())
    else:
        rank_zero_info(
            "manifold-train: no held-out validation set is configured "
            "(val_data_base_dir is unset or not a directory). Validation is DISABLED "
            "- val/* metrics will not be logged. Reusing the train set as val would "
            "leak train metrics into validation, which is refused; set "
            "val_data_base_dir to a held-out BraTS directory to enable."
        )
        extra_trainer_kwargs = {"num_sanity_val_steps": 0}

    return spine.run(
        module=module,
        datamodule=datamodule,
        ctx=ctx,
        default_names=names,
        max_epochs=max_epochs,
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches if val_enabled else 0,
        extra_trainer_kwargs=extra_trainer_kwargs,
        ckpt_path=ckpt_path,
        callback_cfg=callback_cfg_built,
        callback_names_override=callback_names,
        extra_callbacks=extra_callbacks,
    )


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

        # L3 + codex #85 P2: build the backbone LAZILY (rank-0-gated FID stage
        # path) so non-root ranks never pay the ~100 MB load, and rank 0 builds
        # exactly once. The factory is FAIL-SAFE (try/except -> None): a bad/corrupt
        # cache or a no-network host makes FIDCallback skip gracefully instead of
        # aborting training mid-fit, and the online torch.hub fallback stays
        # reachable (no launch-time pre-disable on a missing cache). No eager probe.
        def feature_net_factory():
            try:
                return make_feature_network("resnet50")
            except Exception as exc:  # pragma: no cover - torch.hub/network only on gauss/dev
                rank_zero_info("RadImageNet backbone unavailable (%r); FID will be skipped.", exc)
                return None

    max_epochs = int(args.max_epochs or train_cfg.n_epochs)

    # Thread the optional config blocks (from the train recipe) as overrides so
    # fid.* / checkpoint.* knobs are actually reachable via dotlist/YAML.
    # ADR-0029: the ``fid_eval`` block is renamed to ``fid`` (callback name ==
    # config namespace). Existing dotlist overrides may still arrive as
    # ``fid_eval.*`` — ``_resolve_fid_overrides`` translates them to ``fid.*``
    # while preserving dotlist precedence, and ``checkpoint_cfg`` is passed
    # through to ``run_training`` so ``monitor_metric``, ``mode``, ``filename``,
    # and ``save_last`` are not silently ignored.
    fid_cfg = _resolve_fid_overrides(cfg, args.overrides)
    ckpt_cfg = dict(opt(cfg, "checkpoint", {}))

    fid_kwargs = _dict_subset(
        fid_cfg, ("num_synth", "every_n_epochs", "center_slices_ratio", "cov_ridge")
    )
    # ``save_top_k`` may be supplied either in the legacy ``fid_eval`` block or in
    # the modern ``checkpoint`` block; ``_resolve_save_top_k`` checks the raw
    # dotlist first so an explicit ``fid_eval.save_top_k=2`` wins over the
    # recipe's ``checkpoint.save_top_k: 3`` (the standard recipe default).
    ckpt_save_top_k = _resolve_save_top_k(ckpt_cfg, fid_cfg, args.overrides)

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
        checkpoint_cfg=ckpt_cfg,
        **fid_kwargs,
    )
    print(f"[manifold-train] done; checkpoints under {cfg.model_dir}")
    return 0


def _warm_data(cfg, device) -> tuple[_DataBundle, int]:
    """Build the cold-start warm bundle + the source volume count (production path).

    ADR-0017 / F1+F4 (issue #84): the warm is DEFERRED to ``DataModule.setup()``
    (post-PG) so the per-rank ``i % world == rank`` sharding activates. Returns a
    ``_DataBundle`` carrying ``warm_fn`` (a closure over
    :func:`warm_latent_pipeline` for train + an optional held-out val warm) with NO
    pre-warmed ``latent_ds`` / ``val_latents`` (those are populated inside
    ``setup()``). ``num_examples`` is the source volume count ``len(vol_ds)``
    (known pre-warm) so the Module's LR horizon is set without needing
    ``len(latent_ds)``.

    When ``cfg.val_data_base_dir`` is a BraTS directory, a held-out val
    :class:`~manifold.data.NiftiVolumeDataset` is built (disjoint subjects from the
    organizer ship split — no train/val leakage) and ``warm_fn`` warms it alongside
    train; ``has_val`` is set so :func:`run_training` enables validation. The val
    cache stores UNSCALED latents and reuses the train-estimated ``scale_factor``
    (scale-on-read; one factor over both splits, never re-estimated on val).
    """
    from ..config import autoencoder_divisor
    from ..data.latent_dataset import LatentDataset
    from ..data.latent_pipeline import (
        build_encode_pipeline,
        build_volume_dataset,
        make_encode_fn,
        resolve_warm_device,
        warm_latent_pipeline,
    )
    from ..data.volume_dataset import NiftiVolumeDataset

    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    vol_ds, provider = build_volume_dataset(
        cfg, target_dim=target_dim, include_modality=True, default_modality=int(inf_cfg.modality)
    )
    divisor = autoencoder_divisor(cfg)

    # Held-out val volumes (the FID real set + the val/x0_mae loader). Only a BraTS
    # *directory* is usable (mirrors paired_reward_cli._train_val_manifests); the
    # organizer ship split is disjoint by subject, so there is no train/val leakage.
    val_dir = opt(cfg, "val_data_base_dir", None)
    has_val = bool(val_dir and os.path.isdir(str(val_dir)))
    val_vol_ds = None
    if has_val:
        val_vol_ds = NiftiVolumeDataset(
            str(val_dir), provider, target_dim, divisor, data_base_dir=str(val_dir)
        )
        if not len(val_vol_ds):
            raise FileNotFoundError(
                f"No validation NIfTI found under val_data_base_dir={val_dir}."
            )

    # Build the VAE on CPU pre-PG (the launch-time ``device`` is the default cuda:0,
    # which under DDP would load on GPU 0 before LOCAL_RANK is known). The warm
    # re-stages it onto the per-rank local GPU inside setup() (P1/ADR-0017); FID's
    # decode stages it to the UNet device at eval (rank 0). The CPU encode_fn built
    # here is unused on the cold path (rebuilt on the local device in warm_fn).
    autoencoder, _cpu_encode_fn = build_encode_pipeline(cfg, device=torch.device("cpu"))
    cache_dir = str(opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "latent_cache")))
    scale_sample = int(opt(cfg, "val_subset_size", 32))

    def warm_fn():
        # P1: warm on the per-rank local CUDA device, not the launch-time ``device``
        # (which is cuda:0 before Lightning assigns LOCAL_RANK). Rebuild the encode_fn
        # bound to that device so the sliding-window predictor runs on the right GPU.
        warm_device = resolve_warm_device(device)
        autoencoder.to(warm_device)
        encode_fn = make_encode_fn(autoencoder, warm_device, cfg)
        # Warm the held-out val cache FIRST (while the VAE is still on the warm
        # device): warm_latent_pipeline below frees the train encoder and moves the
        # VAE to CPU. The val cache stores UNSCALED latents; the train-estimated
        # scale_factor is applied as scale-on-read once the train warm completes
        # (scale-consistency: one factor over both splits, never re-estimated on val).
        val_latent_ds = None
        if val_vol_ds is not None:
            import torch.distributed as dist

            rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
            val_latent_ds = LatentDataset(
                val_vol_ds, encode_fn=encode_fn, cache_dir=cache_dir, cache_tag="val"
            )
            val_latent_ds.warm_cache(warm_device, show_progress=rank0)
            val_latent_ds.free_encoder()
        # F3: rank/world derived from dist inside warm_latent_pipeline (post-PG).
        train_pipeline = warm_latent_pipeline(
            vol_ds, encode_fn, autoencoder,
            cache_dir=cache_dir, cache_tag="train",
            device=warm_device,
            scale_factor_sample_size=scale_sample,
        )
        if val_latent_ds is not None:
            val_latent_ds.scaling_factor = float(train_pipeline.scale_factor)
        return train_pipeline.latent_ds, train_pipeline.autoencoder, val_latent_ds

    return (
        _DataBundle(vae=autoencoder, warm_fn=warm_fn, has_val=has_val),
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
