"""``manifold-train-paired-reward`` console entry + the testable paired-reward core.

The console entry (issue #93) composes the OmegaConf experiment config, builds the
condition-aware :class:`~manifold.models.RewardModel` (``in_channels = 2·C_latent``)
+ the precomputed pair inputs (real latent cache + the frozen paired generator, or
a fake via the ``data_provider`` injection seam for the CPU smoke), and calls
``Trainer.fit``. The paired-reward job is standalone: decoupled from Paired-JiT
training and from paired-GRPO, independently resumable.

The integration core :func:`run_paired_reward_training` (Module + datamodule +
``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a tiny CPU smoke
can drive it with a fake generator + toy pairs (the issue's testing seam) instead of
a real latent cache + the frozen paired generator.

**Offline precompute (ADR-0020, inverts ADR-0010).** The train set is a precomputed
``{winner, loser}`` pair dataset (built once from the fake cache); the Module holds
no generator - ``fit`` consumes pairs and is structurally the JiT reward's
``validate`` path. Validation (real-vs-fake pairs) + the generated-end probe are
precomputed **once** (the generator is frozen ⇒ static across epochs) and reused.
"""

from __future__ import annotations

import argparse
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
from ..data.paired_manifests import _train_val_manifests
from ..models.reward_model import RewardModel
from ..modules.paired_reward import PairedRewardModule
from manifold.training.callbacks import CallbackContext, CheckpointSpec, TrainLossSpec
from manifold.training.core import TrainingSpine
from .trainer import is_multi_gpu


@dataclass
class PairedRewardInputs:
    """Precomputed-pair inputs for paired reward training (no generator - ADR-0020).

    The Module holds NO generator: ``fit`` and ``validate`` both consume
    precomputed ``{winner, loser}`` pair datasets (built once from the fake cache).
    The ``data_provider`` seam injects a fake generator + toy pairs for the CPU
    smoke; the real path (``_real_inputs``, issue #95) builds this from the paired
    native export (#94's loader) + the paired latent cache.
    """

    train_pair_ds: Any  #: train: emits {winner, loser} (condition-aware [2·C] concat)
    val_pair_ds: Any  #: precomputed: emits {winner, loser} (real-vs-fake, val subjects)
    val_probe: Any = None  #: precomputed RewardPairDataset (generated-end probe, [2·C]).


def run_paired_reward_training(
    *,
    module: PairedRewardModule,
    inputs: PairedRewardInputs,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    save_top_k: int = 1,
    monitor_metric: str = "val/gen_pair_acc",
    mode: str = "max",
    limit_val_batches: int | float = 1.0,
    seed: int = 0,
    ckpt_path: str | None = None,
    callback_names: list[str] | None = None,
    callback_cfg: dict[str, dict] | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the paired reward module (the core seam).

    The shell (ADR-0032): seed, the mandatory-probe guard, the pre-fit probe
    mutation, a stock ``ModelCheckpoint`` (monitoring the generated-end probe, or
    an unmonitored fallback under DDP where the metric is rank-local), then
    delegate to :class:`TrainingSpine.run`. Returns ``(trainer, ckpt)`` so callers
    can find the written ``.ckpt``.

    Args:
        inputs: the precomputed train/val/probe pair datasets (no generator).
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
        callback_names: optional CLI ``--callbacks`` override (a full name-list
            replacement of the derived defaults; ADR-0032).
        callback_cfg: optional ``{name: {knob: value}}`` YAML knob overrides,
            merged over the shell-derived defaults (the YAML ``callbacks:`` block).
    """
    # Seed deterministically so direct callers (tests, notebooks) get reproducible
    # runs; ``main`` also seeds, harmlessly, before building the module.
    pl.seed_everything(seed, workers=True)
    multi_gpu = is_multi_gpu(devices)

    # Shell-derived checkpoint knobs, then the YAML/CLI callback_cfg override wins,
    # then the DDP monitor=None fallback is re-asserted POST-merge so a YAML
    # monitor_metric cannot restore a rank-local metric under DDP.
    ckpt_monitor = monitor_metric if not multi_gpu else None
    cfg_built: dict[str, dict] = {
        "checkpoint": {
            "monitor_metric": ckpt_monitor,
            "mode": mode,
            "save_top_k": save_top_k,
        }
    }
    for name, knobs in (callback_cfg or {}).items():
        cfg_built.setdefault(name, {}).update(knobs)
    if multi_gpu:
        cfg_built["checkpoint"]["monitor_metric"] = None
    effective_monitor = cfg_built["checkpoint"]["monitor_metric"]
    # Derive the filename from the EFFECTIVE (post-merge) monitor unless supplied.
    if cfg_built["checkpoint"].get("filename") is None:
        cfg_built["checkpoint"]["filename"] = (
            f"reward-{{epoch:03d}}-{{{effective_monitor}:.3f}}"
            if effective_monitor is not None
            else "reward-{epoch:03d}"
        )

    # The generated-end probe is mandatory (ADR-0023) only when the EFFECTIVE
    # monitor is ``val/gen_pair_acc`` — the metric the Module logs solely from the
    # probe. A YAML override to a non-probe metric (or ``null``) legitimately opts
    # out of the probe requirement (codex #183 P2). Under single-GPU the monitored
    # checkpoint would crash Lightning if the probe metric never appears; under DDP
    # the monitor is dropped (rank-local selection is unreliable), so no probe is
    # needed there (effective_monitor is None).
    if effective_monitor == "val/gen_pair_acc" and inputs.val_probe is None:
        raise ValueError(
            "Paired reward training monitors val/gen_pair_acc, but no generated-end "
            "probe was provided (inputs.val_probe is None). The probe is mandatory "
            "(ADR-0023) - build it via build_paired_reward_probe and pass it in, or "
            "set monitor_metric to a logged metric."
        )
    # Score the fixed generated-end probe in training-batch-size chunks (bounds
    # epoch-end memory); attach the probe if the inputs carry one.
    module.probe_batch_size = int(batch_size)
    if inputs.val_probe is not None and getattr(module, "val_probe", None) is None:
        module.set_val_probe(inputs.val_probe.winners, inputs.val_probe.losers)
    datamodule = build_datamodule(
        inputs.train_pair_ds,
        batch_size=batch_size,
        val_dataset=inputs.val_pair_ds,
        num_workers=num_workers,
    )

    spine = TrainingSpine()
    spine.registry.register("train_loss", TrainLossSpec)
    spine.registry.register("checkpoint", CheckpointSpec)
    ctx = CallbackContext(
        module=module,
        vae=None,
        datamodule=datamodule,
        inference_recipe=None,
        model_dir=model_dir,
        seed=seed,
    )
    return spine.run(
        module=module,
        datamodule=datamodule,
        ctx=ctx,
        default_names=["checkpoint"],
        max_epochs=max_epochs,
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches,
        ckpt_path=ckpt_path,
        callback_cfg=cfg_built,
        callback_names_override=callback_names,
    )


# -- console entry -----------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manifold-train-paired-reward",
        description="Train the paired-JiT reward model (real-vs-fake, offline precompute).",
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c",
        "--train",
        default="configs/train/config_paired_reward.yaml",
        help="paired-reward recipe YAML.",
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (reward_model block)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--native-dir",
        default=None,
        help="paired native export dir (frozen generator + VAE scale); required without --data-provider.",
    )
    parser.add_argument(
        "--latents-dir",
        default=None,
        help="paired latent cache dir (the warmed paired cache); required without --data-provider.",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument(
        "--callbacks",
        default=None,
        help="comma-separated callback names; REPLACES the YAML callbacks: list (ADR-0032).",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config -> build -> ``run_paired_reward_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> PairedRewardInputs`` returning a fake generator + toy pairs
    so the full ``main`` path runs without a real generator or latent cache. The
    real path loads the frozen paired generator from ``--native-dir`` and the paired
    cache from ``--latents-dir``, precomputes train/val/probe pairs once, and fits
    (ADR-0020 - the Module holds no generator).
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths, resolve_callback_names

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # The paired-reward job needs no VAE / UNet-checkpoint path; only the output
    # ``model_dir`` is required (the generator comes from --native-dir at precompute).
    require_paths(cfg, keys=("model_dir",))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "paired_reward_train", None) is None:
        raise ValueError(
            "Config has no `paired_reward_train` block - use the paired-reward recipe "
            "(-c configs/train/config_paired_reward.yaml), not a JiT/paired train config."
        )

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        inputs = data_provider(cfg, device)
    else:
        # --native-dir / --latents-dir are NOT argparse-required: that would break
        # the data_provider injection seam (the CPU smoke). Validate them here, only
        # on the real path (issue #95).
        if not args.native_dir or not args.latents_dir:
            raise ValueError(
                "Paired reward training needs --native-dir <paired native export> and "
                "--latents-dir <paired latent cache> (or inject a data_provider for the smoke)."
            )
        inputs = _real_inputs(cfg, args.native_dir, args.latents_dir, device)

    reward_cfg = cfg.reward_model
    latent_c = int(opt(cfg, "latent_channels", 4))
    module = PairedRewardModule(
        RewardModel(
            spatial_dims=int(opt(reward_cfg, "spatial_dims", 3)),
            # The paired reward scores condition-aware ``[x_src, tgt]`` concat pairs
            # ⇒ ``in_channels = 2·C_latent`` structurally (ADR-0019). The network
            # config's ``reward_model.in_channels`` is for the JiT reward (single C);
            # ignore it here — the ``opt(..., 2*C)`` fallback was dead because the
            # config carried ``${latent_channels}`` (codex #96/#99 P1/P2).
            in_channels=2 * latent_c,
            channels=int(opt(reward_cfg, "channels", 64)),
            num_layers_d=int(opt(reward_cfg, "num_layers_d", 3)),
            norm=str(opt(reward_cfg, "norm", "BATCH")),
        ),
        lr=float(cfg.paired_reward_train.lr),
    )

    # ADR-0032: the CLI ``--callbacks`` (comma list) REPLACES the YAML
    # ``callbacks:`` name list; the YAML ``checkpoint:`` knob block forwards as
    # the callback_cfg override. Both are ``None``/empty when neither is supplied,
    # so the shell uses its derived defaults (unchanged behaviour).
    cb_names = resolve_callback_names(args.callbacks, cfg)
    callback_cfg = {"checkpoint": dict(opt(cfg, "checkpoint", {}))}

    run_paired_reward_training(
        module=module,
        inputs=inputs,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or cfg.paired_reward_train.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(cfg.paired_reward_train.batch_size),
        seed=seed,
        ckpt_path=args.resume,
        callback_names=cb_names,
        callback_cfg=callback_cfg,
    )
    print(f"[manifold-train-paired-reward] done; checkpoints under {cfg.model_dir}")
    return 0


def _real_inputs(
    cfg, native_dir: str, latents_dir: str, device: torch.device
) -> PairedRewardInputs:
    """Build the real paired-reward inputs from the ControlNet native export + latent cache.

    Loads the frozen ControlNet generator (``--native-dir``, T7's
    :func:`~manifold.training.controlnet_inputs.load_frozen_controlnet_generator` —
    frozen base + ControlNet + base scheduler + scaling_factor), resolves the
    **paired** train/val split
    (``_train_val_manifests`` / ``val_data_base_dir`` / ``val_fraction`` - NOT JiT
    reward's ``partition_subjects``, ADR-0022), warms the paired latent cache over
    each split (reusing the existing ``paired_train`` cache - disjoint sample_ids
    -> free disk hits), sets each dataset's ``scaling_factor`` to the export's
    (ADR-0021 - scale-consistency: reuse verbatim, never re-estimate), and builds
    the train/val/probe condition-aware pairs via
    :func:`~manifold.data.paired_reward_pairs.build_paired_reward_inputs` (the
    offline fake-cache builder, ADR-0020). The generator is used once; the returned
    inputs carry only precomputed pairs - the Module holds no generator.
    """
    import os

    from ..config import autoencoder_divisor
    from ..data.paired_brats import build_brats_pair_manifest
    from ..data.paired_latent_dataset import PairedLatentDataset, paired_cache_tag
    from ..data.paired_reward_pairs import build_paired_reward_inputs
    from ..data.paired_volume_dataset import PairedNiftiVolumeDataset
    from .controlnet_inputs import load_frozen_controlnet_generator

    # target_dim MUST match the paired_train cache the generator trained on (cache
    # reuse, ADR-0021/0022: sample_ids are derived from the volume + target_dim, so
    # a mismatch breaks disk hits). Read directly (not opt()) - it is load-bearing.
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    divisor = autoencoder_divisor(cfg)

    # The frozen ControlNet generator (raw arm): base UNet + ControlNet + base
    # scheduler + scaling_factor (ADR-0027/T7 — the fake source is the supervised
    # ControlNet's noise→data generation, replacing the deleted src→tgt rollout).
    generator, controlnet, base_scheduler, scaling_factor = load_frozen_controlnet_generator(native_dir)
    generator.to(device).eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    controlnet.to(device).eval()
    for p in controlnet.parameters():
        p.requires_grad_(False)

    # The PAIRED 2-way subject split (ADR-0022): resolve via _train_val_manifests
    # (val_data_base_dir / val_fraction), NOT JiT reward's partition_subjects.
    brats_dir = str(cfg.data_base_dir)
    manifest = build_brats_pair_manifest(brats_dir)
    if not manifest:
        raise FileNotFoundError(
            f"No paired BraTS volumes found under data_base_dir={brats_dir} "
            f"(need >=1 subject with all 4 contrasts)."
        )
    # _train_val_manifests reads ROOT cfg.val_fraction, but the paired-reward recipe
    # defines the held-out fraction under paired_reward.val_fraction. Mirror the nested
    # value to root whenever the native-split DIRECTORY path is not taken - i.e. when
    # val_data_base_dir is unset OR a non-directory (e.g. the BraTS2023 profile's
    # brats_all_val.json, which _train_val_manifests rejects and falls back to the
    # fraction) - else the val split resolves to 0 -> empty val -> the guard below
    # raises (codex #99 P1 / #100 P1). Mirror ONLY when the root key is absent: an
    # explicit root override (a CLI dotlist or a profile that sets val_fraction) wins
    # (codex #100 P2 round-3).
    val_dir = opt(cfg, "val_data_base_dir", None)
    if not (val_dir and os.path.isdir(str(val_dir))):
        from omegaconf import OmegaConf

        if OmegaConf.select(cfg, "val_fraction", default=None) is None:
            pr_val_fraction = float(opt(cfg, "paired_reward.val_fraction", 0.0))
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"val_fraction": pr_val_fraction}))

    train_manifest, val_manifest = _train_val_manifests(cfg, manifest)
    if not val_manifest:
        raise ValueError(
            "Paired reward needs a held-out val split (val_data_base_dir set, or "
            "val_fraction > 0); train data is never reused as val (ADR-0022)."
        )

    # Reuse the existing paired_train cache (disjoint sample_ids -> free disk hits,
    # no write conflict, ADR-0022). No VAE is needed: the cache is the fully-warmed
    # `paired_train` cache from a manifold-train-paired run, so warm_cache reads
    # every volume from disk (encode_fn=None is tolerated on cache hits). A partial
    # cache fails fast (PairedLatentDataset raises a clear "cache miss - no encoder"
    # error) rather than silently re-encoding. --latents-dir overrides latent_cache_dir.
    # The tag folds in target_dim/divisor (issue #147) so a geometry change reuses
    # no stale entry — it produces a disjoint cache file instead of a shape mismatch.
    cache_dir = str(
        latents_dir
        or opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "paired_latent_cache"))
    )
    base_tag = str(opt(cfg, "paired_reward.cache_tag", "paired_train"))
    cache_tag = paired_cache_tag(base_tag, target_dim, divisor)

    def _resolve_tag(vol_ds):
        """Pick the cache tag that actually hits this split's unique volumes (#148).

        The geometry-suffixed tag (issue #147) is preferred; but a cache warmed BEFORE
        the suffix was introduced (the plain legacy ``paired_train`` tag) at the SAME
        geometry is still valid — ``encode_fn=None`` below cannot re-encode, so a
        suffixed-only lookup would cache-miss on every such upgrade. Fall back to the
        legacy tag when the suffixed tag does NOT fully cover the split but the legacy
        tag does; the every-entry shape validation below then guards that the legacy
        entries truly match this geometry (a different-geometry legacy cache is
        rejected there, not silently reused).
        """
        from ..data.latent_dataset import _cache_path

        sids = vol_ds.unique_sample_ids()

        def _hits(tag):
            # File-existence probe only (NOT _load_cache): _load_cache deserializes
            # every latent tensor, which would add a full extra disk read per split
            # before warm_cache reads them again (codex #148 P2). warm_cache's own
            # _load_cache pass stays the single deserialization point.
            return sum(1 for sid in sids if _cache_path(cache_dir, sid, tag).is_file())
        if _hits(cache_tag) == len(sids):
            return cache_tag
        if base_tag != cache_tag and _hits(base_tag) == len(sids):
            rank_zero_info(
                "paired_reward: geometry-tagged cache miss; falling back to the legacy "
                "'%s' tag (shape-validated below).", base_tag,
            )
            return base_tag
        return cache_tag  # neither fully covers -> warm_cache raises a clear miss

    def _ds(manifest_split):
        vol_ds = PairedNiftiVolumeDataset(manifest_split, target_dim=target_dim, divisor=divisor)
        ds = PairedLatentDataset(vol_ds, encode_fn=None, cache_dir=cache_dir, cache_tag=_resolve_tag(vol_ds))
        ds.warm_cache(device, show_progress=False)
        # Scale-on-read uses the EXPORT scaling_factor verbatim (ADR-0021): the
        # generator trained on latents scaled by this factor, so the rollout
        # operates in its training space. Never re-estimate.
        ds.scaling_factor = float(scaling_factor)
        return ds

    train_ds, val_ds = _ds(train_manifest), _ds(val_manifest)

    # The paired_train latent cache is keyed by sample_id + cache_tag (NOT target_dim),
    # so a target_dim mismatch silently reuses stale wrong-shape latents (the cache was
    # built at the paired training's target_dim). When the dataset exposes the real
    # cache interface, validate EVERY entry's spatial shape (both splits) against the
    # reward config's target_dim / divisor and fail fast (codex #99 P2; check every
    # entry, not just the first, so a mixed/partial cache can't slip through - codex
    # #100 P2 round-3). Use CEIL division: PairedNiftiVolumeDataset zero-pads each
    # spatial dim up to a multiple of the divisor before encoding, so the latent
    # spatial is ceil(target_dim / divisor) (floor would false-positive on
    # non-divisible target_dim, codex #100 P2). Test fakes (no source/raw_latent)
    # bypass this - they don't model the encode.
    if hasattr(train_ds, "raw_latent") and hasattr(train_ds, "source"):
        expected_spatial = tuple(-(-d // divisor) for d in target_dim)  # ceil(d / divisor)
        for split_name, ds in (("train", train_ds), ("val", val_ds)):
            for sid in ds.source.unique_sample_ids():
                cached_spatial = tuple(ds.raw_latent(sid).shape[1:])
                if cached_spatial != expected_spatial:
                    raise ValueError(
                        f"Cached paired latent ({split_name}, {sid}) spatial shape "
                        f"{cached_spatial} does not match the reward config's "
                        f"target_dim={target_dim} / divisor={divisor} = {expected_spatial} "
                        f"(ceil). The paired_train cache was built with a different "
                        f"target_dim (or is a mixed/partial cache); point --latents-dir at "
                        f"a matching cache or re-warm it."
                    )

    num_steps = int(opt(cfg, "paired_reward.num_steps", 8))
    probe_num_steps = int(opt(cfg, "paired_reward.precompute_num_steps", num_steps))
    n_probe = int(opt(cfg, "paired_reward.n_probe", 64))
    gen_batch_size = int(opt(cfg, "paired_reward.gen_batch_size", 4))
    return build_paired_reward_inputs(
        train_ds=train_ds,
        val_ds=val_ds,
        generator=generator,
        base_scheduler=base_scheduler,
        controlnet=controlnet,
        num_steps=num_steps,
        probe_num_steps=probe_num_steps,
        n_probe=n_probe,
        batch_size=gen_batch_size,
        seed=int(opt(cfg, "random_seed", 0)),
        device=device,
    )
