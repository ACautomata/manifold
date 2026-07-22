"""``manifold-train-grpo`` console entry + the testable GRPO-training core (#56).

The console entry composes the experiment config, builds the :class:`GRPOModule`
(trainable policy UNet + frozen reward) + the data inputs (real JiT policy + reward,
or a fake via the ``data_provider`` injection seam for the CPU smoke), and calls
``Trainer.fit``. The GRPO job is standalone (issue #55 user story 24): decoupled from
JiT pretraining and from reward training, independently resumable.

The integration core :func:`run_grpo_training` (Module + datamodule +
``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a tiny CPU smoke
can drive it with a fake policy + toy conditioning (the issue's testing seam)
instead of the real JiT checkpoint + trained reward. The real-data launch path
(loading the raw JiT arm + the trained reward, wiring the latent cache) is gated on
the reward model clearing its probe + a tiny-config measurement (#59); the
``data_provider`` seam ships in #56.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.utilities.rank_zero import rank_zero_info  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import opt
from ..data.datamodule import build_datamodule
from ..modules.grpo import GRPOModule
from ..schedulers.scheduling_flow_match_grpo import FlowMatchGRPOScheduler
from manifold.training.callbacks import CallbackContext, CheckpointSpec, FIDSpec, TrainLossSpec
from manifold.training.core import TrainingSpine


@dataclass
class GRPOInputs:
    """Module-construction + data inputs for one GRPO run.

    ``policy`` / ``reward_model`` / ``scheduler`` go to the :class:`GRPOModule` ctor;
    ``train_ds`` / ``val_ds`` emit manifold conditioning (``{spacing, label}``);
    ``latent_shape`` is the rollout latent shape (GRPO is generative — the Module
    samples the group noise). The ``data_provider`` seam injects a fake policy + toy
    conditioning for the CPU smoke; the real path (``_real_inputs``, #59) loads the
    raw JiT arm + the trained reward.

    Optional FID-validation inputs (#58): when ``vae`` / ``real_latents`` /
    ``feature_net`` are ALL present, :func:`run_grpo_training` attaches
    :class:`~manifold.metrics.FIDCallback` (no EMA) and selects the checkpoint on
    ``val/fid`` (mode ``min``) — the anti-reward-hacking screen. When any is ``None``
    (the CPU smoke, or a reward-only run) validation stays ``val/mean_reward`` only.
    """

    policy: Any
    reward_model: Any
    scheduler: Any
    train_ds: Any
    val_ds: Any
    latent_shape: Sequence[int]
    reference_policy: Any = None  # the frozen KL anchor (ADR-0015); None ⇒ no KL (v1)
    controlnet: Any = None  # the trainable ControlNet (set iff --native-dir is a ControlNet export)
    vae: Any = None
    real_latents: Any = None
    feature_net: Any = None
    feature_net_factory: Any = None  # L3: lazy build on the rank-0-gated FID stage
    fid_modality: int = 1
    fid_spacing: Sequence[float] = (1.0, 1.0, 1.0)


def run_grpo_training(
    *,
    module: GRPOModule,
    inputs: GRPOInputs,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    save_top_k: int = 1,
    monitor_metric: str | None = None,
    mode: str | None = None,
    limit_val_batches: int | float = 1.0,
    limit_train_batches: int | float | None = None,
    seed: int = 0,
    ckpt_path: str | None = None,
    num_synth: int = 16,
    center_slices_ratio: float = 0.5,
    cov_ridge: float = 1e-6,
    callback_names: list[str] | None = None,
    callback_cfg: dict[str, dict] | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the GRPO module (the core seam).

    The shell (ADR-0032): seed, the FID-triple decision, the monitor/mode
    derivation, then delegate to :class:`TrainingSpine.run` — the single caller of
    the callback registry. When ``inputs`` carries the FID triple
    (``vae`` / ``real_latents`` / ``feature_net``) an :class:`FIDSpec` callback is
    attached (no EMA — GRPO evaluates the raw policy, #59) and the checkpoint
    selects on ``val/fid`` (min); otherwise validation is ``val/mean_reward`` (max).
    Returns ``(trainer, ckpt)`` so callers can find the written ``.ckpt``.

    When the ControlNet is present (``module.controlnet`` set — the policy was inferred
    from a ControlNet export, ADR-0034), the unconditional FID is suppressed two ways:
    at default-derivation (the constant frozen-base metric is skipped) AND post-merge
    via :class:`TrainingSpine`'s ``forbidden_callbacks`` / ``forbidden_monitors``
    — a YAML / CLI ``--callbacks fid`` override or a ``val/fid`` monitor cannot
    re-enable it.

    Args:
        inputs: the train/val conditioning datasets + the policy/reward/scheduler
            (+ optional FID triple).
        monitor_metric / mode: ``None`` (the default) auto-selects ``val/fid`` (min)
            when the FID triple is present, else ``val/mean_reward`` (max). Pass
            explicitly to override.
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
        callback_names: optional CLI ``--callbacks`` override (a full name-list
            replacement of the derived defaults; ADR-0032).
        callback_cfg: optional ``{name: {knob: value}}`` YAML knob overrides,
            merged over the shell-derived defaults (the YAML ``callbacks:`` block).
    """
    pl.seed_everything(seed, workers=True)
    fid_active = (
        inputs.vae is not None
        and inputs.real_latents is not None
        and (inputs.feature_net is not None or inputs.feature_net_factory is not None)
    )
    # When the ControlNet is present the base UNet is frozen and only the ControlNet
    # trains, but the FID callback's unconditional `module.sample()` rollout ignores
    # the ControlNet — so val/fid would be a CONSTANT frozen-base metric, independent
    # of the learned policy. Selecting on it is meaningless; skip the unconditional
    # FID and select on the conditional reward (val/mean_reward). The post-merge
    # forbidden policy (below) makes this airtight against YAML/CLI overrides that
    # try to re-add it.
    controlnet_present = getattr(module, "controlnet", None) is not None
    if controlnet_present and fid_active:
        rank_zero_info(
            "ControlNet policy (frozen base): skipping unconditional FID — it "
            "ignores the ControlNet (a constant frozen-base metric). Monitoring "
            "val/mean_reward (the conditional-reward progress signal) instead."
        )
        fid_active = False
    if monitor_metric is None:
        monitor_metric = "val/fid" if fid_active else "val/mean_reward"
    # mode is derived from the FINAL (post-merge) monitor below, unless the caller
    # (or a YAML checkpoint.mode knob) supplies it explicitly — so a YAML
    # ``checkpoint.monitor_metric=val/mean_reward`` override without a ``mode``
    # does not keep the shell's ``val/fid``-derived ``min`` (codex #183 P1).

    datamodule = build_datamodule(
        inputs.train_ds, batch_size=batch_size, val_dataset=inputs.val_ds, num_workers=num_workers
    )

    # Shell-derived callback knobs; YAML/CLI callback_cfg wins (merged over).
    cfg_built: dict[str, dict] = {
        "checkpoint": {
            "monitor_metric": monitor_metric,
            "save_top_k": save_top_k,
        }
    }
    if mode is not None:
        cfg_built["checkpoint"]["mode"] = mode
    default_names = ["checkpoint"]
    if fid_active:
        default_names.append("fid")
        cfg_built["fid"] = {
            "num_synth": num_synth,
            "center_slices_ratio": center_slices_ratio,
            "cov_ridge": cov_ridge,
        }
    for name, knobs in (callback_cfg or {}).items():
        cfg_built.setdefault(name, {}).update(knobs)

    # Derive mode + filename from the FINAL (post-merge) monitor unless the merged
    # cfg supplied them explicitly. ``val/fid`` is min; every other GRPO monitor
    # (val/mean_reward, or a YAML override) is max. An unmonitored checkpoint keeps
    # the plain prefix.
    ckpt_cfg = cfg_built["checkpoint"]
    final_monitor = ckpt_cfg.get("monitor_metric")
    if ckpt_cfg.get("mode") is None:
        ckpt_cfg["mode"] = "min" if final_monitor == "val/fid" else "max"
    if ckpt_cfg.get("filename") is None:
        ckpt_cfg["filename"] = (
            f"grpo-{{epoch:03d}}-{{{final_monitor}:.3f}}"
            if final_monitor is not None
            else "grpo-{epoch:03d}"
        )

    # GRPO's FID reference is the held real_latents (ADR-0032) — its conditioning
    # datamodule carries no val_latents, so real_latents is passed explicitly and
    # FIDSpec.build forwards it to FIDCallback. Built only when FID is active.
    inference_recipe = None
    if fid_active:
        inference_recipe = {
            "latent_shape": (1, *module.latent_shape),
            "spacing": list(inputs.fid_spacing),
            "modality": int(inputs.fid_modality),
            "num_inference_steps": module.num_steps,
            "guidance_scale": 1.0,
            "cfg_interval": None,
        }

    ctx = CallbackContext(
        module=module,
        vae=inputs.vae,
        datamodule=datamodule,
        inference_recipe=inference_recipe,
        model_dir=model_dir,
        seed=seed,
        feature_net=inputs.feature_net,
        feature_net_factory=inputs.feature_net_factory,
        real_latents=inputs.real_latents,
    )

    forbidden_callbacks = None
    forbidden_monitors = None
    if controlnet_present:
        reason = (
            "ControlNet policy (frozen base): unconditional FID ignores the "
            "ControlNet — a constant frozen-base metric, independent of the policy"
        )
        forbidden_callbacks = {"fid": reason}
        forbidden_monitors = {"val/fid": reason}

    spine = TrainingSpine()
    spine.registry.register("train_loss", TrainLossSpec)
    spine.registry.register("fid", FIDSpec)
    spine.registry.register("checkpoint", CheckpointSpec)
    return spine.run(
        module=module,
        datamodule=datamodule,
        ctx=ctx,
        default_names=default_names,
        max_epochs=max_epochs,
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches,
        extra_trainer_kwargs=(
            {"limit_train_batches": limit_train_batches}
            if limit_train_batches is not None
            else None
        ),
        ckpt_path=ckpt_path,
        callback_cfg=cfg_built,
        callback_names_override=callback_names,
        forbidden_callbacks=forbidden_callbacks,
        forbidden_monitors=forbidden_monitors,
    )


# -- console entry -----------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manifold-train-grpo",
        description="Post-train the JiT x0-denoiser with Granular-GRPO against the frozen reward.",
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c", "--train", default="configs/train/config_grpo.yaml", help="GRPO recipe YAML."
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (latent metadata)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--native-dir",
        default=None,
        help="native JiT export dir (the raw-arm policy + VAE scale); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--reward-path",
        default=None,
        help="trained RewardModel checkpoint (.ckpt); required without --data-provider.",
    )
    parser.add_argument(
        "--latents-dir",
        default=None,
        help="latent cache dir (conditioning + the FID real reference); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="run a tiny-config measurement (it/s + peak GPU memory) and exit — the "
        "#59 launch gate; size G / eta_step_list / n_epochs before the full run.",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="cap train batches/epoch (a debug knob for the fast v2 re-measure; the full "
             "run leaves it unset).",
    )
    parser.add_argument(
        "--callbacks",
        default=None,
        help="comma-separated callback names; REPLACES the YAML callbacks: list (ADR-0032).",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config → build → ``run_grpo_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> GRPOInputs`` returning a fake policy + toy conditioning so the
    full ``main`` path runs without the real JiT checkpoint + trained reward. The real
    path loads the raw JiT arm from ``--native-dir`` and the reward from
    ``--reward-path`` (#59).
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths, resolve_callback_names

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # GRPO needs no VAE / image data — only the output model_dir is required.
    require_paths(cfg, keys=("model_dir",))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "grpo_train", None) is None:
        raise ValueError(
            "Config has no `grpo_train` block — use the GRPO recipe "
            "(-c configs/train/config_grpo.yaml), not a JiT train or reward config."
        )

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    # Pin each DDP rank to its own GPU before _real_inputs runs the policy rollout
    # on `device` (GPU inference happens here, before trainer.fit sets up DDP).
    # Without this every rank lands on cuda:0 and serializes on one GPU, blowing
    # past the DDP init timeout (sugon 8-DCU).
    if torch.cuda.is_available():
        _lr = int(os.environ.get("LOCAL_RANK", 0))
        if _lr < torch.cuda.device_count():
            torch.cuda.set_device(_lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        inputs = data_provider(cfg, device)
    else:
        # --native-dir / --reward-path / --latents-dir are NOT argparse-required: that
        # would break the data_provider injection seam (the CPU smoke). Validate them
        # here, only on the real path.
        if not args.native_dir or not args.reward_path or not args.latents_dir:
            raise ValueError(
                "GRPO needs --native-dir <native export>, --reward-path <trained "
                "RewardModel .ckpt>, and --latents-dir <latent cache> (or inject a "
                "data_provider for the smoke). Point --native-dir at a raw JiT export "
                "to train the UNet policy, or a supervised ControlNet export (frozen "
                "base + ControlNet) to train the ControlNet policy — the policy is "
                "inferred from the export, not a flag (ADR-0034)."
            )
        # The ControlNet path builds the paired conditioning from data_base_dir (the
        # BraTS manifest); the UNet path reads the latent-cache files directly and needs
        # no data dir. Discriminate up front so a missing data_base_dir fails fast with
        # CLI-override guidance instead of a deep MissingMandatoryValue.
        if _detect_controlnet_export(args.native_dir):
            require_paths(cfg, keys=("model_dir", "data_base_dir"))
        inputs = _real_inputs(cfg, args.native_dir, args.reward_path, args.latents_dir, device)

    gcfg = cfg.grpo_train
    latent_shape = tuple(int(s) for s in opt(gcfg, "latent_shape", [4, 64, 64, 32]))
    module = GRPOModule(
        inputs.policy,
        inputs.reward_model,
        inputs.scheduler if inputs.scheduler is not None else FlowMatchGRPOScheduler(eta=float(opt(gcfg, "eta", 0.7))),
        G=int(opt(gcfg, "G", 8)),
        eta_step_list=list(opt(gcfg, "eta_step_list", [0, 1, 2, 3, 4, 5, 6, 7])),
        clip_range=float(opt(gcfg, "clip_range", 1e-4)),
        lr=float(gcfg.lr),
        adv_clip_max=float(opt(gcfg, "adv_clip_max", 5.0)),
        num_steps=int(opt(gcfg, "num_steps", 15)),
        latent_shape=latent_shape,
        reference_policy=inputs.reference_policy,
        kl_coef=float(opt(gcfg, "kl_coef", 0.0)),
        reward_bound=str(opt(gcfg, "reward_bound", "none")),
        reward_temp=float(opt(gcfg, "reward_temp", 8.0)),
        controlnet=inputs.controlnet,
        freeze_unet=inputs.controlnet is not None,
    )

    if args.measure:
        # The #59 launch-gate measurement: a 1-epoch fit timing + peak GPU memory,
        # to size G / eta_step_list / n_epochs on the target cluster before the full run.
        it_per_s, peak, elapsed = run_grpo_measurement(
            module=module,
            inputs=inputs,
            model_dir=str(cfg.model_dir),
            devices=args.num_gpus if args.num_gpus > 1 else 1,
            batch_size=int(gcfg.batch_size),
            seed=seed,
        )
        print(
            f"[manifold-train-grpo] measure: {it_per_s:.3f} it/s | "
            f"peak GPU {peak / 1e9:.2f} GB | {elapsed:.1f}s"
        )
        return 0

    # ADR-0032: the CLI ``--callbacks`` (comma list) REPLACES the YAML
    # ``callbacks:`` name list; the YAML ``fid:`` / ``checkpoint:`` knob blocks
    # forward as the callback_cfg override. Both are ``None``/empty when neither is
    # supplied, so the shell uses its derived defaults (unchanged behaviour).
    cb_names = resolve_callback_names(args.callbacks, cfg)
    callback_cfg: dict[str, dict] = {"checkpoint": dict(opt(cfg, "checkpoint", {}))}
    fid_block = opt(cfg, "fid", None)
    if fid_block is not None:
        callback_cfg["fid"] = dict(fid_block)

    run_grpo_training(
        module=module,
        inputs=inputs,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or gcfg.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(gcfg.batch_size),
        seed=seed,
        ckpt_path=args.resume,
        limit_train_batches=args.limit_train_batches,
        callback_names=cb_names,
        callback_cfg=callback_cfg,
    )
    print(f"[manifold-train-grpo] done; checkpoints under {cfg.model_dir}")
    return 0


def run_grpo_measurement(
    *,
    module: GRPOModule,
    inputs: GRPOInputs,
    model_dir: str,
    devices: int | str = 1,
    accelerator: str = "auto",
    batch_size: int = 2,
    seed: int = 0,
) -> tuple[float, int, float]:
    """Time a 1-epoch GRPO fit + report it/s + peak GPU memory (the #59 launch gate).

    Sizes ``G`` / ``eta_step_list`` / ``n_epochs`` by measuring the real budget's
    throughput + peak GPU memory on the target cluster before committing to the full
    run. Returns ``(it_per_s, peak_gpu_bytes, elapsed_s)``. Peak memory is 0 off-CUDA
    (the read is GPU-only); a tiny ``--measure`` run on the cluster is the real signal.
    """
    import time

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer, _ = run_grpo_training(
        module=module,
        inputs=inputs,
        model_dir=model_dir,
        max_epochs=1,
        devices=devices,
        accelerator=accelerator,
        batch_size=batch_size,
        seed=seed,
    )
    elapsed = time.perf_counter() - start
    it_per_s = float(trainer.global_step) / elapsed if elapsed > 0 else float("nan")
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return it_per_s, peak, elapsed


def _detect_controlnet_export(native_dir) -> bool:
    """Whether the native export under ``native_dir`` is a ControlNet export.

    The unified real-input builder's discriminator (issue #177 / ADR-0034): reads
    the pipeline's ``model_index.json`` (the per-component layout BOTH
    :class:`~manifold.LatentFlowPipeline` and
    :class:`~manifold.ControlNetLatentFlowPipeline` write on ``save_pretrained``)
    and keys on a declared ``controlnet`` component. This is INDEPENDENT of
    :func:`~manifold.training.controlnet_inputs.load_frozen_controlnet_generator`,
    which ASSUMES ControlNet-ness (it accesses ``pipe.controlnet`` directly) and
    therefore cannot detect it — the discriminator must run BEFORE the loader is
    selected.

    - A ControlNet export lists a ``controlnet`` component AND carries a
      ``controlnet/`` subdir → ``True`` (build the frozen-base + ControlNet path).
    - A raw JiT export lists no ``controlnet`` component → ``False`` (build the
      trainable-UNet path).
    - A dir with no ``model_index.json`` → ``False``: the discriminator only owns
      the ControlNet-vs-JiT routing, so the downstream loader raises its own clear
      "not a manifold pipeline directory" error.
    - A dir that DECLARES a controlnet component but carries no ``controlnet/``
      subdir is a corrupt / half-written export → fail fast with a clear error,
      rather than silently routing it to the UNet path (which would mis-train) or
      crashing deep in ``from_pretrained``.
    """
    index_path = os.path.join(str(native_dir), "model_index.json")
    if not os.path.isfile(index_path):
        return False
    with open(index_path) as f:
        index = json.load(f)
    if "controlnet" not in index.get("components", {}):
        return False
    if not os.path.isdir(os.path.join(str(native_dir), "controlnet")):
        raise FileNotFoundError(
            f"{str(native_dir)!r} declares a ControlNet component in model_index.json "
            f"but carries no controlnet/ subdir — the export is incomplete. Re-export "
            f"with ControlNetLatentFlowPipeline.save_pretrained (the supervised "
            f"stage-1 artifact, ADR-0027) or point --native-dir at a complete export."
        )
    return True


def _real_inputs(
    cfg, native_dir: str, reward_path: str, latents_dir: str, device: torch.device
) -> GRPOInputs:
    """Build the real GRPO inputs, inferring the policy from the native artifact (#177).

    The unified real-input builder (ADR-0034): no ``--grpo-mode`` flag — the policy
    is inferred from the native export under ``--native-dir``. A ControlNet export
    (a ``controlnet`` component in its ``model_index.json``) builds the frozen-base +
    trainable-ControlNet path with paired conditioning; a raw JiT export builds the
    trainable-UNet path. The discriminator (:func:`_detect_controlnet_export`) runs
    BEFORE either loader is selected — ``load_frozen_controlnet_generator`` assumes
    ControlNet-ness and so cannot be the discriminator.

    ``reward_path`` / ``latents_dir`` are NOT argparse-required (that would break the
    ``data_provider`` injection seam for the CPU smoke); they are validated in
    :func:`main`, only on the real path.
    """
    if _detect_controlnet_export(native_dir):
        return _controlnet_real_inputs(cfg, native_dir, reward_path, latents_dir, device)
    return _unet_real_inputs(cfg, native_dir, reward_path, latents_dir, device)


def _load_frozen_reward(cfg, reward_path: str, device: torch.device, *, in_channels: int,
                        latent_c: int | None = None):
    """Load the frozen :class:`RewardModel` from its RewardModule checkpoint (shared).

    Shared by both real-input paths: the reward scores the terminal latent ``z_K``
    unconditionally (``reward(z_K)``, ``in_channels = C_latent``) for both the UNet
    and the ControlNet policy. ``in_channels`` is the RewardModel's input-channel
    count (the network config's ``reward_model.in_channels`` for the UNet path, the
    latent channel count for the ControlNet path). ``latent_c``, when given (the
    ControlNet path), validates the checkpoint's first conv against the single-latent
    contract — a 2·C condition-aware paired-reward ckpt is incompatible and fails fast
    with a readable error (codex #151). ``None`` (the UNet path) skips the check.

    weights_only=True first (no arbitrary-code-execution risk); fall back to False only
    for the weights_only restriction itself (non-allowlisted globals in a full Lightning
    ckpt's callback/optimizer state), NOT for file/IO errors (which must surface).
    ``reward_path`` is the user's OWN trained reward (trusted — mirroring
    ``export_to_native``; never point this at an untrusted ``.ckpt``).
    """
    import pickle

    from ..models.reward_model import RewardModel

    reward_cfg = opt(cfg, "reward_model", {})
    reward_model = RewardModel(
        spatial_dims=int(opt(reward_cfg, "spatial_dims", 3)),
        in_channels=int(in_channels),
        channels=int(opt(reward_cfg, "channels", 64)),
        num_layers_d=int(opt(reward_cfg, "num_layers_d", 3)),
        norm=str(opt(reward_cfg, "norm", "BATCH")),
    )
    try:
        ckpt = torch.load(str(reward_path), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, ValueError):
        rank_zero_info("reward ckpt %s needs weights_only=False (non-tensor state); loading trusted.", reward_path)
        ckpt = torch.load(str(reward_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    reward_sd = {k[len("reward_model."):]: v for k, v in state.items() if k.startswith("reward_model.")}
    if not reward_sd:
        raise ValueError(
            f"No 'reward_model.*' keys in {reward_path} — not a trained RewardModule checkpoint."
        )
    if latent_c is not None:
        # Fail fast on a channel mismatch BEFORE load_state_dict's cryptic shape error:
        # the z_K reward is scored unconditionally (in_channels = C_latent), so a 2·C
        # condition-aware paired-reward ckpt (from manifold-train-paired-reward) is
        # incompatible. Detect via the first conv's input-channel dim (shape [out, in, ...]).
        first_conv = reward_sd.get("discriminator.initial_conv.conv.weight")
        if first_conv is not None and int(first_conv.shape[1]) != latent_c:
            raise ValueError(
                f"Reward checkpoint {reward_path} has in_channels={int(first_conv.shape[1])}, "
                f"but the z_K reward is scored unconditionally and needs a single-latent reward "
                f"(in_channels={latent_c}). This looks like a 2·C condition-aware paired-reward "
                "ckpt (manifold-train-paired-reward); the ControlNet policy does NOT concat x_src "
                "into the reward (the policy x0 sees x_src instead). Point --reward-path at a "
                "single-latent RewardModule checkpoint."
            )
    reward_model.load_state_dict(reward_sd, strict=True)
    reward_model.eval().to(device)
    for p in reward_model.parameters():
        p.requires_grad_(False)
    return reward_model


def _unet_real_inputs(
    cfg, native_dir: str, reward_path: str, latents_dir: str, device: torch.device
) -> GRPOInputs:
    """Build the real GRPO inputs for a raw JiT export (the trainable-UNet policy).

    The raw-arm JiT UNet path, selected by :func:`_real_inputs` when the native
    export is NOT a ControlNet export. The raw-arm JiT UNet (ADR-0006:
    ``export_to_native`` bakes the raw ``state_dict`` weights) is the **trainable**
    policy. The trained :class:`RewardModel` (a ``RewardModule`` Lightning checkpoint
    — its ``reward_model.*`` state_dict) is the **frozen** reward. The latent cache
    furnishes the manifold conditioning (``{spacing, label}`` — GRPO is generative, so
    there is no clean-latent target) over train subjects AND the fixed real-reference
    latents (val subjects, scaled) the FID callback decodes. The RadImageNet
    ``feature_net`` completes the FID triple (#58) — val/fid is meaningful for the
    UNet policy.

    Launch is gated on the ns15 reward clearing ``val/gen_pair_acc > 0.8`` + a
    ``--measure`` run on the target cluster (#59) — not exercisable here (no real
    artifacts on the dev machine), so this mirrors ``reward_cli._real_inputs``'s
    pattern (the data_provider seam covers the CPU smoke).
    """
    from torch.utils.data import Dataset

    from ..data.reward_pairs import load_cached_latents, partition_subjects
    from ..metrics import make_feature_network
    from ..pipelines.latent_flow import LatentFlowPipeline

    # 1. Raw-arm JiT UNet (the trainable policy) + VAE (carries scaling_factor).
    pipe = LatentFlowPipeline.from_pretrained(str(native_dir))
    policy = pipe.unet.to(device)
    for p in policy.parameters():  # GRPO post-trains the policy (the reward is frozen).
        p.requires_grad_(True)
    vae = pipe.vae
    scaling_factor = float(vae.scaling_factor)
    # The frozen KL anchor (ADR-0015): a snapshot of the pretrained policy taken BEFORE
    # any GRPO update. ``copy.deepcopy`` (not a second from_pretrained) so it is
    # bit-identical to the trainable arm's starting weights; the Module freezes + unregisters it.
    reference_policy = copy.deepcopy(policy)

    # 2. Trained RewardModel (frozen) from its RewardModule Lightning checkpoint.
    # The reward_model architecture comes from the network config (``-t``) — opt()
    # falls back to the RewardModel defaults if a non-standard network file omits it.
    reward_cfg = opt(cfg, "reward_model", {})
    reward_model = _load_frozen_reward(
        cfg, reward_path, device, in_channels=int(opt(reward_cfg, "in_channels", 4))
    )

    # 3. Conditioning ({spacing, label}) + real_latents (val subjects, scaled) from cache.
    subject_regex = opt(cfg, "grpo.subject_regex", None)
    items, subject_ids = load_cached_latents(str(latents_dir), subject_regex)
    train_subjects, val_subjects = partition_subjects(
        subject_ids,
        val_fraction=float(opt(cfg, "grpo.val_fraction", 0.2)),
        seed=int(opt(cfg, "random_seed", 0)),
    )
    train_items = [it for it, sid in zip(items, subject_ids) if sid in train_subjects]
    val_items = [it for it, sid in zip(items, subject_ids) if sid in val_subjects]
    if not train_items or not val_items:
        raise ValueError(
            f"Empty train ({len(train_items)}) / val ({len(val_items)}) subject split "
            f"from {len(items)} latents — adjust grpo.val_fraction."
        )
    rank_zero_info(
        "GRPO real inputs: %d train / %d val conditioning subjects (%d / %d latents).",
        len(train_subjects), len(val_subjects), len(train_items), len(val_items),
    )

    class _CondDS(Dataset):
        """Manifold conditioning only (GRPO is generative — the rollout samples noise).

        ``spacing`` / ``label`` are normalized to tensors on read so the default
        collate batches them: cache files may store ``spacing`` as a plain list,
        which the default collate would transpose into a list-of-tensors and break
        ``_conditioning_tensors``'s ``torch.as_tensor``."""

        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return {
                "spacing": torch.as_tensor(self.items[i]["spacing"], dtype=torch.float32),
                "label": torch.as_tensor(self.items[i]["label"], dtype=torch.long),
            }

    # Cap the FID real-reference subset (mirrors the JiT cli): FIDCallback decodes the
    # whole ``real_latents`` tensor in ONE ``_real_planes()`` pass, so an unbounded val
    # set would OOM the FID phase. Seeded prefix ⇒ the reference is fixed across runs.
    val_subset_size = int(opt(cfg, "val_subset_size", 32))
    g = torch.Generator().manual_seed(0)
    val_idx = torch.randperm(len(val_items), generator=g)[:val_subset_size].tolist()
    real_latents = torch.stack([val_items[i]["latent"] for i in val_idx]).float() * scaling_factor

    # The FID backbone IS the anti-reward-hacking screen — fail closed if it cannot
    # load (the launch requires it). The data_provider smoke bypasses _real_inputs.
    feature_net = make_feature_network("resnet50")

    # Preserve the native scheduler's transport settings (the t_eps /
    # num_train_timesteps the JiT checkpoint trained/exported with); only eta is the
    # GRPO addition. Building with defaults would mismatch a non-default t_eps export.
    sched_cfg = pipe.scheduler.config
    scheduler = FlowMatchGRPOScheduler(
        num_train_timesteps=int(sched_cfg.get("num_train_timesteps", 1000)),
        t_eps=float(sched_cfg.get("t_eps", 0.05)),
        eta=float(opt(cfg.grpo_train, "eta", 0.7)),
    )

    latent_shape = tuple(int(s) for s in opt(cfg.grpo_train, "latent_shape", [4, 64, 64, 32]))
    return GRPOInputs(
        policy=policy,
        reward_model=reward_model,
        scheduler=scheduler,
        train_ds=_CondDS(train_items),
        val_ds=_CondDS(val_items),
        latent_shape=latent_shape,
        reference_policy=reference_policy,
        vae=vae,
        real_latents=real_latents,
        feature_net=feature_net,
        fid_modality=int(opt(cfg, "grpo.modality", 1)),
        fid_spacing=list(opt(cfg, "grpo.spacing", [1.0, 1.0, 1.0])),
    )


def _controlnet_real_inputs(
    cfg, native_dir: str, reward_path: str, latents_dir: str, device: torch.device
) -> GRPOInputs:
    """Build the real GRPO inputs for a ControlNet export (frozen base + trainable ControlNet).

    The ControlNet path, selected by :func:`_real_inputs` when the native export IS a
    ControlNet export. Mirrors ``controlnet_cli._real_inputs``'s paired-input
    construction, but starts GRPO from a **supervised ControlNet export**
    (ADR-0027 stage 1):

    - ``--native-dir`` is the **ControlNet native export** (frozen base UNet +
      ControlNet + VAE + scheduler — the layout ``ControlNetLatentFlowPipeline``
      writes / #144 bakes). ``load_frozen_controlnet_generator`` loads both arms
      frozen; we then **unfreeze the ControlNet** (the only trainable arm) and keep
      the base UNet frozen (``requires_grad_(False)`` — the GRPOModule also holds it
      unregistered, off the optimizer/checkpoint).
    - The conditioning is the **paired** latent cache (``--latents-dir``): each batch
      carries ``{src_latent, src_label, tgt_label, spacing}`` (the ControlNet control
      signal + the translation direction). Warmed over the paired train/val split via
      the same ``paired_train`` cache as the reward/supervised stages.
    - The reward scores the terminal latent ``z_K`` **unconditionally** — the same
      single-latent reward (``in_channels = C_latent``) the UNet path uses, loaded
      from a ``RewardModule`` ``.ckpt``. The ControlNet policy does NOT use the 2·C
      condition-aware paired reward: its conditional fidelity is driven by the policy
      x0 (which sees ``x_src``), not by the reward input.

    Scale-consistency (ADR-0021): scale-on-read uses the ControlNet export's
    ``vae.scaling_factor`` verbatim (never re-estimated).
    """
    from torch.utils.data import Dataset

    from ..config import autoencoder_divisor
    from ..data.paired_brats import build_brats_pair_manifest
    from ..data.paired_latent_dataset import PairedLatentDataset
    from ..data.paired_manifests import _train_val_manifests
    from ..data.paired_volume_dataset import PairedNiftiVolumeDataset
    from ..schedulers.scheduling_flow_match_grpo import FlowMatchGRPOScheduler
    from .controlnet_inputs import load_frozen_controlnet_generator

    # 1. Frozen base UNet + ControlNet + base scheduler + scaling_factor from the
    # ControlNet native export (the supervised stage-1 artifact, ADR-0027). Both arms
    # come back frozen + eval; the ControlNet is then made the ONLY trainable arm.
    base, controlnet, base_scheduler, scaling_factor = load_frozen_controlnet_generator(native_dir)
    base.to(device).eval()
    for p in base.parameters():
        p.requires_grad_(False)  # the base stays frozen (held unregistered by GRPOModule)
    controlnet.to(device)
    for p in controlnet.parameters():
        p.requires_grad_(True)  # the ControlNet is the only trainable arm
    # The frozen KL anchor (ADR-0015): a deepcopy of the initial (base, controlnet)
    # pair taken BEFORE any GRPO update. The recipe enables kl_coef (anti-reward-hacking);
    # without this reference the GRPOModule's _transition_kl early-returns and the
    # anchor silently disables (codex #151 P1). The Module freezes + unregisters both.
    reference_policy = (copy.deepcopy(base), copy.deepcopy(controlnet))

    # 2. Reward (frozen) from its RewardModule Lightning checkpoint — validated BEFORE
    # the (expensive) paired cache warm so a bad reward ckpt fails fast (codex #151 /
    # issue #177 AC7). The ControlNet policy scores the terminal latent z_K
    # UNCONDITIONALLY (reward(z_K), in_channels = C_latent) — the same single-latent
    # reward the UNet path uses, NOT the 2·C condition-aware paired reward. The
    # ControlNet's conditional fidelity is driven by the policy x0 (which sees x_src),
    # not by the reward input. latent_c is validated against the single-latent contract.
    latent_c = int(opt(cfg, "latent_channels", 4))
    reward_model = _load_frozen_reward(
        cfg, reward_path, device, in_channels=latent_c, latent_c=latent_c
    )

    # 3. Paired conditioning: the paired train/val split warmed from the paired latent
    # cache. The cache carries UNSCALED latents; scale-on-read uses the export factor.
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
    # Mirror the recipe's grpo.val_fraction to the root key _train_val_manifests reads
    # (only when the native-split dir path is not taken and the root key is unset).
    import os

    from omegaconf import OmegaConf

    val_dir = opt(cfg, "val_data_base_dir", None)
    if not (val_dir and os.path.isdir(str(val_dir))):
        if OmegaConf.select(cfg, "val_fraction", default=None) is None:
            g_val_fraction = float(opt(cfg, "grpo.val_fraction", 0.0))
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"val_fraction": g_val_fraction}))
    train_manifest, val_manifest = _train_val_manifests(cfg, manifest)
    if not val_manifest:
        raise ValueError(
            "The ControlNet GRPO path needs a held-out val split (val_data_base_dir set, or "
            "val_fraction > 0); train data is never reused as val."
        )

    cache_dir = str(
        latents_dir
        or opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "paired_latent_cache"))
    )
    cache_tag = str(opt(cfg, "grpo.cache_tag", "paired_train"))

    def _ds(manifest_split):
        vol_ds = PairedNiftiVolumeDataset(manifest_split, target_dim=target_dim, divisor=divisor)
        ds = PairedLatentDataset(vol_ds, encode_fn=None, cache_dir=cache_dir, cache_tag=cache_tag)
        ds.warm_cache(device, show_progress=False)
        ds.scaling_factor = float(scaling_factor)  # the export factor verbatim (ADR-0021)
        return ds

    train_ds, val_ds = _ds(train_manifest), _ds(val_manifest)

    # The paired cache is keyed by sample_id + cache_tag (NOT target_dim), so a
    # --latents-dir pointing at a cache warmed at a DIFFERENT target_dim silently
    # reuses wrong-shape src latents — the ControlNet then fails at the first
    # forward (its sampled rollout shape != the cached src shape). Mirror the
    # every-entry shape check paired_reward_cli._real_inputs uses (codex #151 P2):
    # validate EVERY unique latent's spatial shape (both splits) against this
    # recipe's target_dim / divisor and fail fast. CEIL division (the volume dataset
    # zero-pads each dim up to a divisor multiple before encoding, so the latent
    # spatial is ceil(target_dim / divisor)). Test fakes (no source/raw_latent)
    # bypass this — they don't model the encode.
    if hasattr(train_ds, "raw_latent") and hasattr(train_ds, "source"):
        expected_spatial = tuple(-(-d // divisor) for d in target_dim)  # ceil(d / divisor)
        for split_name, ds in (("train", train_ds), ("val", val_ds)):
            for sid in ds.source.unique_sample_ids():
                cached_spatial = tuple(ds.raw_latent(sid).shape[1:])
                if cached_spatial != expected_spatial:
                    raise ValueError(
                        f"Cached paired latent ({split_name}, {sid}) spatial shape "
                        f"{cached_spatial} does not match the ControlNet recipe's "
                        f"target_dim={target_dim} / divisor={divisor} = {expected_spatial} "
                        f"(ceil). The paired cache was built with a different target_dim "
                        f"(or is a mixed/partial cache); point --latents-dir at a matching "
                        f"cache or re-warm it."
                    )

    # GRPO is generative — the rollout samples the group noise; the batch carries only
    # the ControlNet condition ({src_latent, src_label, tgt_label, spacing}). Normalize
    # the labels/spacing to tensors on read so the default collate batches them.
    class _PairedCondDS(Dataset):
        def __init__(self, paired_ds):
            self._ds = paired_ds

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, i):
            item = self._ds[i]
            return {
                "src_latent": item["src_latent"],
                "spacing": torch.as_tensor(item["spacing"], dtype=torch.float32),
                "src_label": torch.as_tensor(item["src_label"], dtype=torch.long),
                "tgt_label": torch.as_tensor(item["tgt_label"], dtype=torch.long),
            }

    # 4. GRPO scheduler preserving the export's transport settings (t_eps /
    # num_train_timesteps); only eta is the GRPO addition (mirrors _unet_real_inputs).
    sched_cfg = base_scheduler.config
    scheduler = FlowMatchGRPOScheduler(
        num_train_timesteps=int(sched_cfg.get("num_train_timesteps", 1000)),
        t_eps=float(sched_cfg.get("t_eps", 0.05)),
        eta=float(opt(cfg.grpo_train, "eta", 0.7)),
    )

    latent_shape = tuple(int(s) for s in opt(cfg.grpo_train, "latent_shape", [4, 64, 64, 32]))
    rank_zero_info(
        "GRPO ControlNet real inputs: frozen base + trainable ControlNet (%d train / %d val "
        "paired conditioning); reward on z_K (in_channels=%d); scale_factor=%.6f (export).",
        len(train_ds), len(val_ds), latent_c, float(scaling_factor),
    )
    return GRPOInputs(
        policy=base,  # the FROZEN base (GRPOModule holds it unregistered on the ControlNet path)
        reward_model=reward_model,
        scheduler=scheduler,
        train_ds=_PairedCondDS(train_ds),
        val_ds=_PairedCondDS(val_ds),
        latent_shape=latent_shape,
        reference_policy=reference_policy,  # the (base, controlnet) KL anchor (ADR-0015)
        controlnet=controlnet,  # the ONLY trainable arm
        # No FID triple: the ControlNet path selects on val/mean_reward (the unconditional
        # FID would be a constant frozen-base metric — see run_grpo_training's controlnet skip).
    )
