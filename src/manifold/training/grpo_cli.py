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
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import opt
from ..data.datamodule import build_datamodule
from ..metrics import FIDCallback
from ..modules.grpo import GRPOModule
from ..schedulers.scheduling_flow_match_grpo import FlowMatchGRPOScheduler
from .trainer import build_trainer

_log = logging.getLogger(__name__)


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
    vae: Any = None
    real_latents: Any = None
    feature_net: Any = None
    feature_net_factory: Any = None  # L3: lazy build on the rank-0-gated FID stage
    fid_modality: int = 1
    fid_spacing: Sequence[float] = (1.0, 1.0, 1.0)


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/mean_reward",
    mode: str = "max",
    save_top_k: int = 1,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` monitoring the GRPO progress signal.

    #58 selects on ``val/fid`` (mode ``min``) - the anti-reward-hacking screen (a
    reward-hacked checkpoint scores high reward but high FID, so it is not selected)
    - when the FID callback is attached; #56 monitored ``val/mean_reward`` (mode
    ``max``) for the reward-only tracer. ``auto_insert_metric_name = False`` because
    the metric key contains a ``/``. ``save_last=True`` for resume. ``val/fid`` /
    ``val/mean_reward`` are GLOBAL under DDP now (FID sufficient-stats + mean_reward
    ``sync_dist``; ADR-0025), so the monitor stays on under multi-GPU.
    """
    return ModelCheckpoint(
        dirpath=model_dir,
        filename=f"grpo-{{epoch:03d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


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
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the GRPO module (the core seam).

    Builds a stock ``ModelCheckpoint`` (monitoring ``val/fid`` when the FID callback
    is attached, else ``val/mean_reward``) — and, when ``inputs`` carries the FID
    triple (``vae`` / ``real_latents`` / ``feature_net``), attaches
    :class:`~manifold.metrics.FIDCallback` with **no EMA** (GRPO evaluates the raw
    policy, #59) — then runs ``Trainer.fit`` on the conditioning datamodule + the val
    set. Returns ``(trainer, ckpt)`` so callers can find the written ``.ckpt``.

    Args:
        inputs: the train/val conditioning datasets + the policy/reward/scheduler
            (+ optional FID triple).
        monitor_metric / mode: ``None`` (the default) auto-selects ``val/fid`` (min)
            when the FID triple is present, else ``val/mean_reward`` (max). Pass
            explicitly to override.
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    pl.seed_everything(seed, workers=True)
    fid_active = (
        inputs.vae is not None
        and inputs.real_latents is not None
        and (inputs.feature_net is not None or inputs.feature_net_factory is not None)
    )
    if monitor_metric is None:
        monitor_metric = "val/fid" if fid_active else "val/mean_reward"
    if mode is None:
        # Derive from the FINAL metric (not fid_active): a caller who overrides
        # monitor_metric back to val/mean_reward must get mode=max, not the FID min.
        mode = "min" if monitor_metric == "val/fid" else "max"
    ckpt = _build_checkpoint(
        model_dir, monitor_metric=monitor_metric, mode=mode, save_top_k=save_top_k
    )
    callbacks: list[pl.Callback] = [ckpt]
    if fid_active:
        callbacks.append(
            FIDCallback(
                module=module,
                vae=inputs.vae,
                real_latents=inputs.real_latents,
                feature_net=inputs.feature_net,
                feature_net_factory=inputs.feature_net_factory,
                latent_shape=(1, *module.latent_shape),
                spacing=inputs.fid_spacing,
                modality=int(inputs.fid_modality),
                num_inference_steps=module.num_steps,
                num_synth=num_synth,
                center_slices_ratio=center_slices_ratio,
                cov_ridge=cov_ridge,
                seed=seed,
            )
        )
    datamodule = build_datamodule(
        inputs.train_ds, batch_size=batch_size, val_dataset=inputs.val_ds, num_workers=num_workers
    )
    trainer = build_trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches,
        extra_kwargs=(
            {"limit_train_batches": limit_train_batches}
            if limit_train_batches is not None
            else None
        ),
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    return trainer, ckpt


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

    from ..config import load_config, merge_overrides, require_paths

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        inputs = data_provider(cfg, device)
    else:
        # --native-dir / --reward-path / --latents-dir are NOT argparse-required: that
        # would break the data_provider injection seam (the CPU smoke). Validate them
        # here, only on the real path.
        if not args.native_dir or not args.reward_path or not args.latents_dir:
            raise ValueError(
                "GRPO needs --native-dir <native JiT export (raw arm)>, "
                "--reward-path <trained RewardModel .ckpt>, and "
                "--latents-dir <latent cache> (or inject a data_provider for the smoke)."
            )
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

    run_grpo_training(
        module=module,
        inputs=inputs,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or gcfg.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(gcfg.batch_size),
        save_top_k=int(opt(cfg, "checkpoint.save_top_k", 1)),
        seed=seed,
        ckpt_path=args.resume,
        limit_train_batches=args.limit_train_batches,
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


def _real_inputs(
    cfg, native_dir: str, reward_path: str, latents_dir: str, device: torch.device
) -> GRPOInputs:
    """Build the real GRPO inputs from the raw JiT arm + the trained reward (#59).

    The raw-arm JiT UNet (ADR-0006: ``export_to_native`` bakes the raw ``state_dict``
    weights) is the **trainable**
    policy. The trained :class:`RewardModel` (a ``RewardModule`` Lightning checkpoint —
    its ``reward_model.*`` state_dict) is the **frozen** reward. The latent cache
    furnishes the manifold conditioning (``{spacing, label}`` — GRPO is generative, so
    there is no clean-latent target) over train subjects AND the fixed real-reference
    latents (val subjects, scaled) the FID callback decodes. The RadImageNet
    ``feature_net`` completes the FID triple (#58).

    Launch is gated on the ns15 reward clearing ``val/gen_pair_acc > 0.8`` + a
    ``--measure`` run on the target cluster (#59) — not exercisable here (no real
    artifacts on the dev machine), so this mirrors ``reward_cli._real_inputs``'s
    pattern (the data_provider seam covers the CPU smoke).
    """
    from torch.utils.data import Dataset

    from ..data.reward_pairs import load_cached_latents, partition_subjects
    from ..metrics import make_feature_network
    from ..models.reward_model import RewardModel
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
    reward_model = RewardModel(
        spatial_dims=int(opt(reward_cfg, "spatial_dims", 3)),
        in_channels=int(opt(reward_cfg, "in_channels", 4)),
        channels=int(opt(reward_cfg, "channels", 64)),
        num_layers_d=int(opt(reward_cfg, "num_layers_d", 3)),
        norm=str(opt(reward_cfg, "norm", "BATCH")),
    )
    # weights_only=True first (no arbitrary-code-execution risk); fall back to False
    # only for the weights_only restriction itself (non-allowlisted globals in a full
    # Lightning ckpt's callback/optimizer state), NOT for file/IO errors (which must
    # surface). reward_path is the user's OWN trained reward (trusted — mirroring
    # export_to_native; never point this at an untrusted .ckpt).
    import pickle

    try:
        ckpt = torch.load(str(reward_path), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, ValueError):
        _log.warning("reward ckpt %s needs weights_only=False (non-tensor state); loading trusted.", reward_path)
        ckpt = torch.load(str(reward_path), map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    reward_sd = {k[len("reward_model."):]: v for k, v in state.items() if k.startswith("reward_model.")}
    if not reward_sd:
        raise ValueError(
            f"No 'reward_model.*' keys in {reward_path} — not a trained RewardModule checkpoint."
        )
    reward_model.load_state_dict(reward_sd, strict=True)
    reward_model.eval().to(device)
    for p in reward_model.parameters():
        p.requires_grad_(False)

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
    _log.info(
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
    # whole ``real_latents`` tensor in ONE _real_moments() pass, so an unbounded val
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
