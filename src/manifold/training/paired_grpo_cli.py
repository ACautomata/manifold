"""``manifold-train-paired-grpo`` console entry + the testable G2RPO-training core.

The console entry (issue #103) composes the experiment config, builds the
:class:`PairedGRPOModule` (trainable Paired JiT policy UNet + frozen paired reward)
+ the data inputs (real paired policy + reward, or a fake via the ``data_provider``
injection seam for the CPU smoke), and calls ``Trainer.fit``. The G2RPO job is
standalone: decoupled from Paired JiT pretraining and from paired-reward training,
independently resumable.

The integration core :func:`run_paired_grpo_training` (Module + datamodule +
``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a tiny CPU smoke
can drive it with a fake policy + toy source latents (the issue's testing seam)
instead of the real paired checkpoint + trained reward. The real-data launch path
(loading the slow-EMA paired UNet + the trained paired reward, wiring the paired
latent cache, the η-ramp, the raw-arm export) is gated on the bridge-noise
reward-ranking probe + a tiny-config measurement (#106); the ``data_provider`` seam
ships in #103, ``_real_inputs`` in #104.
"""

from __future__ import annotations

import argparse
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
from ..modules.paired_grpo import PairedGRPOModule
from ..schedulers.scheduling_flow_match_bridge_grpo import FlowMatchBridgeGRPOScheduler
from .trainer import build_trainer, is_multi_gpu

_log = logging.getLogger(__name__)


@dataclass
class PairedGRPOInputs:
    """Module-construction + data inputs for one G2RPO run.

    ``policy`` / ``reward_model`` / ``scheduler`` go to the :class:`PairedGRPOModule`
    ctor; ``train_ds`` / ``val_ds`` emit the source latent + the contrast direction
    (``{src_latent, src_label, tgt_label, spacing}`` — pure-RL, the target volume is
    unused at train). The ``data_provider`` seam injects a fake policy + toy source
    latents for the CPU smoke; the real path (``_real_inputs``, #104) loads the
    slow-EMA paired UNet + the trained paired reward.
    """

    policy: Any
    reward_model: Any
    scheduler: Any
    train_ds: Any
    val_ds: Any
    reference_policy: Any = None  # the frozen KL anchor (ADR-0015); None ⇒ no KL (v1)


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/mean_reward",
    mode: str = "max",
    save_top_k: int = 1,
    multi_gpu: bool = False,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` monitoring the G2RPO progress signal.

    #105 selects on ``val/psnr`` (mode ``max``) — the reproducible deterministic-Heun
    goal metric — when the PSNR callback is attached; #103 (the tracer) monitors
    ``val/mean_reward`` (mode ``max``) for the reward-only smoke. ``auto_insert_metric_name
    = False`` because the metric key contains a ``/``. ``save_last=True`` for resume.
    Under DDP the rank-local ``val/mean_reward`` is dropped (the PSNR callback's
    ``val/psnr`` is the global cross-rank selection metric, kept) — mirroring the
    noise→data GRPO checkpoint DDP fallback.
    """
    if multi_gpu and monitor_metric == "val/mean_reward":
        # val/mean_reward is rank-0-only (validation_step gate); drop the monitor
        # under DDP (save_last + save_top_k=1 keep the latest). val/psnr is a global
        # cross-rank mean (the PSNR callback all_gathers) → monitor stays on (#105).
        return ModelCheckpoint(
            dirpath=model_dir,
            filename="paired-grpo-{epoch:03d}",
            save_last=True,
            save_top_k=1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
    return ModelCheckpoint(
        dirpath=model_dir,
        filename=f"paired-grpo-{{epoch:03d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


def run_paired_grpo_training(
    *,
    module: PairedGRPOModule,
    inputs: PairedGRPOInputs,
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
    extra_callbacks: Sequence[pl.Callback] | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the G2RPO module (the core seam).

    Builds a stock ``ModelCheckpoint`` (monitoring ``val/psnr`` when the PSNR callback
    is attached in #105, else ``val/mean_reward``) + the caller's extra callbacks
    (the PSNR callback, #105), then runs ``Trainer.fit`` on the source-latent
    datamodule + the val set. Returns ``(trainer, ckpt)`` so callers can find the
    written ``.ckpt``.

    Args:
        inputs: the train/val source-latent datasets + the policy/reward/scheduler
            (+ the optional KL reference).
        monitor_metric / mode: ``None`` (the default) auto-selects ``val/psnr`` (max)
            when extra_callbacks carries a PSNR callback, else ``val/mean_reward``
            (max). Pass explicitly to override.
        extra_callbacks: the PSNR callback (#105) and any other non-checkpoint
            callbacks; the checkpoint is always appended here.
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    pl.seed_everything(seed, workers=True)
    multi_gpu = is_multi_gpu(devices)
    psnr_active = extra_callbacks is not None and any(
        type(c).__name__ == "PairedPSNRSSIMCallback" for c in extra_callbacks
    )
    if monitor_metric is None:
        monitor_metric = "val/psnr" if psnr_active else "val/mean_reward"
    if mode is None:
        mode = "max"  # both val/psnr and val/mean_reward are mode=max
    ckpt = _build_checkpoint(
        model_dir, monitor_metric=monitor_metric, mode=mode, save_top_k=save_top_k, multi_gpu=multi_gpu
    )
    callbacks: list[pl.Callback] = list(extra_callbacks) if extra_callbacks else []
    callbacks.append(ckpt)
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
        prog="manifold-train-paired-grpo",
        description="Post-train the Paired JiT UNet with Granular-GRPO (G2RPO) over the "
        "src→tgt Brownian bridge against the frozen paired reward (ADR-0024).",
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c", "--train", default="configs/train/config_paired_grpo.yaml",
        help="G2RPO recipe YAML.",
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (latent metadata)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--native-dir",
        default=None,
        help="native paired export dir (the slow-EMA policy + VAE scale); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--reward-path",
        default=None,
        help="trained paired RewardModel checkpoint (.ckpt); required without --data-provider.",
    )
    parser.add_argument(
        "--latents-dir",
        default=None,
        help="paired latent cache dir (source latents + contrast labels); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="run a tiny-config measurement (it/s + peak GPU memory) and exit — the "
        "#106 launch gate; size G / eta_step_list / n_epochs before the full run.",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="cap train batches/epoch (a debug knob for the fast re-measure; the full "
        "run leaves it unset).",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config → build → ``run_paired_grpo_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> PairedGRPOInputs`` returning a fake policy + toy source
    latents so the full ``main`` path runs without the real paired checkpoint +
    trained reward. The real path loads the slow-EMA paired UNet from ``--native-dir``
    and the paired reward from ``--reward-path`` (#104).
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # G2RPO needs no VAE / image data — only the output model_dir is required.
    require_paths(cfg, keys=("model_dir",))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "paired_grpo_train", None) is None:
        raise ValueError(
            "Config has no `paired_grpo_train` block — use the G2RPO recipe "
            "(-c configs/train/config_paired_grpo.yaml), not a Paired JiT train or "
            "reward config."
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
                "G2RPO needs --native-dir <native paired export (slow-EMA arm)>, "
                "--reward-path <trained paired RewardModel .ckpt>, and "
                "--latents-dir <paired latent cache> (or inject a data_provider for the smoke)."
            )
        inputs = _real_inputs(cfg, args.native_dir, args.reward_path, args.latents_dir, device)

    gcfg = cfg.paired_grpo_train
    module = PairedGRPOModule(
        inputs.policy,
        inputs.reward_model,
        inputs.scheduler if inputs.scheduler is not None
        else FlowMatchBridgeGRPOScheduler(eta=float(opt(gcfg, "eta", 0.7))),
        G=int(opt(gcfg, "G", 8)),
        eta_step_list=list(opt(gcfg, "eta_step_list", [0, 1, 2, 3])),
        clip_range=float(opt(gcfg, "clip_range", 1e-4)),
        lr=float(gcfg.lr),
        adv_clip_max=float(opt(gcfg, "adv_clip_max", 5.0)),
        num_steps=int(opt(gcfg, "num_steps", 8)),
        reference_policy=inputs.reference_policy,
        kl_coef=float(opt(gcfg, "kl_coef", 0.0)),
        reward_bound=str(opt(gcfg, "reward_bound", "none")),
        reward_temp=float(opt(gcfg, "reward_temp", 8.0)),
    )

    if args.measure:
        # The #106 launch-gate measurement: a 1-epoch fit timing + peak GPU memory.
        it_per_s, peak, elapsed = run_paired_grpo_measurement(
            module=module,
            inputs=inputs,
            model_dir=str(cfg.model_dir),
            devices=args.num_gpus if args.num_gpus > 1 else 1,
            batch_size=int(gcfg.batch_size),
            seed=seed,
        )
        print(
            f"[manifold-train-paired-grpo] measure: {it_per_s:.3f} it/s | "
            f"peak GPU {peak / 1e9:.2f} GB | {elapsed:.1f}s"
        )
        return 0

    run_paired_grpo_training(
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
    print(f"[manifold-train-paired-grpo] done; checkpoints under {cfg.model_dir}")
    return 0


def run_paired_grpo_measurement(
    *,
    module: PairedGRPOModule,
    inputs: PairedGRPOInputs,
    model_dir: str,
    devices: int | str = 1,
    accelerator: str = "auto",
    batch_size: int = 2,
    seed: int = 0,
) -> tuple[float, int, float]:
    """Time a 1-epoch G2RPO fit + report it/s + peak GPU memory (the #106 launch gate).

    Sizes ``G`` / ``eta_step_list`` / ``n_epochs`` by measuring the real budget's
    throughput + peak GPU memory on the target cluster before committing to the full
    run. Returns ``(it_per_s, peak_gpu_bytes, elapsed_s)``. Peak memory is 0 off-CUDA
    (the read is GPU-only); a tiny ``--measure`` run on the cluster is the real signal.
    """
    import time

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer, _ = run_paired_grpo_training(
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
) -> PairedGRPOInputs:
    """Build the real G2RPO inputs from the slow-EMA paired UNet + the trained reward (#104).

    The slow-EMA Paired JiT UNet (ADR-0021 — ``load_frozen_paired_generator``'s arm,
    inverted for G2RPO: it is the trainable policy init, not a frozen generator) is
    the **trainable** policy; a bit-identical frozen deep-copy is the KL reference
    (ADR-0015). The trained paired :class:`RewardModel` (``in_channels = 2·C_latent``)
    is the **frozen** reward. The paired latent cache furnishes the source latents +
    contrast labels. Launch is gated on the bridge-noise reward-ranking probe + a
    ``--measure`` run (#106) — not exercisable here (no real artifacts on the dev
    machine), so this mirrors ``paired_reward_cli._real_inputs``'s pattern (the
    data_provider seam covers the CPU smoke).
    """
    raise NotImplementedError(
        "G2RPO _real_inputs ships in #104 (real budget + committed recipe). The #103 "
        "tracer drives run_paired_grpo_training via the data_provider injection seam "
        "(fake policy + toy source latents)."
    )


__all__ = [
    "PairedGRPOInputs",
    "main",
    "run_paired_grpo_measurement",
    "run_paired_grpo_training",
]
