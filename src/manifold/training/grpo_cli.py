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
from ..models.reward_model import RewardModel
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
    """

    policy: Any
    reward_model: Any
    scheduler: Any
    train_ds: Any
    val_ds: Any
    latent_shape: Sequence[int]


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/mean_reward",
    mode: str = "max",
    save_top_k: int = 1,
    multi_gpu: bool = False,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` monitoring the GRPO progress signal.

    #56 monitors ``val/mean_reward`` (mode ``max``) — the only metric this slice
    logs. #58 switches selection to ``val/fid`` (mode ``min``, the anti-reward-hacking
    screen). ``auto_insert_metric_name = False`` because the metric key contains a
    ``/``. ``save_last=True`` for resume. Under DDP the metric is rank-local, so
    monitoring is dropped (``save_last`` + ``save_top_k=1`` keep the latest) —
    mirroring the JiT / reward checkpoint DDP fallback.
    """
    if multi_gpu:
        return ModelCheckpoint(
            dirpath=model_dir,
            filename="grpo-{epoch:03d}",
            save_last=True,
            save_top_k=1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
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
    monitor_metric: str = "val/mean_reward",
    mode: str = "max",
    limit_val_batches: int | float = 1.0,
    seed: int = 0,
    ckpt_path: str | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the GRPO module (the core seam).

    Builds a stock ``ModelCheckpoint`` (monitoring ``val/mean_reward``) and runs
    ``Trainer.fit`` on the conditioning train datamodule + the val set. Returns
    ``(trainer, ckpt)`` so callers can find the written ``.ckpt``.

    Args:
        inputs: the train/val conditioning datasets + the policy/reward/scheduler.
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    pl.seed_everything(seed, workers=True)
    multi_gpu = isinstance(devices, int) and devices > 1
    ckpt = _build_checkpoint(
        model_dir, monitor_metric=monitor_metric, mode=mode, save_top_k=save_top_k, multi_gpu=multi_gpu
    )
    datamodule = build_datamodule(
        inputs.train_ds, batch_size=batch_size, val_dataset=inputs.val_ds, num_workers=num_workers
    )
    trainer = build_trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt],
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
        help="trained RewardModel checkpoint dir; required without --data-provider.",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
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
        # --native-dir / --reward-path are NOT argparse-required: that would break
        # the data_provider injection seam (the CPU smoke). Validate them here, only
        # on the real path.
        if not args.native_dir or not args.reward_path:
            raise ValueError(
                "GRPO needs --native-dir <native JiT export (raw arm)> and "
                "--reward-path <trained RewardModel> (or inject a data_provider for the smoke)."
            )
        inputs = _real_inputs(cfg, args.native_dir, args.reward_path, device)

    gcfg = cfg.grpo_train
    latent_shape = tuple(int(s) for s in opt(gcfg, "latent_shape", [4, 64, 64, 32]))
    module = GRPOModule(
        inputs.policy,
        inputs.reward_model,
        inputs.scheduler if inputs.scheduler is not None else FlowMatchGRPOScheduler(eta=float(opt(gcfg, "eta", 0.7))),
        G=int(opt(gcfg, "G", 8)),
        eta_step_list=list(opt(gcfg, "eta_step_list", [0, 1, 2, 3, 4, 5, 6, 7])),
        eta=float(opt(gcfg, "eta", 0.7)),
        clip_range=float(opt(gcfg, "clip_range", 1e-4)),
        lr=float(gcfg.lr),
        adv_clip_max=float(opt(gcfg, "adv_clip_max", 5.0)),
        num_steps=int(opt(gcfg, "num_steps", 15)),
        latent_shape=latent_shape,
    )

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
    )
    print(f"[manifold-train-grpo] done; checkpoints under {cfg.model_dir}")
    return 0


def _real_inputs(cfg, native_dir: str, reward_path: str, device: torch.device) -> GRPOInputs:
    """Build the real GRPO inputs from the raw JiT arm + the trained reward (#59).

    Loads the raw-arm JiT UNet (the policy — ADR-0006), the trained
    :class:`RewardModel`, and wires the latent-cache conditioning. Gated on the ns15
    reward model clearing ``val/gen_pair_acc > 0.8`` + a tiny-config measurement
    (#59 launch readiness); not built in the #56 tracer.
    """
    raise NotImplementedError(
        "GRPO real-data launch is wired in #59 (launch readiness: raw JiT arm + trained "
        "reward + no-EMA + measurement gate). Inject a data_provider for the #56 CPU smoke."
    )
