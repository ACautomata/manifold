"""``manifold-train-reward`` console entry + the testable reward-training core.

The console entry composes the experiment config, builds the :class:`RewardModel`
+ the pair datamodule (real precomputed pairs, or a fake pair cache via the
``data_provider`` injection seam for the CPU smoke), and calls ``Trainer.fit``.
The reward job is standalone (issue #39 user story 17): decoupled from JiT
pretraining and from GRPO, independently resumable.

The integration core :func:`run_reward_training` (Module + pair datamodule +
``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a tiny CPU
smoke can drive it with a fake pair dataset (the issue's testing seam) instead of
a real precomputed pair cache + a frozen denoiser.
"""

from __future__ import annotations

import argparse
import logging
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
from ..models.reward_model import RewardModel
from ..modules.reward import RewardModule
from .trainer import build_trainer

_log = logging.getLogger(__name__)


@dataclass
class _RewardDataBundle:
    """The pair-data bundle ``main`` passes into :func:`run_reward_training`.

    (Injection seam for the CPU smoke test, which feeds a fake pair dataset
    instead of a real precomputed pair cache + frozen denoiser.)
    """

    pair_ds: Any
    val_pair_ds: Any
    val_probe: Any = None  #: optional fixed generated-end probe RewardPairDataset.


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/pair_acc",
    mode: str = "max",
    save_top_k: int = 1,
    multi_gpu: bool = False,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` monitoring the reward val metric.

    Pairwise accuracy is maximized (``mode="max"``); ``auto_insert_metric_name =
    False`` because the metric key contains a ``/``. ``save_last=True`` for resume.
    Under DDP (``multi_gpu``) the metric is rank-local to rank 0, so monitoring is
    dropped (``save_last`` + ``save_top_k=1`` keep the latest) — mirroring the JiT
    checkpoint's DDP fallback (val/pair_acc is synced for reporting, but the
    selection signal stays single-GPU).
    """
    if multi_gpu:
        return ModelCheckpoint(
            dirpath=model_dir,
            filename="reward-{epoch:03d}",
            save_last=True,
            save_top_k=1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
    return ModelCheckpoint(
        dirpath=model_dir,
        filename=f"reward-{{epoch:03d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


def run_reward_training(
    *,
    module: RewardModule,
    bundle: _RewardDataBundle,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    save_top_k: int = 1,
    monitor_metric: str = "val/pair_acc",
    mode: str = "max",
    limit_val_batches: int | float = 1.0,
    seed: int = 0,
    ckpt_path: str | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the reward module (the core seam).

    Builds a stock ``ModelCheckpoint`` (monitoring the reward val metric) and runs
    ``Trainer.fit`` on the pair datamodule. Returns ``(trainer, ckpt)`` so callers
    can find the written ``.ckpt``.

    Args:
        bundle: the train + val pair datasets (``{"winner","loser"}``-emitting).
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    # Seed deterministically so direct callers (tests, notebooks) get reproducible
    # runs; ``main`` also seeds, harmlessly, before building the module.
    pl.seed_everything(seed, workers=True)
    multi_gpu = isinstance(devices, int) and devices > 1
    ckpt = _build_checkpoint(
        model_dir, monitor_metric=monitor_metric, mode=mode, save_top_k=save_top_k, multi_gpu=multi_gpu
    )
    # Score the fixed generated-end probe in training-batch-size chunks (bounds
    # epoch-end memory); attach the probe if the bundle carries one.
    module.probe_batch_size = int(batch_size)
    if bundle.val_probe is not None and getattr(module, "val_probe", None) is None:
        module.set_val_probe(bundle.val_probe.winners, bundle.val_probe.losers)
    datamodule = build_datamodule(
        bundle.pair_ds,
        batch_size=batch_size,
        val_dataset=bundle.val_pair_ds,
        num_workers=num_workers,
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
        prog="manifold-train-reward", description="Train the GRPO reward model."
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c", "--train", default="configs/train/config_reward.yaml", help="reward recipe YAML."
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (reward_model block)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--pairs-dir", default=None, help="precomputed RewardPairDataset directory."
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config → build → ``run_reward_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> _RewardDataBundle`` returning a fake pair dataset so the
    full ``main`` path runs without a real precomputed pair cache. The real path
    loads precomputed pairs from ``--pairs-dir`` (written by the offline
    generation script, issue #42).
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # The reward job needs no VAE / UNet-checkpoint path (pairs are precomputed);
    # only the output ``model_dir`` is required.
    require_paths(cfg, keys=("model_dir",))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "reward_train", None) is None:
        raise ValueError(
            "Config has no `reward_train` block — use the reward recipe "
            "(-c configs/train/config_reward.yaml), not a JiT train config."
        )

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        bundle = data_provider(cfg, device)
    else:
        bundle = _warm_pairs(cfg, args.pairs_dir)

    reward_cfg = cfg.reward_model
    module = RewardModule(
        RewardModel(
            spatial_dims=int(opt(reward_cfg, "spatial_dims", 3)),
            in_channels=int(opt(reward_cfg, "in_channels", 4)),
            channels=int(opt(reward_cfg, "channels", 64)),
            num_layers_d=int(opt(reward_cfg, "num_layers_d", 3)),
            norm=str(opt(reward_cfg, "norm", "BATCH")),
        ),
        lr=float(cfg.reward_train.lr),
    )

    run_reward_training(
        module=module,
        bundle=bundle,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or cfg.reward_train.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else "auto",
        batch_size=int(cfg.reward_train.batch_size),
        save_top_k=int(opt(cfg, "checkpoint.save_top_k", 1)),
        seed=seed,
        ckpt_path=args.resume,
    )
    print(f"[manifold-train-reward] done; checkpoints under {cfg.model_dir}")
    return 0


def _warm_pairs(cfg, pairs_dir: str | None) -> _RewardDataBundle:
    """Load the real precomputed pair cache (the production data path, issue #42).

    ``RewardPairDataset`` is imported lazily so this module does not require the
    pair-generation stack at import time (and the CPU smoke, which injects its own
    fake bundle, never reaches here).
    """
    if pairs_dir is None:
        pairs_dir = str(opt(cfg, "reward.pairs_dir"))
    if not pairs_dir:
        raise ValueError(
            "No reward pairs directory: pass --pairs-dir <dir> (or set reward.pairs_dir), "
            "the dir scripts/generate_reward_pairs.py wrote."
        )
    from ..data.reward_pairs import load_reward_pairs

    pair_ds, val_pair_ds, val_probe = load_reward_pairs(pairs_dir)
    return _RewardDataBundle(pair_ds=pair_ds, val_pair_ds=val_pair_ds, val_probe=val_probe)
