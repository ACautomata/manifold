"""The trainer stack (issues #26 / #28): trainer + EMA + metrics + CLI + export.

- :func:`build_trainer` — the invariant Lightning ``Trainer`` wiring (AMP, DDP
  ``find_unused_parameters``, the spt registry callback, CSV+TB loggers);
- :class:`DoubleEMACallback` — JiT's ``0.9999`` / ``0.9996`` EMA shadows,
  ``swap_in``/``restore`` for generation on the slow shadow, persisted for resume;
- :class:`TrainLossLogger` / :class:`LatentX0MAE` — ``train/loss_epoch`` and the
  cheap latent-space ``val/x0_mae``;
- :func:`run_training` / :func:`cli.main` — the ``manifold-train`` entry + the
  testable orchestration core (stock ``ModelCheckpoint``);
- :func:`export.export_to_native` — the ADR-0006 ``.ckpt → native`` bridge.
"""

from .cli import run_training
from .ema import DoubleEMACallback
from .export import export_to_native
from .grpo_cli import run_grpo_training
from .metrics import LatentX0MAE, TrainLossLogger
from .paired_reward_cli import run_paired_reward_training
from .reward_cli import run_reward_training
from .trainer import build_trainer

__all__ = [
    "DoubleEMACallback",
    "LatentX0MAE",
    "TrainLossLogger",
    "build_trainer",
    "export_to_native",
    "run_grpo_training",
    "run_paired_reward_training",
    "run_reward_training",
    "run_training",
]

