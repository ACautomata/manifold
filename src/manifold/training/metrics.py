"""Training-metrics callbacks: epoch-mean train loss + latent-space x0-MAE.

Both read the ``state`` dict ``spt.Module``'s training/validation steps return:

- :class:`TrainLossLogger` accumulates the per-batch ``state["loss"]`` and logs
  the all-reduced epoch mean as ``train/loss_epoch`` (``module.log`` reuses
  Lightning's DDP reduction);
- :class:`LatentX0MAE` accumulates ``mean(|pred − target|)`` over the validation
  forward's ``pred`` / ``target`` and logs ``val/x0_mae``. It respects
  ``Trainer.limit_val_batches`` (the Trainer simply runs fewer batches — the
  callback only sees what ran), so the cheap reconstruction signal never
  dominates epoch time.

The generative FID (issue #27) is a separate callback; this is the cheap
latent-space signal alongside it.
"""

from __future__ import annotations

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore


class TrainLossLogger(pl.Callback):
    """Log the epoch-mean training loss as ``train/loss_epoch``."""

    def __init__(self) -> None:
        super().__init__()
        self._sum: float = 0.0
        self._n: int = 0

    def on_train_epoch_start(self, trainer, module) -> None:
        self._sum = 0.0
        self._n = 0

    def on_train_batch_end(self, trainer, module, outputs, *args, **kwargs) -> None:
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
            if torch.is_tensor(loss):
                self._sum += float(loss.detach())
                self._n += 1

    def on_train_epoch_end(self, trainer, module) -> None:
        if self._n:
            module.log("train/loss_epoch", self._sum / self._n)


class LatentX0MAE(pl.Callback):
    """Log the latent-space x0-MAE (``val/x0_mae``) over the validation forward.

    Computed from the Module validation forward's ``pred`` / ``target`` (the
    clean-latent prediction vs the clean latent x0) — a cheap reconstruction-
    quality signal that, unlike the generative FID, needs no sampling. Capped by
    ``Trainer.limit_val_batches`` (the Trainer runs at most that many val
    batches; the callback only averages what ran).
    """

    def __init__(self) -> None:
        super().__init__()
        self._sum: float = 0.0
        self._n: int = 0

    def on_validation_epoch_start(self, trainer, module) -> None:
        self._sum = 0.0
        self._n = 0

    def on_validation_batch_end(
        self, trainer, module, outputs, *args, dataloader_idx: int = 0, **kwargs
    ) -> None:
        if not isinstance(outputs, dict) or "pred" not in outputs or "target" not in outputs:
            return
        mae = (outputs["pred"] - outputs["target"]).abs().mean()
        self._sum += float(mae.detach())
        self._n += 1

    def on_validation_epoch_end(self, trainer, module) -> None:
        if self._n:
            module.log("val/x0_mae", self._sum / self._n)
