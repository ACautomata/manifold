"""Training-metrics callbacks: epoch-mean train loss + latent-space x0-MAE.

Both read the ``state`` dict ``spt.Module``'s training/validation steps return:

- :class:`TrainLossLogger` accumulates the per-batch ``state["loss"]`` and logs
  the all-reduced epoch mean as ``train/loss_epoch``;
- :class:`LatentX0MAE` accumulates ``mean(|pred − target|)`` over the validation
  forward's ``pred`` / ``target`` and logs ``val/x0_mae``. It respects
  ``Trainer.limit_val_batches`` (the Trainer simply runs fewer batches - the
  callback only sees what ran), so the cheap reconstruction signal never
  dominates epoch time.

The generative FID (issue #27) is a separate callback; this is the cheap
latent-space signal alongside it.

DDP honesty (issue #82 / ADR-0016): both accumulate into a
:class:`torchmetrics.MeanMetric`, updating with ``weight=batch_size`` so the
cross-rank reduction is the **true sample-weighted global mean** ``(Σ loss·B) /
(Σ B)`` - NOT a mean-of-per-rank-means (which a naive ``sync_dist=True`` would
give, exact only under equal per-rank batch counts that a future non-padding
sampler would break).

Lightning requires a logged :class:`torchmetrics.Metric` to be an attribute of
the ``LightningModule`` (it scans ``named_modules()`` for ``Metric`` subclasses to
restore state). A callback-held Metric is not auto-registered, so the callback
attaches its ``MeanMetric`` to the module under a private name in
``on_fit_start`` (idempotent - reuses the same instance across epochs so the
state survives). The Metric follows the module's device (Lightning moves module
submodules, not callback-held ones).
"""

from __future__ import annotations

import torch
import torchmetrics as pl_metrics

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover - lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore[assignment]

_ATTR = "_manifold_train_loss_epoch_mean"
_X0_ATTR = "_manifold_val_x0_mae_mean"


def _attach(module, attr: str) -> pl_metrics.MeanMetric:
    """Attach (idempotently) a ``MeanMetric`` to *module* under *attr*.

    Lightning scans ``named_modules()`` for ``Metric`` subclasses when a Metric
    is logged, so the metric must live on the module. Reuses the same instance
    across epochs (state survives) - ``setattr`` only on the first attach.
    """
    m = getattr(module, attr, None)
    if m is None:
        m = pl_metrics.MeanMetric()
        setattr(module, attr, m)
    return m


def _batch_size(batch, outputs) -> float:
    """The per-batch sample count B for sample-weighting the epoch loss.

    The module loss is a scalar mean over the batch; ``MeanMetric`` needs B as the
    weight so the cross-rank epoch aggregate is ``sum(loss·B)/sum(B)`` (the true
    sample-weighted global mean), not a mean-of-per-batch-means. Reads B from the
    batch's leading tensor dim; falls back to 1.0 when no batch tensor is visible
    (a non-dict batch with no recognizable tensor).
    """
    for key in ("latent", "src_latent", "src", "image"):
        v = batch.get(key) if isinstance(batch, dict) else None
        if torch.is_tensor(v) and v.dim() > 0:
            return float(v.shape[0])
    # Fall back to outputs' batch dim if present (a non-scalar loss carries it).
    loss = outputs.get("loss") if isinstance(outputs, dict) else None
    if torch.is_tensor(loss) and loss.dim() > 0:
        return float(loss.shape[0])
    return 1.0


class TrainLossLogger(pl.Callback):
    """Log the epoch-mean training loss as ``train/loss_epoch`` (sample-weighted)."""

    def on_fit_start(self, trainer, module) -> None:
        self._mean = _attach(module, _ATTR)
        self._mean.to(module.device if hasattr(module, "device") else "cpu")

    def on_train_epoch_start(self, trainer, module) -> None:
        self._mean.reset()
        self._mean.to(module.device if hasattr(module, "device") else "cpu")

    def on_train_batch_end(self, trainer, module, outputs, batch, *args, **kwargs) -> None:
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
            if torch.is_tensor(loss):
                # ``weight=batch_size`` makes the epoch aggregate the true
                # sample-weighted global mean across ranks (MeanMetric reduces
                # ``sum(loss·B) / sum(B)``). The module loss is a scalar
                # batch-MEAN (F.mse_loss over the batch), so the weight must be
                # the per-batch SAMPLE COUNT B (not ``loss.shape[0]``, which is
                # the scalar's shape [] -> 1). Derive B from the batch tensor.
                self._mean.update(loss.detach().float(), weight=_batch_size(batch, outputs))

    def on_train_epoch_end(self, trainer, module) -> None:
        # Logging the Metric object (not a float) so Lightning's DDP ``sync``
        # fires the cross-rank reduction to the true weighted mean.
        module.log("train/loss_epoch", self._mean)


class LatentX0MAE(pl.Callback):
    """Log the latent-space x0-MAE (``val/x0_mae``) over the validation forward.

    Computed from the Module validation forward's ``pred`` / ``target`` (the
    clean-latent prediction vs the clean latent x0) - a cheap reconstruction-
    quality signal that, unlike the generative FID, needs no sampling. Capped by
    ``Trainer.limit_val_batches`` (the Trainer runs at most that many val
    batches; the callback only averages what ran).
    """

    def on_fit_start(self, trainer, module) -> None:
        self._mean = _attach(module, _X0_ATTR)
        self._mean.to(module.device if hasattr(module, "device") else "cpu")

    def on_validation_epoch_start(self, trainer, module) -> None:
        self._mean.reset()
        self._mean.to(module.device if hasattr(module, "device") else "cpu")

    def on_validation_batch_end(
        self, trainer, module, outputs, *args, dataloader_idx: int = 0, **kwargs
    ) -> None:
        if not isinstance(outputs, dict) or "pred" not in outputs or "target" not in outputs:
            return
        mae = (outputs["pred"] - outputs["target"]).abs().mean()
        B = float(outputs["pred"].shape[0])
        self._mean.update(mae.detach().float(), weight=B)

    def on_validation_epoch_end(self, trainer, module) -> None:
        module.log("val/x0_mae", self._mean)
