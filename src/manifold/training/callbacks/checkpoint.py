"""Spec for :class:`lightning.pytorch.callbacks.ModelCheckpoint` (ADR-0029).

The stock Lightning ``ModelCheckpoint`` migrated behind the
:class:`CallbackRegistry` (issue #160). The spec declares the **full checkpoint
config surface** as knobs so strict unknown-knob validation does not reject live
configs; ``build`` reproduces the prior ``_build_checkpoint`` two-branch
construction (monitored vs. unmonitored periodic / last).

Its one special behavior lives outside ``build``: a **post-resolve monitor
validation** run by :meth:`CallbackRegistry.validate_monitor`, which checks the
``monitor_metric`` against the resolved callbacks' logged-metrics union the
module's declared metrics (an explicit ``monitor_metric=None`` bypasses it).
"""

from __future__ import annotations

from dataclasses import dataclass

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from manifold.training.callbacks.context import CallbackContext


@dataclass(frozen=True)
class CheckpointSpec:
    """Registry spec for :class:`ModelCheckpoint`.

    The unmonitored path (``monitor_metric=None``) keeps the latest periodic
    checkpoint (``save_top_k=1``) plus ``last.ckpt`` at the ``every_n_epochs``
    cadence - the JiT production fallback when no held-out validation is
    configured. The monitored path tracks ``monitor_metric`` top-``k`` plus last.
    """

    monitor_metric: str | None = None
    save_top_k: int = 3
    save_last: bool = True
    every_n_epochs: int = 1
    mode: str = "min"
    filename: str | None = None

    def build(self, ctx: CallbackContext) -> pl.Callback:
        common = dict(
            dirpath=ctx.model_dir,
            save_last=self.save_last,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
        if self.monitor_metric is None:
            return ModelCheckpoint(
                **common,
                filename="unet3d-{epoch:03d}-{step}",
                save_top_k=1,
                every_n_epochs=max(1, self.every_n_epochs),
            )
        return ModelCheckpoint(
            **common,
            filename=self.filename
            or f"unet3d-{{epoch:03d}}-{{step}}-{{{self.monitor_metric}:.3f}}",
            monitor=self.monitor_metric,
            mode=self.mode,
            save_top_k=self.save_top_k,
        )
