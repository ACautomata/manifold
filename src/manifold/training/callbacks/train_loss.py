"""Spec for :class:`TrainLossLogger` — the no-knob ``train/loss_epoch`` logger.

The first callback migrated behind the :class:`CallbackRegistry` (ADR-0029
tracer-bullet, issue #159). ``TrainLossLogger`` carries no config, so
:class:`TrainLossSpec` is a field-less dataclass; it exists so the registry can
resolve it by name (``"train_loss"``) and so the fail-fast knob validation has a
registered spec to exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightning.pytorch as pl

from manifold.training.callbacks.context import CallbackContext
from manifold.training.metrics import TrainLossLogger


@dataclass(frozen=True)
class TrainLossSpec:
    """Registry spec for :class:`TrainLossLogger` (no knobs).

    Matches the :class:`CallbackSpec` Protocol structurally (a ``build`` method
    over :class:`CallbackContext`) without inheriting it.
    """

    def build(self, ctx: CallbackContext) -> pl.Callback:
        return TrainLossLogger()
