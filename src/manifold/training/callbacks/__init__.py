"""The callback registry (ADR-0029): typed specs + two-phase resolve/build.

- :class:`CallbackContext` — the runtime-objects bag ``build`` injects;
- :class:`CallbackSpec` — the ``@dataclass`` spec contract (knobs + ``build``);
- :class:`CallbackRegistry` — name → spec, ``resolve`` (fail-fast) / ``build``;
- :class:`TrainLossSpec` — the ``train/loss_epoch`` logger spec (issue #159).
"""

from manifold.training.callbacks.context import CallbackContext
from manifold.training.callbacks.registry import CallbackRegistry, CallbackSpec
from manifold.training.callbacks.train_loss import TrainLossSpec

__all__ = [
    "CallbackContext",
    "CallbackRegistry",
    "CallbackSpec",
    "TrainLossSpec",
]
