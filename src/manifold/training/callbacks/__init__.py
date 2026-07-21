"""The callback registry (ADR-0029): typed specs + two-phase resolve/build.

- :class:`CallbackContext` — the runtime-objects bag ``build`` injects;
- :class:`CallbackSpec` — the ``@dataclass`` spec contract (knobs + ``build``);
- :class:`CallbackRegistry` — name → spec, ``resolve`` (fail-fast) / ``build`` /
  ``validate_monitor`` (post-resolve monitor check);
- :class:`TrainLossSpec` — the ``train/loss_epoch`` logger spec (issue #159);
- :class:`FIDSpec` — the ``val/fid`` generative-FID spec (issue #160);
- :class:`CheckpointSpec` — the ``ModelCheckpoint`` spec (issue #160).
"""

from manifold.training.callbacks.checkpoint import CheckpointSpec
from manifold.training.callbacks.context import CallbackContext
from manifold.training.callbacks.fid import FIDSpec
from manifold.training.callbacks.registry import CallbackRegistry, CallbackSpec
from manifold.training.callbacks.train_loss import TrainLossSpec

__all__ = [
    "CallbackContext",
    "CallbackRegistry",
    "CallbackSpec",
    "CheckpointSpec",
    "FIDSpec",
    "TrainLossSpec",
]
