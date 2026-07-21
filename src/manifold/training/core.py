"""The composed training spine — the single caller of the callback registry (ADR-0032).

:class:`TrainingSpine` owns the :class:`CallbackRegistry` and the
assemble→resolve→build→validate→trainer→fit sequence that was duplicated across
the five training CLIs. Each ``run_*`` shell seeds, builds its own module +
datamodule, derives its dynamic default callback-name set, and delegates to
``TrainingSpine.run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

from manifold.training.callbacks import CallbackContext, CallbackRegistry
from manifold.training.trainer import build_trainer


@dataclass
class TrainingSpine:
    """The composed training spine: registry → resolve → build → validate → fit.

    A composed object (not a base class — the OOP rule). Each training CLI holds
    one instance and delegates to :meth:`run` after seeding + building its own
    module + datamodule.

    Attributes:
        registry: The callback name → spec registry. Populated by the shell
            before calling :meth:`run`.
    """

    registry: CallbackRegistry = field(default_factory=CallbackRegistry)

    def run(
        self,
        *,
        module: Any,
        datamodule: Any,
        ctx: CallbackContext,
        default_names: list[str],
        max_epochs: int,
        model_dir: str,
        devices: int | str = "auto",
        accelerator: str = "auto",
        limit_val_batches: int | float | None = None,
        extra_trainer_kwargs: dict | None = None,
        ckpt_path: str | None = None,
        callback_cfg: dict[str, dict] | None = None,
        callback_names_override: list[str] | None = None,
        extra_callbacks: list | None = None,
    ) -> tuple[pl.Trainer, ModelCheckpoint]:
        """Run the full training spine.

        The merge order (ADR-0029): *default_names* → *callback_cfg* knobs →
        *callback_names_override* **replaces** the name list.

        After resolving and building the registry callbacks, any
        *extra_callbacks* (e.g. the hand-appended ``LatentX0MAE``) are
        added before :meth:`CallbackRegistry.validate_monitor` checks the
        checkpoint's ``monitor_metric`` against the full set.

        Args:
            module: The training module.
            datamodule: The data module.
            ctx: The runtime-objects bag for ``CallbackSpec.build``.
            default_names: The per-CLI dynamic default callback name list.
            max_epochs: Training epoch count.
            model_dir: Checkpoint / log output directory.
            devices: Lightning device spec.
            accelerator: Lightning accelerator.
            limit_val_batches: Cap on validation batches.
            extra_trainer_kwargs: Additional ``Trainer`` kwargs.
            ckpt_path: Optional resume checkpoint.
            callback_cfg: Optional ``{name: {knob: value}}`` override map
                (the YAML ``callbacks:`` block).
            callback_names_override: Optional full replacement for
                *default_names* (the CLI ``--callbacks`` flag).
            extra_callbacks: Non-registry callbacks to append after
                registry callbacks (e.g. ``LatentX0MAE``).

        Returns:
            ``(trainer, ckpt)`` so callers can find the written ``.ckpt``.
        """
        names = list(
            callback_names_override
            if callback_names_override is not None
            else default_names
        )
        specs = self.registry.resolve(names, callback_cfg)
        callbacks: list = self.registry.build(specs, ctx)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        self.registry.validate_monitor(specs, module)
        ckpt = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
        trainer = build_trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            model_dir=model_dir,
            devices=devices,
            accelerator=accelerator,
            limit_val_batches=limit_val_batches,
            extra_kwargs=extra_trainer_kwargs,
        )
        trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
        return trainer, ckpt
