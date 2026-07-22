"""The composed training spine — the single caller of the callback registry (ADR-0032).

:class:`TrainingSpine` owns the :class:`CallbackRegistry` and the
assemble→resolve→build→validate→trainer→fit sequence that was duplicated across
the five training CLIs. Each ``run_*`` shell seeds, builds its own module +
datamodule, derives its dynamic default callback-name set, and delegates to
``TrainingSpine.run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ImportError:  # pragma: no cover
    from pytorch_lightning.utilities.rank_zero import rank_zero_info  # type: ignore

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
        forbidden_callbacks: Mapping[str, str] | None = None,
        forbidden_monitors: Mapping[str, str] | None = None,
    ) -> tuple[pl.Trainer, ModelCheckpoint]:
        """Run the full training spine.

        The merge order (ADR-0029): *default_names* → *callback_cfg* knobs →
        *callback_names_override* **replaces** the name list.

        After the merge, *forbidden_callbacks* force-removes named callbacks and
        *forbidden_monitors* rejects a checkpoint monitor_metric — both with a
        loud ``rank_zero_info`` — **before** resolution. This is the post-merge
        guard that lets a shell (e.g. GRPO Mode-2) forbid a callback no YAML /
        CLI override can re-enable, rather than only suppressing it at
        default-derivation (ADR-0032).

        After resolving and building the registry callbacks, any
        *extra_callbacks* (e.g. the hand-appended ``LatentX0MAE``) are
        added before :meth:`CallbackRegistry.validate_monitor` checks the
        checkpoint's ``monitor_metric`` against the registry specs, the
        module's declared metrics, and the extra callbacks' ``logged_metrics``.

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
            forbidden_callbacks: Optional ``{name: reason}`` map; each name is
                force-removed from the merged list (with a loud log) before
                resolution — a YAML / CLI override cannot re-enable it.
            forbidden_monitors: Optional ``{metric: reason}`` map; if the merged
                checkpoint ``monitor_metric`` matches, a loud log precedes a
                ``ValueError`` before resolution.

        Returns:
            ``(trainer, ckpt)`` so callers can find the written ``.ckpt``.
        """
        names = list(
            callback_names_override
            if callback_names_override is not None
            else default_names
        )
        if forbidden_callbacks:
            for name in [n for n in names if n in forbidden_callbacks]:
                rank_zero_info(
                    "TrainingSpine: dropping callback %r post-merge (%s); "
                    "a YAML/CLI override cannot re-enable it here.",
                    name, forbidden_callbacks[name],
                )
                names.remove(name)
        if forbidden_monitors and callback_cfg:
            monitor = (callback_cfg.get("checkpoint") or {}).get("monitor_metric")
            if monitor is not None and monitor in forbidden_monitors:
                rank_zero_info(
                    "TrainingSpine: checkpoint monitor %r is forbidden (%s).",
                    monitor, forbidden_monitors[monitor],
                )
                raise ValueError(
                    f"TrainingSpine.run: checkpoint monitor_metric {monitor!r} is "
                    f"forbidden here ({forbidden_monitors[monitor]})."
                )
        specs = self.registry.resolve(names, callback_cfg)
        callbacks: list = self.registry.build(specs, ctx)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        # validate_monitor scans the registry specs + module + extra_callbacks'
        # logged_metrics, so a checkpoint monitoring a metric an extra callback
        # emits (e.g. LatentX0MAE's val/x0_mae) validates without the shell
        # mutating module.logged_metrics.
        self.registry.validate_monitor(specs, module, extra_callbacks=extra_callbacks)
        ckpt = next((c for c in callbacks if isinstance(c, ModelCheckpoint)), None)
        if ckpt is None:
            # A callback_names_override that drops "checkpoint" leaves no
            # ModelCheckpoint to return. The JiT shell (and the run_* contract)
            # expect one, so fail fast with a clear message rather than a
            # StopIteration deep in next(...).
            raise ValueError(
                f"TrainingSpine.run: the resolved callback list has no "
                f"ModelCheckpoint (names={names!r}). A checkpoint callback is "
                f"required; include 'checkpoint' in default_names or the override."
            )
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
