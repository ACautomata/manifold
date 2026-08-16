"""Spec for :class:`~manifold.metrics.PairedFidelityCallback` — the in-training paired-fidelity monitor.

The observe-only 3D PSNR/SSIM monitor for the supervised ControlNet stage (ADR-0037,
issue #238), mounted on the :class:`CallbackRegistry` exactly like
:class:`~manifold.training.callbacks.FIDSpec` (ADR-0029). Its four knobs
(``subset_size`` / ``every_n_epochs`` / ``num_inference_steps`` / ``seed``) come from a
``paired_fidelity`` config block; the runtime objects (``module`` / ``vae`` /
``paired_data``) are injected from :class:`CallbackContext` at ``build``.

``logged_metrics`` declares ``val/psnr`` / ``val/ssim`` so
:meth:`CallbackRegistry.validate_monitor` accepts them as *validatable* monitors — a
future opt-in switch is possible — but the monitor is **observe-only**: the checkpoint
spec's ``monitor_metric`` stays ``val/x0_mae`` and the callback never enters the loss.

``build`` forwards ``ctx.datamodule`` as the paired-subset source; the callback resolves
the actual paired dataset lazily at the first gated epoch
(``getattr(datamodule, "val_latent_ds", datamodule)``, F5), so the cold path's
post-``setup()`` replacement of ``val_latent_ds`` is honored. The rollout step count is
recipe-primary (ADR-0037): the supervised ControlNet spine fills
``CallbackContext.inference_recipe`` from the existing ``controlnet.num_inference_steps``
knob (issue #239), and the spec's own ``num_inference_steps`` is an optional per-callback
override layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import lightning.pytorch as pl

from manifold.metrics import PairedFidelityCallback
from manifold.training.callbacks.context import CallbackContext


@dataclass(frozen=True)
class PairedFidelitySpec:
    """Registry spec for :class:`PairedFidelityCallback` (the paired-fidelity knobs).

    Matches the :class:`CallbackSpec` Protocol structurally. The fixed-subset / cadence
    / seed knobs are declared here as config fields; the runtime objects come from
    :class:`CallbackContext`. The rollout step count is **recipe-primary** (ADR-0037):
    ``num_inference_steps`` defaults to ``None`` ⇒ read from
    ``ctx.inference_recipe["num_inference_steps"]`` (the ``controlnet.num_inference_steps``
    knob the supervised spine fills in), and an explicit per-callback value overrides it.
    """

    subset_size: int = 8
    every_n_epochs: int = 1
    #: Optional per-callback override for the rollout step count. ``None`` ⇒ take it
    #: from the inference recipe (the shared ``controlnet.num_inference_steps`` knob).
    num_inference_steps: int | None = None
    seed: int = 0

    #: The two metrics this callback logs (``ClassVar`` so they are not config knobs
    #: and do not enter :meth:`CallbackRegistry.resolve`'s knob set). Declared so a
    #: checkpoint *could* monitor them, but observe-only keeps ``val/x0_mae``.
    logged_metrics: ClassVar[frozenset[str]] = frozenset({"val/psnr", "val/ssim"})

    def build(self, ctx: CallbackContext) -> pl.Callback:
        num_steps = self.num_inference_steps
        if num_steps is None:
            # Recipe-primary (ADR-0037 / issue #239): the supervised spine fills
            # ctx.inference_recipe from controlnet.num_inference_steps; the literal 15
            # fallback mirrors the callback's own default for a recipe-less context.
            num_steps = int((ctx.inference_recipe or {}).get("num_inference_steps", 15))
        return PairedFidelityCallback(
            module=ctx.module,
            vae=ctx.vae,
            paired_data=ctx.datamodule,  # resolved lazily to the paired val dataset (F5)
            subset_size=self.subset_size,
            every_n_epochs=self.every_n_epochs,
            num_inference_steps=num_steps,
            seed=self.seed,
        )
