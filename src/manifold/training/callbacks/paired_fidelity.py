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

This ticket is verified at the unit + registry seams and is **not yet wired into real
training** (that is the follow-up). ``build`` forwards ``ctx.datamodule`` as the
paired-subset source; the callback resolves the actual paired dataset lazily at the
first gated epoch (``getattr(datamodule, "val_latent_ds", datamodule)``, F5), so the
cold path's post-``setup()`` replacement of ``val_latent_ds`` is honored.
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

    Matches the :class:`CallbackSpec` Protocol structurally. The generation recipe
    (``num_inference_steps``) and the fixed-subset / cadence / seed knobs are declared
    here as config fields; the runtime objects come from :class:`CallbackContext`.
    """

    subset_size: int = 8
    every_n_epochs: int = 1
    num_inference_steps: int = 15
    seed: int = 0

    #: The two metrics this callback logs (``ClassVar`` so they are not config knobs
    #: and do not enter :meth:`CallbackRegistry.resolve`'s knob set). Declared so a
    #: checkpoint *could* monitor them, but observe-only keeps ``val/x0_mae``.
    logged_metrics: ClassVar[frozenset[str]] = frozenset({"val/psnr", "val/ssim"})

    def build(self, ctx: CallbackContext) -> pl.Callback:
        return PairedFidelityCallback(
            module=ctx.module,
            vae=ctx.vae,
            paired_data=ctx.datamodule,  # resolved lazily to the paired val dataset (F5)
            subset_size=self.subset_size,
            every_n_epochs=self.every_n_epochs,
            num_inference_steps=self.num_inference_steps,
            seed=self.seed,
        )
