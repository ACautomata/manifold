"""Spec for :class:`~manifold.metrics.FIDCallback` — the ``val/fid`` logger.

The generative FID callback migrated behind the :class:`CallbackRegistry`
(ADR-0029, issue #160). Its four knobs (``num_synth`` / ``every_n_epochs`` /
``center_slices_ratio`` / ``cov_ridge``) come from the ``fid`` config block; the
generation recipe (``latent_shape`` … ``cfg_interval``) and the runtime objects
(``module`` / ``vae`` / ``feature_net`` / ``feature_net_factory`` / lazy real
latents) are injected from :class:`CallbackContext` at ``build``.

``logged_metrics`` declares what the callback contributes to the monitorable set
so :meth:`CallbackRegistry.validate_monitor` can validate a checkpoint's
``monitor_metric`` against the resolved callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import lightning.pytorch as pl

from manifold.metrics import FIDCallback
from manifold.training.callbacks.context import CallbackContext


@dataclass(frozen=True)
class FIDSpec:
    """Registry spec for :class:`FIDCallback` (the ``fid`` block knobs).

    Matches the :class:`CallbackSpec` Protocol structurally. The lazy real-latents
    pull (ADR-0017 / F5) is preserved: ``build`` passes ``real_latents`` directly
    when GRPO supplies it via :attr:`CallbackContext.real_latents`, and always
    hands the JiT fallback ``real_latents_source=datamodule`` (the callback pulls
    ``datamodule.val_latents`` lazily once the post-setup warm populates it).
    """

    num_synth: int = 16
    every_n_epochs: int = 1
    center_slices_ratio: float = 0.5
    cov_ridge: float = 1e-6

    #: The single monitored metric this callback logs (``ClassVar`` so it is not a
    #: config knob and does not enter :meth:`CallbackRegistry.resolve`'s knob set).
    logged_metrics: ClassVar[frozenset[str]] = frozenset({"val/fid"})

    def build(self, ctx: CallbackContext) -> pl.Callback:
        inf = ctx.inference_recipe
        return FIDCallback(
            module=ctx.module,
            vae=ctx.vae,
            real_latents=ctx.real_latents,  # GRPO-supplied wins; None for JiT.
            real_latents_source=ctx.datamodule,  # JiT lazy fallback (F5).
            feature_net=ctx.feature_net,
            feature_net_factory=ctx.feature_net_factory,
            latent_shape=inf["latent_shape"],
            spacing=inf["spacing"],
            modality=inf["modality"],
            num_inference_steps=inf["num_inference_steps"],
            guidance_scale=inf["guidance_scale"],
            cfg_interval=inf["cfg_interval"],
            num_synth=self.num_synth,
            every_n_epochs=self.every_n_epochs,
            center_slices_ratio=self.center_slices_ratio,
            cov_ridge=self.cov_ridge,
            seed=ctx.seed,
        )
