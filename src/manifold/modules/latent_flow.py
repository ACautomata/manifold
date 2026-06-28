"""JiT training component: the x0-denoiser objective as a stable-pretraining Module.

A :class:`stable_pretraining.Module` (:class:`spt.Module`) implementing the JiT
x0-denoiser training objective. ``forward(batch, stage)`` samples flow-times
``t`` from the logit-normal distribution, obtains the noised latent from the
**scheduler's** :meth:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler.add_noise`
(the transport is the scheduler's, never re-derived — ADR-0001, so train and
inference cannot drift), runs the UNet to predict the clean latent x0, and forms
the ``(1 − t)⁻²``-weighted MSE against it.

The module owns **no** ``scale_factor`` (ADR-0003): latents in the batch are
already scaled (the VAE's ``encode`` scaled them). This slice delivers the
forward + loss only; optimizer / grad-norm / LR-scheduler wiring belongs to the
deferred trainer stack.

Ported faithfully from the x0 path in ``hope/modules/rflow.py`` (``_forward_x0``);
``t = 1 → data`` matches the scheduler and the sampler.
"""

from __future__ import annotations

from typing import Any

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.unet_3d_condition import UNet3DConditionModel
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler

#: A training batch: a scaled latent + medical conditioning. The data stack
#: (deferred) produces these; this module only consumes them.
SampleDict = dict[str, Any]


class LatentFlowModule(spt.Module):
    """JiT x0-denoiser training module (``spt.Module``).

    Args:
        unet: the :class:`~manifold.UNet3DConditionModel` (predicts x0).
        scheduler: the :class:`~manifold.FlowMatchHeunDiscreteScheduler`; its
            :meth:`add_noise` is the single source of truth for the transport.
        p_mean / p_std: logit-normal timestep sampler parameters
            (``t ~ sigmoid(N(p_mean, p_std))``). Defaults are JiT's published
            values (``-0.8`` / ``0.8``).
        t_eps: floor on ``1 − t`` in the loss denominator (matches the sampler's
            endpoint clamp), avoiding the singularity as ``t → 1``.
        include_modality: whether the UNet takes a class-label condition.
    """

    def __init__(
        self,
        unet: UNet3DConditionModel,
        scheduler: FlowMatchHeunDiscreteScheduler,
        *,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        t_eps: float = 0.05,
        include_modality: bool = True,
    ):
        # NOTE: forward is NOT passed to spt.Module — it would double-bind self.
        # Overriding forward directly is the supported pattern (hope prior art).
        super().__init__(hparams={"p_mean": p_mean, "p_std": p_std, "t_eps": t_eps})
        self.unet = unet
        self.scheduler = scheduler
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = float(t_eps)
        self.include_modality = bool(include_modality)

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Logit-normal ``t ~ sigmoid(N(p_mean, p_std))`` in ``(0, 1)``."""
        logits = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return torch.sigmoid(logits)

    def forward(self, batch: SampleDict, stage: str) -> dict[str, Tensor]:
        """JiT x0-denoiser forward: ``(1 − t)⁻²``-weighted MSE on the clean latent.

        The noised latent comes from ``scheduler.add_noise`` (the transport is the
        scheduler's). The loss is ``mean(((x0 − x0_pred) / max(1 − t, t_eps))²)``.
        """
        latent = batch["latent"]  # already scaled (ADR-0003): no scale_factor here
        device = latent.device
        noise = torch.randn_like(latent)
        t = self._sample_timesteps(latent.shape[0], device)  # (B,) in (0, 1)
        noisy = self.scheduler.add_noise(latent, noise, t)  # z = t·x + (1−t)·e

        spacing = batch["spacing"]
        kwargs: dict[str, Any] = {"spacing": spacing}
        if self.include_modality and "label" in batch:
            kwargs["class_labels"] = batch["label"]

        x0_pred = self.unet(sample=noisy, timestep=t, **kwargs)

        t_b = t.view(-1, *([1] * (latent.ndim - 1)))  # broadcast over spatial dims
        weight = (1.0 - t_b).clamp(min=self.t_eps)
        loss = F.mse_loss(x0_pred.float() / weight, latent.float() / weight)

        out: dict[str, Tensor] = {"loss": loss}
        if stage != "fit":
            out["pred"] = x0_pred.detach()  # the clean-latent prediction
            out["target"] = latent.detach()  # the clean latent x0
        return out
