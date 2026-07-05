"""Paired JiT training component: the src→tgt x0 objective as an ``spt.Module``.

A :class:`stable_pretraining.Module` (:class:`spt.Module`) implementing the Paired
JiT training objective (ADR-0013/0014). ``forward(batch, stage)`` samples flow-
times ``t`` from the logit-normal distribution, obtains the interpolated latent
from the **scheduler's** :meth:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler.add_noise`
as ``z_t = scheduler.add_noise(x_tgt, x_src, t) = t·x_tgt + (1−t)·x_src`` (the
transport is the scheduler's, never re-derived — ADR-0001, so train and inference
cannot drift; the scheduler class is shared unchanged, ADR-0013), runs the UNet on
``concat([z_t, x_src])`` with the summed-label condition ``embed(src)+embed(tgt)``
to predict ``x_tgt``, and forms the ``(1 − t)⁻²``-weighted x0-MSE against it.

The module owns **no** ``scale_factor`` (ADR-0003): both src and tgt latents in
the batch are already scaled (the VAE's ``encode`` scaled them; Slice 2 estimates
one ``scale_factor`` over src+tgt pooled). The manual-optimization wiring a
Lightning ``Trainer`` needs to ``fit`` it (Adam over the UNet) is included; the
cosine-with-warmup schedule, AMP-aware grad-norm hook, and in-training
:meth:`sample` land with the training stack (Slice 4) and the PSNR/SSIM callback
(Slice 3) respectively — they consume the same shared rollout primitive.

The objective mirrors :class:`manifold.modules.LatentFlowModule`; only the
transport endpoints (data src at ``t = 0``, data tgt at ``t = 1``), the doubled
input (``concat``), and the summed-label conditioning differ.
"""

from __future__ import annotations

from typing import Any

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.unet_3d_condition import UNet3DConditionModel
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler

#: A Paired JiT training batch: scaled src + tgt latents, contrast labels, spacing.
#: The data stack (Slice 2) produces these; this module only consumes them.
PairedSampleDict = dict[str, Any]


class PairedLatentFlowModule(spt.Module):
    """Paired JiT src→tgt training module (``spt.Module``).

    Args:
        unet: the :class:`~manifold.UNet3DConditionModel` with
            ``in_channels = 2·C_latent`` (predicts x_tgt from ``concat([z_t, x_src])``).
        scheduler: the :class:`~manifold.FlowMatchHeunDiscreteScheduler`; its
            :meth:`add_noise` is the single source of truth for the transport
            (shared with the noise→data JiT, ADR-0013).
        p_mean / p_std: logit-normal timestep sampler parameters
            (``t ~ sigmoid(N(p_mean, p_std))``). Defaults are JiT's published
            values (``-0.8`` / ``0.8``).
        t_eps: floor on ``1 − t`` in the loss denominator (matches the sampler's
            endpoint clamp), avoiding the singularity as ``t → 1``.
        lr: Adam learning rate (paired is trained from scratch, ADR-0014).
    """

    def __init__(
        self,
        unet: UNet3DConditionModel,
        scheduler: FlowMatchHeunDiscreteScheduler,
        *,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        t_eps: float = 0.05,
        lr: float = 1.0e-4,
    ):
        # NOTE: forward is NOT passed to spt.Module — it would double-bind self.
        # Overriding forward directly is the supported pattern (mirrors LatentFlowModule).
        super().__init__(
            hparams={
                "p_mean": p_mean,
                "p_std": p_std,
                "t_eps": t_eps,
                "lr": lr,
            }
        )
        self.unet = unet
        self.scheduler = scheduler
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = float(t_eps)
        self.lr = float(lr)

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Logit-normal ``t ~ sigmoid(N(p_mean, p_std))`` in ``(0, 1)``."""
        logits = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return torch.sigmoid(logits)

    def forward(self, batch: PairedSampleDict, stage: str) -> dict[str, Tensor]:
        """Paired JiT forward: ``(1 − t)⁻²``-weighted MSE on the target latent.

        The interpolated latent comes from ``scheduler.add_noise(x_tgt, x_src, t)``
        (the transport is the scheduler's). The UNet input is ``concat([z_t, x_src])``
        and the loss is ``mean(((x_tgt − x_tgt_pred) / max(1 − t, t_eps))²)``.
        """
        x_src = batch["src_latent"]  # already scaled (ADR-0003): no scale_factor here
        x_tgt = batch["tgt_latent"]  # already scaled (same VAE / scale_factor)
        device = x_src.device
        t = self._sample_timesteps(x_tgt.shape[0], device)  # (B,) in (0, 1)
        noisy = self.scheduler.add_noise(x_tgt, x_src, t)  # z_t = t·x_tgt + (1−t)·x_src

        sample = torch.cat([noisy, x_src], dim=1)  # in_channels = 2·C_latent (ADR-0014)
        x0_pred = self.unet(
            sample=sample,
            timestep=t,
            spacing=batch["spacing"],
            class_labels_src=batch["src_label"],
            class_labels_tgt=batch["tgt_label"],
        )

        t_b = t.view(-1, *([1] * (x_tgt.ndim - 1)))  # broadcast over spatial dims
        weight = (1.0 - t_b).clamp(min=self.t_eps)
        loss = F.mse_loss(x0_pred.float() / weight, x_tgt.float() / weight)

        out: dict[str, Tensor] = {"loss": loss}
        if stage != "fit":
            out["pred"] = x0_pred.detach()  # the target-latent prediction
            out["target"] = x_tgt.detach()  # the target latent x_tgt
        return out

    def configure_optimizers(self):
        """Adam over every UNet param.

        Full fine-tune from scratch (ADR-0014 — no warm-start). The cosine-with-
        warmup schedule + AMP-aware grad-norm hook are wired by the training stack
        (Slice 4), which extends this method verbatim from ``LatentFlowModule``.
        """
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        return {"optimizer": optimizer}


__all__ = ["PairedLatentFlowModule", "PairedSampleDict"]
