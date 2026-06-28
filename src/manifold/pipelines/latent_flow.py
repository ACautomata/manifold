"""End-to-end latent-flow inference pipeline (JiT x0-denoiser → decoded volume).

Holds the UNet, VAE, and scheduler and turns noise + medical conditions into a
decoded 3D volume in one call: a latent rollout under the true two-evaluation
Heun (:meth:`manifold.schedulers.FlowMatchHeunDiscreteScheduler.euler_step` /
:meth:`heun_correct`), then sliding-window VAE decode. The pipeline owns no
``scale_factor`` — the VAE does (ADR-0003): the rollout operates on already-
scaled latents and ``vae.decode`` undoes the scaling.

Denoising-interval classifier-free guidance (issue #5) wraps the UNet calls
*inside* the pipeline: per Heun evaluation point the conditional (spacing +
class label) and unconditional (label dropped) outputs are combined as
``uncond + guidance_scale·(cond − uncond)``, but only for flow-times ``t`` inside
``cfg_interval``. The scheduler stays pure (Heun on velocities).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..models.autoencoder_kl import AutoencoderKL
from ..models.unet_3d_condition import UNet3DConditionModel
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from .pipeline_utils import DiffusionPipeline


class LatentFlowPipeline(DiffusionPipeline):
    """Latent-flow generation: noise → Heun rollout → sliding-window VAE decode."""

    def __init__(
        self,
        unet: UNet3DConditionModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchHeunDiscreteScheduler,
    ):
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self._internal_dict: dict = {}

    def __call__(
        self,
        target_shape: Sequence[int],
        spacing: Tensor | Sequence[float],
        modality: int,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        cfg_interval: tuple[float, float] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate a decoded volume ``[B, C, D, H, W]`` from pure noise.

        Args:
            target_shape: the latent shape ``[B, C_latent, D, H, W]`` to generate;
                the returned image volume is the VAE upsample of this latent.
            spacing: raw voxel spacing ``[3]`` (or ``[B, 3]``).
            modality: the integer class label (modality) for conditioning.
            num_inference_steps: number of Heun integration steps over ``t: 0 → 1``.
            guidance_scale: classifier-free guidance strength. ``1.0`` (the
                default) runs the no-CFG path (a single conditional forward).
                Values ``!= 1.0`` apply CFG inside ``cfg_interval`` only.
            cfg_interval: ``(low, high)`` — guidance is active for flow-times
                ``t`` with ``low < t < high``. ``None`` (the default) disables
                CFG entirely; a degenerate interval covering no timestep also
                reproduces the no-CFG path exactly.
            generator: optional :class:`torch.Generator` for the starting noise.

        Returns:
            The decoded volume ``[B, C_image, D·, H·, W·]`` with a finite range.
        """
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        batch_size = int(target_shape[0])

        spacing = torch.as_tensor(spacing, device=device)
        class_labels = torch.full((batch_size,), int(modality), dtype=torch.long, device=device)

        def unet_call(z: Tensor, t: float) -> Tensor:
            """One (possibly CFG-combined) UNet evaluation at flow-time ``t``.

            ``guidance_scale == 1.0`` short-circuits to the conditional output —
            no unconditional forward, so the ``= 1`` path is bit-identical to the
            no-CFG path. With an active interval the label is dropped for the
            unconditional pass and the two are combined; outside the interval the
            conditional output is used directly.
            """
            cond = self.unet(
                sample=z, timestep=float(t), spacing=spacing, class_labels=class_labels
            )
            if guidance_scale == 1.0 or cfg_interval is None:
                return cond
            low, high = cfg_interval
            if not (low < t < high):
                return cond
            uncond = self.unet(
                sample=z,
                timestep=float(t),
                spacing=spacing,
                class_labels=torch.zeros_like(class_labels),
            )
            return uncond + guidance_scale * (cond - uncond)

        z = torch.randn(target_shape, generator=generator, device=device, dtype=dtype)
        nodes = self.scheduler.set_timesteps(num_inference_steps, device=device)
        n = int(num_inference_steps)

        self.unet.eval()
        self.vae.eval()
        with torch.inference_mode():
            for i in range(n):
                t = float(nodes[i])
                t_next = float(nodes[i + 1])
                x0_1 = unet_call(z, t)
                z_euler, v1 = self.scheduler.euler_step(x0_1, z, t, t_next)
                if i == n - 1:
                    # Final step is Euler: at t_next = 1 the denominator 1 − t_next
                    # vanishes, so the second Heun evaluation is undefined.
                    z = z_euler
                else:
                    x0_2 = unet_call(z_euler, t_next)
                    z = self.scheduler.heun_correct(x0_2, z, z_euler, v1, t, t_next)
            volume = self.vae.decode(z)
        return volume
