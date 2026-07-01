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

import json
import os
from typing import Any, Sequence

import torch
from torch import Tensor

from ..models.autoencoder_kl import AutoencoderKL
from ..models.unet_3d_condition import UNet3DConditionModel
from ..modules.sampler import sample_latent_flow
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from .pipeline_utils import DiffusionPipeline

#: Top-level index naming the pipeline and its per-component layout.
_MODEL_INDEX_FILE = "model_index.json"


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
        # No serializable ctor args (the components are objects, not config);
        # persistence enumerates them in model_index.json (issue #7).
        self._internal_dict: dict = {}

    # -- persistence (native per-component format; ADR-0003) -----------------

    def save_pretrained(self, save_directory: str) -> None:
        """Write the pipeline as a per-component directory layout.

        ``model_index.json`` identifies the pipeline; each component writes its
        own ``config.json`` (and, for models, ``diffusion_pytorch_model.pt``)
        into its subdirectory.
        """
        os.makedirs(save_directory, exist_ok=True)
        index = {
            "format": "manifold",
            "pipeline_class": type(self).__name__,
            "components": {
                "unet": _qualname(self.unet),
                "vae": _qualname(self.vae),
                "scheduler": _qualname(self.scheduler),
            },
        }
        with open(os.path.join(save_directory, _MODEL_INDEX_FILE), "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        # The scheduler is stateless (config only — no weights).
        self.scheduler.to_json_file(
            os.path.join(save_directory, "scheduler", self.scheduler.config_name)
        )

    @classmethod
    def from_pretrained(cls, save_directory: str) -> LatentFlowPipeline:
        """Load a pipeline written by :meth:`save_pretrained`.

        Reads only the native per-component format.
        """
        index_path = os.path.join(save_directory, _MODEL_INDEX_FILE)
        if not os.path.isdir(save_directory) or not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{save_directory!r} is not a manifold pipeline directory "
                f"(missing {_MODEL_INDEX_FILE})."
            )
        unet = UNet3DConditionModel.from_pretrained(os.path.join(save_directory, "unet"))
        vae = AutoencoderKL.from_pretrained(os.path.join(save_directory, "vae"))
        scheduler = FlowMatchHeunDiscreteScheduler.from_json_file(
            os.path.join(save_directory, "scheduler", FlowMatchHeunDiscreteScheduler.config_name)
        )
        return cls(unet, vae, scheduler)

    def sample_latent(
        self,
        target_shape: Sequence[int],
        spacing: Tensor | Sequence[float],
        modality: int,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        cfg_interval: tuple[float, float] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Run the Heun rollout and return the final latent ``[B, C_latent, D, H, W]``.

        Thin delegate over the shared :func:`~manifold.modules.sample_latent_flow`
        primitive (ADR-0005) — inference is packaging over the same rollout the
        Module's ``sample()`` uses, so the two paths cannot drift. The
        pure-noise start is built from *generator* here; the rollout itself runs
        under cuda autocast. No VAE decode. Exposed so numerical validation can
        compare the latent trajectory directly (issue #18).
        """
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        noise = torch.randn(target_shape, generator=generator, device=device, dtype=dtype)
        return sample_latent_flow(
            self.unet,
            self.scheduler,
            noise,
            spacing,
            modality,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cfg_interval=cfg_interval,
        )

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

        Equivalent to ``self.vae.decode(self.sample_latent(...))``. Args mirror
        :meth:`sample_latent`:

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
        latent = self.sample_latent(
            target_shape,
            spacing,
            modality,
            num_inference_steps,
            guidance_scale=guidance_scale,
            cfg_interval=cfg_interval,
            generator=generator,
        )
        self.vae.eval()
        # Decode under cuda autocast: the migrated VAE carries ``norm_float16``
        # (half-precision norms), so a float32 decode path hits a Half/float
        # dtype mismatch. Disabled off-cuda.
        with (
            torch.inference_mode(),
            torch.autocast(device_type=latent.device.type, enabled=latent.device.type == "cuda"),
        ):
            return self.vae.decode(latent)


# -- model_index helper -----------------------------------------------------


def _qualname(component: Any) -> str:
    """``module.ClassName`` for a component (written to model_index.json)."""
    return f"{type(component).__module__}.{type(component).__name__}"
