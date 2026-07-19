"""End-to-end ControlNet latent-flow inference pipeline (noise → decoded tgt volume).

Holds the **frozen base UNet + the trainable ControlNet + scheduler + VAE** and
turns pure noise + a source control signal + medical conditions into a decoded 3D
target volume in one call: a noise→data ControlNet Heun rollout
(:func:`manifold.modules.controlnet_rollout`), then a VAE decode. The source
``x_src`` is a **control signal** (ControlNet condition), not a transport endpoint —
the transport is the canonical noise→data rectified flow shared with unconditional
JiT generation (ADR-0026/0027). The pipeline owns no ``scale_factor`` — the VAE
does (ADR-0003).

Persistence uses the native per-component checkpoint contract (``model_index.json``
+ one subdirectory per component), so trained components reach inference through the
same export path as the JiT UNet. No CFG in v1 (the ControlNet conditioning has no
unconditional path to guide against — spec Out-of-Scope).

Sibling of :class:`manifold.pipelines.LatentFlowPipeline`.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

import torch
from torch import Tensor

from ..models.autoencoder_kl import AutoencoderKL
from ..models.controlnet_3d import ControlNet3DConditionModel
from ..models.unet_3d_condition import UNet3DConditionModel
from ..modules.controlnet_sampler import controlnet_rollout
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from .pipeline_utils import DiffusionPipeline

#: Top-level index naming the pipeline and its per-component layout.
_MODEL_INDEX_FILE = "model_index.json"


class ControlNetLatentFlowPipeline(DiffusionPipeline):
    """ControlNet translation: noise + x_src control → Heun rollout → VAE decode."""

    def __init__(
        self,
        unet: UNet3DConditionModel,
        controlnet: ControlNet3DConditionModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchHeunDiscreteScheduler,
    ):
        self.unet = unet
        self.controlnet = controlnet
        self.vae = vae
        self.scheduler = scheduler
        # No serializable ctor args (the components are objects, not config);
        # persistence enumerates them in model_index.json (mirrors LatentFlowPipeline).
        self._internal_dict: dict = {}

    # -- persistence (native per-component format; mirrors LatentFlowPipeline) --

    def save_pretrained(self, save_directory: str) -> None:
        """Write the pipeline as a per-component directory layout.

        ``model_index.json`` identifies the pipeline; each component (base UNet,
        ControlNet, VAE, scheduler) writes its own ``config.json`` (and, for models,
        ``diffusion_pytorch_model.pt``) into its subdirectory.
        """
        os.makedirs(save_directory, exist_ok=True)
        index = {
            "format": "manifold",
            "pipeline_class": type(self).__name__,
            "components": {
                "unet": _qualname(self.unet),
                "controlnet": _qualname(self.controlnet),
                "vae": _qualname(self.vae),
                "scheduler": _qualname(self.scheduler),
            },
        }
        with open(os.path.join(save_directory, _MODEL_INDEX_FILE), "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        self.unet.save_pretrained(os.path.join(save_directory, "unet"))
        self.controlnet.save_pretrained(os.path.join(save_directory, "controlnet"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))
        # The scheduler is stateless (config only — no weights).
        self.scheduler.to_json_file(
            os.path.join(save_directory, "scheduler", self.scheduler.config_name)
        )

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "ControlNetLatentFlowPipeline":
        """Load a pipeline written by :meth:`save_pretrained` (native format only)."""
        index_path = os.path.join(save_directory, _MODEL_INDEX_FILE)
        if not os.path.isdir(save_directory) or not os.path.isfile(index_path):
            raise FileNotFoundError(
                f"{save_directory!r} is not a manifold pipeline directory "
                f"(missing {_MODEL_INDEX_FILE})."
            )
        unet = UNet3DConditionModel.from_pretrained(os.path.join(save_directory, "unet"))
        controlnet = ControlNet3DConditionModel.from_pretrained(
            os.path.join(save_directory, "controlnet")
        )
        vae = AutoencoderKL.from_pretrained(os.path.join(save_directory, "vae"))
        scheduler = FlowMatchHeunDiscreteScheduler.from_json_file(
            os.path.join(save_directory, "scheduler", FlowMatchHeunDiscreteScheduler.config_name)
        )
        return cls(unet, controlnet, vae, scheduler)

    def sample_latent(
        self,
        noise: Tensor,
        src_latent: Tensor,
        spacing: Tensor | Sequence[float],
        src_label: int | Tensor,
        tgt_label: int | Tensor,
        num_inference_steps: int,
    ) -> Tensor:
        """Run the ControlNet noise→data rollout; return the target latent ``[B, C, D, H, W]``.

        Thin delegate over the shared
        :func:`~manifold.modules.controlnet_rollout` primitive (the ADR-0005
        analogue) — inference is packaging over the same rollout the supervised
        Module's validation uses, so the two paths cannot drift. No VAE decode.
        Exposed so numerical validation can compare the latent trajectory directly.

        Args:
            noise: the pure-noise start latent ``[B, C_latent, D, H, W]`` (the
                stochastic input — the noise→data transport's ``t = 0`` endpoint).
            src_latent: the source latent control signal ``[B, C_latent, D, H, W]``
                (already scaled); the ControlNet's ``x_src`` condition.
            spacing: raw voxel spacing ``[3]`` or ``[B, 3]``.
            src_label / tgt_label: the (src, tgt) contrast pair (scalar broadcast or
                ``[B]`` per-sample).
            num_inference_steps: Heun integration steps over ``t: 0 → 1``.

        Returns:
            The predicted target latent ``[B, C_latent, D, H, W]``.
        """
        return controlnet_rollout(
            self.unet,
            self.controlnet,
            self.scheduler,
            noise,
            src_latent,
            spacing,
            src_label,
            tgt_label,
            num_inference_steps=num_inference_steps,
        )

    def __call__(
        self,
        noise: Tensor,
        src_latent: Tensor,
        spacing: Tensor | Sequence[float],
        src_label: int,
        tgt_label: int,
        num_inference_steps: int,
    ) -> Tensor:
        """Translate noise + a src control signal into a decoded target volume ``[B, C, D, H, W]``.

        The ControlNet noise→data rollout, then a float32 VAE decode with
        ``norm_float16`` disabled (mirrors ``FIDCallback._eval_decode``), then a
        per-volume min-max normalization to
        ``[0, 1]`` (the published-inference output contract).

        Args:
            noise: the pure-noise start latent ``[B, C_latent, D, H, W]``.
            src_latent: the source latent control signal ``[B, C_latent, D, H, W]``.
            spacing: raw voxel spacing ``[3]`` (or ``[B, 3]``).
            src_label / tgt_label: integer contrast labels for the translation
                direction (the ControlNet direction MLP).
            num_inference_steps: number of Heun integration steps over ``t: 0 → 1``.

        Returns:
            The decoded target volume ``[B, C_image, D, H, W]`` normalized to
            ``[0, 1]`` per volume.
        """
        latent = self.sample_latent(
            noise, src_latent, spacing, src_label, tgt_label, num_inference_steps
        )
        self.vae.eval()
        with torch.inference_mode():
            vol = self._decode_f32(latent)
            return self._minmax_to_unit(vol)

    # -- decode + post-process (mirror FID eval) ----

    def _decode_f32(self, latents: Tensor) -> Tensor:
        """Float32 VAE decode with ``norm_float16`` disabled (mirrors FID eval).

        The migrated VAE's MaisiGroupNorm3D carries ``norm_float16`` (casts its
        output to half unconditionally), so a downstream float32 conv raises a
        Half/float bias-type mismatch unless an outer autocast reconciles it.
        Disabling ``norm_float16`` once (idempotent) lets the whole decode run in
        float32. The latent is moved to the VAE's device; the VAE undoes
        ``scaling_factor`` internally (ADR-0003).
        """
        if not getattr(self, "_norm16_disabled", False):
            for m in self.vae.modules():
                if hasattr(m, "norm_float16"):
                    m.norm_float16 = False
            self._norm16_disabled = True
        vae_device = next(self.vae.parameters()).device
        return self.vae.decode(latents.float().to(vae_device))

    @staticmethod
    def _minmax_to_unit(vol: Tensor) -> Tensor:
        """Per-volume min-max normalization to ``[0, 1]`` (FID feature-arm step).

        Mirrors the per-volume min-max in FID's RadImageNet preprocessing, so the
        published inference output is a normalized image regardless of the raw VAE
        decode range. Per-sample (each volume normalized by its own ``[min, max]``);
        a degenerate zero-range volume maps to zeros.
        """
        b = vol.shape[0]
        flat = vol.reshape(b, -1)  # [B, C*D*H*W]
        mn = flat.amin(dim=1).view(b, 1, 1, 1, 1)
        mx = flat.amax(dim=1).view(b, 1, 1, 1, 1)
        rng = mx - mn
        rng = torch.where(rng > 0, rng, torch.ones_like(rng))  # avoid div-by-zero
        return (vol - mn) / rng


# -- model_index helper -----------------------------------------------------


def _qualname(component: Any) -> str:
    """``module.ClassName`` for a component (written to model_index.json)."""
    return f"{type(component).__module__}.{type(component).__name__}"
