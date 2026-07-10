"""End-to-end Paired JiT inference pipeline (src latent → Heun rollout → decoded tgt).

Holds the UNet (``in_channels = 2·C_latent``), VAE, and the **shared** scheduler
and turns a source latent + contrast labels into a decoded 3D target volume in one
call: a start-from-src Heun rollout over the reused scheduler transport
(:func:`manifold.modules.sample_paired_latent_flow`), then sliding-window VAE
decode. The pipeline owns no ``scale_factor`` — the VAE does (ADR-0003): the
rollout operates on already-scaled latents and ``vae.decode`` undoes the scaling.

Sibling of :class:`manifold.pipelines.LatentFlowPipeline`; the rollout starts from
a source data latent (not Gaussian noise) and the conditioning is the summed
contrast embedding. Deterministic given ``x_src`` (no stochastic input) — no
generator is taken. Persistence (``save_pretrained`` / ``from_pretrained`` writing
the ``2·C_latent`` UNet config) lands with the training stack (Slice 4).
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

import torch
from torch import Tensor

from ..models.autoencoder_kl import AutoencoderKL
from ..models.unet_3d_condition import UNet3DConditionModel
from ..modules.paired_sampler import sample_paired_latent_flow
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from .pipeline_utils import DiffusionPipeline

#: Top-level index naming the pipeline and its per-component layout.
_MODEL_INDEX_FILE = "model_index.json"


class PairedLatentFlowPipeline(DiffusionPipeline):
    """Paired JiT translation: src latent → Heun rollout → sliding-window VAE decode."""

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
        # persistence enumerates them in model_index.json (mirrors LatentFlowPipeline).
        self._internal_dict: dict = {}

    # -- persistence (native per-component format; mirrors LatentFlowPipeline) --

    def save_pretrained(self, save_directory: str) -> None:
        """Write the pipeline as a per-component directory layout.

        ``model_index.json`` identifies the pipeline; each component writes its
        own ``config.json`` (and, for models, ``diffusion_pytorch_model.pt``)
        into its subdirectory. The UNet's ``in_channels = 2·C_latent`` config
        round-trips through its own ``config.json`` (seam #6, ADR-0014).
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
    def from_pretrained(cls, save_directory: str) -> "PairedLatentFlowPipeline":
        """Load a pipeline written by :meth:`save_pretrained` (native format only)."""
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
        src_latent: Tensor,
        spacing: Tensor | Sequence[float],
        src_label: int | Tensor,
        tgt_label: int | Tensor,
        num_inference_steps: int,
    ) -> Tensor:
        """Run the start-from-src Heun rollout; return the target latent ``[B, C, D, H, W]``.

        Thin delegate over the shared
        :func:`~manifold.modules.sample_paired_latent_flow` primitive (the
        ADR-0005 analogue) — inference is packaging over the same rollout the
        Module's ``sample()`` (Slice 3) uses, so the two paths cannot drift. No VAE
        decode. Exposed so the end-to-end reconstruct test (Slice 1) and numerical
        validation can compare the latent directly.

        ``src_label`` / ``tgt_label`` accept a scalar ``int`` (one direction
        broadcast across the batch — the published-inference contract used by
        :meth:`__call__`) or a ``[B]`` long tensor of per-sample labels (the
        validation contract — a Paired JiT val batch mixes all 12 within-subject
        contrast directions, so the PSNR/SSIM callback passes per-sample labels so
        each volume is conditioned on its own src→tgt pair).
        """
        return sample_paired_latent_flow(
            self.unet,
            self.scheduler,
            src_latent,
            spacing,
            src_label,
            tgt_label,
            num_inference_steps=num_inference_steps,
        )

    def __call__(
        self,
        src_latent: Tensor,
        spacing: Tensor | Sequence[float],
        src_label: int,
        tgt_label: int,
        num_inference_steps: int,
    ) -> Tensor:
        """Translate a src latent into a normalized decoded target volume ``[B, C, D, H, W]``.

        The start-from-src Heun rollout, then a float32 VAE decode with
        ``norm_float16`` disabled (mirrors ``FIDCallback._eval_decode``: more
        correct than the autocast half-precision path used before), then a
        per-volume min-max normalization to ``[0, 1]`` (the FID feature-arm
        preprocessing, so the published inference output is a normalized image).
        Deterministic
        given ``src_latent`` (the rollout has no stochastic input — ADR-0013).

        Args:
            src_latent: the source latent ``[B, C_latent, D, H, W]`` (already
                scaled); the rollout starts here and the returned image volume is
                the VAE upsample of the predicted target latent.
            spacing: raw voxel spacing ``[3]`` (or ``[B, 3]``).
            src_label / tgt_label: integer contrast labels whose embeddings are
                summed for the translation direction (ADR-0014).
            num_inference_steps: number of Heun integration steps over ``t: 0 → 1``.

        Returns:
            The decoded target volume ``[B, C_image, D, H, W]`` normalized to
            ``[0, 1]`` per volume.
        """
        latent = self.sample_latent(
            src_latent, spacing, src_label, tgt_label, num_inference_steps
        )
        self.vae.eval()
        with torch.inference_mode():
            vol = self._decode_f32(latent)
            return self._minmax_to_unit(vol)

    # -- decode + post-process (mirror FID eval) ----------------------------

    def _decode_f32(self, latents: Tensor) -> Tensor:
        """Float32 VAE decode with ``norm_float16`` disabled (mirrors FID eval).

        Mirrors ``FIDCallback._eval_decode`` (and ``PairedPSNRSSIMCallback``):
        the migrated VAE's MaisiGroupNorm3D carries ``norm_float16`` (casts its
        output to half unconditionally), so a downstream float32 conv raises a
        Half/float bias-type mismatch unless an outer autocast reconciles it.
        Disabling ``norm_float16`` once (idempotent) lets the whole decode run in
        float32 - both robust and more correct than half precision. The latent is
        moved to the VAE's device; the VAE undoes ``scaling_factor`` internally
        (ADR-0003).
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

        Mirrors the per-volume min-max in FID's RadImageNet preprocessing
        (``_RadImageNetFeatures.forward``: one global min/max over the whole
        volume), so the published inference output is a normalized image regardless
        of the raw VAE decode range. Per-sample (each volume in the batch is
        normalized by its own ``[min, max]``); a degenerate zero-range volume maps
        to zeros.
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
