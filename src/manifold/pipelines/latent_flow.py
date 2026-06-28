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

        Reads only the native per-component format. A hope flat checkpoint
        (``{unet_state_dict, scale_factor, ema}``) is **not** accepted — convert
        it first with :func:`convert_hope_checkpoint` (ADR-0003).
        """
        index_path = os.path.join(save_directory, _MODEL_INDEX_FILE)
        if not os.path.isdir(save_directory) or not os.path.isfile(index_path):
            _reject_if_hope_flat(save_directory)  # raises a helpful error
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


# -- checkpoint conversion: hope flat format -> native (ADR-0003) ------------


def _qualname(component: Any) -> str:
    """``module.ClassName`` for a component (written to model_index.json)."""
    return f"{type(component).__module__}.{type(component).__name__}"


def _reject_if_hope_flat(path: str) -> None:
    """Raise a clear error if *path* is a hope flat checkpoint, else return.

    hope flat checkpoints are a single dict ``{unet_state_dict, scale_factor,
    ema}``; ``from_pretrained`` must refuse them so the converter stays the sole
    migration path in.
    """
    if os.path.isfile(path):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # not a torch checkpoint (or unsafe pickle) — fall through
            return
        if isinstance(payload, dict) and "unet_state_dict" in payload:
            raise ValueError(
                f"{path!r} is a hope flat checkpoint (has 'unet_state_dict'), not a "
                "manifold pipeline directory. Convert it first with "
                "`manifold.pipelines.latent_flow.convert_hope_checkpoint` (or the "
                "scripts/convert_hope_checkpoint.py CLI)."
            )


def _select_inference_weights(hope: dict, *, prefer_ema: bool) -> tuple[dict, str]:
    """Pick the inference UNet weights from a hope flat checkpoint.

    hope's inference samples the EMA copy: when EMA shadows are present (and not
    disabled), the slowest shadow (largest decay) is the published model and is
    baked as the inference weights; otherwise the raw ``unet_state_dict``.
    Returns ``(state_dict, source)`` where *source* names which was used.
    """
    ema = hope.get("ema")
    if prefer_ema and isinstance(ema, dict):
        shadows = ema.get("shadows") or []
        if shadows:
            decays = ema.get("decays") or []
            idx = max(range(len(shadows)), key=lambda i: decays[i]) if decays else len(shadows) - 1
            return shadows[idx], f"ema[decay={decays[idx]}]" if decays else "ema[last]"
    if "unet_state_dict" not in hope:
        raise KeyError("hope checkpoint has no 'unet_state_dict' (and no usable EMA shadow).")
    return hope["unet_state_dict"], "unet_state_dict"


def convert_hope_checkpoint(
    hope_checkpoint: str | dict,
    output_directory: str,
    unet: UNet3DConditionModel,
    vae: AutoencoderKL,
    scheduler: FlowMatchHeunDiscreteScheduler,
    *,
    prefer_ema: bool = True,
) -> str:
    """Convert a hope flat checkpoint to manifold's native per-component format.

    Maps ``unet_state_dict → unet`` (or, by default, the **slowest EMA shadow** —
    hope's inference copy — baked as the UNet weights), and
    ``scale_factor → vae.scaling_factor``. The provided ``unet``/``vae``/
    ``scheduler`` carry the target component configs; their weights/scale are set
    from the checkpoint, then the pipeline is written via
    :meth:`LatentFlowPipeline.save_pretrained`. Returns *output_directory*.
    """
    hope = (
        # hope flat checkpoints hold only tensors + simple containers (state
        # dicts, the scale_factor tensor, and an {shadows, decays} EMA dict), so
        # weights_only=True loads them safely without unpickling arbitrary code.
        torch.load(hope_checkpoint, map_location="cpu", weights_only=True)
        if isinstance(hope_checkpoint, str)
        else hope_checkpoint
    )
    if not isinstance(hope, dict) or "unet_state_dict" not in hope:
        raise KeyError("Not a hope flat checkpoint: expected a dict with 'unet_state_dict'.")

    weights, source = _select_inference_weights(hope, prefer_ema=prefer_ema)
    # hope's ``unet_state_dict`` (and EMA shadows) are the RAW MAISI UNet's state
    # dict — no wrapper prefix — so they load into the wrapped backbone directly.
    unet.unet.load_state_dict(weights, strict=True)  # bake inference weights into the UNet

    scale_factor = hope.get("scale_factor")
    if scale_factor is None:
        raise KeyError("hope checkpoint has no 'scale_factor' to map onto vae.scaling_factor.")
    with torch.no_grad():
        vae.scaling_factor.fill_(float(scale_factor))

    LatentFlowPipeline(unet, vae, scheduler).save_pretrained(output_directory)
    return output_directory
