"""Export a training ``.ckpt`` to the native per-component inference dir (ADR-0006).

A one-shot bridge from a Lightning training checkpoint to the directory layout
:meth:`~manifold.LatentFlowPipeline.from_pretrained` loads: bake the inference
UNet, take the VAE (carrying ``scaling_factor``) and scheduler, and write via
:meth:`~manifold.LatentFlowPipeline.save_pretrained`. This is the sole
checkpoint -> inference path now that the hope->native converter has been
retired (ADR-0007).

The **raw ``state_dict`` UNet weights** are baked (EMA training was removed;
ADR-0006) - aligned with the ``ModelCheckpoint`` ``val/fid`` monitor, so the
exported "best" checkpoint is best for the weights that are published.
"""

from __future__ import annotations

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..pipelines.latent_flow import LatentFlowPipeline

#: The wrapper's UNet params live under ``self.unet`` (the MAISI backbone), so
#: the Lightning ``state_dict`` (rooted at the module) uses the ``"unet.unet."``
#: prefix (the wrapper's ``self.unet`` + the backbone's ``unet``).
_STATE_PREFIX = "unet.unet."

#: A supervised-ControlNet checkpoint (ADR-0027 stage 1) registers ONLY the
#: trainable ControlNet — the frozen base UNet is held unregistered
#: (``object.__setattr__``), so it is absent from the Lightning ``state_dict``.
#: The ControlNet params are rooted at the module's ``self.controlnet``.
_CONTROLNET_PREFIX = "controlnet."


def _bake_backbone(unet, backbone_state: dict, *, strict: bool) -> None:
    """Load raw-MAISI-keyed weights into the wrapped backbone (``unet.unet``)."""
    unet.unet.load_state_dict(backbone_state, strict=strict)


def export_to_native(
    ckpt_path: str,
    output_dir: str,
    *,
    unet,
    vae,
    scheduler,
    pipeline_cls=None,
    controlnet=None,
) -> str:
    """Convert a training ``.ckpt`` into a native per-component inference dir.

    Args:
        ckpt_path: a Lightning ``.ckpt`` written by ``ModelCheckpoint`` (full
            state - ``save_weights_only=False``). Loaded with
            ``weights_only=False``: only point this at training checkpoints you
            wrote.
        unet / vae / scheduler: fresh components built from the network config;
            the UNet's backbone weights are overwritten from the checkpoint, the
            VAE carries its (already-set) ``scaling_factor``.
        pipeline_cls: the pipeline class whose ``save_pretrained`` writes the dir
            - default :class:`~manifold.LatentFlowPipeline`; pass
            :class:`~manifold.ControlNetLatentFlowPipeline` for the ControlNet
            export (the reward's frozen generator base+ControlNet, ADR-0027). One
            export path: the baking is MAISI-backbone-keyed, so it is reused
            verbatim (the base UNet wraps the same MAISI backbone).
        controlnet: optional fresh ControlNet (built from the network config). When
            given, the checkpoint's ``controlnet.*`` weights are baked into it and
            the pipeline is written as a base UNet + ControlNet + VAE + scheduler
            dir (issue #144 — the supervised ControlNet stage-1 export). The
            checkpoint is a supervised-ControlNet ckpt whose ONLY registered arm is
            the ControlNet (the base is held unregistered, so it carries no
            ``unet.unet.*`` keys); pass the base ``unet`` already holding the frozen
            JiT weights (e.g. loaded from the JiT native export) — it is written
            verbatim, not re-baked.

    Returns:
        A short string naming which weights were baked (``"unet_state_dict"`` for a
        plain JiT export; ``"controlnet_state_dict"`` when ``controlnet`` is given).
    """
    if pipeline_cls is None:
        pipeline_cls = LatentFlowPipeline
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if controlnet is not None:
        # Supervised ControlNet export (ADR-0027 stage 1 -> #144): bake the ckpt's
        # ``controlnet.*`` weights into the fresh ControlNet. The base UNet is NOT
        # re-baked — a supervised ckpt registers only the ControlNet (the base is
        # unregistered), so the caller passes the base already holding the frozen JiT
        # weights. Write the 4-component ControlNet pipeline dir.
        cn_state = {
            k[len(_CONTROLNET_PREFIX):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith(_CONTROLNET_PREFIX)
        }
        if not cn_state:
            raise ValueError(
                f"No 'controlnet.*' keys in {ckpt_path} — not a supervised ControlNet "
                "checkpoint (the ControlNet is its only registered arm)."
            )
        controlnet.load_state_dict(cn_state, strict=True)
        source = "controlnet_state_dict"
        rank_zero_info(
            "Export baking %s (base UNet passed through) -> %s", source, output_dir
        )
        pipeline_cls(unet, controlnet, vae, scheduler).save_pretrained(output_dir)
        return source

    backbone = {
        k[len(_STATE_PREFIX):]: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith(_STATE_PREFIX)
    }
    _bake_backbone(unet, backbone, strict=True)
    source = "unet_state_dict"
    rank_zero_info("Export baking %s as the inference UNet -> %s", source, output_dir)

    pipeline_cls(unet, vae, scheduler).save_pretrained(output_dir)
    return source
