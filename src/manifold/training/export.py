"""Export a training ``.ckpt`` to the native per-component inference dir (ADR-0006).

A one-shot bridge from a Lightning training checkpoint to the directory layout
:meth:`~manifold.LatentFlowPipeline.from_pretrained` loads: bake the inference
UNet, take the VAE (carrying ``scaling_factor``) and scheduler, and write via
:meth:`~manifold.LatentFlowPipeline.save_pretrained`. This is the sole
checkpoint → inference path now that the hope→native converter has been
retired (ADR-0007).

By default the **raw ``state_dict`` UNet weights** are baked — aligned with the
``ModelCheckpoint`` raw-FID monitor (``val/fid_raw``), so the exported "best"
checkpoint is best for the weights that are published. Pass ``prefer_ema=True``
to bake the slowest EMA shadow instead, for runs where the 0.9999 EMA has
converged (warm-start / long horizon, as hope trains). The EMA shadows live in
the checkpoint's callback state (``callbacks['DoubleEMACallback']``); the
slowest (largest decay) is the generation/eval model the FID callback sampled.
"""

from __future__ import annotations

import logging

import torch

from ..pipelines.latent_flow import LatentFlowPipeline
from .ema import slowest_shadow_index

_log = logging.getLogger(__name__)

#: The wrapper's UNet params live under ``self.unet`` (the MAISI backbone), so
#: ``module.unet.named_parameters()`` (the EMA shadow source) yields keys with a
#: single ``"unet."`` prefix, while the Lightning ``state_dict`` (rooted at the
#: module) uses ``"unet.unet."``.
_WRAPPER_PREFIX = "unet."
_STATE_PREFIX = "unet.unet."


def _slowest_ema_shadow(ckpt: dict) -> tuple[dict, float] | tuple[None, None]:
    """Find the EMA callback state and return ``(slowest_shadow, decay)``.

    Scans the checkpoint's ``callbacks`` for the double-EMA state (keyed by class
    name) and selects the largest-decay shadow — the published model.
    """
    for state in ckpt.get("callbacks", {}).values():
        if isinstance(state, dict) and "shadows" in state and "decays" in state:
            shadows = state["shadows"]
            if not shadows:
                return None, None
            idx = slowest_shadow_index(state["decays"])
            return shadows[idx], float(state["decays"][idx])
    return None, None


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
    prefer_ema: bool = False,
) -> str:
    """Convert a training ``.ckpt`` into a native per-component inference dir.

    Args:
        ckpt_path: a Lightning ``.ckpt`` written by ``ModelCheckpoint`` (full
            state — ``save_weights_only=False``). Loaded with
            ``weights_only=False``: only point this at training checkpoints you
            wrote.
        unet / vae / scheduler: fresh components built from the network config;
            the UNet's backbone weights are overwritten from the checkpoint, the
            VAE carries its (already-set) ``scaling_factor``.
        prefer_ema: bake the slowest EMA shadow when present; else (the default)
            the raw ``state_dict`` UNet weights. The default ``False`` aligns the
            published weights with what ``ModelCheckpoint`` selects — the raw-FID
            monitor (``val/fid_raw``) — so the exported "best" checkpoint is best
            for the weights that are actually published. Pass ``True`` for runs
            where the 0.9999 EMA has converged (warm-start / long horizon, as
            hope trains), publishing that instead.

    Returns:
        A short string naming which weights were baked (e.g. ``ema[decay=...]``).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    shadow, decay = _slowest_ema_shadow(ckpt) if prefer_ema else (None, None)
    if shadow is not None:
        # EMA shadow keys carry the single wrapper prefix; strip to backbone keys.
        backbone = {k[len(_WRAPPER_PREFIX):]: v for k, v in shadow.items() if k.startswith(_WRAPPER_PREFIX)}
        # strict=False: the shadow covers parameters (EMA tracks params); any
        # buffers (none on the GroupNorm MAISI backbone) keep construction values.
        _bake_backbone(unet, backbone, strict=False)
        source = f"ema[decay={decay}]"
    else:
        backbone = {
            k[len(_STATE_PREFIX):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith(_STATE_PREFIX)
        }
        _bake_backbone(unet, backbone, strict=True)
        source = "unet_state_dict"
    _log.info("Export baking %s as the inference UNet -> %s", source, output_dir)

    LatentFlowPipeline(unet, vae, scheduler).save_pretrained(output_dir)
    return source
