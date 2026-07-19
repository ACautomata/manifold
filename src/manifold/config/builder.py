"""Build manifold components from a composed experiment config (ADR-0004).

The construction seam: a composed :class:`~omegaconf.DictConfig` carries plain-
kwarg network blocks (``autoencoder`` / ``diffusion_unet`` / ``scheduler``) and
these helpers instantiate the manifold wrappers directly from them — **not** via a
MONAI ``ConfigParser`` (a MONAI parser would instantiate MAISI directly and bypass
the wrappers' timestep-scaling / spacing×1e2 / ``scale_factor`` logic, ADR-0001).

Shared dims interpolate through ``${spatial_dims}`` / ``${image_channels}`` /
``${latent_channels}``; resolving a network sub-block does not resolve the env
``???`` paths (they are never referenced from the network blocks).
"""

from __future__ import annotations

from omegaconf import DictConfig, ListConfig, OmegaConf

from ..models.autoencoder_kl import AutoencoderKL
from ..models.controlnet_3d import ControlNet3DConditionModel
from ..models.unet_3d_condition import UNet3DConditionModel
from ..pipelines.latent_flow import LatentFlowPipeline
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler


def _block_kwargs(cfg: DictConfig, key: str) -> dict:
    """Resolve one network sub-block to a plain kwargs dict.

    ``resolve=True`` resolves ``${...}`` interpolations against the composed root
    without resolving the env ``???`` paths (those sit under different top-level
    keys the network block never references).
    """
    block = OmegaConf.to_container(getattr(cfg, key), resolve=True)
    if not isinstance(block, dict):
        raise ValueError(f"Config block {key!r} must be a mapping, got {type(block).__name__}.")
    return block


def build_vae(cfg: DictConfig) -> AutoencoderKL:
    """Instantiate the :class:`AutoencoderKL` from the ``autoencoder`` block."""
    return AutoencoderKL(**_block_kwargs(cfg, "autoencoder"))


def build_unet(cfg: DictConfig) -> UNet3DConditionModel:
    """Instantiate the :class:`UNet3DConditionModel` from the ``diffusion_unet`` block.

    The shipped ``config_network.yaml`` still carries ``paired_direction_offset``
    (the ControlNet's direction-MLP knob, ADR-0028), but the base UNet wrapper no
    longer accepts it (the paired branch was removed in #130 T8). Pop it so the
    shared block builds the base cleanly — ``build_controlnet`` forwards it.
    """
    kwargs = _block_kwargs(cfg, "diffusion_unet")
    kwargs.pop("paired_direction_offset", None)  # ControlNet-only (ADR-0028); not a base arg
    return UNet3DConditionModel(**kwargs)


def build_controlnet(cfg: DictConfig) -> ControlNet3DConditionModel:
    """Instantiate the :class:`ControlNet3DConditionModel` from the ``diffusion_unet`` block.

    The ControlNet clones the frozen base UNet's encoder (ADR-0026), so it shares
    the base's architecture knobs from the SAME ``diffusion_unet`` block. Two keys
    differ from the base: ``out_channels`` is base-only (the ControlNet emits
    residuals, not a denoised output conv — it has no ``out`` projection) and
    ``controlnet_cond_channels`` is the ``x_src`` control signal's width, which
    equals ``in_channels`` (a VAE latent of the same dimensionality as ``z_t``).
    """
    kwargs = _block_kwargs(cfg, "diffusion_unet")
    kwargs.pop("out_channels", None)  # base-only — the ControlNet has no output conv
    kwargs.setdefault("controlnet_cond_channels", kwargs["in_channels"])  # x_src latent width
    return ControlNet3DConditionModel(**kwargs)


def build_scheduler(cfg: DictConfig) -> FlowMatchHeunDiscreteScheduler:
    """Instantiate the :class:`FlowMatchHeunDiscreteScheduler` from the ``scheduler`` block."""
    return FlowMatchHeunDiscreteScheduler(**_block_kwargs(cfg, "scheduler"))


def build_pipeline(cfg: DictConfig) -> LatentFlowPipeline:
    """Build a live :class:`LatentFlowPipeline` (UNet + VAE + Scheduler) from *cfg*.

    The composed config builds the pipeline at launch; it never persists
    (component config stays JSON — ADR-0004).
    """
    return LatentFlowPipeline(build_unet(cfg), build_vae(cfg), build_scheduler(cfg))


def autoencoder_divisor(cfg: DictConfig) -> int:
    """VAE spatial downsampling factor: ``2 ** (num_downsample - 1)``.

    The MAISI autoencoder downsamples by 2× per stage after the first, so the
    latent volume is the image volume divided by ``2**(n-1)`` where ``n`` is the
    number of ``num_channels`` entries. The data stack pads volumes to a multiple
    of it (issue #16).
    """
    nc = OmegaConf.to_container(cfg.autoencoder.num_channels, resolve=True)
    num_downsample = max(1, len(nc) if isinstance(nc, (list, ListConfig)) else 1)
    return 2 ** (num_downsample - 1)
