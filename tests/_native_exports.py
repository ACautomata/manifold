"""Shared native-export helpers for tests that need a minimal on-disk export.

``save_controlnet_export`` writes a minimal ControlNet native export (the
per-component ``ControlNetLatentFlowPipeline`` layout: base UNet + ControlNet + VAE +
scheduler) — the fixture several GRPO / native-artifact-detection tests drive
``load_frozen_controlnet_generator`` / ``_detect_controlnet_export`` against. Extracted
out of the deleted paired-reward test suite (ADR-0034) so those tests survive its
deletion.
"""

from __future__ import annotations

import torch
from torch import nn

from manifold import (
    AutoencoderKL,
    ControlNet3DConditionModel,
    ControlNetLatentFlowPipeline,
    FlowMatchHeunDiscreteScheduler,
    UNet3DConditionModel,
)


def save_controlnet_export(native_dir) -> None:
    """Write a minimal ControlNet native export under ``native_dir``.

    The export uses the ``ControlNetLatentFlowPipeline`` per-component layout (a
    ``controlnet/`` subdirectory) that ``load_frozen_controlnet_generator`` and
    ``_detect_controlnet_export`` key on. The base ``out`` conv is re-initialized so the
    ControlNet's residual effect is visible (a zero-init ``out`` would mask it).
    """
    torch.manual_seed(0)
    base = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in base.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    controlnet.load_base_encoder_weights(base)
    ControlNetLatentFlowPipeline(
        base, controlnet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    ).save_pretrained(str(native_dir))
