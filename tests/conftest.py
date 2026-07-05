"""Shared fixtures: tiny MONAI MAISI components on CPU (no GPU, no real data).

Tiny tensors and tiny/mock networks, so the four
components are exercised in isolation. The tiny config is verified to construct
and forward on CPU: a 2-level VAE (latent divisor 2) and a 2-level UNet (one
downsample, mid block at spatial 2 so GroupNorm never collapses to size 1).
"""

from __future__ import annotations

import pytest
import torch

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowModule,
    LatentFlowPipeline,
    PairedLatentFlowPipeline,
    UNet3DConditionModel,
)


@pytest.fixture
def vae() -> AutoencoderKL:
    torch.manual_seed(0)
    return AutoencoderKL(scaling_factor=0.5)


@pytest.fixture
def unet() -> UNet3DConditionModel:
    torch.manual_seed(0)
    return UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)


@pytest.fixture
def scheduler() -> FlowMatchHeunDiscreteScheduler:
    return FlowMatchHeunDiscreteScheduler()


@pytest.fixture
def pipeline(unet, vae, scheduler) -> LatentFlowPipeline:
    return LatentFlowPipeline(unet, vae, scheduler)


@pytest.fixture
def latent_module() -> LatentFlowModule:
    """A tiny trainable module for trainer / FID tests (lr sized for descent)."""
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    return LatentFlowModule(
        unet,
        FlowMatchHeunDiscreteScheduler(),
        lr=1e-2,
        lr_warmup_steps=0,
        num_train_examples=6,
        train_batch_size=2,
        n_epochs=2,
    )


# -- Paired JiT (src→tgt flow) fixtures --------------------------------------
# A paired UNet doubles the input channels: concat([z_t, x_src]) → 2·C_latent = 8.
# ``vae`` above is reused (one frozen VAE for both endpoints — ADR-0014).


@pytest.fixture
def paired_unet() -> UNet3DConditionModel:
    torch.manual_seed(0)
    return UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )


@pytest.fixture
def paired_scheduler() -> FlowMatchHeunDiscreteScheduler:
    return FlowMatchHeunDiscreteScheduler()


@pytest.fixture
def paired_pipeline(paired_unet, vae, paired_scheduler) -> PairedLatentFlowPipeline:
    return PairedLatentFlowPipeline(paired_unet, vae, paired_scheduler)
