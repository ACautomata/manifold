"""Denoising-interval classifier-free guidance (issue #5).

The scheduler's Heun math is untouched; CFG wraps the UNet calls inside the
pipeline. ``guidance_scale == 1`` must reproduce the no-CFG path exactly, a
degenerate interval covering no timestep must too, and ``guidance_scale > 1``
with an active interval must differ.

These tests use a tiny **mock** UNet whose output demonstrably depends on
``class_labels`` (per the PRD's "tiny/mock networks" test style). The real MONAI
MAISI UNet at tiny config attenuates the per-channel conditioning bias through
its GroupNorms to ~0, so it cannot exercise CFG; the mock isolates the *pipeline*
combination logic, which is what this slice changes. The real UNet is exercised
end-to-end in ``test_pipeline_inference.py`` / ``test_persistence.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowPipeline,
)

LATENT_SHAPE = (1, 4, 4, 4, 4)
SEED = 1234


class _CFGMockUNet(nn.Module):
    """Tiny UNet stand-in whose output depends on ``class_labels``.

    Mimics :class:`manifold.UNet3DConditionModel.forward`'s signature so the
    pipeline calls it unchanged. The per-channel class bias is large enough that
    conditional vs unconditional outputs clearly differ.
    """

    def __init__(self, channels: int = 4, num_classes: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_classes, channels)
        nn.init.normal_(self.embed.weight, std=2.0)
        self.channels = channels

    def forward(self, sample, timestep, spacing, class_labels=None, context=None):
        out = sample + 0.1 * float(timestep)
        if class_labels is not None:
            bias = self.embed(class_labels)  # [B, channels]
            out = out + bias.view(bias.shape[0], -1, 1, 1, 1)
        return out


@pytest.fixture
def cfg_pipeline():
    torch.manual_seed(0)
    unet = _CFGMockUNet()
    vae = AutoencoderKL(scaling_factor=0.5)
    return LatentFlowPipeline(unet, vae, FlowMatchHeunDiscreteScheduler())


def _run(pipeline, **over):
    kwargs = dict(
        target_shape=LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=4,
    )
    kwargs.update(over)
    return pipeline(generator=torch.Generator().manual_seed(SEED), **kwargs)


def test_guidance_one_matches_no_cfg(cfg_pipeline):
    """guidance_scale=1.0 is bit-identical to the no-CFG path (no uncond eval)."""
    no_cfg = _run(cfg_pipeline, guidance_scale=1.0, cfg_interval=None)
    with_interval = _run(cfg_pipeline, guidance_scale=1.0, cfg_interval=(0.0, 1.0))
    assert torch.allclose(no_cfg, with_interval)


def test_guidance_above_one_differs(cfg_pipeline):
    """guidance_scale > 1 with an active interval changes the output."""
    base = _run(cfg_pipeline, guidance_scale=1.0)
    guided = _run(cfg_pipeline, guidance_scale=3.0, cfg_interval=(0.0, 1.0))
    assert not torch.allclose(base, guided)


def test_degenerate_interval_matches_no_cfg(cfg_pipeline):
    """An interval covering no timestep reproduces the no-CFG output exactly."""
    no_cfg = _run(cfg_pipeline, guidance_scale=3.0, cfg_interval=None)
    degenerate = _run(cfg_pipeline, guidance_scale=3.0, cfg_interval=(1.0, 1.0))
    assert torch.allclose(no_cfg, degenerate)


def test_in_interval_step_costs_four_evals():
    """With CFG active at both Heun points, a step costs 4 UNet evals.

    For ``n`` steps with interval ``(0, 1)`` (active at every node except the
    pure-noise start ``t = 0``, which the strict ``low < t`` excludes — matching
    hope): each of the ``2n − 2`` non-start eval points costs 2 calls (cond +
    uncond), the ``t = 0`` point costs 1, for ``4n − 3`` total. A single-call
    scheme would cost ``2n − 1``; this rules that out.
    """
    unet = _CFGMockUNet()
    counts = {"n": 0}
    real_forward = unet.forward

    def counting_forward(*args, **kwargs):
        counts["n"] += 1
        return real_forward(*args, **kwargs)

    unet.forward = counting_forward  # type: ignore[method-assign]
    vae = AutoencoderKL(scaling_factor=0.5)
    pipe = LatentFlowPipeline(unet, vae, FlowMatchHeunDiscreteScheduler())
    n = 3
    pipe(
        LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=n,
        guidance_scale=3.0,
        cfg_interval=(0.0, 1.0),
        generator=torch.Generator().manual_seed(SEED),
    )
    assert counts["n"] == 4 * n - 3
