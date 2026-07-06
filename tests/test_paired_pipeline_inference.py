"""Paired JiT Pipeline tests (Seam 2, issue #66).

``PairedLatentFlowPipeline`` turns a source latent + contrast labels into a
decoded ``[B, C, D, H, W]`` volume with the expected dtype and a finite range, and
is deterministic given ``src_latent`` (the rollout has no stochastic input —
ADR-0013). The highest-level behavioral test fits a few steps on one synthetic
pair, then reconstructs ``tgt`` from ``src`` within tolerance — exercising
transport + sampler + concat conditioning + decode together. An oracle-UNet
variant pins the transport/sampler composition exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    PairedLatentFlowModule,
    PairedLatentFlowPipeline,
    UNet3DConditionModel,
)

# Tiny shapes consistent with the default 2-level VAE (latent divisor 2):
# latent [1,4,4,4,4] decodes to image [1,1,8,8,8].
C_LATENT = 4
LATENT_SHAPE = (1, C_LATENT, 4, 4, 4)
IMAGE_SHAPE = (1, 1, 8, 8, 8)


def test_pipeline_returns_decoded_volume(paired_pipeline):
    vol = paired_pipeline(
        torch.randn(LATENT_SHAPE),
        spacing=[1.0, 1.0, 1.0],
        src_label=0,
        tgt_label=1,
        num_inference_steps=4,
    )
    assert vol.shape == IMAGE_SHAPE
    assert vol.dtype == torch.float32
    assert torch.isfinite(vol).all()


def test_pipeline_is_deterministic_given_src(paired_pipeline):
    """Deterministic given x_src (no stochastic input) — same src → identical tgt."""
    src = torch.randn(LATENT_SHAPE)
    args = dict(spacing=[1.5, 1.5, 2.0], src_label=0, tgt_label=2, num_inference_steps=3)
    a = paired_pipeline(src, **args)
    b = paired_pipeline(src, **args)
    assert torch.allclose(a, b)


def test_pipeline_sample_latent_shape(paired_pipeline):
    latent = paired_pipeline.sample_latent(
        torch.randn(LATENT_SHAPE),
        spacing=[1.0, 1.0, 1.0],
        src_label=0,
        tgt_label=1,
        num_inference_steps=4,
    )
    assert latent.shape == LATENT_SHAPE


def test_pipeline_holds_no_scale_factor(paired_pipeline):
    """The pipeline owns no scale_factor; the VAE does (ADR-0003)."""
    assert not hasattr(paired_pipeline, "scale_factor")
    assert set(paired_pipeline.__dict__) >= {"unet", "vae", "scheduler"}
    assert hasattr(paired_pipeline.vae, "scaling_factor")


class _TargetUNet(nn.Module):
    """Oracle UNet that always predicts ``x_tgt`` exactly (ignores its input).

    Pins the transport + sampler composition: with the model returning ``x_tgt``
    at every point, the Heun rollout integrates ``z`` from ``x_src`` to ``x_tgt``
    exactly (the discrete Heun is exact for this linear velocity field; the final
    Euler step cancels the ``1 − t`` denominator). Carries a dummy parameter so
    ``next(unet.parameters())`` resolves in the shared sampler primitive.
    """

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, sample, timestep, spacing, class_labels=None, context=None, *,
                class_labels_src=None, class_labels_tgt=None):
        return self.target


def test_oracle_rollout_lands_on_tgt_exactly():
    """With an oracle UNet (predicts x_tgt at every point), the rollout reconstructs
    x_tgt from x_src to float precision — proving transport + sampler compose."""
    torch.manual_seed(0)
    x_src = torch.randn(LATENT_SHAPE)
    x_tgt = torch.randn(LATENT_SHAPE)
    pipeline = PairedLatentFlowPipeline(
        _TargetUNet(x_tgt), AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    )
    latent = pipeline.sample_latent(
        x_src, spacing=[1.0, 1.0, 1.0], src_label=0, tgt_label=1, num_inference_steps=4
    )
    # The discrete Heun is algebraically exact here (see _TargetUNet docstring);
    # float32 accumulation over 4 steps stays well within this tolerance.
    assert torch.allclose(latent, x_tgt, atol=1e-5), (
        f"max abs diff {float((latent - x_tgt).abs().max())}"
    )


def test_reconstruct_tgt_from_src_after_fit():
    """Fit a few steps on one synthetic pair, then reconstruct tgt from src.

    A real (trainable) paired UNet is shared between the Module and the Pipeline.
    Training reduces the loss substantially; the rollout's output then lands close
    to ``x_tgt`` (the model has memorized the single pair, approximating the
    oracle behavior proven separately above).
    """
    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    # Re-init MAISI's zero-init output conv so the full UNet is trainable.
    for p in unet.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)

    scheduler = FlowMatchHeunDiscreteScheduler()
    module = PairedLatentFlowModule(unet, scheduler, lr=2e-2, lr_warmup_steps=0)
    pipeline = PairedLatentFlowPipeline(unet, AutoencoderKL(scaling_factor=0.5), scheduler)

    torch.manual_seed(1)
    x_src = torch.randn(LATENT_SHAPE)
    x_tgt = torch.randn(LATENT_SHAPE)
    batch = {
        "src_latent": x_src,
        "tgt_latent": x_tgt,
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }

    optimizer = module.configure_optimizers()["optimizer"]
    loss0 = module(batch, "fit")["loss"].item()
    for _ in range(200):
        optimizer.zero_grad()
        module(batch, "fit")["loss"].backward()
        optimizer.step()
    loss1 = module(batch, "fit")["loss"].item()
    assert loss1 < loss0 / 10.0  # strong fit on the single pair

    latent = pipeline.sample_latent(
        x_src, spacing=[1.0, 1.0, 1.0], src_label=0, tgt_label=1, num_inference_steps=8
    )
    # Relative L2 closeness to x_tgt (loose: the tiny UNet approximates the oracle).
    rel_l2 = float((latent - x_tgt).norm() / x_tgt.norm())
    assert rel_l2 < 0.25, f"relative L2 {rel_l2:.4f}, loss0={loss0:.4f} loss1={loss1:.4f}"
