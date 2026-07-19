"""ControlNet pipeline inference smoke + persistence (issue #141 / ADR-0027).

``ControlNetLatentFlowPipeline`` built from tiny components turns noise + a source
control signal into a decoded ``[B, C, D, H, W]`` target volume; the native
per-component ``save_pretrained`` → ``from_pretrained`` round-trip reproduces the
rollout bit-for-bit; and the VAE decode's ``norm_float16`` disable (the ControlNet
pipeline's decode-correctness mechanism, mirroring ``LatentFlowPipeline``'s
autocast decode — issue #18) is exercised. Sibling of ``test_pipeline_inference.py``
+ ``test_persistence.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from manifold import (
    AutoencoderKL,
    ControlNet3DConditionModel,
    ControlNetLatentFlowPipeline,
    FlowMatchHeunDiscreteScheduler,
    UNet3DConditionModel,
)

# Latent [1,4,8,8,4] decodes to image [1,1,16,16,8] (the default 2-level VAE is
# spatial-divisor 2). Matches the tiny ControlNet module-training fixture shapes.
LATENT_SHAPE = (1, 4, 8, 8, 4)
IMAGE_SHAPE = (1, 1, 16, 16, 8)


def _frozen_base() -> UNet3DConditionModel:
    """A tiny base UNet with the zero-init output conv re-initialized.

    MONAI MAISI zero-initializes the final output projection, so at init the base
    output is identically zero and the ControlNet's residual-injection effect on the
    output is masked. Re-initializing the all-zero ``out`` params (emulating a
    warm-started base) lets the full base-output→ControlNet path run end-to-end.
    (Mirrors ``tests/test_controlnet_module_training._frozen_base``.)
    """
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in unet.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return unet


def _controlnet(base: UNet3DConditionModel) -> ControlNet3DConditionModel:
    torch.manual_seed(1)
    controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    controlnet.load_base_encoder_weights(base)
    return controlnet


def _pipeline() -> ControlNetLatentFlowPipeline:
    base = _frozen_base()
    vae = AutoencoderKL(scaling_factor=0.5)
    return ControlNetLatentFlowPipeline(base, _controlnet(base), vae, FlowMatchHeunDiscreteScheduler())


# -- end-to-end translation --------------------------------------------------


def test_pipeline_returns_decoded_volume():
    """noise + src control -> decoded target volume of the right shape/dtype/range."""
    pipe = _pipeline()
    noise = torch.randn(*LATENT_SHAPE)
    src = torch.randn(*LATENT_SHAPE)
    vol = pipe(noise, src, spacing=[1.0, 1.0, 1.0], src_label=1, tgt_label=2, num_inference_steps=4)
    assert vol.shape == IMAGE_SHAPE
    assert vol.dtype == torch.float32
    assert torch.isfinite(vol).all()


def test_pipeline_output_is_minmax_normalized_to_unit():
    """The published output contract: each volume normalized to [0, 1]."""
    pipe = _pipeline()
    vol = pipe(
        torch.randn(*LATENT_SHAPE), torch.randn(*LATENT_SHAPE),
        spacing=[1.0, 1.0, 1.0], src_label=1, tgt_label=2, num_inference_steps=3,
    )
    assert vol.min().item() >= -1e-5
    assert vol.max().item() <= 1.0 + 1e-5


def test_pipeline_is_deterministic_given_noise():
    """The rollout is deterministic given the (noise, src) inputs — same in, same out."""
    pipe = _pipeline()
    noise = torch.randn(*LATENT_SHAPE)
    src = torch.randn(*LATENT_SHAPE)
    a = pipe(noise, src, spacing=[1.0, 1.0, 1.0], src_label=1, tgt_label=2, num_inference_steps=3)
    b = pipe(noise, src, spacing=[1.0, 1.0, 1.0], src_label=1, tgt_label=2, num_inference_steps=3)
    assert torch.allclose(a, b)


def test_pipeline_holds_no_scale_factor():
    """The pipeline owns no scale_factor; the VAE does (ADR-0003)."""
    pipe = _pipeline()
    assert not hasattr(pipe, "scale_factor")
    assert set(pipe.__dict__) >= {"unet", "controlnet", "vae", "scheduler"}
    assert hasattr(pipe.vae, "scaling_factor")


# -- native per-component round-trip ----------------------------------------


def test_save_load_round_trip_reproduces_rollout(tmp_path):
    """save_pretrained -> from_pretrained reproduces the rollout bit-for-bit."""
    pipe = _pipeline()
    pipe.save_pretrained(str(tmp_path / "pipe"))
    loaded = ControlNetLatentFlowPipeline.from_pretrained(str(tmp_path / "pipe"))

    torch.manual_seed(7)
    noise = torch.randn(*LATENT_SHAPE)
    src = torch.randn(*LATENT_SHAPE)
    a = pipe.sample_latent(noise, src, [1.0, 1.0, 1.0], 1, 2, num_inference_steps=4)
    b = loaded.sample_latent(noise, src, [1.0, 1.0, 1.0], 1, 2, num_inference_steps=4)
    assert torch.equal(a, b)


def test_components_keep_their_class_after_load(tmp_path):
    pipe = _pipeline()
    pipe.save_pretrained(str(tmp_path / "pipe"))
    loaded = ControlNetLatentFlowPipeline.from_pretrained(str(tmp_path / "pipe"))
    assert type(loaded.unet) is UNet3DConditionModel
    assert type(loaded.controlnet) is ControlNet3DConditionModel
    assert type(loaded.vae) is AutoencoderKL
    assert type(loaded.scheduler) is FlowMatchHeunDiscreteScheduler


def test_scaling_factor_round_trips(tmp_path):
    pipe = _pipeline()
    pipe.save_pretrained(str(tmp_path / "pipe"))
    loaded = ControlNetLatentFlowPipeline.from_pretrained(str(tmp_path / "pipe"))
    assert loaded.vae.scaling_factor.item() == pytest.approx(0.5)


# -- decode correctness: norm_float16 disable (the ControlNet autocast analogue) --


def test_pipeline_decode_disables_norm_float16_idempotently():
    """The ControlNet decode disables the migrated VAE's ``norm_float16`` once.

    The migrated VAE's MaisiGroupNorm3D carries ``norm_float16`` (casts its output to
    half unconditionally), so a downstream float32 conv raises a Half/float mismatch.
    ``LatentFlowPipeline`` solves this with a cuda ``autocast`` decode region
    (``test_persistence.test_pipeline_runs_rollout_under_autocast``); the ControlNet
    pipeline instead runs a float32 decode with ``norm_float16`` disabled once
    (idempotent) — its decode-correctness mechanism, the ADR-0027 analogue.
    """
    pipe = _pipeline()
    norm_modules = [m for m in pipe.vae.modules() if hasattr(m, "norm_float16")]
    assert norm_modules, "the tiny VAE fixture has no norm_float16 modules (fixture drift)"
    # The first call disables norm_float16 on every such module (idempotent flag set).
    pipe(
        torch.randn(*LATENT_SHAPE), torch.randn(*LATENT_SHAPE),
        spacing=[1.0, 1.0, 1.0], src_label=1, tgt_label=2, num_inference_steps=2,
    )
    assert all(not m.norm_float16 for m in norm_modules)
    assert pipe._norm16_disabled is True
