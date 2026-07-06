"""Paired JiT persistence tests (Seam 6, issue #69).

``PairedLatentFlowPipeline.save_pretrained`` / ``from_pretrained`` round-trip the
native per-component directory format including the doubled-``in_channels`` UNet
config (``2·C_latent``), and a reloaded pipeline infers identically to the
original — the same src latent + labels produces the same decoded volume.
"""

from __future__ import annotations

import torch

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    PairedLatentFlowPipeline,
    UNet3DConditionModel,
)

C_LATENT = 4
LATENT_SHAPE = (1, C_LATENT, 4, 4, 4)


def _pipeline() -> PairedLatentFlowPipeline:
    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    return PairedLatentFlowPipeline(unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler())


def test_save_from_pretrained_round_trips_doubled_in_channels(tmp_path):
    """The reloaded UNet keeps ``in_channels = 2·C_latent`` (ADR-0014 concat input)."""
    pipe = _pipeline()
    pipe.save_pretrained(str(tmp_path / "paired"))

    reloaded = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired"))
    assert reloaded.unet.config["in_channels"] == 2 * C_LATENT
    assert reloaded.unet.config["out_channels"] == C_LATENT


def test_reloaded_pipeline_infers_identically(tmp_path):
    """A reloaded pipeline translates the same src latent identically (state_dict exact)."""
    pipe = _pipeline()
    src = torch.randn(LATENT_SHAPE)
    args = dict(spacing=[1.0, 1.0, 1.0], src_label=0, tgt_label=1, num_inference_steps=3)
    before = pipe(src, **args)

    pipe.save_pretrained(str(tmp_path / "paired"))
    reloaded = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired"))
    after = reloaded(src, **args)
    # torch.save/load state_dict is exact → bit-identical inference.
    assert torch.equal(before, after)


def test_reloaded_sample_latent_matches(tmp_path):
    """The latent rollout (pre-decode) also round-trips identically."""
    pipe = _pipeline()
    src = torch.randn(LATENT_SHAPE)
    args = dict(spacing=[1.0, 1.0, 1.0], src_label=0, tgt_label=1, num_inference_steps=4)
    before = pipe.sample_latent(src, **args)

    pipe.save_pretrained(str(tmp_path / "paired"))
    reloaded = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired"))
    after = reloaded.sample_latent(src, **args)
    assert torch.equal(before, after)


def test_from_pretrained_rejects_non_pipeline_directory(tmp_path):
    """A directory without model_index.json is rejected with a clear error."""
    import pytest

    with pytest.raises(FileNotFoundError, match="model_index"):
        PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "nonexistent"))
