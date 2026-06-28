"""Pipeline inference smoke test (Seam 1, inference half — issue #4).

``LatentFlowPipeline`` built from directly-constructed tiny components turns pure
noise into a decoded ``[B, C, D, H, W]`` volume with the expected dtype and a
finite range. (The migration half of Seam 1 — converter → from_pretrained →
pipeline — is covered in ``test_persistence.py`` for issue #7, and CFG in
``test_cfg.py`` for issue #5.)
"""

from __future__ import annotations

import torch

# Tiny shapes consistent with the default 2-level VAE (latent divisor 2):
# latent [1,4,4,4,4] decodes to image [1,1,8,8,8].
LATENT_SHAPE = (1, 4, 4, 4, 4)
IMAGE_SHAPE = (1, 1, 8, 8, 8)


def test_pipeline_returns_decoded_volume(pipeline):
    vol = pipeline(
        LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=4,
        generator=torch.Generator().manual_seed(0),
    )
    # Latent [1,4,4,4,4] decodes to image [1,1,8,8,8] (VAE divisor 2).
    assert vol.shape == IMAGE_SHAPE
    assert vol.dtype == torch.float32
    assert torch.isfinite(vol).all()


def test_pipeline_is_deterministic_with_generator(pipeline):
    args = dict(
        target_shape=LATENT_SHAPE,
        spacing=[1.5, 1.5, 2.0],
        modality=0,
        num_inference_steps=3,
    )
    a = pipeline(generator=torch.Generator().manual_seed(42), **args)
    b = pipeline(generator=torch.Generator().manual_seed(42), **args)
    assert torch.allclose(a, b)


def test_pipeline_different_seed_differs(pipeline):
    args = dict(
        target_shape=LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=1,
        num_inference_steps=3,
    )
    a = pipeline(generator=torch.Generator().manual_seed(1), **args)
    b = pipeline(generator=torch.Generator().manual_seed(2), **args)
    assert not torch.allclose(a, b)


def test_pipeline_holds_no_scale_factor(pipeline):
    """The pipeline owns no scale_factor; the VAE does (ADR-0003)."""
    assert not hasattr(pipeline, "scale_factor")
    # Only the three components — the scale lives inside the VAE, not here.
    assert set(pipeline.__dict__) >= {"unet", "vae", "scheduler"}
    assert hasattr(pipeline.vae, "scaling_factor")
