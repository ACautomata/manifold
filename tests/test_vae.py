"""VAE round-trip + scaling-ownership tests (ADR-0003, Seam 3).

``AutoencoderKL`` owns ``scaling_factor``: ``encode`` returns a scaled latent,
``decode`` undoes the scaling internally. These tests pin the invariant directly
(decode undoes encode's scaling exactly) rather than reconstruction quality,
which would require a trained VAE.
"""

from __future__ import annotations

import pytest
import torch

# Tiny shapes consistent with the default 2-level VAE (latent divisor 2).
IMAGE_SHAPE = (1, 1, 8, 8, 8)
LATENT_SHAPE = (1, 4, 4, 4, 4)


def test_scaling_factor_is_a_buffer(vae):
    names = dict(vae.named_buffers())
    assert "scaling_factor" in names


def test_encode_returns_scaled_latent(vae):
    """encode(x) == encode_stage_2_inputs(x) * scaling_factor (same RNG seed)."""
    img = torch.randn(*IMAGE_SHAPE)
    torch.manual_seed(7)
    z_raw = vae.autoencoder.encode_stage_2_inputs(img)
    torch.manual_seed(7)
    z_scaled = vae.encode(img)
    assert torch.allclose(z_scaled, z_raw * vae.scaling_factor)


def test_decode_undoes_scaling_internally(vae):
    """decode(z * sf) == decode_stage_2_outputs(z): the scaling cancels."""
    z = torch.randn(*LATENT_SHAPE)
    decoded_scaled = vae.decode(z * vae.scaling_factor)
    decoded_raw = vae.autoencoder.decode_stage_2_outputs(z.to(vae._backbone_dtype))
    assert decoded_scaled.shape == decoded_raw.shape
    assert torch.allclose(decoded_scaled, decoded_raw, atol=1e-5)


def test_roundtrip_shape_dtype_finite(vae):
    img = torch.randn(*IMAGE_SHAPE)
    vol = vae.decode(vae.encode(img))
    assert vol.shape == img.shape
    assert vol.dtype == img.dtype
    assert torch.isfinite(vol).all()


def test_scaling_factor_round_trips_through_save_load(vae, tmp_path):
    """scaling_factor is a buffer, so it persists with the state dict."""
    vae.save_pretrained(str(tmp_path / "vae"))
    from manifold import AutoencoderKL

    reloaded = AutoencoderKL.from_pretrained(str(tmp_path / "vae"))
    assert torch.allclose(reloaded.scaling_factor, vae.scaling_factor)
    assert reloaded.scaling_factor.item() == pytest.approx(0.5)


@pytest.mark.parametrize("scaling_factor", [0.25, 1.0, 4.0])
def test_decode_undoes_scaling_for_any_factor(scaling_factor):
    """The round-trip is independent of the scaling_factor value (it cancels)."""
    from manifold import AutoencoderKL

    torch.manual_seed(0)
    vae = AutoencoderKL(scaling_factor=scaling_factor)
    z = torch.randn(*LATENT_SHAPE)
    decoded_scaled = vae.decode(z * scaling_factor)
    decoded_raw = vae.autoencoder.decode_stage_2_outputs(z.to(vae._backbone_dtype))
    assert torch.allclose(decoded_scaled, decoded_raw, atol=1e-5)
