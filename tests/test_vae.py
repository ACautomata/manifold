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


def test_encode_raw_returns_unscaled_latent(vae):
    """encode_raw is the unscaled affordance; encode is encode_raw * scaling_factor.

    The data stack warms its cache of unscaled latents via encode_raw and
    estimates scaling_factor over it (ADR-0003 addendum, issue #16); encode keeps
    the public scaled-latent contract.
    """
    img = torch.randn(*IMAGE_SHAPE)
    torch.manual_seed(7)
    z_raw = vae.encode_raw(img)
    torch.manual_seed(7)
    z_stage2 = vae.autoencoder.encode_stage_2_inputs(img)
    assert torch.allclose(z_raw, z_stage2)  # no scaling applied
    # encode composes from encode_raw (the scale-on-read contract). encode and
    # encode_raw each draw a fresh reparameterization eps, so reset the seed
    # before each to compare identical draws (mirrors test_encode_returns_scaled_latent).
    torch.manual_seed(7)
    encoded = vae.encode(img)
    torch.manual_seed(7)
    assert torch.allclose(encoded, vae.encode_raw(img) * vae.scaling_factor)


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"norm_float16": True, "num_splits": 4, "dim_split": 1},
        {"norm_float16": False, "num_splits": 1, "dim_split": 0},
        {"norm_float16": False, "num_splits": 2, "dim_split": 1},
        {"with_encoder_nonlocal_attn": False, "include_fc": False, "use_convtranspose": False},
    ],
)
def test_widened_knobs_accepted_and_captured(kwargs):
    """The widened VAE construction surface is accepted and round-trips via config.

    These are GPU / memory knobs (``norm_float16`` runs the backbone in half;
    ``num_splits`` / ``dim_split`` chunk encode/decode) whose forward behaviour
    cannot be exercised on tiny CPU tensors, so the assertion is construction +
    config capture — that the construction kwargs load without a
    TypeError and persist identically (mirrors the scale-cancellation parametrize
    varying a knob; defaults preserved so the tiny fixtures stay green).
    """
    from manifold import AutoencoderKL

    vae = AutoencoderKL(**kwargs)
    for key, value in kwargs.items():
        assert vae.config[key] == value
