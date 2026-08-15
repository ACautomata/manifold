"""Paired-fidelity metric tests (issue #227, ADR-0036).

The numerical seam: two ``[0,1]`` image volumes ``[B, C, D, H, W]`` (the generated
target vs the real target) → PSNR + SSIM scalars. The metric wraps MONAI
``PSNRMetric(max_val=1.0)`` / ``SSIMMetric(spatial_dims=3, data_range=1.0)`` — never
hand-rolled — so these tests pin the *external* scalar behaviour, not MONAI internals
(prior art: the numerical seam in ``test_fid.py``).

Volumes are tiny but ≥ 11 voxels per side because MONAI's SSIM default ``win_size=11``
must fit inside the spatial extent.
"""

from __future__ import annotations

import math

import pytest
import torch

from manifold.metrics import PairedFidelityMetrics, PairedFidelityScores


def _unit_volume(shape=(2, 1, 16, 16, 16), *, seed=0):
    """A deterministic structured ``[0,1]`` volume (smooth, so SSIM is non-degenerate)."""
    g = torch.Generator().manual_seed(seed)
    coarse = torch.rand(shape[0], shape[1], 4, 4, 4, generator=g)
    up = torch.nn.functional.interpolate(
        coarse, size=shape[2:], mode="trilinear", align_corners=False
    )
    return up.clamp(0.0, 1.0)


def test_identical_volumes_give_maximal_psnr_and_unit_ssim():
    """Identity is the fidelity ceiling: PSNR → inf (zero error), SSIM == 1."""
    metric = PairedFidelityMetrics()
    vol = _unit_volume()
    scores = metric(vol, vol.clone())
    assert isinstance(scores, PairedFidelityScores)
    assert math.isinf(scores.psnr), "identical volumes must give maximal (inf) PSNR"
    assert scores.ssim == pytest.approx(1.0, abs=1e-6)


def test_psnr_matches_reference_formula_for_known_mse():
    """PSNR is pinned to its public definition ``10·log10(max_val²/MSE)``.

    A constant offset on a constant volume gives an exactly-known MSE, so the
    expected PSNR comes from the formula (an independent source of truth), not from
    re-running MONAI. With ``data_range = max_val = 1.0`` and MSE = 0.01 the expected
    value is exactly ``10·log10(1/0.01) = 20`` dB — pinning ``data_range=1.0``.
    """
    real = torch.full((2, 1, 16, 16, 16), 0.5)
    generated = real + 0.1  # MSE == 0.01 exactly; stays inside [0,1]
    scores = PairedFidelityMetrics()(generated, real)
    expected_psnr = 10.0 * math.log10(1.0**2 / 0.01)
    assert scores.psnr == pytest.approx(expected_psnr, abs=1e-3)


def test_perturbation_drops_both_metrics():
    """A perturbation lowers fidelity: finite PSNR, SSIM strictly below 1."""
    metric = PairedFidelityMetrics()
    real = _unit_volume(seed=0)
    generated = (real + 0.1 * _unit_volume(seed=1)).clamp(0.0, 1.0)
    scores = metric(generated, real)
    assert math.isfinite(scores.psnr)
    assert 0.0 < scores.ssim < 1.0


def test_fidelity_decreases_monotonically_with_noise():
    """More noise → worse fidelity: PSNR and SSIM both decrease monotonically."""
    metric = PairedFidelityMetrics()
    real = _unit_volume(seed=0)
    noise = _unit_volume(seed=2) - 0.5
    small = (real + 0.05 * noise).clamp(0.0, 1.0)
    large = (real + 0.3 * noise).clamp(0.0, 1.0)

    clean = metric(real, real.clone())
    small_scores = metric(small, real)
    large_scores = metric(large, real)

    assert clean.psnr > small_scores.psnr > large_scores.psnr
    assert clean.ssim > small_scores.ssim > large_scores.ssim


def test_spatial_dims_3_operates_on_5d_volumes():
    """The metric consumes ``[B, C, D, H, W]`` and scores across all three spatial
    dims — a 2D SSIM would not accept a 5D volume. Identity gives the 3D ceiling."""
    metric = PairedFidelityMetrics()
    vol = _unit_volume(shape=(1, 1, 16, 16, 16))
    scores = metric(vol, vol.clone())
    assert scores.ssim == pytest.approx(1.0, abs=1e-6)


def test_mismatched_shapes_raise():
    """generated and real must be the same shape — a pairing bug must fail loudly
    (the wrapper raises a clear ValueError before MONAI sees the mismatched tensors)."""
    metric = PairedFidelityMetrics()
    a = _unit_volume(shape=(2, 1, 16, 16, 16))
    b = _unit_volume(shape=(2, 1, 16, 16, 8))
    with pytest.raises(ValueError, match="matching shapes"):
        metric(a, b)
