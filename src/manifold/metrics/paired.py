"""Paired-fidelity metric: 3D PSNR + 3D SSIM of a generated target vs the real target.

The full-reference fidelity metric of the ControlNet translation policy (ADR-0036,
issue #227) — the paired counterpart to **Unbiased FID**, which is reference-free and
fits the unconditional JiT. Where FID asks "is this volume realistic", this metric
asks "does the generated ``tgt`` match the real ``tgt``" — exactly the fidelity signal
the realism Reward Model is blind to by construction (ADR-0034).

Both volumes are VAE-decoded and then per-sample min-max normalized to ``[0,1]`` (the
Pipeline's published-output convention), so ``data_range = 1.0``. MONAI
``PSNRMetric(max_val=1.0)`` / ``SSIMMetric(spatial_dims=3, data_range=1.0)`` are
*reused*, never hand-rolled — matching the RewardModel-wraps-MONAI ethos. PSNR is
``+inf`` on identical volumes (zero error); that ceiling is surfaced honestly rather
than clamped to a finite value.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from monai.metrics import PSNRMetric, SSIMMetric
from torch import Tensor


class PairedFidelityScores(NamedTuple):
    """The two full-reference scalars: ``psnr`` (dB, ``+inf`` when identical) and
    ``ssim`` (in ``[-1, 1]``, ``1.0`` when identical)."""

    psnr: float
    ssim: float


class PairedFidelityMetrics:
    """3D PSNR + 3D SSIM between two ``[0,1]`` volumes ``[B, C, D, H, W]``.

    Composes MONAI's ``PSNRMetric`` / ``SSIMMetric`` (composition, not inheritance) and
    reduces their per-sample ``[B, 1]`` outputs to batch-mean scalars. The batch mean
    is the reported fidelity over the paired set.
    """

    def __init__(self, *, data_range: float = 1.0, spatial_dims: int = 3):
        """Args:
            data_range: the dynamic range of the input volumes (``1.0`` for the
                Pipeline's min-max-to-unit output convention). Sets PSNR ``max_val``
                and SSIM ``data_range``.
            spatial_dims: spatial rank of the volumes (3 for volumetric ``tgt``).
        """
        self._psnr = PSNRMetric(max_val=data_range)
        self._ssim = SSIMMetric(spatial_dims=spatial_dims, data_range=data_range)

    def __call__(self, generated: Tensor, real: Tensor) -> PairedFidelityScores:
        """Score *generated* against *real* (both ``[B, C, D, H, W]``, ``[0,1]``).

        Returns the batch-mean PSNR (``+inf`` if a pair is identical) and SSIM. The
        two volumes must share a shape; MONAI raises on a mismatch.
        """
        if generated.shape != real.shape:
            raise ValueError(
                f"Paired fidelity needs matching shapes, got generated "
                f"{tuple(generated.shape)} vs real {tuple(real.shape)}."
            )
        with torch.no_grad():
            psnr = float(self._psnr(generated, real).mean())
            ssim = float(self._ssim(generated, real).mean())
        return PairedFidelityScores(psnr=psnr, ssim=ssim)
