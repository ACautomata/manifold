"""Unbiased 2.5D Fréchet distance + RadImageNet feature extraction (issue #27).

Ports only the unbiased core of hope's metric module:

- :func:`frechet_distance_unbiased` — the small-sample-bias-corrected Fréchet
  distance. The plug-in (biased) FID over-estimates on small ``N`` because the
  empirical squared mean-gap ``‖μ̂₁ − μ̂₂‖²`` has expectation
  ``‖μ₁ − μ₂‖² + Tr(Σ₁)/n₁ + Tr(Σ₂)/n₂``; we subtract that ``Tr(Σ)/n`` mean-
  term, use the unbiased ``1/(N−1)`` covariance, and ridge-stabilise the
  covariance before the matrix square root.
- :func:`get_features_2p5d` — the 2.5D feature extraction (three orthogonal
  XY/YZ/ZX planes, stacked per volume).
- :func:`make_feature_network` — the RadImageNet ResNet50 factory (via
  ``torch.hub``; gauss must pre-populate ``$TORCH_HOME`` for offline use).

The InceptionV3 / SqueezeNet backbones, the bootstrap CI, and the standalone
2.5D orchestrator are **not** ported (out of scope). ``scipy`` and
``torchvision`` are required dependencies.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch import nn

try:
    from scipy.linalg import sqrtm as _scipy_sqrtm
except ImportError as _e:  # pragma: no cover — scipy is a required dep
    raise ImportError("manifold.metrics.fid requires scipy (pip install scipy)") from _e


def _cov_unbiased(x: Tensor) -> Tensor:
    """Unbiased sample covariance ``1/(N−1)·Σ(x−μ)(x−μ)ᵀ`` of an ``[N, D]`` matrix."""
    n = x.shape[0]
    if n < 2:
        raise ValueError(f"Need ≥2 samples for an unbiased covariance, got {n}.")
    mu = x.mean(dim=0, keepdim=True)
    centered = x - mu
    return centered.t() @ centered / (n - 1)


def _tr(matrix: Tensor) -> Tensor:
    """Trace of a square matrix (a scalar tensor)."""
    return matrix.diagonal(dim1=-2, dim2=-1).sum(dim=-1)


def _principal_sqrtm(matrix: Tensor) -> Tensor:
    """Principal matrix square root (real part of ``scipy.linalg.sqrtm``).

    *matrix* is symmetric PSD (a covariance product, ridge-stabilised), so the
    principal root is real; the imaginary residual from the numerical ``sqrtm``
    is dropped.
    """
    arr = matrix.detach().cpu().numpy()
    root = _scipy_sqrtm(arr)
    return torch.as_tensor(root.real, dtype=matrix.dtype, device=matrix.device)


def frechet_distance_unbiased(
    gen: Tensor, real: Tensor, *, ridge: float = 1e-6
) -> Tensor:
    """Small-sample-**unbiased** Fréchet distance between two feature sets.

    Args:
        gen: synthetic features ``[N1, D]``.
        real: real features ``[N2, D]``.
        ridge: covariance ridge (``Σ += ridge·I``) added before the matrix
            square root for numerical stability.

    Returns:
        A scalar tensor — the unbiased Fréchet distance. Subtracts the
        ``Tr(Σ̂)/n`` mean-term bias, so on a small validation set it is not
        dominated by the plug-in estimator's upward bias.
    """
    if gen.shape[1] != real.shape[1]:
        raise ValueError(
            f"Feature dim mismatch: gen {gen.shape[1]} vs real {real.shape[1]}."
        )
    n1, n2 = gen.shape[0], real.shape[0]
    mu_g = gen.mean(dim=0)
    mu_r = real.mean(dim=0)
    sigma_g = _cov_unbiased(gen)
    sigma_r = _cov_unbiased(real)

    # Unbiased mean term: ‖μ̂_g − μ̂_r‖² − Tr(Σ_g)/n1 − Tr(Σ_r)/n2.
    mean_term = (mu_g - mu_r).pow(2).sum() - _tr(sigma_g) / n1 - _tr(sigma_r) / n2

    eye = torch.eye(sigma_g.shape[0], dtype=sigma_g.dtype, device=sigma_g.device) * ridge
    cov_prod = (sigma_g + eye) @ (sigma_r + eye)
    cov_term = _tr(sigma_g) + _tr(sigma_r) - 2.0 * _tr(_principal_sqrtm(cov_prod))
    return (mean_term + cov_term).clamp(min=0.0)


def get_features_2p5d(
    volumes: Tensor,
    feature_net: Callable[[Tensor], Tensor],
    *,
    center_slices_ratio: float = 0.5,
) -> list[Tensor]:
    """Extract 2.5D features over the XY / YZ / ZX planes (stacked per volume).

    For each volume ``[B, C, D, H, W]`` and each of the three orthogonal axes,
    take ``max(1, round(center_slices_ratio·axis_len))`` center slices, run each
    plane (a 2D image ``[B, C, h, w]``) through *feature_net*, and collect the
    feature vectors per plane. Returns three ``[M_axis, D_feat]`` tensors — one
    per plane — so the caller can compute a per-plane Fréchet distance.

    Args:
        volumes: decoded image volumes ``[B, C, D, H, W]``.
        feature_net: callable ``[K, C, h, w] -> [K, D_feat]`` (the RadImageNet
            backbone, pre-wrapped with its BGR + ImageNet-mean preprocessing).
        center_slices_ratio: fraction of each axis's center slices to sample.
    """
    if volumes.dim() != 5:
        raise ValueError(f"Expected [B, C, D, H, W] volumes, got shape {tuple(volumes.shape)}.")
    per_plane: list[list[Tensor]] = [[], [], []]
    for axis in (2, 3, 4):  # D, H, W
        length = volumes.shape[axis]
        k = max(1, int(round(center_slices_ratio * length)))
        if k >= length:
            idx = list(range(length))
        else:
            start = (length - k) // 2
            idx = list(range(start, start + k))
        for i in idx:
            plane = volumes.index_select(axis, torch.tensor(i, device=volumes.device))
            # Collapse the singleton axis → [B, C, h, w] 2D image.
            plane = plane.squeeze(axis)
            if plane.dim() == 3:  # single batch → add batch dim for the net
                plane = plane.unsqueeze(0)
            per_plane[axis - 2].append(feature_net(plane).detach())
    return [torch.cat(feats, dim=0) if feats else torch.empty(0) for feats in per_plane]


def make_feature_network(name: str = "resnet50") -> nn.Module:
    """Load a RadImageNet feature backbone via ``torch.hub``.

    Wraps the model in a BGR + ImageNet-mean preprocessing module so it maps a
    single-channel medical slice ``[B, 1, H, W]`` to ``[B, D_feat]`` features.
    gauss (behind a jump host) must pre-populate ``$TORCH_HOME`` so the load runs
    offline — set ``TORCH_HOME`` and ``torch.hub.load(..., source='local')`` will
    find the cached ``Warvito/radimagenet_model`` repo.

    Only ``"resnet50"`` (RadImageNet) is supported here; the InceptionV3 /
    SqueezeNet backbones are out of scope.
    """
    if name != "resnet50":
        raise ValueError(f"Only 'resnet50' (RadImageNet) is supported, got {name!r}.")
    model = torch.hub.load(
        "Warvito/radimagenet_model",
        "radimagenet_resnet50",
        source="local" if _torchhub_cached() else "github",
        trust_repo=True,
    )
    return _RadImageNetFeatures(model)


def _torchhub_cached() -> bool:
    import os

    import torch.hub

    home = os.environ.get("TORCH_HOME", torch.hub.get_dir())
    return os.path.isdir(os.path.join(home, "Warvito_radimagenet_model_master"))


# ImageNet channel statistics (RadImageNet models are trained with ImageNet
# normalisation; the input is replicated to 3 channels from the single medical
# slice and mean-subtracted). BGR ordering is applied inside the wrapper.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _RadImageNetFeatures(nn.Module):
    """Wrap a RadImageNet classifier as a single-channel → features backbone.

    Replicates the medical slice to 3 channels, applies ImageNet mean/std
    normalisation (BGR channel order, matching RadImageNet's preprocessing), and
    returns the penultimate feature vector.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 1, H, W] in ~[0, 1]; replicate to 3-channel, BGR order.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.flip(1)  # RGB -> BGR
        x = (x - self._mean) / self._std
        with torch.no_grad():
            feats = self.model.features(x) if hasattr(self.model, "features") else self.model(x)
        return feats.flatten(1)
