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
- :func:`make_feature_network` — the RadImageNet ResNet50 factory (loads the
  cached ``RadImageNet-ResNet50_notop.pth`` state_dict offline; falls back to
  ``torch.hub`` only when the checkpoint is absent).

The InceptionV3 / SqueezeNet backbones, the bootstrap CI, and the standalone
2.5D orchestrator are **not** ported (out of scope). ``scipy`` and
``torchvision`` are required dependencies.
"""

from __future__ import annotations

import os
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


def frechet_from_moments(
    mu_g: Tensor, mu_r: Tensor, sigma_g: Tensor, sigma_r: Tensor,
    n1: int, n2: int, *, ridge: float = 1e-6,
) -> Tensor:
    """Unbiased Frechet distance from precomputed per-set moments.

    Identical math to :func:`frechet_distance_unbiased` once each set's mean and
    unbiased covariance are known. Extracted so a DDP-sharded FID can accumulate
    sufficient statistics ``(sum_x, sum_xxT, n)`` per rank, all-reduce them to
    global moments, and call this for the exact global FID (no feature-matrix
    gather). ``n1``/``n2`` are the GLOBAL per-set counts (the unbiased ``Tr(S)/n``
    mean-term bias is a function of the global count + global covariance).
    """
    mean_term = (mu_g - mu_r).pow(2).sum() - _tr(sigma_g) / n1 - _tr(sigma_r) / n2
    eye = torch.eye(sigma_g.shape[0], dtype=sigma_g.dtype, device=sigma_g.device) * ridge
    cov_prod = (sigma_g + eye) @ (sigma_r + eye)
    cov_term = _tr(sigma_g) + _tr(sigma_r) - 2.0 * _tr(_principal_sqrtm(cov_prod))
    return (mean_term + cov_term).clamp(min=0.0)


def features_to_sufficient_stats(features: Tensor) -> tuple[Tensor, Tensor, int]:
    """``[N, D]`` feature matrix -> ``(sum_x[D], sum_xxT[D, D], n)`` for distributed FID.

    The sufficient statistics for a per-set mean and unbiased covariance: each rank
    reduces these three (sum) across ranks to recover the global moments exactly
    (see :func:`moments_from_sufficient_stats`), avoiding a full ``[N, D]`` gather.
    """
    n = features.shape[0]
    return features.sum(dim=0), features.t() @ features, n


def moments_from_sufficient_stats(sum_x: Tensor, sum_xxT: Tensor, n: int) -> tuple[Tensor, Tensor, int]:
    """All-reduced sufficient stats -> ``(mu, sigma_unbiased, n)`` (exact global moments).

    ``mu = sum_x / n`` and ``sigma = (sum_xxT - n * mu outer mu) / (n - 1)`` ==
    the raw ``1/(n-1) * sum (x - mu)(x - mu)^T`` over the full (global) feature set.
    """
    mu = sum_x / n
    sigma = (sum_xxT - n * torch.outer(mu, mu)) / (n - 1)
    return mu, sigma, n


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
    return frechet_from_moments(
        gen.mean(dim=0), real.mean(dim=0),
        _cov_unbiased(gen), _cov_unbiased(real),
        n1, n2, ridge=ridge,
    )


def get_features_2p5d(
    volumes: Tensor,
    feature_net: Callable[[Tensor], Tensor],
    *,
    center_slices_ratio: float = 0.5,
) -> list[Tensor]:
    """Extract 2.5D features over the XY / YZ / ZX planes (stacked per volume).

    For each volume ``[B, C, D, H, W]`` and each of the three orthogonal axes,
    take ``max(1, round(center_slices_ratio·axis_len))`` center slices, batch them
    into one plane (``[K, C, h, w]``) per (volume, axis), run each plane-batch
    through *feature_net*, and collect the feature vectors per plane. Returns
    three ``[M_axis, D_feat]`` tensors — one per plane — so the caller can compute
    a per-plane Fréchet distance.

    The feature net is called once per (volume, plane) — NOT once per slice — so
    its per-plane-batch min-max normalization matches hope's
    ``radimagenet_intensity_normalisation`` (one global min/max over all of a
    volume's slices for that plane). hope computes that min-max per volume (its
    ``get_features_2p5d`` is invoked one volume at a time), so we loop per volume
    here rather than batching volumes together.

    Args:
        volumes: decoded image volumes ``[B, C, D, H, W]``.
        feature_net: callable ``[K, C, h, w] -> [K, D_feat]`` (the RadImageNet
            backbone, pre-wrapped with its BGR + per-plane-batch min-max +
            ImageNet-mean preprocessing).
        center_slices_ratio: fraction of each axis's center slices to sample.
    """
    if volumes.dim() != 5:
        raise ValueError(f"Expected [B, C, D, H, W] volumes, got shape {tuple(volumes.shape)}.")
    per_plane: list[list[Tensor]] = [[], [], []]
    for v in range(volumes.shape[0]):
        vol = volumes[v : v + 1]  # (1, C, D, H, W) — hope min-maxes per volume
        for axis in (2, 3, 4):  # D, H, W
            length = vol.shape[axis]
            k = max(1, int(round(center_slices_ratio * length)))
            if k >= length:
                idx = list(range(length))
            else:
                start = (length - k) // 2
                idx = list(range(start, start + k))
            slices = []
            for i in idx:
                # Select slice i along the axis, collapse it → [1, C, h, w].
                plane = vol.index_select(axis, torch.tensor(i, device=vol.device)).squeeze(axis)
                slices.append(plane)
            plane_batch = torch.cat(slices, dim=0)  # (K, C, h, w)
            per_plane[axis - 2].append(feature_net(plane_batch).detach())
    return [torch.cat(feats, dim=0) if feats else torch.empty(0) for feats in per_plane]


def make_feature_network(name: str = "resnet50") -> nn.Module:
    """Load a RadImageNet ResNet50 feature backbone.

    On offline hosts (gauss, behind a jump host), builds the backbone directly
    from the cached ``RadImageNet-ResNet50_notop.pth`` state_dict — no
    ``torch.hub`` and no network. (``torch.hub.load`` is offline-broken there: the
    cached repo dir uses a hyphen + ``_main`` suffix that the ``source='local'``
    branch resolver mishandles, and the entry point never lands on ``sys.path``.)
    Only when the checkpoint is *absent* does it fall back to a GitHub ``torch.hub``
    load for hosts with network.

    Wraps the model in a BGR + ImageNet-mean preprocessing module so it maps a
    single-channel medical slice ``[B, 1, H, W]`` to ``[B, D_feat]`` features.

    Only ``"resnet50"`` (RadImageNet) is supported here; the InceptionV3 /
    SqueezeNet backbones are out of scope.
    """
    if name != "resnet50":
        raise ValueError(f"Only 'resnet50' (RadImageNet) is supported, got {name!r}.")
    ckpt = _radimagenet_checkpoint_path()
    if os.path.isfile(ckpt):
        model = _load_radimagenet_resnet50(ckpt)
    else:  # online fallback for hosts without the cached checkpoint
        model = torch.hub.load(
            _RADIMAGENET_REPO,
            "radimagenet_resnet50",
            source="github",
            trust_repo=True,
        )
    return _RadImageNetFeatures(model)


# Cached RadImageNet assets: the ``_notop`` ResNet50 state_dict (no classifier
# head) under ``$TORCH_HOME/checkpoints/``, and the (online-only) hub repo.
_RADIMAGENET_CKPT = "RadImageNet-ResNet50_notop.pth"
_RADIMAGENET_REPO = "Warvito/radimagenet-models"


def _radimagenet_checkpoint_path() -> str:
    """Resolve the cached RadImageNet ResNet50 (no-top) checkpoint path.

    Honours ``TORCH_HOME`` then ``torch.hub.get_dir()``; the checkpoint lives in
    ``<hub>/checkpoints/``. The path is returned whether or not the file exists —
    :func:`make_feature_network` falls back to the online hub load when it does
    not, so the absence is not an error here.
    """
    home = os.environ.get("TORCH_HOME") or torch.hub.get_dir()
    return os.path.join(home, "checkpoints", _RADIMAGENET_CKPT)


def _match_radimagenet_arch(model: nn.Module) -> None:
    """Reshape torchvision's ``resnet50`` to the RadImageNet architecture in place.

    RadImageNet's ResNet50 (converted from a Keras model) differs from
    torchvision's in three inference-relevant ways — without *all three* the
    loaded weights compute the wrong features despite every key/shape matching:

    * **conv bias**: Keras convs carry a trained bias (53 keys); torchvision uses
      ``bias=False``. Dropping them skews features by ~3% (cosine 0.97).
    * **bottleneck stride**: Keras puts the downsampling stride on ``conv1`` (the
      1×1); torchvision puts it on ``conv2`` (the 3×3). Same weight shapes, very
      different receptive field (this is the divergence the shape-check hides).
    * **BN ``eps``**: 1.001e-5 vs torchvision's 1e-5.

    After this the features match the canonical ``torch.hub`` model bit-for-bit.
    """
    from torchvision.models.resnet import Bottleneck

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1.001e-5
        elif isinstance(module, Bottleneck) and module.conv2.stride[0] != 1:
            # Move the downsampling stride conv2 (3×3) -> conv1 (1×1).
            module.conv1.stride = module.conv2.stride
            module.conv2.stride = (1, 1)
    _enable_conv_bias(model)


def _enable_conv_bias(model: nn.Module) -> None:
    """Rebuild ``model``'s conv layers with ``bias=True`` in place (recursive)."""
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Conv2d):
            biased = nn.Conv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=True,
            )
            biased.weight = child.weight  # reuse the initialised Parameter
            setattr(model, name, biased)
        else:
            _enable_conv_bias(child)


def _load_radimagenet_resnet50(ckpt_path: str) -> nn.Module:
    """Build a bias-True ``resnet50`` and strict-load the cached state_dict.

    The checkpoint is the ``_notop`` variant (no ``fc`` head), so ``fc`` and
    ``avgpool`` are replaced with :class:`~torch.nn.Identity` — the model then
    returns the post-``layer4`` spatial map (the penultimate feature tensor),
    matching ``torch.hub.load(..., 'radimagenet_resnet50')``. No network.
    """
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    _match_radimagenet_arch(model)
    model.avgpool = nn.Identity()
    model.fc = nn.Identity()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model


# Caffe-mode channel mean (RadImageNet's training preprocessing: [0,1] ->
# RGB->BGR -> mean-subtract ONLY, no std division — the hub model has no internal
# normalisation, so this lands directly on conv1). Ordered BGR to pair with the
# post-flip channel order inside the wrapper; the single medical slice is
# replicated to 3 channels before the flip.
_IMAGENET_MEAN = (0.406, 0.456, 0.485)


class _RadImageNetFeatures(nn.Module):
    """Wrap a RadImageNet classifier as a single-channel → features backbone.

    Replicates the medical slice to 3 channels, applies caffe-mode mean
    subtraction in BGR channel order (RadImageNet's training preprocessing:
    ``[0,1]`` → RGB→BGR → mean-subtract, no std division), and returns the
    penultimate feature vector.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        # Global-average-pool so the backbone yields the canonical 2048-dim FID
        # feature. The offline loader leaves ``avgpool`` as Identity (returning a
        # spatial map); torchvision's forward flattens that map before we see it, so
        # pooling must happen *inside* the backbone via a real AdaptiveAvgPool2d (fc
        # stays Identity — no classifier). NOT a divergence from hope: the hub
        # RadImageNet ResNet50 has no avgpool/fc (its _forward_impl returns the
        # layer4 (B,2048,7,7) map) and hope pools it to (B,2048) outside via
        # spatial_average; this AdaptiveAvgPool2d matches that bit-for-bit.
        if isinstance(model.avgpool, nn.Identity):
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        # x: [K, 1, H, W] raw decoded slice batch for ONE plane of ONE volume.
        # hope's RadImageNet preprocessing (metrics.fid.radimagenet_intensity_
        # normalisation): replicate 1->3, RGB->BGR, per-plane-batch min-max to
        # [0,1], then caffe-mode ImageNet-mean subtract (no std division). The
        # min-max is over the WHOLE plane-batch (all K slices × channels), so
        # get_features_2p5d must call this once per (volume, plane) — not per slice.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.flip(1)  # RGB -> BGR
        minv = torch.min(x)
        maxv = torch.max(x)
        x = (x - minv) / (maxv - minv + 1e-10)  # per-plane-batch min-max to [0,1]
        x = x - self._mean  # caffe-mode: mean-subtract only (no std division)
        with torch.no_grad():
            feats = self.model.features(x) if hasattr(self.model, "features") else self.model(x)
        return feats.flatten(1)
