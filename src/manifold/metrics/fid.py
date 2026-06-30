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
        # Global-average-pool so the backbone yields the canonical 2048-dim FID
        # feature. The offline loader leaves ``avgpool`` as Identity (returning a
        # spatial map); torchvision's forward flattens that map before we see it, so
        # pooling must happen *inside* the backbone via a real AdaptiveAvgPool2d (fc
        # stays Identity — no classifier). hope flattened the map (D~131k -> 69 GB
        # covariance); we pool to 2048 (17 MB) — a deliberate, tested divergence.
        if isinstance(model.avgpool, nn.Identity):
            model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
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
