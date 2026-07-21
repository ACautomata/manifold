"""FeatureExtractor — 2.5D feature extraction via injectable feature_net.

Thin callable wrapper around :func:`~manifold.metrics.fid.math.get_features_2p5d`
with a pre-bound ``center_slices_ratio``.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from manifold.metrics.fid.math import get_features_2p5d


class FeatureExtractor:
    """2.5D extraction via injectable feature_net.

    Args:
        feature_net: callable ``[K, C, h, w] -> [K, D_feat]`` (the RadImageNet
            backbone, or a test fake).
        center_slices_ratio: fraction of each axis's center slices to sample.
    """

    def __init__(
        self,
        feature_net: Callable[[Tensor], Tensor],
        *,
        center_slices_ratio: float = 0.5,
    ) -> None:
        self._feature_net = feature_net
        self._center_slices_ratio = float(center_slices_ratio)

    def __call__(self, volumes: Tensor) -> list[Tensor]:
        """Extract per-plane features.

        Args:
            volumes: decoded image volumes ``[B, C, D, H, W]``.

        Returns:
            Three ``[M_axis, D_feat]`` tensors — one per plane (XY/YZ/ZX).
        """
        return get_features_2p5d(
            volumes, self._feature_net,
            center_slices_ratio=self._center_slices_ratio,
        )
