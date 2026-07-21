"""SufficientStatsReducer — symmetric per-plane all_reduce to global (mu, sigma, n).

Every rank enters one all_reduce per plane (always 3 planes), contributing zero
stats for an empty local shard so the collective cannot deadlock. A plane with
global count < 2 yields None (FID undefined for a single sample).
"""

from __future__ import annotations

import torch
from torch import Tensor

from manifold.metrics.fid.math import features_to_sufficient_stats, moments_from_sufficient_stats


class SufficientStatsReducer:
    """Symmetric per-plane all_reduce of sufficient stats to global moments.

    Args:
        feat_dim: the feature vector dimension (probed once at stage time;
            deterministic across ranks).
    """

    def __init__(self, feat_dim: int) -> None:
        self._feat_dim = int(feat_dim)

    def __call__(
        self,
        planes: list[Tensor | tuple[Tensor, Tensor, int]],
        device: torch.device,
    ) -> list[tuple[Tensor, Tensor, int] | None]:
        """All-reduce per-plane sufficient stats to global ``(mu, sigma, n)``.

        Each input is either a feature tensor ``[M_axis, D_feat]`` or a
        pre-computed sufficient-stats tuple ``(sum_x, sum_xxT, n)`` — the
        latter avoids fallible float conversion + ``features.T @ features``
        inside the reduction sequence (codex #171 P1).

        Args:
            planes: list of 3 items (feature tensor or pre-computed stats).
            device: the device for zero-stat allocation.

        Returns:
            List of length 3. Each entry is ``(mu, sigma, n)`` if global n >= 2,
            else None.
        """
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = int(torch.distributed.get_world_size())
        d = self._feat_dim
        out: list[tuple[Tensor, Tensor, int] | None] = []
        for item in planes:
            if isinstance(item, Tensor):
                # Feature tensor [N, D] (possibly empty).
                feats = item
                if feats.numel() == 0 or feats.shape[0] == 0:
                    sum_x = torch.zeros(d, device=device, dtype=torch.float32)
                    sum_xxT = torch.zeros(d, d, device=device, dtype=torch.float32)
                    n = 0
                else:
                    sum_x, sum_xxT, n = features_to_sufficient_stats(feats.float())
            else:
                # Pre-computed sufficient stats (sum_x, sum_xxT, n).
                sum_x, sum_xxT, n = item  # type: ignore[misc]
            if world > 1:
                torch.distributed.all_reduce(sum_x)
                torch.distributed.all_reduce(sum_xxT)
                n_t = torch.tensor([float(n)], device=device, dtype=torch.float32)
                torch.distributed.all_reduce(n_t)
                n = int(n_t.item())
            if n >= 2:
                mu, sigma, _ = moments_from_sufficient_stats(sum_x, sum_xxT, n)
                out.append((mu, sigma, n))
            else:
                out.append(None)
        return out
