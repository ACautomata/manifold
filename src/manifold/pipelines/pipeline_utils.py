"""Pipeline base class, mimicking ``diffusers.DiffusionPipeline``.

A manifold Pipeline holds the model/scheduler components that turn noise +
conditions into a decoded volume, and persists them as a per-component directory
layout described by a ``model_index.json``. The base provides the config mixin
and the registry helpers; each concrete pipeline implements
``from_pretrained`` / ``save_pretrained`` for its own component layout. It does
**not** subclass ``diffusers.DiffusionPipeline`` (ADR-0001).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..configuration import ConfigMixin


class DiffusionPipeline(ConfigMixin):
    """Base for manifold inference pipelines."""

    #: Top-level index file naming the pipeline and its components.
    config_name = "model_index.json"


def min_max_to_unit(vol: Tensor) -> Tensor:
    """Per-volume min-max normalization to ``[0, 1]`` (the published-output contract).

    The Pipeline's published-inference output convention (ADR-0036): each volume is
    normalized by its own ``[min, max]`` so the published image is in ``[0, 1]``
    regardless of the raw VAE decode range, and the paired-fidelity metric
    (:class:`~manifold.metrics.PairedFidelityMetrics`) can assume ``data_range = 1.0``.
    Per-sample (each volume in the batch by its own range); a degenerate zero-range
    volume maps to zeros. Shared by the ControlNet pipeline's decode path and the
    before/after eval driver, so both normalize identically.
    """
    b = vol.shape[0]
    flat = vol.reshape(b, -1)  # [B, C*D*H*W]
    mn = flat.amin(dim=1).view(b, 1, 1, 1, 1)
    mx = flat.amax(dim=1).view(b, 1, 1, 1, 1)
    rng = mx - mn
    rng = torch.where(rng > 0, rng, torch.ones_like(rng))  # avoid div-by-zero
    return (vol - mn) / rng
