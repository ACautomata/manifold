"""Partial per-sample flow-match Heun scheduler (GRPO reward pair generation).

A :class:`PartialFlowMatchHeunScheduler` **subclasses**
:class:`FlowMatchHeunDiscreteScheduler` and adds only :meth:`set_timesteps_partial`,
producing a per-sample ``(B, n+1)`` grid over ``t_start → 1`` (the JiT scheduler's
grid is batch-wide ``0 → 1``). It **inherits** the transport
(:meth:`~FlowMatchHeunDiscreteScheduler.add_noise`, already per-sample-capable)
and the Heun step math (:meth:`~FlowMatchHeunDiscreteScheduler.euler_step` /
:meth:`~FlowMatchHeunDiscreteScheduler.heun_correct`, widened to accept ``(B,)`
timesteps) **verbatim** — it never reimplements them. Forking either would feed
the frozen JiT denoiser out-of-distribution noise (it was trained on
``z = t·x + (1−t)·e``) and the reward signal would be meaningless; forking also
violates the ADR-0001/0002 single-source-of-truth (ADR-0008).
"""

from __future__ import annotations

import torch
from torch import Tensor

from .scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler


class PartialFlowMatchHeunScheduler(FlowMatchHeunDiscreteScheduler):
    """Per-sample partial-range flow-match Heun scheduler (reward pairs).

    Identical transport + Heun steps to the JiT scheduler (inherited); only the
    integration grid differs — each sample integrates from its own ``t_start`` to
    clean (``t = 1``), so one batch can span a continuous corruption spectrum
    (winners near-clean, losers near-noise) under a single shared step budget.
    """

    def set_timesteps_partial(self, t_start: Tensor, num_steps: int, *, device=None) -> Tensor:
        """Per-sample grid ``(B, n+1)`` over ``[t_start, 1]`` with ``n = num_steps``.

        Each sample's row is ``linspace(t_start[b], 1, n+1)``. ``n`` steps advance
        every sample from its own corruption level to clean, so the per-sample step
        size ``δt = (1 − t_start[b]) / n`` **differs across the batch** (a winner
        denoised from near-clean takes small steps; a loser from near-noise takes
        large ones) even though the step *count* is shared. The endpoint column is
        ``1`` for every sample (the Heun final-step Euler, where ``1 − t_next``
        vanishes).

        A scalar ``t_start`` is treated as a single-sample batch ``(1,)``.
        """
        n = int(num_steps)
        if n <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps!r}.")
        t_start = torch.as_tensor(t_start, dtype=torch.float32, device=device)
        if t_start.dim() == 0:
            t_start = t_start.unsqueeze(0)
        fracs = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=torch.float32)  # (n+1,)
        # grid[b, i] = t_start[b] + (1 − t_start[b]) · (i / n)
        return t_start[:, None] + (1.0 - t_start[:, None]) * fracs[None, :]
