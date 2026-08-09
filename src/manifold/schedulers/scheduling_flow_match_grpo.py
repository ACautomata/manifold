"""GRPO flow-match Heun scheduler: singular-branch SDE transition math (ADR-0011).

A :class:`FlowMatchGRPOScheduler` **subclasses** :class:`FlowMatchHeunDiscreteScheduler`
and adds only the stochastic transition the GRPO rollout branches on — the x-pred
**equimarginal reverse-time SDE** of the JiT transport, exposed as the math-only
:meth:`sde_step_mean`. It **inherits** the transport
(:meth:`~FlowMatchHeunDiscreteScheduler.add_noise`), the anchor grid
(:meth:`~FlowMatchHeunDiscreteScheduler.set_timesteps`), and the Heun reverse step
(:meth:`~FlowMatchHeunDiscreteScheduler.euler_step` /
:meth:`~FlowMatchHeunDiscreteScheduler.heun_correct`) **verbatim** — forking any of
those would break the ADR-0001/0002/0008 single-source-of-truth and feed the policy
out-of-distribution noise.

The SDE drift reuses ``euler_step``'s velocity ``v_θ = (x_θ − z)/(1 − t)`` — the SDE
mean is *Euler advance + Langevin correction* — so transport/Heun math is inherited,
never reimplemented. ``sde_step_mean`` returns the Gaussian ``(mean, std)`` of the
transition ``π_θ(z_{k+1} | z_k, t_k) = N(z_k + Δt·b_θ, σ²_t·Δt·I)``; the caller draws
``z_{k+1} = mean + std·ξ`` and owns the log-prob/replay (decision-encoding from the
derivation prototype, ADR-0011).

Diffusion schedule: equimarginal reverse-SDE ``σ_t = η·sqrt((1−t)/t)`` (η default
0.7). The noise-end (``t → 0``) blowup is clamped via ``t_safe = max(t, 1/n)``
(``n = num_inference_steps``, mirroring Granular's ``σ==1 → σ_max``); the clean-end
``1/(1 − t)`` is ``euler_step``'s (the start node never reaches ``t = 1``).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor

from ..configuration import register_to_config
from .scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler, Timestep


class FlowMatchGRPOScheduler(FlowMatchHeunDiscreteScheduler):
    """Flow-match Heun + the GRPO singular-branch SDE transition (``sde_step_mean``).

    Identical transport + Heun steps to the JiT scheduler (inherited); adds the
    equimarginal reverse-SDE transition the GRPO rollout branches on, as math-only
    ``(mean, std)``. The caller draws the sample and computes the old/new log-prob.

    Args:
        eta: the diffusion scale ``η`` in ``σ_t = η·sqrt((1−t)/t)`` — the single
            exploration knob (``η → 0`` recovers the deterministic Heun anchor).
    """

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, t_eps: float = 0.05, eta: float = 0.7):
        """Args:
        num_train_timesteps / t_eps: inherited unchanged (see the parent).
        eta: the equimarginal ``σ_t`` scale (default 0.7, ADR-0011).
        """
        super().__init__(num_train_timesteps=num_train_timesteps, t_eps=t_eps)
        self.eta = float(eta)
        #: The anchor grid resolution; set by :meth:`set_timesteps`. The noise-end
        #: clamp ``t_safe = max(t, 1/n)`` reads it (the floor is the first interior
        #: grid node, mirroring Granular's ``σ_max``).
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int, *, device=None) -> Tensor:
        """The JiT anchor grid ``linspace(0, 1, n+1)`` (inherited); also stash ``n``.

        Stashing ``num_inference_steps`` fixes the noise-end floor ``1/n`` for
        :meth:`sde_step_mean`. The grid itself is the parent's, unchanged.
        """
        ts = super().set_timesteps(num_inference_steps, device=device)
        self.num_inference_steps = int(num_inference_steps)
        return ts

    def rollout_range(
        self,
        x0_fn: Callable[[Tensor, Timestep], Tensor],
        z_start: Tensor,
        nodes: Tensor,
        start_i: int,
        end_i: int,
    ) -> list[Tensor]:
        """Deterministic Heun over the node interval ``[start_i, end_i]``; returns ``[z_start_i, …, z_end_i]``.

        The GRPO training path's interval-slice loop (ADR-0005, issue #211): runs the
        inherited two-eval Heun step over each node pair ``nodes[i] → nodes[i+1]`` —
        predictor at ``z``, Euler advance, corrector at the Euler point — with the
        final-step Euler guard when ``t_next == 1`` (the ``1 − t_next`` corrector
        denominator diverges; the same convention as
        :meth:`FlowMatchHeunDiscreteScheduler.heun_rollout`). Returns the visited
        latents ``[z_start_i, z_end_i]`` inclusive so the caller can index the anchor
        nodes by step. Used for both the shared anchor (``start_i=0``) and each
        branch's suffix (``start_i=k+1``).

        ``x0_fn`` is the injected x0 source ``x0_fn(z, t) → x0`` (the clean-latent
        prediction; the ControlNet policy pre-binds its conditionals and batch
        dispatch — ``None`` for the UNet policy is folded in by the caller). ``nodes``
        is the batch-wide anchor grid (:meth:`set_timesteps`); the step math is the
        inherited :meth:`FlowMatchHeunDiscreteScheduler.euler_step` /
        :meth:`FlowMatchHeunDiscreteScheduler.heun_correct` — never reimplemented
        (ADR-0001/0002). The interval shape differs from the full-range
        :meth:`FlowMatchHeunDiscreteScheduler.heun_rollout` primitive (anchor + suffix
        slices of the GRPO grid), so the loop lives here rather than being forced
        into it.
        """
        zs = [z_start]
        z = z_start
        for i in range(start_i, end_i):
            t = nodes[i]
            t_next = nodes[i + 1]
            x0_1 = x0_fn(z, t)
            z_euler, v1 = self.euler_step(x0_1, z, t, t_next)
            if float(t_next) >= 1.0:
                # Final step is Euler: at t_next = 1 the denominator 1 − t_next
                # vanishes, so the second Heun evaluation is undefined.
                z = z_euler
            else:
                x0_2 = x0_fn(z_euler, t_next)
                z = self.heun_correct(x0_2, z, z_euler, v1, t, t_next)
            zs.append(z)
        return zs

    def _sigma_t(self, t_b):
        """Equimarginal ``σ_t = η·sqrt((1−t)/t)`` with the noise-end clamp applied.

        Returns ``(σ_t, t_safe)`` where ``t_safe = max(t, 1/n)`` is the clamped time
        used for BOTH ``σ_t`` and the Langevin ``1/t`` denominator (so the whole
        ``(σ²/2t)`` coefficient stays finite at ``t = 0``). ``t_b`` is a python float
        (the scalar anchor path) or a broadcast ``(B, 1, …)`` tensor (the per-sample
        path) — :meth:`FlowMatchHeunDiscreteScheduler._step_t`'s two forms.
        """
        if self.num_inference_steps is None:
            raise RuntimeError(
                "sde_step_mean requires the anchor grid: call set_timesteps(n) first "
                "(the noise-end clamp 1/n needs the grid resolution)."
            )
        floor = 1.0 / self.num_inference_steps
        if isinstance(t_b, Tensor):
            t_safe = t_b.clamp(min=floor)
            sigma = self.eta * torch.sqrt((1.0 - t_safe) / t_safe)
        else:
            t_safe = max(float(t_b), floor)
            sigma = self.eta * math.sqrt((1.0 - t_safe) / t_safe)
        return sigma, t_safe

    def sde_step_mean(
        self, model_output: Tensor, sample: Tensor, t, t_next
    ) -> tuple[Tensor, Tensor]:
        """The x-pred equimarginal reverse-SDE Euler-Maruyama ``(mean, std)`` (math only).

        Drift ``b_θ = v_θ + (σ²_t/2t)·x_θ`` where ``v_θ = (x_θ − z)/(1 − t)`` is the
        velocity :meth:`euler_step` derives (unclamped step-start denom — the start
        node never reaches ``t = 1``). The transition is
        ``N(z + Δt·b_θ, σ²_t·Δt·I)``; this returns its ``(mean, std)`` so the caller
        draws ``z_{k+1}`` and owns the log-prob:

        - ``mean = z + Δt·(v_θ + (σ²_t/2t)·x_θ)``  (Euler advance + Langevin correction)
        - ``std  = σ_t·√Δt``

        Both endpoint blowups are handled: the clean-end ``1/(1−t)`` is ``euler_step``'s
        convention; the noise-end ``σ_t`` / ``1/t`` use ``t_safe = max(t, 1/n)``.

        Args:
            model_output: the UNet's clean-latent prediction ``x_θ`` at ``(z, t)``.
            sample: the step-start latent ``z_k``.
            t / t_next: step-start / step-end flow-times (scalars or ``(B,)``).

        Returns:
            ``(mean, std)`` — ``mean`` matches ``sample``'s shape/dtype; ``std`` is a
            python float (scalar ``t``) or a broadcast ``(B, 1, …)`` tensor (``(B,)``).
        """
        # Euler advance + the inherited velocity v_θ (transport/Heun math, not forked).
        z_euler, _ = self.euler_step(model_output, sample, t, t_next)
        t_b = self._step_t(t, sample)
        t_next_b = self._step_t(t_next, sample)
        dt = t_next_b - t_b
        sigma, t_safe = self._sigma_t(t_b)
        langevin = (sigma ** 2) / (2.0 * t_safe)  # (σ²/2t_safe) — finite at t = 0
        mean = z_euler.float() + (dt * langevin) * model_output.float()
        std = sigma * (dt ** 0.5)
        return mean.to(sample.dtype), std
