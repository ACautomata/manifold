"""Bridge flow-match Heun scheduler: the data-to-data Brownian-bridge singular-branch
SDE transition (ADR-0024 / G2RPO).

A :class:`FlowMatchBridgeGRPOScheduler` **subclasses**
:class:`FlowMatchHeunDiscreteScheduler` **directly** (NOT the equimarginal
:class:`FlowMatchGRPOScheduler`) and adds only the stochastic transition the G2RPO
rollout branches on — the **data-to-data Brownian bridge** pinned at the predicted
target ``x̂₁_θ``. It inherits the transport (:meth:`add_noise`), the anchor grid
(:meth:`set_timesteps`), and the Heun reverse step (:meth:`euler_step` /
:meth:`heun_correct`) **verbatim**.

The bridge SDE is the forward Doob h-transform of the rectified-flow transport
pinned at ``Z_1 = x̂₁_θ``:

.. math::
    dZ_t = \\frac{\\hat x_1 - Z_t}{1 - t}\\, dt + \\sqrt{\\eta}\\, dW_t ,

whose drift is exactly the euler velocity ``(x̂₁ − z)/(1 − t)`` with **no Langevin /
score term** (§5 of the derivation — the score is absorbed into the pin; the
equimarginal reverse-SDE scheduler adds ``(σ²/2t)·x_θ`` because it *time-reverses*
a noise→data marginal, which the bridge does not). The exact transition over one
step ``[t, t_next]`` (constant diffusion ⇒ closed-form, no Euler-Maruyama) is

.. math::
    \\mu_\\theta = \\text{euler\\_step}(\\hat x_1, z, t, t_{next}) ,\\quad
    \\sigma^2 = \\eta\\,\\Delta t\\,\\frac{1 - t_{next}}{1 - t} ,

i.e. ``§6-mean == §7-mean == euler_step`` (an algebraic identity — the drift is
linear in ``Z`` and ``β(t)=1/(1−t)``); §7's std **vanishes at the terminal**
(``t_next → 1`` ⇒ ``σ² → 0`` ⇒ ``z_K → x̂₁``, stable), so there is **no ``t_safe``
clamp** (unlike the equimarginal scheduler's noise-end ``max(t, 1/n)`` floor — the
bridge has no noise-end; both endpoints are data, so ``σ`` never blows up).

``σ`` is **θ-independent** (depends only on ``(t, t_next, η)``), so the policy and
frozen-reference transitions share equal variance and the diagonal-Gaussian KL
collapses to ``0.5·‖μ_θ − μ_ref‖²/σ²`` (trace + log-det cancel) — the entire
``grpo.py`` spine (``gaussian_log_prob``, ``_transition_kl``, ``group_advantage``,
``clipped_surrogate_loss``, the multi-step PPO inner loop) reuses verbatim. The
bridge scheduler is one method (``sde_step_mean``) off ``euler_step``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..configuration import register_to_config
from .scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler


class FlowMatchBridgeGRPOScheduler(FlowMatchHeunDiscreteScheduler):
    """Flow-match Heun + the bridge singular-branch SDE transition (``sde_step_mean``).

    Identical transport + Heun steps to the Paired JiT scheduler (inherited); adds
    the data-to-data Brownian-bridge transition the G2RPO rollout branches on, as
    math-only ``(mean, std)``. The caller draws the sample and computes the
    old/new log-prob.

    Args:
        eta: the bridge diffusion scale ``η`` in ``σ² = η·Δt·(1−t_next)/(1−t)`` —
            the single exploration knob (``η → 0`` recovers the deterministic Paired
            JiT Heun anchor).
    """

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, t_eps: float = 0.05, eta: float = 0.7):
        """Args:
        num_train_timesteps / t_eps: inherited unchanged (see the parent).
        eta: the bridge ``σ²`` scale (default 0.7, ADR-0024).
        """
        super().__init__(num_train_timesteps=num_train_timesteps, t_eps=t_eps)
        self.eta = float(eta)

    def sde_step_mean(
        self, model_output: Tensor, sample: Tensor, t, t_next
    ) -> tuple[Tensor, Tensor]:
        """The bridge SDE exact transition ``(mean, std)`` (math only, no Langevin).

        ``mean = euler_step(x̂₁, z, t, t_next)[0]`` — the §5 forward-Doob-h-transform
        drift is *exactly* the euler velocity (the score is absorbed into the pin;
        no ``(σ²/2t)·x_θ`` correction). ``std = √(η·Δt·(1−t_next)/(1−t))`` — the §7
        exact transition (constant diffusion ⇒ closed-form), which vanishes at the
        terminal ``t_next → 1``. No ``t_safe`` clamp: both endpoints are data, so
        ``σ`` is finite everywhere on the grid.

        Args:
            model_output: the UNet's clean-latent prediction ``x̂₁_θ`` at ``(z, t)``.
            sample: the step-start latent ``z_k``.
            t / t_next: step-start / step-end flow-times (scalars or ``(B,)``).

        Returns:
            ``(mean, std)`` — ``mean`` matches ``sample``'s shape/dtype; ``std`` is a
            python float (scalar ``t``) or a broadcast ``(B, 1, …)`` tensor (``(B,)``).
        """
        # §5 / §6 / §7 share this mean (algebraic identity); §5 adds no Langevin term.
        z_euler, _ = self.euler_step(model_output, sample, t, t_next)
        t_b = self._step_t(t, sample)
        t_next_b = self._step_t(t_next, sample)
        dt = t_next_b - t_b
        var = self.eta * dt * (1.0 - t_next_b) / (1.0 - t_b)
        if isinstance(var, Tensor):
            std = torch.sqrt(var)
        else:
            std = math.sqrt(var)
        return z_euler, std


__all__ = ["FlowMatchBridgeGRPOScheduler"]
