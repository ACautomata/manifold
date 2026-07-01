"""Rectified-flow scheduler with a true two-evaluation Heun reverse step.

Owns the rectified-flow **transport** ``z = t·x + (1−t)·e`` (``t∈[0,1]``, ``t→1``
is clean data) and the reverse-time Heun integrator for the JiT x0-denoiser.
This is manifold's own scheduler — it deliberately collides with a ``diffusers``
class name to signal its role, but is **not** a re-export: a true trapezoidal
Heun *requires* evaluating the model at the Euler-advanced point, so the reverse
API is two-phase (a predictor :meth:`euler_step` and a corrector
:meth:`heun_correct`) rather than diffusers' single-call ``step()`` (ADR-0002).

The transport is shared verbatim with the training module, which obtains its
noised latent via :meth:`add_noise` rather than re-deriving it (ADR-0001) so
train and inference cannot drift.

Conventions:

- ``t = 1`` → clean data, ``t = 0`` → pure noise; sampling integrates ``t: 0 → 1``.
- The model output is interpreted as the clean-latent prediction x0
  (``prediction_type = "sample"``); the velocity is *derived* from it as
  ``v = (x0 − z) / (1 − t)`` by inverting the interpolation.
- The step-**start** denominator ``1 − t`` is **unclamped** (the start node never
  reaches 1, so it is never singular; clamping it would shrink the final Euler
  velocity and leave residual noise). Only the Heun **endpoint** denominator
  ``1 − t_next`` is clamped at ``t_eps`` (``t_next`` can hit 1).
"""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

from ..configuration import register_to_config
from .scheduling_utils import SchedulerMixin

#: A timestep may be a python/0-d scalar (inference, one flow-time node) or a
#: per-sample ``(B,)`` tensor (training, logit-normal sample).
Timestep = Union[float, int, Tensor]


class FlowMatchHeunDiscreteScheduler(SchedulerMixin):
    """Rectified-flow transport + true two-evaluation Heun reverse step (JiT).

    ``prediction_type`` is ``"sample"``: the coupled UNet predicts the clean
    latent x0, and the velocity is derived from that prediction.
    """

    prediction_type = "sample"

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, t_eps: float = 0.05):
        """Args:
        num_train_timesteps: only a label here; the UNet wrapper owns the
            time-embedding scale. Kept on the scheduler so a converted
            checkpoint's value round-trips alongside the transport config.
        t_eps: clamp on ``1 − t`` at the Heun endpoint (and in the derived
            velocity there), avoiding the singularity as ``t_next → 1``.
        """
        self.num_train_timesteps = int(num_train_timesteps)
        self.t_eps = float(t_eps)
        self.timesteps: Tensor | None = None

    # -- forward transport (shared with the training module) -----------------

    @staticmethod
    def _bcast_t(t: Timestep, reference: Tensor) -> Tensor:
        """Broadcast a scalar / ``(B,)`` timestep over a sample's spatial dims."""
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=reference.device)
        if t_tensor.dim() == 0:
            return t_tensor  # scalar broadcasts against the sample directly
        shape = [t_tensor.shape[0]] + [1] * (reference.dim() - 1)
        return t_tensor.view(shape)

    def add_noise(self, original_samples: Tensor, noise: Tensor, timesteps: Timestep) -> Tensor:
        """The rectified-flow transport ``z = t·x + (1 − t)·e``.

        ``t = 1`` returns the clean sample ``x``; ``t = 0`` returns ``e``. The
        training module calls this to obtain its noised latent (single source of
        truth), and inference's pure-noise start is the ``t = 0`` endpoint.
        """
        t = self._bcast_t(timesteps, original_samples)
        return t * original_samples + (1.0 - t) * noise

    # -- inference grid ------------------------------------------------------

    def set_timesteps(self, num_inference_steps: int, *, device=None) -> Tensor:
        """The JiT integration nodes ``t: 0 → 1`` (``num_inference_steps + 1``).

        Each of the ``num_inference_steps`` steps advances ``t_i → t_{i+1}`` from
        pure noise (``t_0 = 0``) to clean data (``t_n = 1``). Stored on
        ``self.timesteps`` and returned.
        """
        n = int(num_inference_steps)
        if n <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps!r}.")
        self.timesteps = torch.linspace(0.0, 1.0, n + 1, device=device, dtype=torch.float32)
        return self.timesteps

    def scale_model_input(self, sample: Tensor, timestep=None) -> Tensor:
        """Flow-matching needs no input scaling — returned unchanged.

        Present for diffusers-parity ergonomics; the pipeline does not depend on it.
        """
        return sample

    # -- reverse step: true two-evaluation Heun ------------------------------

    def euler_step(
        self, model_output: Tensor, sample: Tensor, t: float, t_next: float
    ) -> tuple[Tensor, Tensor]:
        """Predictor: derive the step-start velocity and advance to the Euler point.

        ``v1 = (x0_pred − z) / (1 − t)`` with the denominator **unclamped** (the
        step-start node ``t`` never reaches 1, so it is never singular). Returns
        the Euler-advanced point ``z + v1·dt`` and ``v1`` (the corrector needs it).
        """
        denom = 1.0 - float(t)
        v1 = (model_output.float() - sample.float()) / denom
        dt = float(t_next) - float(t)
        z_euler = sample.float() + v1 * dt
        return z_euler.to(sample.dtype), v1

    def heun_correct(
        self,
        model_output: Tensor,
        sample: Tensor,
        z_euler: Tensor,
        v1: Tensor,
        t: float,
        t_next: float,
    ) -> Tensor:
        """Corrector: derive the endpoint velocity and return the trapezoidal average.

        ``v2 = (x0_pred_euler − z_euler) / max(1 − t_next, t_eps)`` with the
        denominator **clamped** at ``t_eps`` (``t_next`` can hit 1), then the
        trapezoidal Heun update ``z + 0.5·(v1 + v2)·dt``.
        """
        denom = max(1.0 - float(t_next), float(self.t_eps))
        v2 = (model_output.float() - z_euler.float()) / denom
        dt = float(t_next) - float(t)
        out = sample.float() + 0.5 * (v1 + v2) * dt
        return out.to(sample.dtype)
