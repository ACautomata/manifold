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

from typing import Callable, Union

import torch
from torch import Tensor

from ..configuration import register_to_config
from .scheduling_utils import SchedulerMixin

#: A timestep may be a python/0-d scalar (inference, one flow-time node) or a
#: per-sample ``(B,)`` tensor (training's logit-normal sample, or a per-sample
#: partial-denoise step where each sample sits at its own flow-time).
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

    @staticmethod
    def _step_t(t: Timestep, reference: Tensor) -> Union[float, Tensor]:
        """Coerce a reverse-step endpoint to its arithmetic form.

        A scalar (python ``float``/``int`` or 0-d tensor) becomes a python
        ``float`` — the byte-identical scalar fast path (the JiT train/inference
        callers pass scalars, and their outputs must not change). A ``(B,)``
        tensor is broadcast over the sample's spatial dims as ``(B, 1, 1, …)`` so
        each sample advances by its own ``dt`` and divides by its own ``1 − t``
        (per-sample partial denoise, ADR-0008).
        """
        if isinstance(t, Tensor):
            t = torch.as_tensor(t, dtype=torch.float32, device=reference.device)
            if t.dim() == 0:
                return float(t)
            return FlowMatchHeunDiscreteScheduler._bcast_t(t, reference)
        return float(t)

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
        self, model_output: Tensor, sample: Tensor, t: Timestep, t_next: Timestep
    ) -> tuple[Tensor, Tensor]:
        """Predictor: derive the step-start velocity and advance to the Euler point.

        ``v1 = (x0_pred − z) / (1 − t)`` with the denominator **unclamped** (the
        step-start node ``t`` never reaches 1, so it is never singular). Returns
        the Euler-advanced point ``z + v1·dt`` and ``v1`` (the corrector needs it).

        ``t`` / ``t_next`` may be scalars (the JiT path — byte-identical to the
        scalar arithmetic) or ``(B,)`` tensors (per-sample partial denoise; each
        sample divides by its own ``1 − t`` and advances by its own ``dt``).
        """
        t_b = self._step_t(t, sample)
        t_next_b = self._step_t(t_next, sample)
        denom = 1.0 - t_b
        v1 = (model_output.float() - sample.float()) / denom
        dt = t_next_b - t_b
        z_euler = sample.float() + v1 * dt
        return z_euler.to(sample.dtype), v1

    def heun_correct(
        self,
        model_output: Tensor,
        sample: Tensor,
        z_euler: Tensor,
        v1: Tensor,
        t: Timestep,
        t_next: Timestep,
    ) -> Tensor:
        """Corrector: derive the endpoint velocity and return the trapezoidal average.

        ``v2 = (x0_pred_euler − z_euler) / max(1 − t_next, t_eps)`` with the
        denominator **clamped** at ``t_eps`` (``t_next`` can hit 1), then the
        trapezoidal Heun update ``z + 0.5·(v1 + v2)·dt``.

        ``t`` / ``t_next`` may be scalars or ``(B,)`` tensors (the clamp is
        elementwise over the batch in the tensor path).
        """
        t_b = self._step_t(t, sample)
        t_next_b = self._step_t(t_next, sample)
        one_minus = 1.0 - t_next_b
        if isinstance(one_minus, Tensor):
            denom = one_minus.clamp(min=self.t_eps)
        else:
            denom = max(one_minus, float(self.t_eps))
        v2 = (model_output.float() - z_euler.float()) / denom
        dt = t_next_b - t_b
        out = sample.float() + 0.5 * (v1 + v2) * dt
        return out.to(sample.dtype)

    # -- the shared full-range/per-sample rollout primitive (ADR-0005) ---------

    def heun_rollout(
        self,
        x0_fn: Callable[[Tensor, Timestep], Tensor],
        z_start: Tensor,
        grid: Tensor,
        *,
        grad: str,
    ) -> Tensor:
        """The shared x0 Heun rollout primitive — the Euler→guard→Heun loop (ADR-0005).

        The full-range / per-sample rollouts converge here: ``num_steps`` Heun steps
        over ``grid``'s nodes under the true two-evaluation Heun — predictor at
        ``z_t``, Euler advance, corrector at ``z_{t+dt}`` — Euler on the final
        step, where ``t_next = 1`` makes the ``1 − t_next`` corrector denominator
        vanish. The loop body carries zero branching on the grid shape: each node
        is passed to :meth:`euler_step` / :meth:`heun_correct` verbatim, and both
        accept a python-float node (the scalar full-range path, bit-identical to
        the pre-primitive arithmetic) or a ``(B,)`` node (the per-sample partial
        path — each sample divides by its own ``1 − t`` and advances by its own dt,
        ADR-0008). The grad context is two-state: ``"inference"``
        (``torch.inference_mode()``) for the forward-only paths, ``"no_grad"``
        (``torch.no_grad()``) when the output must stay backward-safe — the online
        reward path's discriminator backward forbids saving an ``inference_mode``
        tensor (issue #49/50); both run identical math, only the tensor flag differs.
        Autocast on cuda mirrors the deployed sampler; disabled off-cuda so CPU
        results are bit-identical to the no-autocast path.

        Args:
            x0_fn: the x0 source — ``x0_fn(z, t) → x0``, the clean-latent prediction
                (the injected seam: UNet direct, CFG-wrapped, or ControlNet
                residual forward, per caller).
            z_start: the rollout's start latent ``[B, C, D, H, W]`` (pure noise at
                ``t = 0``, or the caller's noised start for the partial path — the
                ``add_noise`` start stays with the caller).
            grid: the integration nodes — ``(n+1,)`` (scalar full-range,
                :meth:`set_timesteps` product) or ``(B, n+1)`` (per-sample partial,
                :meth:`PartialFlowMatchHeunScheduler.set_timesteps_partial` product).
            grad: ``"inference"`` or ``"no_grad"`` (the two-state grad context).

        Returns:
            The denoised latent ``[B, C, D, H, W]``.
        """
        if grad not in ("inference", "no_grad"):
            raise ValueError(
                f"grad must be 'inference' or 'no_grad' (no third state — a "
                f"full-grad Heun rollout does not exist), got {grad!r}."
            )
        device = z_start.device
        ctx = torch.inference_mode() if grad == "inference" else torch.no_grad()
        n = grid.shape[-1] - 1  # steps: a (n+1,) scalar grid or a (B, n+1) per-sample grid
        z = z_start
        with ctx:
            # Autocast the Heun rollout on cuda (mirrors sample_latent_flow); disabled
            # off-cuda, so CPU results are bit-identical to the no-autocast path.
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                for i in range(n):
                    t = grid[..., i]  # scalar (full-range) or (B,) (per-sample)
                    t_next = grid[..., i + 1]
                    x0_1 = x0_fn(z, t)
                    z_euler, v1 = self.euler_step(x0_1, z, t, t_next)
                    if i == n - 1:
                        # Final step is Euler: at t_next = 1 the denominator 1 − t_next
                        # vanishes, so the second Heun evaluation is undefined.
                        z = z_euler
                    else:
                        x0_2 = x0_fn(z_euler, t_next)
                        z = self.heun_correct(x0_2, z, z_euler, v1, t, t_next)
        return z
