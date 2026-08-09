"""The x0 Heun rollout — a single shared primitive (ADR-0005).

The true two-evaluation Heun reverse step (predictor at ``z_t``, Euler-advance,
corrector at ``z_{t+dt}``; Euler on the final step where ``1 − t_next`` vanishes)
is the single source of truth for x0-latent generation.
:meth:`LatentFlowModule.sample` (in-training generation — the FID callback) and
:meth:`LatentFlowPipeline.sample_latent` (inference) both delegate here, so the
train and infer paths cannot drift. Sampler parity against the upstream ``hope``
implementation was verified during migration (``tests/parity/validate_against_hope.py``).

The mechanism is unchanged from ADR-0002: the scheduler's
:meth:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler.euler_step` /
:meth:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler.heun_correct` run
under ``inference_mode`` + cuda autocast, with
denoising-interval classifier-free guidance wrapping the UNet calls.
``guidance_scale == 1.0`` short-circuits to the conditional output — the no-CFG
path is bit-identical to a single conditional forward.

Two refinements of the issue's prose signature, both in the service of the
single-source-of-truth invariant and ADR-0001 (the scheduler owns the transport
and reverse step):

- the primitive takes the **materialized** ``noise`` (the stochastic input)
  rather than a ``generator`` — each caller builds the noise from its own
  generator, so the rollout itself is deterministic given the noise; and
- it takes **no separate** ``t_eps``: the Heun endpoint clamp is the scheduler's
  (``scheduler.t_eps``), owned there alongside the transport (so it is not a
  rollout parameter). ``Module.sample`` and ``Pipeline.sample_latent`` both
  reach the clamp through the scheduler they already hold.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def sample_latent_flow(
    unet,
    scheduler,
    noise: Tensor,
    spacing: Tensor | Sequence[float],
    modality: int,
    *,
    num_inference_steps: int,
    guidance_scale: float = 1.0,
    cfg_interval: tuple[float, float] | None = None,
) -> Tensor:
    """Run the true two-evaluation Heun rollout from pure noise → final latent.

    The single shared x0 Heun rollout (ADR-0005). ``Module.sample`` and
    ``Pipeline.sample_latent`` both call this; because the Module's
    training-time generation and the Pipeline share one rollout, the rollout
    uses whatever weights the caller passes (the Module passes live optimizer
    weights; the inference Pipeline passes its loaded weights) - no EMA shadow is
    ever swapped into ``unet``.

    The rollout puts the UNet in ``eval()`` and runs under ``inference_mode`` +
    cuda autocast so the latent-trajectory tolerance for numerical validation is
    achievable (disabled off-cuda, so CPU results are bit-identical to the
    no-autocast path).

    Args:
        unet: the x0-denoiser UNet (predicts the clean latent).
        scheduler: the :class:`FlowMatchHeunDiscreteScheduler`; its ``t_eps`` is
            the Heun endpoint clamp, and its ``set_timesteps`` /
            ``euler_step`` / ``heun_correct`` run.
        noise: the pure-noise start latent ``[B, C, D, H, W]`` — the stochastic
            input (callers build it from a generator).
        spacing: raw voxel spacing ``[3]`` or ``[B, 3]`` (scaled ×1e2 in the UNet).
        modality: the integer class label for conditioning.
        num_inference_steps: Heun integration steps over ``t: 0 → 1``.
        guidance_scale: CFG strength; ``1.0`` (the default) runs the no-CFG path
            exactly. Values ``!= 1.0`` apply CFG inside ``cfg_interval`` only.
        cfg_interval: ``(low, high)`` — guidance is active for flow-times ``t``
            with ``low < t < high``. ``None`` (the default) disables CFG.

    Returns:
        The final latent ``[B, C, D, H, W]``.
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    batch_size = noise.shape[0]

    spacing_t = torch.as_tensor(spacing, device=device)
    class_labels = torch.full((batch_size,), int(modality), dtype=torch.long, device=device)

    def unet_call(z: Tensor, t: float) -> Tensor:
        """One (possibly CFG-combined) UNet evaluation at flow-time ``t``.

        ``guidance_scale == 1.0`` short-circuits to the conditional output — no
        unconditional forward, so the ``= 1`` path is bit-identical to the
        no-CFG path. With an active interval the label is dropped for the
        unconditional pass and the two are combined; outside the interval the
        conditional output is used directly.
        """
        cond = unet(sample=z, timestep=float(t), spacing=spacing_t, class_labels=class_labels)
        if guidance_scale == 1.0 or cfg_interval is None:
            return cond
        low, high = cfg_interval
        if not (low < t < high):
            return cond
        uncond = unet(
            sample=z,
            timestep=float(t),
            spacing=spacing_t,
            class_labels=torch.zeros_like(class_labels),
        )
        return uncond + guidance_scale * (cond - uncond)

    z = noise.to(device=device, dtype=dtype)
    nodes = scheduler.set_timesteps(num_inference_steps, device=device)

    unet.eval()
    # The shared heun_rollout loop (the ADR-0005 convergence point for the
    # full-range / per-sample rollouts) runs the Euler→guard→Heun cycle with
    # the CFG-wrapped unet_call as the injected x0_fn; cuda autocast on /
    # off-cuda disabled (CPU results bit-identical to the pre-primitive path).
    return scheduler.heun_rollout(unet_call, z, nodes, grad="inference")
