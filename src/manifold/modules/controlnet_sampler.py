"""The ControlNet x0 Heun rollout — a single shared noise→data primitive (ADR-0005).

A sibling of :func:`manifold.modules.sampler.sample_latent_flow` for paired MRI
translation on a **frozen noise→data base UNet + a trainable ControlNet**
(ADR-0026/0027). The transport is the **canonical noise→data** rectified flow
(``z = t·x_tgt + (1−t)·ε``, ``t: 0 → 1`` from pure noise to clean data) — identical
to unconditional JiT generation. The source latent ``x_src`` enters only as a
**control signal** to the ControlNet, never as a transport endpoint: at every Heun
evaluation point the ControlNet consumes ``(z_t, t, spacing, class_labels_src,
class_labels_tgt, controlnet_cond=x_src)`` and emits residual injections, which the
frozen base UNet then consumes through its native
``down_block_additional_residuals`` / ``mid_block_additional_residual`` forward
args to produce the x0 prediction.

This is the single source of truth for the ControlNet rollout: the inference
:class:`~manifold.pipelines.ControlNetLatentFlowPipeline`, the supervised
:class:`~manifold.modules.ControlNetLatentFlowModule` validation, the GRPO Mode-2
suffix, and the reward fake-builder all delegate here, so the paths cannot drift.

Two regimes share the per-step injection helper (never a duplicated Heun loop):

- :func:`controlnet_rollout` — the full ``0 → 1`` rollout from pure noise (the
  base :class:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler` batch-wide
  grid), used by the pipeline / supervised-module validation / GRPO suffix; and
- :func:`controlnet_partial_rollout` — the per-sample ``t_start → 1`` rollout over
  the :class:`~manifold.schedulers.PartialFlowMatchHeunScheduler` grid (the reward
  generated-end probe, ADR-0023 analogue): the corrupt endpoint is the *noise*
  side of the noise→data transport, so ``z_start = add_noise(x_tgt, ε, t_start)``
  with fresh Gaussian ``ε`` (higher ``t_start`` ⇒ nearer the real target ⇒
  higher-quality fake ⇒ the probe's winner is the higher-``t`` sample).

The direction conditioning (``class_labels_src`` / ``class_labels_tgt``) feeds the
ControlNet's direction MLP; the frozen base separately receives only the
target-contrast ``class_labels`` for its own modality embedding. Deterministic
given the noise (``controlnet_rollout``) or the seeded ``ε`` (partial), matching
the noise→data generator's re-seeded reproducibility (ADR-0027).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..schedulers.scheduling_partial_flow_match_heun import PartialFlowMatchHeunScheduler
from .paired_sampler import _as_label_tensor


def _controlnet_x0(
    unet,
    controlnet,
    z: Tensor,
    t,
    x_src: Tensor,
    spacing_t: Tensor,
    src_labels: Tensor,
    tgt_labels: Tensor,
) -> Tensor:
    """One ControlNet-conditioned x0 evaluation at flow-time ``t``.

    Runs the ControlNet to obtain the residual injections, then the frozen base
    UNet with those residuals. The ControlNet carries the (src, tgt) direction
    pair; the base receives only the target-contrast label. ``t`` may be a scalar
    (the full rollout's batch-wide node) or a ``(B,)`` tensor (the partial
    rollout's per-sample node) — both the ControlNet and the base scale it.
    """
    down_res, mid_res = controlnet(
        sample=z,
        controlnet_cond=x_src,
        timestep=t,
        spacing=spacing_t,
        class_labels_src=src_labels,
        class_labels_tgt=tgt_labels,
    )
    return unet(
        sample=z,
        timestep=t,
        spacing=spacing_t,
        class_labels=tgt_labels,
        down_block_additional_residuals=down_res,
        mid_block_additional_residual=mid_res,
    )


def controlnet_rollout(
    unet,
    controlnet,
    scheduler,
    noise: Tensor,
    x_src: Tensor,
    spacing: Tensor | Sequence[float],
    src_label: int | Tensor,
    tgt_label: int | Tensor,
    *,
    num_inference_steps: int,
) -> Tensor:
    """Run the ControlNet noise→data Heun rollout from pure noise → target latent.

    The single shared ControlNet x0 Heun rollout (ADR-0005 analogue) over the
    **canonical noise→data transport**: starts at ``z_0 = noise`` (Gaussian, the
    ``t = 0`` endpoint) and integrates ``t: 0 → 1`` on the base scheduler's
    batch-wide grid, injecting the ControlNet residuals at every evaluation. The
    source ``x_src`` is the control signal only — it is not a transport endpoint.

    Puts the UNet + ControlNet in ``eval()`` and runs under ``inference_mode`` +
    cuda autocast (disabled off-cuda, so CPU results are bit-identical to the
    no-autocast path), mirroring :func:`sample_latent_flow`.

    Args:
        unet: the frozen noise→data base UNet (predicts x0).
        controlnet: the trainable :class:`~manifold.ControlNet3DConditionModel`.
        scheduler: the :class:`FlowMatchHeunDiscreteScheduler`; its ``t_eps`` is
            the Heun endpoint clamp, and its ``set_timesteps`` / ``euler_step`` /
            ``heun_correct`` run.
        noise: the pure-noise start latent ``[B, C_latent, D, H, W]`` — the
            stochastic input (callers build it from a generator).
        x_src: the source latent control signal ``[B, C_latent, D, H, W]``
            (already scaled into the base's training space).
        spacing: raw voxel spacing ``[3]`` or ``[B, 3]`` (scaled ×1e2 internally).
        src_label / tgt_label: the (src, tgt) contrast pair. Scalar ``int``
            (broadcast — one direction per call) or ``[B]`` long tensor of
            per-sample labels (validation mixes directions).
        num_inference_steps: Heun integration steps over ``t: 0 → 1``.

    Returns:
        The predicted target latent ``[B, C_latent, D, H, W]``.
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    batch_size = noise.shape[0]

    spacing_t = torch.as_tensor(spacing, device=device)
    src_labels = _as_label_tensor(src_label, batch_size, device)
    tgt_labels = _as_label_tensor(tgt_label, batch_size, device)

    x_src_dev = x_src.to(device=device, dtype=dtype)

    def unet_call(z: Tensor, t: float) -> Tensor:
        return _controlnet_x0(
            unet, controlnet, z, float(t), x_src_dev, spacing_t, src_labels, tgt_labels
        )

    z = noise.to(device=device, dtype=dtype)
    nodes = scheduler.set_timesteps(num_inference_steps, device=device)
    n = int(num_inference_steps)

    unet.eval()
    controlnet.eval()
    with torch.inference_mode():
        # Autocast the Heun rollout on cuda (mirrors sample_latent_flow); disabled
        # off-cuda, so CPU results are bit-identical to the no-autocast path.
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for i in range(n):
                t = float(nodes[i])
                t_next = float(nodes[i + 1])
                x0_1 = unet_call(z, t)
                z_euler, v1 = scheduler.euler_step(x0_1, z, t, t_next)
                if i == n - 1:
                    # Final step is Euler: at t_next = 1 the denominator 1 − t_next
                    # vanishes, so the second Heun evaluation is undefined.
                    z = z_euler
                else:
                    x0_2 = unet_call(z_euler, t_next)
                    z = scheduler.heun_correct(x0_2, z, z_euler, v1, t, t_next)
    return z


def controlnet_partial_rollout(
    unet,
    controlnet,
    scheduler: PartialFlowMatchHeunScheduler,
    x_src: Tensor,
    x_tgt: Tensor,
    t_start: Tensor,
    spacing: Tensor | Sequence[float],
    src_label: int | Tensor,
    tgt_label: int | Tensor,
    *,
    num_steps: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Partial ControlNet noise→data Heun rollout from per-sample ``t_start`` → clean.

    The ControlNet reward generated-end **probe** primitive (ADR-0023 analogue) —
    the single-source-of-truth sibling of :func:`controlnet_rollout` for the
    partial regime. On the noise→data transport the corrupt endpoint is the
    **noise** side, so the rollout starts from
    ``z_start = add_noise(x_tgt, ε, t_start) = t_start·x_tgt + (1−t_start)·ε``
    with fresh Gaussian ``ε`` (drawn from *generator* for seeded determinism).
    Higher ``t_start`` ⇒ ``z`` nearer the real target ⇒ a higher-quality fake, so
    the probe's **winner is the higher-``t``** sample. Each sample integrates its
    own ``[t_start, 1]`` range on the per-sample grid
    (:meth:`PartialFlowMatchHeunScheduler.set_timesteps_partial`).

    The ControlNet conditions on ``x_src`` (the control signal) and the (src, tgt)
    direction pair at every evaluation, exactly as in the full rollout. Runs under
    ``inference_mode`` (the probe is precomputed once to disk and scored
    forward-only), mirroring :func:`controlnet_rollout`.

    Args:
        unet: the frozen noise→data base UNet (predicts x0).
        controlnet: the trainable :class:`~manifold.ControlNet3DConditionModel`.
        scheduler: a :class:`PartialFlowMatchHeunScheduler`; its
            :meth:`set_timesteps_partial` builds the per-sample grid and its
            (inherited) ``add_noise`` / ``euler_step`` / ``heun_correct`` run.
        x_src: the source latent control signal ``[B, C_latent, D, H, W]``.
        x_tgt: the real target latent ``[B, C_latent, D, H, W]`` (the ``t = 1``
            endpoint); the rollout's far end.
        t_start: ``(B,)`` flow-times in ``[0, 1)`` — each sample's start (the
            probe caps ``t_start ∈ [0, 0.5)`` so samples stay genuinely fake).
        spacing: raw voxel spacing ``[3]`` or ``[B, 3]`` (scaled ×1e2 internally).
        src_label / tgt_label: the (src, tgt) contrast pair (scalar broadcast or
            ``[B]`` per-sample).
        num_steps: Heun steps over each sample's ``[t_start, 1]`` range (shared
            budget; per-sample ``δt`` differs).
        generator: optional :class:`torch.Generator` for the corruption noise
            ``ε`` (seeded ⇒ a reproducible probe).

    Returns:
        The generated target latent ``[B, C_latent, D, H, W]``.
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    batch_size = x_src.shape[0]
    if t_start.shape[0] != batch_size:
        raise ValueError(
            f"x_src batch ({batch_size}) != t_start ({t_start.shape[0]}); "
            "the probe batch must align."
        )

    spacing_t = torch.as_tensor(spacing, device=device)
    if spacing_t.dim() == 2 and spacing_t.shape[0] != batch_size:
        raise ValueError(
            f"per-sample spacing rows ({spacing_t.shape[0]}) != batch ({batch_size}); "
            "a [B,3] spacing must match the batch size."
        )
    src_labels = _as_label_tensor(src_label, batch_size, device)
    tgt_labels = _as_label_tensor(tgt_label, batch_size, device)

    x_src_dev = x_src.to(device=device, dtype=dtype)
    x_tgt_dev = x_tgt.to(device=device, dtype=dtype)
    # z_start = t_start·x_tgt + (1−t_start)·ε (the noise→data transport); the
    # corrupt endpoint is Gaussian ε (drawn from the seeded generator when given).
    eps = torch.randn(x_tgt_dev.shape, generator=generator, device=device, dtype=dtype)
    z = scheduler.add_noise(x_tgt_dev, eps, t_start.to(device=device, dtype=dtype))
    nodes = scheduler.set_timesteps_partial(t_start, num_steps, device=device)  # (B, n+1)

    def unet_call(z_t: Tensor, t: Tensor) -> Tensor:
        return _controlnet_x0(
            unet, controlnet, z_t, t, x_src_dev, spacing_t, src_labels, tgt_labels
        )

    unet.eval()
    controlnet.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for i in range(num_steps):
                t = nodes[:, i]  # (B,)
                t_next = nodes[:, i + 1]  # (B,)
                x0_1 = unet_call(z, t)
                z_euler, v1 = scheduler.euler_step(x0_1, z, t, t_next)
                if i == num_steps - 1:
                    # Final step is Euler: at t_next = 1 the denominator 1 − t_next
                    # vanishes, so the second Heun evaluation is undefined.
                    z = z_euler
                else:
                    x0_2 = unet_call(z_euler, t_next)
                    z = scheduler.heun_correct(x0_2, z, z_euler, v1, t, t_next)
    return z


__all__ = ["controlnet_partial_rollout", "controlnet_rollout"]
