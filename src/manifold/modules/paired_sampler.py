"""The Paired JiT x0 Heun rollout — a single shared start-from-src primitive.

A sibling of :func:`manifold.modules.sampler.sample_latent_flow` for src→tgt
translation. The same two-evaluation Heun reverse step runs over the **same**
:class:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler` transport and
integrator (ADR-0013 — the scheduler class is reused unchanged); the only
differences are:

- the rollout **starts from the source latent** ``x_src`` (the ``t = 0`` endpoint is
  a data latent, not Gaussian noise) — ``z_0 = x_src``;
- the UNet sees ``concat([z_t, x_src])`` at every evaluation, so the model can
  disambiguate the mid-``t`` mix (ADR-0014); and
- the conditioning is the summed contrast embedding ``embed(src) + embed(tgt)``,
  fed through the wrapper's paired pathway.

Because the transport is deterministic given ``x_src`` (no stochastic input), the
rollout is reproducible with no generator / re-seeding — distinct from the
noise→data generator, which starts from sampled noise.

Like :func:`sample_latent_flow`, this is the single source of truth for the Paired
JiT rollout (the ADR-0005 analogue): the future ``PairedLatentFlowModule.sample``
(in-training generation — the PSNR/SSIM callback) and
:meth:`PairedLatentFlowPipeline.sample_latent` both delegate here, so the train and
infer paths cannot drift.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..schedulers.scheduling_partial_flow_match_heun import PartialFlowMatchHeunScheduler


def sample_paired_latent_flow(
    unet,
    scheduler,
    x_src: Tensor,
    spacing: Tensor | Sequence[float],
    src_label: int | Tensor,
    tgt_label: int | Tensor,
    *,
    num_inference_steps: int,
) -> Tensor:
    """Run the two-evaluation Heun rollout from ``x_src`` → the target latent.

    The single shared Paired JiT x0 Heun rollout (ADR-0005 analogue). Starts at
    ``z_0 = x_src``, concats ``x_src`` into every UNet evaluation, and runs the
    scheduler's ``set_timesteps`` / ``euler_step`` / ``heun_correct`` over
    ``t: 0 → 1``. Puts the UNet in ``eval()`` and runs under ``inference_mode`` +
    cuda autocast (disabled off-cuda, so CPU results are bit-identical to the
    no-autocast path), mirroring :func:`sample_latent_flow`.

    Args:
        unet: the Paired JiT UNet (``in_channels = 2·C_latent``); predicts x_tgt.
        scheduler: the :class:`FlowMatchHeunDiscreteScheduler` (shared, not a new
            class — ADR-0013); its ``t_eps`` is the Heun endpoint clamp.
        x_src: the source latent ``[B, C_latent, D, H, W]`` — the ``t = 0``
            endpoint and the rollout's starting point.
        spacing: raw voxel spacing ``[3]`` or ``[B, 3]`` (scaled ×1e2 in the UNet).
        src_label / tgt_label: the contrast labels whose embeddings are summed for
            the translation direction (ADR-0014). Either a scalar ``int`` broadcast
            across the batch (the inference contract — one direction per call) or a
            ``[B]`` long tensor of per-sample labels. Per-sample labels are required
            for validation, whose batch mixes all 12 within-subject contrast
            directions: a scalar would condition every sample on sample 0's
            direction. The UNet wrapper accepts ``[B]`` labels (training forwards
            ``batch["src_label"]`` directly), so a tensor is passed through unchanged.
        num_inference_steps: Heun integration steps over ``t: 0 → 1``.

    Returns:
        The predicted target latent ``[B, C_latent, D, H, W]``.
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    batch_size = x_src.shape[0]

    spacing_t = torch.as_tensor(spacing, device=device)
    src_labels = _as_label_tensor(src_label, batch_size, device)
    tgt_labels = _as_label_tensor(tgt_label, batch_size, device)

    x_src_dev = x_src.to(device=device, dtype=dtype)

    def unet_call(z: Tensor, t: float) -> Tensor:
        """One UNet evaluation at flow-time ``t`` on ``concat([z, x_src])``."""
        sample = torch.cat([z, x_src_dev], dim=1)
        return unet(
            sample=sample,
            timestep=float(t),
            spacing=spacing_t,
            class_labels_src=src_labels,
            class_labels_tgt=tgt_labels,
        )

    z = x_src_dev
    nodes = scheduler.set_timesteps(num_inference_steps, device=device)
    n = int(num_inference_steps)

    unet.eval()
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


def partial_paired_rollout(
    unet,
    scheduler: PartialFlowMatchHeunScheduler,
    x_src: Tensor,
    x_tgt: Tensor,
    t_start: Tensor,
    spacing: Tensor | Sequence[float],
    src_label: int | Tensor,
    tgt_label: int | Tensor,
    *,
    num_steps: int,
) -> Tensor:
    """Partial src->tgt Heun rollout from per-sample ``t_start`` -> clean (``t = 1``).

    The paired reward's generated-end **probe** primitive (ADR-0023) - the
    single-source-of-truth (ADR-0005) sibling of :func:`sample_paired_latent_flow`
    for the *partial* regime. Two changes from the full rollout:

    - the rollout **starts from** ``z = scheduler.add_noise(x_tgt, x_src, t_start)``
      (= ``t_start·x_tgt + (1−t_start)·x_src``), NOT ``x_src``. Higher ``t_start``
      ⇒ ``z`` nearer the real target (less translation needed) ⇒ a higher-quality
      fake - so the probe's **winner is the higher-``t``** sample (ADR-0023);
    - per-sample ``(B,)`` flow-times from
      :meth:`PartialFlowMatchHeunScheduler.set_timesteps_partial` (each sample
      integrates its own ``[t_start, 1]`` range), NOT the batch-wide scalar
      ``nodes[i]`` of the full rollout.

    The UNet still sees ``concat([z_t, x_src])`` at every evaluation and the summed
    contrast embedding ``embed(src) + embed(tgt)`` (ADR-0014) - the paired
    conditioning is unchanged. ``add_noise`` and ``set_timesteps_partial`` are
    endpoint-agnostic (the JiT ``src=clean``/``tgt=endpoint`` split is reused for
    the src/tgt transport here verbatim), so no scheduler math is forked.

    Runs under ``inference_mode`` (mirroring :func:`sample_paired_latent_flow`): the
    probe is precomputed once to disk (ADR-0020) and scored forward-only in
    validation, so no downstream ``backward`` needs a flag-cleared tensor (unlike
    the JiT online rollout, which uses ``no_grad`` for that reason - ADR-0023).

    Args:
        unet: the Paired JiT UNet (``in_channels = 2·C_latent``); predicts x_tgt.
        scheduler: a :class:`PartialFlowMatchHeunScheduler`; its
            :meth:`set_timesteps_partial` builds the per-sample grid and its
            (inherited) ``euler_step`` / ``heun_correct`` run the steps.
        x_src: the source latent ``[B, C_latent, D, H, W]`` (the ``t = 0`` endpoint
            and the concat conditioning - already scaled into the UNet's space).
        x_tgt: the real target latent ``[B, C_latent, D, H, W]`` (the ``t = 1``
            endpoint - already scaled); the rollout's far end.
        t_start: ``(B,)`` flow-times in ``[0, 1)`` - each sample's start (the probe
            caps ``t_start ∈ [0, 0.5)`` so samples stay genuinely fake, ADR-0023).
        spacing: raw voxel spacing ``[3]`` or ``[B, 3]`` (scaled ×1e2 in the UNet).
        src_label / tgt_label: the contrast labels whose embeddings are summed for
            the translation direction (ADR-0014). Scalar ``int`` (broadcast) or
            ``[B]`` long tensor of per-sample labels (a val batch mixes directions).
        num_steps: Heun steps over each sample's ``[t_start, 1]`` range (shared
            budget; per-sample ``δt`` differs).

    Returns:
        The generated target latent ``[B, C_latent, D, H, W]``.
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    batch_size = x_src.shape[0]
    # Surface a mismatched batch as a clear error (not a MAISI-internal crash):
    # the probe issues one [B] batch; a z / t_start / per-sample-spacing mismatch
    # would otherwise blow up deep inside the UNet.
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
    # z_start = t_start·x_tgt + (1−t_start)·x_src (the scheduler transport); at
    # t_start = 0 this is x_src (the full rollout's start), at t_start -> 1 it is x_tgt.
    z = scheduler.add_noise(x_tgt_dev, x_src_dev, t_start.to(device=device, dtype=dtype))
    nodes = scheduler.set_timesteps_partial(t_start, num_steps, device=device)  # (B, n+1)

    def unet_call(z_t: Tensor, t: Tensor) -> Tensor:
        """One UNet evaluation at per-sample flow-time ``t`` on ``concat([z_t, x_src])``."""
        sample = torch.cat([z_t, x_src_dev], dim=1)
        return unet(
            sample=sample,
            timestep=t,
            spacing=spacing_t,
            class_labels_src=src_labels,
            class_labels_tgt=tgt_labels,
        )

    unet.eval()
    with torch.inference_mode():
        # Autocast the Heun rollout on cuda (mirrors sample_paired_latent_flow);
        # disabled off-cuda, so CPU results are bit-identical to the no-autocast path.
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


def _as_label_tensor(labels: int | Tensor, batch_size: int, device: torch.device) -> Tensor:
    """Coerce ``labels`` to a ``[batch_size]`` long tensor on ``device``.

    A scalar — a Python ``int`` or a 0-d tensor like ``torch.tensor(0)`` — is
    broadcast (the inference contract — one direction per call); a ``[B]`` tensor
    of per-sample labels is passed through unchanged (the validation contract — a
    val batch mixes all 12 contrast directions). Fails fast on a tensor whose
    length disagrees with the batch — a silent broadcast there would condition
    samples on the wrong contrast.
    """
    if torch.is_tensor(labels):
        # A 0-d tensor is a scalar — broadcast it (preserves the prior
        # ``int(src_label)`` behavior for scalar-as-tensor callers).
        if labels.ndim == 0:
            return torch.full((batch_size,), int(labels.item()), dtype=torch.long, device=device)
        if labels.shape != (batch_size,):
            raise ValueError(
                f"per-sample label tensor shape {tuple(labels.shape)} != batch_size "
                f"{batch_size}; pass a scalar int to broadcast one direction, or a "
                f"[B] tensor of per-sample labels."
            )
        return labels.to(device=device, dtype=torch.long)
    return torch.full((batch_size,), int(labels), dtype=torch.long, device=device)
