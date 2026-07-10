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
