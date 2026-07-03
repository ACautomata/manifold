"""The partial-denoise Heun rollout — a per-sample ``t_start`` primitive (ADR-0005/0008).

The reward analogue of :func:`~manifold.modules.sample_latent_flow`: denoise a
latent from a **per-sample** flow-time ``t_start`` to clean (``t = 1``) under the
true two-evaluation Heun, using the
:class:`~manifold.PartialFlowMatchHeunScheduler`'s per-sample grid and the
(shared) Heun step math. Module-owned, scheduler-delegated (ADR-0005): the grid
(:meth:`set_timesteps_partial`) and the steps
(:meth:`~manifold.FlowMatchHeunDiscreteScheduler.euler_step` / :meth:`heun_correct`,
inherited verbatim) are the scheduler's; only the loop lives here. The denoiser is
the frozen JiT export (ADR-0006) — the GRPO starting policy — run under
``inference_mode`` so no state is captured.

The caller noises the clean latent via the scheduler's transport
(``scheduler.add_noise(clean, noise, t_start)``) so the rollout's start matches
the UNet's training distribution exactly (ADR-0001).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from ..schedulers.scheduling_partial_flow_match_heun import PartialFlowMatchHeunScheduler


def partial_denoise_rollout(
    unet,
    scheduler: PartialFlowMatchHeunScheduler,
    z_start: Tensor,
    t_start: Tensor,
    spacing: Tensor | Sequence[float],
    modality: int,
    *,
    num_steps: int,
) -> Tensor:
    """Denoise from per-sample ``t_start`` → clean under the true two-evaluation Heun.

    Args:
        unet: the frozen JiT x0-denoiser (predicts the clean latent).
        scheduler: a :class:`PartialFlowMatchHeunScheduler`; its
            :meth:`set_timesteps_partial` builds the per-sample grid and its
            (inherited) ``euler_step`` / ``heun_correct`` run.
        z_start: the noised latent ``[B, C, D, H, W]`` at flow-time ``t_start``
            (callers noise via ``scheduler.add_noise``).
        t_start: ``(B,)`` flow-times — each sample's corruption level.
        spacing: raw voxel spacing ``[3]`` or ``[B, 3]`` (scaled ×1e2 in the UNet).
        modality: integer class label (broadcast across the batch) **or** a
            per-sample ``(B,)`` long tensor (a heterogeneous, multi-contrast cache).
        num_steps: Heun steps over each sample's ``[t_start, 1]`` range (shared
            budget; per-sample ``δt`` differs).

    Returns:
        The denoised latent ``[B, C, D, H, W]``.
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    batch_size = z_start.shape[0]

    spacing_t = torch.as_tensor(spacing, device=device)
    # modality may be a scalar (broadcast across the batch) or a per-sample
    # ``(B,)`` long tensor (a heterogeneous, multi-contrast cache). spacing may be
    # ``[3]`` (broadcast) or ``[B, 3]`` (per-sample) — the UNet wrapper handles both.
    if isinstance(modality, Tensor):
        class_labels = modality.to(device=device, dtype=torch.long)
    else:
        class_labels = torch.full((batch_size,), int(modality), dtype=torch.long, device=device)
    nodes = scheduler.set_timesteps_partial(t_start, num_steps, device=device)  # (B, n+1)

    z = z_start.to(device=device, dtype=dtype)
    unet.eval()
    with torch.inference_mode():
        # Autocast the Heun rollout on cuda (mirrors sample_latent_flow); disabled
        # off-cuda so CPU results are bit-identical to the no-autocast path.
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for i in range(num_steps):
                t = nodes[:, i]  # (B,)
                t_next = nodes[:, i + 1]  # (B,)
                x0_1 = unet(sample=z, timestep=t, spacing=spacing_t, class_labels=class_labels)
                z_euler, v1 = scheduler.euler_step(x0_1, z, t, t_next)
                if i == num_steps - 1:
                    # Final step is Euler: at t_next = 1 the denominator 1 − t_next
                    # vanishes, so the second Heun evaluation is undefined.
                    z = z_euler
                else:
                    x0_2 = unet(
                        sample=z_euler, timestep=t_next, spacing=spacing_t, class_labels=class_labels
                    )
                    z = scheduler.heun_correct(x0_2, z, z_euler, v1, t, t_next)
    return z
