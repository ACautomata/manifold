"""ControlNet-GRPO input-assembly helpers.

Owns the frozen-ControlNet-generator loader consumed by the GRPO real-input path
(the ControlNet-GRPO policy loader, :func:`~manifold.training.grpo_cli._controlnet_real_inputs`
— the ControlNet path of the unified builder). Relocated from the now-deleted
``manifold.data.paired_reward_pairs`` (issue #176, ADR-0034) ahead of the paired-reward
pairs builder's deletion (#174) so dropping it did not take the loader with it; the
loader's only consumer is this GRPO real-input path.

The loader assumes the native dir is a ControlNet export (it accesses the
pipeline's ``controlnet`` component directly) and therefore **cannot** be used as
the discriminator for "is this a ControlNet export?" - the unified builder builds
that discriminator separately (:func:`~manifold.training.grpo_cli._detect_controlnet_export`).
"""

from __future__ import annotations

from pathlib import Path

from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler

__all__ = ["load_frozen_controlnet_generator"]


def load_frozen_controlnet_generator(native_dir: str | Path):
    """Load the frozen ControlNet generator from a ControlNet native export.

    The ControlNet counterpart of the (deleted) paired-JiT generator loader
    (ADR-0027/T7). The native dir is the layout written by
    :meth:`~manifold.ControlNetLatentFlowPipeline.save_pretrained` (the raw arm, no
    EMA). The generator is the **supervised ControlNet's** noise->data policy: a
    **frozen base UNet** + a **frozen ControlNet** whose residuals steer it. Its
    only consumer is the GRPO real-input ControlNet path
    (:func:`~manifold.training.grpo_cli._controlnet_real_inputs`), which loads both
    arms frozen and then unfreezes the ControlNet as the only trainable arm.

    - The scheduler is the **base** :class:`FlowMatchHeunDiscreteScheduler` (the
      ControlNet-path generation is a full ``0 -> 1`` rollout), NOT the Partial
      subclass.
    - Both arms come back frozen + eval + grad-disabled (the caller unfreezes the
      ControlNet for training). The VAE's ``scaling_factor`` is returned so
      callers scale the raw paired-cache src latents into the generator's training
      space (ADR-0021).

    Returns:
        ``(base_unet, controlnet, scheduler, scaling_factor)`` - the frozen + eval
        base UNet, the frozen ControlNet, the base scheduler, and the VAE scaling
        factor.
    """
    from ..pipelines.controlnet_latent_flow import ControlNetLatentFlowPipeline

    pipe = ControlNetLatentFlowPipeline.from_pretrained(str(native_dir))
    # The base scheduler (NOT the Partial subclass): the ControlNet-path generation
    # is a full 0->1 rollout.
    scheduler = FlowMatchHeunDiscreteScheduler(**pipe.scheduler.config)
    scaling_factor = float(pipe.vae.scaling_factor)
    pipe.unet.eval()
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    pipe.controlnet.eval()
    for p in pipe.controlnet.parameters():
        p.requires_grad_(False)
    return pipe.unet, pipe.controlnet, scheduler, scaling_factor
