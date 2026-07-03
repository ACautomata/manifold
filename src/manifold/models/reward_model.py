"""Reward Model: a MONAI PatchGAN discriminator behind a mean-pooling head (GRPO).

A diffusers-style :class:`~manifold.ModelMixin` wrapper around MONAI's
:class:`~monai.networks.nets.PatchDiscriminator` (3D, latent-channel input). The
reward is the per-sample mean of the discriminator's final **raw** patch-logit
map — no final sigmoid (the MONAI head is already ``conv_only``; σ appears only
inside the Bradley–Terry training loss, so the model emits a raw scalar whose
*differences* are calibrated by the loss). The reward scores latent quality
unconditionally and is never decoded to image space.

This is a **Model** in the four-component sense (a network wrapper specialized to
scoring); GRPO calls its :meth:`forward` on policy rollouts for the advantage
signal. It owns no generation rollout.
"""

from __future__ import annotations

from monai.networks.nets import PatchDiscriminator
from torch import Tensor

from ..configuration import register_to_config
from .modeling_utils import ModelMixin


class RewardModel(ModelMixin):
    """PatchGAN reward scorer: latent ``[B, C, D, H, W]`` → per-sample reward ``[B]``.

    The reward is the mean of the discriminator's final patch-logit map per
    sample (no sigmoid) — a raw realism/fidelity score. Construction mirrors the
    MONAI PatchDiscriminator knobs; ``channels`` is the first-layer filter count
    (doubled per subsequent layer, per MONAI's convention).
    """

    @register_to_config
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 4,
        channels: int = 64,
        num_layers_d: int = 3,
        out_channels: int = 1,
        norm: str = "BATCH",
    ):
        """Args:
            spatial_dims: spatial rank of the latent (3 for volumetric latents).
            in_channels: latent channel count (``latent_channels``).
            channels: first conv filter count (doubled per layer — MONAI convention).
            num_layers_d: number of strided conv layers between the initial and
                final conv (the PatchGAN depth).
            out_channels: final-conv output channels (1 = one patch-logit map).
            norm: normalization in the middle conv layers (MONAI default ``BATCH``).
        """
        super().__init__()
        self.num_layers_d = int(num_layers_d)
        self.discriminator = PatchDiscriminator(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            out_channels=out_channels,
            num_layers_d=num_layers_d,
            norm=norm,
        )

    def forward(self, latent: Tensor) -> Tensor:
        """Score each latent → per-sample mean patch-logit reward ``[B]`` (no sigmoid).

        The final element of the PatchGAN output list is the raw patch-logit map
        ``[B, out_channels, ...]``; its per-sample spatial mean is the scalar
        reward. Mean-pooling (not a learned head) keeps the reward a direct
        average of the patch realism signal.
        """
        try:
            patch_map = self.discriminator(latent)[-1]  # [B, out_channels, *spatial]
        except (RuntimeError, ValueError) as exc:  # pragma: no cover - size-collapse is config-dependent
            # PatchGAN spatial collapse surfaces as either a conv "Kernel size >
            # input size" RuntimeError or a BatchNorm "more than 1 value per
            # channel" ValueError (a 1×1×1 patch map). Convert either to a clear,
            # actionable error naming the knob to lower.
            msg = str(exc)
            if any(s in msg for s in ("Kernel size", "input size", "more than 1 value per channel")):
                raise ValueError(
                    f"RewardModel(num_layers_d={self.num_layers_d}) cannot score a latent of "
                    f"spatial {tuple(latent.shape[2:])}: a spatial dim collapsed below the patch "
                    f"conv's receptive field. Reduce reward_model.num_layers_d or use larger latents."
                ) from exc
            raise
        return patch_map.flatten(1).mean(dim=1)  # [B]
