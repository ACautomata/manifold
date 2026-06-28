"""3D conditioning UNet wrapper (predicts the clean latent x0).

Wraps the MONAI MAISI diffusion UNet
(:class:`monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi`)
behind a diffusers-style :class:`UNet3DConditionModel`. The MAISI backbone is
wrapped, never reimplemented; its native medical conditioning (``spacing`` +
``class_labels``, optional ``context``) is kept as-is rather than remapped onto
diffusers' ``encoder_hidden_states`` (ADR-0001).

The wrapper owns the timestep-embedding scale ``num_train_timesteps`` (the MAISI
UNet expects timesteps in ``[0, num_train_timesteps]``; callers pass flow-time
``t ∈ [0, 1]`` and the wrapper scales internally) and the spacing conditioning
range (``spacing`` is multiplied by ``1e2`` internally, matching the range the
MAISI spacing MLP was trained in — a property of the wrapped backbone).
"""

from __future__ import annotations

from typing import Sequence

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from torch import Tensor

from ..configuration import register_to_config
from .modeling_utils import ModelMixin

#: Spacing is fed to the MAISI conditioning MLP in this scaled range (voxel
#: spacing × 100), matching how the backbone was trained.
_SPACING_SCALE = 1e2


class UNet3DConditionModel(ModelMixin):
    """3D conditioning UNet whose forward predicts the clean latent x0."""

    @register_to_config
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 4,
        out_channels: int = 4,
        num_channels: Sequence[int] = (8, 8),
        num_res_blocks: int | Sequence[int] = (1, 1),
        attention_levels: Sequence[bool] = (False, False),
        norm_num_groups: int = 8,
        num_head_channels: int | Sequence[int] = 4,
        with_conditioning: bool = False,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        include_spacing_input: bool = True,
        include_top_region_index_input: bool = False,
        include_bottom_region_index_input: bool = False,
        resblock_updown: bool = False,
        include_fc: bool = False,
        use_flash_attention: bool = False,
        num_train_timesteps: int = 1000,
    ):
        """Construct the wrapper.

        Args:
            num_train_timesteps: the MAISI UNet time-embedding scale; ``timestep``
                passed to :meth:`forward` (a flow-time ``t ∈ [0, 1]``) is
                multiplied by it before the backbone forward.
            Remaining args are the MAISI diffusion UNet's construction config; the
            new knobs default to MAISI's values so the tiny-CPU fixtures are unchanged.
        """
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        # MAISI accepts a scalar (broadcast across levels) or a per-level
        # sequence for ``num_res_blocks``; pass a scalar through unchanged and
        # tuple a sequence (current behaviour).
        nr_blocks = num_res_blocks if isinstance(num_res_blocks, int) else tuple(num_res_blocks)
        self.unet = DiffusionModelUNetMaisi(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=tuple(num_channels),
            num_res_blocks=nr_blocks,
            attention_levels=tuple(attention_levels),
            norm_num_groups=norm_num_groups,
            num_head_channels=num_head_channels,
            with_conditioning=with_conditioning,
            cross_attention_dim=cross_attention_dim,
            num_class_embeds=num_class_embeds,
            include_spacing_input=include_spacing_input,
            include_top_region_index_input=include_top_region_index_input,
            include_bottom_region_index_input=include_bottom_region_index_input,
            resblock_updown=resblock_updown,
            include_fc=include_fc,
            use_flash_attention=use_flash_attention,
        )

    def _scaled_timesteps(self, timestep, batch_size: int, device, dtype) -> Tensor:
        """Flow-time ``t ∈ [0, 1]`` → the backbone's ``[0, num_train_timesteps]`` grid.

        A scalar timestep is broadcast to the batch (the inference seam, one node
        per step); a ``(B,)`` tensor timestep is scaled elementwise (the training
        seam, logit-normal per-sample ``t``).
        """
        ts = torch.as_tensor(timestep, device=device)
        if ts.dim() == 0:
            ts = ts.expand(batch_size)
        return (ts.float() * self.num_train_timesteps).to(dtype)

    @staticmethod
    def _batched_spacing(spacing: Tensor, batch_size: int) -> Tensor:
        """Raw voxel spacing → ``[B, 3]`` conditioning tensor (×1e2)."""
        if spacing.dim() == 1:
            spacing = spacing.unsqueeze(0)
        if spacing.shape[0] == 1 and batch_size > 1:
            spacing = spacing.expand(batch_size, -1)
        return spacing * _SPACING_SCALE

    def forward(
        self,
        sample: Tensor,
        timestep,
        spacing: Tensor,
        class_labels: Tensor | None = None,
        context: Tensor | None = None,
    ) -> Tensor:
        """Predict the clean latent x0 given a noised latent and medical conditions.

        Args:
            sample: the noised latent ``z`` ``[B, C, D, H, W]``.
            timestep: flow-time ``t ∈ [0, 1]`` (scalar or ``(B,)``); scaled to the
                backbone's embedding grid internally.
            spacing: raw voxel spacing ``[3]`` or ``[B, 3]``; scaled ×1e2 internally.
            class_labels: optional modality label ``[B]`` (long).
            context: optional cross-attention context.
        """
        b = sample.shape[0]
        timesteps = self._scaled_timesteps(timestep, b, sample.device, sample.dtype)
        spacing_tensor = self._batched_spacing(spacing, b)
        return self.unet(
            x=sample,
            timesteps=timesteps,
            context=context,
            class_labels=class_labels,
            spacing_tensor=spacing_tensor,
        )
