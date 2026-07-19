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

Paired JiT (ADR-0014 addendum): the wrapper additionally accepts a ``(src, tgt)``
contrast-label pair and injects a learned nonlinear combination at the backbone's
native class-embedding injection point. A lightweight MLP projects
``concat([embed(src), embed(tgt+offset)])`` from R^{2·time_embed_dim} to
R^{time_embed_dim}, replacing the original linear sum ``embed(src) + embed(tgt)``
so translation direction is a learned nonlinear combination with greater
discriminability across the 12 contrast directions. The MAISI backbone builds its
``class_embedding`` internally and adds it to the time embedding inside
``_get_time_and_class_embedding``; rather than reimplement that private assembly,
the paired path transiently swaps the backbone's ``class_embedding`` module for
:class:`_PinnedClassEmbedding` — a thin stand-in that returns the precomputed MLP
output — around the single backbone forward (try/finally restored). The noise→data
callers (a single ``class_labels`` tensor) never enter this branch and are
byte-unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from torch import Tensor
from torch import nn

from ..configuration import register_to_config
from .modeling_utils import ModelMixin

#: Spacing is fed to the MAISI conditioning MLP in this scaled range (voxel
#: spacing × 100), matching how the backbone was trained.
_SPACING_SCALE = 1e2


class _PinnedClassEmbedding(nn.Module):
    """Transient stand-in for the MAISI ``class_embedding`` (Paired JiT, ADR-0014).

    Returns a **precomputed** conditioning vector regardless of the label index
    passed in. Installed on the backbone (``backbone.class_embedding = ...``) only
    for the duration of one Paired-JiT forward, so the backbone's internal
    ``self.class_embedding(class_labels)`` call — which it then adds to the time
    embedding — yields the MLP-computed conditioning vector. The vector is a plain
    attribute (not a ``Parameter``/``Buffer``): it carries autograd history from
    where it was computed (the concatenated embedding rows + MLP), so gradients
    still reach the embedding table and the MLP; and because it is not a registered
    parameter/buffer it never enters a ``state_dict`` (the swap is instantaneous and
    restored in ``finally``).
    """

    def __init__(self, embedding: Tensor):
        super().__init__()
        self.embedding = embedding

    def forward(self, class_labels: Tensor) -> Tensor:  # noqa: ANN001
        # The label index is ignored — the summed vector was computed by the caller.
        return self.embedding


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
        paired_direction_offset: int = 0,
    ):
        """Construct the wrapper.

        Args:
            num_train_timesteps: the MAISI UNet time-embedding scale; ``timestep``
                passed to :meth:`forward` (a flow-time ``t ∈ [0, 1]``) is
                multiplied by it before the backbone forward.
            paired_direction_offset: optional integer added to the **target**
                contrast label before its class-embedding lookup in the Paired JiT
                conditioning path. The default ``0`` keeps ``cond = MLP(concat(
                embed(src), embed(tgt)))`` (the symmetric ADR-0014 behaviour). A
                non-zero offset shifts the target rows so the MLP sees
                ``concat(embed(src), embed(tgt+offset))``, which is no longer
                commutative in src<->tgt, breaking the A<->B symmetry.
            Remaining args are the MAISI diffusion UNet's construction config; the
            new knobs default to MAISI's values so the tiny-CPU fixtures are unchanged.
        """
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.paired_direction_offset = int(paired_direction_offset)
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
        # Paired conditioning MLP: concat([embed(src), embed(tgt+offset)]) ∈
        # R^{2·time_embed_dim} → R^{time_embed_dim} with a SiLU hidden layer.
        # Replaces the linear sum embed(src)+embed(tgt+offset) so translation
        # direction is a learned nonlinear combination — the two embedding rows
        # are no longer forced to share an additive space (ADR-0014 addendum).
        _tem_dim = tuple(num_channels)[0] * 4   # MAISI time_embed_dim
        self.paired_cond_mlp = nn.Sequential(
            nn.Linear(_tem_dim * 2, _tem_dim),
            nn.SiLU(),
            nn.Linear(_tem_dim, _tem_dim),
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
        *,
        class_labels_src: Tensor | None = None,
        class_labels_tgt: Tensor | None = None,
        down_block_additional_residuals: tuple[Tensor, ...] | None = None,
        mid_block_additional_residual: Tensor | None = None,
    ) -> Tensor:
        """Predict the clean latent x0 given a noised latent and medical conditions.

        Args:
            sample: the noised latent ``z`` ``[B, C, D, H, W]`` (Paired JiT passes
                ``concat([z_t, x_src])`` with ``C = 2·C_latent``).
            timestep: flow-time ``t ∈ [0, 1]`` (scalar or ``(B,)``); scaled to the
                backbone's embedding grid internally.
            spacing: raw voxel spacing ``[3]`` or ``[B, 3]``; scaled ×1e2 internally.
            class_labels: optional modality label ``[B]`` (long) — the noise→data
                JiT path. Byte-unchanged when the paired kwargs are absent.
            context: optional cross-attention context.
            class_labels_src / class_labels_tgt: the Paired JiT (src, tgt) contrast
                pair ``[B]`` (long). When **both** are passed the wrapper injects
                ``MLP(concat(embed(src), embed(tgt+paired_direction_offset)))`` at the
                backbone's class-embedding injection point (ADR-0014 addendum); they
                must be passed together and require ``num_class_embeds`` to be set.
            down_block_additional_residuals / mid_block_additional_residual: the
                ControlNet residual injections (ADR-0026) — passed straight through to the
                MONAI backbone's native forward args. ``None`` (the noise→data JiT path)
                leaves the backbone byte-unchanged.
        """
        b = sample.shape[0]
        timesteps = self._scaled_timesteps(timestep, b, sample.device, sample.dtype)
        spacing_tensor = self._batched_spacing(spacing, b)

        if class_labels_src is not None or class_labels_tgt is not None:
            # Paired JiT conditioning path (ADR-0014 addendum). Requires both
            # a class-embedding table (num_class_embeds set on the backbone).
            if class_labels_src is None or class_labels_tgt is None:
                raise ValueError(
                    "class_labels_src and class_labels_tgt must be passed together "
                    "(Paired JiT conditioning, ADR-0014)."
                )
            embedding = getattr(self.unet, "class_embedding", None)
            if embedding is None:
                raise ValueError(
                    "Paired JiT summed-label conditioning requires num_class_embeds "
                    "to be set on the UNet (the backbone has no class_embedding table)."
                )
            # The paired conditioning flows through the MLP:
            # concat([embed(src), embed(tgt+offset)]) → Linear → SiLU → Linear → cond.
            # Autograd reaches the embedding table and the MLP through the
            # concatenated embedding rows; the _PinnedClassEmbedding swap
            # below is transparent to gradients.
            tgt_rows = class_labels_tgt + self.paired_direction_offset
            num_rows = embedding.weight.shape[0]
            if int(self.paired_direction_offset) != 0:
                lo = int(min(class_labels_src.min().item(), tgt_rows.min().item()))
                hi = int(max(class_labels_src.max().item(), tgt_rows.max().item()))
                if lo < 0 or hi >= num_rows:
                    raise ValueError(
                        f"paired label out of range after offset: src range "
                        f"[{class_labels_src.min().item()}, {class_labels_src.max().item()}], "
                        f"tgt+offset range [{tgt_rows.min().item()}, {tgt_rows.max().item()}] "
                        f"(paired_direction_offset={self.paired_direction_offset}, "
                        f"num_class_embeds={num_rows}); use a smaller offset or a larger "
                        f"num_class_embeds."
                    )
            # MLP over concatenated embeddings replaces the linear sum of
            # ADR-0014 — the two rows are no longer forced to share a single
            # additive space; the nonlinear projection can learn per-direction
            # conditioning with greater discriminability.
            cond = self.paired_cond_mlp(
                torch.cat([embedding(class_labels_src), embedding(tgt_rows)], dim=-1)
            )
            pinned = _PinnedClassEmbedding(cond)
            original = self.unet.class_embedding
            self.unet.class_embedding = pinned
            try:
                # class_labels=src is a sentinel: the pinned module ignores it and
                # returns cond, which the backbone adds to the time embedding —
                # net effect emb += MLP(concat(embed(src), embed(tgt+offset))).
                return self.unet(
                    x=sample,
                    timesteps=timesteps,
                    context=context,
                    class_labels=class_labels_src,
                    spacing_tensor=spacing_tensor,
                    down_block_additional_residuals=down_block_additional_residuals,
                    mid_block_additional_residual=mid_block_additional_residual,
                )
            finally:
                self.unet.class_embedding = original

        return self.unet(
            x=sample,
            timesteps=timesteps,
            context=context,
            class_labels=class_labels,
            spacing_tensor=spacing_tensor,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )
