"""3D ControlNet wrapper — a trainable adapter on the frozen noise→data JiT UNet.

A canonical ControlNet (Zhang & Agrawala 2023) for paired MRI translation: a clone of
the frozen base UNet's encoder (``conv_in`` + input-embedding path + ``down_blocks`` +
``middle_block``) plus zero-conv layers, whose residuals are injected into the **frozen**
:class:`~manifold.models.unet_3d_condition.UNet3DConditionModel` through the MONAI
``DiffusionModelUNetMaisi``'s native forward args ``down_block_additional_residuals`` /
``mid_block_additional_residual`` (ADR-0026). The MONAI MAISI UNet is a diffusers-port
that already exposes these args and the encoder submodules; this wrapper composes the
same public MONAI block builders (``get_down_block`` / ``get_mid_block`` /
``Convolution``) the base uses internally — it never reimplements them (ADR-0001).

The source ``x_src`` enters as a **control signal**, not a transport endpoint: a
``controlnet_cond_embedding`` conv maps it to the ``conv_in`` output width and adds it
post-``conv_in`` (the diffusers ``ControlNetModel`` precedent), so the ControlNet's own
``conv_in`` stays clone-able from the base. The zero-conv layers are zero-initialized, so
the initial residuals are zero and the model's initial behavior is the pretrained JiT UNet
unchanged (a safe warm-start).

The src→tgt contrast-direction conditioning lives on the ControlNet's class-embedding
path: a direction MLP over ``concat(embed(src), embed(tgt+offset))`` (ADR-0028),
substituted for the plain class-embedding lookup in the time-embedding assembly. The
(frozen) base UNet separately receives the target-contrast ``class_labels`` for its own
modality embedding; the ControlNet carries the translation direction.

``forward`` returns ``(down_block_additional_residuals, mid_block_additional_residual)``
— the caller (training Module or inference pipeline) passes them to the frozen base UNet.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from monai.networks.blocks import Convolution
from monai.networks.nets.diffusion_model_unet import (
    get_down_block,
    get_mid_block,
    get_timestep_embedding,
    zero_module,
)
from monai.utils import ensure_tuple_rep
from torch import Tensor
from torch import nn

from ..configuration import register_to_config
from .modeling_utils import ModelMixin

#: Spacing is fed to the MAISI conditioning MLP in this scaled range — must match the
#: frozen base UNet wrapper (models/unet_3d_condition.py), whose backbone was trained in it.
_SPACING_SCALE = 1e2


def _embedding_module(input_dim: int, embed_dim: int) -> nn.Sequential:
    """MAISI's ``_create_embedding_module`` (Linear → SiLU → Linear), mirrored verbatim."""
    return nn.Sequential(nn.Linear(input_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))


class ControlNet3DConditionModel(ModelMixin):
    """3D ControlNet whose forward emits base-UNet residual injections."""

    @register_to_config
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 4,
        controlnet_cond_channels: int = 4,
        num_channels: Sequence[int] = (8, 8),
        num_res_blocks: int | Sequence[int] = (1, 1),
        attention_levels: Sequence[bool] = (False, False),
        norm_num_groups: int = 8,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 4,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        include_spacing_input: bool = True,
        include_top_region_index_input: bool = False,
        include_bottom_region_index_input: bool = False,
        include_fc: bool = False,
        use_flash_attention: bool = False,
        num_train_timesteps: int = 1000,
        paired_direction_offset: int = 0,
    ):
        """Construct the ControlNet.

        Args:
            in_channels: channels of the noisy latent ``z_t`` the ``conv_in`` takes —
                equals the base UNet's ``in_channels`` (``C_latent``), so ``conv_in`` is
                clone-able from the base.
            controlnet_cond_channels: channels of the control signal ``x_src`` (a scaled
                VAE latent ⇒ ``C_latent``). Mapped to ``num_channels[0]`` by
                ``controlnet_cond_embedding`` and added post-``conv_in``.
            num_train_timesteps: the timestep-embedding scale; the flow-time ``t ∈ [0, 1]``
                passed to :meth:`forward` is multiplied by it (must match the frozen base).
            paired_direction_offset: integer added to the **target** contrast label before
                its class-embedding lookup in the direction MLP (breaks A↔B symmetry,
                ADR-0014 addendum carried onto the ControlNet).
            Remaining args mirror the MAISI UNet config so ``down_blocks`` / ``middle_block``
                match the base (required both for weight warm-start and for the residual
                shapes to align with the base's ``_apply_down_blocks``).
        """
        super().__init__()
        self.num_train_timesteps = int(num_train_timesteps)
        self.paired_direction_offset = int(paired_direction_offset)
        self.num_class_embeds = num_class_embeds

        block_out = tuple(num_channels)
        nr_tuple = ensure_tuple_rep(num_res_blocks, len(block_out)) if isinstance(num_res_blocks, int) else tuple(num_res_blocks)
        heads = ensure_tuple_rep(num_head_channels, len(block_out)) if isinstance(num_head_channels, int) else tuple(num_head_channels)
        time_embed_dim = block_out[0] * 4
        # temb width after the input-embedding assembly (time + [class] + spacing concat),
        # matching the base's new_time_embed_dim so down/mid blocks are interchangeable.
        new_time_embed_dim = time_embed_dim
        if include_top_region_index_input:
            new_time_embed_dim += time_embed_dim
        if include_bottom_region_index_input:
            new_time_embed_dim += time_embed_dim
        if include_spacing_input:
            new_time_embed_dim += time_embed_dim
        self.include_spacing_input = include_spacing_input
        self.include_top_region_index_input = include_top_region_index_input
        self.include_bottom_region_index_input = include_bottom_region_index_input
        self.with_conditioning = with_conditioning
        self._block_out = block_out

        # input conv — clone-able from the base (takes z_t, C_latent).
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=block_out[0],
            strides=1, kernel_size=3, padding=1, conv_only=True,
        )
        # control signal pathway: x_src (C_latent) → conv_in width, added post-conv_in.
        self.controlnet_cond_embedding = Convolution(
            spatial_dims=spatial_dims,
            in_channels=controlnet_cond_channels,
            out_channels=block_out[0],
            strides=1, kernel_size=3, padding=1, conv_only=True,
        )

        # input embeddings — mirror the base's _get_time_and_class/_get_input_embeddings.
        self.time_embed = _embedding_module(block_out[0], time_embed_dim)
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        if include_top_region_index_input:
            self.top_region_index_layer = _embedding_module(4, time_embed_dim)
        if include_bottom_region_index_input:
            self.bottom_region_index_layer = _embedding_module(4, time_embed_dim)
        if include_spacing_input:
            self.spacing_layer = _embedding_module(3, time_embed_dim)
        # direction MLP: concat([embed(src), embed(tgt+offset)]) → time_embed_dim, replacing
        # the plain class-embedding lookup (ADR-0028).
        self.paired_cond_mlp = nn.Sequential(
            nn.Linear(time_embed_dim * 2, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # encoder — identical config to the base (clone-able; residual shapes align).
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out[0]
        for i in range(len(block_out)):
            input_channel = output_channel
            output_channel = block_out[i]
            is_final_block = i == len(block_out) - 1
            self.down_blocks.append(
                get_down_block(
                    spatial_dims=spatial_dims,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=new_time_embed_dim,
                    num_res_blocks=nr_tuple[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    add_downsample=not is_final_block,
                    resblock_updown=resblock_updown,
                    with_attn=(attention_levels[i] and not with_conditioning),
                    with_cross_attn=(attention_levels[i] and with_conditioning),
                    num_head_channels=heads[i],
                    transformer_num_layers=transformer_num_layers,
                    cross_attention_dim=cross_attention_dim,
                    include_fc=include_fc,
                    use_flash_attention=use_flash_attention,
                )
            )
        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=block_out[-1],
            temb_channels=new_time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=heads[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            include_fc=include_fc,
            use_flash_attention=use_flash_attention,
        )

        # zero-convolution residual outputs (zero-init ⇒ initial residuals are zero).
        down_sample_channels = self._down_sample_channels(block_out, nr_tuple)
        self.controlnet_down_zero_convs = nn.ModuleList(
            zero_module(
                Convolution(
                    spatial_dims=spatial_dims, in_channels=ch, out_channels=ch,
                    strides=1, kernel_size=1, padding=0, conv_only=True,
                )
            )
            for ch in down_sample_channels
        )
        self.controlnet_mid_zero_conv = zero_module(
            Convolution(
                spatial_dims=spatial_dims, in_channels=block_out[-1], out_channels=block_out[-1],
                strides=1, kernel_size=1, padding=0, conv_only=True,
            )
        )

    @staticmethod
    def _down_sample_channels(block_out: tuple[int, ...], nr_tuple: tuple[int, ...]) -> list[int]:
        """Channel count of each collected down-block residual sample.

        Mirrors the base's ``_apply_down_blocks`` collection order: the ``conv_in`` output
        first, then each down-block's resnet outputs (+ a downsampler output when present).
        All samples within block ``i`` carry ``block_out[i]`` channels.
        """
        channels = [block_out[0]]
        for i in range(len(block_out)):
            is_final_block = i == len(block_out) - 1
            channels += [block_out[i]] * (nr_tuple[i] + (0 if is_final_block else 1))
        return channels

    def _scaled_timesteps(self, timestep, batch_size: int, device, dtype) -> Tensor:
        ts = torch.as_tensor(timestep, device=device)
        if ts.dim() == 0:
            ts = ts.expand(batch_size)
        return (ts.float() * self.num_train_timesteps).to(dtype)

    @staticmethod
    def _batched_spacing(spacing: Tensor, batch_size: int) -> Tensor:
        if spacing.dim() == 1:
            spacing = spacing.unsqueeze(0)
        if spacing.shape[0] == 1 and batch_size > 1:
            spacing = spacing.expand(batch_size, -1)
        return spacing * _SPACING_SCALE

    def forward(
        self,
        sample: Tensor,
        controlnet_cond: Tensor,
        timestep,
        spacing: Tensor,
        class_labels_src: Tensor,
        class_labels_tgt: Tensor,
        context: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        """Emit the base-UNet residual injections for one ControlNet eval.

        Args:
            sample: the noisy latent ``z_t`` ``[B, C_latent, D, H, W]`` (same as the base
                UNet input).
            controlnet_cond: the source latent ``x_src`` ``[B, C_latent, D, H, W]``.
            timestep: flow-time ``t ∈ [0, 1]`` (scalar or ``(B,)``); scaled internally.
            spacing: raw voxel spacing ``[3]`` or ``[B, 3]``; scaled ×1e2 internally.
            class_labels_src / class_labels_tgt: the (src, tgt) contrast pair ``[B]``
                (long) feeding the direction MLP.
            context: optional cross-attention context.

        Returns:
            ``(down_block_additional_residuals, mid_block_additional_residual)`` — pass
            these to the frozen base UNet's ``forward`` of the same names.
        """
        if self.num_class_embeds is None:
            raise ValueError("ControlNet direction conditioning requires num_class_embeds to be set.")
        b = sample.shape[0]
        timesteps = self._scaled_timesteps(timestep, b, sample.device, sample.dtype)
        spacing_tensor = self._batched_spacing(spacing, b)

        # time + direction conditioning (direction replaces the plain class-embedding lookup).
        t_emb = get_timestep_embedding(timesteps, self._block_out[0]).to(dtype=sample.dtype)
        emb = self.time_embed(t_emb)
        tgt_rows = class_labels_tgt + self.paired_direction_offset
        cond = self.paired_cond_mlp(
            torch.cat([self.class_embedding(class_labels_src), self.class_embedding(tgt_rows)], dim=-1)
        )
        emb = emb + cond.to(dtype=emb.dtype)
        # input-embedding assembly (spacing concat), matching the base's new_time_embed_dim.
        if self.include_spacing_input:
            emb = torch.cat((emb, self.spacing_layer(spacing_tensor)), dim=1)

        # encoder pass — mirrors the base's _apply_down_blocks collection order.
        h = self.conv_in(sample)
        h = h + self.controlnet_cond_embedding(controlnet_cond)
        down_block_res_samples: list[Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            down_block_res_samples.extend(res_samples)
        if len(down_block_res_samples) != len(self.controlnet_down_zero_convs):
            raise RuntimeError(
                f"ControlNet collected {len(down_block_res_samples)} down residuals but "
                f"{len(self.controlnet_down_zero_convs)} zero-convs exist — the MONAI "
                f"down-block structure does not match _down_sample_channels."
            )
        down_block_additional_residuals = tuple(
            zc(res) for zc, res in zip(self.controlnet_down_zero_convs, down_block_res_samples)
        )

        h = self.middle_block(h, emb, context)
        mid_block_additional_residual = self.controlnet_mid_zero_conv(h)
        return down_block_additional_residuals, mid_block_additional_residual

    def load_base_encoder_weights(self, base_unet) -> None:
        """Warm-start from a frozen base :class:`UNet3DConditionModel`.

        Copies the base's encoder submodules (``conv_in``, ``time_embed``,
        ``class_embedding``, ``spacing_layer``, ``down_blocks``, ``middle_block``) into the
        matching ControlNet submodules. The ``controlnet_cond_embedding``, zero-convs, and
        direction MLP keep their construction init (zero-convs are zero ⇒ initial residuals
        are zero ⇒ initial behavior is the pretrained base unchanged).
        """
        src = base_unet.unet  # the wrapped MONAI MAISI backbone
        self.conv_in.load_state_dict(src.conv_in.state_dict())
        self.time_embed.load_state_dict(src.time_embed.state_dict())
        if hasattr(self, "class_embedding") and hasattr(src, "class_embedding"):
            self.class_embedding.load_state_dict(src.class_embedding.state_dict())
        if hasattr(self, "spacing_layer") and hasattr(src, "spacing_layer"):
            self.spacing_layer.load_state_dict(src.spacing_layer.state_dict())
        for cn_block, base_block in zip(self.down_blocks, src.down_blocks):
            cn_block.load_state_dict(base_block.state_dict())
        self.middle_block.load_state_dict(src.middle_block.state_dict())
