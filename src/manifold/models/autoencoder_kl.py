"""VAE wrapper owning ``scaling_factor`` (ADR-0003).

Wraps the MONAI MAISI autoencoder
(:class:`monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi`)
behind a diffusers-style :class:`AutoencoderKL`. The MAISI backbone is wrapped,
never reimplemented. ``scaling_factor`` (= ``1/std(z)``) is a buffer + config on
this wrapper: :meth:`encode` returns a **scaled** latent and :meth:`decode`
undoes the scaling internally, so the training Module and inference Pipeline
never reference it. This absorbs hope's scattered ``latent * scale_factor``
(module) and ``z / scale_factor`` (decode) into one place.

In this slice ``scaling_factor`` is supplied from a converted checkpoint or a
test value; its data-driven estimation (``1/std(z)`` over a cache) ships with the
(deferred) data stack.
"""

from __future__ import annotations

from typing import Sequence

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
)
from monai.inferers import sliding_window_inference
from torch import Tensor

from ..configuration import register_to_config
from .modeling_utils import ModelMixin


class AutoencoderKL(ModelMixin):
    """Image ↔ scaled-latent VAE (wraps the MONAI MAISI autoencoder).

    ``encode`` returns ``encode_stage_2_inputs(x) * scaling_factor`` (the
    reparameterized latent, scaled); ``decode`` divides by ``scaling_factor``
    then decodes via sliding window, returning image space.
    """

    @register_to_config
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        num_channels: Sequence[int] = (8, 8),
        num_res_blocks: Sequence[int] = (1, 1),
        attention_levels: Sequence[bool] = (False, False),
        latent_channels: int = 4,
        norm_num_groups: int = 8,
        scaling_factor: float = 1.0,
        num_splits: int = 1,
        save_mem: bool = False,
    ):
        """Args:
        scaling_factor: latent normalization scalar (``1/std(z)``); owned here
            as a buffer. ``encode`` multiplies by it, ``decode`` divides by it.
        num_splits / save_mem: MAISI's internal encoder/decoder block-split
            memory knob. Defaults disable it (``num_splits=1``) so tiny CPU
            tensors do not trip the split logic; production decode uses
            sliding window (see :meth:`decode`).
        Remaining args are the MAISI autoencoder's construction config.
        """
        super().__init__()
        self.autoencoder = AutoencoderKlMaisi(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=tuple(num_channels),
            num_res_blocks=tuple(num_res_blocks),
            attention_levels=tuple(attention_levels),
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            num_splits=num_splits,
            save_mem=save_mem,
        )
        self.register_buffer(
            "scaling_factor", torch.as_tensor(float(scaling_factor), dtype=torch.float32)
        )

    @property
    def _backbone_dtype(self) -> torch.dtype:
        return next(self.autoencoder.parameters()).dtype

    def encode(self, images: Tensor) -> Tensor:
        """Image → **scaled** latent (``encode_stage_2_inputs(x) * scaling_factor``)."""
        z = self.autoencoder.encode_stage_2_inputs(images)
        return z * self.scaling_factor.to(z.dtype)

    def decode(
        self,
        latents: Tensor,
        *,
        roi_size: Sequence[int] | None = None,
        sw_batch_size: int = 1,
        overlap: float = 0.25,
    ) -> Tensor:
        """Scaled latent → image, undoing the scaling internally.

        The latent is divided by ``scaling_factor`` before the MAISI decoder. For
        large 3D volumes decoding runs through MONAI sliding-window inference so
        it does not OOM; with ``roi_size`` ≥ the latent volume (the default) it
        decodes the whole volume in a single patch.
        """
        z = (latents.to(self.scaling_factor.dtype) / self.scaling_factor).to(self._backbone_dtype)
        if roi_size is None:
            roi_size = tuple(latents.shape[2:])
        return sliding_window_inference(
            z,
            roi_size=tuple(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=self.autoencoder.decode_stage_2_outputs,
            overlap=overlap,
        )
