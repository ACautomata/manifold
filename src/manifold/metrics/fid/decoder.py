"""LatentDecoder — VAE decode for FID eval in float32 on the staged device.

``norm_float16`` makes MaisiGroupNorm3D cast its output to float16
unconditionally, so a downstream float32 conv raises a bias-type mismatch
unless an outer autocast reconciles them — which a validation hook cannot
rely on. FID is an *evaluation* metric, so float32 decode is both robust and
more correct than half precision.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentDecoder:
    """VAE decode with norm_float16 handling, float32-on-device.

    On the first call, iterates the VAE modules and sets ``norm_float16 = False``
    on any module that carries it (MaisiGroupNorm3D workaround). Thereafter the
    flag is cached and the iteration is skipped.

    Args:
        vae: the held frozen VAE; its ``.decode()`` decodes latents to images.
    """

    def __init__(self, vae: nn.Module) -> None:
        self._vae = vae
        self._norm16_disabled: bool = False

    @property
    def norm16_disabled(self) -> bool:
        """Whether norm_float16 has been disabled (for test assertions)."""
        return self._norm16_disabled

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents in float32 on the VAE's device.

        Args:
            latents: ``[N, C, D, H, W]`` latent tensor.

        Returns:
            Decoded image tensor ``[N, C_out, D_out, H_out, W_out]``.
        """
        if not self._norm16_disabled:
            for m in self._vae.modules():
                if hasattr(m, "norm_float16"):
                    m.norm_float16 = False
            self._norm16_disabled = True
        vae_device = next(self._vae.parameters()).device
        return self._vae.decode(latents.float().to(vae_device))
