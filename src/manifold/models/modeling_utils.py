"""Model base class, mimicking ``diffusers.ModelMixin``.

A manifold Model is an :class:`torch.nn.Module` that wraps a backbone (here, a
MONAI MAISI network) behind a thin adapter and never reimplements it. The base
adds config persistence (via :class:`~manifold.configuration.ConfigMixin`) and a
diffusers-idiomatic ``save_pretrained`` / ``from_pretrained`` pair that writes a
per-component ``config.json`` + ``diffusion_pytorch_model.pt``. It does **not**
subclass ``diffusers.ModelMixin`` (ADR-0001).
"""

from __future__ import annotations

import os

import torch
from torch import nn

from ..configuration import ConfigMixin


class ModelMixin(nn.Module, ConfigMixin):
    """Base for manifold network wrappers (VAE, UNet).

    Subclasses decorate ``__init__`` with
    :func:`~manifold.configuration.register_to_config` so their config round-trips.
    State is persisted with :func:`torch.save` (a plain state dict) rather than
    safetensors, to keep the dependency surface at torch + monai only.
    """

    #: State-dict file name inside a component directory (diffusers-idiomatic).
    weights_name = "diffusion_pytorch_model.pt"

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.to_json_file(os.path.join(save_directory, self.config_name))
        torch.save(self.state_dict(), os.path.join(save_directory, self.weights_name))

    @classmethod
    def from_pretrained(cls, save_directory: str, *, map_location: str = "cpu"):
        model = cls.from_json_file(os.path.join(save_directory, cls.config_name))
        state = torch.load(
            os.path.join(save_directory, cls.weights_name),
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state)
        return model
