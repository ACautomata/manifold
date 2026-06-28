"""Lightweight configuration base, mimicking ``diffusers.ConfigMixin``.

Manifold components store their constructor arguments as a JSON-serializable
config dict so each can be re-instantiated from disk (``from_config`` /
``from_json_file``). This is manifold's own minimal implementation; ``diffusers``
is **not** a dependency and these classes do not subclass it (ADR-0001).

Mirrors diffusers' two entry points: the :func:`register_to_config` decorator
(captures ``__init__`` arguments as the config) and the ``ConfigMixin`` base
(``config`` property, ``to_json_file`` / ``from_json_file``).
"""

from __future__ import annotations

import functools
import inspect
import json
import os
from typing import Any


def register_to_config(init):
    """Capture a component's ``__init__`` arguments as its config dict.

    Equivalent to diffusers' ``register_to_config`` decorator: after the wrapped
    ``__init__`` runs, ``self._internal_dict`` holds the bound arguments minus
    ``self``. Values are JSON-normalized lazily on read (see :func:`_to_jsonable`)
    so a non-serializable default such as a ``torch.dtype`` round-trips as a
    string. Used by models and schedulers whose constructor arguments are plain
    JSON-able config (the pipeline does not use it — its components are objects,
    not config, and are enumerated in ``model_index.json`` instead).
    """

    @functools.wraps(init)
    def wrapper(self, *args, **kwargs):
        sig = inspect.signature(init)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()
        config = {k: v for k, v in bound.arguments.items() if k != "self"}
        init(self, *args, **kwargs)
        self._internal_dict = dict(config)
        return None

    return wrapper


def _to_jsonable(value: Any) -> Any:
    """Best-effort conversion of a config value to a JSON-serializable form."""
    import torch

    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    # Uncommon for our configs; fall back to a string rather than crash on save.
    return str(value)


class ConfigMixin:
    """Minimal config-persistence mixin (mirrors ``diffusers.ConfigMixin``)."""

    #: File name a component writes its config to inside its directory.
    config_name = "config.json"

    @property
    def config(self) -> dict:
        """The JSON-serializable constructor config of this component."""
        return {k: _to_jsonable(v) for k, v in self._internal_dict.items()}

    def to_json_file(self, json_path: str) -> None:
        parent = os.path.dirname(json_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(self.config, f, indent=2, sort_keys=True)

    @classmethod
    def from_json_file(cls, json_path: str):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

    @classmethod
    def from_config(cls, config: dict, **kwargs):
        """Instantiate from a config dict, with keyword overrides."""
        return cls(**{**config, **kwargs})
