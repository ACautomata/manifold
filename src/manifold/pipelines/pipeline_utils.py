"""Pipeline base class, mimicking ``diffusers.DiffusionPipeline``.

A manifold Pipeline holds the model/scheduler components that turn noise +
conditions into a decoded volume, and persists them as a per-component directory
layout described by a ``model_index.json``. The base provides the config mixin
and the registry helpers; each concrete pipeline implements
``from_pretrained`` / ``save_pretrained`` for its own component layout. It does
**not** subclass ``diffusers.DiffusionPipeline`` (ADR-0001).
"""

from __future__ import annotations

from ..configuration import ConfigMixin


class DiffusionPipeline(ConfigMixin):
    """Base for manifold inference pipelines."""

    #: Top-level index file naming the pipeline and its components.
    config_name = "model_index.json"
