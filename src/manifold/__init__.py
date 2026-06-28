"""Manifold: pretraining & medical-imaging experiments.

Built on top of :mod:`stable_pretraining` and :mod:`monai`. The package mirrors
the diffusers four-component layout (models / schedulers / pipeline) with
manifold-defined lightweight base classes — it does **not** subclass the
``diffusers`` library (ADR-0001).
"""

from .configuration import ConfigMixin
from .models import AutoencoderKL, ModelMixin, UNet3DConditionModel
from .pipelines import DiffusionPipeline, LatentFlowPipeline
from .schedulers import FlowMatchHeunDiscreteScheduler, SchedulerMixin

__version__ = "0.1.0"

__all__ = [
    "AutoencoderKL",
    "ConfigMixin",
    "DiffusionPipeline",
    "FlowMatchHeunDiscreteScheduler",
    "LatentFlowPipeline",
    "ModelMixin",
    "SchedulerMixin",
    "UNet3DConditionModel",
]
