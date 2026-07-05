"""Manifold: pretraining & medical-imaging experiments.

Built on top of :mod:`stable_pretraining` and :mod:`monai`. The package mirrors
the diffusers four-component layout (models / schedulers / training module /
pipeline) with manifold-defined lightweight base classes — it does **not**
subclass the ``diffusers`` library (ADR-0001).
"""

from .configuration import ConfigMixin
from .models import AutoencoderKL, ModelMixin, RewardModel, UNet3DConditionModel
from .modules import LatentFlowModule, PairedLatentFlowModule
from .pipelines import DiffusionPipeline, LatentFlowPipeline, PairedLatentFlowPipeline
from .schedulers import (
    FlowMatchGRPOScheduler,
    FlowMatchHeunDiscreteScheduler,
    PartialFlowMatchHeunScheduler,
    SchedulerMixin,
)

__version__ = "0.1.0"

__all__ = [
    "AutoencoderKL",
    "ConfigMixin",
    "DiffusionPipeline",
    "FlowMatchGRPOScheduler",
    "FlowMatchHeunDiscreteScheduler",
    "LatentFlowModule",
    "LatentFlowPipeline",
    "ModelMixin",
    "PairedLatentFlowModule",
    "PairedLatentFlowPipeline",
    "PartialFlowMatchHeunScheduler",
    "RewardModel",
    "SchedulerMixin",
    "UNet3DConditionModel",
]
