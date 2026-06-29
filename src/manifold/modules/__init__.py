"""Manifold training modules: stable-pretraining ``spt.Module`` components.

The training Module owns training-only concerns (logit-normal timestep sampling,
the ``(1−t)⁻²`` loss weight, the MSE); the transport and reverse step live in the
scheduler (ADR-0001).
"""

from .latent_flow import LatentFlowModule, SampleDict
from .sampler import sample_latent_flow

__all__ = ["LatentFlowModule", "SampleDict", "sample_latent_flow"]
