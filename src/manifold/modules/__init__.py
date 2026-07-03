"""Manifold training modules: stable-pretraining ``spt.Module`` components.

The training Module owns training-only concerns (logit-normal timestep sampling,
the ``(1−t)⁻²`` loss weight, the MSE); the transport and reverse step live in the
scheduler (ADR-0001).
"""

from .latent_flow import LatentFlowModule, SampleDict
from .partial_denoise import partial_denoise_rollout
from .reward import RewardBatch, RewardModule, bradley_terry_loss, reward_roc_auc
from .sampler import sample_latent_flow

__all__ = [
    "LatentFlowModule",
    "RewardBatch",
    "RewardModule",
    "SampleDict",
    "bradley_terry_loss",
    "partial_denoise_rollout",
    "reward_roc_auc",
    "sample_latent_flow",
]
