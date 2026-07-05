"""Manifold training modules: stable-pretraining ``spt.Module`` components.

The training Module owns training-only concerns (logit-normal timestep sampling,
the ``(1−t)⁻²`` loss weight, the MSE); the transport and reverse step live in the
scheduler (ADR-0001).
"""

from .grpo import (
    GRPOBatch,
    GRPOModule,
    RolloutStep,
    clipped_surrogate_loss,
    gaussian_log_prob,
    group_advantage,
    singular_branch_rollout,
)
from .latent_flow import LatentFlowModule, SampleDict
from .paired_latent_flow import PairedLatentFlowModule, PairedSampleDict
from .paired_sampler import sample_paired_latent_flow
from .partial_denoise import partial_denoise_rollout
from .reward import RewardBatch, RewardModule, bradley_terry_loss, reward_roc_auc
from .sampler import sample_latent_flow

__all__ = [
    "GRPOBatch",
    "GRPOModule",
    "LatentFlowModule",
    "PairedLatentFlowModule",
    "PairedSampleDict",
    "RewardBatch",
    "RewardModule",
    "RolloutStep",
    "SampleDict",
    "bradley_terry_loss",
    "clipped_surrogate_loss",
    "gaussian_log_prob",
    "group_advantage",
    "partial_denoise_rollout",
    "reward_roc_auc",
    "sample_latent_flow",
    "sample_paired_latent_flow",
    "singular_branch_rollout",
]
