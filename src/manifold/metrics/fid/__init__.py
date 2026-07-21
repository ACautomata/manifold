"""FID subpackage: composable helpers + callback (ADR-0030).

Public API:
- :class:`FIDCallback` — the per-epoch Lightning callback.
- :class:`VramStage` — context manager: stage/restore VAE + feature_net.
- :class:`FixedSampleRollout` — rank-strided seeded generation.
- :class:`LatentDecoder` — VAE decode in float32 with norm_float16 handling.
- :class:`FeatureExtractor` — 2.5D feature extraction via injectable feature_net.
- :class:`SufficientStatsReducer` — symmetric per-plane all_reduce of sufficient stats.
"""

from manifold.metrics.fid.callback import FIDCallback
from manifold.metrics.fid.decoder import LatentDecoder
from manifold.metrics.fid.extractor import FeatureExtractor
from manifold.metrics.fid.math import (
    frechet_distance_unbiased,
    get_features_2p5d,
    make_feature_network,
)
from manifold.metrics.fid.reducer import SufficientStatsReducer
from manifold.metrics.fid.rollout import FixedSampleRollout
from manifold.metrics.fid.vram import VramStage

__all__ = [
    "FIDCallback",
    "FeatureExtractor",
    "FixedSampleRollout",
    "LatentDecoder",
    "SufficientStatsReducer",
    "VramStage",
    "frechet_distance_unbiased",
    "get_features_2p5d",
    "make_feature_network",
]
