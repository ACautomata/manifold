"""Generative-validation metrics (issue #27): the unbiased 2.5D FID.

- :func:`frechet_distance_unbiased` — the small-sample-bias-corrected Fréchet
  distance (``Tr(Σ)/n`` mean-term subtraction + covariance ridge);
- :func:`get_features_2p5d` — three-plane (XY/YZ/ZX) feature extraction;
- :func:`make_feature_network` — the RadImageNet ResNet50 backbone factory
  (loads the cached ``_notop`` state_dict offline, ``torch.hub`` fallback);
- :class:`FIDCallback` — the per-epoch Lightning callback (fixed samples,
  re-seeded generation noise, single-GPU / rank-0).
- :class:`MetricsPlotCallback` — re-renders all logged metrics to a line-chart
  PNG every epoch + at fit end (crash-robust on remote DCU).
"""

from .fid.callback import FIDCallback
from .fid.decoder import LatentDecoder
from .fid.extractor import FeatureExtractor
from .fid.math import frechet_distance_unbiased, get_features_2p5d, make_feature_network
from .fid.reducer import SufficientStatsReducer
from .fid.rollout import FixedSampleRollout
from .fid.vram import VramStage
from .metric_plot_callback import MetricsPlotCallback

__all__ = [
    "FIDCallback",
    "FeatureExtractor",
    "FixedSampleRollout",
    "LatentDecoder",
    "MetricsPlotCallback",
    "SufficientStatsReducer",
    "VramStage",
    "frechet_distance_unbiased",
    "get_features_2p5d",
    "make_feature_network",
]
