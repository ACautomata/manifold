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
- :class:`PairedFidelityMetrics` — 3D PSNR + 3D SSIM of a generated target vs the
  real target (MONAI-backed, ADR-0036); the ControlNet translation-fidelity screen.
- :class:`PairedFidelityCallback` — the observe-only in-training paired-fidelity
  monitor (``val/psnr`` / ``val/ssim``) for supervised ControlNet (ADR-0037).
- :class:`VaeStage` — the VAE-only VRAM stage/restore context manager (composed by
  ``VramStage``; used directly by the paired-fidelity decode).
"""

from .fid.callback import FIDCallback
from .fid.decoder import LatentDecoder
from .fid.extractor import FeatureExtractor
from .fid.math import frechet_distance_unbiased, get_features_2p5d, make_feature_network
from .fid.reducer import SufficientStatsReducer
from .fid.rollout import FixedSampleRollout
from .fid.vram import VramStage
from .metric_plot_callback import MetricsPlotCallback
from .paired import PairedFidelityMetrics, PairedFidelityScores
from .paired_callback import PairedFidelityCallback
from .vae_stage import VaeStage

__all__ = [
    "FIDCallback",
    "FeatureExtractor",
    "FixedSampleRollout",
    "LatentDecoder",
    "MetricsPlotCallback",
    "PairedFidelityCallback",
    "PairedFidelityMetrics",
    "PairedFidelityScores",
    "SufficientStatsReducer",
    "VaeStage",
    "VramStage",
    "frechet_distance_unbiased",
    "get_features_2p5d",
    "make_feature_network",
]
