"""Generative-validation metrics (issue #27): the unbiased 2.5D FID.

- :func:`frechet_distance_unbiased` — the small-sample-bias-corrected Fréchet
  distance (``Tr(Σ)/n`` mean-term subtraction + covariance ridge);
- :func:`get_features_2p5d` — three-plane (XY/YZ/ZX) feature extraction;
- :func:`make_feature_network` — the RadImageNet ResNet50 backbone factory
  (loads the cached ``_notop`` state_dict offline, ``torch.hub`` fallback);
- :class:`FIDCallback` — the per-epoch Lightning callback (fixed samples,
  re-seeded generation noise, single-GPU / rank-0).
- :class:`PairedPSNRSSIMCallback` — the Paired JiT per-epoch pixel-space 3D
  PSNR/SSIM callback (issue #68; deterministic given ``x_src`` — no re-seed).
- :class:`MetricsPlotCallback` — re-renders all logged metrics to a line-chart
  PNG every epoch + at fit end (crash-robust on remote DCU).
"""

from .fid import frechet_distance_unbiased, get_features_2p5d, make_feature_network
from .fid_callback import FIDCallback
from .metric_plot_callback import MetricsPlotCallback
from .psnr_ssim_callback import PairedPSNRSSIMCallback

__all__ = [
    "FIDCallback",
    "MetricsPlotCallback",
    "PairedPSNRSSIMCallback",
    "frechet_distance_unbiased",
    "get_features_2p5d",
    "make_feature_network",
]
