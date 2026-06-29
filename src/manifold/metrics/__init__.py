"""Generative-validation metrics (issue #27): the unbiased 2.5D FID.

- :func:`frechet_distance_unbiased` — the small-sample-bias-corrected Fréchet
  distance (``Tr(Σ)/n`` mean-term subtraction + covariance ridge);
- :func:`get_features_2p5d` — three-plane (XY/YZ/ZX) feature extraction;
- :func:`make_feature_network` — the RadImageNet ResNet50 ``torch.hub`` factory;
- :class:`FIDCallback` — the per-epoch Lightning callback (fixed samples,
  re-seeded generation noise, single-GPU / rank-0).
"""

from .fid import frechet_distance_unbiased, get_features_2p5d, make_feature_network
from .fid_callback import FIDCallback

__all__ = [
    "FIDCallback",
    "frechet_distance_unbiased",
    "get_features_2p5d",
    "make_feature_network",
]
