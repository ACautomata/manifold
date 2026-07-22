"""BraTS latent-prep data stack (issue #16).

Turns BraTS2023 NIfTIs into the training batch the :class:`~manifold.LatentFlowModule`
consumes: a NIfTI volume dataset (RAS reorient + the preprocessing transforms), a
BraTS label provider, a latent dataset that warms an **unscaled** cache via the
VAE's ``encode_raw`` and returns **scaled** latents at ``__getitem__`` (scale-on-
read, ADR-0003 addendum), and the latent-prep orchestration +
``spt.data.DataModule`` factory.
"""

from .base import LabelProvider, MedicalDataset, SampleDict
from .datamodule import build_datamodule
from .labels import (
    BRATS_CONTRASTS,
    DEFAULT_BRATS_LABELS,
    BratsLabelProvider,
    FixedLabelProvider,
    ManifestLabelProvider,
    detect_brats_contrast,
    label_provider_from_config,
    load_brats_labels,
)
from .latent_dataset import EncodeFn, LatentDataset, estimate_scale_factor
from .latent_pipeline import (
    LatentPipeline,
    build_encode_pipeline,
    build_volume_dataset,
    load_vae,
    warm_latent_pipeline,
)
from .paired_brats import build_brats_pair_manifest
from .paired_latent_dataset import (
    PairedLatentDataset,
    estimate_paired_scale_factor,
)
from .paired_manifests import _train_val_manifests
from .paired_reward_pairs import (
    build_paired_reward_pairs,
    build_paired_reward_probe,
)
from .paired_volume_dataset import PairedNiftiVolumeDataset
from .reward_pairs import (
    RewardPairDataset,
    generate_generated_end_probe,
    generate_reward_pairs,
    load_frozen_denoiser,
    load_reward_pairs,
    save_reward_pairs,
)
from .transforms import floor_to_divisible, normalize_to_01, pad_to_divisible, resize_to
from .volume_dataset import NiftiVolumeDataset, collect_nifti_paths

__all__ = [
    "BRATS_CONTRASTS",
    "BratsLabelProvider",
    "DEFAULT_BRATS_LABELS",
    "EncodeFn",
    "FixedLabelProvider",
    "LabelProvider",
    "LatentDataset",
    "LatentPipeline",
    "ManifestLabelProvider",
    "MedicalDataset",
    "NiftiVolumeDataset",
    "PairedLatentDataset",
    "PairedNiftiVolumeDataset",
    "RewardPairDataset",
    "SampleDict",
    "_train_val_manifests",
    "build_brats_pair_manifest",
    "build_datamodule",
    "build_encode_pipeline",
    "build_paired_reward_pairs",
    "build_paired_reward_probe",
    "build_volume_dataset",
    "collect_nifti_paths",
    "detect_brats_contrast",
    "estimate_paired_scale_factor",
    "estimate_scale_factor",
    "floor_to_divisible",
    "generate_generated_end_probe",
    "generate_reward_pairs",
    "label_provider_from_config",
    "load_brats_labels",
    "load_frozen_controlnet_generator",
    "load_frozen_denoiser",
    "load_reward_pairs",
    "load_vae",
    "normalize_to_01",
    "pad_to_divisible",
    "resize_to",
    "save_reward_pairs",
    "warm_latent_pipeline",
]


def __getattr__(name):
    """Lazy re-export of the relocated ControlNet-GRPO loader (issue #176).

    ``load_frozen_controlnet_generator`` moved from this package's
    ``paired_reward_pairs`` module to ``manifold.training.controlnet_inputs``
    (its only consumer is the GRPO real-input path). It stays re-exported here for
    the previously-exposed ``manifold.data`` surface (issue #176 acceptance), but
    imported lazily to avoid a ``data -> training -> data.datamodule`` import cycle
    (the training CLI modules import ``data.datamodule`` at top level).
    """
    if name == "load_frozen_controlnet_generator":
        from ..training.controlnet_inputs import load_frozen_controlnet_generator as _loader

        globals()[name] = _loader  # cache: subsequent access skips __getattr__
        return _loader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
