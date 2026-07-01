"""Label providers: filename/metadata -> int class label.

BraTS is **one** provider, not baked into trainers. Also retains the central
contrast/modality-code helpers plus a :class:`FixedLabelProvider` /
:class:`ManifestLabelProvider`. Train-time label
augmentation (modality dropout) stays deferred with the trainer.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from .base import LabelProvider

#: Canonical BraTS contrast suffixes (BraTS2023 publication order).
BRATS_CONTRASTS: tuple[str, ...] = ("t1n", "t1c", "t2w", "t2f")

#: Built-in fallback labels — in sync with ``configs/modality_mapping.json``'s
#: ``brats2023_*`` keys (34/35/36/37), distinct from upstream skull-stripped
#: codes (29/17/30/31) so BraTS2023 fine-tuning gets fresh class-embedding rows.
DEFAULT_BRATS_LABELS: dict[str, int] = {
    "t1n": 34,
    "t1c": 35,
    "t2w": 36,
    "t2f": 37,
}


def load_brats_labels(modality_mapping_path: str | os.PathLike | None) -> dict[str, int]:
    """Load ``{contrast: label}`` from the central modality-mapping JSON.

    Reads each ``brats2023_<contrast>`` key from the flat code dict in
    *modality_mapping_path* and returns ``{contrast: int(value)}``. Returns a
    copy of :data:`DEFAULT_BRATS_LABELS` when the file is missing or has no
    ``brats2023_*`` keys, so callers always get a usable mapping.
    """
    if not modality_mapping_path or not os.path.exists(modality_mapping_path):
        return dict(DEFAULT_BRATS_LABELS)
    with open(modality_mapping_path) as f:
        codes = json.load(f)
    out: dict[str, int] = {}
    for contrast in BRATS_CONTRASTS:
        key = f"brats2023_{contrast}"
        if key in codes:
            out[contrast] = int(codes[key])
    return out or dict(DEFAULT_BRATS_LABELS)


def detect_brats_contrast(filename: str) -> str | None:
    """Detect a BraTS contrast from a filename (``...-t1n.nii.gz`` -> ``t1n``).

    Matches a contrast key from :data:`BRATS_CONTRASTS` as a hyphen- or
    underscore-separated suffix of the stem. Returns ``None`` when no known
    contrast is found (segmentation masks, unknown modalities, …).
    """
    stem = filename.lower()
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    for contrast in BRATS_CONTRASTS:
        if stem.endswith(f"-{contrast}") or stem.endswith(f"_{contrast}") or stem == contrast:
            return contrast
    return None


class BratsLabelProvider:
    """Map a BraTS filename suffix to its class label via ``{contrast: label}``.

    Files without a known contrast (e.g. ``*-seg.nii.gz`` masks) map to ``None``
    and are skipped.
    """

    def __init__(
        self,
        labels: Mapping[str, int] | None = None,
        *,
        modality_mapping_path: str | os.PathLike | None = None,
    ) -> None:
        self.labels: dict[str, int] = (
            dict(labels) if labels is not None else load_brats_labels(modality_mapping_path)
        )

    def __call__(self, filename: str, meta: dict[str, Any]) -> int | None:  # noqa: ARG002
        contrast = detect_brats_contrast(filename)
        return None if contrast is None else self.labels.get(contrast)


class ManifestLabelProvider:
    """Read the label from manifest metadata (``meta[field]``).

    For datasets whose label is an explicit per-item field rather than encoded
    in the filename. Falls back to *default* when absent.
    """

    def __init__(self, field: str = "label", default: int | None = None) -> None:
        self.field = field
        self.default = default

    def __call__(self, filename: str, meta: dict[str, Any]) -> int | None:  # noqa: ARG002
        val = meta.get(self.field, self.default)
        return None if val is None else int(val)


class FixedLabelProvider:
    """Return one constant label (single-contrast / unconditional datasets)."""

    def __init__(self, label: int) -> None:
        self.label = int(label)

    def __call__(self, filename: str, meta: dict[str, Any]) -> int | None:  # noqa: ARG002
        return self.label


def label_provider_from_config(
    args: Any,
    *,
    include_modality: bool,
    default_modality: int = 0,
) -> LabelProvider:
    """Pick a label provider from the merged config.

    Selection is **explicit** via ``args.dataset_type`` (``"brats"`` or
    ``"fixed"``); presence of ``modality_mapping_path`` alone is NOT enough —
    every shipped env config sets that path, but only BraTS-named files should map
    through :class:`BratsLabelProvider`. CT/MR datasets without contrast suffixes
    use :class:`FixedLabelProvider` so :class:`NiftiVolumeDataset` keeps every
    file (otherwise the dataset comes up empty).

    Defaults to ``"fixed"`` (the safe choice for arbitrary inputs); set
    ``dataset_type: brats`` in the env config to opt in to filename-based contrast
    detection.
    """
    dataset_type = str(getattr(args, "dataset_type", "fixed")).lower()
    if include_modality and dataset_type == "brats":
        mapping_path = getattr(args, "modality_mapping_path", None)
        return BratsLabelProvider(modality_mapping_path=mapping_path)
    return FixedLabelProvider(default_modality)
