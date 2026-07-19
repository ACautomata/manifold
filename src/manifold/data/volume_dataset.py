"""NIfTI volume dataset emitting ``SampleDict`` (image + spacing + label).

All dataset-specific knowledge (NIfTI loading, RAS reorientation, intensity
normalization, resize, manifest/dir resolution, label assignment) lives HERE —
trainers only ever see the dict. The label comes from a pluggable
:class:`~manifold.data.LabelProvider`, so BraTS / CT / MR / fixed-label are each
just a different provider passed in at construction.

Ported from hope's ``data/volume_dataset.py`` (transforms verbatim — ADR: the
latents must match a hope-trained model's training distribution).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from .base import LabelProvider, MedicalDataset
from .transforms import normalize_to_01, pad_to_divisible, resize_to


def collect_nifti_paths(input_path: str) -> list[str]:
    """Collect ``.nii`` / ``.nii.gz`` paths from *input_path* (file or directory)."""
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        gz = {str(f) for f in p.rglob("*.nii.gz")}
        nii = {str(f) for f in p.rglob("*.nii")}
        return sorted(gz | nii)
    raise FileNotFoundError(f"Input path not found: {input_path}")


def _load_manifest(source: str) -> list[dict[str, Any]]:
    """Read a MONAI-style manifest: ``{"training": [{...}, ...]}`` or a bare list."""
    with open(source) as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("training", "test", "validation"):
            if key in data and isinstance(data[key], list):
                return data[key]
        return [data]  # fallback: treat the dict as a single item list
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported manifest format in {source}: {type(data).__name__}")


class NiftiVolumeDataset(MedicalDataset):
    """Dataset of 3D NIfTI volumes -> ``{"image","spacing","label","sample_id","meta"}``.

    Args:
        source: a manifest JSON path (``{"training":[{"image", ...}]}`` or a list)
            or a directory scanned recursively for ``.nii``/``.nii.gz``.
        label_provider: maps ``(filename, meta)`` to an int class label, or
            ``None`` to skip the file (e.g. BraTS segmentation masks).
        target_dim: every volume is trilinear-resized to this ``(D0, D1, D2)`` so
            all samples yield a uniform latent shape (required for ``batch_size>1``).
        divisor: VAE spatial downsampling factor; volumes are zero-padded to a
            multiple of it as a safety net.
        data_base_dir: base directory to resolve relative ``image`` paths in a
            manifest (absolute paths used as-is).
    """

    def __init__(
        self,
        source: str,
        label_provider: LabelProvider,
        target_dim: tuple[int, int, int],
        divisor: int,
        data_base_dir: str | None = None,
    ) -> None:
        self.label_provider = label_provider
        self.target_dim: tuple[int, int, int] = (
            int(target_dim[0]),
            int(target_dim[1]),
            int(target_dim[2]),
        )
        self.divisor = int(divisor)
        self._items: list[tuple[str, dict[str, Any], int]] = self._collect(source, data_base_dir)

    # -- construction --------------------------------------------------------

    def _collect(self, source: str, base_dir: str | None) -> list[tuple[str, dict[str, Any], int]]:
        """Build ``(path, meta, label)`` triples, dropping files the provider skips.

        Relative manifest ``image`` paths resolve against the manifest's own
        directory unless the caller gave a distinct directory ``base_dir``.
        """
        entries: list[tuple[str, dict[str, Any]]] = []
        if os.path.isfile(source) and source.endswith(".json"):
            manifest_dir = os.path.dirname(os.path.abspath(source))
            if base_dir is None or base_dir == source or not os.path.isdir(base_dir):
                rel_base = manifest_dir
            else:
                rel_base = base_dir
            for item in _load_manifest(source):
                img = item["image"] if isinstance(item, dict) else str(item)
                meta = dict(item) if isinstance(item, dict) else {}
                path = img if os.path.isabs(img) else os.path.join(rel_base, img)
                entries.append((path, meta))
        else:
            for path in collect_nifti_paths(source):
                entries.append((path, {}))

        items: list[tuple[str, dict[str, Any], int]] = []
        skipped = 0
        for path, meta in entries:
            label = self.label_provider(os.path.basename(path), meta)
            if label is None or not os.path.exists(path):
                skipped += 1
                continue
            items.append((path, dict(meta), int(label)))
        if skipped:
            rank_zero_info(
                f"NiftiVolumeDataset: kept {len(items)}, skipped {skipped} "
                f"file(s) (no label from provider or missing on disk)."
            )
        return items

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path, meta, label = self._items[index]
        img = nib.as_closest_canonical(nib.load(path))
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        spacing = np.asarray(nib.affines.voxel_sizes(img.affine), dtype=np.float32)

        # The class label doubles as the modality code for intensity windowing
        # (>=8 -> MR percentile, <8 -> CT HU window), exactly as the original
        # trainer did (BraTS labels 34-37 take the MR branch).
        data = normalize_to_01(data, label)
        data = resize_to(data, self.target_dim)
        data, _ = pad_to_divisible(data, self.divisor)

        image = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0)  # [1, D0, D1, D2]
        # ``sample_id`` doubles as the latent-cache key (see LatentDataset). Two
        # NIfTIs in different subject dirs often share a basename; append a short
        # hash of the absolute path so the key is collision-free yet grep-able.
        digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
        sample_id = f"{os.path.basename(path)}__{digest}"
        return {
            "image": image,
            "spacing": torch.from_numpy(spacing).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
            "meta": meta,
        }

    def label_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, _, label in self._items:
            counts[label] = counts.get(label, 0) + 1
        return counts

    # -- introspection -------------------------------------------------------

    @property
    def paths(self) -> list[str]:
        return [p for p, _, _ in self._items]

    def sample_ids(self) -> list[str]:
        """Stable, collision-free per-volume IDs (basename + path hash)."""
        return [
            f"{os.path.basename(p)}__{hashlib.sha1(os.path.abspath(p).encode('utf-8')).hexdigest()[:12]}"
            for p, _, _ in self._items
        ]
