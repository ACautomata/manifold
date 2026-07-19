"""Paired NIfTI volume dataset emitting both src and tgt images per pair.

A dataset-agnostic pair reader: it consumes a manifest of
``(src, tgt, src_label, tgt_label)`` tuples (paths + integer contrast labels) and
emits both endpoints' images in one ``__getitem__`` so the latent-prep stack can
encode each **unique volume once** and share it across every pair that references
it (ADR-0014 — no 12× duplication per 4-contrast subject).

The class knows nothing about BraTS: contrast enumeration / subject grouping lives
in :func:`manifold.data.build_brats_pair_manifest`, which produces the manifest.
NIfTI loading and the preprocessing transforms (RAS reorientation, intensity
normalization, resize, pad-to-divisible) are verbatim from
:class:`~manifold.data.NiftiVolumeDataset` — the latents must match a
hope-trained model's training distribution, so the transforms are reused, not
re-derived (the volume-side twin of ADR-0013's transport reuse).

The ``sample_id`` (``f"{basename}__{sha1(abspath)[:12]}"``) is the collision-free
cache key shared with :class:`~manifold.data.PairedLatentDataset`; both endpoints
of a pair are deduped into a single ``{sample_id: (path, label)}`` map so the
encode cost is independent of pair count.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info

from .base import MedicalDataset
from .transforms import normalize_to_01, pad_to_divisible, resize_to


class PairedNiftiVolumeDataset(MedicalDataset):
    """Dataset of ``(src, tgt)`` NIfTI volume pairs -> the paired image contract.

    Emits ``{src_image, tgt_image, spacing, src_label, tgt_label, src_id,
    tgt_id}`` where both images are ``(1, D0, D1, D2)`` floats normalized to ~[0, 1]
    and ``spacing`` is the source's voxel size (BraTS src/tgt are co-registered, so
    src spacing == tgt spacing). The pair's two ``sample_id``\\ s are looked up in a
    deduped ``{sample_id: (path, label)}`` volume map built once at construction.

    Args:
        manifest: list of ``{"src","tgt","src_label","tgt_label"}`` dicts (paths +
            integer contrast labels). Relative ``src``/``tgt`` paths resolve
            against *data_base_dir* when given.
        target_dim: every volume is trilinear-resized to this ``(D0, D1, D2)`` so
            all samples yield a uniform latent shape (required for ``batch_size>1``).
        divisor: VAE spatial downsampling factor; volumes are zero-padded to a
            multiple of it as a safety net.
        data_base_dir: base directory to resolve relative ``src``/``tgt`` paths
            (absolute paths used as-is). Defaults to the CWD.
    """

    def __init__(
        self,
        manifest: list[dict[str, Any]],
        target_dim: tuple[int, int, int],
        divisor: int,
        data_base_dir: str | None = None,
    ) -> None:
        self.target_dim: tuple[int, int, int] = (
            int(target_dim[0]),
            int(target_dim[1]),
            int(target_dim[2]),
        )
        self.divisor = int(divisor)
        self.data_base_dir = data_base_dir
        self._pairs: list[dict[str, Any]] = []
        #: Deduped volume map — one entry per unique ``(subject, contrast)`` volume.
        #: The latent cache iterates :meth:`unique_sample_ids` and encodes each
        #: exactly once, however many pairs reference it (ADR-0014 shared cache).
        self._volumes: dict[str, tuple[str, int]] = {}
        self._build(manifest)

    # -- construction --------------------------------------------------------

    def _build(self, manifest: list[dict[str, Any]]) -> None:
        base = self.data_base_dir or "."
        for item in manifest:
            src_raw = str(item["src"])
            tgt_raw = str(item["tgt"])
            src_path = src_raw if os.path.isabs(src_raw) else os.path.join(base, src_raw)
            tgt_path = tgt_raw if os.path.isabs(tgt_raw) else os.path.join(base, tgt_raw)
            src_label = int(item["src_label"])
            tgt_label = int(item["tgt_label"])
            src_id = self._sample_id(src_path)
            tgt_id = self._sample_id(tgt_path)
            self._register_volume(src_id, src_path, src_label)
            self._register_volume(tgt_id, tgt_path, tgt_label)
            self._pairs.append(
                {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "src_label": src_label,
                    "tgt_label": tgt_label,
                }
            )
        if self._pairs:
            rank_zero_info(
                f"PairedNiftiVolumeDataset: {len(self._pairs)} pairs over "
                f"{len(self._volumes)} unique volumes "
                f"(encode cost = unique volumes, not pairs)."
            )

    def _register_volume(self, sample_id: str, path: str, label: int) -> None:
        """Insert into the deduped volume map, enforcing label consistency.

        The same volume (by ``sample_id``) may appear as src of one pair and tgt
        of another; its label must be identical every time (a contrast is a fixed
        property of the file). A mismatch means the manifest is malformed.
        """
        existing = self._volumes.get(sample_id)
        if existing is None:
            self._volumes[sample_id] = (path, label)
            return
        _, existing_label = existing
        if existing_label != label:
            raise ValueError(
                f"PairedNiftiVolumeDataset: label mismatch for {sample_id} "
                f"({existing_label} vs {label}) — a volume's contrast label must "
                "be consistent across all pairs that reference it."
            )

    @staticmethod
    def _sample_id(path: str) -> str:
        """``f"{basename}__{sha1(abspath)[:12]}"`` — the shared latent-cache key."""
        digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
        return f"{os.path.basename(path)}__{digest}"

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self._pairs[index]
        src = self._load_volume(pair["src_id"])
        tgt = self._load_volume(pair["tgt_id"])
        return {
            "src_image": src["image"],
            "tgt_image": tgt["image"],
            # BraTS volumes are co-registered → src spacing == tgt spacing.
            "spacing": src["spacing"],
            "src_label": torch.tensor(pair["src_label"], dtype=torch.long),
            "tgt_label": torch.tensor(pair["tgt_label"], dtype=torch.long),
            "src_id": pair["src_id"],
            "tgt_id": pair["tgt_id"],
        }

    def pair_meta(self, index: int) -> dict[str, Any]:
        """The pair's metadata WITHOUT loading volumes (the latent dataset's hot path).

        Returns ``{src_id, tgt_id, src_label, tgt_label}`` from the in-memory pair
        list — no NIfTI read, no transforms. :class:`PairedLatentDataset.__getitem__`
        uses this (not :meth:`__getitem__`) once the cache is warm, so a training-
        batch fetch is pure RAM lookup: the latents and spacing already live in the
        shared ``{sample_id: item}`` cache. :meth:`__getitem__` (which loads both
        volumes) stays for the image contract + inspection.
        """
        pair = self._pairs[index]
        return {
            "src_id": pair["src_id"],
            "tgt_id": pair["tgt_id"],
            "src_label": torch.tensor(pair["src_label"], dtype=torch.long),
            "tgt_label": torch.tensor(pair["tgt_label"], dtype=torch.long),
        }

    # -- introspection -------------------------------------------------------

    def unique_sample_ids(self) -> list[str]:
        """The deduped volume set — the latent cache iterates this exactly once."""
        return list(self._volumes.keys())

    @property
    def paths(self) -> list[str]:
        return [path for path, _ in self._volumes.values()]

    # -- internals -----------------------------------------------------------

    def _load_volume(self, sample_id: str) -> dict[str, Any]:
        """Load + preprocess one volume by ``sample_id`` (the nibabel sequence).

        Verbatim from :class:`~manifold.data.NiftiVolumeDataset`: RAS-reorient via
        ``as_closest_canonical``, ``get_fdata``, voxel sizes off the affine, then
        ``normalize_to_01`` → ``resize_to`` → ``pad_to_divisible`` (the same
        transforms the noise→data path uses, so a paired latent matches the same
        VAE training distribution).
        """
        path, label = self._volumes[sample_id]
        img = nib.as_closest_canonical(nib.load(path))
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        spacing = np.asarray(nib.affines.voxel_sizes(img.affine), dtype=np.float32)
        # The class label doubles as the modality code for intensity windowing
        # (>=8 -> MR percentile, <8 -> CT HU window) — exactly as NiftiVolumeDataset.
        data = normalize_to_01(data, label)
        data = resize_to(data, self.target_dim)
        data, _ = pad_to_divisible(data, self.divisor)
        image = torch.from_numpy(np.ascontiguousarray(data)).float().unsqueeze(0)  # [1, D0, D1, D2]
        return {
            "image": image,
            "spacing": torch.from_numpy(spacing).float(),
            "label": label,
        }


__all__ = ["PairedNiftiVolumeDataset"]
