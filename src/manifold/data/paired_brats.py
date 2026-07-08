"""BraTS pair builder: a directory of BraTS volumes -> the paired manifest.

Walks a BraTS directory, groups volumes by subject, and emits all ordered
within-subject contrast pairs — for a 4-contrast subject (``t1n, t1c, t2w, t2f``)
that is the ``4 × 3 = 12`` non-self permutations (ADR-0014 — any-to-any pairing;
a single model serves every contrast direction via the summed-label embedding).

The builder is the **only** BraTS-specific code on the paired path:
:class:`~manifold.data.PairedNiftiVolumeDataset` is dataset-agnostic and consumes
the manifest this produces. Subject grouping uses the BraTS filename convention
``<subject>-<contrast>.nii.gz`` where ``<subject>`` is everything before the
trailing ``-t1n`` / ``-t1c`` / ``-t2w`` / ``-t2f`` suffix (detected via
:func:`manifold.data.detect_brats_contrast`). Files with no detected contrast
(segmentation masks, unknown modalities) are dropped, and subjects missing any of
the four contrasts are skipped entirely (a partial subject yields zero pairs).
"""

from __future__ import annotations

import math
import os
from typing import Any

from .labels import BRATS_CONTRASTS, DEFAULT_BRATS_LABELS, detect_brats_contrast
from .volume_dataset import collect_nifti_paths


def _split_subject_contrast(filename: str) -> tuple[str | None, str | None]:
    """Split ``<subject>-<contrast>.nii.gz`` into ``(subject, contrast)``.

    The contrast suffix is detected case-insensitively (mirroring
    :func:`~manifold.data.detect_brats_contrast`), but the returned *subject* keeps
    the original filename casing. A file with no known contrast returns
    ``(None, None)`` (segmentation masks, unknown modalities — the caller drops it).
    """
    stem = filename
    lower = filename.lower()
    for ext in (".nii.gz", ".nii"):
        if lower.endswith(ext):
            stem = stem[: -len(ext)]
            lower = lower[: -len(ext)]
            break
    for contrast in BRATS_CONTRASTS:
        if lower == contrast:
            return "", contrast
        if lower.endswith(f"-{contrast}") or lower.endswith(f"_{contrast}"):
            # Strip the separator + contrast token, keeping the original-cased subject.
            return stem[: -(len(contrast) + 1)], contrast
    return None, None


def build_brats_pair_manifest(
    brats_dir: str,
    labels: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Build the paired manifest for a BraTS directory.

    Walks *brats_dir* recursively for ``.nii``/``.nii.gz``, groups files by
    subject, and for each subject that has all four contrasts (``t1n, t1c, t2w,
    t2f``) emits the 12 ordered within-subject contrast pairs (self-pairs
    excluded). Files with no detected contrast are dropped; subjects missing any
    contrast are skipped.

    Args:
        brats_dir: directory scanned recursively for BraTS NIfTIs.
        labels: ``{contrast: int_label}`` mapping (defaults to
            :data:`~manifold.data.DEFAULT_BRATS_LABELS`).

    Returns:
        A list of ``{"src","tgt","src_label","tgt_label"}`` dicts ready for
        :class:`~manifold.data.PairedNiftiVolumeDataset`. The order is deterministic
        (subjects sorted, contrasts in :data:`BRATS_CONTRASTS` order).
    """
    label_map = dict(labels) if labels is not None else dict(DEFAULT_BRATS_LABELS)
    # subject -> {contrast: abspath}; insertion order is the directory scan order.
    per_subject: dict[str, dict[str, str]] = {}
    for path in collect_nifti_paths(brats_dir):
        filename = os.path.basename(path)
        subject, contrast = _split_subject_contrast(filename)
        if subject is None or contrast is None:
            continue  # seg mask or unknown modality — dropped
        if contrast not in label_map:
            continue
        per_subject.setdefault(subject, {})[contrast] = path

    manifest: list[dict[str, Any]] = []
    for subject in sorted(per_subject):
        present = per_subject[subject]
        # A subject must have ALL four contrasts to be pairable; a partial subject
        # contributes zero pairs (no half-built pairs).
        if any(c not in present for c in BRATS_CONTRASTS):
            continue
        for src_c in BRATS_CONTRASTS:
            for tgt_c in BRATS_CONTRASTS:
                if src_c == tgt_c:
                    continue  # self-pair excluded
                manifest.append(
                    {
                        "src": present[src_c],
                        "tgt": present[tgt_c],
                        "src_label": label_map[src_c],
                        "tgt_label": label_map[tgt_c],
                    }
                )
    return manifest


def split_brats_pair_manifest(
    manifest: list[dict[str, Any]],
    val_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a BraTS pair manifest into ``(train, val)`` by **subject**.

    Groups the pairs by subject (re-derived from each pair's ``src`` path via
    :func:`_split_subject_contrast`), sorts the subjects deterministically, and
    assigns the **last** ``ceil(val_fraction · n_subjects)`` subjects to val, the
    rest to train. Because the split is by subject, no subject's volume appears in
    both splits → no train/val leakage: every contrast of a held-out subject is
    held out, whether it appears as the src or the tgt of any pair.

    The "last subjects" choice (not a random draw) keeps the split reproducible
    with no RNG seed and stable across runs/resumes, so runs are directly
    comparable on the same held-out set. BraTS-GLI subject IDs are arbitrary
    filename labels (not ordered by site/scanner), so a fixed contiguous block is
    an unbiased held-out set.

    Args:
        manifest: the full :func:`build_brats_pair_manifest` output.
        val_fraction: fraction of subjects to hold out (``0 < f``).
            ``<= 0`` → all subjects in train, empty val (the val=train fallback);
            ``>= 1`` → all-but-one in val (always keeps ≥1 train subject).

    Returns:
        ``(train_manifest, val_manifest)`` — each a list of the same
        ``{"src","tgt","src_label","tgt_label"}`` dicts, over disjoint subject sets.
        ``train_manifest`` is never empty when the input has ≥1 subject (a single
        subject with ``val_fraction > 0`` stays in train), so the train-only scale
        estimate downstream never faces an empty cache.
    """
    if val_fraction <= 0.0:
        return list(manifest), []
    per_subject: dict[str, list[dict[str, Any]]] = {}
    for item in manifest:
        subject, _ = _split_subject_contrast(os.path.basename(str(item["src"])))
        if subject is None:
            continue  # malformed pair (no detectable contrast) — skip, matches the builder
        per_subject.setdefault(subject, []).append(item)
    subjects = sorted(per_subject)
    if not subjects:
        return [], []
    if val_fraction >= 1.0:
        n_val = len(subjects)
    else:
        n_val = max(1, math.ceil(val_fraction * len(subjects)))
    # Always keep ≥1 train subject: a single subject with val_fraction>0 stays in
    # train, and val_fraction>=1 holds out all-but-one. Prevents an empty train
    # dataset (and an opaque torch.stack([]) crash in the train-only scale estimate).
    n_val = min(n_val, len(subjects) - 1)
    val_subjects = set(subjects[-n_val:]) if n_val > 0 else set()
    train_manifest: list[dict[str, Any]] = []
    val_manifest: list[dict[str, Any]] = []
    for s in subjects:
        (val_manifest if s in val_subjects else train_manifest).extend(per_subject[s])
    return train_manifest, val_manifest


__all__ = ["build_brats_pair_manifest", "split_brats_pair_manifest"]
