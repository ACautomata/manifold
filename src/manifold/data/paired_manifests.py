"""Shared paired-manifest subject splitter (ADR-0022, issue #175).

``_train_val_manifests`` resolves the (train, val) paired manifests from the
configured split mode. It is the ADR-0022 two-way subject split consumed by **both**
the ControlNet supervised CLI and the GRPO CLI — a shared paired-manifest helper,
not paired-reward-specific. It lives here (a neutral ``data`` home) so the
paired-reward pipeline's deletion (ADR-0034) can drop the paired-reward CLI without
taking the splitter with it. Relocated verbatim from the paired-reward CLI — no
behavior change (same splits, same ``val_fraction`` mirroring, same
native-split-vs-``val_fraction`` fallback).
"""

from __future__ import annotations

from lightning.pytorch.utilities.rank_zero import rank_zero_info

from ..config import opt


def _train_val_manifests(cfg, manifest):
    """Resolve the (train, val) paired manifests from the configured split mode.

    Two mutually-exclusive modes (mirrors the ``val_data_base_dir`` /
    ``val_fraction`` env-config contract):

    - ``cfg.val_data_base_dir`` set AND an existing directory → the **native
      held-out split**: ``manifest`` (built from ``data_base_dir``) is the full
      train set, and val is built from ``val_data_base_dir`` — a BraTS directory
      in the same form as ``data_base_dir`` (NOT a manifest JSON; the paired path
      is BraTS-dir-based via :func:`build_brats_pair_manifest`). Use this when the
      dataset ships its own disjoint train/val (e.g. BraTS-2024-GLI's 1621 train /
      188 val) — the organizer-split subjects are disjoint, so there is no
      train/val leakage. A non-directory ``val_data_base_dir`` (e.g. the manifest
      JSON the BraTS2023 profile sets) is ignored with a warning and falls back to
      ``val_fraction`` (the pre-native-split behavior).
    - otherwise → ``cfg.val_fraction`` subject-level split of ``manifest``
      (``0`` → val=train fallback). A ``null``/``???``/absent
      ``val_data_base_dir`` reads as unset via :func:`~manifold.config.opt`.

    Inlined from the deleted ``paired_cli`` (T8); relocated to the shared ``data``
    package (issue #175) so the paired-reward CLI's deletion does not take the
    shared splitter with it.
    """
    import os

    from .paired_brats import build_brats_pair_manifest, split_brats_pair_manifest

    val_dir = opt(cfg, "val_data_base_dir", None)
    # The native-split path needs a BraTS *directory* (build_brats_pair_manifest
    # scans NIfTIs); a manifest JSON (e.g. the BraTS2023 profile's
    # brats_all_val.json) or a missing path is not usable here. Fall back to
    # val_fraction (the pre-native-split behavior) instead of building an empty
    # val set and crashing (codex #78, P1).
    if val_dir and os.path.isdir(str(val_dir)):
        val_manifest = build_brats_pair_manifest(str(val_dir))
        if not val_manifest:
            raise FileNotFoundError(
                f"No paired BraTS volumes found under val_data_base_dir={val_dir} "
                f"(need >=1 subject with all 4 contrasts)."
            )
        return manifest, val_manifest
    if val_dir:
        rank_zero_info(
            "paired val_data_base_dir=%s is not a directory; the native train/val "
            "split needs a BraTS directory (not a manifest JSON). Falling back to "
            "the val_fraction subject split.",
            val_dir,
        )
    val_fraction = float(opt(cfg, "val_fraction", 0.0))
    return split_brats_pair_manifest(manifest, val_fraction)


__all__ = ["_train_val_manifests"]
