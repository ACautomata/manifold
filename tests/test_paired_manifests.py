"""Tests for the shared paired-manifest splitter (issue #175).

``_train_val_manifests`` is the ADR-0022 two-way subject split consumed by BOTH the
ControlNet supervised CLI and the GRPO CLI. It was relocated out of the paired-reward
CLI (which a later ticket deletes) into a neutral shared ``data`` home —
``manifold.data.paired_manifests`` — with **no behavior change**: same splits, same
``val_fraction`` mirroring, same native-split-vs-``val_fraction`` fallback.

These tests pin the new import location + the three split-mode behaviors, isolating
the filesystem seams (``build_brats_pair_manifest``) so they run on CPU.
"""

from __future__ import annotations

import omegaconf
import pytest


def _manifest(subjects, prefix="/d"):
    """A fake paired manifest with one src/tgt pair per subject."""
    return [
        {
            "src": f"{prefix}/{s}-t1n.nii.gz",
            "tgt": f"{prefix}/{s}-t1c.nii.gz",
            "src_label": 0,
            "tgt_label": 1,
        }
        for s in subjects
    ]


def test_train_val_manifests_importable_from_data_package():
    """The splitter is exported from the shared ``data`` package (issue #175 AC)."""
    import manifold.data as data_pkg
    from manifold.data import paired_manifests
    from manifold.data.paired_manifests import _train_val_manifests

    assert paired_manifests._train_val_manifests is _train_val_manifests
    # Exported from the package __init__ (AC: "exported from the data package __init__").
    assert data_pkg._train_val_manifests is _train_val_manifests


def test_train_val_manifests_val_fraction_split(monkeypatch):
    """val_data_base_dir unset → subject-level ``val_fraction`` split (ADR-0022)."""
    from manifold.data import paired_brats as pb
    from manifold.data.paired_manifests import _train_val_manifests

    manifest = _manifest([f"s{i}" for i in range(4)])
    # Native-split path not taken (val_data_base_dir unset) → build_brats_pair_manifest
    # must NOT be consulted for a val dir; the split falls to split_brats_pair_manifest.
    monkeypatch.setattr(
        pb,
        "build_brats_pair_manifest",
        lambda *a, **k: pytest.fail(
            "native-split path must not fire when val_data_base_dir is unset"
        ),
    )

    cfg = omegaconf.OmegaConf.create({"val_fraction": 0.25})
    train, val = _train_val_manifests(cfg, manifest)

    train_subjects = {p["src"].split("/")[-1].split("-")[0] for p in train}
    val_subjects = {p["src"].split("/")[-1].split("-")[0] for p in val}
    # 4 subjects, ceil(0.25*4)=1 held out → the LAST sorted subject (s3), disjoint.
    assert val_subjects == {"s3"}
    assert train_subjects == {"s0", "s1", "s2"}
    assert train_subjects.isdisjoint(val_subjects)


def test_train_val_manifests_native_split_dir(monkeypatch, tmp_path):
    """val_data_base_dir an existing BraTS dir → native held-out split returned."""
    from manifold.data import paired_brats as pb
    from manifold.data.paired_manifests import _train_val_manifests

    train_manifest = _manifest([f"t{i}" for i in range(3)], prefix="/train")
    val_manifest = _manifest([f"v{i}" for i in range(2)], prefix="/val")

    val_dir = tmp_path / "val_brats"
    val_dir.mkdir()

    seen = {}

    def fake_build(brats_dir, *a, **k):
        seen["dir"] = brats_dir
        return val_manifest

    monkeypatch.setattr(pb, "build_brats_pair_manifest", fake_build)

    cfg = omegaconf.OmegaConf.create({"val_data_base_dir": str(val_dir), "val_fraction": 0.9})
    train, val = _train_val_manifests(cfg, train_manifest)

    # Native split: train is the full incoming manifest untouched; val is built from
    # the val dir (NOT the val_fraction split — the 0.9 must be ignored).
    assert seen["dir"] == str(val_dir)
    assert train == train_manifest
    assert val == val_manifest


def test_train_val_manifests_native_split_empty_val_raises(monkeypatch, tmp_path):
    """Native-split dir with no pairable subjects → FileNotFoundError (unchanged)."""
    from manifold.data import paired_brats as pb
    from manifold.data.paired_manifests import _train_val_manifests

    val_dir = tmp_path / "empty_brats"
    val_dir.mkdir()
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: [])

    cfg = omegaconf.OmegaConf.create({"val_data_base_dir": str(val_dir)})
    with pytest.raises(FileNotFoundError, match="No paired BraTS volumes"):
        _train_val_manifests(cfg, _manifest(["s0"]))


def test_train_val_manifests_non_directory_val_dir_falls_back(monkeypatch, tmp_path):
    """A non-directory val_data_base_dir (a manifest JSON) → warn + val_fraction split."""
    from manifold.data import paired_brats as pb
    from manifold.data.paired_manifests import _train_val_manifests

    # A FILE (not a dir): the BraTS2023 profile's manifest-JSON case (codex #78).
    not_a_dir = tmp_path / "brats_all_val.json"
    not_a_dir.write_text("[]")

    def _no_native(*a, **k):
        pytest.fail("non-directory val_data_base_dir must not take the native-split path")

    monkeypatch.setattr(pb, "build_brats_pair_manifest", _no_native)

    manifest = _manifest([f"s{i}" for i in range(4)])
    cfg = omegaconf.OmegaConf.create({"val_data_base_dir": str(not_a_dir), "val_fraction": 0.25})
    train, val = _train_val_manifests(cfg, manifest)
    val_subjects = {p["src"].split("/")[-1].split("-")[0] for p in val}
    assert val_subjects == {"s3"}  # fell back to the val_fraction subject split


def test_train_val_manifests_val_fraction_zero_returns_train_fallback():
    """val_fraction 0 (and no val dir) → val=train fallback, empty val (unchanged)."""
    from manifold.data.paired_manifests import _train_val_manifests

    manifest = _manifest(["s0", "s1"])
    cfg = omegaconf.OmegaConf.create({"val_fraction": 0.0})
    train, val = _train_val_manifests(cfg, manifest)
    assert train == manifest
    assert val == []
