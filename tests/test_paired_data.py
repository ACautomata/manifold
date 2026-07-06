"""Paired JiT data-stack tests (Seam 3 + Seam 4 + integration, issue #67).

Covers the four acceptance criteria:

- **Seam #3 (data contract):** :class:`PairedNiftiVolumeDataset` emits
  ``(src_image, tgt_image, spacing, src_label, tgt_label, src_id, tgt_id)`` and
  :class:`PairedLatentDataset` emits ``(src_latent, tgt_latent, src_label,
  tgt_label, spacing)`` with BOTH latents scaled (scale-on-read, ADR-0003).
- **Shared cache:** warming a 12-pair / 1-subject dataset encodes each of the 4
  unique volumes EXACTLY ONCE — not 12×2 (ADR-0014 — latents cached per
  ``(subject, contrast)`` and shared across the pairs that reference them).
- **scale_factor pooled:** :func:`estimate_paired_scale_factor` sets
  ``vae.scaling_factor = 1/std(z)`` over the UNION of unique latents (src∪tgt by
  construction); both src and tgt ``__getitem__`` latents equal ``raw * scale``.
- **Seam #4 (BraTS builder):** a 4-contrast subject yields exactly 12 ordered
  pairs (self-pairs excluded), the ``-seg`` mask is dropped, and a subject
  missing a contrast is skipped entirely.

The integration test feeds a warmed paired batch through
:class:`~manifold.PairedLatentFlowModule` for one ``fit`` step and asserts a
finite loss with a working ``.backward()`` — the batch contract matches the
Module (no contract mismatch). CPU, synthetic, fake ``encode_fn``, ``tmp_path``;
all fixtures local.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    PairedLatentFlowModule,
    UNet3DConditionModel,
)
from manifold.data import (
    DEFAULT_BRATS_LABELS,
    PairedLatentDataset,
    PairedNiftiVolumeDataset,
    build_brats_pair_manifest,
    estimate_paired_scale_factor,
)


def _write_nifti(path: str, shape=(10, 10, 6)) -> None:
    """Write a fake NIfTI with a ``(1,1,2,1)`` affine (voxel size 2 on the Z axis)."""
    arr = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    nib.Nifti1Image(arr, affine=np.diag([1.0, 1.0, 2.0, 1.0])).to_filename(path)


def _mock_encode_fn(scale_std: float = 3.0):
    """A deterministic-per-RNG-state unscaled-latent encoder (std != 1 → scale != 1).

    Distinct from ``tests.test_data._mock_encode_fn``: it does NOT reset the seed
    per call, so each unique volume gets a distinct latent (``std`` over the cache
    is non-zero, so ``estimate_paired_scale_factor`` is finite). Seed once outside.
    """

    def fn(images: torch.Tensor) -> torch.Tensor:
        return torch.randn(images.shape[0], 4, 4, 4, 4) * scale_std

    return fn


def _counting_encode_fn(counter: list[int], scale_std: float = 3.0):
    """An encode fn that counts how many times it runs (to assert unique reuse)."""

    def fn(images: torch.Tensor) -> torch.Tensor:
        counter[0] += 1
        return torch.randn(images.shape[0], 4, 4, 4, 4) * scale_std

    return fn


def _write_brats_subject(dir_, contrasts=("t1n", "t1c", "t2w", "t2f"), subject="BraTS-GLI-0000-000") -> None:
    for c in contrasts:
        _write_nifti(str(dir_ / f"{subject}-{c}.nii.gz"))


# -- Seam #3: PairedNiftiVolumeDataset contract --------------------------------


def test_paired_volume_dataset_emits_contract(tmp_path) -> None:
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 34, "tgt_label": 35}]

    ds = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)
    assert len(ds) == 1
    assert len(ds.unique_sample_ids()) == 2  # src + tgt are distinct files

    item = ds[0]
    assert {"src_image", "tgt_image", "spacing", "src_label", "tgt_label", "src_id", "tgt_id"} <= set(item)
    assert item["src_image"].shape == (1, 8, 8, 8)
    assert item["tgt_image"].shape == (1, 8, 8, 8)
    assert item["spacing"].shape == (3,)
    assert item["spacing"][2].item() == pytest.approx(2.0)  # read off the affine
    assert item["src_label"].dtype == torch.long
    assert item["src_label"].item() == 34
    assert item["tgt_label"].item() == 35
    assert item["src_id"].startswith("a-t1n.nii.gz__") and "__" in item["src_id"]
    assert item["tgt_id"].startswith("a-t1c.nii.gz__") and "__" in item["tgt_id"]


def test_paired_volume_dataset_dedups_shared_volumes(tmp_path) -> None:
    """A volume that is tgt of one pair and src of another is stored ONCE."""
    a = tmp_path / "a-t1n.nii.gz"
    b = tmp_path / "b-t1c.nii.gz"
    c = tmp_path / "c-t2w.nii.gz"
    for p in (a, b, c):
        _write_nifti(str(p))
    manifest = [
        {"src": str(a), "tgt": str(b), "src_label": 34, "tgt_label": 35},
        {"src": str(b), "tgt": str(c), "src_label": 35, "tgt_label": 36},
    ]
    ds = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)
    assert len(ds) == 2
    assert len(ds.unique_sample_ids()) == 3  # not 4 — `b` is shared


def test_paired_volume_dataset_rejects_inconsistent_labels(tmp_path) -> None:
    """The same volume must carry one label across all pairs (contrast is a file property)."""
    a = tmp_path / "a-t1n.nii.gz"
    b = tmp_path / "b-t1c.nii.gz"
    _write_nifti(str(a))
    _write_nifti(str(b))
    manifest = [
        {"src": str(a), "tgt": str(b), "src_label": 34, "tgt_label": 35},
        {"src": str(b), "tgt": str(a), "src_label": 99, "tgt_label": 34},  # b now 99, was 35
    ]
    with pytest.raises(ValueError, match="label mismatch"):
        PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)


# -- Seam #3: PairedLatentDataset contract + scale-on-read ----------------------


def test_paired_latent_dataset_requires_warm_before_getitem(tmp_path) -> None:
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 34, "tgt_label": 35}]
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)
    ds = PairedLatentDataset(vol, encode_fn=_mock_encode_fn())
    with pytest.raises(RuntimeError, match="warm_cache"):
        _ = ds[0]


def test_paired_latent_dataset_emits_contract(tmp_path) -> None:
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 34, "tgt_label": 35}]
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)

    torch.manual_seed(0)
    ds = PairedLatentDataset(vol, encode_fn=_mock_encode_fn())
    ds.warm_cache(torch.device("cpu"), show_progress=False)

    item = ds[0]
    assert {"src_latent", "tgt_latent", "src_label", "tgt_label", "spacing"} <= set(item)
    assert item["src_latent"].shape == (4, 4, 4, 4)
    assert item["tgt_latent"].shape == (4, 4, 4, 4)
    assert item["src_label"].dtype == torch.long
    assert item["src_label"].item() == 34
    assert item["tgt_label"].item() == 35
    assert torch.isfinite(item["src_latent"]).all()
    assert torch.isfinite(item["tgt_latent"]).all()


def test_scale_on_read_applies_to_both_src_and_tgt(tmp_path) -> None:
    """scaling_factor multiplies BOTH endpoints at __getitem__ (ADR-0003 scale-on-read)."""
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 34, "tgt_label": 35}]
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)

    torch.manual_seed(0)
    ds = PairedLatentDataset(vol, encode_fn=_mock_encode_fn())
    ds.warm_cache(torch.device("cpu"), show_progress=False)
    ds.scaling_factor = 2.5  # arbitrary, != 1 — exercises scale-on-read for both endpoints

    pair = vol[0]
    item = ds[0]
    assert torch.allclose(item["src_latent"], ds.raw_latent(pair["src_id"]) * 2.5)
    assert torch.allclose(item["tgt_latent"], ds.raw_latent(pair["tgt_id"]) * 2.5)


def test_getitem_serves_from_ram_without_loading_volumes(tmp_path) -> None:
    """A warmed ``__getitem__`` does ZERO NIfTI reads (the training hot path).

    Once the shared cache is warm, fetching a training batch must be pure RAM
    lookup — ``pair_meta`` skips the volume dataset's volume-loading ``__getitem__``
    and the spacing is read from the cached latent item. Spying on the volume
    dataset's ``_load_volume`` asserts the train-time fetch never touches disk.
    """
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 34, "tgt_label": 35}]
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)

    torch.manual_seed(0)
    ds = PairedLatentDataset(vol, encode_fn=_mock_encode_fn())
    ds.warm_cache(torch.device("cpu"), show_progress=False)

    calls = {"n": 0}
    real = vol._load_volume

    def spy(sample_id):  # noqa: ANN001
        calls["n"] += 1
        return real(sample_id)

    vol._load_volume = spy
    try:
        item = ds[0]  # a train-time fetch
    finally:
        vol._load_volume = real
    assert calls["n"] == 0  # no NIfTI read at train time
    # Spacing still arrives (from the cached src volume, captured at warm time).
    assert item["spacing"].shape == (3,)
    assert torch.isfinite(item["spacing"]).all()


# -- Shared cache: encode each unique volume once -----------------------------


def test_shared_cache_encodes_each_unique_volume_once(tmp_path) -> None:
    """12 pairs / 1 subject / 4 contrasts → exactly 4 encode calls (NOT 12×2).

    The shared unique-volume cache is the central ADR-0014 invariant: latents are
    cached per ``(subject, contrast)`` and reused across every pair referencing
    them, so the encode cost is the unique-volume count, never the pair count.
    """
    _write_brats_subject(tmp_path)  # 4 contrasts
    _write_nifti(str(tmp_path / "BraTS-GLI-0000-000-seg.nii.gz"))  # mask — dropped

    manifest = build_brats_pair_manifest(str(tmp_path))
    assert len(manifest) == 12  # 4 contrasts × 3 non-self permutations
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)
    assert len(vol.unique_sample_ids()) == 4  # the seg mask did not become a volume

    counter = [0]
    ds = PairedLatentDataset(vol, encode_fn=_counting_encode_fn(counter))
    ds.warm_cache(torch.device("cpu"), show_progress=False)
    assert counter[0] == 4  # each unique volume encoded EXACTLY once — not 24


# -- scale_factor pooled over the union of unique latents ---------------------


def test_estimate_paired_scale_factor_pools_union_and_sets_vae(tmp_path) -> None:
    """1/std(z) over the UNIQUE latents (= src∪tgt); both endpoints equal raw*scale."""
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 34, "tgt_label": 35}]
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)

    torch.manual_seed(0)
    ds = PairedLatentDataset(vol, encode_fn=_mock_encode_fn())
    ds.warm_cache(torch.device("cpu"), show_progress=False)

    vae = AutoencoderKL(scaling_factor=1.0)
    assert vae.scaling_factor.item() == 1.0
    scale = estimate_paired_scale_factor(ds, vae, sample_size=64)

    ids = vol.unique_sample_ids()
    expected_std = torch.stack([ds.raw_latent(sid) for sid in ids]).std()
    assert scale.item() == pytest.approx(1.0 / expected_std.item())
    assert vae.scaling_factor.item() == pytest.approx(scale.item())
    assert vae.scaling_factor.item() != 1.0  # estimated, not the placeholder

    # Both src and tgt __getitem__ latents equal raw * scale (one pooled scale).
    pair = vol[0]
    item = ds[0]
    assert torch.allclose(item["src_latent"], ds.raw_latent(pair["src_id"]) * scale)
    assert torch.allclose(item["tgt_latent"], ds.raw_latent(pair["tgt_id"]) * scale)


def test_paired_disk_cache_reuses_across_runs(tmp_path) -> None:
    """A warmed disk cache serves a second run with NO encoder (keyed by sample_id)."""
    cache_dir = str(tmp_path / "cache")
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 34, "tgt_label": 35}]
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)

    torch.manual_seed(0)
    first = PairedLatentDataset(vol, encode_fn=_mock_encode_fn(), cache_dir=cache_dir)
    first.warm_cache(torch.device("cpu"), show_progress=False)
    raw_first = first.raw_latent(vol.unique_sample_ids()[0]).clone()
    assert any(p.suffix == ".pt" for p in tmp_path.glob("cache/*.pt"))

    # Second dataset: no encoder — must serve entirely from the disk cache.
    torch.manual_seed(0)
    second = PairedLatentDataset(vol, encode_fn=None, cache_dir=cache_dir)
    second.warm_cache(torch.device("cpu"), show_progress=False)
    assert torch.allclose(second.raw_latent(vol.unique_sample_ids()[0]), raw_first)


# -- Seam #4: BraTS pair builder ---------------------------------------------


def test_build_brats_pair_manifest_one_subject_12_pairs(tmp_path) -> None:
    _write_brats_subject(tmp_path)
    _write_nifti(str(tmp_path / "BraTS-GLI-0000-000-seg.nii.gz"))  # must be dropped

    manifest = build_brats_pair_manifest(str(tmp_path))
    assert len(manifest) == 12  # 4 × 3 ordered, self excluded

    # No self-pairs; all 12 (src, tgt) path-pairs are distinct.
    for pair in manifest:
        assert pair["src"] != pair["tgt"]
    pair_keys = {(p["src"], p["tgt"]) for p in manifest}
    assert len(pair_keys) == 12

    # The seg mask is absent from every pair's paths.
    for pair in manifest:
        assert "seg" not in pair["src"] and "seg" not in pair["tgt"]


def test_build_brats_pair_manifest_skips_missing_contrast(tmp_path) -> None:
    """A subject missing any contrast contributes zero pairs (no half-built pairs)."""
    _write_brats_subject(tmp_path, contrasts=("t1n", "t1c", "t2w"))  # only 3 of 4
    manifest = build_brats_pair_manifest(str(tmp_path))
    assert manifest == []


def test_build_brats_pair_manifest_two_subjects(tmp_path) -> None:
    """Per-subject grouping: 2 complete subjects → 24 pairs (subjects don't cross)."""
    for subject in ("BraTS-GLI-0000-000", "BraTS-GLI-0000-001"):
        _write_brats_subject(tmp_path, subject=subject)
    manifest = build_brats_pair_manifest(str(tmp_path))
    assert len(manifest) == 24
    # No pair crosses subjects: src and tgt share the same subject prefix.
    for pair in manifest:
        src_sub = pair["src"].rsplit("-t1n", 1)[0].rsplit("-t1c", 1)[0].rsplit("-t2w", 1)[0].rsplit("-t2f", 1)[0]
        tgt_sub = pair["tgt"].rsplit("-t1n", 1)[0].rsplit("-t1c", 1)[0].rsplit("-t2w", 1)[0].rsplit("-t2f", 1)[0]
        assert src_sub == tgt_sub


def test_build_brats_pair_manifest_labels_map_correctly(tmp_path) -> None:
    _write_brats_subject(tmp_path)
    manifest = build_brats_pair_manifest(str(tmp_path), labels=DEFAULT_BRATS_LABELS)

    # The (t1n → t1c) pair carries labels 34 → 35.
    for pair in manifest:
        if pair["src"].endswith("-t1n.nii.gz") and pair["tgt"].endswith("-t1c.nii.gz"):
            assert pair["src_label"] == DEFAULT_BRATS_LABELS["t1n"] == 34
            assert pair["tgt_label"] == DEFAULT_BRATS_LABELS["t1c"] == 35
            break
    else:
        pytest.fail("t1n→t1c pair not found in manifest")


# -- Integration: batch contract feeds PairedLatentFlowModule -----------------


def test_paired_latent_dataset_batch_feeds_paired_module_one_step(tmp_path) -> None:
    """The paired batch contract matches :class:`PairedLatentFlowModule` — one fit
    step yields a finite loss whose ``.backward()`` reaches the UNet (no contract
    mismatch). Uses ``in_channels = 2·C_latent = 8`` (ADR-0014 concat conditioning)."""
    src = tmp_path / "a-t1n.nii.gz"
    tgt = tmp_path / "a-t1c.nii.gz"
    _write_nifti(str(src))
    _write_nifti(str(tgt))
    manifest = [{"src": str(src), "tgt": str(tgt), "src_label": 0, "tgt_label": 1}]
    vol = PairedNiftiVolumeDataset(manifest, target_dim=(8, 8, 8), divisor=4)

    torch.manual_seed(0)
    ds = PairedLatentDataset(vol, encode_fn=_mock_encode_fn())
    ds.warm_cache(torch.device("cpu"), show_progress=False)

    # The dataset emits the contract; add the batch dim the Module expects.
    item = ds[0]
    batch = {
        "src_latent": item["src_latent"].unsqueeze(0),  # [1, 4, 4, 4, 4]
        "tgt_latent": item["tgt_latent"].unsqueeze(0),
        "src_label": item["src_label"].unsqueeze(0),  # [1]
        "tgt_label": item["tgt_label"].unsqueeze(0),
        "spacing": item["spacing"],
    }

    unet = UNet3DConditionModel(  # in_channels = 2·C_latent = 8 (ADR-0014)
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    scheduler = FlowMatchHeunDiscreteScheduler()
    module = PairedLatentFlowModule(unet, scheduler)

    out = module(batch, "fit")
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in unet.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
