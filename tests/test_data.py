"""BraTS latent-prep data stack tests (issue #16).

Covers the transforms (verbatim from hope), the BraTS label provider (contrast
detection + seg drop), the NIfTI volume dataset (RAS reorient + the
``{image, spacing, label, sample_id, meta}`` contract), and the latent dataset's
scale-on-read: warm an **unscaled** cache via an injected encode_fn, estimate
``1/std(z)``, set ``vae.scaling_factor``, and return **scaled** latents matching
the Module's ``{latent, spacing, label}`` batch contract (ADR-0003 addendum).
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch
from stable_pretraining.data import DataModule as SptDataModule

from manifold import AutoencoderKL
from manifold.data import (
    BratsLabelProvider,
    FixedLabelProvider,
    LatentDataset,
    NiftiVolumeDataset,
    build_datamodule,
    detect_brats_contrast,
    estimate_scale_factor,
    floor_to_divisible,
    label_provider_from_config,
    load_brats_labels,
    normalize_to_01,
    pad_to_divisible,
    resize_to,
)
from manifold.data.base import MedicalDataset


def _write_nifti(path: str, shape=(10, 10, 6)) -> None:
    arr = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    nib.Nifti1Image(arr, affine=np.diag([1.0, 1.0, 2.0, 1.0])).to_filename(path)


# -- transforms -------------------------------------------------------------


def test_normalize_mr_percentile_no_clip_vs_ct_fixed_window() -> None:
    mr = np.array([0.0, 1.0, 2.0, 1000.0], dtype=np.float32)  # p99.5 ~ 25.5
    out = normalize_to_01(mr, modality=34)  # MR branch (>= 8)
    assert out.dtype == np.float32
    # the high tail (1000) is NOT clipped — it lands well above 1 (MR dynamic range)
    assert out[-1] > 1.0
    # CT branch: [-1000,1000] hard-clipped to [0,1]
    ct = np.array([-2000.0, 0.0, 2000.0], dtype=np.float32)
    out_ct = normalize_to_01(ct, modality=1)
    assert out_ct[0] == 0.0 and out_ct[-1] == 1.0  # clipped at both ends


def test_pad_and_floor_to_divisible() -> None:
    arr = np.zeros((10, 10, 6), dtype=np.float32)
    padded, orig = pad_to_divisible(arr, 4)
    assert padded.shape == (12, 12, 8) and orig == (10, 10, 6)
    floored, new = floor_to_divisible(arr, 4)
    assert floored.shape == (8, 8, 4) and new == (8, 8, 4)


def test_resize_to_is_trilinear_align_corners_false() -> None:
    vol = np.random.default_rng(1).standard_normal((6, 6, 6)).astype(np.float32)
    out = resize_to(vol, (8, 8, 8))
    assert out.shape == (8, 8, 8)


# -- labels -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("BraTS-GLI-0000-000-t1n.nii.gz", "t1n"),
        ("case_t1c.nii.gz", "t1c"),
        ("x-t2w.nii", "t2w"),
        ("x-t2f.nii.gz", "t2f"),
        ("case-seg.nii.gz", None),  # mask dropped
        ("unknown.nii.gz", None),
    ],
)
def test_detect_brats_contrast(name, expected) -> None:
    assert detect_brats_contrast(name) == expected


def test_load_brats_labels_default_mapping() -> None:
    labels = load_brats_labels(None)
    assert labels == {"t1n": 34, "t1c": 35, "t2w": 36, "t2f": 37}


def test_label_provider_from_config_brats_vs_fixed(tmp_path) -> None:
    from omegaconf import OmegaConf

    brats = OmegaConf.create({"dataset_type": "brats", "modality_mapping_path": None})
    fixed = OmegaConf.create({"dataset_type": "fixed"})
    assert isinstance(label_provider_from_config(brats, include_modality=True), BratsLabelProvider)
    assert isinstance(label_provider_from_config(fixed, include_modality=True), FixedLabelProvider)
    # include_modality=False always falls back to fixed.
    assert isinstance(label_provider_from_config(brats, include_modality=False), FixedLabelProvider)


# -- volume dataset ---------------------------------------------------------


def test_volume_dataset_emits_contract_and_skips_masks(tmp_path) -> None:
    _write_nifti(str(tmp_path / "case-t1n.nii.gz"))
    _write_nifti(str(tmp_path / "case-t2w.nii.gz"))
    _write_nifti(str(tmp_path / "case-seg.nii.gz"))  # must be skipped

    ds = NiftiVolumeDataset(
        str(tmp_path),
        BratsLabelProvider({"t1n": 34, "t1c": 35, "t2w": 36, "t2f": 37}),
        target_dim=(8, 8, 8),
        divisor=4,
    )
    assert len(ds) == 2  # seg dropped
    assert ds.label_counts() == {34: 1, 36: 1}

    item = ds[0]
    assert {"image", "spacing", "label", "sample_id", "meta"} <= set(item)
    assert item["image"].shape == (1, 8, 8, 8)
    assert item["spacing"].shape == (3,)
    assert item["spacing"][2].item() == pytest.approx(2.0)  # read off the affine
    assert item["label"].dtype == torch.long
    assert item["label"].item() in {34, 36}
    # sample_id = basename + path-hash (collision-free cache key).
    assert item["sample_id"].startswith(("case-t1n.nii.gz__", "case-t2w.nii.gz__"))
    assert "__" in item["sample_id"]


def test_volume_dataset_manifest_source(tmp_path) -> None:
    p1 = tmp_path / "a-t1n.nii.gz"
    p2 = tmp_path / "b-t1c.nii.gz"
    _write_nifti(str(p1))
    _write_nifti(str(p2))
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps({"training": [{"image": str(p1)}, {"image": str(p2)}]}))

    ds = NiftiVolumeDataset(
        str(manifest),
        BratsLabelProvider({"t1n": 34, "t1c": 35, "t2w": 36, "t2f": 37}),
        target_dim=(8, 8, 8),
        divisor=4,
    )
    assert len(ds) == 2
    assert ds.label_counts() == {34: 1, 35: 1}


# -- latent dataset (scale-on-read) -----------------------------------------


class _FakeVolumeDataset(MedicalDataset):
    """Image-emitting source with deterministic latents (no NIfTI, no VAE)."""

    def __init__(self, n: int = 4) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {
            "image": torch.randn(1, 8, 8, 8),
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "label": torch.tensor(34, dtype=torch.long),
            "sample_id": f"fake__{i:02d}",
        }

    def sample_ids(self) -> list[str]:
        return [f"fake__{i:02d}" for i in range(self.n)]


def _mock_encode_fn(scale_std: float = 3.0):
    """A deterministic unscaled-latent encoder (std != 1 so scale != 1)."""

    def fn(images: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(42)
        return torch.randn(images.shape[0], 4, 4, 4, 4) * scale_std

    return fn


def test_warm_estimate_returns_scaled_latents_and_sets_vae_buffer(tmp_path) -> None:
    """warm_cache + estimate_scale_factor → scaled-latent batch + vae.scaling_factor."""
    vae = AutoencoderKL(scaling_factor=1.0)  # placeholder
    assert vae.scaling_factor.item() == 1.0

    ds = LatentDataset(_FakeVolumeDataset(5), encode_fn=_mock_encode_fn())
    ds.warm_cache(torch.device("cpu"), show_progress=False)
    raw0 = ds.raw_latent(0).clone()
    expected_std = torch.stack([ds.raw_latent(i) for i in range(5)]).std().item()

    scale = estimate_scale_factor(ds, vae, sample_size=5)
    assert scale.item() == pytest.approx(1.0 / expected_std)
    # vae.scaling_factor changed from the placeholder AND matches the estimate.
    assert vae.scaling_factor.item() != 1.0
    assert vae.scaling_factor.item() == pytest.approx(scale.item())

    item = ds[0]
    assert {"latent", "spacing", "label"} <= set(item)  # the Module's batch contract
    assert torch.allclose(item["latent"], raw0 * vae.scaling_factor.item())  # scale-on-read
    assert item["label"].dtype == torch.long
    assert torch.isfinite(item["latent"]).all()


def test_latent_dataset_requires_warm_before_getitem() -> None:
    ds = LatentDataset(_FakeVolumeDataset(2), encode_fn=_mock_encode_fn())
    with pytest.raises(RuntimeError, match="warm_cache"):
        _ = ds[0]


def test_two_tier_disk_cache_reuses_across_runs(tmp_path) -> None:
    """A warmed disk cache is reused on a second run (deterministic sample_id key)."""
    cache_dir = str(tmp_path / "cache")

    first = LatentDataset(_FakeVolumeDataset(3), encode_fn=_mock_encode_fn(), cache_dir=cache_dir)
    first.warm_cache(torch.device("cpu"), show_progress=False)
    raw_first = first.raw_latent(0).clone()
    assert any(p.suffix == ".pt" for p in tmp_path.glob("cache/*.pt"))  # disk cache written

    # Second dataset: no encoder — must serve entirely from the disk cache.
    second = LatentDataset(_FakeVolumeDataset(3), encode_fn=None, cache_dir=cache_dir)
    second.warm_cache(torch.device("cpu"), show_progress=False)
    assert torch.allclose(second.raw_latent(0), raw_first)  # identical, cache-reused


def test_build_datamodule_returns_spt_datamodule() -> None:
    ds = LatentDataset(_FakeVolumeDataset(4), encode_fn=_mock_encode_fn())
    ds.warm_cache(torch.device("cpu"), show_progress=False)
    estimate_scale_factor(ds, AutoencoderKL(scaling_factor=1.0), sample_size=4)
    dm = build_datamodule(ds, batch_size=2)
    assert isinstance(dm, SptDataModule)
    batch = next(iter(dm.train_dataloader()))
    assert {"latent", "spacing", "label"} <= set(batch)
    assert batch["latent"].shape[0] == 2  # batch_size


def test_warm_latent_pipeline_orchestration(tmp_path) -> None:
    """warm_latent_pipeline warms, frees the encoder, and sets vae.scaling_factor.

    The encode_fn + VAE are injected (the smoke-test seam — it does not call
    load_vae), exercising the orchestration glue end-to-end.
    """
    from manifold.data import warm_latent_pipeline

    vae = AutoencoderKL(scaling_factor=1.0)
    bundle = warm_latent_pipeline(
        _FakeVolumeDataset(4),
        encode_fn=_mock_encode_fn(),
        autoencoder=vae,
        cache_dir=str(tmp_path / "cache"),
        cache_tag="test",
        device=torch.device("cpu"),
        logger=None,
        scale_factor_sample_size=4,
    )
    assert bundle.latent_ds.encode_fn is None  # encoder freed after warm
    assert bundle.scale_factor.item() == pytest.approx(vae.scaling_factor.item())
    assert bundle.scale_factor.item() != 1.0  # estimated, not the placeholder
    # The bundle's dataset serves scaled latents (scale-on-read).
    item = bundle.latent_ds[0]
    assert torch.allclose(
        item["latent"], bundle.latent_ds.raw_latent(0) * vae.scaling_factor.item()
    )
