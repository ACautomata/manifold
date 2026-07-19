"""Paired latent cache geometry-keying tests (issue #147).

The paired latent disk cache is keyed by ``<stem>__<cache_tag>__<sha1(sample_id)>.pt``
and the ``sample_id`` is path-derived only (it carries no ``target_dim``/``divisor``).
:func:`paired_cache_tag` folds the encode geometry into the tag so two runs differing
only in ``target_dim`` or ``divisor`` produce disjoint cache entries (no shared stale
reuse), and reading a cache written at a different geometry fails fast instead of
returning a wrong-shape latent.
"""

from __future__ import annotations

import torch

from manifold.data.latent_dataset import _cache_path
from manifold.data.paired_latent_dataset import PairedLatentDataset, paired_cache_tag


class _StubVolDS:
    """A ``PairedNiftiVolumeDataset`` stand-in (the latent dataset's source seam).

    Exposes the surface ``PairedLatentDataset`` touches: ``unique_sample_ids`` (the
    warm driver), ``pair_meta`` (the hot-path fetch), ``_load_volume`` (the encode
    source), and the ``target_dim`` / ``divisor`` geometry. Serves a tiny fake image.
    """

    def __init__(self, target_dim=(4, 4, 4), divisor=2, sample_ids=("v0",)):
        self.target_dim = tuple(target_dim)
        self.divisor = int(divisor)
        self._ids = list(sample_ids)

    def __len__(self):
        return len(self._ids)

    def unique_sample_ids(self):
        return list(self._ids)

    def pair_meta(self, index):
        sid = self._ids[index]
        return {
            "src_id": sid,
            "tgt_id": sid,
            "src_label": torch.tensor(0, dtype=torch.long),
            "tgt_label": torch.tensor(1, dtype=torch.long),
        }

    def _load_volume(self, sample_id):
        d = self.target_dim
        return {
            "image": torch.ones(1, *d),
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "label": 0,
        }


def _identity_encode(images):
    """Echo the image as the latent (C=1), so the cached latent tracks the image shape."""
    return images


def test_paired_cache_tag_encodes_geometry():
    """The tag carries target_dim + divisor; either change yields a different tag."""
    base = paired_cache_tag("paired_train", (256, 256, 128), 4)
    assert base == "paired_train__d256x256x128__div4"
    # target_dim change → different tag.
    assert paired_cache_tag("paired_train", (128, 128, 128), 4) != base
    # divisor change → different tag.
    assert paired_cache_tag("paired_train", (256, 256, 128), 8) != base
    # Reading geometry off the volume dataset (attrs) matches passing it explicitly.
    vol = _StubVolDS(target_dim=(256, 256, 128), divisor=4)
    assert paired_cache_tag("paired_train", vol, vol) == base


def test_geometry_change_produces_disjoint_cache_entries(tmp_path):
    """Encoding at one geometry, then requesting another, yields NO stale hit.

    Two ``PairedLatentDataset`` warm runs differing only in ``divisor`` write to
    disjoint files (the geometry is in the tag), so the second run re-encodes rather
    than reusing the first geometry's latent. The two cache paths differ.
    """
    vol_a = _StubVolDS(target_dim=(4, 4, 4), divisor=2)
    vol_b = _StubVolDS(target_dim=(4, 4, 4), divisor=4)  # same dim, different divisor
    tag_a = paired_cache_tag("paired_train", vol_a, vol_a)
    tag_b = paired_cache_tag("paired_train", vol_b, vol_b)

    # The two geometries map the SAME sample_id to DIFFERENT cache paths.
    assert _cache_path(str(tmp_path), "v0", tag_a) != _cache_path(str(tmp_path), "v0", tag_b)

    ds_a = PairedLatentDataset(vol_a, encode_fn=_identity_encode, cache_dir=str(tmp_path), cache_tag=tag_a)
    ds_a.warm_cache(torch.device("cpu"), show_progress=False)
    # One file written, at geometry A's tag.
    files = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert len(files) == 1
    assert tag_a in files[0] and tag_b not in files[0]


def test_read_at_a_different_geometry_raises_clear_error(tmp_path):
    """A cache written at geometry A is invisible at geometry B → a fresh encode (or a
    clear "cache miss — no encoder" error when no encoder is set), never a stale hit.

    Encodes with an encoder present, then re-opens the same cache dir with a DIFFERENT
    geometry tag and no encoder: the read must NOT find the stale latent (disjoint
    filename) and must fail fast with the clear cache-miss error.
    """
    import pytest

    vol_a = _StubVolDS(target_dim=(4, 4, 4), divisor=2)
    tag_a = paired_cache_tag("paired_train", vol_a, vol_a)
    ds_a = PairedLatentDataset(vol_a, encode_fn=_identity_encode, cache_dir=str(tmp_path), cache_tag=tag_a)
    ds_a.warm_cache(torch.device("cpu"), show_progress=False)

    # Same cache dir, but a different geometry (divisor 4) and NO encoder: the stale
    # geometry-A latent is not visible under the geometry-B tag, so the read is a miss.
    vol_b = _StubVolDS(target_dim=(4, 4, 4), divisor=4)
    tag_b = paired_cache_tag("paired_train", vol_b, vol_b)
    ds_b = PairedLatentDataset(vol_b, encode_fn=None, cache_dir=str(tmp_path), cache_tag=tag_b)
    with pytest.raises(RuntimeError, match="cache miss"):
        ds_b.warm_cache(torch.device("cpu"), show_progress=False)
