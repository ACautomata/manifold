#!/usr/bin/env python
"""[DEBUG-a1f3] hope-vs-manifold FID differential regression.

Diagnosing-bugs skill: the Phase-1 loop, now tightened to ASSERT the D1 fix.
NOT a committed unit test (that lives in tests/test_fid.py) — this is the
hope-vs-manifold *differential* harness: same deterministic fake backbone on
both sides, identical inputs, so any difference is purely preprocessing.
Exit code 1 on any assertion failure so it can gate a deploy.

Asserts (locally verifiable, no RadImageNet weights needed):
  STAGE A — preprocessing op-parity: manifold's _RadImageNetFeatures applies the
            SAME ops as hope's _to_bgr + radimagenet_intensity_normalisation
            (replicate -> BGR -> per-batch min-max -> ImageNet-mean-subtract).
  STAGE B — scale-invariance (the D1 fix): scaling the input volume by c leaves
            BOTH hope and manifold features invariant. Pre-fix manifold was
            scale-SENSITIVE (the bug); it must now be scale-INVARIANT.
  STAGE C — FID no longer scale-inflated: manifold-FID is stable across scales.

Gauss-only (real RadImageNet backbone, full 3D per-plane parity) is a separate
tier — documented at the bottom, skipped locally (RadImageNet _notop not cached).
"""
# tag: DEBUG-a1f3
import os
import sys
from pathlib import Path

MANIFOLD_SRC = os.environ.get(
    "MANIFOLD_SRC", str(Path(__file__).resolve().parents[1] / "src")
)
HOPE_SRC = os.environ.get("HOPE_SRC")
if not HOPE_SRC:
    raise SystemExit("Set HOPE_SRC=/path/to/hope/src to run this diff harness")
for p in (MANIFOLD_SRC, HOPE_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch

torch.manual_seed(0)
from hope.metrics import fid as hfid
from manifold.metrics import fid as mfid

FAILURES: list[str] = []


def _check(name: str, ok: bool, detail: str) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


class FakeBackbone(torch.nn.Module):
    """Deterministic 3-channel -> 8-d feature net (shared weights both sides)."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.copy_(torch.randn_like(self.conv.weight) * 0.1)
            self.conv.bias.copy_(torch.randn_like(self.conv.bias) * 0.1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = torch.nn.Identity()  # _RadImageNetFeatures probes/replaces this

    def forward(self, x):  # [B,3,H,W] -> [B,8]
        return self.pool(self.conv(x)).flatten(1)


class _Passthrough(torch.nn.Module):
    """Returns input unchanged so the wrapper's preprocessing is observable."""

    def __init__(self):
        super().__init__()
        self.avgpool = torch.nn.Identity()  # probed by _RadImageNetFeatures.__init__

    def forward(self, x):
        return x


def make_volume(scale: float, layout: str) -> torch.Tensor:
    """Deterministic 3D volume at the given intensity scale.

    layout='hope'    -> (1,1,H,W,D)
    layout='manifold'-> (1,1,D,H,W)  (same voxel grid, permuted to native order)
    """
    g = torch.Generator().manual_seed(123)
    base = torch.rand((1, 1, 24, 24, 16), generator=g)
    base[:, :, 8:16, 8:16, 5:11] += 0.6  # bright structure so slices aren't empty
    vol = base * scale
    return vol if layout == "hope" else vol.permute(0, 1, 4, 2, 3).contiguous()


def shared_backbone_pair():
    """Same backbone weights, wrapped for each pipeline's contract."""
    backbone = FakeBackbone().eval()
    return backbone, mfid._RadImageNetFeatures(backbone)


def stage_a_preprocessing_parity():
    print("\n=== STAGE A — preprocessing op-parity (manifold wrapper vs hope) ===")
    s = torch.rand(5, 1, 8, 8) * 1.3 - 0.05  # unclamped BraTS-MR-like range
    # hope: _to_bgr (replicate+flip) then radimagenet_intensity_normalisation
    h_prep = hfid.radimagenet_intensity_normalisation(hfid._to_bgr(s.clone()))
    # manifold: the wrapper does replicate -> BGR -> min-max -> mean-sub internally
    m_prep = mfid._RadImageNetFeatures(_Passthrough())(s)
    max_abs = (h_prep.flatten(1).float() - m_prep.float()).abs().max().item()
    _check("preprocessing tensor parity", max_abs < 1e-6, f"max_abs={max_abs:.2e}")


def stage_b_scale_invariance():
    print("\n=== STAGE B — scale-invariance (the D1 regression bar) ===")
    backbone_h, net_m = shared_backbone_pair()
    for c in (1.0, 3.0):
        vh1, vh2 = make_volume(1.0, "hope"), make_volume(c, "hope")
        vm1, vm2 = make_volume(1.0, "manifold"), make_volume(c, "manifold")
        with torch.no_grad():
            fh1 = hfid.get_features_2p5d(vh1, backbone_h, center_slices_ratio=1.0)
            fh2 = hfid.get_features_2p5d(vh2, backbone_h, center_slices_ratio=1.0)
            fm1 = mfid.get_features_2p5d(vm1, net_m, center_slices_ratio=1.0)
            fm2 = mfid.get_features_2p5d(vm2, net_m, center_slices_ratio=1.0)
        hope_drift = max((a.float() - b.float()).abs().max().item() for a, b in zip(fh1, fh2))
        mani_drift = max((a.float() - b.float()).abs().max().item() for a, b in zip(fm1, fm2))
        _check(f"hope scale-invariant c={c}", hope_drift < 1e-4, f"drift={hope_drift:.2e}")
        _check(
            f"manifold scale-invariant c={c} (D1 fix)",
            mani_drift < 1e-4,
            f"drift={mani_drift:.2e}",
        )


def stage_c_fid_not_scale_inflated():
    print("\n=== STAGE C — FID no longer scale-inflated ===")
    backbone_h, net_m = shared_backbone_pair()

    def feats_hope(vols):
        out = [[] for _ in range(3)]
        for v in vols:
            f = hfid.get_features_2p5d(v.unsqueeze(0), backbone_h, center_slices_ratio=1.0)
            for i in range(3):
                out[i].append(f[i])
        return [torch.cat(p, 0) for p in out]

    def feats_mani(vols):
        out = [[] for _ in range(3)]
        for v in vols:
            f = mfid.get_features_2p5d(v.unsqueeze(0), net_m, center_slices_ratio=1.0)
            for i in range(3):
                out[i].append(f[i])
        return [torch.cat(p, 0) for p in out]

    def fid_of(feat_fn, real, synth):
        r, s = feat_fn(real), feat_fn(synth)
        return sum(float(mfid.frechet_distance_unbiased(a.float(), b.float())) for a, b in zip(s, r)) / 3

    g = torch.Generator().manual_seed(7)

    def synth_set(scale, n=4):
        h, m = [], []
        for _ in range(n):
            b = torch.rand((1, 1, 24, 24, 16), generator=g)
            b[:, :, 8:16, 8:16, 5:11] += 0.5
            b = b * scale
            h.append(b.squeeze(0))
            m.append(b.permute(0, 1, 4, 2, 3).squeeze(0).contiguous())
        return h, m

    fids = {}
    for scale in (1.0, 3.0):
        rh, rm = synth_set(scale)
        sh, sm = synth_set(scale)
        fids[scale] = (fid_of(feats_hope, rh, sh), fid_of(feats_mani, rm, sm))
    h1, m1 = fids[1.0]
    h3, m3 = fids[3.0]
    print(f"  hope-FID:    scale=1 -> {h1:.3e}   scale=3 -> {h3:.3e}")
    print(f"  mani-FID:    scale=1 -> {m1:.3e}   scale=3 -> {m3:.3e}")
    # Pre-fix manifold-FID grew with scale (the inflation symptom); it must now be
    # scale-stable (same order of magnitude at scale=3 as at scale=1).
    _check("manifold-FID scale-stable", m3 < max(m1 * 5.0, 1e-3), f"m1={m1:.2e} m3={m3:.2e}")


if __name__ == "__main__":
    print(f"[DEBUG-a1f3] torch={torch.__version__}")
    print(f"[DEBUG-a1f3] hope_fid={hfid.__file__}")
    print(f"[DEBUG-a1f3] manifold_fid={mfid.__file__}")
    stage_a_preprocessing_parity()
    stage_b_scale_invariance()
    stage_c_fid_not_scale_inflated()
    print(
        "\n[DEBUG-a1f3] Gauss-only tier (NOT run locally): load the REAL RadImageNet "
        "ResNet50 on both sides (manifold make_feature_network('resnet50') offline "
        "loader vs hope make_feature_network('radimagenet_resnet50') hub local "
        "fallback) on the SAME decoded volume and assert per-plane features "
        "allclose(atol=1e-4). Requires the cached _notop checkpoint + hub repo under "
        "$TORCH_HOME (gauss only)."
    )
    if FAILURES:
        print(f"\n[DEBUG-a1f3] {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\n[DEBUG-a1f3] ALL ASSERTIONS PASS — manifold FID preprocessing matches hope.")
