#!/usr/bin/env python
"""[DEBUG-a1f3-real] gauss-only: real RadImageNet backbone feature parity.

The local scripts/_debug_fid_diff.py proves D1 (per-plane-batch min-max) with a
shared FAKE backbone. This is the real-backbone tier: load the ACTUAL RadImageNet
ResNet50 both ways (manifold offline loader vs hope torch.hub local fallback —
verified bit-identical on gauss), feed the SAME 2D slice-batch through both with
their respective preprocessing, and assert the features match. Same input scale
invariance is checked too.

Run in the gauss `hope` env (needs the cached _notop ckpt + hub repo under
$TORCH_HOME). Exits 1 on divergence.
"""
# tag: DEBUG-a1f3-real
import os
import sys

MANIFOLD_SRC = os.environ.get("MANIFOLD_SRC", "/data72/junran/manifold/src")
HOPE_SRC = os.environ.get("HOPE_SRC", "/data72/junran/hope/src")
for p in (MANIFOLD_SRC, HOPE_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

torch.manual_seed(0)
from hope.metrics import fid as hfid
from manifold.metrics import fid as mfid

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[real-parity] device={dev}")

net_h = hfid.make_feature_network("radimagenet_resnet50", device=dev).eval()
net_m = mfid.make_feature_network("resnet50").to(dev).eval()

FAIL = []


def _check(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


# Same 2D slice-batch (K,1,H,W) at an unclamped BraTS-MR-like scale.
s = (torch.rand(6, 1, 24, 24) * 1.3 - 0.05).to(dev)
with torch.no_grad():
    # hope: _to_bgr + radimagenet_intensity_normalisation, then raw backbone
    h_in = hfid._to_bgr(s.clone())
    h_prep = hfid.radimagenet_intensity_normalisation(h_in)
    feat_h = hfid.spatial_average(net_h(h_prep), keepdim=False)
    # manifold: the wrapper does preprocessing + backbone + flatten
    feat_m = mfid._RadImageNetFeatures(net_m)(s)
max_abs = (feat_h.float() - feat_m.float()).abs().max().item()
_check("real-backbone feature parity (D1+D2)", max_abs < 1e-4, f"max_abs={max_abs:.3e}")

# Scale invariance through the real backbone (the D1 property).
with torch.no_grad():
    f1 = mfid._RadImageNetFeatures(net_m)(s)
    f3 = mfid._RadImageNetFeatures(net_m)(s * 3.0)
drift = (f1.float() - f3.float()).abs().max().item()
_check("manifold scale-invariant (real backbone)", drift < 1e-4, f"drift={drift:.3e}")

if FAIL:
    print(f"\n[real-parity] {len(FAIL)} FAILURE(S): {FAIL}")
    raise SystemExit(1)
print("\n[real-parity] PASS — real RadImageNet features match between manifold and hope.")
