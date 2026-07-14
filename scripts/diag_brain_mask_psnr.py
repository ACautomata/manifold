#!/usr/bin/env python3
# NOTE (post-EMA-removal, 2026-07-14): EMA training was removed from manifold
# (DoubleEMACallback deleted). This historical diagnostic reads the EMA shadow
# state from a checkpoint's callbacks['DoubleEMACallback'], which NEW
# checkpoints no longer carry. It only runs against PRE-removal EMA checkpoints;
# on a current checkpoint it raises KeyError on 'DoubleEMACallback'.
"""Diagnostic: is val/psnr background-dominated? (BraTS is skull-stripped.)

The copy-src baseline already reaches ~23.8 dB full-volume (diag_paired_ceiling),
which is suspiciously high for a contrast-translation task. BraTS volumes are
skull-stripped → a large fraction of voxels are identically 0 in both src and tgt,
so a full-volume MSE is dominated by the trivially-matched background.

This script measures, on 8 val pairs (mixed contrast directions):
  - brain fraction (|tgt > 0| / |tgt|) — how much of the volume is background
  - copy-src full-volume PSNR vs brain-masked PSNR (mask = tgt > 0)
  - SLOW-EMA model (8-step) full-volume PSNR vs brain-masked PSNR

PSNR uses data_range = target[max - min] and is SCALE-INVARIANT (multiplying both
pred and tgt by a constant leaves it unchanged), so the VAE scaling_factor need not
be estimated — a placeholder (1.0) gives identical PSNR. (Cross-checked: copy-src
full PSNR must reproduce ~23.8 from diag_paired_ceiling.)

Usage (gauss, GPU 3):  CUDA_VISIBLE_DEVICES=3 python3 scripts/diag_brain_mask_psnr.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

CKPT = "/data72/junran/manifold-runtime/brats2023_gli_paired_jit/paired-025-48802-24.645.ckpt"
CACHE_DIR = "/data72/junran/manifold-runtime/brats2023_gli_paired_jit/paired_latent_cache"
ENV_CFG = "/data72/junran/manifold/configs/env/environment_brats2023.yaml"
TRAIN_CFG = "/data72/junran/manifold/configs/train/config_paired_jit.yaml"
NET_CFG = "/data72/junran/manifold/configs/network/config_network.yaml"
N_VAL = 8
_PAIRED_DOTLIST = [
    "diffusion_unet.in_channels=8",
    "data_base_dir=/data72/dataset/ASNR-MICCAI-BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    "trained_autoencoder_path=/data72/junran/hope-runtime/models/autoencoder_v1.pt",
    "model_dir=/data72/junran/manifold-runtime/brats2023_gli_paired_jit",
    "latent_cache_dir=/data72/junran/manifold-runtime/brats2023_gli_paired_jit/paired_latent_cache",
]


def _build():
    from omegaconf import OmegaConf
    from manifold.config import load_config, merge_overrides
    from manifold.config import autoencoder_divisor
    from manifold.data.latent_pipeline import build_encode_pipeline
    from manifold.data.paired_brats import build_brats_pair_manifest
    from manifold.data.paired_latent_dataset import PairedLatentDataset
    from manifold.data.paired_volume_dataset import PairedNiftiVolumeDataset
    from manifold.config.builder import build_unet, build_scheduler

    cfg = load_config(ENV_CFG, TRAIN_CFG, NET_CFG)
    cfg = merge_overrides(cfg, {"num_gpus": 1}, _PAIRED_DOTLIST)
    OmegaConf.resolve(cfg)
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    manifest = build_brats_pair_manifest(str(cfg.data_base_dir))
    vol_ds = PairedNiftiVolumeDataset(manifest, target_dim=target_dim,
                                      divisor=autoencoder_divisor(cfg))
    device = torch.device("cuda")
    vae, encode_fn = build_encode_pipeline(cfg, device=device, logger=None)
    # Warm loads from the on-disk cache (fast). scale_factor left as placeholder
    # (1.0) — PSNR is scale-invariant, so no estimate_paired_scale_factor needed.
    ds = PairedLatentDataset(vol_ds, encode_fn=encode_fn, cache_dir=CACHE_DIR, cache_tag="paired_train")
    ds.warm_cache(device, logger=None, show_progress=False)
    ds.free_encoder()
    for m in vae.modules():
        if hasattr(m, "norm_float16"):
            m.norm_float16 = False
    vae.cuda().eval()
    batches = [ds[i] for i in range(min(N_VAL, len(ds)))]
    unet = build_unet(cfg).cuda()
    sched = build_scheduler(cfg)
    return batches, vae, unet, sched


def _psnr_full_and_masked(pred_vol, tgt_vol):
    """Return (full_psnr, brain_psnr, brain_frac) for one [1,C,D,H,W] volume.

    brain mask = (tgt > 0). data_range uses the FULL target range in both cases
    (so masked-PSNR is comparable to full-PSNR — only the MSE region changes).
    """
    t = tgt_vol.float()
    p = pred_vol.float()
    dr = float(t.max() - t.min())
    if dr <= 0:
        return None
    brain = t > 0
    brain_frac = float(brain.float().mean())
    full_mse = float((p - t).pow(2).mean())
    brain_mse = float((p - t).pow(2)[brain].mean()) if brain.any() else float("nan")
    full = 10.0 * math.log10((dr * dr) / full_mse) if full_mse > 0 else float("inf")
    masked = 10.0 * math.log10((dr * dr) / brain_mse) if brain_mse > 0 else float("inf")
    return full, masked, brain_frac


@torch.no_grad()
def rollout(unet, sched, src, spacing, sl, tl, n=8):
    from manifold.modules.paired_sampler import sample_paired_latent_flow
    return sample_paired_latent_flow(unet, sched, src, spacing, sl, tl, num_inference_steps=n)


def main():
    print("=== warming cache + building (scale placeholder; PSNR scale-invariant) ===")
    batches, vae, unet, sched = _build()

    d = torch.load(CKPT, map_location="cpu", weights_only=True)
    unet.load_state_dict(d["callbacks"]["DoubleEMACallback"]["shadows"][0], strict=True)
    unet.eval()

    cs_full = cs_mask = md_full = md_mask = 0.0
    bfrac = 0.0
    n = 0
    print(f"\n{'i':>2} {'dir':>9} {'brain%':>7} | {'copy full':>9} {'copy brain':>10} | {'model full':>10} {'model brain':>11}")
    for i, b in enumerate(batches):
        src = b["src_latent"].unsqueeze(0).cuda()
        tgt = b["tgt_latent"].unsqueeze(0).cuda()
        sp = b["spacing"].unsqueeze(0).cuda()
        sl = int(b["src_label"].item()); tl = int(b["tgt_label"].item())
        with torch.inference_mode():
            sv = vae.decode(src.float()).cpu()
            tv = vae.decode(tgt.float()).cpu()
            pv = vae.decode(rollout(unet, sched, src, sp, sl, tl).float().cuda()).cpu()
        cf = _psnr_full_and_masked(sv, tv)
        mf = _psnr_full_and_masked(pv, tv)
        if cf is None or mf is None:
            continue
        names = {34: "t1n", 35: "t1c", 36: "t2w", 37: "t2f"}  # DEFAULT_BRATS_LABELS
        print(f"{i:>2} {names.get(sl,'?')+'->'+names.get(tl,'?'):>9} {cf[2]*100:>6.1f}% | "
              f"{cf[0]:>9.2f} {cf[1]:>10.2f} | {mf[0]:>10.2f} {mf[1]:>11.2f}")
        cs_full += cf[0]; cs_mask += cf[1]; md_full += mf[0]; md_mask += mf[1]
        bfrac += cf[2]; n += 1

    print(f"\n=== MEAN over {n} samples ===")
    print(f"  brain fraction         : {bfrac/n*100:.1f}% of voxels are non-background")
    print(f"  copy-src  full  PSNR   : {cs_full/n:6.2f}")
    print(f"  copy-src  brain PSNR   : {cs_mask/n:6.2f}   <- true task difficulty")
    print(f"  model     full  PSNR   : {md_full/n:6.2f}   (logged metric, SLOW-EMA 8-step)")
    print(f"  model     brain PSNR   : {md_mask/n:6.2f}   <- model's real added value")
    print(f"  model gain over copy (full) : {md_full/n - cs_full/n:+.2f} dB")
    print(f"  model gain over copy (brain): {md_mask/n - cs_mask/n:+.2f} dB")


if __name__ == "__main__":
    main()
