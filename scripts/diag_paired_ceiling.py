#!/usr/bin/env python3
"""Diagnostic: locate the PSNR ceiling of Paired JiT at the ep25 peak ckpt.

Question: can optimizer/scheduler tuning lift val/psnr from ~24.6 toward 29?
This script isolates WHERE the ceiling lives by comparing, on the same 32 train
val-batches (val=train in this codebase), the decoded-PSNR of:

  A. SLOW-EMA 8-step rollout   (the logged val/psnr arm; expect ~24.7)
  B. RAW (optimizer) 8-step     (is the raw model better than the slow EMA?)
  C. FAST-EMA 8-step            (decay 0.9996 — less lag than slow)
  D. single-shot at t=nodes[0]  (ONE UNet call, no Heun integration — isolates
                                  per-step predictive quality from integrator error)
  E. copy-src baseline          (pred = x_src; the "do nothing" floor)

Interpretation:
  - If RAW >> SLOW-EMA       → EMA lag is capping the metric (tunable: shorter EMA).
  - If single-shot >> rollout → the Heun integrator is lossy (a sampler fix, NOT an
                                optimizer/scheduler fix, would help).
  - If single-shot ≈ rollout ≈ 24.6 → the per-step model itself is the ceiling;
                                optimizer/scheduler tuning cannot reach 29.
  - copy-src brackets the floor and shows how hard src→tgt is.

Reuses eval_paired_step_sweep.py's data/decode path (trained VAE + scale_factor).
The tgt volume is decoded ONCE and reused across all arms (decode is the cost).

Usage (gauss, GPU 3 — does not touch the GPU-0 training):
    CUDA_VISIBLE_DEVICES=3 python3 scripts/diag_paired_ceiling.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Any

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
VAL_SUBSET = 4   # batches × 8 = 32 samples (matches trainer val_subset_size)
BATCH_SIZE = 8
SCALE_SAMPLE = 64
N_STEPS = 8

_PAIRED_DOTLIST = [
    "diffusion_unet.in_channels=8",
    "data_base_dir=/data72/dataset/ASNR-MICCAI-BraTS2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData",
    "trained_autoencoder_path=/data72/junran/hope-runtime/models/autoencoder_v1.pt",
    "model_dir=/data72/junran/manifold-runtime/brats2023_gli_paired_jit",
    "latent_cache_dir=/data72/junran/manifold-runtime/brats2023_gli_paired_jit/paired_latent_cache",
]


def _build_cfg():
    from omegaconf import OmegaConf
    from manifold.config import load_config, merge_overrides
    cfg = load_config(ENV_CFG, TRAIN_CFG, NET_CFG)
    cfg = merge_overrides(cfg, {"num_gpus": 1}, _PAIRED_DOTLIST)
    OmegaConf.resolve(cfg)
    return cfg


def build_val_data():
    from manifold.config import autoencoder_divisor
    from manifold.data.latent_pipeline import build_encode_pipeline
    from manifold.data.paired_brats import build_brats_pair_manifest
    from manifold.data.paired_latent_dataset import (
        PairedLatentDataset, estimate_paired_scale_factor,
    )
    from manifold.data.paired_volume_dataset import PairedNiftiVolumeDataset

    cfg = _build_cfg()
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    divisor = autoencoder_divisor(cfg)
    manifest = build_brats_pair_manifest(str(cfg.data_base_dir))
    vol_ds = PairedNiftiVolumeDataset(manifest, target_dim=target_dim, divisor=divisor)
    device = torch.device("cuda")
    autoencoder, encode_fn = build_encode_pipeline(cfg, device=device, logger=None)
    latent_ds = PairedLatentDataset(
        vol_ds, encode_fn=encode_fn, cache_dir=CACHE_DIR, cache_tag="paired_train")
    latent_ds.warm_cache(device, logger=None, show_progress=False)
    latent_ds.free_encoder()
    estimate_paired_scale_factor(latent_ds, autoencoder, sample_size=SCALE_SAMPLE, logger=None)
    print(f"  scale_factor={float(autoencoder.scaling_factor):.6f}")
    for m in autoencoder.modules():
        if hasattr(m, "norm_float16"):
            m.norm_float16 = False
    autoencoder.cuda().eval()
    n_val = min(VAL_SUBSET * BATCH_SIZE, len(latent_ds))
    batches = [latent_ds[i] for i in range(n_val)]
    return batches, autoencoder


@torch.no_grad()
def heun_rollout(unet, scheduler, src, spacing, src_label, tgt_label, n_steps):
    from manifold.modules.paired_sampler import sample_paired_latent_flow
    return sample_paired_latent_flow(
        unet, scheduler, src, spacing, src_label, tgt_label, num_inference_steps=n_steps)


@torch.no_grad()
def first_node(unet, scheduler, src, spacing, src_label, tgt_label, n_steps):
    """Single UNet call at the rollout's first node (t=nodes[0]) on concat([src,src])."""
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    nodes = scheduler.set_timesteps(n_steps, device=device)
    t0 = float(nodes[0])
    sd = src.to(device=device, dtype=dtype)
    sp = torch.as_tensor(spacing, device=device)
    sl = torch.full((src.shape[0],), int(src_label), dtype=torch.long, device=device)
    tl = torch.full((src.shape[0],), int(tgt_label), dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        return unet(sample=torch.cat([sd, sd], dim=1), timestep=t0, spacing=sp,
                    class_labels_src=sl, class_labels_tgt=tl).float()


def _psnr_ssim(pred_vol, tgt_vol):
    from torchmetrics.functional import structural_similarity_index_measure
    ps, ss, n = 0.0, 0.0, 0
    for i in range(pred_vol.shape[0]):
        p = pred_vol[i:i + 1].float(); t = tgt_vol[i:i + 1].float()
        dr = float(t.max() - t.min())
        if dr <= 0.0:
            continue
        mse = float((p - t).pow(2).mean())
        if mse == 0.0:
            continue
        ps += 10.0 * math.log10((dr * dr) / mse)
        ss += float(structural_similarity_index_measure(p, t, data_range=dr))
        n += 1
    return (ps / max(n, 1)), (ss / max(n, 1)), n


def main():
    print(f"=== ckpt: {os.path.basename(CKPT)} ===")
    d = torch.load(CKPT, map_location="cpu", weights_only=True)
    raw_sd = d["state_dict"]
    ema = d["callbacks"]["DoubleEMACallback"]
    shadows, decays = ema["shadows"], ema["decays"]
    print(f"epoch={d.get('epoch','?')}  decays={decays}")

    # Bare-UNet raw weights: strip the Lightning module prefix.
    prefix = "unet."
    raw_unet_sd = {k[len(prefix):]: v for k, v in raw_sd.items() if k.startswith(prefix)}
    print(f"raw UNet tensors: {len(raw_unet_sd)}  (shadow[0]: {len(shadows[0])})")

    cfg = _build_cfg()
    from manifold.config.builder import build_unet, build_scheduler
    unet = build_unet(cfg).cuda()
    scheduler = build_scheduler(cfg)

    print(f"\n=== warming cache + {VAL_SUBSET*BATCH_SIZE} val batches ===")
    batches, vae = build_val_data()

    # Decode the tgt volume ONCE and reuse across all arms.
    print("=== decoding tgt volumes (cached) ===")
    tgt_vols = []
    for batch in batches:
        tgt = batch["tgt_latent"].unsqueeze(0).cuda()
        with torch.inference_mode():
            tgt_vols.append(vae.decode(tgt.float()).float().cpu())

    def run_arm(name, sd, predict):
        unet.load_state_dict(sd, strict=True); unet.eval()
        ps, ss, n, t0 = 0.0, 0.0, 0, time.time()
        for batch, tv in zip(batches, tgt_vols):
            src = batch["src_latent"].unsqueeze(0).cuda()
            spacing = batch["spacing"].unsqueeze(0).cuda()
            sl = int(batch["src_label"].item()); tl = int(batch["tgt_label"].item())
            pred_lat = predict(unet, scheduler, src, spacing, sl, tl, N_STEPS)
            with torch.inference_mode():
                pred_vol = vae.decode(pred_lat.float().cuda()).float().cpu()
            p, s, nv = _psnr_ssim(pred_vol, tv)
            ps += p * nv; ss += s * nv; n += nv
        print(f"  {name:>28s}: psnr={ps/max(n,1):7.3f}  ssim={ss/max(n,1):.4f}  "
              f"({time.time()-t0:.0f}s, {n} samples)")

    print(f"\n=== arms (8-step Heun, {len(batches)} samples) ===")
    run_arm("A. SLOW-EMA rollout (0.9999)", shadows[0], heun_rollout)
    run_arm("B. RAW rollout (optimizer)", raw_unet_sd, heun_rollout)
    if len(shadows) > 1:
        run_arm("C. FAST-EMA rollout (0.9996)", shadows[1], heun_rollout)
    run_arm("D. SLOW-EMA single-shot t0", shadows[0], first_node)
    run_arm("E. copy-src (pred=x_src)", shadows[0], lambda u, s, src, sp, sl, tl, n: src)


if __name__ == "__main__":
    main()
