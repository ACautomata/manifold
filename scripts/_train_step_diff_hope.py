#!/usr/bin/env python
"""[DEBUG-b2c1] one-step JiT training loss/gradient differential: hope vs manifold.

Diagnosing-bugs skill (JiT-flow question). Gauss-only one-off (NOT a CI seam):
needs the hope package + diffusers + the JiT/x0 checkpoint + the converted
manifold pipeline, none of which are available locally. Run in the gauss `hope`
conda env after `pip install -e . --no-deps` for manifold.

Question: for a SINGLE training step, do hope's ``_forward_x0`` and manifold's
``LatentFlowModule.forward`` produce the same loss + same UNet gradients, given
identical (latent, noise, timestep)? The workflow read both line-by-line and
found them bit-identical for matching scale_factor / num_train_timesteps=1000 /
noise_scale=1.0 / no label-aug; this harness confirms it numerically with the
REAL shared backbone.

Method (inline both sides so noise E and timestep t are fixed explicitly — no
RNG-seed coupling): load the hope JiT UNet (raw MAISI) AND the manifold wrapper
UNet (from the converted pipeline) — same underlying weights — then:

  hope:      z = t*(X*sf) + (1-t)*E ; x0 = unet_h(x=z, timesteps=t*1000, **cond)
             weight = (1-t).clamp(t_eps) ; loss = mse(x0/weight, (X*sf)/weight)
  manifold:  z = scheduler.add_noise(X*sf, E, t)  # == t*(X*sf)+(1-t)*E
             x0 = unet_m(sample=z, timestep=t, spacing=SP, **cond)  # wrapper scales t*1000
             weight = (1-t).clamp(t_eps) ; loss = mse(x0/weight, (X*sf)/weight)

``X*sf`` is the pre-scaled latent (manifold's convention — hope applies sf
internally, so feeding X_raw to hope == feeding X_raw*sf to manifold). With the
same weights, x0_h == x0_m → loss_h == loss_m → identical grads.

Prints: per-side loss, |loss_h-loss_m|, and the UNet grad-norm each side + their
ratio. Exits 1 if the loss diff or grad-norm ratio is out of tolerance.

Example (gauss, hope env)::

    python scripts/_train_step_diff_hope.py \\
        --manifold-pipeline /data72/junran/manifold-runtime/checkpoints/jit \\
        --hope-ckpt /data72/junran/hope-runtime/brats2023_finetune_jit/diff_unet_3d_rflow-brats2023_jit.pt \\
        --hope-root /data72/junran/hope
"""
# tag: DEBUG-b2c1
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


def _parse_list(s: str) -> list:
    return list(ast.literal_eval(s))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifold-pipeline", required=True)
    parser.add_argument("--hope-ckpt", required=True)
    parser.add_argument("--hope-root", default="/data72/junran/hope")
    parser.add_argument("--hope-env", default=None)
    parser.add_argument("--hope-train", default=None)
    parser.add_argument("--hope-network", default=None)
    parser.add_argument(
        "--dim", default="[16,16,16]",
        help="latent spatial shape [D,H,W] (small to fit a fwd+bwd on one GPU).",
    )
    parser.add_argument("--p-mean", type=float, default=-0.8)
    parser.add_argument("--p-std", type=float, default=0.8)
    parser.add_argument("--t-eps", type=float, default=0.05)
    parser.add_argument("--loss-tol", type=float, default=1e-4)
    parser.add_argument("--grad-ratio-tol", type=float, default=1e-3)
    args = parser.parse_args(argv)

    import torch
    import torch.nn.functional as F

    from manifold import LatentFlowPipeline
    from manifold.schedulers.scheduling_flow_match_heun import (
        FlowMatchHeunDiscreteScheduler,
    )

    import hope  # noqa: F401
    from hope.config import load_config as hope_load_config
    from hope.models.networks import load_unet
    from hope.sampling.conditions import build_inference_conditions

    hope_root = Path(args.hope_root)
    env_path = args.hope_env or str(hope_root / "configs/env/environment_brats2023.yaml")
    train_path = args.hope_train or str(hope_root / "configs/train/config_rflow_jit.yaml")
    net_path = args.hope_network or str(hope_root / "configs/network/config_network_rflow.yaml")
    cfg = hope_load_config(env_path, train_path, net_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = _parse_list(args.dim)  # [D, H, W]
    latent_channels = int(cfg.latent_channels)
    modality = int(cfg.diffusion_unet_inference["modality"])
    ntt = int(cfg.noise_scheduler.get("num_train_timesteps", 1000))

    # -- shared, fixed stochastic inputs --------------------------------------
    g = torch.Generator(device=device).manual_seed(20260701)
    shape = [1, latent_channels, *dim]
    X_raw = torch.randn(shape, generator=g, device=device)          # unscaled latent
    E = torch.randn(shape, generator=g, device=device)              # fixed noise
    t = torch.tensor([0.3, 0.5, 0.7, 0.95], device=device)[: shape[0]].float()
    if t.shape[0] < shape[0]:
        t = t.repeat(shape[0])[: shape[0]]
    t_b = t.view(-1, *([1] * (X_raw.ndim - 1)))

    # -- hope UNet (raw MAISI) + scale_factor ---------------------------------
    unet_h, scale_h = load_unet(cfg, device, ckpt_path=args.hope_ckpt, strict=True, ema=True)
    scale_h = scale_h.to(device)
    Xsf = X_raw * scale_h  # the effective (scaled) latent both sides train on
    cond_h = build_inference_conditions(cfg, device, modality)  # {'class_labels': ...}
    spacing = list(cfg.diffusion_unet_inference["spacing"])

    # -- manifold wrapper UNet (same weights, via the converted pipeline) ------
    pipe = LatentFlowPipeline.from_pretrained(args.manifold_pipeline)
    pipe.unet.to(device)
    unet_m = pipe.unet
    # manifold conditions: spacing + class_labels (same label as hope)
    cond_m = {"spacing": torch.tensor([spacing], device=device, dtype=torch.float32)}
    if "class_labels" in cond_h:
        cond_m["class_labels"] = cond_h["class_labels"]

    w = (1.0 - t_b).clamp(min=args.t_eps)

    # -- hope loss (inline _forward_x0) ---------------------------------------
    z_h = t_b * Xsf + (1.0 - t_b) * E
    ts_h = (t * ntt).float()
    unet_h.zero_grad(set_to_none=True)
    x0_h = unet_h(x=z_h, timesteps=ts_h, **cond_h)
    loss_h = F.mse_loss(x0_h.float() / w, Xsf.float() / w)
    loss_h.backward(retain_graph=True)
    grad_norm_h = _grad_norm(unet_h)

    # -- manifold loss (inline forward; scheduler.add_noise + wrapper unet) ---
    sched = FlowMatchHeunDiscreteScheduler()
    z_m = sched.add_noise(Xsf, E, t)  # == t_b*Xsf + (1-t_b)*E  (assert below)
    assert torch.allclose(z_h, z_m, atol=1e-6), "add_noise diverges from hope's inline transport"
    unet_m.zero_grad(set_to_none=True)
    x0_m = unet_m(sample=z_m, timestep=t, **cond_m)  # wrapper scales t -> t*ntt
    loss_m = F.mse_loss(x0_m.float() / w, Xsf.float() / w)
    loss_m.backward()
    grad_norm_m = _grad_norm(unet_m)

    loss_diff = (loss_h - loss_m).abs().item()
    ratio = (grad_norm_m / grad_norm_h).item() if grad_norm_h > 0 else float("nan")
    print(f"[train-diff] device={device} shape={shape} t={t.tolist()} ntt={ntt}")
    print(f"[train-diff] loss_hope  = {loss_h.item():.8f}")
    print(f"[train-diff] loss_mani  = {loss_m.item():.8f}")
    print(f"[train-diff] |loss diff| = {loss_diff:.3e}  (tol {args.loss_tol:.0e})")
    print(f"[train-diff] grad_norm_hope = {grad_norm_h:.6e}")
    print(f"[train-diff] grad_norm_mani = {grad_norm_m:.6e}")
    print(f"[train-diff] grad ratio mani/hope = {ratio:.6f}  (tol ±{args.grad_ratio_tol:.0e})")

    ok = loss_diff < args.loss_tol and abs(ratio - 1.0) < args.grad_ratio_tol
    print(f"[train-diff] {'PASS' if ok else 'FAIL'} — JiT training step {'matches' if ok else 'DIVERGES FROM'} hope.")
    return 0 if ok else 1


def _grad_norm(module) -> float:
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().pow(2).sum())
    return sq**0.5


if __name__ == "__main__":
    raise SystemExit(main())
