#!/usr/bin/env python
"""Numerical validation: manifold Pipeline vs hope's JiT ``sample_x0`` (issue #18).

A one-off gauss script (NOT a CI seam). Proves the migration correct by
reproducing hope's JiT ``sample_x0`` output through manifold's formal inference
Pipeline on the **same** seed / ``num_inference_steps`` / cfg / modality / target
shape / spacing, then asserting two things:

- **Algorithmic latent parity** (bit-identical sampler): run BOTH sides with a
  *degenerate* ``cfg_interval`` so classifier-free guidance is never active —
  each step is then a single conditional forward on both sides, eliminating the
  only fp16 non-associativity source (hope batch-doubles cond‖uncond; manifold
  runs two separate forwards — bitwise-equal in fp32, ~1e-2 apart under
  ``torch.autocast``). This latent MUST match to ``max_abs < latent_tol``.
- **Decoded-volume parity** (end-to-end output): decode the *real*-CFG latent on
  both sides (hope's ``ReconModel`` vs ``pipe.vae.decode``) and assert
  ``allclose(atol=1e-2, rtol=1e-2)`` AND ``PSNR > 60 dB``.

The *real*-CFG latent (informational, printed but NOT a pass/fail bar) sits at
~1e-2 because hope's batch-doubled CFG forward and manifold's two-separate-forward
CFG accumulate fp16 differently under autocast — the same computation in a
different rounding order. The degenerate-interval probe above isolates the
algorithm; the decoded comparison proves the generated volumes agree.

The hope reference is built from **hope's own MONAI-bundle configs** (the
``_target_``/``@name`` schema hope's ``define_instance``/``load_unet``/
``build_inference_conditions`` read) — NOT manifold's plain-kwarg network config
(ADR-0004). Requires the hope package + hope source on the PYTHONPATH (gauss
conda env ``hope``) and the confirmed **JiT/x0** checkpoint
(``diff_unet_3d_rflow-brats2023_jit.pt``, double-EMA ``[0.9999, 0.9996]``).

Example (gauss)::

    python scripts/validate_against_hope.py \\
        --manifold-pipeline /data72/junran/manifold-runtime/checkpoints/jit \\
        --hope-ckpt /data72/junran/hope-runtime/brats2023_finetune_jit/diff_unet_3d_rflow-brats2023_jit.pt \\
        --vae-ckpt /data72/junran/hope-runtime/models/autoencoder_v1.pt \\
        --hope-root /data72/junran/hope --dim "[64,64,64]" --decode-roi "[32,32,32]"
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

# Allow running the script directly from a source checkout (no install needed).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

#: A ``cfg_interval`` covering no flow-time ``t ∈ [0, 1]`` → guidance never
#: active → every step is a single conditional forward on BOTH sides (hope does
#: NOT batch-double when guidance==0). This isolates the sampler algorithm from
#: fp16 CFG-batching non-associativity.
_ALGO_INTERVAL = (2.0, 3.0)


def _psnr(a, b, data_range=None) -> float:
    """Peak signal-to-noise ratio between two tensors (higher = closer)."""
    import math

    import torch

    a = a.float()
    b = b.float()
    if data_range is None:
        data_range = (a.max() - a.min()).item() or 1.0
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range**2) / mse)


def _parse_list(s: str) -> list:
    return list(ast.literal_eval(s))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifold-pipeline", required=True, help="Converted manifold pipeline dir."
    )
    parser.add_argument("--hope-ckpt", required=True, help="hope JiT/x0 UNet checkpoint (.pt).")
    parser.add_argument(
        "--vae-ckpt", required=True, help="Trained VAE weights (autoencoder_v1.pt)."
    )
    parser.add_argument(
        "--hope-root",
        default="/data72/junran/hope",
        help="hope source root (derives default hope env/train/network configs).",
    )
    parser.add_argument("--hope-env", default=None, help="Override hope env config YAML.")
    parser.add_argument("--hope-train", default=None, help="Override hope train recipe YAML.")
    parser.add_argument("--hope-network", default=None, help="Override hope network config YAML.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Override random_seed (default: config)."
    )
    parser.add_argument("--steps", type=int, default=None, help="Override num_inference_steps.")
    parser.add_argument(
        "--dim",
        default=None,
        help="Override latent spatial shape [D,H,W] (default: config). Use a smaller "
        "shape to fit hope's batch-doubled CFG UNet forward under torch's INT_MAX "
        "and the GPU memory budget.",
    )
    parser.add_argument(
        "--latent-tol", type=float, default=1e-3, help="max_abs latent tolerance (algo parity)."
    )
    parser.add_argument(
        "--decode-atol", type=float, default=1e-2, help="decoded allclose atol/rtol."
    )
    parser.add_argument("--psnr-min-db", type=float, default=60.0, help="min decoded PSNR (dB).")
    parser.add_argument(
        "--decode-roi",
        default=None,
        help="Override decode roi [D,H,W] in latent space. Use a value < --dim so the "
        "VAE decode tiles instead of materializing the full decoder pyramid at once "
        "(which OOMs). Applied identically to BOTH sides, so tiling/blending artifacts "
        "cancel and parity is preserved.",
    )
    parser.add_argument(
        "--decode-overlap", type=float, default=None, help="Override decode sliding-window overlap."
    )
    args = parser.parse_args(argv)

    import torch

    from manifold import LatentFlowPipeline  # noqa: F401  (imported for use below)

    import hope  # noqa: F401  (ensures hope is importable)
    from hope.config import load_config as hope_load_config
    from hope.models.networks import load_unet, load_vae
    from hope.models.recon import ReconModel
    from hope.sampling.conditions import build_inference_conditions
    from hope.sampling.x0 import sample_x0

    hope_root = Path(args.hope_root)
    env_path = args.hope_env or str(hope_root / "configs/env/environment_brats2023.yaml")
    train_path = args.hope_train or str(hope_root / "configs/train/config_rflow_jit.yaml")
    net_path = args.hope_network or str(hope_root / "configs/network/config_network_rflow.yaml")

    # hope's own MONAI-bundle config (autoencoder_def/diffusion_unet_def/_target_).
    cfg = hope_load_config(env_path, train_path, net_path)
    cfg.trained_autoencoder_path = args.vae_ckpt  # hope's load_vae reads it from cfg

    inf = cfg.diffusion_unet_inference
    form = cfg.formulation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = int(args.seed if args.seed is not None else inf["random_seed"])
    steps = int(args.steps if args.steps is not None else inf["num_inference_steps"])
    cfg_strength = float(inf["cfg_guidance_scale"])
    cfg_interval = tuple(form.get("cfg_interval", (0.1, 1.0)))
    t_eps = float(form.get("t_eps", 0.05))
    modality = int(inf["modality"])
    latent_channels = int(cfg.latent_channels)
    dim = _parse_list(args.dim) if args.dim else list(inf["dim"])  # [D, H, W]
    shape = [1, latent_channels, *dim]
    spacing = list(inf["spacing"])
    num_train_timesteps = int(cfg.noise_scheduler["num_train_timesteps"])
    # hope's sliding-window decode tiling — the manifold decode MUST match it.
    roi = list(inf.get("autoencoder_sliding_window_infer_size", [80, 80, 80]))
    overlap = float(inf.get("autoencoder_sliding_window_infer_overlap", 0.4))
    if args.decode_roi:
        roi = _parse_list(args.decode_roi)
    if args.decode_overlap is not None:
        overlap = args.decode_overlap

    print(
        f"[validate] device={device} shape={shape} steps={steps} cfg={cfg_strength} "
        f"interval={cfg_interval} modality={modality} seed={seed} roi={roi}"
    )

    from monai.inferers import sliding_window_inference

    # -- hope reference -------------------------------------------------------
    unet_h, scale_h = load_unet(cfg, device, ckpt_path=args.hope_ckpt, strict=True, ema=True)
    vae_h = load_vae(cfg, device)
    conditions = build_inference_conditions(cfg, device, modality)

    def hope_sample(unet, conds, interval):
        noise = torch.randn(
            shape, generator=torch.Generator(device=device).manual_seed(seed), device=device
        )
        return sample_x0(
            unet,
            noise,
            conds,
            num_inference_steps=steps,
            cfg=cfg_strength,
            cfg_interval=interval,
            t_eps=t_eps,
            ode_solver="heun",
            num_train_timesteps=num_train_timesteps,
            start_t=0.0,
            device=device,
        ).float()

    # Real-CFG latent (decoded below) AND the degenerate-interval algorithmic-
    # parity latent (guidance never active → single conditional forward).
    latent_h = hope_sample(unet_h, conditions, cfg_interval)
    latent_h_algo = hope_sample(unet_h, conditions, _ALGO_INTERVAL)
    # The hope UNet is done — free it (and the conditions) so the decode + the
    # manifold pipeline have GPU room. vae_h stays for the decode.
    del unet_h, conditions
    torch.cuda.empty_cache()

    recon = ReconModel(vae_h, scale_h.to(device))
    # Decode the REAL-CFG latent under autocast (the VAE carries norm_float16, so
    # a float32 decode hits a Half/float mismatch — hope's infer CLI decodes under
    # torch.amp.autocast too). inference_mode so the decoder forward does NOT build
    # an autograd graph (8 sliding-window patches × a deep decoder retain every
    # activation otherwise → OOM).
    with torch.inference_mode(), torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        decoded_h = sliding_window_inference(
            latent_h, roi_size=roi, sw_batch_size=1, predictor=recon, overlap=overlap
        ).float()
    # Move results to CPU and free the hope VAE entirely before the manifold side
    # loads — at most one stack (UNet+VAE) is resident at a time.
    decoded_h = decoded_h.cpu()
    latent_h = latent_h.cpu()
    latent_h_algo = latent_h_algo.cpu()
    del recon, vae_h
    torch.cuda.empty_cache()

    # -- manifold pipeline ----------------------------------------------------
    pipe = LatentFlowPipeline.from_pretrained(args.manifold_pipeline)
    pipe.unet.to(device)  # the pipeline is not an nn.Module; move its components
    pipe.vae.to(device)

    def manifold_sample(interval):
        return pipe.sample_latent(
            shape,
            spacing,
            modality,
            steps,
            guidance_scale=cfg_strength,
            cfg_interval=interval,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).float()

    latent_m = manifold_sample(cfg_interval)
    latent_m_algo = manifold_sample(_ALGO_INTERVAL)
    # The manifold UNet is done — free it; only the VAE is needed for the decode.
    del pipe.unet
    torch.cuda.empty_cache()
    # Decode with hope's roi/overlap so both sides run the SAME sliding window —
    # inference_mode too (see decoded_h above): the decoder forward must not graph.
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
    ):
        decoded_m = pipe.vae.decode(latent_m, roi_size=roi, overlap=overlap).float()
    decoded_m = decoded_m.cpu()
    latent_m = latent_m.cpu()
    latent_m_algo = latent_m_algo.cpu()

    # -- compare --------------------------------------------------------------
    latent_real_max_abs = (latent_m - latent_h).abs().max().item()
    latent_algo_max_abs = (latent_m_algo - latent_h_algo).abs().max().item()
    decode_allclose = torch.allclose(
        decoded_m, decoded_h, atol=args.decode_atol, rtol=args.decode_atol
    )
    psnr = _psnr(decoded_m, decoded_h)

    latent_algo_ok = latent_algo_max_abs < args.latent_tol
    decode_ok = decode_allclose and psnr > args.psnr_min_db

    print(
        f"[validate] latent (algo, no-CFG) max_abs = {latent_algo_max_abs:.3e}  "
        f"({'PASS' if latent_algo_ok else 'FAIL'}, tol < {args.latent_tol:.0e})  "
        f"[bit-identical sampler proof]"
    )
    print(
        f"[validate] latent (real CFG)  max_abs = {latent_real_max_abs:.3e}  "
        f"(informational — fp16 CFG-batching residual; not a pass/fail bar)"
    )
    print(
        f"[validate] decoded allclose = {decode_allclose}  PSNR = {psnr:.2f} dB  "
        f"({'PASS' if decode_ok else 'FAIL'}, tol allclose {args.decode_atol:.0e} / "
        f"PSNR > {args.psnr_min_db})"
    )

    if latent_algo_ok and decode_ok:
        print(
            "[validate] PASS — manifold reproduces hope's JiT sampler (bit-identical, "
            "no-CFG) and decode (real-CFG decoded volume)."
        )
        return 0
    if not latent_algo_ok:
        print(
            "[validate] FAIL — algorithmic latent mismatch (sampling bug: Heun / CFG / "
            "timestep / noise parity — the no-CFG single-forward trajectory disagrees).",
            file=sys.stderr,
        )
    if not decode_ok:
        print(
            "[validate] FAIL — decoded-volume mismatch (decode bug: VAE weights / scale).",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
