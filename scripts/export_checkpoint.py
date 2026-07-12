#!/usr/bin/env python
"""Export a manifold training ``.ckpt`` to the native per-component inference dir.

The ADR-0006 bridge: load a Lightning ``.ckpt``, bake the inference UNet (the
**raw UNet weights** by default, matching the ``val/fid_raw`` checkpoint monitor;
``--ema`` bakes the slowest EMA shadow instead), and write a directory
:meth:`manifold.LatentFlowPipeline.from_pretrained` (or
:meth:`manifold.PairedLatentFlowPipeline.from_pretrained` with ``--pipeline paired``)
loads.

Example (gauss, JiT noise->data)::

    python scripts/export_checkpoint.py \\
        --ckpt /data72/junran/manifold-runtime/lightning/last.ckpt \\
        --network-config configs/network/config_network.yaml \\
        --vae-checkpoint models/autoencoder_v1.pt \\
        --output /data72/junran/manifold-runtime/checkpoints/jit_exported

Example (paired src->tgt, slow-EMA arm - the reward's frozen generator)::

    python scripts/export_checkpoint.py \\
        --ckpt <paired_run>/last.ckpt \\
        --network-config configs/network/config_network.yaml \\
        --vae-checkpoint models/autoencoder_v1.pt \\
        --pipeline paired \\
        --scaling-factor $(python -c "import torch;print(torch.load('<paired_run>/paired_scaling_factor.pt', weights_only=True))") \\
        --output <paired_native>

    ``--pipeline paired`` forces the slow-EMA arm (ADR-0021) and builds the 2·C_latent
    UNet; ``--scaling-factor`` (read from the paired run's paired_scaling_factor.pt)
    bakes the generator's training scale into the exported VAE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from a source checkout (no install needed).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from manifold.config import (  # noqa: E402
    build_scheduler,
    build_unet,
    build_vae,
    load_config,
)
from manifold.training.export import export_to_native  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True, help="Lightning training .ckpt (full state).")
    parser.add_argument("--output", required=True, help="Output manifold pipeline directory.")
    parser.add_argument(
        "--network-config", required=True, help="OmegaConf network YAML (VAE + UNet construction)."
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Path to the trained VAE weights (autoencoder_v1.pt); loaded into "
        "vae.autoencoder before export so the exported VAE decodes.",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="bake the slowest EMA shadow instead of the raw UNet weights (the "
        "default bakes raw, matching the val/fid_raw checkpoint monitor; use this "
        "for warm-start / long-horizon runs where the 0.9999 EMA has converged, or "
        "for the paired reward whose val/psnr monitors the slow-EMA arm - ADR-0021).",
    )
    parser.add_argument(
        "--pipeline",
        choices=("jit", "paired"),
        default="jit",
        help="which native pipeline to write: 'jit' (LatentFlowPipeline, the "
        "noise->data JiT - the default) or 'paired' (PairedLatentFlowPipeline, the "
        "src->tgt translation - the reward's frozen generator, ADR-0021).",
    )
    parser.add_argument(
        "--scaling-factor",
        type=float,
        default=None,
        help="the VAE scaling_factor (1/std(z)) to bake into the exported VAE. The "
        "network YAML carries a 1.0 placeholder; paired training writes the real "
        "value to <model_dir>/paired_scaling_factor.pt. REQUIRED for --pipeline "
        "paired (the reward pairs scale src latents by vae.scaling_factor - "
        "ADR-0021; codex #98 P1). Optional (defaults to the YAML value) for jit.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.network_config, None, args.network_config)

    # Paired src->tgt checkpoints train a 2·C_latent condition-aware UNet: the input
    # is the [x_src, x_tgt_noisy] concat (in_channels=2·C), the output predicts the
    # tgt velocity (out_channels=C, unchanged). The stock network config builds
    # in_channels=${latent_channels}=4; override in_channels before construction or
    # the backbone load hits a conv-weight size mismatch (codex #98 P1).
    if args.pipeline == "paired":
        from omegaconf import OmegaConf

        latent_c = int(cfg.get("latent_channels", 4))
        cfg = OmegaConf.merge(cfg, {"diffusion_unet": {"in_channels": 2 * latent_c}})

    unet = build_unet(cfg)
    vae = build_vae(cfg)
    scheduler = build_scheduler(cfg)
    if args.vae_checkpoint:
        import torch

        from omegaconf import OmegaConf

        from manifold.data.latent_pipeline import load_vae as _load_vae

        cfg = OmegaConf.merge(cfg, {"trained_autoencoder_path": args.vae_checkpoint})
        vae = _load_vae(cfg, torch.device("cpu"))

    # The paired reward's frozen generator runs on the generator's TRAINING scale
    # (1/std(z), ADR-0021); the reward pairs scale src latents by vae.scaling_factor.
    # The network YAML carries a 1.0 placeholder, so paired exports must override it
    # (read from <paired_model_dir>/paired_scaling_factor.pt, written after the warm)
    # or the rollout receives unscaled src -> garbage fakes (codex #98 P1).
    if args.scaling_factor is not None:
        vae.scaling_factor.fill_(float(args.scaling_factor))
    elif args.pipeline == "paired":
        raise ValueError(
            "--scaling-factor is required for --pipeline paired: the reward's frozen "
            "generator must be exported at its training scale (1/std(z), ADR-0021). "
            "Read it from <paired_model_dir>/paired_scaling_factor.pt (written by "
            "manifold-train-paired after the VAE warm)."
        )

    pipeline_cls = None
    if args.pipeline == "paired":
        from manifold.pipelines.paired_latent_flow import PairedLatentFlowPipeline

        pipeline_cls = PairedLatentFlowPipeline

    # Paired checkpoint selection + the reward's frozen-generator contract monitor the
    # slow-EMA arm (val/psnr @ slow-EMA, ADR-0021). Force it for paired exports rather
    # than silently baking raw non-EMA weights (codex #98 P2).
    prefer_ema = args.ema or args.pipeline == "paired"
    if args.pipeline == "paired" and not args.ema:
        print("[export_checkpoint] paired export: forcing slow-EMA arm (ADR-0021).")

    source = export_to_native(
        args.ckpt,
        args.output,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        prefer_ema=prefer_ema,
        pipeline_cls=pipeline_cls,
    )
    print(f"Exported {args.ckpt} -> {args.output} ({source}; pipeline={args.pipeline}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
