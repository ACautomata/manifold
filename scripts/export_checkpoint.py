#!/usr/bin/env python
"""Export a manifold training ``.ckpt`` to the native per-component inference dir.

The ADR-0006 bridge (sibling to ``convert_hope_checkpoint``): load a Lightning
``.ckpt``, bake the **slowest EMA shadow** as the inference UNet, and write a
directory :meth:`manifold.LatentFlowPipeline.from_pretrained` loads.

Example (gauss)::

    python scripts/export_checkpoint.py \\
        --ckpt /data72/junran/manifold-runtime/lightning/last.ckpt \\
        --network-config configs/network/config_network.yaml \\
        --vae-checkpoint models/autoencoder_v1.pt \\
        --output /data72/junran/manifold-runtime/checkpoints/jit_exported
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
        "--no-ema", action="store_true", help="bake the raw UNet weights instead of the EMA shadow."
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.network_config, None, args.network_config)
    unet = build_unet(cfg)
    vae = build_vae(cfg)
    scheduler = build_scheduler(cfg)
    if args.vae_checkpoint:
        import torch

        from manifold.data.latent_pipeline import load_vae as _load_vae

        vae = _load_vae(cfg, torch.device("cpu"))

    source = export_to_native(
        args.ckpt,
        args.output,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        prefer_ema=not args.no_ema,
    )
    print(f"Exported {args.ckpt} -> {args.output} ({source}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
