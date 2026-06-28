#!/usr/bin/env python
"""Convert a hope flat checkpoint into manifold's native per-component format.

The one-shot migration path (ADR-0003). Maps ``unet_state_dict → unet`` (or the
slowest EMA shadow, baked as the inference weights) and ``scale_factor →
vae.scaling_factor``, and — when ``--vae-checkpoint`` is given — loads the trained
VAE weights (``autoencoder_v1.pt``) into ``vae.autoencoder`` before save, so the
converted checkpoint's VAE actually decodes. Writes a directory
``LatentFlowPipeline.from_pretrained`` can load.

The target UNet/VAE construction kwargs are read from the OmegaConf **network
config** (issue #15): the same YAML a training run composes. Point it at the
network config of the checkpoint you are converting.

Example (gauss)::

    python scripts/convert_hope_checkpoint.py \\
        --hope diff_unet_3d_rflow-brats2023.pt \\
        --vae-checkpoint models/autoencoder_v1.pt \\
        --network-config configs/network/config_network.yaml \\
        --output /data72/junran/manifold-runtime/checkpoints/jit
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
from manifold.pipelines import convert_hope_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hope", required=True, help="Path to the hope flat UNet checkpoint.")
    parser.add_argument("--output", required=True, help="Output manifold pipeline directory.")
    parser.add_argument(
        "--network-config",
        required=True,
        help="OmegaConf network YAML (VAE + UNet construction kwargs).",
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=None,
        help="Path to the trained VAE weights (autoencoder_v1.pt). Loads into "
        "vae.autoencoder before save so the converted VAE decodes.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Ignore EMA shadows; bake the raw unet_state_dict as the inference weights.",
    )
    args = parser.parse_args(argv)

    # The converter only needs the network (construction) blocks, so the network
    # config is loaded for the env slot too (its path keys are unused here).
    cfg = load_config(args.network_config, None, args.network_config)
    unet = build_unet(cfg)
    vae = build_vae(cfg)
    scheduler = build_scheduler(cfg)
    convert_hope_checkpoint(
        args.hope,
        args.output,
        unet,
        vae,
        scheduler,
        prefer_ema=not args.no_ema,
        vae_checkpoint=args.vae_checkpoint,
    )
    msg = f"Converted {args.hope} -> {args.output}"
    if args.vae_checkpoint:
        msg += f" (VAE weights from {args.vae_checkpoint})"
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
