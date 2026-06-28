#!/usr/bin/env python
"""Convert a hope flat checkpoint into manifold's native per-component format.

The one-shot migration path (ADR-0003). Maps ``unet_state_dict → unet`` (or the
slowest EMA shadow, baked as the inference weights) and ``scale_factor →
vae.scaling_factor``, writing a directory ``LatentFlowPipeline.from_pretrained``
can load.

The target UNet/VAE configs are given as a tiny JSON (the same keys the manifold
component constructors take); point it at the config of the checkpoint you are
converting.

Example::

    python scripts/convert_hope_checkpoint.py \\
        --hope hope.pt --output manifold_pipeline \\
        --unet-config unet.json --vae-config vae.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running the script directly from a source checkout (no install needed).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from manifold import (  # noqa: E402
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    UNet3DConditionModel,
)
from manifold.pipelines import convert_hope_checkpoint  # noqa: E402


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hope", required=True, help="Path to the hope flat checkpoint.")
    parser.add_argument("--output", required=True, help="Output manifold pipeline directory.")
    parser.add_argument("--unet-config", required=True, help="JSON of UNet3DConditionModel config.")
    parser.add_argument("--vae-config", required=True, help="JSON of AutoencoderKL config.")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Ignore EMA shadows; bake the raw unet_state_dict as the inference weights.",
    )
    args = parser.parse_args(argv)

    unet = UNet3DConditionModel(**_load_json(args.unet_config))
    vae = AutoencoderKL(**_load_json(args.vae_config))
    scheduler = FlowMatchHeunDiscreteScheduler()
    convert_hope_checkpoint(
        args.hope,
        args.output,
        unet,
        vae,
        scheduler,
        prefer_ema=not args.no_ema,
    )
    print(f"Converted {args.hope} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
