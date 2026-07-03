#!/usr/bin/env python
"""Generate offline GRPO reward preference pairs (issue #39/#42).

Build ``(winner, loser)`` latent pairs by noising each clean latent to a per-sample
flow-time ``t`` (winner ``t_w ~ U[0.5,1)``, loser ``t_l ~ U[0,0.5)`` — the
scheduler's ``t→1 = clean`` convention) and denoising back to clean with the
**frozen JiT denoiser** loaded from a native export (ADR-0006 — the GRPO starting
policy), then cache them to a ``RewardPairDataset`` with a held-out-subject
validation split. Pairs are built once (the denoiser is frozen, so they are static).

Clean latents are read from a directory of ``.pt`` files — either a plain
``[C, D, H, W]`` latent tensor or a latent-cache item dict (``{"latent", ...,
"sample_id"}``); the subject id is the dict's ``sample_id`` if present, else the
file stem. Files sharing a stem prefix up to the first ``__`` are one subject.

Example (gauss)::

    python scripts/generate_reward_pairs.py \\
        --native-dir /data72/junran/manifold-runtime/checkpoints/jit_exported \\
        --latents-dir /data72/junran/manifold-runtime/latent_cache \\
        --output-dir /data72/junran/manifold-runtime/reward_pairs \\
        --num-steps 4 --modality 1 --spacing 1.0 1.0 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script directly from a source checkout (no install needed).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from manifold.data.reward_pairs import (  # noqa: E402
    generate_generated_end_probe,
    generate_reward_pairs,
    load_frozen_denoiser,
    save_reward_pairs,
)


def _load_clean_latents(latents_dir: Path) -> tuple[torch.Tensor, list[str]]:
    """Load clean latents + subject ids from a directory of ``.pt`` files."""
    files = sorted(p for p in latents_dir.glob("*.pt") if not p.name.endswith(".tmp.r0.p0"))
    if not files:
        raise FileNotFoundError(f"No .pt latents under {latents_dir}.")
    latents, subject_ids = [], []
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(item, dict) and "latent" in item:
            latent = item["latent"]
            sid = str(item.get("sample_id", path.stem))
        else:
            latent = item
            sid = path.stem.split("__", 1)[0]  # group by cache stem prefix
        latents.append(latent.float())
        subject_ids.append(sid)
    return torch.stack(latents), subject_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--native-dir", required=True, help="native JiT export dir (frozen denoiser).")
    parser.add_argument("--latents-dir", required=True, help="directory of clean .pt latents.")
    parser.add_argument("--output-dir", required=True, help="output RewardPairDataset directory.")
    parser.add_argument("--num-steps", type=int, default=4, help="shared Heun step budget.")
    parser.add_argument("--modality", type=int, default=1, help="class label for conditioning.")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="voxel spacing.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="held-out subject fraction.")
    parser.add_argument("--batch-size", type=int, default=4, help="generation batch size.")
    parser.add_argument(
        "--n-probe",
        type=int,
        default=64,
        help="max clean latents for the generated-end probe (both t ∈ [0, 0.5]).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (split + noise + t).")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean, subject_ids = _load_clean_latents(Path(args.latents_dir))
    denoiser, scheduler = load_frozen_denoiser(args.native_dir)
    denoiser.to(device)

    train, val = generate_reward_pairs(
        clean,
        subject_ids,
        denoiser,
        scheduler,
        spacing=args.spacing,
        modality=args.modality,
        num_steps=args.num_steps,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    # Generated-end probe (both samples t ∈ [0, 0.5], ordered by t) over a capped
    # subset — directly tests ranking within the all-generated regime.
    probe = generate_generated_end_probe(
        clean[: args.n_probe],
        denoiser,
        scheduler,
        spacing=args.spacing,
        modality=args.modality,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    save_reward_pairs(args.output_dir, train, val, probe=probe)
    print(
        f"[generate_reward_pairs] wrote {len(train)} train / {len(val)} val / "
        f"{len(probe)} probe pairs to {args.output_dir}."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
