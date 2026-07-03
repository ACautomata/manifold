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


def _subject_id(sample_id: str, regex) -> str:
    """Derive a true subject id from a cache sample_id / file stem.

    BraTS contrast files of one subject differ in the trailing contrast token
    (``BraTS-GLI-0000-000-t1n`` vs ``-t1c``); the full id would split one subject
    across train/val. By default the last ``-<token>`` is stripped (BraTS
    convention). ``--subject-regex`` (one capture group) overrides for other naming.
    """
    import re

    if regex:
        m = re.match(regex, sample_id)
        return m.group(1) if (m and m.groups()) else sample_id
    return sample_id.rsplit("-", 1)[0] if "-" in sample_id else sample_id


def _load_clean_latents(
    latents_dir: Path, subject_regex, scaling_factor: float
) -> tuple[torch.Tensor, list[str], list, list]:
    """Load **scaled** clean latents + subject ids + per-sample spacing/label.

    Latent-cache items are ``{"latent", "spacing", "label", "sample_id"}`` storing
    the **unscaled** latent (scale-on-read happens at training ``__getitem__``);
    *scaling_factor* (the native VAE's) is applied here so the returned latents are
    already in the frozen denoiser's training space — no raw, unscaled tensor ever
    reaches the rollout. Per-sample ``spacing`` / ``label`` are collected so a
    heterogeneous (multi-contrast) cache conditions each rollout correctly; a
    plain-tensor file yields ``None`` (fall back to globals).
    """
    files = sorted(p for p in latents_dir.glob("*.pt") if not p.name.endswith(".tmp.r0.p0"))
    if not files:
        raise FileNotFoundError(f"No .pt latents under {latents_dir}.")
    latents, subject_ids, spacings, labels = [], [], [], []
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(item, dict) and "latent" in item:
            latent = item["latent"]
            sid_raw = str(item.get("sample_id", path.stem))
            spacings.append(item.get("spacing"))
            labels.append(item.get("label"))
        else:
            latent = item
            sid_raw = path.stem
            spacings.append(None)
            labels.append(None)
        latents.append(latent.float() * float(scaling_factor))  # → denoiser's scaled space
        subject_ids.append(_subject_id(sid_raw, subject_regex))
    return torch.stack(latents), subject_ids, spacings, labels


def _maybe_per_sample(items, n: int, fallback):
    """Build a per-sample tensor if every item is present, else the broadcast fallback."""
    if items and all(it is not None for it in items):
        return torch.stack([torch.as_tensor(it) for it in items])
    return fallback


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--native-dir", required=True, help="native JiT export dir (frozen denoiser + VAE scale).")
    parser.add_argument("--latents-dir", required=True, help="directory of clean .pt latents (latent cache).")
    parser.add_argument("--output-dir", required=True, help="output RewardPairDataset directory.")
    parser.add_argument("--num-steps", type=int, default=4, help="shared Heun step budget.")
    parser.add_argument("--modality", type=int, default=1, help="class label (used if cache items lack a label).")
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="voxel spacing (used if cache items lack spacing).")
    parser.add_argument(
        "--subject-regex",
        default=None,
        help="regex (one capture group) extracting a subject id from the sample_id; "
        "default strips the last '-<token>' (BraTS <subject>-<contrast> convention).",
    )
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
    # Load the frozen denoiser first so its VAE scaling_factor scales the cached
    # latents into the denoiser's training space inside _load_clean_latents (the
    # latent cache stores UNSCALED latents — ADR-0003).
    denoiser, scheduler, scaling_factor = load_frozen_denoiser(args.native_dir)
    denoiser.to(device)
    clean, subject_ids, spacings, labels = _load_clean_latents(
        Path(args.latents_dir), args.subject_regex, scaling_factor
    )

    spacing_arg = _maybe_per_sample(spacings, len(clean), args.spacing)
    modality_arg = _maybe_per_sample(labels, len(clean), args.modality)

    train, val = generate_reward_pairs(
        clean,
        subject_ids,
        denoiser,
        scheduler,
        spacing=spacing_arg,
        modality=modality_arg,
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
        spacing=spacing_arg,
        modality=modality_arg,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    save_reward_pairs(args.output_dir, train, val, probe=probe)
    print(
        f"[generate_reward_pairs] wrote {len(train)} train / {len(val)} val / "
        f"{len(probe)} probe pairs to {args.output_dir} (scale ×{scaling_factor})."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
