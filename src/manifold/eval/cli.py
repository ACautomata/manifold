"""``manifold-eval`` console entry (issue #229, ADR-0036 / ADR-0033).

The before/after-GRPO eval on **real checkpoints**: take the *before* native
export dir and the *after* (GRPO) training ``.ckpt``, export the after ckpt to
a native dir via the existing export bridge (:func:`~manifold.training.export.
export_to_native`, ADR-0006), build both pipelines with ``from_pretrained``,
and run the #228 :class:`~manifold.eval.BeforeAfterEval` driver — same-noise
generation → frozen-VAE decode → ``[0, 1]`` → 2.5D slice grids (+ 3D PSNR/SSIM
vs the real target on the paired policy). Writes slice-grid PNGs and a metrics
JSON under ``--output``.

The policy is inferred from the **before** artifact's ``model_index.json``
(``pipeline_class`` self-description, issue #7) — no mode flag (the ADR-0034
artifact-inference convention): a JiT export runs the unconditional pass, a
supervised-ControlNet export the paired pass. The export's components come
from the before pipeline itself (``from_pretrained``), so the after export
reuses the very UNet structure / VAE weights + scale / scheduler the before
side — and the GRPO run — were built on: no separate network config,
``--vae-checkpoint``, or ``--base-native-dir`` is needed, and the frozen
VAE/scheduler are identical across the two sides by construction.

Example (JiT, unconditional before|after grid)::

    manifold-eval \\
        --before-dir /data72/junran/manifold-runtime/checkpoints/jit_exported \\
        --after-ckpt /data72/junran/manifold-runtime/grpo_jit/lightning/last.ckpt \\
        --output runs/eval_jit_before_after

Example (ControlNet, paired before|after|real grid + PSNR/SSIM)::

    manifold-eval \\
        --before-dir <supervised_controlnet_export> \\
        --after-ckpt <grpo_controlnet>/lightning/last.ckpt \\
        --output runs/eval_controlnet_before_after \\
        --data-base-dir <brats_root> --val-data-base-dir <brats_val_root> \\
        --latents-dir <paired_latent_cache> --num-pairs 8
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manifold-eval",
        description=(
            "Before/after-GRPO eval on real checkpoints: export the after ckpt "
            "to native (ADR-0006), build both pipelines, run the same-noise "
            "before/after driver, and write slice-grid PNGs + a metrics JSON."
        ),
    )
    parser.add_argument(
        "--before-dir", required=True, help="the BEFORE native export dir (JiT or supervised ControlNet)."
    )
    parser.add_argument(
        "--after-ckpt", required=True, help="the AFTER (GRPO) Lightning training .ckpt."
    )
    parser.add_argument(
        "--output", required=True, help="run dir for the slice-grid PNGs + metrics.json."
    )
    parser.add_argument("--seed", type=int, default=0, help="the fixed eval seed (the pairing invariant).")
    parser.add_argument(
        "--num-inference-steps", type=int, default=15,
        help="Heun integration steps for both sides (default: the GRPO recipe's num_steps).",
    )
    parser.add_argument(
        "--device", default=None,
        help="torch device for generation/decode (default: cuda if available, else cpu).",
    )
    # -- unconditional (JiT) knobs --------------------------------------------
    parser.add_argument(
        "--num-samples", type=int, default=4,
        help="[jit] batch size of the same-noise before/after generation."
    )
    parser.add_argument(
        "--latent-shape", default="4,64,64,32",
        help="[jit] per-sample latent shape C,D,H,W (default: the GRPO recipe's latent_shape).",
    )
    parser.add_argument("--modality", type=int, default=1, help="[jit] the conditioning class label.")
    parser.add_argument(
        "--spacing", default="1.0,1.0,1.0", help="[jit] raw voxel spacing (3 comma-separated floats)."
    )
    # -- paired (ControlNet) knobs ---------------------------------------------
    parser.add_argument(
        "--data-base-dir", default=None,
        help="[controlnet] BraTS data root (the paired manifest source); required "
        "without --pairs-provider (the CPU smoke seam).",
    )
    parser.add_argument(
        "--val-data-base-dir", default=None,
        help="[controlnet] native held-out BraTS split dir (the training run's val split; "
        "omit to fall back to --val-fraction).",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="[controlnet] subject-level val fraction when no native split dir is given."
    )
    parser.add_argument(
        "--latents-dir", default=None,
        help="[controlnet] the warmed paired latent cache dir; required without "
        "--pairs-provider.",
    )
    parser.add_argument(
        "--cache-tag", default="paired_train",
        help="[controlnet] the paired cache's logical tag (geometry-suffixed as in training).",
    )
    parser.add_argument(
        "--target-dim", default="256,256,128",
        help="[controlnet] the volume target shape D,H,W the cache was warmed at "
        "(default: the supervised recipe's dim).",
    )
    parser.add_argument(
        "--num-pairs", type=int, default=8,
        help="[controlnet] how many validation pairs to evaluate (a chosen subject set).",
    )
    parser.add_argument(
        "--src-label", type=int, default=None,
        help="[controlnet] filter to one control direction's src contrast (default: all).",
    )
    parser.add_argument(
        "--tgt-label", type=int, default=None,
        help="[controlnet] filter to one control direction's tgt contrast (default: all).",
    )
    return parser.parse_args(argv)


def _pipeline_class_of(before_dir: str):
    """The before artifact's pipeline class, resolved from its own ``model_index``.

    The native format self-describes (``pipeline_class``, issue #7), so the
    policy needs no flag: a JiT export names ``LatentFlowPipeline``, a
    supervised-ControlNet export ``ControlNetLatentFlowPipeline``. Unknown
    classes fail fast here with the supported list (the ADR-0034
    artifact-inference convention; ``grpo_cli._detect_controlnet_export`` keys
    on the declared ``controlnet`` component for the same routing — this reader
    keeps the eval entry free of a training-stack import).
    """
    index_path = os.path.join(str(before_dir), "model_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"{str(before_dir)!r} is not a manifold pipeline directory (missing "
            f"model_index.json) — point --before-dir at a native export."
        )
    with open(index_path) as f:
        cls_name = json.load(f)["pipeline_class"]
    # Heavy MONAI-backed classes: imported here (not at module top) per the
    # ADR-0033 cold-start convention shared with export_cli / grpo_cli.
    from ..pipelines.controlnet_latent_flow import ControlNetLatentFlowPipeline
    from ..pipelines.latent_flow import LatentFlowPipeline

    known = {
        "LatentFlowPipeline": LatentFlowPipeline,
        "ControlNetLatentFlowPipeline": ControlNetLatentFlowPipeline,
    }
    if cls_name not in known:
        raise ValueError(
            f"--before-dir holds a {cls_name!r} export; the before/after eval supports "
            f"{sorted(known)}."
        )
    return known[cls_name]


def _to_device(pipeline, device: torch.device) -> None:
    """Move a pipeline's weight components to ``device`` (no ``Pipeline.to``)."""
    pipeline.unet.to(device)
    pipeline.vae.to(device)
    if getattr(pipeline, "controlnet", None) is not None:
        pipeline.controlnet.to(device)


def _load_paired_val_pairs(args: argparse.Namespace, before_pipeline):
    """The real ControlNet validation pairs: {src,tgt} latent + direction + spacing.

    Reuses the training data stack read-only: the BraTS pair manifest → the
    same train/val split resolution (``_train_val_manifests``) → the warmed
    paired latent cache (disk hits only — the eval never encodes; a miss means
    the cache was not warmed for this geometry and fails fast). Scale-on-read
    uses the before export's ``vae.scaling_factor`` verbatim (ADR-0021 /
    ADR-0003: never re-estimated). Pairs are filtered to the chosen control
    direction (``--src-label`` / ``--tgt-label``) and capped at ``--num-pairs``.
    """
    from omegaconf import OmegaConf

    from ..data.paired_brats import build_brats_pair_manifest
    from ..data.paired_latent_dataset import PairedLatentDataset, paired_cache_tag
    from ..data.paired_manifests import _train_val_manifests
    from ..data.paired_volume_dataset import PairedNiftiVolumeDataset

    target_dim = tuple(int(d) for d in str(args.target_dim).split(","))
    # The VAE spatial divisor from the before export's own VAE config — the
    # same 2**(len(num_channels)-1) the network-config builder derives.
    divisor = 2 ** (len(before_pipeline.vae.config["num_channels"]) - 1)

    manifest = build_brats_pair_manifest(str(args.data_base_dir))
    if not manifest:
        raise FileNotFoundError(
            f"No paired BraTS volumes found under --data-base-dir={args.data_base_dir} "
            "(need >=1 subject with all 4 contrasts)."
        )
    split_cfg = OmegaConf.create(
        {"val_data_base_dir": args.val_data_base_dir, "val_fraction": float(args.val_fraction)}
    )
    _, val_manifest = _train_val_manifests(split_cfg, manifest)
    if not val_manifest:
        raise ValueError(
            "The ControlNet eval needs a held-out val split (--val-data-base-dir set, "
            "or --val-fraction > 0); train data is never reused as eval pairs."
        )

    vol_ds = PairedNiftiVolumeDataset(val_manifest, target_dim=target_dim, divisor=divisor)
    ds = PairedLatentDataset(
        vol_ds,
        encode_fn=None,  # disk hits only — the eval never encodes
        cache_dir=str(args.latents_dir),
        cache_tag=paired_cache_tag(str(args.cache_tag), target_dim, divisor),
    )
    ds.warm_cache(torch.device("cpu"), show_progress=False)
    ds.scaling_factor = float(before_pipeline.vae.scaling_factor)

    pairs = []
    for i in range(len(ds)):
        item = ds[i]
        if args.src_label is not None and int(item["src_label"]) != args.src_label:
            continue
        if args.tgt_label is not None and int(item["tgt_label"]) != args.tgt_label:
            continue
        pairs.append(item)
        if len(pairs) >= int(args.num_pairs):
            break
    if not pairs:
        raise ValueError(
            "No validation pairs matched "
            f"(src_label={args.src_label}, tgt_label={args.tgt_label}) — check the "
            "direction filter or raise --num-pairs."
        )
    return pairs


def main(argv: list[str] | None = None, *, pairs_provider=None) -> int:
    """Console entry: export after → build pipelines → run the before/after driver.

    ``pairs_provider`` is the injection seam for the CPU smoke (mirroring
    ``grpo_cli``'s ``data_provider``): a callable ``(args) -> list of
    {src_latent, tgt_latent, spacing, src_label, tgt_label}`` replacing the
    ControlNet path's BraTS-manifest + latent-cache loading with tiny fixtures.
    """
    args = _parse_args(argv)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    from ..training.export import export_to_native
    from .before_after import BeforeAfterEval

    pipe_cls = _pipeline_class_of(args.before_dir)

    # 1. The probe pipeline furnishes the export's components (UNet structure,
    #    the frozen VAE with its scaling_factor, the scheduler's transport
    #    settings) — the exact arms the before side and the GRPO run were built
    #    on, so the after export needs no separate network config / VAE
    #    checkpoint / base-native-dir. Its UNet/ControlNet are baked IN PLACE
    #    by the bridge, so the probe is discarded after this step.
    probe = pipe_cls.from_pretrained(args.before_dir)
    controlnet = getattr(probe, "controlnet", None)
    after_dir = os.path.join(args.output, "after_native")
    export_to_native(
        args.after_ckpt,
        after_dir,
        unet=probe.unet,
        vae=probe.vae,
        scheduler=probe.scheduler,
        controlnet=controlnet,
        pipeline_cls=pipe_cls if controlnet is not None else None,
    )

    # 2. Fresh pipelines for both sides (the probe's arms were overwritten by
    #    the bake; reloading keeps the before side's weights pristine).
    before_pipe = pipe_cls.from_pretrained(args.before_dir)
    after_pipe = pipe_cls.from_pretrained(after_dir)
    _to_device(before_pipe, device)
    _to_device(after_pipe, device)

    os.makedirs(args.output, exist_ok=True)
    driver = BeforeAfterEval()
    if controlnet is not None:
        if pairs_provider is not None:
            pairs = pairs_provider(args)
        else:
            if not args.data_base_dir or not args.latents_dir:
                raise ValueError(
                    "The ControlNet eval needs --data-base-dir <BraTS root> and "
                    "--latents-dir <paired latent cache> (or inject a pairs_provider "
                    "for the smoke)."
                )
            pairs = _load_paired_val_pairs(args, before_pipe)
        src = torch.stack([torch.as_tensor(p["src_latent"]).float() for p in pairs]).to(device)
        real = torch.stack([torch.as_tensor(p["tgt_latent"]).float() for p in pairs]).to(device)
        spacing = torch.as_tensor([p["spacing"] for p in pairs], dtype=torch.float32)
        src_label = torch.as_tensor([p["src_label"] for p in pairs], dtype=torch.long)
        tgt_label = torch.as_tensor([p["tgt_label"] for p in pairs], dtype=torch.long)
        result = driver.run_paired(
            before_pipe,
            after_pipe,
            noise_shape=tuple(src.shape),
            src_latent=src,
            real_tgt_latent=real,
            spacing=spacing,
            src_label=src_label,
            tgt_label=tgt_label,
            num_inference_steps=int(args.num_inference_steps),
            seed=int(args.seed),
            out_dir=args.output,
        )
    else:
        latent_shape = tuple(int(s) for s in str(args.latent_shape).split(","))
        spacing = [float(s) for s in str(args.spacing).split(",")]
        result = driver.run_unconditional(
            before_pipe,
            after_pipe,
            target_shape=(int(args.num_samples), *latent_shape),
            spacing=spacing,
            modality=int(args.modality),
            num_inference_steps=int(args.num_inference_steps),
            seed=int(args.seed),
            out_dir=args.output,
        )

    print(f"[manifold-eval] wrote {len(result.grids)} slice grid(s) + metrics.json -> {args.output}")
    if result.real is not None:
        for side in ("before", "after"):
            print(
                f"[manifold-eval] {side}: psnr={result.metrics[side]['psnr']:.4f} "
                f"ssim={result.metrics[side]['ssim']:.4f} (vs the real target)"
            )
    print(f"[manifold-eval] after export -> {after_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
