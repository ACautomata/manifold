"""Export a manifold training ``.ckpt`` to the native per-component inference dir.

The ADR-0006 bridge: load a Lightning ``.ckpt``, bake the inference UNet (the
**raw UNet weights**; EMA training was removed), and write a directory
:meth:`manifold.LatentFlowPipeline.from_pretrained` (or
:meth:`manifold.PairedLatentFlowPipeline.from_pretrained` with ``--pipeline paired``)
loads.

Example (gauss, JiT noise->data)::

    manifold-export \\
        --ckpt /data72/junran/manifold-runtime/lightning/last.ckpt \\
        --network-config configs/network/config_network.yaml \\
        --vae-checkpoint models/autoencoder_v1.pt \\
        --output /data72/junran/manifold-runtime/checkpoints/jit_exported

Example (paired src->tgt - the reward's frozen generator)::

    manifold-export \\
        --ckpt <paired_run>/last.ckpt \\
        --network-config configs/network/config_network.yaml \\
        --vae-checkpoint models/autoencoder_v1.pt \\
        --pipeline paired \\
        --scaling-factor $(python -c "import torch;print(torch.load('<paired_run>/paired_scaling_factor.pt', weights_only=True))") \\
        --output <paired_native>

    ``--pipeline paired`` builds the 2\u00b7C_latent UNet (the paired
    condition-aware concat); ``--scaling-factor`` (read from the paired run's
    paired_scaling_factor.pt) bakes the generator's training scale into the
    exported VAE.
"""

from __future__ import annotations

import argparse

from manifold.config import (
    build_scheduler,
    build_unet,
    build_vae,
    load_config,
)
from manifold.training.export import export_to_native


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
        "--pipeline",
        choices=("jit", "paired", "controlnet"),
        default="jit",
        help="which native pipeline to write: 'jit' (LatentFlowPipeline, the "
        "noise->data JiT - the default), 'paired' (PairedLatentFlowPipeline, the "
        "src->tgt translation - the reward's frozen generator, ADR-0021), or "
        "'controlnet' (ControlNetLatentFlowPipeline - the supervised ControlNet "
        "stage-1 export, base UNet + ControlNet + VAE, ADR-0027/issue #144).",
    )
    parser.add_argument(
        "--base-native-dir",
        default=None,
        help="REQUIRED for --pipeline controlnet: the JiT native export dir the "
        "supervised ControlNet was trained against. A supervised ControlNet ckpt "
        "registers ONLY the trainable ControlNet (the frozen base is held "
        "unregistered, off the checkpoint), so the frozen base UNet + VAE scale are "
        "loaded from this export and passed through verbatim; the ckpt bakes only "
        "the controlnet.* weights.",
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

    # Paired src->tgt checkpoints train a 2\u00b7C_latent condition-aware UNet: the input
    # is the [x_src, x_tgt_noisy] concat (in_channels=2\u00b7C), the output predicts the
    # tgt velocity (out_channels=C, unchanged). The stock network config builds
    # in_channels=${latent_channels}=4; override in_channels before construction or
    # the backbone load hits a conv-weight size mismatch (codex #98 P1).
    if args.pipeline == "paired":
        from omegaconf import OmegaConf

        latent_c = int(cfg.get("latent_channels", 4))
        cfg = OmegaConf.merge(cfg, {"diffusion_unet": {"in_channels": 2 * latent_c}})

    if args.pipeline == "controlnet":
        # Supervised ControlNet export (ADR-0027 stage 1 -> issue #144): the ckpt
        # registers ONLY the trainable ControlNet, so the frozen base UNet + VAE scale
        # come from the JiT native export it was trained against (--base-native-dir),
        # passed through verbatim; export_to_native bakes only the controlnet.* weights.
        if not args.base_native_dir:
            raise ValueError(
                "--base-native-dir is required for --pipeline controlnet: the frozen "
                "base UNet + VAE scale are loaded from that JiT native export (a "
                "supervised ControlNet ckpt carries only controlnet.* keys, no base)."
            )
        from manifold import ControlNetLatentFlowPipeline, LatentFlowPipeline
        from manifold.config.builder import build_controlnet

        base_pipe = LatentFlowPipeline.from_pretrained(args.base_native_dir)
        unet = base_pipe.unet  # the frozen base (passed through verbatim, NOT re-baked)
        vae = base_pipe.vae    # carries the training scaling_factor
        scheduler = base_pipe.scheduler
        controlnet = build_controlnet(cfg)
        controlnet.load_base_encoder_weights(unet)
        source = export_to_native(
            args.ckpt,
            args.output,
            unet=unet,
            controlnet=controlnet,
            vae=vae,
            scheduler=scheduler,
            pipeline_cls=ControlNetLatentFlowPipeline,
        )
        print(f"Exported {args.ckpt} -> {args.output} ({source}; pipeline=controlnet).")
        return 0

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

    source = export_to_native(
        args.ckpt,
        args.output,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        pipeline_cls=pipeline_cls,
    )
    print(f"Exported {args.ckpt} -> {args.output} ({source}; pipeline={args.pipeline}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
