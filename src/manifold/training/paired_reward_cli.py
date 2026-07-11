"""``manifold-train-paired-reward`` console entry + the testable paired-reward core.

The console entry (issue #93) composes the OmegaConf experiment config, builds the
condition-aware :class:`~manifold.models.RewardModel` (``in_channels = 2·C_latent``)
+ the precomputed pair inputs (real latent cache + the frozen paired generator, or
a fake via the ``data_provider`` injection seam for the CPU smoke), and calls
``Trainer.fit``. The paired-reward job is standalone: decoupled from Paired-JiT
training and from paired-GRPO, independently resumable.

The integration core :func:`run_paired_reward_training` (Module + datamodule +
``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a tiny CPU smoke
can drive it with a fake generator + toy pairs (the issue's testing seam) instead of
a real latent cache + the frozen paired generator.

**Offline precompute (ADR-0020, inverts ADR-0010).** The train set is a precomputed
``{winner, loser}`` pair dataset (built once from the fake cache); the Module holds
no generator - ``fit`` consumes pairs and is structurally the JiT reward's
``validate`` path. Validation (real-vs-fake pairs) + the generated-end probe are
precomputed **once** (the generator is frozen ⇒ static across epochs) and reused.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import opt
from ..data.datamodule import build_datamodule
from ..models.reward_model import RewardModel
from ..modules.paired_reward import PairedRewardModule
from .reward_cli import _build_checkpoint
from .trainer import build_trainer, is_multi_gpu

_log = logging.getLogger(__name__)


@dataclass
class PairedRewardInputs:
    """Precomputed-pair inputs for paired reward training (no generator - ADR-0020).

    The Module holds NO generator: ``fit`` and ``validate`` both consume
    precomputed ``{winner, loser}`` pair datasets (built once from the fake cache).
    The ``data_provider`` seam injects a fake generator + toy pairs for the CPU
    smoke; the real path (``_real_inputs``, issue #95) builds this from the paired
    native export (#94's loader) + the paired latent cache.
    """

    train_pair_ds: Any  #: train: emits {winner, loser} (condition-aware [2·C] concat)
    val_pair_ds: Any  #: precomputed: emits {winner, loser} (real-vs-fake, val subjects)
    val_probe: Any = None  #: precomputed RewardPairDataset (generated-end probe, [2·C]).


def run_paired_reward_training(
    *,
    module: PairedRewardModule,
    inputs: PairedRewardInputs,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    save_top_k: int = 1,
    monitor_metric: str = "val/gen_pair_acc",
    mode: str = "max",
    limit_val_batches: int | float = 1.0,
    seed: int = 0,
    ckpt_path: str | None = None,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the paired reward module (the core seam).

    Builds a stock ``ModelCheckpoint`` (monitoring the generated-end probe) and runs
    ``Trainer.fit`` on the precomputed train-pair datamodule + the precomputed val
    pairs. Returns ``(trainer, ckpt)`` so callers can find the written ``.ckpt``.

    Args:
        inputs: the precomputed train/val/probe pair datasets (no generator).
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    # Seed deterministically so direct callers (tests, notebooks) get reproducible
    # runs; ``main`` also seeds, harmlessly, before building the module.
    pl.seed_everything(seed, workers=True)
    multi_gpu = is_multi_gpu(devices)
    # The generated-end probe is mandatory (ADR-0023): ``val/gen_pair_acc`` is the
    # load-bearing monitor, and ``PairedRewardModule`` only logs it when a probe is
    # attached. Under single-GPU the checkpoint monitors the metric, so a missing
    # probe would crash Lightning's ``ModelCheckpoint`` (monitor key never seen).
    # Fail fast with a clear error rather than a MisconfigurationException at fit.
    # DDP drops the monitor (rank-local selection is unreliable), so no probe is
    # needed there.
    if not multi_gpu and monitor_metric == "val/gen_pair_acc" and inputs.val_probe is None:
        raise ValueError(
            "Paired reward training monitors val/gen_pair_acc, but no generated-end "
            "probe was provided (inputs.val_probe is None). The probe is mandatory "
            "(ADR-0023) - build it via build_paired_reward_probe and pass it in, or "
            "set monitor_metric to a logged metric."
        )
    ckpt = _build_checkpoint(
        model_dir, monitor_metric=monitor_metric, mode=mode, save_top_k=save_top_k, multi_gpu=multi_gpu
    )
    # Score the fixed generated-end probe in training-batch-size chunks (bounds
    # epoch-end memory); attach the probe if the inputs carry one.
    module.probe_batch_size = int(batch_size)
    if inputs.val_probe is not None and getattr(module, "val_probe", None) is None:
        module.set_val_probe(inputs.val_probe.winners, inputs.val_probe.losers)
    datamodule = build_datamodule(
        inputs.train_pair_ds,
        batch_size=batch_size,
        val_dataset=inputs.val_pair_ds,
        num_workers=num_workers,
    )
    trainer = build_trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt],
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    return trainer, ckpt


# -- console entry -----------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manifold-train-paired-reward",
        description="Train the paired-JiT reward model (real-vs-fake, offline precompute).",
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c", "--train", default="configs/train/config_paired_reward.yaml",
        help="paired-reward recipe YAML.",
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (reward_model block)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--native-dir",
        default=None,
        help="paired native export dir (frozen generator + VAE scale); required without --data-provider.",
    )
    parser.add_argument(
        "--latents-dir",
        default=None,
        help="paired latent cache dir (the warmed paired cache); required without --data-provider.",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config -> build -> ``run_paired_reward_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> PairedRewardInputs`` returning a fake generator + toy pairs
    so the full ``main`` path runs without a real generator or latent cache. The
    real path loads the frozen paired generator from ``--native-dir`` and the paired
    cache from ``--latents-dir``, precomputes train/val/probe pairs once, and fits
    (ADR-0020 - the Module holds no generator).
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # The paired-reward job needs no VAE / UNet-checkpoint path; only the output
    # ``model_dir`` is required (the generator comes from --native-dir at precompute).
    require_paths(cfg, keys=("model_dir",))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "paired_reward_train", None) is None:
        raise ValueError(
            "Config has no `paired_reward_train` block - use the paired-reward recipe "
            "(-c configs/train/config_paired_reward.yaml), not a JiT/paired train config."
        )

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        inputs = data_provider(cfg, device)
    else:
        # --native-dir / --latents-dir are NOT argparse-required: that would break
        # the data_provider injection seam (the CPU smoke). Validate them here, only
        # on the real path (issue #95).
        if not args.native_dir or not args.latents_dir:
            raise ValueError(
                "Paired reward training needs --native-dir <paired native export> and "
                "--latents-dir <paired latent cache> (or inject a data_provider for the smoke)."
            )
        inputs = _real_inputs(cfg, args.native_dir, args.latents_dir, device)

    reward_cfg = cfg.reward_model
    module = PairedRewardModule(
        RewardModel(
            spatial_dims=int(opt(reward_cfg, "spatial_dims", 3)),
            in_channels=int(opt(reward_cfg, "in_channels", 2 * int(opt(cfg, "latent_channels", 4)))),
            channels=int(opt(reward_cfg, "channels", 64)),
            num_layers_d=int(opt(reward_cfg, "num_layers_d", 3)),
            norm=str(opt(reward_cfg, "norm", "BATCH")),
        ),
        lr=float(cfg.paired_reward_train.lr),
    )

    run_paired_reward_training(
        module=module,
        inputs=inputs,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or cfg.paired_reward_train.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(cfg.paired_reward_train.batch_size),
        save_top_k=int(opt(cfg, "checkpoint.save_top_k", 1)),
        seed=seed,
        ckpt_path=args.resume,
    )
    print(f"[manifold-train-paired-reward] done; checkpoints under {cfg.model_dir}")
    return 0


def _real_inputs(cfg, native_dir: str, latents_dir: str, device: torch.device) -> PairedRewardInputs:
    """Build the real paired-reward inputs from the paired native export + latent cache.

    Loads the frozen paired generator (``--native-dir``, issue #94's
    :func:`~manifold.data.paired_reward_pairs.load_frozen_paired_generator` - slow-EMA
    arm + base scheduler + scaling_factor), resolves the **paired** train/val split
    (``_train_val_manifests`` / ``val_data_base_dir`` / ``val_fraction`` - NOT JiT
    reward's ``partition_subjects``, ADR-0022), warms the paired latent cache over
    each split (reusing the existing ``paired_train`` cache - disjoint sample_ids
    -> free disk hits), sets each dataset's ``scaling_factor`` to the export's
    (ADR-0021 - scale-consistency: reuse verbatim, never re-estimate), and builds
    the train/val/probe condition-aware pairs via
    :func:`~manifold.data.paired_reward_pairs.build_paired_reward_inputs` (the
    offline fake-cache builder, ADR-0020). The generator is used once; the returned
    inputs carry only precomputed pairs - the Module holds no generator.
    """
    import os

    from ..config import autoencoder_divisor
    from ..data.paired_brats import build_brats_pair_manifest
    from ..data.paired_latent_dataset import PairedLatentDataset
    from ..data.paired_reward_pairs import build_paired_reward_inputs, load_frozen_paired_generator
    from ..data.paired_volume_dataset import PairedNiftiVolumeDataset
    from .paired_cli import _train_val_manifests

    # target_dim MUST match the paired_train cache the generator trained on (cache
    # reuse, ADR-0021/0022: sample_ids are derived from the volume + target_dim, so
    # a mismatch breaks disk hits). Read directly (not opt()) - it is load-bearing.
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    divisor = autoencoder_divisor(cfg)

    # The frozen paired generator (slow-EMA arm, base scheduler, scaling_factor).
    generator, base_scheduler, scaling_factor = load_frozen_paired_generator(native_dir)
    generator.to(device).eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    # The PAIRED 2-way subject split (ADR-0022): resolve via _train_val_manifests
    # (val_data_base_dir / val_fraction), NOT JiT reward's partition_subjects.
    brats_dir = str(cfg.data_base_dir)
    manifest = build_brats_pair_manifest(brats_dir)
    if not manifest:
        raise FileNotFoundError(
            f"No paired BraTS volumes found under data_base_dir={brats_dir} "
            f"(need >=1 subject with all 4 contrasts)."
        )
    train_manifest, val_manifest = _train_val_manifests(cfg, manifest)
    if not val_manifest:
        raise ValueError(
            "Paired reward needs a held-out val split (val_data_base_dir set, or "
            "val_fraction > 0); train data is never reused as val (ADR-0022)."
        )

    # Reuse the existing paired_train cache (disjoint sample_ids -> free disk hits,
    # no write conflict, ADR-0022). No VAE is needed: the cache is the fully-warmed
    # `paired_train` cache from a manifold-train-paired run, so warm_cache reads
    # every volume from disk (encode_fn=None is tolerated on cache hits). A partial
    # cache fails fast (PairedLatentDataset raises a clear "cache miss - no encoder"
    # error) rather than silently re-encoding. --latents-dir overrides latent_cache_dir.
    cache_dir = str(
        latents_dir
        or opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "paired_latent_cache"))
    )
    cache_tag = str(opt(cfg, "paired_reward.cache_tag", "paired_train"))

    def _ds(manifest_split):
        vol_ds = PairedNiftiVolumeDataset(manifest_split, target_dim=target_dim, divisor=divisor)
        ds = PairedLatentDataset(vol_ds, encode_fn=None, cache_dir=cache_dir, cache_tag=cache_tag)
        ds.warm_cache(device, logger=_log, show_progress=False)
        # Scale-on-read uses the EXPORT scaling_factor verbatim (ADR-0021): the
        # generator trained on latents scaled by this factor, so the rollout
        # operates in its training space. Never re-estimate.
        ds.scaling_factor = float(scaling_factor)
        return ds

    train_ds, val_ds = _ds(train_manifest), _ds(val_manifest)

    num_steps = int(opt(cfg, "paired_reward.num_steps", 8))
    probe_num_steps = int(opt(cfg, "paired_reward.precompute_num_steps", num_steps))
    n_probe = int(opt(cfg, "paired_reward.n_probe", 64))
    gen_batch_size = int(opt(cfg, "paired_reward.gen_batch_size", 4))
    return build_paired_reward_inputs(
        train_ds=train_ds,
        val_ds=val_ds,
        generator=generator,
        base_scheduler=base_scheduler,
        num_steps=num_steps,
        probe_num_steps=probe_num_steps,
        n_probe=n_probe,
        batch_size=gen_batch_size,
        seed=int(opt(cfg, "random_seed", 0)),
        device=device,
    )
