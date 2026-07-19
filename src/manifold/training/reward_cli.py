"""``manifold-train-reward`` console entry + the testable reward-training core.

The console entry composes the experiment config, builds the :class:`RewardModel`
+ the data inputs (real latent cache + frozen denoiser, or a fake via the
``data_provider`` injection seam for the CPU smoke), and calls ``Trainer.fit``.
The reward job is standalone (issue #39 user story 17): decoupled from JiT
pretraining and from GRPO, independently resumable.

The integration core :func:`run_reward_training` (Module + datamodule +
``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a tiny CPU
smoke can drive it with a fake denoiser + toy clean-latent dataset (the issue's
testing seam) instead of a real latent cache + the frozen JiT denoiser.

**Online rollout-in-the-loop training (ADR-0010, issues #48/#50/#51).** The train
set is now a **clean-latent** dataset (the Module rolls fresh preference pairs
each step); validation (full-range pairs) + the generated-end probe are
precomputed **once at startup** (the denoiser is frozen ⇒ static across epochs)
and reused. The offline pair-generation script is retained for this one-time
precompute (and offline inspection); the offline *train*-pair path is superseded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch

try:
    import lightning.pytorch as pl
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.utilities.rank_zero import rank_zero_info  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import opt
from ..data.datamodule import build_datamodule
from ..models.reward_model import RewardModel
from ..modules.reward import RewardModule
from .trainer import build_trainer, is_multi_gpu


@dataclass
class RewardInputs:
    """Module-construction + data inputs for online reward training.

    ``denoiser`` / ``scheduler`` / ``num_steps`` go to the :class:`RewardModule`
    ctor (the online fit-step rollout); ``clean_ds`` is the train set (clean
    latents — rolled fresh each step); ``val_pair_ds`` + ``val_probe`` are
    precomputed once at startup (the denoiser is frozen ⇒ static across epochs).
    The ``data_provider`` seam injects a fake denoiser + toy datasets for the CPU
    smoke; the real path (``_real_inputs``) builds this from ``--native-dir`` +
    ``--latents-dir``.
    """

    denoiser: Any
    scheduler: Any
    num_steps: int
    clean_ds: Any  #: train: emits {latent, spacing, label}
    val_pair_ds: Any  #: precomputed: emits {winner, loser}
    val_probe: Any = None  #: precomputed RewardPairDataset (generated-end probe).


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/gen_pair_acc",
    mode: str = "max",
    save_top_k: int = 1,
    multi_gpu: bool = False,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` monitoring the generated-end probe.

    The checkpoint monitors ``val/gen_pair_acc`` (the GRPO-regime metric — ranking
    within the all-generated regime; ``mode="max"``). ``val/pair_acc`` and
    ``val/roc_auc`` are logged for diagnosis but are not the selection signal.
    ``auto_insert_metric_name = False`` because the metric key contains a ``/``.
    ``save_last=True`` for resume. Under DDP (``multi_gpu``) the metric is
    rank-local to rank 0, so monitoring is dropped (``save_last`` +
    ``save_top_k=1`` keep the latest) — mirroring the JiT checkpoint's DDP fallback
    (the probe is synced for reporting, but the selection signal stays single-GPU).
    """
    if multi_gpu:
        return ModelCheckpoint(
            dirpath=model_dir,
            filename="reward-{epoch:03d}",
            save_last=True,
            save_top_k=1,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
            save_weights_only=False,
        )
    return ModelCheckpoint(
        dirpath=model_dir,
        filename=f"reward-{{epoch:03d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=save_top_k,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )


def run_reward_training(
    *,
    module: RewardModule,
    inputs: RewardInputs,
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
    """Assemble callbacks + ``Trainer`` and ``fit`` the reward module (the core seam).

    Builds a stock ``ModelCheckpoint`` (monitoring the generated-end probe) and
    runs ``Trainer.fit`` on the clean-latent train datamodule (online rollout per
    step) + the precomputed val pairs. Returns ``(trainer, ckpt)`` so callers can
    find the written ``.ckpt``.

    Args:
        inputs: the clean-latent train dataset + precomputed val/probe + the
            frozen denoiser/scheduler (already on-device in the real path).
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    # Seed deterministically so direct callers (tests, notebooks) get reproducible
    # runs; ``main`` also seeds, harmlessly, before building the module. The
    # per-step t/noise draws then reproduce across runs.
    pl.seed_everything(seed, workers=True)
    multi_gpu = is_multi_gpu(devices)
    ckpt = _build_checkpoint(
        model_dir, monitor_metric=monitor_metric, mode=mode, save_top_k=save_top_k, multi_gpu=multi_gpu
    )
    # Score the fixed generated-end probe in training-batch-size chunks (bounds
    # epoch-end memory); attach the probe if the inputs carry one.
    module.probe_batch_size = int(batch_size)
    if inputs.val_probe is not None and getattr(module, "val_probe", None) is None:
        module.set_val_probe(inputs.val_probe.winners, inputs.val_probe.losers)
    datamodule = build_datamodule(
        inputs.clean_ds,
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
        prog="manifold-train-reward", description="Train the GRPO reward model (online rollout-in-the-loop)."
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c", "--train", default="configs/train/config_reward.yaml", help="reward recipe YAML."
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (reward_model block)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--native-dir",
        default=None,
        help="native JiT export dir (frozen denoiser + VAE scale); required without --data-provider.",
    )
    parser.add_argument(
        "--latents-dir",
        default=None,
        help="directory of clean .pt latents (the latent cache); required without --data-provider.",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config → build → ``run_reward_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> RewardInputs`` returning a fake denoiser + toy clean-latent
    dataset so the full ``main`` path runs without a real latent cache. The real
    path loads the frozen denoiser from ``--native-dir`` and the latent cache from
    ``--latents-dir``, precomputes val/probe once, and rolls fresh train pairs each
    step (ADR-0010).
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # The reward job needs no VAE / UNet-checkpoint path; only the output
    # ``model_dir`` is required (the denoiser comes from --native-dir).
    require_paths(cfg, keys=("model_dir",))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "reward_train", None) is None:
        raise ValueError(
            "Config has no `reward_train` block — use the reward recipe "
            "(-c configs/train/config_reward.yaml), not a JiT train config."
        )

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        inputs = data_provider(cfg, device)
    else:
        # --native-dir / --latents-dir are NOT argparse-required: that would break
        # the data_provider injection seam (the CPU smoke). Validate them here,
        # only on the real path.
        if not args.native_dir or not args.latents_dir:
            raise ValueError(
                "Online reward training needs --native-dir <native JiT export> and "
                "--latents-dir <latent cache> (or inject a data_provider for the smoke)."
            )
        inputs = _real_inputs(cfg, args.native_dir, args.latents_dir, device)

    reward_cfg = cfg.reward_model
    module = RewardModule(
        RewardModel(
            spatial_dims=int(opt(reward_cfg, "spatial_dims", 3)),
            in_channels=int(opt(reward_cfg, "in_channels", 4)),
            channels=int(opt(reward_cfg, "channels", 64)),
            num_layers_d=int(opt(reward_cfg, "num_layers_d", 3)),
            norm=str(opt(reward_cfg, "norm", "BATCH")),
        ),
        lr=float(cfg.reward_train.lr),
        denoiser=inputs.denoiser,
        scheduler=inputs.scheduler,
        num_steps=inputs.num_steps,
    )

    run_reward_training(
        module=module,
        inputs=inputs,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or cfg.reward_train.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(cfg.reward_train.batch_size),
        save_top_k=int(opt(cfg, "checkpoint.save_top_k", 1)),
        seed=seed,
        ckpt_path=args.resume,
    )
    print(f"[manifold-train-reward] done; checkpoints under {cfg.model_dir}")
    return 0


def _real_inputs(cfg, native_dir: str, latents_dir: str, device: torch.device) -> RewardInputs:
    """Build the real online-reward inputs from the native export + latent cache.

    Loads the frozen JiT denoiser (``--native-dir``), partitions subjects once
    (seeded) and constructs the **train** clean-latent dataset over **train
    subjects only** (no leakage at the dataloader — the discriminator never trains
    on a validation subject), and precomputes the full-range val pairs + the
    generated-end probe over val subjects **once** (both static across epochs — the
    denoiser is frozen). The denoiser/scheduler are returned for the Module ctor.
    """
    from ..data.reward_pairs import (
        CleanLatentDataset,
        generate_full_range_val_pairs,
        generate_generated_end_probe,
        load_cached_latents,
        load_frozen_denoiser,
        maybe_per_sample,
        partition_subjects,
    )

    num_steps = int(opt(cfg, "reward_train.num_steps", 2))  # per-step TRAIN rollout (the cost lever)
    # The one-time val/probe precompute uses a larger budget (more accurate, one-off
    # cost) than the every-step train rollout — issue #48: num_steps=4 reserved for
    # the precompute, num_steps=2 for train.
    precompute_num_steps = int(opt(cfg, "reward.precompute_num_steps", 4))
    val_fraction = float(opt(cfg, "reward.val_fraction", 0.2))
    subject_regex = opt(cfg, "reward.subject_regex", None)
    default_spacing = opt(cfg, "reward.spacing", [1.0, 1.0, 1.0])
    default_modality = int(opt(cfg, "reward.modality", 1))
    gen_batch_size = int(opt(cfg, "reward.gen_batch_size", 4))
    n_probe = int(opt(cfg, "reward.n_probe", 64))

    denoiser, scheduler, scaling_factor = load_frozen_denoiser(native_dir)
    denoiser.to(device).eval()
    for p in denoiser.parameters():
        p.requires_grad_(False)

    items, subject_ids = load_cached_latents(latents_dir, subject_regex)
    train_subjects, val_subjects = partition_subjects(
        subject_ids, val_fraction=val_fraction, seed=int(opt(cfg, "random_seed", 0))
    )
    train_items = [it for it, sid in zip(items, subject_ids) if sid in train_subjects]
    val_items = [it for it, sid in zip(items, subject_ids) if sid in val_subjects]
    if not train_items or not val_items:
        raise ValueError(
            f"Empty train ({len(train_items)}) or val ({len(val_items)}) subject split "
            f"from {len(items)} latents — adjust reward.val_fraction."
        )
    rank_zero_info(
        "online reward inputs: %d train / %d val subjects (%d / %d latents).",
        len(train_subjects), len(val_subjects), len(train_items), len(val_items),
    )
    clean_ds = CleanLatentDataset(train_items, scaling_factor)

    # Precompute val (full-range, mirroring the train distribution) + probe
    # ([0, 0.5) generated regime) over VAL subjects once. Clean latents are
    # pre-scaled into the denoiser's training space (the gen functions expect
    # scaled inputs; the train path scales on read — scale applied exactly once
    # in both, ADR-0003).
    val_clean = torch.stack([it["latent"] for it in val_items]).float() * float(scaling_factor)
    spacing = maybe_per_sample([it["spacing"] for it in val_items], default_spacing)
    modality = maybe_per_sample([it["label"] for it in val_items], default_modality)
    val_pair_ds = generate_full_range_val_pairs(
        val_clean, denoiser, scheduler, spacing=spacing, modality=modality,
        num_steps=precompute_num_steps, batch_size=gen_batch_size, seed=int(opt(cfg, "random_seed", 0)), device=device,
    )
    probe = generate_generated_end_probe(
        val_clean[: min(n_probe, len(val_items))], denoiser, scheduler, spacing=spacing, modality=modality,
        num_steps=precompute_num_steps, batch_size=gen_batch_size, seed=int(opt(cfg, "random_seed", 0)), device=device,
    )
    return RewardInputs(
        denoiser=denoiser, scheduler=scheduler, num_steps=num_steps,
        clean_ds=clean_ds, val_pair_ds=val_pair_ds, val_probe=probe,
    )
