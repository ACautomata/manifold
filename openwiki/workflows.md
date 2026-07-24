---
type: Guide
title: Key Workflows
description: JiT and Paired JiT training, reward/GRPO stages, inference, checkpoints, and export.
tags: [workflows, training, inference, checkpoints, export]
---

# Key workflows

## JiT training

`manifold-train` composes environment, recipe, and network configs, builds a cold data bundle, and lets `DataModule.setup()` warm the latent cache after the distributed process group exists. This allows rank-sharded warming rather than duplicating pre-DDP work (`src/manifold/training/cli.py`, `src/manifold/data/warm_datamodule.py`; ADR-0017).

The path is:

1. Discover NIfTI inputs and labels from a directory or manifest.
2. Load and freeze the pretrained MAISI VAE.
3. Sliding-window encode volumes into an unscaled disk/RAM latent cache.
4. Estimate VAE `scaling_factor = 1/std(z)` and scale latent reads.
5. Train `LatentFlowModule` with logit-normal timesteps and JiT's weighted clean-latent prediction loss.
6. Write full-state Lightning checkpoints.
7. Export the selected checkpoint to a native inference directory.

Important constraint: the regular noise-to-data production flow disables validation when no held-out validation source is wired. It refuses to reuse training data as validation; configured FID knobs alone do not guarantee that FID runs (`src/manifold/training/cli.py`).

## Paired training

`manifold-train-paired` builds subject-level train/validation splits, warms shared per-volume latents, and trains all source-to-target directions. Validation runs deterministic source-started Heun inference and reports decoded full-volume PSNR/SSIM (`src/manifold/training/paired_cli.py`, `src/manifold/metrics/psnr_ssym_callback.py`).

Paired JiT conditioning uses a learned MLP that combines source and target contrast embeddings (`concat([embed(src), embed(tgt+offset)])`), replacing the earlier linear sum. This provides greater discriminability across the 12 contrast directions. The optional `paired_direction_offset` config shifts the target embedding row to break A<->B symmetry when needed.

Useful recipe controls in `configs/train/config_paired_jit.yaml` include:

- `val_fraction`: subject-level holdout fraction; keep it nonzero for honest validation.
- `paired_eval.num_inference_steps`: currently 8, based on the repository's step sweep.
- `paired_eval.check_val_every_n_epoch`: set to the training horizon for one final validation pass.
- `diffusion_unet_train.lr_warmup_ratio`: preferred over a fixed count for short runs; warmup steps are clamped so peak LR can be reached.

## Reward and GRPO stages

The console surfaces are:

```text
manifold-train-reward
manifold-train-grpo
manifold-train-controlnet
```

`manifold-train-reward` loads a frozen native generator and trains the shared mode-agnostic PatchGAN scorer on partial-denoise preference pairs. `manifold-train-grpo` then loads a policy and that frozen reward model, forks stochastic trajectories, and scores each terminal latent `z_K` unconditionally. The same reward serves both supported policy paths; the deleted paired-reward pipeline is not part of the current workflow.

There is one GRPO recipe, `configs/train/config_grpo.yaml`. The native artifact supplied through `--native-dir` selects the policy automatically: a raw JiT export trains the UNet, while a supervised ControlNet export trains the ControlNet on its frozen base UNet. There is no mode flag or separate ControlNet-GRPO preset. Shared settings remain under `grpo_train`; `grpo_train.lr` is the UNet default, while optional `controlnet.lr` applies only to the ControlNet path and falls back to `grpo_train.lr` when absent. The ControlNet path also reads `diffusion_unet_inference` for paired cache geometry.

For ControlNet, run `manifold-train-controlnet` first to produce the supervised native export, then pass that export to `manifold-train-grpo`. During GRPO, source conditioning, supervised initialization, and the KL anchor carry translation fidelity; the shared reward contributes realism only. Unconditional FID is suppressed for this path because it ignores the ControlNet and would measure the frozen base, so checkpoint selection uses `val/mean_reward`. The raw UNet path can use `val/fid` when the FID inputs are present.

This workflow depends on the native artifact contract described in [Architecture and source map](architecture.md#configuration-and-persistence). Consult `src/manifold/training/reward_cli.py`, `controlnet_cli.py`, and `grpo_cli.py` for current arguments. Focused guards live in `tests/test_grpo.py` (routing and policy-specific learning rates) and `tests/test_config.py` (the removed mode-specific preset must not return); see [Operations and testing](operations-and-testing.md#standard-checks) for the broader test matrix.

## Checkpoint and export contract

There are two artifact types:

- **Training checkpoint (`.ckpt`)** — full Lightning state for resume and selection.
- **Native checkpoint directory** — deployable UNet/VAE/scheduler components loaded by a pipeline.

Export is the sole supported bridge:

```bash
python scripts/export_checkpoint.py \
  --ckpt <run>/last.ckpt \
  --network-config configs/network/config_network.yaml \
  --vae-checkpoint <vae.pt> \
  --output <native-dir>
```

Paired export additionally selects the paired pipeline and supplies the scaling factor; inspect `scripts/export_checkpoint.py --help` for the exact current flags.

EMA training was removed in commit `e89b05d`. `src/manifold/training/export.py` now extracts the raw UNet backbone under the `unet.unet.` state-dict prefix and always reports `unet_state_dict`. Do not pass retired `--ema`/`prefer_ema` options, configure `ema_decays`, or expect `val/fid_avg` and `val/fid_raw`; the single validation metric is `val/fid`, evaluated on the live raw model. Reward and GRPO policy loading follows the same raw-weight contract.

Only load trusted `.ckpt` files: export calls `torch.load(..., weights_only=False)` because Lightning checkpoints contain full training state.

## Inference

`LatentFlowPipeline` generates from noise; `PairedLatentFlowPipeline` translates a source latent. Both package the UNet, scheduler, and frozen VAE and expose native save/load behavior. NIfTI writing is outside the pipeline boundary: pipelines return decoded `[B,C,D,H,W]` tensors.

When changing inference, verify that module sampling and pipeline sampling still share the same rollout primitive. Relevant tests are `tests/test_pipeline_inference.py`, `test_paired_pipeline_inference.py`, `test_scheduler.py`, and persistence tests.
