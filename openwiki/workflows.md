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

`manifold-train-paired` builds subject-level train/validation splits, warms shared per-volume latents, and trains all source-to-target directions. Validation runs deterministic source-started Heun inference and reports decoded full-volume PSNR/SSIM (`src/manifold/training/paired_cli.py`, `src/manifold/metrics/psnr_ssim_callback.py`).

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
manifold-train-paired-reward
manifold-train-paired-grpo
```

The reward stages load a frozen native generator and train a scorer. The GRPO stages load both a generator/policy and frozen reward model, then optimize singular branches of stochastic trajectories. For paired reward, real targets beat generated targets and the scorer receives the source latent alongside the candidate target. For paired GRPO, the stochastic Brownian bridge is a training mechanism; native paired inference remains deterministic.

Consult `src/manifold/training/reward_cli.py`, `paired_reward_cli.py`, `grpo_cli.py`, and `paired_grpo_cli.py` for current arguments. Treat `scripts/generate_reward_pairs.py` cautiously: it documents an older offline reward-pair workflow, while the current noise-to-data reward module performs online fit-step rollout and precomputes only validation/probe data (`src/manifold/modules/reward.py`, `CONTEXT.md`).

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

EMA training was removed in commit `e89b05d`. `src/manifold/training/export.py` now extracts the raw UNet backbone under the `unet.unet.` state-dict prefix and always reports `unet_state_dict`. Do not pass retired `--ema`/`prefer_ema` options, configure `ema_decays`, or expect `val/fid_avg` and `val/fid_raw`; the single validation metric is `val/fid`, evaluated on the live raw model. Downstream paired reward and paired GRPO generator loading follows the same raw-weight contract.

Only load trusted `.ckpt` files: export calls `torch.load(..., weights_only=False)` because Lightning checkpoints contain full training state.

## Inference

`LatentFlowPipeline` generates from noise; `PairedLatentFlowPipeline` translates a source latent. Both package the UNet, scheduler, and frozen VAE and expose native save/load behavior. NIfTI writing is outside the pipeline boundary: pipelines return decoded `[B,C,D,H,W]` tensors.

When changing inference, verify that module sampling and pipeline sampling still share the same rollout primitive. Relevant tests are `tests/test_pipeline_inference.py`, `test_paired_pipeline_inference.py`, `test_scheduler.py`, and persistence tests.
