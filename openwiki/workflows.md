---
type: Guide
title: Key Workflows
description: JiT, supervised ControlNet translator, reward/GRPO training stages, before/after evaluation, inference, checkpoints, and export.
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

`manifold-train-controlnet` is the supervised paired translator (ADR-0027 stage 1; the old `paired_cli` and the separate paired-reward pipeline were retired — the latter in ADR-0034). It loads a frozen JiT native export via `--native-dir`, warms shared per-volume latents via `--latents-dir`, and trains a trainable ControlNet over the frozen base through `ControlNetLatentFlowModule`. Validation currently uses the latent-space `val/x0_mae` callback (`src/manifold/training/metrics.py`), which is fast and runs through the shared `controlnet_rollout` primitive that native inference also uses (`src/manifold/modules/controlnet_sampler.py`; ADR-0005). ADR-0037 accepts a complementary fixed-subset `val/psnr` / `val/ssim` monitor, but that callback is not implemented yet and will remain observe-only, not a checkpoint selector. The native supervised export then becomes stage-1 input to `manifold-train-grpo` for the ControlNet policy path.

Paired conditioning uses a learned MLP that combines source and target contrast embeddings (`concat([embed(src), embed(tgt+offset)])`), with the optional `paired_direction_offset` shifting the target embedding row to break A<->B symmetry. The learned MLP provides greater discriminability across the 12 contrast directions and replaces the earlier linear sum.

Useful recipe controls in `configs/train/config_controlnet_supervised.yaml` include:

- `controlnet.num_inference_steps`: Heun steps for the validation rollout; mirror the JiT denoiser's production inference count.
- `controlnet.val_fraction`: held-out subject fraction when `env.val_data_base_dir` is not a BraTS directory (the shipped `environment_brats2023.yaml` points it at a manifest JSON, which `_train_val_manifests` rejects and falls back to the fraction).
- `diffusion_unet_train.lr_warmup_ratio`: preferred over a fixed count for short runs; warmup steps are clamped so peak LR can be reached.

## Reward and GRPO stages

The console surfaces are:

```text
manifold-train-reward
manifold-train-grpo
manifold-train-controlnet
manifold-eval
```

`manifold-train-reward` loads a frozen native generator and trains the shared mode-agnostic PatchGAN scorer on partial-denoise preference pairs. `manifold-train-grpo` then loads a policy and that frozen reward model, forks stochastic trajectories, and scores each terminal latent `z_K` unconditionally. The same reward serves both supported policy paths; the deleted paired-reward pipeline is not part of the current workflow.

There is one GRPO recipe, `configs/train/config_grpo.yaml`. The native artifact supplied through `--native-dir` selects the policy automatically: a raw JiT export trains the UNet, while a supervised ControlNet export trains the ControlNet on its frozen base UNet. There is no mode flag or separate ControlNet-GRPO preset. Shared settings remain under `grpo_train`; `grpo_train.lr` is the UNet default, while optional `controlnet.lr` applies only to the ControlNet path and falls back to `grpo_train.lr` when absent. The ControlNet path also reads `diffusion_unet_inference` for paired cache geometry.

For ControlNet, run `manifold-train-controlnet` first to produce the supervised native export, then pass that export to `manifold-train-grpo`. During GRPO, source conditioning, supervised initialization, and the KL anchor carry translation fidelity; the shared reward contributes realism only. Unconditional FID is suppressed for this path because it ignores the ControlNet and would measure the frozen base, so checkpoint selection uses `val/mean_reward`. The suppression is enforced post-merge through `TrainingSpine.run(..., forbidden_callbacks={"fid": ...}, forbidden_monitors={"val/fid": ...})` (ADR-0032) — a YAML or `--callbacks` override cannot re-enable FID or its monitor on the ControlNet policy path. The raw UNet path can use `val/fid` when the FID inputs are present.

### GRPO scheduler: `rollout_range` single Heun interval loop (ADR-0005)

The GRPO rollout (`singular_branch_rollout` in `src/manifold/modules/grpo.py`)
walks two slices per branch:

- the **anchor slice** `0 → max_k` (one Heun trajectory across the deterministic
  prefix, shared by every sibling), and
- the **suffix slice** `k+1 → n` (the per-sibling stochastic branch that
  produces the GRPO group).

Both slices now run through `FlowMatchGRPOScheduler.rollout_range` (commit
`d13691d`, issue #211, ADR-0005 single-copy). The method lives on the
scheduler, not as a free function in `grpo.py` — both the partial and GRPO
schedulers inherit the parent `FlowMatchHeunDiscreteScheduler.heun_rollout`
verbatim, and the GRPO scheduler adds `rollout_range` on top. The acceptance
tests are the source-level guards `tests/test_grpo.py::test_grpo_rollout_loop_lives_on_the_scheduler_not_module_functions`
(no `_heun_one_step` / `_heun_rollout` free function survives in `grpo.py`)
and `tests/test_scheduler.py::test_heun_rollout_primitive_subclasses_inherit_verbatim`
(the three schedulers share the inherited primitive by identity), plus
`test_grpo_scheduler_rollout_range_is_the_anchor_suffix_interval_loop` which
exercises the anchor and the suffix slice through the scheduler method.

This workflow depends on the native artifact contract described in [Architecture and source map](architecture.md#configuration-and-persistence). Consult `src/manifold/training/reward_cli.py`, `controlnet_cli.py`, and `grpo_cli.py` for current arguments. Focused guards live in `tests/test_grpo.py` (routing and policy-specific learning rates) and `tests/test_config.py` (the removed mode-specific preset must not return); see [Operations and testing](operations-and-testing.md#standard-checks) for the broader test matrix.

## Before/after evaluation

Use the shipped `manifold-eval` to compare the supervised native export with the post-GRPO Lightning checkpoint. It infers the policy from the before artifact's `model_index.json`, exports the after checkpoint through the normal checkpoint-to-native bridge, reloads both pipelines, and runs the same-seed evaluation described in [Before/after GRPO evaluation](evaluation.md#runtime-flow). JiT records only seed, step count, and sample count; ControlNet additionally records full-reference 3D PSNR/SSIM against each real target.

```bash
manifold-eval \
  --before-dir <jit-native-export> \
  --after-ckpt <grpo-jit>/lightning/last.ckpt \
  --output runs/eval_jit \
  --device cuda
```

For ControlNet, the before directory must be the **supervised** export. Evaluation reads held-out `(src, tgt)` pairs from a warmed cache and refuses an empty validation split or cache miss; it does not estimate a new VAE scaling factor.

```bash
manifold-eval \
  --before-dir <supervised-controlnet-export> \
  --after-ckpt <grpo-controlnet>/lightning/last.ckpt \
  --output runs/eval_controlnet \
  --data-base-dir <brats-root> \
  --val-data-base-dir <brats-held-out-root> \
  --latents-dir <paired-latent-cache> \
  --num-pairs 8 \
  --device cuda
```

The command writes `metrics.json`, `slice_grid_<i>.png`, and `<output>/after_native`. A separate `ComparisonPageBuilder` turns those local artifacts plus optional JiT `metrics.csv` files into a self-contained HTML report; it is not another console entry point. Keep this offline comparison distinct from ADR-0037's accepted in-training monitor, which is planned but not active.

## Checkpoint and export contract

There are two artifact types:

- **Training checkpoint (`.ckpt`)** — full Lightning state for resume and selection.
- **Native checkpoint directory** — deployable UNet/VAE/scheduler components loaded by a pipeline.

Export is the sole supported bridge. The `scripts/export_checkpoint.py` shell was retired in ADR-0033; the same logic is now the `manifold-export` console entry:

```bash
manifold-export \
  --ckpt <run>/last.ckpt \
  --network-config configs/network/config_network.yaml \
  --vae-checkpoint <vae.pt> \
  --output <native-dir>
```

`--pipeline {jit,paired,controlnet}` selects which inference component tree to write (default `jit`). The ControlNet export additionally requires `--base-native-dir` (the JiT native export the supervised ControlNet was trained against — the ControlNet `.ckpt` registers only the trainable residuals). `--pipeline paired` is currently a stale code reference (the `PairedLatentFlowPipeline` module was deleted alongside the paired-reward pipeline in ADR-0034); do not use it until it is either restored or removed from the `argparse` choices. Inspect `manifold-export --help` for the current flags.

`manifold-eval` invokes this same bridge internally rather than accepting a separate network config. It uses the before native directory to load the probe component structure, bakes the after `.ckpt` to `<eval-output>/after_native`, discards the mutated probe, and reloads both native directories before generation. This prevents the in-place export bake from contaminating the before side and keeps VAE, scheduler, and architecture identical by construction.

EMA training was removed in commit `e89b05d`. `src/manifold/training/export.py` now extracts the raw UNet backbone under the `unet.unet.` state-dict prefix and always reports `unet_state_dict`. Do not pass retired `--ema`/`prefer_ema` options, configure `ema_decays`, or expect `val/fid_avg` and `val/fid_raw`; the single validation metric is `val/fid`, evaluated on the live raw model. Reward and GRPO policy loading follows the same raw-weight contract.

Supervised ControlNet `.ckpt` files follow the same contract under
[`FrozenArmMixin`](frozen-arm-and-device-policy.md#frozenarmmixin-register--dual-exclude)
(ADR-0031 A1): the trainable ControlNet is the only arm that survives
`state_dict()`; the frozen base is registered on the host (so Lightning owns
its device placement) but stripped from the checkpoint by the `state_dict`
override and strict-loaded as an allow-list by `load_state_dict`. A supervised
ControlNet ckpt therefore carries only `controlnet.*` keys; a missing or
unexpected `controlnet.*` key still raises (no blanket `strict=False`).
Export bakes those `controlnet.*` keys into a fresh ControlNet built from the
network config and restores the frozen base from `--base-native-dir` (the JiT
native export the supervised stage trained against).

Only load trusted `.ckpt` files: export calls `torch.load(..., weights_only=False)` because Lightning checkpoints contain full training state.

## Inference

Two native inference pipelines package the components and expose native save/load behavior:

- `LatentFlowPipeline` — JiT noise-to-data generator (UNet + scheduler + frozen VAE).
- `ControlNetLatentFlowPipeline` — supervised paired translator (frozen JiT base UNet + trainable ControlNet + scheduler + frozen VAE).

The previous `PairedLatentFlowPipeline` (the 2·C src→tgt UNet paired-JiT inference pipeline) was retired with the paired-reward pipeline (ADR-0034); `manifold.pipelines.paired_latent_flow` no longer exists, and the `--pipeline paired` choice in `manifold-export` is a stale reference — use `--pipeline jit` (default) or `--pipeline controlnet`. The JiT and ControlNet pipelines share the per-component `save_pretrained` / `from_pretrained` contract.

NIfTI writing is outside the pipeline boundary: pipelines return decoded `[B,C,D,H,W]` tensors.

When changing inference, verify that module sampling and pipeline sampling still share the same rollout primitive (`src/manifold/modules/sampler.py`, `controlnet_sampler.py`). Relevant tests are `tests/test_pipeline_inference.py`, `test_controlnet_pipeline_inference.py`, `test_scheduler.py`, and persistence tests.

## Change navigation

| Change area | Where to start | Focused tests | Minimal validation |
| --- | --- | --- | --- |
| Cache build / reconstruction | `src/manifold/data/latent_pipeline.py`, `src/manifold/data/paired_latent_dataset.py`, `src/manifold/data/latent_dataset.py` | `tests/test_paired_latent_cache.py`, `tests/test_ddp_warm.py` | `pytest tests/test_paired_latent_cache.py tests/test_ddp_warm.py -q` |
| Supervised ControlNet stage | `src/manifold/training/controlnet_cli.py`, `src/manifold/modules/controlnet_latent_flow.py`, `configs/train/config_controlnet_supervised.yaml` | `tests/test_controlnet_module_training.py`, `tests/test_controlnet_cli.py` | `pytest tests/test_controlnet_module_training.py tests/test_controlnet_cli.py -q` |
| Reward + GRPO stages | `src/manifold/training/{reward_cli,grpo_cli}.py`, `src/manifold/modules/{reward,grpo}.py` | `tests/test_reward.py`, `tests/test_reward_pairs.py`, `tests/test_grpo.py` | `pytest tests/test_reward.py tests/test_reward_pairs.py tests/test_grpo.py -q` |
| `manifold-export` bridge + persistence | `src/manifold/training/export.py`, `src/manifold/training/export_cli.py`, `src/manifold/pipelines/{latent_flow,controlnet_latent_flow}.py` | `tests/test_persistence.py`, `tests/test_controlnet_cli.py` | `pytest tests/test_persistence.py tests/test_controlnet_cli.py -q` |
| `manifold-eval` paired fidelity + comparison page | `src/manifold/eval/cli.py`, `src/manifold/eval/before_after.py`, `src/manifold/eval/comparison_page.py`, `src/manifold/metrics/paired.py` | `tests/test_paired_fidelity.py`, `tests/test_before_after_eval.py`, `tests/test_eval_cli.py`, `tests/test_comparison_page.py` | `pytest tests/test_paired_fidelity.py tests/test_before_after_eval.py tests/test_eval_cli.py tests/test_comparison_page.py -q` |
| Native inference pipelines | `src/manifold/pipelines/{latent_flow,controlnet_latent_flow}.py`, `src/manifold/modules/{sampler,controlnet_sampler}.py` | `tests/test_pipeline_inference.py`, `tests/test_controlnet_pipeline_inference.py`, `tests/test_scheduler.py` | `pytest tests/test_pipeline_inference.py tests/test_controlnet_pipeline_inference.py tests/test_scheduler.py -q` |

Scope boundaries: each row maps a workflow seam to one CLI plus its module and pipeline siblings; do not rerun the full suite for a single-row change. New `--pipeline paired` restoration or a new in-training monitor must add the matching test before becoming supported.
