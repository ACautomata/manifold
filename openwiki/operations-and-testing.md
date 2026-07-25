---
type: Guide
title: Operations and Testing
description: Setup, validation behavior, distributed metrics, runbook cautions, and focused test commands for Manifold.
tags: [operations, testing, distributed, validation, ddp]
---

# Operations and testing

## Standard checks

```bash
pytest
ruff check .
```

For focused changes, start with the nearest tests:

| Area | Tests |
|---|---|
| Config and training orchestration | `tests/test_config.py`, `test_training_cli.py`, `test_paired_training_cli.py`, `test_training.py` |
| Data, warming, and split isolation | `tests/test_data.py`, `test_paired_data.py`, `test_ddp_warm.py` |
| Scheduler/module/pipeline behavior | `tests/test_scheduler.py`, `test_module_training.py`, `test_pipeline_inference.py`, paired equivalents |
| FID and paired image metrics | `tests/test_fid.py`, `test_paired_metrics.py` |
| Distributed validation | `tests/test_ddp.py`, `test_ddp_detection.py`, `test_ddp_metrics.py`, `test_ddp_val_honesty.py` |
| Reward and policy learning | `tests/test_reward.py`, `test_reward_pairs.py`, `test_grpo.py`, `test_controlnet_module_training.py` |
| Persistence/export | `tests/test_persistence.py`, `test_paired_persistence.py`, export assertions in training/reward/GRPO tests |

`tests/ddp.py` is the multi-process helper/harness used by DDP tests. Run the focused distributed tests after changing rank gates, sampler assumptions, reduction code, validation callbacks, trainer device selection, or checkpoint monitors.

## Distributed validation contract

ADR-0025 supersedes ADR-0016. Current behavior is:

- **Paired PSNR/SSIM:** every rank decodes its `DistributedSampler` shard. The callback accumulates per-volume sums/count and manually `all_reduce`s them at epoch end. It logs global `val/psnr` and `val/ssim` without `sync_dist`.
- **FID:** synthetic and real examples are rank-strided. Each plane reduces sufficient statistics `(sum_x, sum_xxT, n)`, reconstructs global moments, and computes unbiased FID without gathering feature matrices. Empty local shards contribute zero statistics; only the global count must be at least two.
- **GRPO reward:** every rank validates and logs `val/mean_reward` with `sync_dist=True`.
- **Checkpoint monitors:** `val/fid`, `val/psnr`, and `val/mean_reward` remain active under DDP because the monitored values are global.

Key implementations are `src/manifold/metrics/fid/`, `src/manifold/modules/grpo.py`, and the training callback/CLI paths in `src/manifold/training/`.

Do not follow the stale checkpoint comments in `configs/train/config_rflow_jit.yaml` and `config_paired_jit.yaml` that still describe rank-0-only DDP metrics and unmonitored fallback. ADR-0025 and current callback/CLI code are authoritative.

## Distributed validation runbook

The all-rank policy reverses a rank-0 workaround for a reproducible 8-DCU/DTK stall during concurrent full-volume MAISI VAE decode. NVIDIA 8-GPU and single-DCU runs were reported healthy, but ADR-0025 explicitly marks the sugon verification as pending.

Before relying on multi-DCU best-by-metric selection, run one validation epoch on all eight ranks with a small subset, following the ADR's probe parameters:

```text
--max-epochs 1 check_val_every_n_epoch=1 val_subset_size=4
```

Profile all ranks through the validation epoch and confirm that every rank exits decode and logs the same global metric. The VAE network config currently uses `num_splits: 4`, `dim_split: 1`, and `save_mem: true`, but this configuration was already present during the original stall; do not claim it is a proven fix.

If the probe hangs, ADR-0025's fallback order is:

1. Serialize GPU decode across ranks while retaining global reduction.
2. Move decode to CPU (`sw_device='cpu'`).

The metric contract stays global; only the decode strategy should change. A return to rank-0-only metrics would again make checkpoint selection shard-biased.

## Diagnosing deadlock vs. slow validation

ADR-0025 includes diagnostic guidance for distinguishing the DCU deadlock from slow validation. The symptom triad "processes `Sl` (sleeping) + log mtime stalled + no tqdm output" is a false positive — it also describes healthy, fully-loaded validation under 8-DDP.

Before diagnosing a deadlock, use load-bearing signals:

- `hy-smi` (after `source /opt/dtk/env.sh`): DCU% near 0 with no progress = stalled; DCU% ~100% = computing (slow, not deadlocked).
- `SIGTERM` response: the 2026-07-14 stall ignored `SIGTERM` (required `SIGKILL`); a merely-slow validation terminates on `SIGTERM`.
- `py-spy` on all ranks: identical frozen frame in `sliding_window_inference -> _conv_forward` = deadlock.

The "Sl + log stalled" triad alone is insufficient; do not act on it without confirming one of the above signals.

## DDP failure modes to guard

- Every rank must enter collectives in the same order. FID synchronizes feature-network disablement before entering moment reductions so a rank-local load failure cannot strand other ranks.
- Empty FID shards must not be sent through MAISI decode; they contribute correctly sized zero sufficient statistics.
- A one-sample local shard is valid and must contribute its first/second-order sums; covariance validity is checked only after global reduction.
- Do not combine manual `all_reduce` with `sync_dist=True` for the same value.
- Preserve rank-strided FID seeds (`seed + global_index`) so the distributed sample union matches the requested global sample count rather than multiplying it by world size.

These cases are covered principally by `tests/test_fid.py`, `test_ddp_metrics.py`, and `test_ddp_val_honesty.py`.

## Validation and checkpoint cautions

- Noise-to-data production validation is disabled unless a held-out source is wired; the code refuses train-as-validation leakage. In that case checkpointing falls back to periodic/last rather than monitored FID.
- Paired validation should use a nonzero subject-level `val_fraction`; `0` permits a train-as-validation fallback and is not an honest generalization estimate.
- Metric callbacks decode in float32 with MAISI `norm_float16` disabled, then restore the VAE to CPU to free training VRAM.
- Current metrics and native exports use raw optimizer weights. Remove references to EMA arms from automation and dashboards.
- Export uses full-state deserialization; only process checkpoints produced by a trusted run.

## Diagnostics

`scripts/eval_paired_step_sweep.py` evaluates paired integration-step choices. `scripts/diag_brain_mask_psnr.py`, `diag_paired_ceiling.py`, and `diag_raw_rollout.py` are targeted investigation tools rather than the primary training path. Read their arguments and assumptions before using them against a new dataset or checkpoint.
ainst a new dataset or checkpoint.
