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
| Config and training orchestration | `tests/test_config.py`, `test_training_cli.py`, `test_controlnet_cli.py`, `test_training.py` |
| Data, warming, and split isolation | `tests/test_data.py`, `test_paired_latent_cache.py`, `test_paired_manifests.py`, `test_ddp_warm.py`, `test_controlnet_warm_defer.py` |
| Scheduler/module/pipeline behavior | `tests/test_scheduler.py`, `test_module_training.py`, `test_pipeline_inference.py`, `test_controlnet_module_training.py`, `test_controlnet_pipeline_inference.py` |
| FID and image metrics | `tests/test_fid.py`, `test_fid_helpers.py`, `test_paired_fidelity.py`, `test_metric_plot.py` |
| Distributed validation | `tests/test_ddp.py`, `test_ddp_detection.py`, `test_ddp_metrics.py`, `test_ddp_val_honesty.py`, `test_controlnet_ddp_monitor.py` |
| Reward and policy learning | `tests/test_reward.py`, `test_reward_pairs.py`, `test_grpo.py`, `test_controlnet_module_training.py` |
| Callback registry and training spine | `tests/test_callback_registry.py`, plus the registry / monitor / `forbidden_callbacks` assertions in `test_training_cli.py` |
| Frozen-arm contract (register + dual-exclude) | `tests/test_frozen_arm_mixin.py`, `tests/test_controlnet_module_training.py::test_base_is_registered_but_dual_excluded` |
| Per-rank device policy (`DevicePolicy`, post-PG warm) | `tests/test_device_policy.py`, plus `test_ddp_warm.py::test_p1_*` |
| Persistence/export | `tests/test_persistence.py`, export assertions in training/reward/GRPO tests |
| Before/after eval and reporting | `tests/test_paired_fidelity.py`, `test_before_after_eval.py`, `test_eval_cli.py`, `test_comparison_page.py` |

`tests/ddp.py` is the multi-process helper/harness used by DDP tests. Run the focused distributed tests after changing rank gates, sampler assumptions, reduction code, validation callbacks, trainer device selection, or checkpoint monitors.

## Distributed validation contract

ADR-0025 supersedes ADR-0016. Current behavior is:

- **Latent-space x0-MAE (`val/x0_mae`):** every rank computes the cheap reconstruction-MAE on its `DistributedSampler` shard through `LatentX0MAE` (`src/manifold/training/metrics.py`), which attaches a `torchmetrics.MeanMetric` to the module (sample-weighted) so Lightning's DDP reduction produces the true global mean.
- **FID:** synthetic and real examples are rank-strided. Each plane reduces sufficient statistics `(sum_x, sum_xxT, n)`, reconstructs global moments, and computes unbiased FID without gathering feature matrices. Empty local shards contribute zero statistics; only the global count must be at least two.
- **GRPO reward:** every rank validates and logs `val/mean_reward` with `sync_dist=True`.
- **Checkpoint monitors:** configured `val/fid`, `val/x0_mae`, and `val/mean_reward` monitors remain global under DDP. ADR-0037 accepts an in-training paired `val/psnr` / `val/ssim` observer, but no such callback is active yet; when implemented, it must log under synchronized global reduction without entering `monitor_metric` or the loss. The shipped offline comparison instead uses `BeforeAfterEval` + `PairedFidelityMetrics` for a deterministic seed. See [Before/after GRPO evaluation](evaluation.md#accepted-in-training-monitor-planned-not-active).

Key implementations are `src/manifold/metrics/fid/`, `src/manifold/modules/grpo.py`, and the training callback/CLI paths in `src/manifold/training/`.

Do not follow the stale checkpoint comments in `configs/train/config_rflow_jit.yaml` that still describe rank-0-only DDP metrics and unmonitored fallback. The `config_paired_jit.yaml` referenced in older docs no longer exists; the supervised paired translator is `configs/train/config_controlnet_supervised.yaml`. ADR-0025 and current callback/CLI code are authoritative.

## Before/after evaluation runbook

`manifold-eval` is the only eval console entry. It owns a fresh export of the after `.ckpt`, two clean native reloads, deterministic generation, and the eval artifacts. The implementation and artifact schema are the canonical reference in [Before/after GRPO evaluation](evaluation.md#outputs-and-metric-semantics).

For a quick shipped-surface check, confirm that the installed entry point exposes the contract before starting a full run:

```bash
manifold-eval --help
```

Then use the [workflow examples](workflows.md#beforeafter-evaluation). The key runbook constraints are:

- `--before-dir` must be a native directory with a self-describing `model_index.json`; a model-dir or a missing `pipeline_class` fails before export. Never pass an arbitrary directory merely because it contains weights.
- `--after-ckpt` is processed through the standard full-state export bridge. The command writes the derived native tree to `<output>/after_native`; it does not overwrite the before export or return a NIfTI file.
- `--seed`, `--num-inference-steps`, shape/conditioning, and the VAE/scheduler are the pairing contract. Hold them fixed when comparing checkpoints; changing them makes before/after noise or trajectory differences ambiguous.
- ControlNet requires a nonempty held-out split and a warmed cache under the geometry-suffixed cache tag. Match the cache's target geometry and `vae.scaling_factor`; eval reads the scaling factor from the before export and never re-estimates it.
- The report and slice-grid renderer imports Matplotlib lazily on the headless `Agg` backend, but a missing or broken rendering dependency fails the run instead of silently dropping visuals.
- A JiT run records provenance in `metrics.json` but intentionally has no `before.psnr` or `before.ssim`. Those full-reference fields are valid only for ControlNet; the unconditional comparison uses same-noise grids and training FID/reward history.

Focused validation for any metric, driver, CLI, or report change:

```bash
pytest tests/test_paired_fidelity.py tests/test_before_after_eval.py tests/test_eval_cli.py tests/test_comparison_page.py -q
```

Conditional integration check: if a `pyproject.toml` console entry, package barrel, hatch artifact, or installed `manifold-eval` import path changes, build/install the wheel and run the CLI `main(argv)` smoke rather than relying only on the internal tests. This repository has no generated publish mirror or second eval entry point; the hatch package mirror is derived from `pyproject.toml`.

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
- ControlNet-supervised validation (`manifold-train-controlnet`) should use a nonzero subject-level `val_fraction` (the recipe default is `0.2`); `0` permits a train-as-validation fallback and is not an honest generalization estimate. The shared splitter lives in `src/manifold/data/paired_manifests.py` and supports a native-split directory when `env.val_data_base_dir` is a real BraTS directory.
- FID callbacks and `BeforeAfterEval` decode in float32 with MAISI `norm_float16` disabled. Before/after evaluation then applies the same per-volume `min_max_to_unit` convention as the ControlNet pipeline and scores PSNR/SSIM with `data_range=1.0`.
- Current metrics and native exports use raw optimizer weights. Remove references to EMA arms from automation and dashboards.
- Export uses full-state deserialization; only process checkpoints produced by a trusted run.

## Diagnostics

The `scripts/` directory was eliminated in ADR-0033; the helper scripts that used to live there (`scripts/eval_paired_step_sweep.py`, `scripts/diag_brain_mask_psnr.py`, etc.) are not part of this tree. The retained investigation tool is `tests/parity/validate_against_hope.py`, a sampler-parity probe kept as a `<1e-3` proof (ADR-0005) that the modules-side sampler and the inference pipeline roll out the same trajectory. Read its arguments and assumptions before using it against a new dataset or checkpoint.
