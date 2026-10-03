---
type: Guide
title: Operations and Testing
description: Setup, validation behavior, distributed metrics, runbook cautions, and focused test commands for Manifold.
tags: [operations, testing, distributed, validation, ddp]
verified:
  - by: openwiki/0.7.0
    at: 2026-10-03T13:11:49.042Z
sources:
  - id: openwiki-source-5ec900d9e277a0cf8e19af49
    resource: repo://docs/adr/0025-all-ddp-validation-no-rank0-gate.md
  - id: openwiki-source-7fe7772700d519e2f220697b
    resource: repo://docs/adr/0030-fid-callback-split.md
  - id: openwiki-source-1f077c4de341ab7dab070abb
    resource: repo://docs/adr/0037-controlnet-fidelity-in-training-monitor.md
  - id: openwiki-source-f006b01e509da9dc1e507d6c
    resource: repo://docs/prd/ddp-correctness.md
  - id: openwiki-source-121ee09471abdc662d9d3b48
    resource: repo://src/manifold/eval/before_after.py
  - id: openwiki-source-5efb4260c1cd509a006515a9
    resource: repo://src/manifold/metrics/fid/callback.py
  - id: openwiki-source-76af229eab7cbd1fa104f790
    resource: repo://src/manifold/metrics/fid/rollout.py
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-4a976a922e9a1286c789e272
    resource: repo://src/manifold/training/callbacks/paired_fidelity.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
  - id: openwiki-source-703199deac93d733d94afae7
    resource: repo://src/manifold/training/metrics.py
  - id: openwiki-source-32dd8914128afdc039c854cb
    resource: repo://tests/test_ddp_metrics.py
  - id: openwiki-source-c07abd7a60c092df1a8401ed
    resource: repo://tests/test_fid_helpers.py
  - id: openwiki-source-43f6a22439495a06382f037c
    resource: repo://tests/test_paired_fidelity_ddp.py
generated: { by: "openwiki/0.7.0", at: "2026-10-03T13:11:49.042Z" }
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
| Distributed validation | `tests/test_ddp.py`, `test_ddp_detection.py`, `test_ddp_metrics.py`, `test_ddp_val_honesty.py`, `test_ddp_warm.py`, `test_controlnet_ddp_monitor.py`, `test_paired_fidelity_ddp.py` |
| Reward and policy learning | `tests/test_reward.py`, `test_reward_pairs.py`, `test_grpo.py`, `test_controlnet_module_training.py` |
| Callback registry and training spine | `tests/test_callback_registry.py`, plus the registry / monitor / `forbidden_callbacks` assertions in `test_training_cli.py` |
| Frozen-arm contract (register + dual-exclude) | `tests/test_frozen_arm_mixin.py`, `tests/test_controlnet_module_training.py::test_base_is_registered_but_dual_excluded` |
| Per-rank device policy (`DevicePolicy`, post-PG warm) | `tests/test_device_policy.py`, plus `test_ddp_warm.py::test_p1_*` |
| Persistence/export | `tests/test_persistence.py`, export assertions in training/reward/GRPO tests |
| Before/after eval and reporting | `tests/test_paired_fidelity.py`, `test_before_after_eval.py`, `test_eval_cli.py`, `test_comparison_page.py` |
| In-training paired-fidelity monitor (ADR-0037) | `tests/test_paired_fidelity_callback.py` (wiring), `tests/test_paired_fidelity_ddp.py` (DDP), plus `tests/test_callback_registry.py::test_paired_fidelity_*` (registry contract) |
| FID composable helpers + collective-count hardening (ADR-0030) | `tests/test_fid_helpers.py` (VramStage, FixedSampleRollout, LatentDecoder, SufficientStatsReducer, error-rendezvous) |

`tests/ddp.py` is the multi-process helper/harness used by DDP tests. Run the focused distributed tests after changing rank gates, sampler assumptions, reduction code, validation callbacks, trainer device selection, or checkpoint monitors.

## Distributed validation contract

ADR-0025 supersedes ADR-0016. Current behavior is:

- **Latent-space x0-MAE (`val/x0_mae`):** every rank computes the cheap reconstruction-MAE on its `DistributedSampler` shard through `LatentX0MAE` (`src/manifold/training/metrics.py`), which attaches a `torchmetrics.MeanMetric` to the module (sample-weighted) so Lightning's DDP reduction produces the true global mean.
- **FID:** synthetic and real examples are rank-strided. Each plane reduces sufficient statistics `(sum_x, sum_xxT, n)`, reconstructs global moments, and computes unbiased FID without gathering feature matrices. Empty local shards contribute zero statistics; only the global count must be at least two. The staged phase is hardened by an **error-flag `all_reduce(MAX)` rendezvous** (ADR-0030) before every reduction-bearing phase so a rank-local exception cannot strand other ranks — see [DDP failure modes to guard](#ddp-failure-modes-to-guard) for the collective-count invariant.
- **GRPO reward:** every rank validates and logs `val/mean_reward` with `sync_dist=True` (the val dataloader is evenly sharded, so the synced epoch mean is the global mean — mirroring the `reward.py val/pair_acc` convention).
- **In-training paired fidelity (`val/psnr` / `val/ssim`, ADR-0037):** the supervised ControlNet stage ships the observe-only `PairedFidelityCallback` (`src/manifold/metrics/paired_callback.py`) registered via `PairedFidelitySpec` (`src/manifold/training/callbacks/paired_fidelity.py`). It runs the module's full Heun `controlnet_rollout` on a small fixed paired subset under fixed seeded noise. Under DDP, every rank evaluates the **same** fixed subset redundantly — DDP-synchronized weights + identical seeded noise + identical fixed input produce identical per-rank results — so the cross-rank reduction is just Lightning's `torchmetrics` sync on the two module-attached `MeanMetric`s. There is no FID-style sharding and no error-rendezvous machinery (ADR-0030 collective-count invariance holds trivially). The monitor is **observe-only**: it never drives checkpoint selection (`monitor_metric` stays `val/x0_mae`), never enters the loss, and never touches the optimizer or EMA. The fixed-subset source is resolved lazily through `getattr(source, "val_latent_ds", source)` so the cold-path `setup()` replacement of `val_latent_ds` is honored (F5). See [Active in-training monitor: `PairedFidelityCallback` + `PairedFidelitySpec`](evaluation.md#active-in-training-monitor-pairedfidelitycallback--pairedfidelityspec) and [Callback registry and training spine](callback-registry.md#built-in-specs) for the registry contract.
- **Checkpoint monitors:** configured `val/fid`, `val/x0_mae`, `val/mean_reward`, and the paired-fidelity `val/psnr` / `val/ssim` are all global under DDP (the latter via redundant-observation; monitor selection stays on `val/x0_mae`).

Key implementations are `src/manifold/metrics/fid/` (FIDCallback + ADR-0030 helpers), `src/manifold/metrics/paired_callback.py` (PairedFidelityCallback), `src/manifold/training/callbacks/paired_fidelity.py` (PairedFidelitySpec), `src/manifold/modules/grpo.py`, and the training callback/CLI paths in `src/manifold/training/`.

Do not follow the stale checkpoint comments in `configs/train/config_rflow_jit.yaml` that still describe rank-0-only DDP metrics and unmonitored fallback. The `config_paired_jit.yaml` referenced in older docs no longer exists; the supervised paired translator is `configs/train/config_controlnet_supervised.yaml`. ADR-0025, ADR-0030, ADR-0037, and current callback/CLI code are authoritative.

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

- **Collective-count invariance (ADR-0030 hardening):** `FIDCallback.on_validation_epoch_end` performs an **error-flag `all_reduce(MAX)`** before each reduction-bearing step (stage, real-decode, synth-decode). Any rank-local exception sets the flag locally, the rendezvous fires, and every rank takes the same abort branch together (`val/fid=+inf`) — a rank-local exception cannot strand peers in a missing `all_reduce`. This generalizes the older "disable-flag" `all_reduce` to **any** exception in the staged phase, not only backbone-factory failures. The number and order of collectives is identical on every rank in every code path; tests in `tests/test_fid_helpers.py` assert the invariant under a forced rank-local exception.
- **Empty FID shards:** an empty local shard contributes correctly sized zero sufficient statistics `(sum_x, sum_xxT, n=0)` rather than skipping the `all_reduce`. The collective cannot deadlock; covariance validity is checked only after global reduction (only the global count must be ≥ 2 for a plane's FID to be computed).
- **One-sample local shard is valid:** it contributes its first/second-order sums to the global reduction. Do not collapse an `n=1` shard into the empty-shard path — the unbiased covariance's `n-1` denominator depends on the global count, not the local count.
- **Manual `all_reduce` vs `sync_dist=True`:** never combine them for the same value. The two reductions double-count: `sync_dist` would average across ranks, the manual `all_reduce` would too, and the logged value would be a mean-of-rank-means or worse.
- **Rank-strided FID seeds:** `FixedSampleRollout` uses `seed + i` for `i % world == rank` so the global synth set is the union across ranks rather than `world ×` rank 0. A non-strided seed would generate `num_synth * world` volumes per rank and silently distort `val/fid`.
- **Per-rank feature extraction under all-rank FID (ADR-0025):** every rank builds `feature_net` (the ~100 MB RadImageNet ResNet50 load) and extracts features for its own shard. The lazy `feature_net_factory` invocation lives in `VramStage.__enter__`; the load fires once per rank, not once total. The dual-write to `feature_net` keeps the existing direct-injection seam (CPU smoke tests pass a fake `feature_net`) working under DDP.
- **`PairedFidelityCallback` DDP is redundant, not sharded:** every rank evaluates the same fixed paired subset under the same seeded noise on DDP-synchronized weights — the per-rank result is identical, and the cross-rank reduction is Lightning's `torchmetrics` sync on the two module-attached `MeanMetric`s. There is no FID-style sharding and no error-rendezvous machinery on this path; `tests/test_paired_fidelity_ddp.py` locks both the no-hang and the per-rank value-equality claims (identical `val/psnr` and `val/ssim` on both ranks).

These cases are covered principally by `tests/test_fid.py`, `tests/test_fid_helpers.py`, `tests/test_ddp_metrics.py`, `tests/test_ddp_val_honesty.py`, and `tests/test_paired_fidelity_ddp.py`.

## Validation and checkpoint cautions

- Noise-to-data production validation is disabled unless a held-out source is wired; the code refuses train-as-validation leakage. In that case checkpointing falls back to periodic/last rather than monitored FID.
- ControlNet-supervised validation (`manifold-train-controlnet`) should use a nonzero subject-level `val_fraction` (the recipe default is `0.2`); `0` permits a train-as-validation fallback and is not an honest generalization estimate. The shared splitter lives in `src/manifold/data/paired_manifests.py` and supports a native-split directory when `env.val_data_base_dir` is a real BraTS directory.
- FID callbacks and `BeforeAfterEval` decode in float32 with MAISI `norm_float16` disabled. Before/after evaluation then applies the same per-volume `min_max_to_unit` convention as the ControlNet pipeline and scores PSNR/SSIM with `data_range=1.0`.
- The in-training `PairedFidelityCallback` is **observe-only**: it never drives checkpoint selection (`monitor_metric` stays `val/x0_mae`), never enters the loss, and never touches the optimizer or EMA. The `paired_fidelity` spec is registered on the supervised ControlNet CLI's default callback list but is forbidden on the ControlNet-GRPO path until the deliberate ADR-0037 follow-up lands. Do not promote `val/psnr` / `val/ssim` to a checkpoint monitor until the signal has been watched.
- The in-training `val/psnr` / `val/ssim` and the offline `metrics.json` PSNR/SSIM share the same metric, normalization, and rollout primitive by construction. A change that breaks one side without the other is a comparison bug — touching `PairedFidelityMetrics`, `min_max_to_unit`, or the rollout step count crosses both surfaces.
- Current metrics and native exports use raw optimizer weights. Remove references to EMA arms from automation and dashboards.
- Export uses full-state deserialization; only process checkpoints produced by a trusted run.

Focused validation per area:

```bash
# Eval surface (offline before/after, console entry, report builder, paired-fidelity metric)
pytest tests/test_paired_fidelity.py tests/test_before_after_eval.py tests/test_eval_cli.py tests/test_comparison_page.py -q

# In-training paired-fidelity monitor (ADR-0037): wiring + registry contract
pytest tests/test_paired_fidelity_callback.py "tests/test_callback_registry.py::test_paired_fidelity_spec_*" -q

# DDP contracts: distributed validation, mean-metric honesty, paired-fidelity DDP
pytest tests/test_ddp.py tests/test_ddp_detection.py tests/test_ddp_metrics.py tests/test_ddp_val_honesty.py tests/test_ddp_warm.py tests/test_controlnet_ddp_monitor.py tests/test_paired_fidelity_ddp.py -q

# FID composable helpers + collective-count hardening (ADR-0030)
pytest tests/test_fid.py tests/test_fid_helpers.py -q
```

DDP tests reuse the CPU 2-rank harness in `tests/ddp.py`; a hang surfaces as a pytest timeout and the per-rank JSON result is the gate.

## Diagnostics

The `scripts/` directory was eliminated in ADR-0033; the helper scripts that used to live there (`scripts/eval_paired_step_sweep.py`, `scripts/diag_brain_mask_psnr.py`, etc.) are not part of this tree. The retained investigation tool is `tests/parity/validate_against_hope.py`, a sampler-parity probe kept as a `<1e-3` proof (ADR-0005) that the modules-side sampler and the inference pipeline roll out the same trajectory. Read its arguments and assumptions before using it against a new dataset or checkpoint.
