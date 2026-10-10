---
type: Guide
title: Operations and Testing
description: Setup, validation behavior, distributed metrics contract (ADR-0025), before/after eval runbook, deadlock-vs-slow-validation diagnostics, and the focused test matrix for the Manifold codebase.
tags: [operations, testing, distributed, validation, ddp]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-10T14:21:58.700Z
sources:
  - id: openwiki-source-c43f56ac0f1e2fe9b56cde75
    resource: repo://configs/train/config_rflow_jit.yaml
  - id: openwiki-source-3144653c6b086e1c0a9159f1
    resource: repo://docs/adr/0016-multigpu-validation-rank0-only-honest-guards.md
  - id: openwiki-source-5ec900d9e277a0cf8e19af49
    resource: repo://docs/adr/0025-all-ddp-validation-no-rank0-gate.md
  - id: openwiki-source-121ee09471abdc662d9d3b48
    resource: repo://src/manifold/eval/before_after.py
  - id: openwiki-source-e22b4e5926edbd63eb845214
    resource: repo://src/manifold/eval/cli.py
  - id: openwiki-source-5efb4260c1cd509a006515a9
    resource: repo://src/manifold/metrics/fid/callback.py
  - id: openwiki-source-7072934e9d047112d13b6121
    resource: repo://src/manifold/metrics/fid/decoder.py
  - id: openwiki-source-03f0e1e662125e7037089ff8
    resource: repo://src/manifold/metrics/fid/math.py
  - id: openwiki-source-3a58c59be60433f8b9221b6c
    resource: repo://src/manifold/metrics/fid/reducer.py
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-2472d02c3afe41b7e0d68982
    resource: repo://src/manifold/metrics/paired.py
  - id: openwiki-source-4a976a922e9a1286c789e272
    resource: repo://src/manifold/training/callbacks/paired_fidelity.py
  - id: openwiki-source-cb9c9c415ab8026b88012d9e
    resource: repo://src/manifold/training/cli.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
  - id: openwiki-source-703199deac93d733d94afae7
    resource: repo://src/manifold/training/metrics.py
  - id: openwiki-source-0d7cf619ef7d1ca1ec91473d
    resource: repo://tests/ddp.py
  - id: openwiki-source-5321f561d4730eeeb15c3502
    resource: repo://tests/parity/validate_against_hope.py
  - id: openwiki-source-36db33c03ac89b676295a74d
    resource: repo://tests/test_callback_registry.py
  - id: openwiki-source-32dd8914128afdc039c854cb
    resource: repo://tests/test_ddp_metrics.py
  - id: openwiki-source-3cf412e052ac52d4a8567215
    resource: repo://tests/test_ddp_val_honesty.py
  - id: openwiki-source-54aad16b0d9e880d1e1286dc
    resource: repo://tests/test_device_policy.py
  - id: openwiki-source-358bc7e072fb230f264b7b03
    resource: repo://tests/test_eval_cli.py
  - id: openwiki-source-c07abd7a60c092df1a8401ed
    resource: repo://tests/test_fid_helpers.py
  - id: openwiki-source-40c4b839a934368c7a102446
    resource: repo://tests/test_fid.py
  - id: openwiki-source-9c5f114a4443114e6c506f09
    resource: repo://tests/test_frozen_arm_mixin.py
  - id: openwiki-source-4e3bcc3726ac2935c40ee848
    resource: repo://tests/test_paired_fidelity_callback.py
  - id: openwiki-source-43f6a22439495a06382f037c
    resource: repo://tests/test_paired_fidelity_ddp.py
  - id: openwiki-source-d30a274dc99c74b634e7fd45
    resource: repo://tests/test_paired_reward_deleted.py
generated: { by: "openwiki/0.7.2", at: "2026-10-10T14:21:58.700Z" }
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
| Scheduler / module / pipeline behavior | `tests/test_scheduler.py`, `test_module_training.py`, `test_pipeline_inference.py`, `test_controlnet_module_training.py`, `test_controlnet_pipeline_inference.py` |
| FID, paired fidelity, and metrics rendering | `tests/test_fid.py`, `test_fid_helpers.py`, `test_paired_fidelity.py`, `test_paired_fidelity_callback.py`, `test_metric_plot.py` |
| Distributed validation + paired-monitor DDP | `tests/test_ddp.py`, `test_ddp_detection.py`, `test_ddp_metrics.py`, `test_ddp_val_honesty.py`, `test_controlnet_ddp_monitor.py`, `test_paired_fidelity_ddp.py` |
| Reward and policy learning | `tests/test_reward.py`, `test_reward_pairs.py`, `test_grpo.py`, `test_paired_reward_deleted.py`, `test_controlnet_module_training.py` |
| Callback registry and training spine | `tests/test_callback_registry.py`, plus the registry / monitor / `forbidden_callbacks` assertions in `test_training_cli.py` |
| Frozen-arm contract (register + dual-exclude) | `tests/test_frozen_arm_mixin.py`, `tests/test_controlnet_module_training.py::test_base_is_registered_but_dual_excluded` |
| Per-rank device policy (`DevicePolicy`, post-PG warm) | `tests/test_device_policy.py`, plus `test_ddp_warm.py::test_p1_*` |
| Persistence / export and the `[project.scripts]` surface | `tests/test_persistence.py`, `tests/test_paired_reward_deleted.py` (parses `pyproject.toml` directly), export assertions in training / reward / GRPO tests |
| Before / after eval and reporting | `tests/test_paired_fidelity.py`, `test_before_after_eval.py`, `test_eval_cli.py`, `test_comparison_page.py` |

`tests/ddp.py` is the multi-process helper / harness used by DDP tests. Run the focused distributed tests after changing rank gates, sampler assumptions, reduction code, validation callbacks, trainer device selection, or checkpoint monitors.

## Distributed-validation contract (ADR-0025, supersedes ADR-0016)

All-rank validation, no rank-0 gate. Concretely:

- **Latent-space x0-MAE (`val/x0_mae`).** Every rank computes the cheap reconstruction-MAE on its `DistributedSampler` shard through `LatentX0MAE` (`src/manifold/training/metrics.py`). The callback attaches a `torchmetrics.MeanMetric` to the module and updates it with `weight=batch_size`, so the cross-rank aggregate is the true sample-weighted global mean `Σ(loss·B) / ΣB` — not a mean-of-per-rank-means (a naive `sync_dist=True` would collapse to the latter under a future non-padding sampler). The unit lock at `tests/test_ddp_metrics.py::test_m6_val_x0_mae_is_global_weighted_mean` pins this property end-to-end, with `test_m6_meanmetric_unit_math_is_weighted_not_rank_means` securing it at the math layer.
- **FID.** Synthetic and real examples are rank-strided. Each plane reduces sufficient statistics `(sum_x, sum_xxT, n)` through `SufficientStatsReducer` (`src/manifold/metrics/fid/reducer.py`); `moments_from_sufficient_stats` reconstructs the global moments and `frechet_from_moments` runs the unbiased Fréchet math — **no feature-matrix gather** (the rejected ADR-0016 alternative would have been one). The collective is always three per-plane `all_reduce`s plus an `n` reduce; empty local shards contribute correctly-sized zero statistics, so the collective cannot deadlock on a partial shard. A one-sample local shard is valid and contributes its first / second-order sums; FID is undefined only when the *global* `n < 2`. Synthetic seeds use `seed + i` for `i % world == rank`, so the global synth set is the union across ranks, not `world × rank-0`. `FIDCallback._gated` is cadence-only — no `is_global_zero` gate (regression-locked by `tests/test_ddp_val_honesty.py::test_l1_fid_no_rank0_gate`).
- **GRPO reward (`val/mean_reward`).** `validation_step` runs the rollout + scoring on every rank; per-rank padding-excluded contributions are reduced at epoch end via `torch.distributed.all_reduce(ReduceOp.SUM)`, then logged. `sync_dist=True` is the cross-rank reduce — it is **not** combined with a manual `all_reduce` for the same value. The test gates at `tests/test_ddp_val_honesty.py::test_m3_m4_grpo_validation_step_uses_global_sum_count` source-guard the shape.
- **Supervised ControlNet paired fidelity (`val/psnr`, `val/ssim`, ADR-0037).** Shipped and **active by default** in `manifold-train-controlnet` (`default_names = ["train_loss", "checkpoint", "paired_fidelity"]`). `PairedFidelityCallback` (`src/manifold/metrics/paired_callback.py`) uses a redundant-evaluation design: every rank runs the same fixed paired subset under the same seeded noise on DDP-synchronized weights, so per-rank results are identical and Lightning's torchmetrics sync is the only cross-rank interaction — no FID-style sharding or error-rendezvous machinery needed. The monitor stays observe-only: never a checkpoint `monitor_metric` (that is `val/x0_mae`), never a loss term. See [In-training paired-fidelity monitor](evaluation.md#in-training-paired-fidelity-monitor-adr-0037-shipped-observe-only) for the wired layers and `tests/test_paired_fidelity_ddp.py` for the 2-rank consistency gate.
- **Checkpoint monitors under DDP.** `val/fid`, `val/x0_mae`, and `val/mean_reward` stay on under multi-GPU (the metrics are now global). The `is_multi_gpu` / `not multi_gpu` monitor guards were removed from `cli.py`, `paired_cli.py` (deleted), `grpo_cli.py`, and `paired_grpo_cli.py` (deleted). The GRPO ControlNet path still suppresses **unconditional FID** (a constant frozen-base metric, ADR-0034) — see [Change navigation](workflows.md#change-navigation) — via `forbidden_callbacks={"fid"}` and `forbidden_monitors={"val/fid"}` in `TrainingSpine.run`, but the only metric-emitting monitor that path runs is `val/mean_reward`.

Key implementations live in `src/manifold/metrics/fid/` (the `FIDCallback` and the ADR-0030 composable helpers `VramStage`, `FixedSampleRollout`, `LatentDecoder`, `FeatureExtractor`, `SufficientStatsReducer`), `src/manifold/modules/grpo.py`, `src/manifold/training/metrics.py`, `src/manifold/training/callbacks/`, `src/manifold/training/cli.py`, `src/manifold/training/grpo_cli.py`, and `src/manifold/training/controlnet_cli.py`.

Do not follow the stale comments in `configs/train/config_rflow_jit.yaml` that still describe rank-0-only DDP metrics and an unmonitored fallback — these pre-date ADR-0025 and are not consistent with the current callback / CLI code. The `config_paired_jit.yaml` referenced in older docs no longer exists; the supervised paired translator is `configs/train/config_controlnet_supervised.yaml`. ADR-0025 and the current callback / CLI code are authoritative.

## Before/after evaluation runbook

`manifold-eval` is the only eval console entry. It owns a fresh export of the after `.ckpt`, two clean native reloads, deterministic generation, and the eval artifacts. The implementation and artifact schema are the canonical reference in [Before/after GRPO evaluation](evaluation.md#outputs-and-metric-semantics).

For a quick shipped-surface check, confirm that the installed entry point exposes the contract before starting a full run:

```bash
manifold-eval --help
```

Then use the [workflow examples](workflows.md#beforeafter-evaluation). The current issue-#229 / ADR-0036 argument semantics:

- **`--before-dir` must be a native directory whose `model_index.json` self-describes a `pipeline_class`** (one of the two pipeline names the shipped eval supports, dispatched by `_pipeline_class_of` — there is no `--policy` / `--mode` flag). A missing index, an unknown `pipeline_class`, or a half-written ControlNet export (declares a `controlnet` component without a `controlnet/` subdir) fails fast with a clear pointer before any export work. A bare weights directory is not enough.
- **`--after-ckpt`** is processed through the standard full-state checkpoint→native export bridge (the same one `manifold-export` uses). The command bakes the after weights into a derived native tree at `<output>/after_native`. It does **not** overwrite the before export: the export mutates a probe pipeline in place and the before tree is then reloaded from disk, so the two sides stay independent. It does not return a NIfTI file — output is the populated `<output>` directory.
- **`--before-dir` and `--after-ckpt` together** must point at the same module shape. A raw-arm JiT checkpoint registers under `unet.unet.*`; a supervised ControlNet checkpoint registers only `controlnet.*` (the frozen base is dual-excluded off the checkpoint — see [FrozenArmMixin](frozen-arm-and-device-policy.md#frozenarmmixin-register--dual-exclude), ADR-0031 A1), so the bridge restores the base from the before export rather than baking it twice.
- **`--seed`, `--num-inference-steps`, shape / conditioning, and the VAE / scheduler** are the pairing contract. Hold them fixed when comparing checkpoints; changing them makes before / after noise or trajectory differences ambiguous.
- ControlNet requires a nonempty held-out split (a native `val_data_base_dir` or a positive `--val-fraction`) and a warmed cache under the geometry-suffixed `cache_tag`. Match the cache's target geometry and `vae.scaling_factor`; eval reads the scaling factor from the before export and never re-estimates it. The real-data path also reuses `build_brats_pair_manifest` + `_train_val_manifests` so train / val subjects stay disjoint.
- The report and slice-grid renderer import Matplotlib lazily on the headless `Agg` backend, but a missing or broken rendering dependency fails the run instead of silently dropping visuals. The PNG writes use atomic rename (`<png>.tmp` → `<png>`).
- A JiT run records provenance in `metrics.json` but intentionally has no `before.psnr` or `before.ssim`. Those full-reference fields are valid only for ControlNet (a real target is required); the unconditional comparison uses same-noise grids plus the training FID / reward history.

Focused validation for any metric, driver, CLI, report, or pairing change:

```bash
pytest tests/test_paired_fidelity.py tests/test_paired_fidelity_callback.py \
       tests/test_paired_fidelity_ddp.py tests/test_before_after_eval.py \
       tests/test_eval_cli.py tests/test_comparison_page.py -q
```

Conditional integration check: if a `pyproject.toml` console entry, package barrel, hatch artifact, or installed `manifold-eval` import path changes, build / install the wheel and run the CLI `main(argv)` smoke rather than relying only on the internal tests. This repository has no generated publish mirror or second eval entry point; the hatch package mirror is derived from `pyproject.toml`. The deletion-safety guard `tests/test_paired_reward_deleted.py` parses `[project.scripts]` directly and fails on any reintroduction of the retired `manifold-train-paired` / `manifold-train-paired-reward` entries (ADR-0034).

## Distributed-validation runbook

The all-rank policy reverses a rank-0 workaround for a reproducible 8-DCU / DTK stall during concurrent full-volume MAISI VAE decode (ADR-0025). NVIDIA 8-GPU and single-DCU runs were reported healthy, but ADR-0025 explicitly marks the sugon verification as pending — the VAE `num_splits: 4` / `dim_split: 1` / `save_mem: true` config already active at the time of the deadlock is one of the ranked root-cause candidates, not a confirmed fix.

Before relying on multi-DCU best-by-metric selection, run one validation epoch on all eight ranks with a small subset, following the ADR's probe parameters:

```text
--max-epochs 1 check_val_every_n_epoch=1 val_subset_size=4
```

Profile all ranks through the validation epoch and confirm that every rank exits decode and logs the same global metric.

If the probe hangs, ADR-0025's fallback order is:

1. Serialize GPU decode across ranks while retaining global reduction.
2. Move decode to CPU (`sw_device='cpu'`).

The metric contract stays global; only the decode strategy should change. A return to rank-0-only metrics would again make checkpoint selection shard-biased.

## Diagnosing deadlock vs. slow validation

ADR-0025 includes diagnostic guidance for distinguishing the DCU deadlock from slow validation. The symptom triad "processes `Sl` (sleeping) + log mtime stalled + no tqdm output" is a false positive — it also describes healthy, fully-loaded validation under 8-DDP: Lightning writes no tqdm progress to a redirected (non-tty) log during `validation_step`, and the host main threads sleep while the DCUs compute.

Before diagnosing a deadlock, use load-bearing signals:

- `hy-smi` (after `source /opt/dtk/env.sh`): DCU% near 0 with no progress = stalled (deadlock candidate); DCU% ~100% = computing (slow, not deadlocked).
- `SIGTERM` response: the 2026-07-14 stall ignored `SIGTERM` (required `SIGKILL`); a merely-slow validation terminates on `SIGTERM`.
- (Decisive) `py-spy` on all ranks: identical frozen frame in `sliding_window_inference -> _conv_forward` = deadlock.

The "Sl + log stalled" triad alone is insufficient; do not act on it without confirming one of the above signals.

## DDP failure modes to guard

- Every rank must enter collectives in the same order. FID synchronizes feature-network disablement before entering moment reductions so a rank-local load failure cannot strand other ranks. ADR-0030 adds four error-rendezvous points inside `FIDCallback.on_validation_epoch_end` (stage, disabled-flag, real-preproc, synth): each barrier `all_reduce`s a flag via `ReduceOp.MAX` so all ranks take the same abort branch together instead of one rank skipping a collective while peers block in it. Collective-count invariance — same collectives, same order, every path — is the new testable property; `tests/test_fid_helpers.py::test_on_validation_epoch_end_has_error_rendezvous` asserts at-least-four rendezvous points.
- Empty FID shards must not be sent through MAISI decode; they contribute correctly sized zero sufficient statistics. The reducer guards on `numel() == 0`, not `shape[0] < 2` (a one-sample shard is valid — `tests/test_fid_helpers.py::test_reducer_n1_shard_contributes_stats` pins the distinction).
- A one-sample local shard contributes its first / second-order sums; covariance validity is checked only after the global `all_reduce` of `n`.
- Do not combine manual `all_reduce` with `sync_dist=True` for the same value (it gets reduced twice).
- Preserve rank-strided FID seeds (`seed + global_index`) so the distributed sample union matches the requested global sample count rather than multiplying it by world size.
- `FixedSampleMaterializer` + decoded-image cache was deliberately deferred (ADR-0030) — building it now would reintroduce an inter-callback channel + ordering dependency for the only current generative metric that exists.
- The supervised ControlNet paired-fidelity monitor is **redundant per rank**, not sharded: identical fixed subset + identical seeded noise + DDP-synchronized weights ⇒ identical per-rank values, so any drift indicates a divergence in the rolled weights (rather than a real metric change). Cross-rank reduction is plain `MeanMetric` sync.

These cases are covered principally by `tests/test_fid.py`, `test_fid_helpers.py`, `test_ddp_metrics.py`, `test_ddp_val_honesty.py`, `test_ddp.py`, `test_paired_fidelity_ddp.py`, and `test_controlnet_ddp_monitor.py`.

## Validation and checkpoint cautions

- Noise-to-data production validation is disabled unless a held-out source is wired; the code refuses train-as-validation leakage. In that case checkpointing falls back to periodic / last rather than monitored FID.
- ControlNet-supervised validation (`manifold-train-controlnet`) should use a nonzero subject-level `val_fraction` (the recipe default is `0.2`); `0` permits a train-as-validation fallback and is not an honest generalization estimate. The shared splitter lives in `src/manifold/data/paired_manifests.py` and supports a native-split directory when `env.val_data_base_dir` is a real BraTS directory.
- FID callbacks and `BeforeAfterEval` decode in float32 with MAISI `norm_float16` disabled (the `LatentDecoder` helper sets `norm_float16 = False` on every VAE module the first time it runs, then caches the skip). Before / after evaluation then applies the same per-volume `min_max_to_unit` convention as the ControlNet pipeline and scores PSNR / SSIM with `data_range=1.0`.
- `val/psnr` / `val/ssim` (ADR-0037) are observed in-training on the supervised ControlNet stage but are **never** a checkpoint monitor and never a loss term. Checkpoint selection stays on `val/x0_mae`. The fixed paired subset reuses the same `min_max_to_unit` → `PairedFidelityMetrics(data_range=1.0)` ordering as the offline `manifold-eval`, so the in-training curve and the before / after number are directly comparable — see [In-training paired-fidelity monitor](evaluation.md#in-training-paired-fidelity-monitor-adr-0037-shipped-observe-only).
- Current metrics and native exports use raw optimizer weights (EMA training was removed in commit `e89b05d`; `val/fid_avg` / `val/fid_raw`, `ema_decays`, and the `--ema` / `--prefer-ema` export options no longer exist). Remove references to EMA arms from automation and dashboards.
- Export uses full-state deserialization (`weights_only=False` for Lightning `.ckpt` files); only process checkpoints produced by a trusted run.

## Diagnostics

The `scripts/` directory was eliminated in ADR-0033; the helper scripts that used to live there (`scripts/eval_paired_step_sweep.py`, `scripts/diag_brain_mask_psnr.py`, etc.) are not part of this tree. The retained investigation tool is `tests/parity/validate_against_hope.py`, a sampler-parity probe kept as a `<1e-3` proof (ADR-0005) that the modules-side sampler and the inference pipeline roll out the same trajectory. Read its arguments and assumptions before using it against a new dataset or checkpoint.
