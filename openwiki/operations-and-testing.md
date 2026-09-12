---
type: Guide
title: Operations and Testing
description: Setup, validation behavior, distributed metrics, runbook cautions, deadlock vs slow-validation diagnostic, and focused test commands for Manifold.
tags: [operations, testing, distributed, validation, ddp]
verified:
  - by: openwiki/0.5.1
    at: 2026-09-12T12:03:27.299Z
sources:
  - id: openwiki-source-5ec900d9e277a0cf8e19af49
    resource: repo://docs/adr/0025-all-ddp-validation-no-rank0-gate.md
  - id: openwiki-source-703199deac93d733d94afae7
    resource: repo://src/manifold/training/metrics.py
generated: { by: "openwiki/0.5.1", at: "2026-09-12T12:03:27.299Z" }
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
| Distributed validation | `tests/test_ddp.py`, `test_ddp_detection.py`, `test_ddp_metrics.py`, `test_ddp_val_honesty.py`, `test_controlnet_ddp_monitor.py`, `test_paired_fidelity_ddp.py` |
| Reward and policy learning | `tests/test_reward.py`, `test_reward_pairs.py`, `test_grpo.py`, `test_paired_reward_deleted.py`, `test_controlnet_module_training.py` |
| Callback registry and training spine | `tests/test_callback_registry.py`, `test_paired_fidelity_callback.py`, plus the registry / monitor / `forbidden_callbacks` assertions in `test_training_cli.py` |
| Frozen-arm contract (register + dual-exclude) | `tests/test_frozen_arm_mixin.py`, `tests/test_controlnet_module_training.py::test_base_is_registered_but_dual_excluded` |
| Per-rank device policy (`DevicePolicy`, post-PG warm) | `tests/test_device_policy.py`, plus `test_ddp_warm.py::test_p1_*` |
| Persistence/export | `tests/test_persistence.py`, export assertions in training/reward/GRPO tests |
| Before/after eval and reporting | `tests/test_paired_fidelity.py`, `test_paired_fidelity_callback.py`, `test_before_after_eval.py`, `test_eval_cli.py`, `test_comparison_page.py` |

`tests/ddp.py` is the multi-process helper/harness used by DDP tests (CPU 2-rank gloo, `run_ddp_two_rank`, the `*_ddp_worker` fixtures). Run the focused distributed tests after changing rank gates, sampler assumptions, reduction code, validation callbacks, trainer device selection, or checkpoint monitors.

## Distributed validation contract

ADR-0025 supersedes ADR-0016. Current behavior is:

- **Latent-space x0-MAE (`val/x0_mae`):** every rank computes the cheap reconstruction-MAE on its `DistributedSampler` shard through `LatentX0MAE` (`src/manifold/training/metrics.py`), which attaches a `torchmetrics.MeanMetric` to the module (sample-weighted, `weight=batch_size`) so Lightning's DDP reduction produces the true global mean — not a mean-of-per-rank-means.
- **FID:** synthetic and real examples are rank-strided. Each plane reduces sufficient statistics `(sum_x, sum_xxT, n)`, reconstructs global moments, and computes unbiased FID without gathering feature matrices. Empty local shards contribute zero statistics; only the global count must be at least two.
- **GRPO reward:** every rank validates and logs `val/mean_reward` with `sync_dist=True`.
- **Paired fidelity (supervised ControlNet):** ADR-0037 ships `PairedFidelityCallback` (`src/manifold/metrics/paired_callback.py`) as the supervised ControlNet stage's default callback (registered as `PairedFidelitySpec`). It rolls a fixed paired subset through `controlnet_rollout` + frozen-VAE decode + `min_max_to_unit` and scores with `PairedFidelityMetrics(data_range=1.0)`; all ranks redundantly evaluate the same fixed subset (DDP-synchronized weights + identical seeded noise ⇒ identical result), so the only cross-rank reduction is Lightning's `torchmetrics` sync. `val/psnr` / `val/ssim` are observe-only: never a checkpoint monitor (that stays `val/x0_mae`), never a loss term. The ControlNet-GRPO path's blanket `forbidden_callbacks={"fid": …}` deliberately still excludes paired fidelity until the explicit follow-up lands — see [Before/after GRPO evaluation](evaluation.md#in-training-monitor-supervised-stage-shipped-controlnet-grpo-is-the-follow-up).
- **Checkpoint monitors:** configured `val/fid`, `val/x0_mae`, and `val/mean_reward` monitors remain global under DDP. `ModelCheckpoint` monitors `val/x0_mae` on the supervised ControlNet path (the active checkpoint selector), `val/fid` on the JiT path, and `val/mean_reward` on the GRPO paths — all global under DDP.

Key implementations are `src/manifold/metrics/fid/`, `src/manifold/metrics/paired_callback.py`, `src/manifold/modules/grpo.py`, and the training callback/CLI paths in `src/manifold/training/`.

Do not follow the stale checkpoint comments in `configs/train/config_rflow_jit.yaml` that still describe rank-0-only DDP metrics and unmonitored fallback (e.g. "under DDP (FID is rank-0-only) manifold-train falls back to save_last + every_n_epochs with no monitor"). The supervised paired translator is `configs/train/config_controlnet_supervised.yaml`. ADR-0025, ADR-0030, ADR-0037, and the current callback/CLI code are authoritative.

## Before/after evaluation runbook

`manifold-eval` is the only eval console entry. It owns a fresh export of the after `.ckpt`, two clean native reloads, deterministic generation, and the eval artifacts. The implementation and artifact schema are the canonical reference in [Before/after GRPO evaluation](evaluation.md#outputs-and-metric-semantics).

For a quick shipped-surface check, confirm that the installed entry point exposes the contract before starting a full run:

```bash
manifold-eval --help
```

Then use the [workflow examples](workflows.md#beforeafter-evaluation). The key runbook constraints are:

- `--before-dir` must be a native directory with a self-describing `model_index.json` (`_pipeline_class_of` rejects missing or unknown `pipeline_class` with `FileNotFoundError` / `ValueError` before export). Never pass an arbitrary directory merely because it contains weights — the eval infers JiT vs. ControlNet from this artifact's `pipeline_class` and there is no mode flag (ADR-0034).
- `--after-ckpt` is processed through the standard full-state export bridge (`manifold.training.export.export_to_native`). The CLI loads a probe pipeline from the before export, runs `export_to_native(args.after_ckpt, <output>/after_native, ...)`, then reloads both sides as fresh pipelines (the probe's arms were overwritten in place). A ControlNet after export reuses the before export's frozen base; only the `controlnet.*` weights are baked.
- `--seed`, `--num-inference-steps`, shape/conditioning, and the VAE/scheduler are the pairing contract. Hold them fixed when comparing checkpoints; changing them makes before/after noise or trajectory differences ambiguous. The JiT path gives each side a fresh generator seeded identically; the ControlNet path draws one noise tensor and passes it to both pipelines.
- ControlNet requires a held-out split (`--val-data-base-dir` as a native BraTS directory or `--val-fraction > 0`) and a warmed paired latent cache under the geometry-suffixed cache tag (`paired_cache_tag(<tag>, target_dim, divisor)`). The eval reads `vae.scaling_factor` from the before export verbatim and never re-estimates it, never encodes — a cache miss fails fast.
- The report and slice-grid renderer imports Matplotlib lazily on the headless `Agg` backend (`comparison_page.py`, `slice_grid.py`), but a missing or broken rendering dependency fails the run instead of silently dropping visuals.
- A JiT run records only `policy` / `seed` / `num_inference_steps` / `num_samples` / `grids` in `metrics.json` (provenance). The paired `before.psnr` / `before.ssim` (and `after.psnr` / `after.ssim`) fields are valid only for the ControlNet path — they are computed by `PairedFidelityMetrics` on `min_max_to_unit`-normalized `[0, 1]` volumes with `data_range=1.0`. The eval never returns a NIfTI file; it writes JSON + PNGs under `--output`.

Focused validation for any metric, driver, CLI, or report change:

```bash
pytest tests/test_paired_fidelity.py tests/test_paired_fidelity_callback.py tests/test_paired_fidelity_ddp.py tests/test_before_after_eval.py tests/test_eval_cli.py tests/test_comparison_page.py -q
```

Conditional integration check: if a `pyproject.toml` console entry, package barrel, hatch artifact, or installed `manifold-eval` import path changes, build/install the wheel and run the CLI `main(argv)` smoke rather than relying only on the internal tests. This repository has no generated publish mirror or second eval entry point; the hatch package mirror is derived from `pyproject.toml`.

## Distributed validation runbook

The all-rank policy reverses a rank-0 workaround for a reproducible 8-DCU/DTK stall during concurrent full-volume MAISI VAE decode (py-spy: all ranks frozen in `sliding_window_inference -> _conv_forward`). NVIDIA 8-GPU and single-DCU runs were reported healthy, but ADR-0025 explicitly marks the sugon verification as pending.

Before relying on multi-DCU best-by-metric selection, run one validation epoch on all eight ranks with a small subset, following the ADR's probe parameters:

```text
--max-epochs 1 check_val_every_n_epoch=1 val_subset_size=4
```

Profile all ranks through the validation epoch and confirm that every rank exits decode and logs the same global metric. The shipped VAE network config (`configs/network/config_network.yaml`) sets `num_splits: 4`, `dim_split: 1`, `save_mem: true`, and `norm_float16: true` for the autoencoder block; the first three block-split the decoder convs via `MaisiConvolution.forward`, matching the UNet block which runs healthy under 8-DDP training. `LatentDecoder` (`src/manifold/metrics/fid/decoder.py`) additionally disables MAISI `norm_float16` once on first call so the float32 decode path stays type-consistent. **None of these knobs is a proven fix for the DCU stall** — that config was already active when the 2026-07-14 deadlock was diagnosed, and the `non_blocking` H2D + per-block `gc.collect` + `_empty_cuda_cache` relief path it activates is itself one of the ranked root-cause candidates. Only the sugon probe (never run) resolves it.

If the probe hangs, ADR-0025's fallback order is:

1. Serialize GPU decode across ranks while retaining global reduction.
2. Move decode to CPU (`sw_device='cpu'`, the knob the encode path already uses).

The metric contract stays global; only the decode strategy should change. A return to rank-0-only metrics would again make checkpoint selection shard-biased.

## Diagnosing deadlock vs. slow validation

ADR-0025 includes diagnostic guidance for distinguishing the DCU deadlock from slow validation. The symptom triad "processes `Sl` (sleeping) + log mtime stalled + no tqdm output" is a false positive — it also describes healthy, fully-loaded validation under 8-DDP (Lightning writes no tqdm progress to a redirected (non-tty) log during `validation_step`, and the host main threads sleep while the DCUs compute). On 2026-07-18 an epoch-0 Paired JiT validation on sugon 8-DCU was mis-diagnosed as this deadlock from those symptoms alone; `hy-smi` then showed all 8 DCUs at 100% utilization — the validation was simply slow, not deadlocked.

Before diagnosing a deadlock, use load-bearing signals:

- `hy-smi` (`/opt/hyhal/bin/hy-smi`, after `source /opt/dtk/env.sh`): DCU% near 0 with no progress = stalled (deadlock candidate); DCU% ~100% = computing (slow, not deadlocked).
- `SIGTERM` response: the 2026-07-14 stall ignored `SIGTERM` (required `SIGKILL`); a merely-slow validation terminates on `SIGTERM`.
- `py-spy` on all ranks: identical frozen frame in `sliding_window_inference -> _conv_forward` = deadlock.

The "Sl + log stalled" triad alone is insufficient; do not act on it (kill / restart / disable validation) without confirming one of the above signals.

## DDP failure modes to guard

The FID pipeline stages compose under ADR-0030's **collective-count invariant**: every rank enters the same collectives (the four error-flag rendezvous + the per-plane sufficient-stats `all_reduce`) in the same order in every code path. When you add a reduction or a new fallible phase, mirror the existing pattern. The invariants to preserve:

- **Rank-symmetry in collective entry.** Every rank must enter collectives in the same order. FID synchronizes feature-network disablement before entering moment reductions so a rank-local load failure cannot strand other ranks. Any new fallible phase needs an error-flag `all_reduce(MAX)` *before* the reduction-bearing phase, not after.
- **Empty local shards are valid.** A rank with `num_synth < world_size` produces zero synth latents (see `FixedSampleRollout`: `for i in range(rank, num_synth, world)`), and a rank with `real_latents[rank::world]` of size 0 must not be sent through MAISI decode. Empty shards contribute correctly sized zero sufficient statistics (`[D]` / `[D, D]` / `n=0`), and the collective cannot deadlock because every rank enters the same `all_reduce`.
- **One-sample local shards are valid.** A single-sample local shard yields first/second-order sums; covariance validity is checked only after global reduction (`moments_from_sufficient_stats` requires `n ≥ 2`). Below that threshold the reducer returns `None` for that plane, and `frechet_from_moments` skips it.
- **No double-reduce.** Do not combine manual `all_reduce` with `sync_dist=True` for the same value. The training-metrics callbacks (`train/loss_epoch`, `val/x0_mae`) use a `torchmetrics.MeanMetric` attached to the module with `weight=batch_size`; the Metric reduces `sum(loss·B)/sum(B)` across ranks — true sample-weighted global mean — and Lightning's `sync` fires on `module.log("name", metric)`. Adding `sync_dist=True` on top would double-reduce.
- **Rank-strided seeds.** `FixedSampleRollout` seeds each generated latent as `seed + i` for `i in range(rank, num_synth, world)`. Preserving this is what makes the distributed synthetic set equal the union across ranks rather than `world × rank-0`. Any new generative metric that does its own sampling must mirror it.

These cases are covered principally by `tests/test_fid.py`, `tests/test_fid_helpers.py`, `tests/ddp.py`, `tests/test_ddp_metrics.py`, `tests/test_ddp_val_honesty.py`, and `tests/test_paired_fidelity_ddp.py` (the redundant-subset paired-fidelity gate, ADR-0037).

## Validation and checkpoint cautions

- **Noise-to-data production validation refuses train-as-validation leakage.** `build_datamodule` raises if `val_dataset is None` and `allow_train_as_val=False` is not opted into (`src/manifold/data/datamodule.py`); `LatentWarmDataModule` mirrors the guard. The warm path only enables the smoke-only train-as-val prefix when explicitly opted into. In those smoke paths checkpointing falls back to `save_last` + `every_n_epochs` rather than a monitor-driven `val/fid` selection.
- **ControlNet-supervised validation (`manifold-train-controlnet`) should use a nonzero subject-level `val_fraction`.** The recipe default in `configs/train/config_controlnet_supervised.yaml` is `0.2` (and the supervised `environment_brats2023.yaml` / `environment_sugon.yaml` overrides point `val_data_base_dir` at a real BraTS directory). A `val_fraction` of `0` permits a train-as-validation fallback and is not an honest generalization estimate. The shared splitter lives in `src/manifold/data/paired_manifests.py` (`_train_val_manifests`) and supports a native-split directory when `env.val_data_base_dir` is a real BraTS directory; that path takes priority over the fraction.
- **Float32 decode with `norm_float16` disabled.** Both `FIDCallback` (via `LatentDecoder`) and `BeforeAfterEval` decode in float32 with MAISI `norm_float16` disabled; the `ControlNetLatentFlowPipeline` decode path mirrors the same idempotent disable. Before/after evaluation applies the per-volume `min_max_to_unit` convention as the ControlNet pipeline and scores PSNR/SSIM with `data_range=1.0` (`PairedFidelityMetrics`).
- **Raw UNet only (no EMA).** Current metrics and native exports use raw optimizer weights; EMA training was removed in ADR-0006 / ADR-0021 (commit `e89b05d`). The shipped `val/fid` monitor and the export-baked UNet are deliberately aligned so the exported "best" is best for the weights that are published. Remove references to EMA arms (`DoubleEMACallback`, slow EMA, etc.) from automation, dashboards, and evaluation tooling.
- **Trusted-checkpoint-only export.** `export_to_native` calls `torch.load(ckpt_path, map_location="cpu", weights_only=False)` — full-state deserialization. Only point it at training checkpoints your own run produced; an attacker-controlled `.ckpt` is pickle-RCE surface. The ControlNet branch additionally requires a registered `controlnet.*` key set; otherwise it raises.
- **`find_unused_parameters=True` is load-bearing.** The supervised ControlNet path depends on it (`src/manifold/training/trainer.py`) because the `_PinnedClassEmbedding` shim can deliver no gradient for a dropped label row. Any DDP-related edit or warm refactor must keep this flag (the `docs/prd/ddp-correctness.md` Risks section calls it out explicitly).

## Diagnostics

The `scripts/` directory was eliminated in ADR-0033; the helper scripts that used to live there (`scripts/eval_paired_step_sweep.py`, `scripts/diag_brain_mask_psnr.py`, etc.) are not part of this tree. The deleted diagnostics died because EMA training was removed (they read `callbacks['DoubleEMACallback']` from new `.ckpt`s that no longer carry it). The retained investigation tool is `tests/parity/validate_against_hope.py`, a sampler-parity probe kept as a `<1e-3` proof (ADR-0005) that the modules-side sampler and the inference pipeline roll out the same trajectory against a real hope MONAI-bundle reference. Read its arguments and assumptions (hope source on `PYTHONPATH`, confirmed JiT/x0 checkpoint + double-EMA `0.9999/0.9996` weights, fp16 CFG-batching residual) before using it against a new dataset or checkpoint.
