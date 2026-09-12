---
type: Guide
title: Quickstart
description: Routing entry for the Manifold wiki; what the wiki covers, how it is organized, and where to go next for each change area.
tags: [quickstart, navigation, overview]
verified:
  - by: openwiki/0.5.1
    at: 2026-09-12T12:03:27.299Z
sources:
  - id: openwiki-source-1f077c4de341ab7dab070abb
    resource: repo://docs/adr/0037-controlnet-fidelity-in-training-monitor.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
  - id: openwiki-source-09e4eceb7e21a9e44d0eeb8d
    resource: repo://src/manifold/eval/__init__.py
  - id: openwiki-source-e22b4e5926edbd63eb845214
    resource: repo://src/manifold/eval/cli.py
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-2472d02c3afe41b7e0d68982
    resource: repo://src/manifold/metrics/paired.py
  - id: openwiki-source-4a976a922e9a1286c789e272
    resource: repo://src/manifold/training/callbacks/paired_fidelity.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
generated: { by: "openwiki/0.5.1", at: "2026-09-12T12:03:27.299Z" }
---

# Quickstart

Manifold is a 3D medical-imaging research codebase built on top of
[stable-pretraining](https://github.com/galilai-group/stable-pretraining) and
[MONAI](https://monai.io/). It follows the [diffusers](https://github.com/huggingface/diffusers)
four-component layout (models / schedulers / training module / pipeline) and
trains noise→data rectiflow generators (JiT), a supervised ControlNet translator
on a frozen JiT base UNet, a mode-agnostic realism `RewardModel`, and a
group-relative policy-optimization (GRPO) stage that can post-train either the
UNet or a supervised ControlNet policy against the *same* shared reward
(ADR-0034). The shipped evaluation stack provides deterministic
same-noise before/after comparisons and paired 3D PSNR/SSIM reporting; the
supervised ControlNet stage additionally runs an **observe-only** in-training
`val/psnr` / `val/ssim` monitor (`PairedFidelitySpec`, ADR-0037). The wiki is
the evidence index; source code and tests remain authoritative.

## Layout of the wiki

- [Architecture and source map](architecture.md) — component boundaries, data/config layers, runtime flows, native artifact contract, and change guidance.
- [Key workflows](workflows.md) — JiT training, supervised ControlNet, reward + GRPO stages, checkpoint → native export, inference, eval, and the GRPO scheduler `rollout_range` contract.
- [Before/after GRPO evaluation](evaluation.md) — `manifold-eval`, same-noise artifact comparison, paired PSNR/SSIM, slice grids, report artifacts, and the shipped supervised-stage in-training fidelity monitor (ControlNet-GRPO extension is the deliberate ADR-0037 follow-up).
- [Operations and testing](operations-and-testing.md) — standard checks, distributed-validation contract, runbook cautions, deadlock vs slow-validation diagnostic, and focused test commands.
- [Callback registry and training spine](callback-registry.md) — `CallbackRegistry` two-phase resolve/build, the spec contract, monitor validation, and `TrainingSpine.run` merge order (ADR-0029 / ADR-0032).
- [Frozen arms and per-rank device policy](frozen-arm-and-device-policy.md) — `FrozenArmMixin` (register + dual-exclude, ADR-0031 A1) and `DevicePolicy` (the per-rank CUDA decision that replaced `resolve_warm_device` and the pre-PG `set_device` twin, ADR-0035).

## Entrypoints at a glance

Declared in `pyproject.toml` under `[project.scripts]` (the source of truth for
console entry points; the deletion guards in
`tests/test_paired_reward_deleted.py` read it directly):

```text
manifold-train            # noise→data JiT training (ADR-0017 warm)
manifold-train-reward     # mode-agnostic PatchGAN realism reward (ADR-0009/0010)
manifold-train-grpo       # UNet *or* ControlNet policy GRPO — inferred from --native-dir
manifold-train-controlnet # supervised ControlNet translator on a frozen JiT base (ADR-0027)
manifold-export           # checkpoint -> native inference dir (--pipeline {jit,controlnet})
manifold-eval             # same-noise before/after GRPO eval -> metrics.json + slice grids
```

The historical `manifold-train-paired` / `manifold-train-paired-reward` entries
are gone; ADR-0034 deleted the condition-aware paired reward pipeline. The
`--pipeline paired` flag in `manifold-export` references a now-removed module
(`manifold.pipelines.paired_latent_flow`) and is not part of the current
workflow — use `--pipeline jit` (default) or `--pipeline controlnet`.

## Task routing

| Change area / intent | Wiki page | Entry points | Important symbols / types | Focused tests | Minimal validation |
|---|---|---|---|---|---|
| Edit transport, integration, scheduler grid | [architecture.md](architecture.md#runtime-flows), [workflows.md](workflows.md#inference) | `src/manifold/schedulers/scheduling_flow_match_heun.py`, `src/manifold/modules/sampler.py`, `src/manifold/pipelines/latent_flow.py` | `FlowMatchHeunDiscreteScheduler`, `PartialFlowMatchHeunScheduler`, `sample_latent_flow`, `FlowMatchGRPOScheduler` | `tests/test_scheduler.py`, `test_pipeline_inference.py`, `test_module_training.py` | `pytest tests/test_scheduler.py tests/test_pipeline_inference.py -q` |
| Edit JiT training loop, latent cache, scaling | [architecture.md](architecture.md#runtime-flows), [workflows.md](workflows.md#jit-training) | `src/manifold/training/cli.py`, `src/manifold/data/{latent_dataset,latent_pipeline,warm_datamodule}.py`, `src/manifold/modules/latent_flow.py` | `LatentFlowModule`, `LatentDataset`, `estimate_scale_factor`, `manifold-train` | `tests/test_training_cli.py`, `test_training.py`, `test_data.py`, `test_ddp_warm.py` | `pytest tests/test_training_cli.py tests/test_ddp_warm.py -q` |
| Edit ControlNet conditioning or supervised training | [architecture.md](architecture.md#runtime-flows), [workflows.md](workflows.md#paired-training) | `src/manifold/models/controlnet_3d.py`, `src/manifold/modules/{controlnet_latent_flow,controlnet_sampler}.py`, `src/manifold/data/{paired_brats,paired_latent_dataset,paired_manifests,paired_volume_dataset}.py`, `src/manifold/training/controlnet_cli.py` | `ControlNet3DConditionModel`, `ControlNetLatentFlowModule`, `controlnet_rollout`, `_train_val_manifests`, `manifold-train-controlnet` | `tests/test_controlnet_cli.py`, `test_controlnet_module_training.py`, `test_controlnet_pipeline_inference.py`, `test_paired_manifests.py`, `test_paired_latent_cache.py` | `pytest tests/test_controlnet_cli.py tests/test_controlnet_pipeline_inference.py -q` |
| Edit reward training (preference pairs, online rollout) | [workflows.md](workflows.md#reward-and-grpo-stages) | `src/manifold/modules/reward.py`, `src/manifold/modules/partial_denoise.py`, `src/manifold/data/reward_pairs.py`, `src/manifold/training/reward_cli.py` | `RewardModule`, `partial_denoise_rollout`, `bradley_terry_loss`, `PartialFlowMatchHeunScheduler` | `tests/test_reward.py`, `test_reward_pairs.py`, `test_paired_reward_deleted.py` | `pytest tests/test_reward.py tests/test_reward_pairs.py -q` |
| Edit GRPO (UNet or ControlNet policy) | [workflows.md](workflows.md#reward-and-grpo-stages), [workflows.md](workflows.md#grpo-scheduler-rollout_range-single-heun-interval-loop-adr-0005), [architecture.md](architecture.md#reward-and-policy-post-training) | `src/manifold/modules/grpo.py`, `src/manifold/schedulers/scheduling_flow_match_grpo.py`, `src/manifold/training/grpo_cli.py`, `src/manifold/training/controlnet_inputs.py` | `GRPOModule`, `singular_branch_rollout`, `clipped_surrogate_loss`, `FlowMatchGRPOScheduler.rollout_range`, `_detect_controlnet_export`, `_controlnet_real_inputs`, `_unet_real_inputs`, `load_frozen_controlnet_generator`, `manifold-train-grpo` | `tests/test_grpo.py`, `test_scheduler.py::test_grpo_*`, `test_paired_reward_deleted.py` | `pytest tests/test_grpo.py tests/test_scheduler.py -q` |
| Edit checkpoint / export contract | [workflows.md](workflows.md#checkpoint-and-export-contract) | `src/manifold/training/{export_cli,export}.py`, `src/manifold/pipelines/{latent_flow,controlnet_latent_flow}.py`, `configs/network/config_network.yaml` | `export_to_native`, `LatentFlowPipeline`, `ControlNetLatentFlowPipeline`, `manifold-export` | `tests/test_persistence.py`, `test_pipeline_inference.py`, `test_controlnet_pipeline_inference.py`, `test_config.py` | `pytest tests/test_persistence.py tests/test_controlnet_pipeline_inference.py -q` |
| Add / change offline before/after evaluation, paired fidelity scoring, or reporting | [evaluation.md](evaluation.md), [workflows.md](workflows.md#beforeafter-evaluation), [operations-and-testing.md](operations-and-testing.md#beforeafter-evaluation-runbook) | `src/manifold/eval/{cli,before_after,comparison_page,slice_grid}.py`, `src/manifold/metrics/paired.py`, `src/manifold/pipelines/pipeline_utils.py` | `BeforeAfterEval`, `BeforeAfterResult`, `PairedFidelityMetrics`, `ComparisonPageBuilder`, `min_max_to_unit` | `tests/test_paired_fidelity.py`, `test_before_after_eval.py`, `test_eval_cli.py`, `test_comparison_page.py` | `pytest tests/test_paired_fidelity.py tests/test_before_after_eval.py tests/test_eval_cli.py tests/test_comparison_page.py -q` |
| Edit validation / FID / x0-MAE / reward / paired-fidelity metric | [operations-and-testing.md](operations-and-testing.md#distributed-validation-contract), [callback-registry.md](callback-registry.md) | `src/manifold/metrics/fid/*`, `src/manifold/metrics/{paired,paired_callback}.py`, `src/manifold/training/metrics.py`, `src/manifold/training/callbacks/{fid,paired_fidelity,registry}.py` | `FIDCallback`, `LatentX0MAE`, `FIDSpec`, `PairedFidelityCallback`, `PairedFidelitySpec`, `PairedFidelityMetrics`, `CallbackRegistry`, `CallbackContext` | `tests/test_fid.py`, `test_fid_helpers.py`, `test_paired_fidelity.py`, `test_paired_fidelity_callback.py`, `test_paired_fidelity_ddp.py`, `test_metric_plot.py`, `test_ddp_metrics.py`, `test_callback_registry.py`, `test_ddp_val_honesty.py`, `test_controlnet_ddp_monitor.py` | `pytest tests/test_fid.py tests/test_ddp_metrics.py tests/test_paired_fidelity_callback.py -q` |
| Add / change a training callback | [callback-registry.md](callback-registry.md) | `src/manifold/training/callbacks/{registry,context,train_loss,fid,checkpoint,paired_fidelity}.py`, `src/manifold/training/core.py` | `CallbackRegistry`, `CallbackSpec`, `TrainingSpine.run`, `forbidden_callbacks`, `forbidden_monitors` | `tests/test_callback_registry.py`, `tests/test_training_cli.py`, `tests/test_paired_fidelity_callback.py` | `pytest tests/test_callback_registry.py -q` |
| Add / change a frozen arm or the per-rank device decision | [frozen-arm-and-device-policy.md](frozen-arm-and-device-policy.md) | `src/manifold/modules/frozen_arm.py`, `src/manifold/training/device_policy.py`, `src/manifold/training/{cli,grpo_cli,reward_cli,controlnet_cli}.py`, `src/manifold/modules/{grpo,controlnet_latent_flow,reward}.py` | `FrozenArmMixin._register_frozen_arm`, `FrozenArmMixin._frozen_arm_names`, `DevicePolicy.pin`, `DevicePolicy.warm_device` | `tests/test_frozen_arm_mixin.py`, `tests/test_device_policy.py`, `tests/test_ddp_warm.py` | `pytest tests/test_frozen_arm_mixin.py tests/test_device_policy.py -q` |
| Diagnose a hung / slow multi-DCU validation epoch | [operations-and-testing.md](operations-and-testing.md#diagnosing-deadlock-vs-slow-validation) | `src/manifold/training/cli.py`, `src/manifold/metrics/fid/*` | `FIDCallback`, `LatentX0MAE`, MAISI `sliding_window_inference` | `tests/ddp.py` helper, `tests/test_ddp*.py` | `pytest tests/test_ddp.py -q` |
| Add / remove a console entry point | [workflows.md](workflows.md#reward-and-grpo-stages) | `pyproject.toml` (`[project.scripts]`), `src/manifold/eval/cli.py` | `manifold-eval` and `main` | `tests/test_paired_reward_deleted.py` (parses `pyproject.toml` directly), `tests/test_eval_cli.py` (ships the full CLI workflow) | `pytest tests/test_paired_reward_deleted.py tests/test_eval_cli.py -q` |

## High-level repository layout

```text
manifold/
├── pyproject.toml              # console scripts + hatch build config (single source of truth)
├── README.md
├── AGENTS.md / CLAUDE.md       # agent skill pointers (do not hand-edit OpenWiki section)
├── CONTEXT.md                  # domain vocabulary — use its terms
├── configs/
│   ├── env/                    # environment profiles (brats2023, euler, sugon)
│   ├── network/config_network.yaml
│   └── train/                  # rflow_jit / controlnet_supervised / reward / grpo recipes
├── docs/adr/                   # 37 ADRs; ADR-0037 is the latest accepted (in-training paired-fidelity monitor)
├── src/manifold/
│   ├── config/                 # OmegaConf loader + component builder
│   ├── models/                 # diffusers-style network wrappers (UNet, VAE, ControlNet, Reward)
│   ├── schedulers/             # rectified-flow transport + Heun/Partial/GRPO schedulers
│   ├── modules/                # stable-pretraining training Modules (JiT, ControlNet, Reward, GRPO, sampler)
│   ├── pipelines/              # native inference: LatentFlowPipeline, ControlNetLatentFlowPipeline, pipeline_utils
│   ├── data/                   # volumes, manifests, latent cache, warm datamodule, BraTS pairing
│   ├── metrics/                # FID + paired-fidelity metric + paired-fidelity in-training callback + plot
│   ├── training/               # CLIs, callbacks, TrainingSpine, FrozenArmMixin / DevicePolicy callers
│   └── eval/                   # manifold-eval: BeforeAfterEval driver, slice grids, comparison page
└── tests/                      # focus tests + tests/ddp.py multi-process harness + tests/parity/
```

## Change-agent checklist

- **Transport/integration:** change the scheduler and the shared sampler path together; run scheduler, pipeline, and module tests to prevent train/inference drift.
- **Latent scaling:** preserve VAE ownership and the unscaled-cache contract; check VAE, data, persistence, and pipeline tests.
- **Paired conditioning / pairing:** keep BraTS discovery outside the generic dataset contract and preserve subject-level split isolation (`_train_val_manifests` is shared by both CLIs after ADR-0022 / ADR-0034).
- **Reward / GRPO policy:** the discriminator is the native artifact under `--native-dir` (a ControlNet export exposes a `controlnet` component in `model_index.json`); there is **no** `--grpo-mode` flag. Both policies score `z_K` unconditionally with the same `RewardModel` (ADR-0034). The ControlNet policy path drops FID and rejects `val/fid` as a checkpoint monitor via `TrainingSpine.forbidden_callbacks` / `forbidden_monitors` (ADR-0032); checkpoint selection uses `val/mean_reward`.
- **GRPO scheduler:** the anchor / suffix Heun loop lives on `FlowMatchGRPOScheduler.rollout_range` — a scheduler method, not a module-level function. The partial and GRPO schedulers inherit `FlowMatchHeunDiscreteScheduler.heun_rollout` verbatim; do not reintroduce `_heun_one_step` / `_heun_rollout` free functions in `src/manifold/modules/grpo.py` (ADR-0005, issue #211).
- **Frozen arms:** new frozen-arm wiring MUST go through `FrozenArmMixin._register_frozen_arm` (ADR-0031 A1). Do not reintroduce the `object.__setattr__` bypass or a hand-rolled state-dict override — the mixin is the single owner of the register + dual-exclude contract.
- **Per-rank device:** pre-PG shells MUST resolve the per-rank CUDA device via `DevicePolicy.pin()`; the post-PG VAE warm MUST go through `DevicePolicy.warm_device(fallback)`. Do not reintroduce the inline `set_device` twin, the bare `torch.device("cuda")` in the controlnet CLI, or `manifold.data.latent_pipeline.resolve_warm_device` (ADR-0035).
- **Callbacks:** add new training callbacks via the `CallbackRegistry` (ADR-0029). The `TrainingSpine.run` merge order — defaults → knobs → `--callbacks` override → `forbidden_callbacks` / `forbidden_monitors` — is the single source of truth for which callbacks fire and which monitors are allowed (ADR-0032). New specs are `@dataclass` classes with knobs as fields and a `build(ctx)` method; a spec that logs a metric declares `logged_metrics: ClassVar[frozenset[str]]`.
- **Paired fidelity/eval:** preserve `min_max_to_unit` → `PairedFidelityMetrics(data_range=1.0)` and same-noise before/after pairing across the pipeline, offline driver, report, and the supervised-stage in-training callback. Change the `manifold.eval` barrel, console wiring, metrics contract, `manifold.metrics.paired` / `manifold.metrics.paired_callback`, and the page schema together when extending the shipped eval surface.
- **In-training paired monitor:** `PairedFidelitySpec` is **shipped on the supervised ControlNet stage** (ADR-0037). It is registered only in `manifold-train-controlnet` (`spine.registry.register("paired_fidelity", …)`) and logs `val/psnr` / `val/ssim`; the checkpoint monitor stays `val/x0_mae` (the spec is **observe-only** — never a checkpoint selector, never a loss term, runs under `inference_mode`). All DDP ranks evaluate the same fixed subset redundantly (collective-count invariance is trivial — no FID-style sharding / error-rendezvous). The ControlNet-GRPO path's blanket `forbidden_callbacks={"fid": …}` still excludes the paired-fidelity callback until the ADR-0037 deliberate follow-up lands; do not assume the GRPO shell sees `paired_fidelity`.
- **Metrics:** distinguish per-rank accumulation from global reduction. Manual `all_reduce` must not also use `sync_dist=True` for the same value. Per-rank accumulation through `torchmetrics.MeanMetric(weight=batch_size)` (e.g. `LatentX0MAE`) gives the true global mean without a manual reducer; do not convert it to a per-rank mean that needs a custom `all_reduce`.
- **Checkpoint behavior:** update training callbacks, export, downstream frozen-generator loaders, and tests as one contract.
- **Console scripts:** `pyproject.toml` is the single source of truth. The paired-reward deletion guard parses it directly, while `tests/test_eval_cli.py` exercises the full `manifold-eval` install-facing workflow; `ComparisonPageBuilder` remains library-only by design.

## Backlog

- Resolve the stale `--pipeline paired` reference in `src/manifold/training/export_cli.py` (`choices=("jit", "paired", "controlnet")` still references a now-deleted module); either drop the `paired` choice or restore `manifold.pipelines.paired_latent_flow.PairedLatentFlowPipeline` before it can be exercised again. ADR-0034 retired the pipeline; the CLI choice and the `--pipeline paired` doc block survive as historical references.
- Land the deliberate ADR-0037 follow-up: extend the supervised-stage paired-fidelity monitor to the ControlNet-GRPO policy path. That requires releasing the blanket `forbidden_callbacks={"fid": …}` for *this* callback specifically (the FID exclusion rationale does not cover `controlnet_rollout`-based paired fidelity) and keeping the VAE on GPU for the decode. Until that lands, only the supervised `manifold-train-controlnet` stage observes `val/psnr` / `val/ssim`.
