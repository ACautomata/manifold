---
type: Reference
title: Architecture and Source Map
description: Component boundaries, data/config layers, evaluation/reporting boundaries, the runtime flows (JiT, supervised ControlNet, in-training paired-fidelity monitor, reward/GRPO, before/after eval), and where to look in source.
tags: [architecture, source-map, components, data-flow]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-24T12:31:51.781Z
sources:
  - id: openwiki-source-39c3295efc089133e87a9c80
    resource: repo://CONTEXT.md
  - id: openwiki-source-f30679e5671d5ddfc684fd24
    resource: repo://docs/adr/0006-training-ckpt-lightning-native-via-export.md
  - id: openwiki-source-862c417e516477791e237829
    resource: repo://docs/adr/0026-controlnet-via-monai-native-residual-interface.md
  - id: openwiki-source-1b5659f7853fdac71576f4bd
    resource: repo://docs/adr/0034-one-realism-reward-both-grpo-policies-delete-condition-aware.md
  - id: openwiki-source-e6eba8b3aeb5f992ffb439cc
    resource: repo://src/manifold/__init__.py
  - id: openwiki-source-02e48983991c862f88057f2d
    resource: repo://src/manifold/config/builder.py
  - id: openwiki-source-9054f9930c7ee1758c84b2c7
    resource: repo://src/manifold/config/loader.py
  - id: openwiki-source-be17d6b0aae1753017c77cef
    resource: repo://src/manifold/configuration.py
  - id: openwiki-source-ca412c5f25d3825e8f38c161
    resource: repo://src/manifold/data/paired_brats.py
  - id: openwiki-source-0ed03478d86efdffd1501c9e
    resource: repo://src/manifold/data/paired_manifests.py
  - id: openwiki-source-121ee09471abdc662d9d3b48
    resource: repo://src/manifold/eval/before_after.py
  - id: openwiki-source-e22b4e5926edbd63eb845214
    resource: repo://src/manifold/eval/cli.py
  - id: openwiki-source-301a360ed5641c7f24af6d5a
    resource: repo://src/manifold/eval/slice_grid.py
  - id: openwiki-source-3a58c59be60433f8b9221b6c
    resource: repo://src/manifold/metrics/fid/reducer.py
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-2472d02c3afe41b7e0d68982
    resource: repo://src/manifold/metrics/paired.py
  - id: openwiki-source-a4a6e4b18ed70c8f8195546e
    resource: repo://src/manifold/metrics/vae_stage.py
  - id: openwiki-source-425df5dedfde0932318ead5e
    resource: repo://src/manifold/models/controlnet_3d.py
  - id: openwiki-source-86c224873935ceedbf0b0d6b
    resource: repo://src/manifold/models/reward_model.py
  - id: openwiki-source-c869059bcd63f8f47204feef
    resource: repo://src/manifold/models/unet_3d_condition.py
  - id: openwiki-source-aefad772862dded74740a681
    resource: repo://src/manifold/modules/controlnet_latent_flow.py
  - id: openwiki-source-0718666398e4c020f19a709c
    resource: repo://src/manifold/modules/controlnet_sampler.py
  - id: openwiki-source-487431944da97854e8cb6d33
    resource: repo://src/manifold/modules/frozen_arm.py
  - id: openwiki-source-ffd9f2f79ca81385e197ae7f
    resource: repo://src/manifold/modules/grpo.py
  - id: openwiki-source-81c1e136efa0c7240a130f1a
    resource: repo://src/manifold/modules/sampler.py
  - id: openwiki-source-63410e878c74053bcaeb98f8
    resource: repo://src/manifold/pipelines/latent_flow.py
  - id: openwiki-source-b99ca7a395e97656837ca104
    resource: repo://src/manifold/pipelines/pipeline_utils.py
  - id: openwiki-source-4a976a922e9a1286c789e272
    resource: repo://src/manifold/training/callbacks/paired_fidelity.py
  - id: openwiki-source-bf73652a42104a8f9ef7e65e
    resource: repo://src/manifold/training/callbacks/registry.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
  - id: openwiki-source-be99bd216fffaa3e1d9212c0
    resource: repo://src/manifold/training/core.py
  - id: openwiki-source-deca8f8f4a3d2dbbba47ac16
    resource: repo://src/manifold/training/device_policy.py
  - id: openwiki-source-d4e091552e1ad8d0fe58e7b2
    resource: repo://src/manifold/training/export.py
  - id: openwiki-source-4bdb3113f633b5c6a86fd01d
    resource: repo://src/manifold/training/grpo_cli.py
  - id: openwiki-source-c6b41726b73d1e18b82404a3
    resource: repo://src/manifold/training/trainer.py
  - id: openwiki-source-43f6a22439495a06382f037c
    resource: repo://tests/test_paired_fidelity_ddp.py
generated: { by: "openwiki/0.7.2", at: "2026-10-24T12:31:51.781Z" }
---

# Architecture and source map

## Component model

Manifold deliberately mirrors the diffusers vocabulary while keeping training and inference concerns separate (`CONTEXT.md`).

| Layer | Responsibility | Primary sources |
|---|---|---|
| Models | Thin wrappers around MONAI MAISI VAE/UNet/ControlNet implementations plus PatchGAN reward scoring. The VAE owns latent scaling and sliding-window encode/decode. | `src/manifold/models/` |
| Schedulers | Rectified-flow transport, timestep/sigma grids, model-input scaling, Heun integration, and stochastic GRPO/bridge transitions. They contain no training loop. | `src/manifold/schedulers/` |
| Modules | stable-pretraining training components: objectives, optimizer/schedule wiring, rollout and validation steps for JiT, the supervised ControlNet translator, reward, and GRPO. | `src/manifold/modules/` |
| Pipelines | Native inference composition of UNet, ControlNet (or frozen UNet plus ControlNet), scheduler, and VAE, with `save_pretrained`/`from_pretrained`. | `src/manifold/pipelines/` |
| Training orchestration | CLI parsing, config composition, data warming, the `CallbackRegistry` + `TrainingSpine` assembly pipeline, Lightning trainer construction, checkpointing, and export. | `src/manifold/training/`, `src/manifold/metrics/` |
| Metrics callbacks | Per-epoch FID, latent-space x0-MAE, GRPO reward, the observe-only in-training paired-fidelity monitor (`val/psnr` / `val/ssim`), and automatic metrics line-chart rendering. | `src/manifold/metrics/`, `src/manifold/training/metrics.py`, `src/manifold/metrics/paired_callback.py`, `src/manifold/training/callbacks/paired_fidelity.py` |
| Offline evaluation and reporting | `manifold-eval` policy routing, same-noise before/after generation, shared decode normalization, MONAI paired PSNR/SSIM, 2.5D slice grids, and self-contained HTML comparison. | `src/manifold/eval/`, `src/manifold/metrics/paired.py`, `src/manifold/pipelines/pipeline_utils.py` |

The shared rollout primitives are intentional: training-time sampling and native inference delegate to the same sampler behavior rather than maintaining parallel integrators (`src/manifold/modules/sampler.py`, `controlnet_sampler.py`; ADR-0005).

```mermaid
flowchart LR
    Cache["VAE latent cache"] --> JiT["LatentFlowModule"]
    Cache --> ControlNet["ControlNetLatentFlowModule"]
    ControlNet --> Monitor["Paired-fidelity callback"]
    Monitor --> Curve["val/psnr and val/ssim, observe-only"]
    JiT --> Export["manifold-export"]
    ControlNet --> Export
    Export --> Eval["manifold-eval"]
    Eval --> Core["BeforeAfterEval"]
    Core --> Metric["Paired PSNR and SSIM"]
    Core --> Artifacts["metrics JSON and slice grids"]
```

*Figure: Training and native artifacts converge at export, then the evaluation path scores paired targets and writes portable artifacts; the supervised ControlNet branch additionally carries the in-training paired-fidelity monitor, which never touches the export path.*

## Runtime flows

### Noise-to-data JiT

A frozen VAE encodes volumes into an unscaled cache. The data layer estimates a single scaling factor and applies it on read. `LatentFlowModule` trains the conditional UNet to predict the clean latent from interpolated noise, while `FlowMatchHeunDiscreteScheduler` owns transport and integration. `LatentFlowPipeline` starts inference from Gaussian noise, applies optional interval-restricted classifier-free guidance, integrates from `t=0` to `t=1`, and decodes the result.

Start with:

- `src/manifold/data/latent_pipeline.py`, `latent_dataset.py`, `warm_datamodule.py`
- `src/manifold/modules/latent_flow.py`, `sampler.py`
- `src/manifold/schedulers/scheduling_flow_match_heun.py`
- `src/manifold/pipelines/latent_flow.py`
- `src/manifold/training/cli.py`

### Supervised paired translator (ControlNet)

The paired `x_src -> x_tgt` translator is a **trainable ControlNet on a frozen JiT base UNet** — see the component model above and `docs/adr/0026-controlnet-via-monai-native-residual-interface.md`, `0027-controlnet-supervised-then-grpo-two-stage.md`. The ControlNet takes `z_t` through its own cloned `conv_in` and adds the `x_src` control signal through `controlnet_cond_embedding` **after** `conv_in`; nothing is concatenated into the frozen base, because ADR-0026 rejected input concat (the base `conv_in` is frozen at `C_latent` channels). The ControlNet emits per-block residuals that the frozen base consumes through an out-of-place forward (in-place adds would break the grad-bearing residual path). The frozen base UNet is held **registered + dual-excluded** via [`FrozenArmMixin`](frozen-arm-and-device-policy.md#frozenarmmixin-register--dual-exclude) (ADR-0031 A1): registered so Lightning owns device placement, dual-excluded (`requires_grad=False` + `state_dict` strip + `train()` re-eval) so it is off the optimizer and off the checkpoint. The ControlNet's (src, tgt) contrast embeddings combine through `concat([embed(src), embed(tgt+offset)])` with the optional `paired_direction_offset` flipping the symmetry. The supervised loss is the `(1 - t)^-2`-weighted x0-MSE the base itself was trained with (ADR-0002 / ADR-0027).

The paired reward pipeline was deleted in ADR-0034 (paired-reward CLI, condition-aware `2·C` reward, offline pair precompute); the ControlNet path no longer needs a condition-aware reward because translation fidelity now comes from `x_src` conditioning + supervised init + the KL anchor, and the *single* realism reward (`RewardModel`, `C_latent`, partial-denoise pairs) scores `z_K` unconditionally for both GRPO policies.

BraTS-specific code groups volumes by subject and contrast, creates subject-disjoint splits, and enumerates all ordered non-self pairs. The dataset contract itself remains generic: source/target latents, labels, and spacing. The shared two-way subject splitter `_train_val_manifests` lives in `src/manifold/data/paired_manifests.py` (relocated from the deleted paired-reward CLI, consumed by `controlnet_cli` and `grpo_cli`).

<!-- openwiki: broken internal link [evaluation.md#accepted-in-training-monitor-planned-not-active] heading anchor "accepted-in-training-monitor-planned-not-active" does not exist in "evaluation.md". Fix the href or restore the target, then delete this comment. -->
The supervised checkpoint monitor stays latent-space `val/x0_mae`. ADR-0037's observe-only fixed-subset `val/psnr` / `val/ssim` callback **is** implemented in the current tree (`src/manifold/metrics/paired_callback.py` + `src/manifold/training/callbacks/paired_fidelity.py`) and is **default-on** for `manifold-train-controlnet`; it never selects checkpoints, never enters the loss, and never touches the optimizer. See [In-training paired-fidelity monitor](#in-training-paired-fidelity-monitor-supervised-controlnet) below. (The [Before/after GRPO evaluation](evaluation.md#accepted-in-training-monitor-planned-not-active) page still describes this monitor as "planned, not active" — that section is stale against the current tree and is superseded by the account here and on the dedicated [Paired-fidelity monitor](paired-fidelity-monitor.md) page.)

Start with:

- `src/manifold/data/paired_brats.py`, `paired_volume_dataset.py`, `paired_latent_dataset.py`, `paired_manifests.py`
- `src/manifold/models/controlnet_3d.py`
- `src/manifold/modules/controlnet_latent_flow.py`, `controlnet_sampler.py`
- `src/manifold/pipelines/controlnet_latent_flow.py`
- `src/manifold/training/controlnet_cli.py`
- `configs/train/config_controlnet_supervised.yaml`

### In-training paired-fidelity monitor (supervised ControlNet)

The monitor is the observe-only per-epoch drift screen ADR-0037 adds to the supervised ControlNet stage: on each gated validation epoch it rolls a **fixed paired subset** under **fixed initial noise** through the module's own Heun rollout, decodes, and scores the generated target against the real target with the same 3D PSNR/SSIM the offline comparison uses.

- **Wiring.** `PairedFidelityCallback` (`src/manifold/metrics/paired_callback.py`) is a `pl.Callback`; `PairedFidelitySpec` (`src/manifold/training/callbacks/paired_fidelity.py`) is its registry spec. `run_controlnet_training` registers the spec under the name `paired_fidelity` and passes `default_names=["train_loss", "checkpoint", "paired_fidelity"]`, so the monitor is **on by default** for `manifold-train-controlnet` (`src/manifold/training/controlnet_cli.py`). Both names are exported from `src/manifold/metrics/__init__.py` and `src/manifold/training/callbacks/__init__.py`.
- **Observe-only contract.** The spec declares `logged_metrics = {"val/psnr", "val/ssim"}` so the registry's `validate_monitor` accepts them as *validatable* monitors, but the checkpoint spec's `monitor_metric` stays `val/x0_mae`; the callback never drives checkpoint selection, never enters the loss, and never touches the optimizer or EMA. Generation runs under `torch.inference_mode()`, so no gradient is ever formed.
- **Mechanics.** The paired validation dataset is resolved **lazily at the first gated epoch** (`getattr(datamodule, "val_latent_ds", datamodule)`), because the cold warm path replaces `val_latent_ds` inside `DataModule.setup()`. A seeded `randperm` prefix of `subset_size` samples is collated and **cached**, so only the model changes between epochs; the generation noise is drawn from a fresh `torch.Generator` re-seeded with the callback's `seed` on every gated epoch, which keeps sampling stochasticity out of the epoch-over-epoch comparison.
- **Shared machinery, not parallel machinery.** The rollout is the module's own `ControlNetLatentFlowModule.sample()` — the shared `controlnet_rollout` primitive the inference Pipeline also uses (ADR-0005). The decode reuses `VaeStage` (stage the VAE onto the device for the decode, restore it to CPU afterwards), `LatentDecoder`, `min_max_to_unit`, and `PairedFidelityMetrics` (`data_range = 1.0`), each already tested — so the in-training curve is directly comparable with the offline before/after number.
- **Step count is recipe-primary.** The supervised spine fills `CallbackContext.inference_recipe` from the existing `controlnet.num_inference_steps` knob (default 15), and `PairedFidelitySpec.build` reads the step count from there unless the spec's own `num_inference_steps` knob overrides it.
- **DDP.** Every rank runs the monitor on the **same** fixed subset redundantly (DDP-synchronized weights + identical seeded noise + identical fixed input ⇒ identical result), so the cross-rank reduction is just Lightning's torchmetrics sync on the two module-attached `MeanMetric`s. There is no sharding and no FID-style error-rendezvous machinery, so collective-count invariance (ADR-0030) holds trivially.
- **Scope.** Only the supervised ControlNet path registers the spec; the GRPO-stage extension remains a separate follow-up (ADR-0037).

```mermaid
sequenceDiagram
    participant Trainer as Lightning Trainer
    participant Callback as Paired-fidelity callback
    participant Module as ControlNetLatentFlowModule
    participant VAE as Frozen VAE
    participant Metric as PairedFidelityMetrics

    Trainer->>Callback: on_validation_epoch_end
    Callback->>Callback: gate on every_n_epochs
    Callback->>Callback: select and cache the fixed paired subset
    Callback->>Module: sample under seeded noise in inference mode
    Module-->>Callback: generated target latent
    Callback->>VAE: stage to device and decode generated and real target
    VAE-->>Callback: decoded volumes
    Callback->>Callback: min_max_to_unit both volumes
    Callback->>Metric: score generated against real
    Metric-->>Callback: psnr and ssim
    Callback->>Trainer: log val/psnr and val/ssim
```

*Figure: The monitor reuses the module's own rollout and the shared decode → min-max → MONAI metric path, then logs two scalars that never drive checkpoint selection.*

Detailed behavior, knob semantics, and the focused test anchors live on the dedicated [Paired-fidelity monitor](paired-fidelity-monitor.md) page; the registry-side extension recipe is in [Callback registry and training spine](callback-registry.md#change-guidance).

### Reward and policy post-training

`RewardModel` wraps a MONAI PatchGAN discriminator and pools its output to a scalar. Reward training learns a mode-agnostic realism score from partial-denoise corruption pairs. The unified `GRPOModule` can optimize either the JiT UNet policy or a warm-started ControlNet on a frozen base UNet; both paths fork stochastic transitions and score the terminal latent `z_K` unconditionally with the *same* reward before applying the clipped group-relative objective. The policy is inferred from the native artifact passed to `--native-dir`, not a flag. For the ControlNet path, translation fidelity comes from `x_src` conditioning, supervised initialization, and the KL anchor rather than a separate condition-aware reward (ADR-0034 deleted the paired-reward pipeline). See the operational routing and recipe contract in [Reward and GRPO stages](workflows.md#reward-and-grpo-stages).

Start with `src/manifold/models/reward_model.py`, `src/manifold/modules/{reward,grpo}.py`, `src/manifold/modules/controlnet_sampler.py`, and `src/manifold/training/{reward_cli,grpo_cli,controlnet_cli}.py`.

### Before/after GRPO evaluation

The shipped `manifold-eval` command exports the post-GRPO checkpoint against the before export's component structure, reloads both artifacts, and sends them through `BeforeAfterEval`. The driver creates identical initial noise and conditioning for each seed, decodes every latent with the frozen VAE, and applies the shared `min_max_to_unit` contract. JiT emits a `before | after` provenance-only metric record; ControlNet additionally scores each generated target against its real target with MONAI 3D PSNR/SSIM. One 2.5D three-plane grid is written per sample. See [Before/after GRPO evaluation](evaluation.md#runtime-flow) for the runtime sequence, public API, artifact schema, and the report builder.

Start with `src/manifold/eval/cli.py`, `src/manifold/eval/before_after.py`, `src/manifold/eval/comparison_page.py`, `src/manifold/metrics/paired.py`, and `src/manifold/pipelines/pipeline_utils.py`.

## Configuration and persistence

Experiment YAML is composed by `src/manifold/config/loader.py` and built into components by `builder.py`. Later top-level blocks replace earlier ones unless `_base_` explicitly requests inheritance. This launch-time OmegaConf layer is separate from persisted component JSON handled by `src/manifold/configuration.py`.

Native inference directories contain component configuration/weights (including `model_index.json` and component subdirectories). Lightning `.ckpt` files are training state and are not loaded directly by pipelines; export is the bridge. `manifold-eval` depends on this boundary: its before directory supplies the loadable policy template and self-described `pipeline_class`, while the existing export bridge bakes the after `.ckpt` into `<output>/after_native`. The eval CLI therefore infers JiT versus ControlNet from the artifact rather than accepting a policy flag. See [Checkpoint and export contract](workflows.md#checkpoint-and-export-contract) and the eval [policy dispatch contract](evaluation.md#policy-dispatch-and-artifact-contract).

## Change guidance

- **Transport/integration:** change the scheduler and shared sampler path together; run scheduler, pipeline, and module tests to prevent train/inference drift.
- **Latent scaling:** preserve VAE ownership and the unscaled-cache contract; check VAE, data, persistence, and pipeline tests.
- **Paired conditioning/pairing:** keep BraTS discovery outside the generic dataset contract and preserve subject-level split isolation.
- **Paired fidelity/evaluation:** preserve the `min_max_to_unit` → `PairedFidelityMetrics(data_range=1.0)` ordering and same-noise before/after contract across the pipeline, offline eval, and the in-training monitor. A normalization, artifact, or report-schema change is a cross-component change, not a local patch. The in-training half of that contract is specified on the [Paired-fidelity monitor](paired-fidelity-monitor.md) page.
- **Metrics:** distinguish per-rank accumulation from global reduction. Manual all-reduced metrics must not also use `sync_dist`, or they will be reduced twice.
- **Checkpoint behavior:** update training callbacks, export, downstream frozen-generator loaders, and tests as one contract.
- **Frozen arms:** new frozen-arm wiring MUST go through `FrozenArmMixin._register_frozen_arm`, not via `object.__setattr__` or any custom state-dict override — the mixin is the single owner of the register + dual-exclude contract (ADR-0031 A1). The frozen arms stay in `parameters()` (Lightning owns device placement) but carry no grad and emit no checkpoint key.
- **Per-rank device:** shells MUST resolve the per-rank CUDA device through `DevicePolicy.pin()` (pre-PG) and `DevicePolicy.warm_device(fallback)` (post-PG VAE warm); do not reintroduce the inline `set_device` twin, the bare `torch.device("cuda" if torch.cuda.is_available() else "cpu")` in the controlnet path, or `manifold.data.latent_pipeline.resolve_warm_device` (ADR-0035).
- **Callbacks:** new callbacks MUST go through the `CallbackRegistry` two-phase resolve/build; the `TrainingSpine.run` merge order is the single source of truth for which callbacks fire, which knobs apply, and which monitors are allowed (ADR-0029 / ADR-0032).

For component-level change navigation (entry points, focused tests, minimal validation), see the [Quickstart task routing](quickstart.md#task-routing) and the stage-level table in [Workflows change navigation](workflows.md#change-navigation).
