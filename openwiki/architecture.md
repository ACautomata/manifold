---
type: Reference
title: Architecture and Source Map
description: Component boundaries, data/config layers, evaluation/reporting boundaries, domain vocabulary, and where to look in source.
tags: [architecture, source-map, components, data-flow]
verified:
  - by: openwiki/0.5.1
    at: 2026-09-12T12:03:27.299Z
sources:
  - id: openwiki-source-39c3295efc089133e87a9c80
    resource: repo://CONTEXT.md
  - id: openwiki-source-7b138eeb87c5724abe63cefb
    resource: repo://docs/adr/0001-diffusers-style-foundation.md
  - id: openwiki-source-788f7be6d37301b814c0e56a
    resource: repo://docs/adr/0002-true-heun-two-eval.md
  - id: openwiki-source-20987cd1438d636c011347c1
    resource: repo://docs/adr/0003-scale-factor-on-vae-and-native-ckpt.md
  - id: openwiki-source-9a7a71e2c4736624f873ed0e
    resource: repo://docs/adr/0004-omegaconf-layer-above-configmixin.md
  - id: openwiki-source-110359de612a3f65e17e363d
    resource: repo://docs/adr/0005-sampler-owned-by-module-pipeline-delegates.md
  - id: openwiki-source-f30679e5671d5ddfc684fd24
    resource: repo://docs/adr/0006-training-ckpt-lightning-native-via-export.md
  - id: openwiki-source-c9f8735566089eaa66d8da6a
    resource: repo://docs/adr/0007-retire-hope-converter.md
  - id: openwiki-source-862c417e516477791e237829
    resource: repo://docs/adr/0026-controlnet-via-monai-native-residual-interface.md
  - id: openwiki-source-969b5d5f156d14e4982cd6a3
    resource: repo://docs/adr/0028-two-mode-grpo-unify-controlnet-delete-bridge.md
  - id: openwiki-source-bdb26e011dbdfd0846537627
    resource: repo://docs/adr/0031-device-ownership.md
  - id: openwiki-source-1b5659f7853fdac71576f4bd
    resource: repo://docs/adr/0034-one-realism-reward-both-grpo-policies-delete-condition-aware.md
  - id: openwiki-source-e835c1371b59a53aa9b0005e
    resource: repo://docs/adr/0036-controlnet-fidelity-offline-3d-psnr-ssim.md
  - id: openwiki-source-1f077c4de341ab7dab070abb
    resource: repo://docs/adr/0037-controlnet-fidelity-in-training-monitor.md
  - id: openwiki-source-e6eba8b3aeb5f992ffb439cc
    resource: repo://src/manifold/__init__.py
  - id: openwiki-source-02e48983991c862f88057f2d
    resource: repo://src/manifold/config/builder.py
  - id: openwiki-source-9054f9930c7ee1758c84b2c7
    resource: repo://src/manifold/config/loader.py
  - id: openwiki-source-be17d6b0aae1753017c77cef
    resource: repo://src/manifold/configuration.py
  - id: openwiki-source-121ee09471abdc662d9d3b48
    resource: repo://src/manifold/eval/before_after.py
  - id: openwiki-source-e22b4e5926edbd63eb845214
    resource: repo://src/manifold/eval/cli.py
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-2472d02c3afe41b7e0d68982
    resource: repo://src/manifold/metrics/paired.py
  - id: openwiki-source-c81310ba9d1d09ac19a656cf
    resource: repo://src/manifold/models/autoencoder_kl.py
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
  - id: openwiki-source-6cdf24eed10631ed02fffcbf
    resource: repo://src/manifold/modules/latent_flow.py
  - id: openwiki-source-82594a5d307aebca75251b76
    resource: repo://src/manifold/modules/reward.py
  - id: openwiki-source-81c1e136efa0c7240a130f1a
    resource: repo://src/manifold/modules/sampler.py
  - id: openwiki-source-56674f106efd2c16c47c3e08
    resource: repo://src/manifold/pipelines/controlnet_latent_flow.py
  - id: openwiki-source-63410e878c74053bcaeb98f8
    resource: repo://src/manifold/pipelines/latent_flow.py
  - id: openwiki-source-b99ca7a395e97656837ca104
    resource: repo://src/manifold/pipelines/pipeline_utils.py
  - id: openwiki-source-041a3584d4049736545ceb90
    resource: repo://src/manifold/schedulers/scheduling_flow_match_grpo.py
  - id: openwiki-source-2dc5bf79aced1021bf03e1f2
    resource: repo://src/manifold/schedulers/scheduling_flow_match_heun.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
  - id: openwiki-source-d4e091552e1ad8d0fe58e7b2
    resource: repo://src/manifold/training/export.py
generated: { by: "openwiki/0.5.1", at: "2026-09-12T12:03:27.299Z" }
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
| Metrics callbacks | Per-epoch FID, latent-space x0-MAE, GRPO reward, and automatic metrics line-chart rendering. | `src/manifold/metrics/`, `src/manifold/training/metrics.py` |
| Offline evaluation and reporting | `manifold-eval` policy routing, same-noise before/after generation, shared decode normalization, MONAI paired PSNR/SSIM, 2.5D slice grids, and self-contained HTML comparison. | `src/manifold/eval/`, `src/manifold/metrics/paired.py`, `src/manifold/pipelines/pipeline_utils.py` |

The shared rollout primitives are intentional: training-time sampling and native inference delegate to the same sampler behavior rather than maintaining parallel integrators. `LatentFlowModule.sample` and `LatentFlowPipeline.sample_latent` both call `sample_latent_flow` (`src/manifold/modules/sampler.py`); `ControlNetLatentFlowModule.sample`, `ControlNetLatentFlowPipeline.__call__`, the GRPO ControlNet-policy suffix, and the reward fake-builder all call `controlnet_rollout` / `controlnet_x0` (`src/manifold/modules/controlnet_sampler.py`); the GRPO anchor + suffix share `FlowMatchGRPOScheduler.rollout_range` (ADR-0005).

```mermaid
flowchart LR
    Cache["VAE latent cache"] --> JiT["LatentFlowModule"]
    Cache --> ControlNet["ControlNetLatentFlowModule"]
    JiT --> Export["manifold-export"]
    ControlNet --> Export
    Export --> Eval["manifold-eval"]
    Eval --> Core["BeforeAfterEval"]
    Core --> Metric["Paired PSNR and SSIM"]
    Core --> Artifacts["metrics JSON and slice grids"]
```

*Figure: Training and native artifacts converge at export, then the evaluation path scores paired targets and writes portable artifacts.*

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

The paired `x_src -> x_tgt` translator is a **trainable ControlNet on a frozen JiT base UNet** — see the component model above and `docs/adr/0026-controlnet-via-monai-native-residual-interface.md`, `0027-controlnet-supervised-then-grpo-two-stage.md`. The ControlNet mirrors the base UNet's encoder (`conv_in` + the input-embedding path + `down_blocks` + `middle_block`) and adds a `controlnet_cond_embedding` conv that maps the source latent `x_src` to the `conv_in` output width and **adds** it post-`conv_in` (a separate control pathway — it is *not* concatenated into `z_t`). It then emits per-block residuals that the frozen base consumes through an **out-of-place** forward (in-place adds in MONAI's `_apply_down_blocks` would break the grad-bearing residual path; ADR-0026's corrected hazard). The frozen base UNet is held **registered + dual-excluded** via [`FrozenArmMixin`](frozen-arm-and-device-policy.md#frozenarmmixin-register--dual-exclude) (ADR-0031 A1): registered so Lightning owns device placement, dual-excluded (`requires_grad=False` + `state_dict` strip + `train()` re-eval) so it is off the optimizer and off the checkpoint. The ControlNet's (src, tgt) contrast embeddings combine through `concat([embed(src), embed(tgt+offset)])` with the optional `paired_direction_offset` flipping the symmetry; the base itself sees only the target-contrast label. The supervised loss is the `(1 - t)^-2`-weighted x0-MSE the base itself was trained with (ADR-0002 / ADR-0027). Zero-init zero-convs keep the initial ControlNet residuals at zero, so a fresh supervised ControlNet reproduces the pretrained base UNet byte-identically (a safe warm-start).

The paired reward pipeline was deleted in ADR-0034 (paired-reward CLI, condition-aware `2·C` reward, offline pair precompute); the ControlNet path no longer needs a condition-aware reward because translation fidelity now comes from `x_src` conditioning + supervised init + the KL anchor, and the *single* realism reward (`RewardModel`, `C_latent`, partial-denoise pairs) scores `z_K` unconditionally for both GRPO policies.

BraTS-specific code groups volumes by subject and contrast, creates subject-disjoint splits, and enumerates all ordered non-self pairs. The dataset contract itself remains generic: source/target latents, labels, and spacing. The shared two-way subject splitter `_train_val_manifests` lives in `src/manifold/data/paired_manifests.py` (relocated from the deleted paired-reward CLI, consumed by `controlnet_cli` and `grpo_cli`).

<!-- openwiki: broken internal link [evaluation.md#accepted-in-training-monitor-planned-not-active] heading anchor "accepted-in-training-monitor-planned-not-active" does not exist in "evaluation.md". Fix the href or restore the target, then delete this comment. -->
The active supervised checkpoint monitor remains latent-space `val/x0_mae`. ADR-0037 accepts an observe-only fixed-subset `val/psnr` / `val/ssim` callback, but that callback is not implemented in the current tree; treat the decision and its extension surface as planned work described in [Before/after GRPO evaluation](evaluation.md#accepted-in-training-monitor-planned-not-active).

Start with:

- `src/manifold/data/paired_brats.py`, `paired_volume_dataset.py`, `paired_latent_dataset.py`, `paired_manifests.py`
- `src/manifold/models/controlnet_3d.py`
- `src/manifold/modules/controlnet_latent_flow.py`, `controlnet_sampler.py`
- `src/manifold/pipelines/controlnet_latent_flow.py`
- `src/manifold/training/controlnet_cli.py`
- `configs/train/config_controlnet_supervised.yaml`

### Reward and policy post-training

`RewardModel` wraps a MONAI PatchGAN discriminator and pools its output to a scalar. Reward training learns a mode-agnostic realism score from partial-denoise corruption pairs. The unified `GRPOModule` can optimize either the JiT UNet policy or a warm-started ControlNet on a frozen base UNet; both paths fork stochastic transitions and score the terminal latent `z_K` unconditionally with the *same* reward before applying the clipped group-relative objective. The policy is inferred from the native artifact passed to `--native-dir`, not a flag. For the ControlNet path, translation fidelity comes from `x_src` conditioning, supervised initialization, and the KL anchor rather than a separate condition-aware reward (ADR-0034 deleted the paired-reward pipeline). See the operational routing and recipe contract in [Reward and GRPO stages](workflows.md#reward-and-grpo-stages).

Start with `src/manifold/models/reward_model.py`, `src/manifold/modules/{reward,grpo}.py`, `src/manifold/modules/controlnet_sampler.py`, and `src/manifold/training/{reward_cli,grpo_cli,controlnet_cli}.py`.

### Before/after GRPO evaluation

The shipped `manifold-eval` command exports the post-GRPO checkpoint against the before export's component structure, reloads both artifacts, and sends them through `BeforeAfterEval`. The driver creates identical initial noise and conditioning for each seed, decodes every latent with the frozen VAE, and applies the shared `min_max_to_unit` contract. JiT emits a `before | after` provenance-only metric record; ControlNet additionally scores each generated target against its real target with MONAI 3D PSNR/SSIM. One 2.5D three-plane grid is written per sample. See [Before/after GRPO evaluation](evaluation.md#runtime-flow) for the runtime sequence, public API, artifact schema, and the accepted-but-unimplemented in-training monitor.

Start with `src/manifold/eval/cli.py`, `src/manifold/eval/before_after.py`, `src/manifold/eval/comparison_page.py`, `src/manifold/metrics/paired.py`, and `src/manifold/pipelines/pipeline_utils.py`.

## Configuration and persistence

Manifold has two config systems by design (ADR-0004). The **experiment config** is OmegaConf: env (paths) → train/inference recipe → network-construction kwargs, composed by `src/manifold/config/loader.py` and built into components by `builder.py`. Each later top-level YAML block replaces earlier ones wholesale (not deep-merged), `_base_` provides inheritance *inside* a single file, required paths are `???` (OmegaConf MISSING — fail-fast on read, with `require_paths` surfacing the unset key), and CLI `--<key>` flags plus Hydra-style dotlist overrides layer on top (dotlist wins). The composed config builds components at launch and never persists them.

The **component config** is the JSON `config.json` each component writes via `register_to_config` / `ConfigMixin` (`src/manifold/configuration.py`) and round-trips through `from_pretrained` / `save_pretrained` — the diffusers-style persistence contract for a trained component, independent of how it was launched. Manifold's mixin is its own minimal implementation (it does **not** subclass `diffusers.ConfigMixin`; ADR-0001).

Native inference directories contain per-component subdirectories (`unet/`, `vae/`, `scheduler/`, plus `controlnet/` for the ControlNet pipeline), each carrying its `config.json` (and `diffusion_pytorch_model.pt` for models). The directory's top-level `model_index.json` self-describes the pipeline (`pipeline_class` + component qualnames). Lightning `.ckpt` files are training state and are not loaded directly by pipelines; export is the bridge (`src/manifold/training/export.py` → `manifold-export`, ADR-0006, now the sole checkpoint → inference path after ADR-0007 retired the hope→native converter). The export bakes the **raw** optimizer weights as the inference UNet (EMA training was removed; the `val/fid` monitor and the export are deliberately aligned so the exported "best" is best for the weights that are published). The ControlNet export reuses the same one-shot bridge with `pipeline_cls=ControlNetLatentFlowPipeline`; only the `controlnet.*` keys are baked from the supervised checkpoint (the frozen base is held unregistered and passed through from `--base-native-dir`).

`manifold-eval` depends on this boundary: its before directory supplies the loadable policy template and self-described `pipeline_class`, while the export bridge bakes the after `.ckpt` into `<output>/after_native`. The eval CLI therefore infers JiT versus ControlNet from the artifact's `pipeline_class` rather than accepting a policy flag (ADR-0034). See [Checkpoint and export contract](workflows.md#checkpoint-and-export-contract) and the eval [policy dispatch contract](evaluation.md#policy-dispatch-and-artifact-contract).

## Change guidance

- **Transport/integration:** change the scheduler and shared sampler path together; run scheduler, pipeline, and module tests to prevent train/inference drift.
- **Latent scaling:** preserve VAE ownership and the unscaled-cache contract; check VAE, data, persistence, and pipeline tests.
- **Paired conditioning/pairing:** keep BraTS discovery outside the generic dataset contract and preserve subject-level split isolation.
- **Paired fidelity/evaluation:** preserve the `min_max_to_unit` → `PairedFidelityMetrics(data_range=1.0)` ordering and same-noise before/after contract across the pipeline, offline eval, and the future in-training monitor. A normalization, artifact, or report-schema change is a cross-component change, not a local patch.
- **Metrics:** distinguish per-rank accumulation from global reduction. Manual all-reduced metrics must not also use `sync_dist`, or they will be reduced twice.
- **Checkpoint behavior:** update training callbacks, export, downstream frozen-generator loaders, and tests as one contract.
- **Frozen arms:** new frozen-arm wiring MUST go through `FrozenArmMixin._register_frozen_arm`, not via `object.__setattr__` or any custom state-dict override — the mixin is the single owner of the register + dual-exclude contract (ADR-0031 A1). The frozen arms stay in `parameters()` (Lightning owns device placement) but carry no grad and emit no checkpoint key.
- **Per-rank device:** shells MUST resolve the per-rank CUDA device through `DevicePolicy.pin()` (pre-PG) and `DevicePolicy.warm_device(fallback)` (post-PG VAE warm); do not reintroduce the inline `set_device` twin, the bare `torch.device("cuda" if torch.cuda.is_available() else "cpu")` in the controlnet path, or `manifold.data.latent_pipeline.resolve_warm_device` (ADR-0035).
- **Callbacks:** new callbacks MUST go through the `CallbackRegistry` two-phase resolve/build; the `TrainingSpine.run` merge order is the single source of truth for which callbacks fire, which knobs apply, and which monitors are allowed (ADR-0029 / ADR-0032).

For component-level change navigation (entry points, focused tests, minimal validation), see the [Quickstart task routing](quickstart.md#task-routing) and the stage-level table in [Workflows change navigation](workflows.md#change-navigation).
