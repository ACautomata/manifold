---
type: Reference
title: Architecture and Source Map
description: Component boundaries, data/config layers, evaluation/reporting boundaries, domain vocabulary, and where to look in source.
tags: [architecture, source-map, components, data-flow]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-10T14:21:58.700Z
sources:
  - id: openwiki-source-862c417e516477791e237829
    resource: repo://docs/adr/0026-controlnet-via-monai-native-residual-interface.md
  - id: openwiki-source-4ee4003f33469f012ed392fe
    resource: repo://docs/adr/0027-controlnet-supervised-then-grpo-two-stage.md
  - id: openwiki-source-1b5659f7853fdac71576f4bd
    resource: repo://docs/adr/0034-one-realism-reward-both-grpo-policies-delete-condition-aware.md
  - id: openwiki-source-e6eba8b3aeb5f992ffb439cc
    resource: repo://src/manifold/__init__.py
  - id: openwiki-source-be17d6b0aae1753017c77cef
    resource: repo://src/manifold/configuration.py
  - id: openwiki-source-09e4eceb7e21a9e44d0eeb8d
    resource: repo://src/manifold/eval/__init__.py
  - id: openwiki-source-e22b4e5926edbd63eb845214
    resource: repo://src/manifold/eval/cli.py
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-666d28928b5f89a8354d80ac
    resource: repo://src/manifold/models/__init__.py
  - id: openwiki-source-425df5dedfde0932318ead5e
    resource: repo://src/manifold/models/controlnet_3d.py
  - id: openwiki-source-f25390467e603a12f73af39d
    resource: repo://src/manifold/models/modeling_utils.py
  - id: openwiki-source-0415468d2196bb148ba1d91e
    resource: repo://src/manifold/modules/__init__.py
  - id: openwiki-source-aefad772862dded74740a681
    resource: repo://src/manifold/modules/controlnet_latent_flow.py
  - id: openwiki-source-0718666398e4c020f19a709c
    resource: repo://src/manifold/modules/controlnet_sampler.py
  - id: openwiki-source-487431944da97854e8cb6d33
    resource: repo://src/manifold/modules/frozen_arm.py
  - id: openwiki-source-81c1e136efa0c7240a130f1a
    resource: repo://src/manifold/modules/sampler.py
  - id: openwiki-source-f571e3a5231f7055f805276d
    resource: repo://src/manifold/pipelines/__init__.py
  - id: openwiki-source-56674f106efd2c16c47c3e08
    resource: repo://src/manifold/pipelines/controlnet_latent_flow.py
  - id: openwiki-source-63410e878c74053bcaeb98f8
    resource: repo://src/manifold/pipelines/latent_flow.py
  - id: openwiki-source-918f26e9f16813877a79b7b5
    resource: repo://src/manifold/schedulers/__init__.py
  - id: openwiki-source-2dc5bf79aced1021bf03e1f2
    resource: repo://src/manifold/schedulers/scheduling_flow_match_heun.py
  - id: openwiki-source-1d5844760600cb6bb2564175
    resource: repo://src/manifold/training/__init__.py
  - id: openwiki-source-812f4876e72062b85443a7c2
    resource: repo://src/manifold/training/callbacks/__init__.py
  - id: openwiki-source-4a976a922e9a1286c789e272
    resource: repo://src/manifold/training/callbacks/paired_fidelity.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
  - id: openwiki-source-bf926767b4a66a5c439a26ca
    resource: repo://src/manifold/training/export_cli.py
  - id: openwiki-source-d4e091552e1ad8d0fe58e7b2
    resource: repo://src/manifold/training/export.py
  - id: openwiki-source-36db33c03ac89b676295a74d
    resource: repo://tests/test_callback_registry.py
  - id: openwiki-source-7131b0d80871bbec1bff1bff
    resource: repo://tests/test_controlnet_cli.py
generated: { by: "openwiki/0.7.2", at: "2026-10-10T14:21:58.700Z" }
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

The shared rollout primitives are intentional: training-time sampling and native inference delegate to the same sampler behavior rather than maintaining parallel integrators (`src/manifold/modules/sampler.py`, `controlnet_sampler.py`; ADR-0005).

```mermaid
flowchart LR
    subgraph Data["Data (layer 1)"]
        Cache["VAE latent cache"]
    end
    subgraph Training["Training orchestration (layer 5)"]
        JiT["LatentFlowModule"]
        ControlNet["ControlNetLatentFlowModule"]
        Reward["RewardModule"]
        GRPO["GRPOModule"]
    end
    subgraph Pipes["Inference pipelines (layer 4)"]
        JitPipe["LatentFlowPipeline"]
        CnPipe["ControlNetLatentFlowPipeline"]
    end
    subgraph Eval["Offline eval and reporting (layer 7)"]
        EvalCmd["manifold-eval"]
        Driver["BeforeAfterEval"]
        Metric["PairedFidelityMetrics"]
        Artifacts["metrics JSON and slice grids"]
    end
    Cache --> JiT
    Cache --> ControlNet
    JiT --> Export["manifold-export (ADR-0006)"]
    ControlNet --> Export
    Reward --> Export
    GRPO --> Export
    Export -- native per-component dir --> JitPipe
    Export -- native per-component dir --> CnPipe
    JitPipe --> EvalCmd
    CnPipe --> EvalCmd
    EvalCmd --> Driver
    Driver --> Metric
    Driver --> Artifacts
```

*Figure: Seven-layer runtime — Data (1) feeds Training modules (5) JiT / ControlNet / Reward / GRPO; `manifold-export` (ADR-0006) writes the native per-component directory that the Inference pipelines (4) reload; Eval (7) reloads both pipelines and writes portable artifacts.*

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

The paired `x_src -> x_tgt` translator is a **trainable ControlNet on a frozen JiT base UNet** — see the component model above and `docs/adr/0026-controlnet-via-monai-native-residual-interface.md`, `0027-controlnet-supervised-then-grpo-two-stage.md`. The ControlNet consumes `concat([z_t, x_src, src_label, tgt_label])` and emits per-block residuals that the frozen base consumes through an out-of-place forward (in-place adds would break the grad-bearing residual path). The frozen base UNet is held **registered + dual-excluded** via [`FrozenArmMixin`](frozen-arm-and-device-policy.md#frozenarmmixin-register--dual-exclude) (ADR-0031 A1): registered so Lightning owns device placement, dual-excluded (`requires_grad=False` + `state_dict` strip + `train()` re-eval) so it is off the optimizer and off the checkpoint. The ControlNet's (src, tgt) contrast embeddings combine through `concat([embed(src), embed(tgt+offset)])` with the optional `paired_direction_offset` flipping the symmetry. The supervised loss is the `(1 - t)^-2`-weighted x0-MSE the base itself was trained with (ADR-0002 / ADR-0027).

The paired reward pipeline was deleted in ADR-0034 (paired-reward CLI, condition-aware `2·C` reward, offline pair precompute); the ControlNet path no longer needs a condition-aware reward because translation fidelity now comes from `x_src` conditioning + supervised init + the KL anchor, and the *single* realism reward (`RewardModel`, `C_latent`, partial-denoise pairs) scores `z_K` unconditionally for both GRPO policies.

BraTS-specific code groups volumes by subject and contrast, creates subject-disjoint splits, and enumerates all ordered non-self pairs. The dataset contract itself remains generic: source/target latents, labels, and spacing. The shared two-way subject splitter `_train_val_manifests` lives in `src/manifold/data/paired_manifests.py` (relocated from the deleted paired-reward CLI, consumed by `controlnet_cli` and `grpo_cli`).

The active supervised checkpoint monitor remains latent-space `val/x0_mae`. ADR-0037's observe-only fixed-subset `val/psnr` / `val/ssim` monitor is implemented and active in the current tree: `PairedFidelitySpec` (`src/manifold/training/callbacks/paired_fidelity.py`) is registered with the `CallbackRegistry` and emitted by default in the supervised `controlnet_cli` `default_names` (`["train_loss", "checkpoint", "paired_fidelity"]`); the built `PairedFidelityCallback` (`src/manifold/metrics/paired_callback.py`) runs the module's own full ControlNet Heun rollout on a seeded fixed paired subset each gated epoch, decodes via `LatentDecoder`, normalizes with `min_max_to_unit`, scores with `PairedFidelityMetrics`, and logs `val/psnr` / `val/ssim`. It stays observe-only — `PairedFidelitySpec.logged_metrics = frozenset({"val/psnr", "val/ssim"})` declares them as *validatable* monitors but never displaces `val/x0_mae` as the checkpoint selector (the recipe-primary rollout step count is threaded from the existing `controlnet.num_inference_steps` knob through `CallbackContext.inference_recipe`, issue #239). The ControlNet-GRPO monitor extension remains a separate follow-up (paired rollout uses the trainable ControlNet, so the unconditional-FID rationale does not apply).

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

The shipped `manifold-eval` command exports the post-GRPO checkpoint against the before export's component structure, reloads both artifacts, and sends them through `BeforeAfterEval`. The driver creates identical initial noise and conditioning for each seed, decodes every latent with the frozen VAE, and applies the shared `min_max_to_unit` contract. JiT emits a `before | after` provenance-only metric record; ControlNet additionally scores each generated target against its real target with MONAI 3D PSNR/SSIM. One 2.5D three-plane grid is written per sample. See [Before/after GRPO evaluation](evaluation.md#runtime-flow) for the runtime sequence, public API, and artifact schema; the offline and the in-training (ADR-0037 `PairedFidelityCallback`) metric paths share the same `PairedFidelityMetrics(data_range=1.0)` contract and `min_max_to_unit` normalization, so the offline number and the logged `val/psnr` / `val/ssim` curve are directly comparable.

Start with `src/manifold/eval/cli.py`, `src/manifold/eval/before_after.py`, `src/manifold/eval/comparison_page.py`, `src/manifold/metrics/paired.py`, and `src/manifold/pipelines/pipeline_utils.py`.

## Configuration and persistence

Experiment YAML is composed by `src/manifold/config/loader.py` and built into components by `builder.py`. Later top-level blocks replace earlier ones unless `_base_` explicitly requests inheritance. This launch-time OmegaConf layer is separate from persisted component JSON handled by `src/manifold/configuration.py`.

Native inference directories contain component configuration/weights (including `model_index.json` and component subdirectories). Lightning `.ckpt` files are training state and are not loaded directly by pipelines; export is the bridge. `manifold-eval` depends on this boundary: its before directory supplies the loadable policy template and self-described `pipeline_class`, while the existing export bridge bakes the after `.ckpt` into `<output>/after_native`. The eval CLI therefore infers JiT versus ControlNet from the artifact rather than accepting a policy flag. See [Checkpoint and export contract](workflows.md#checkpoint-and-export-contract) and the eval [policy dispatch contract](evaluation.md#policy-dispatch-and-artifact-contract).

### Native per-component pipeline format

Both shipped inference pipelines serialize to and reload from the same per-component directory layout — `model_index.json` at the root, one subdirectory per component, each component writing its own `config.json` (and, for model components, a `diffusion_pytorch_model.pt` weights file). Concretely (`src/manifold/pipelines/latent_flow.py`, `controlnet_latent_flow.py`):

```text
<native_dir>/
├── model_index.json            # {"format": "manifold", "pipeline_class": ..., "components": {...}}
├── unet/
│   ├── config.json             # the wrapper kwargs captured by register_to_config
│   └── diffusion_pytorch_model.pt
├── controlnet/                 # ControlNet pipeline only
│   ├── config.json
│   └── diffusion_pytorch_model.pt
├── vae/
│   ├── config.json             # scaling_factor lives here (ADR-0003)
│   └── diffusion_pytorch_model.pt
└── scheduler/
    └── scheduler_config.json   # stateless config only - no weights
```

`save_pretrained` writes `model_index.json` with `format="manifold"`, `pipeline_class=type(self).__name__` (so a `ControlNetLatentFlowPipeline` writes `"ControlNetLatentFlowPipeline"` and a `LatentFlowPipeline` writes `"LatentFlowPipeline"`), and a `components` map that enumerates each held component by its `module.ClassName` qualname (the `_qualname` helper). Each model component is persisted via `ModelMixin.save_pretrained` → `config.json` + `torch.save(state_dict())` (`map_location="cpu"`, `weights_only=True`); the scheduler is stateless and writes only its `ConfigMixin`-derived JSON via `to_json_file`. `from_pretrained` is the mirror — it requires `model_index.json`, refuses a directory without one with a clear `FileNotFoundError`, then constructs each component from its subdirectory and instantiates the pipeline class named in the index. The two pipelines therefore share one persistence contract: only the `components` map and the wrapper's `__init__` signature differ.

This contract is what `manifold-eval` dispatches on — `_pipeline_class_of` reads `pipeline_class` from the before artifact's `model_index.json`, picks the matching pipeline, and runs `run_unconditional` (JiT) or `run_paired` (ControlNet) accordingly. No mode flag is accepted. See [Before/after GRPO evaluation — policy dispatch](evaluation.md#policy-dispatch-and-artifact-contract) for the runtime sequence and the after-export write path.

## Change guidance

- **Transport/integration:** change the scheduler and shared sampler path together; run scheduler, pipeline, and module tests to prevent train/inference drift.
- **Latent scaling:** preserve VAE ownership and the unscaled-cache contract; check VAE, data, persistence, and pipeline tests.
- **Paired conditioning/pairing:** keep BraTS discovery outside the generic dataset contract and preserve subject-level split isolation.
- **Paired fidelity/evaluation:** preserve the `min_max_to_unit` → `PairedFidelityMetrics(data_range=1.0)` ordering and same-noise before/after contract across the pipeline, offline eval, and the active in-training `PairedFidelityCallback` monitor. A normalization, artifact, or report-schema change is a cross-component change, not a local patch.
- **Metrics:** distinguish per-rank accumulation from global reduction. Manual all-reduced metrics must not also use `sync_dist`, or they will be reduced twice.
- **Checkpoint behavior:** update training callbacks, export, downstream frozen-generator loaders, and tests as one contract.
- **Frozen arms:** new frozen-arm wiring MUST go through `FrozenArmMixin._register_frozen_arm`, not via `object.__setattr__` or any custom state-dict override — the mixin is the single owner of the register + dual-exclude contract (ADR-0031 A1). The frozen arms stay in `parameters()` (Lightning owns device placement) but carry no grad and emit no checkpoint key.
- **Per-rank device:** shells MUST resolve the per-rank CUDA device through `DevicePolicy.pin()` (pre-PG) and `DevicePolicy.warm_device(fallback)` (post-PG VAE warm); do not reintroduce the inline `set_device` twin, the bare `torch.device("cuda" if torch.cuda.is_available() else "cpu")` in the controlnet path, or `manifold.data.latent_pipeline.resolve_warm_device` (ADR-0035).
- **Callbacks:** new callbacks MUST go through the `CallbackRegistry` two-phase resolve/build; the `TrainingSpine.run` merge order is the single source of truth for which callbacks fire, which knobs apply, and which monitors are allowed (ADR-0029 / ADR-0032).

For component-level change navigation (entry points, focused tests, minimal validation), see the [Quickstart task routing](quickstart.md#task-routing) and the stage-level table in [Workflows change navigation](workflows.md#change-navigation).
