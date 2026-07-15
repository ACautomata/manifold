# Architecture and source map

## Component model

Manifold deliberately mirrors the diffusers vocabulary while keeping training and inference concerns separate (`CONTEXT.md`).

| Layer | Responsibility | Primary sources |
|---|---|---|
| Models | Thin wrappers around MONAI MAISI VAE/UNet implementations plus PatchGAN reward scoring. The VAE owns latent scaling and sliding-window encode/decode. | `src/manifold/models/` |
| Schedulers | Rectified-flow transport, timestep/sigma grids, model-input scaling, Heun integration, and stochastic GRPO/bridge transitions. They contain no training loop. | `src/manifold/schedulers/` |
| Modules | stable-pretraining training components: objectives, optimizer/schedule wiring, rollout and validation steps for JiT, paired JiT, reward, and GRPO. | `src/manifold/modules/` |
| Pipelines | Native inference composition of UNet, scheduler, and VAE, with `save_pretrained`/`from_pretrained`. | `src/manifold/pipelines/` |
| Training orchestration | CLI parsing, config composition, data warming, callbacks, Lightning trainer construction, checkpointing, and export. | `src/manifold/training/`, `src/manifold/metrics/` |

The shared rollout primitives are intentional: training-time sampling and native inference delegate to the same sampler behavior rather than maintaining parallel integrators (`src/manifold/modules/sampler.py`, `paired_sampler.py`; ADR-0005).

## Runtime flows

### Noise-to-data JiT

A frozen VAE encodes volumes into an unscaled cache. The data layer estimates a single scaling factor and applies it on read. `LatentFlowModule` trains the conditional UNet to predict the clean latent from interpolated noise, while `FlowMatchHeunDiscreteScheduler` owns transport and integration. `LatentFlowPipeline` starts inference from Gaussian noise, applies optional interval-restricted classifier-free guidance, integrates from `t=0` to `t=1`, and decodes the result.

Start with:

- `src/manifold/data/latent_pipeline.py`, `latent_dataset.py`, `warm_datamodule.py`
- `src/manifold/modules/latent_flow.py`, `sampler.py`
- `src/manifold/schedulers/scheduling_flow_match_heun.py`
- `src/manifold/pipelines/latent_flow.py`
- `src/manifold/training/cli.py`

### Paired JiT

Paired JiT maps a source latent at `t=0` to a target latent at `t=1`. The model sees `concat([z_t, x_src])`, so its input channel count is twice the latent channel count. Source and target contrast embeddings are summed to encode the requested translation. The production recipe uses uniform `x0` MSE; the earlier `(1-t)^-2` weighting overemphasized high-`t` examples and encouraged copy-source collapse (`configs/train/config_paired_jit.yaml`).

BraTS-specific code groups volumes by subject and contrast, creates subject-disjoint splits, and enumerates all ordered non-self pairs. The dataset contract itself remains generic: source/target latents, labels, and spacing.

Start with:

- `src/manifold/data/paired_brats.py`, `paired_volume_dataset.py`, `paired_latent_dataset.py`
- `src/manifold/modules/paired_latent_flow.py`, `paired_sampler.py`
- `src/manifold/pipelines/paired_latent_flow.py`
- `src/manifold/training/paired_cli.py`

### Reward and policy post-training

`RewardModel` wraps a MONAI PatchGAN discriminator and pools its output to a scalar. Noise-to-data reward training compares generated preference pairs; paired reward is condition-aware and scores `concat([x_src, tgt])`, so it can penalize a realistic but incorrect copy of the source. GRPO modules fork individual stochastic transitions, score terminal outcomes, and optimize a clipped group-relative objective. The paired variant uses the Brownian-bridge scheduler only during training.

Start with `src/manifold/models/reward_model.py`, `src/manifold/modules/{reward,paired_reward,grpo,paired_grpo}.py`, and the corresponding `src/manifold/training/*_cli.py` files.

## Configuration and persistence

Experiment YAML is composed by `src/manifold/config/loader.py` and built into components by `builder.py`. Later top-level blocks replace earlier ones unless `_base_` explicitly requests inheritance. This launch-time OmegaConf layer is separate from persisted component JSON handled by `src/manifold/configuration.py`.

Native inference directories contain component configuration/weights (including `model_index.json` and component subdirectories). Lightning `.ckpt` files are training state and are not loaded directly by pipelines; export is the bridge. See [Checkpoint and export contract](workflows.md#checkpoint-and-export-contract).

## Change guidance

- **Transport/integration:** change the scheduler and shared sampler path together; run scheduler, pipeline, and module tests to prevent train/inference drift.
- **Latent scaling:** preserve VAE ownership and the unscaled-cache contract; check VAE, data, persistence, and pipeline tests.
- **Paired conditioning/pairing:** keep BraTS discovery outside the generic dataset contract and preserve subject-level split isolation.
- **Metrics:** distinguish per-rank accumulation from global reduction. Manual all-reduced metrics must not also use `sync_dist`, or they will be reduced twice.
- **Checkpoint behavior:** update training callbacks, export, downstream frozen-generator loaders, and tests as one contract.
