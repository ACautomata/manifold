# Manifold

Diffusers-style latent-flow generation for 3D medical volumes. Four decoupled
components — Models, Scheduler, training Module, inference Pipeline — mirror the
diffusers layout; training stays on stable-pretraining, backbones stay on MONAI
MAISI. The codebase *mimics* diffusers' structure and naming with its own base
classes; it does not subclass the `diffusers` library.

## Language

### The four components

**Model**:
A diffusers-style network wrapper under `models/`. Today the VAE
(`AutoencoderKlMaisi`) and the UNet (`DiffusionModelUNetMaisi`); both *wrap* MONAI
MAISI classes behind a thin adapter, never reimplemented.
_Avoid_: network, backbone (those denote the wrapped MONAI class itself).

**Scheduler**:
The component under `schedulers/` owning the rectified-flow transport and the
reverse-time step: `add_noise`, `set_timesteps`, the sigmas/timesteps grid,
`scale_model_input`, the Heun `step`, and `prediction_type="sample"` (x0). Owns
no training concern — deliberately, so it is reusable as the single source of
truth for the forward process.
_Avoid_: noise scheduler (MONAI-flavored), sampler.

**Module** (training Module):
The `spt.Module` under `modules/` owning training-only concerns: logit-normal
timestep sampling, the `(1−t)⁻²` loss weight, the MSE on the clean-latent
prediction, the optimizer/LR-schedule/grad-norm wiring, and the x0 Heun
`sample()` it delivers for in-training generation (the same rollout primitive the
inference Pipeline delegates to — ADR-0005). The stable-pretraining training
interface.
_Avoid_: trainer, Lightning module.

**Pipeline**:
The component under `pipelines/` that turns noise + conditions into a decoded
volume: latent → Heun rollout → sliding-window VAE decode. Returns a decoded
volume tensor `[B,C,D,H,W]` (NIfTI writing is a separate saver); holds the UNet,
scheduler, and VAE; loads/saves via `from_pretrained` / `save_pretrained`.
_Avoid_: inferer, generator.

### The JiT latent flow

**JiT** (x0-denoiser):
The latent-flow formulation where the network predicts the clean latent `x0`,
trained with a `(1−t)⁻²`-weighted MSE and sampled by a Heun reverse-step. From
*"Back to Basics: Greatly Improved Image Generation."* **Not** `torch.jit`.
_Avoid_: recipe, denoiser mode, x0 mode.

**Transport**:
The forward noising `z = t·x + (1−t)·e`, `t∈[0,1]`, `t→1` is clean data. The one
forward process the Scheduler and the Module must agree on.
_Avoid_: forward process, noise schedule, corruption.

**scale_factor**:
The latent normalization scalar `1/std(z)` estimated from VAE latents by the data
stack (over an unscaled cache; ADR-0003) — at inference it comes from the converted
checkpoint instead. Owned by the VAE wrapper — `encode` returns scaled latents,
`decode` undoes the scaling — so the Module and Pipeline never reference it. A
domain property: identical across ranks.
_Avoid_: latent scale, norm factor.

**conditioning**:
The spacing tensor + class-label vector the UNet consumes — medical geometry and
modality, not text embeddings.
_Avoid_: context, encoder hidden states (those are diffusers text-conditioned terms).

### Persistence

**Checkpoint** (native):
A per-component manifold checkpoint, read/written by
`Pipeline.from_pretrained` / `save_pretrained`. Old hope flat checkpoints
(`{unet_state_dict, scale_factor, ema}`) are not read directly — they pass through a
one-shot converter (ADR-0003).
_Avoid_: weights file, model file.

### Training

**Training pipeline**:
The `manifold-train` console entry + Lightning `Trainer` + `spt.Module` + callbacks that turn
warmed latents into a trained checkpoint. Built on `spt.Module` (manual
optimization), never a hand-rolled loop. Owned under the `training/` package
(cli + trainer + EMA + train-metrics callbacks + export) and `metrics/` (the FID).
_Avoid_: train script, run script.

**Training checkpoint** (.ckpt):
The Lightning `ModelCheckpoint` artifact — full training state (UNet + optimizer +
LR-schedule + EMA callback state + epoch); the resume and best-by-FID medium.
Distinct from the **native** checkpoint: training writes `.ckpt`, the inference
Pipeline reads the native per-component dir (reached via export; ADR-0006).
_Avoid_: weights file (say training-ckpt or native).

**Export**:
The one-shot `.ckpt → native` per-component conversion that bakes the slowest EMA
shadow as the inference UNet — the bridge from a training checkpoint to the
inference Pipeline. Distinct from the hope→native converter.
_Avoid_: convert (say export or hope-converter).

**Fixed-sample validation**:
In-training evaluation that reuses the SAME small validation subset AND re-seeds
the generation noise every epoch, so only the model changes between epochs —
isolating quality drift from sampling stochasticity.
_Avoid_: rolling validation, random-sample eval.

**Unbiased FID**:
The small-sample-bias-corrected Fréchet distance — subtracts the exact
`Tr(Σ)/n` upward bias of the `‖μ̂₁−μ̂₂‖²` mean term. The validation FID
estimator; RadImageNet ResNet50 backbone, 2.5D (three orthogonal planes).
Distinct from the legacy biased plug-in FID.
_Avoid_: biased FID, plain FID.

### Configuration

**Experiment config**:
The OmegaConf run-driver — env (paths) → train/inference recipe → network-
construction kwargs, composed top-level (later files replace earlier whole, with
`_base_` inheritance), required paths as `???` (fail-fast) and CLI/dotlist
overrides. Builds components at launch; never persists them (ADR-0004).
_Avoid_: run config, hydra config, "the config".

**Component config**:
The JSON `config.json` a component writes via `register_to_config` / `ConfigMixin`
and `from_pretrained` / `save_pretrained` round-trips — the diffusers-style
persistence contract for a trained component, independent of how it was launched
(ADR-0004).
_Avoid_: model config, "the config" (say which — Experiment or Component).
