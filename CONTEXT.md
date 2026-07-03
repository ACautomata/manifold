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
stack (over an unscaled cache; ADR-0003) — at inference it comes from the exported
native checkpoint (ADR-0006) instead. Owned by the VAE wrapper — `encode` returns scaled latents,
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
(`{unet_state_dict, scale_factor, ema}`) are no longer ingested — the one-shot
hope converter was retired once the migration completed (ADR-0007); training
checkpoints reach inference via export (ADR-0006).
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
shadow as the inference UNet — the sole bridge from a training checkpoint to the
inference Pipeline (ADR-0006), now that the hope→native converter is retired
(ADR-0007).
_Avoid_: convert (say export).

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

### Reward model (GRPO)

**Reward Model**:
The scorer that maps a latent to a scalar reward — a MONAI `PatchGANDiscriminator`
behind a thin pooling head, never a hand-rolled net. Its scoring `forward` is the
inference path GRPO calls during policy learning; it owns **no** generation rollout.
A **Model** in the four-component sense (a network wrapper), specialized to scoring.
_Avoid_: discriminator (that denotes the wrapped MONAI class itself), critic.

**Reward Module** (reward-training Module):
The `spt.Module` owning reward-training-only concerns — preference-pair
construction, the pairwise preference loss, and its optimizer wiring. In online
training (ADR-0010) it **holds the frozen JiT x0-denoiser** (unregistered, so off
the checkpoint/optimizer) and rolls fresh preference pairs each fit step
(*online rollout*); validation pairs + the generated-end probe are precomputed
once. Distinct from the **Reward Model** (the network it trains) and the JiT
**Module** (whose denoiser it consumes frozen).
_Avoid_: reward trainer, reward Lightning module.

**Online rollout** (reward):
The per-fit-step preference-pair rollout the Reward Module runs with the frozen
JiT denoiser: for each clean latent draw `t_a, t_b ~ U[0, 1)`, noise to each, and
partial-denoise both — winner = larger-`t` half, loser = smaller-`t` (label by
input corruption level). Both halves share the `[0, 1)` distribution, so the
winner/loser corruption ranges **overlap** (de-saturation), and the denoiser
produces no gradients. Distinct from the one-time **val/probe precompute** (also a
partial-denoise rollout, but run once at startup over held-out subjects).
_Avoid_: pair cache, online sampling.

**Preference pair** (winner / loser):
Two latents built by noising the same clean latent to a flow-time `t` then denoising
back to clean with the frozen JiT denoiser (ADR-0010). Both `t`'s are drawn from the
**full-range** `[0, 1)`; the **winner** is the larger-`t` (less-corrupted) half, the
**loser** the smaller-`t` — labeled by *input* corruption level, both denoised with
the same Heun step budget. `t` uses half-open `U[0, 1)` so `t = 1` is never sampled
(there the step-start denominator `1 − t` would vanish). Because the two `t`'s share
one distribution and are merely ordered, the winner/loser corruption ranges overlap —
no single threshold separates them. Built on the Scheduler's transport and `t→1 = clean`
convention (ADR-0001), so "more noise" is *smaller* `t`. (Pre-ADR-0010 these were
disjoint `[0.5, 1) / [0, 0.5)` halves, which saturated `val/pair_acc` at 0.997.)
_Avoid_: positive/negative sample (say winner/loser — those name the half-pair, not
a label).

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
