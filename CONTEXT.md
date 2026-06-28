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
timestep sampling, the `(1−t)⁻²` loss weight, and the MSE on the clean-latent
prediction. The stable-pretraining training interface.
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
The latent normalization scalar `1/std(z)` estimated from VAE latents. Owned by the
VAE wrapper — `encode` returns scaled latents, `decode` undoes the scaling — so the
Module and Pipeline never reference it (ADR-0003). A domain property: identical
across ranks.
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
