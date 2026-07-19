# ControlNet on the frozen JiT UNet via MONAI's native residual-injection forward args

Paired MRI generation is rebuilt as a **canonical ControlNet** — a trainable clone of
the noise→data JiT UNet's encoder (`conv_in` + input-embedding path + `down_blocks` +
`middle_block`) plus zero-conv layers — whose residuals are injected into the
**frozen** pretrained JiT UNet through the MONAI `DiffusionModelUNetMaisi`'s native
forward args `down_block_additional_residuals` / `mid_block_additional_residual`. The
MAISI UNet is a diffusers-port that already exposes these args (verified:
`monai/apps/generation/maisi/networks/diffusion_model_unet_maisi.py` `forward` L368,
applies down residuals L348-353 and the mid residual L405-406) and the named encoder
submodules (`conv_in`, `time_embed`, `class_embedding`, `spacing_layer`,
`down_blocks`, `middle_block`). No MONAI fork, no monkey-patch, no input concat into
the base. This replaces the src→tgt Paired JiT transport (ADR-0013, superseded) and
its `2·C_latent` concat conditioning (ADR-0014, retired); the transport reverts to
noise→data (`z = t·x_tgt + (1−t)·ε`), with `x_src` demoted from the `t = 0` endpoint
to a control signal.

## Why

- **The MAISI UNet is not a sealed black box.** It is a faithful diffusers-port whose
  `forward` already accepts the ControlNet residual lists and whose encoder is built
  from the same public MONAI block builders (`get_down_block`, `get_mid_block`,
  `Convolution`) the base UNet uses internally. A canonical ControlNet — a mirrored
  trainable encoder + zero-conv — is therefore feasible by *composing* MONAI
  components, never reimplementing them (ADR-0001 honored).
- **Input concat is incompatible with a frozen base.** The frozen base `conv_in`
  expects `C_latent` channels; the control signal cannot enter the base by concat
  without unfreezing `conv_in`. Residual injection via the base's native forward args
  is the only way to keep the base **byte-frozen** while adding a trainable control
  pathway. (The control instead enters the *ControlNet* by an additive cond embedding
  after the ControlNet's own `conv_in`, so the ControlNet's `conv_in` stays clone-able
  from the base.)
- **Warm-start is the point, and ControlNet makes it safe.** Reusing the pretrained
  noise→data JiT UNet frozen (a strong realism prior) + a lightweight trainable adapter
  is the entire reason to switch. This reverses ADR-0014's warm-start rejection —
  justified because the adapter leaves the base frozen, whereas ADR-0014 rejected
  warm-starting a `2·C` *finetuned* UNet. Zero-init zero-conv means the initial
  residuals are zero, so the model's initial behavior is the pretrained JiT UNet
  unchanged (a safe warm-start).
- **`x_src` is a latent, so the cond embedding is cheap.** `x_src` is already a scaled
  VAE latent in the same `C_latent` space as `z_t`; a single conv maps it to the
  `conv_in` output width and adds it post-`conv_in` (the diffusers `ControlNetModel`
  precedent). No source-encoder subnetwork, no cross-attention surgery.

## Considered options (rejected)

- **Input concat into the base (`concat([z_t, x_src])`, ADR-0014 style):** rejected —
  the base is frozen at `C_latent`; concat would require unfreezing `conv_in`, breaking
  the freeze constraint. (Concat *inside the ControlNet* conv_in is also rejected for
  warm-start: it makes the ControlNet `conv_in` non-cloneable.)
- **Fork MONAI to expose internals / monkey-patch forward hooks:** rejected — violates
  ADR-0001 and is unnecessary; the residual-injection interface is already public.
- **Custom 3D UNet written from scratch:** rejected — violates ADR-0001 and discards
  MAISI's medical conditioning (`spacing` + `class_labels`).
- **Cross-attention / FiLM conditioning:** rejected — needs a new source encoder or
  invasive per-block surgery; the residual-injection ControlNet is the proven,
  minimal-surgery choice.

## Consequences

- New model wrapper `ControlNet3DConditionModel` (`models/controlnet_3d.py`). It composes
  MONAI components matching the base config: a `conv_in` **cloned from the base**
  (takes `z_t`, `C_latent`), a `controlnet_cond_embedding` (new conv: `x_src` →
  `block_out_channels[0]`, added to the `conv_in` output), the cloned input-embedding
  path (`time_embed`, `class_embedding`, `spacing_layer`), `down_blocks` and
  `middle_block` (via the same MONAI builders), and **zero-conv** layers (zero-init)
  producing the down/mid residual lists. It carries the direction-conditioning head
  (ADR-0028). `forward` returns
  `(down_block_additional_residuals, mid_block_additional_residual)`; the caller passes
  them to the frozen base UNet's forward.
- The base wrapper `models/unet_3d_condition.py` is **cleaned**: `paired_direction_offset`,
  `paired_cond_mlp`, and `_PinnedClassEmbedding` are removed (the direction head moves to
  the ControlNet, ADR-0028). The noise→data call path is byte-unchanged.
- **Initialization = warm-start.** Load the base UNet's encoder weights
  (`conv_in`, `time_embed`, `class_embedding`, `spacing_layer`, `down_blocks`,
  `middle_block`) into the matching ControlNet submodules; zero-init the zero-convs;
  small-init the `controlnet_cond_embedding` and the direction MLP. Initial residuals = 0
  ⇒ initial behavior = the pretrained JiT UNet.
- Inference: noise→data Heun rollout where each UNet eval is
  `base(z_t, t, spacing, class_labels=tgt_label, down_block_additional_residuals=…,
  mid_block_additional_residual=…)` with the residuals from the ControlNet (ADR-0027
  pipeline).

## Hazard correction (verified 2026-07-19, MONAI 1.6): in-place residual adds bite the **supervised** stage too

The Step-1 handoff recorded the in-place hazard as *supervised-safe* — "safe when the
base is frozen AND its input `z_t` is a no-grad data tensor (the supervised regime);
breaks only when `z_t` requires grad (the GRPO perturbed step)". **That boundary is
wrong.** Direct probes (this branch, MONAI 1.6/CPU) show the MONAI backbone's in-place
residual adds break **any** backward that flows from the base output back to the
ControlNet — which is exactly the supervised stage's `(1−t)⁻²`-MSE on `x_tgt`, not just
GRPO:

- `_apply_down_blocks` does `down_block_res_sample += down_block_additional_residual`
  **in place**. The `down_block_additional_residual` is the ControlNet's *grad-bearing
  output tensor*; mutating it in place bumps its autograd version, so the ControlNet's
  own backward (which saved that tensor's pre-add value) raises *"one of the variables
  needed for gradient computation has been modified by an inplace operation … version 1;
  expected version 0"*. This fires even with `mid_block_additional_residual=None` and
  even when `z_t` requires no grad.
- `forward` does `h += mid_block_additional_residual` in place on the `middle_block`
  output `h`, which the up-block backward needs — same version-error mechanism.

The reason Step-1's "gradient routing verified" claim passed is that its check used a
loss that did **not** flow through the base output (a synthetic loss on the residual
tensors directly). The real supervised loss — `MSE(base_output_with_residuals, x_tgt)` —
does flow through the base output, and there the in-place adds are fatal.

**Fix (implemented):** the base wrapper `UNet3DConditionModel` runs an **out-of-place**
re-implementation of the backbone forward on the residual-injection path
(`_forward_with_residuals`): the two in-place adds become `sample + residual` and
`h = h + mid_block_additional_residual`. The forward output is **bit-identical** to
MONAI's native forward (verified), the frozen base's parameters are untouched, and the
noise→data path (no residuals) still calls MONAI's own forward unchanged. The backbone
remains composed, never subclassed (ADR-0001); only the injection path is re-implemented.
The supervised stage, two-mode GRPO (Mode-2 perturbed step), and the inference rollout
all route through this single out-of-place path, so the hazard is neutralized once for
all consumers.

