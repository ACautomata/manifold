# ControlNet two-stage: supervised pre-training then GRPO Mode-2

The ControlNet is first **supervised-pre-trained**, then **GRPO post-trained** —
mirroring the JiT two-stage (`LatentFlowModule` → `GRPOModule`).

1. **Supervised** — a new `ControlNetLatentFlowModule` (`modules/controlnet_latent_flow.py`)
   holds the frozen base UNet + the trainable ControlNet, runs the **noise→data**
   transport (`z = t·x_tgt + (1−t)·ε`), and trains **only the ControlNet** with the
   `(1−t)⁻²`-weighted x0-MSE on `x_tgt` (the velocity-MSE the base JiT was trained
   with), the ControlNet conditioned on `x_src`. Optional `l1_weight` knob (default 0).
   This produces a real src→tgt translation policy.
2. **GRPO Mode-2** (ADR-0028) — freeze the base, continue training only the ControlNet
   against the condition-aware reward, initialized from the supervised checkpoint.

The ControlNet is **warm-started** from the base UNet's encoder weights + zero-init
zero-conv (ADR-0026). Replaces the supervised Paired JiT (ADR-0013/0014, superseded/retired).

## Why

- **GRPO needs a meaningful starting policy — RL from scratch has no signal** (the JiT
  lesson: a random/blank denoiser's rollouts are indistinguishable to the reward, so the
  group-relative advantage collapses). A supervised ControlNet that already translates
  src→tgt gives GRPO a non-trivial baseline; the reward then pushes realism/fidelity
  *past* the MSE ceiling. (User decision: supervised stage is required.)
- **Velocity-MSE `(1−t)⁻²` matches the base.** The frozen base was trained with this
  loss, so the ControlNet learns residuals compatible with the base's x0-prediction.
  L1 is held as an optional knob (Paired JiT memory: uniform-MSE + 0.5·L1 +
  direction_offset was the src→tgt ceiling-winner, but that transport is gone — start
  velocity-MSE alone, add L1 empirically).
- **The supervised policy dissolves the reward chicken-egg (ADR-0024).** The reward's
  fake cache now comes from the supervised ControlNet — a real translator — so the
  within-fake ranking the reward needs is naturally available. The bridge-noise launch
  gate (ADR-0024) demotes from blocking to an optional sanity probe.
- **`x_src` is the control, `x_tgt` is the target — the paired dataset is reused as-is.**
  `PairedLatentDataset` already emits `{src_latent, tgt_latent, src_label, tgt_label,
  spacing}`; the supervised stage consumes `src_latent` as the ControlNet condition and
  `tgt_latent` as the x0 target. The transport changes (noise→data, not src→tgt
  interpolation); the data does not.

## Considered options (rejected)

- **Skip supervised, GRPO the ControlNet from the zero-conv (unconditional) init:**
  rejected — cold-start RL; the ControlNet must discover "inject `x_src`" purely from
  the reward, a fragile signal-death/regime with no precedent in this codebase (the
  user's initial lean, corrected after grilling).
- **Reuse the (deleted) Paired JiT weights as ControlNet init:** rejected —
  architecturally incompatible (Paired JiT is a `2·C` full UNet, not an adapter on a
  frozen base); weights do not transfer.
- **Auxiliary MSE in GRPO Mode-2:** held in reserve (not v1) — GRPO's purpose is to
  escape the supervised PSNR ceiling; an aux-MSE would anchor back to it (ADR-0024's
  rationale carries over). The KL-to-init anchor (ADR-0015) is the first anti-hacking
  escalation.

## Consequences

- New `ControlNetLatentFlowModule` composes `LatentFlowModule`'s helpers (`t_sampler`,
  optimizer/LR/grad-norm) by composition, not inheritance (project OOP rule); it freezes
  the base UNet's parameters (no grad, off the optimizer) and trains only the ControlNet.
- Validation: deterministic noise→data Heun + ControlNet residuals
  (`pipelines/controlnet_latent_flow.py`), PSNR/SSIM via the `PairedPSNRSSIMCallback`
  pattern, fixed-sample re-seed (the noise→data analogue of Paired JiT's determinism —
  here reproducibility comes from re-seeded generation noise, since the `t = 0`
  endpoint is Gaussian again).
- **No EMA (raw arm).** ~~the ControlNet carries an EMA arm for validation (mirrors the
  JiT supervised EMA).~~ **Corrected 2026-07-19:** the "JiT supervised EMA" this line
  referenced no longer exists — EMA training was removed repo-wide by ADR-0006
  (2026-07-14); every supervised module, the export, and the validation callbacks run
  the **raw optimizer arm**. The ControlNet supervised stage follows suit: validation,
  checkpoint selection, and the export all use the raw ControlNet weights (no EMA
  shadow). GRPO Mode-2 also trains raw (ADR-0012).
- **Export:** a native ControlNet artifact `{unet (frozen base state), controlnet,
  scaling_factor}` via a new `training/export.py` arm.
  `load_frozen_controlnet_generator(native_dir)` returns
  `(base_unet, controlnet, scheduler, scaling_factor)` — the single loader the reward
  fake-builder and GRPO Mode-2 init both consume (replaces ADR-0021's
  `load_frozen_paired_generator`).
- **Launch gate into GRPO Mode-2:** (1) supervised `val/psnr` above a floor; (2) reward
  `train_acc > 0.7` (real-vs-fake separable); (3) optional bridge-noise sanity probe
  (ADR-0024, demoted). All pass before the real `--max-epochs` of GRPO.
