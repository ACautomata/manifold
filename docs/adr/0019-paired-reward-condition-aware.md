# Condition-aware paired reward — score concat([x_src, tgt]), in_channels = 2·C_latent

The paired reward discriminator scores the **concatenation** `concat([x_src, tgt])`
along channels — i.e. it is constructed `RewardModel(in_channels = 2·C_latent)` and
the **caller** concats before scoring; `RewardModel.forward` is unchanged (it is
channel-agnostic; `in_channels` is ctor-only). The reward therefore asks "is this
tgt a faithful translation *of* this src?", not "is this tgt a realistic latent?"

## Why

- **An unconditional realism reward rewards copy-src.** `x_src` is itself a real VAE
  latent, so a discriminator scoring `tgt` alone sees a copied source as maximally
  real — it would push GRPO toward copying the source, which is the documented
  dominant paired-JiT failure (`model ≈ copy-src`, the PSNR-ceiling regime;
  ADR-0013/0014). Conditioning on `x_src` lets the discriminator learn that
  `tgt == src` is not a legitimate translation (for cross-contrast pairs where they
  must differ) and catch the shortcut.
- **It mirrors the paired UNet's own conditioning.** The UNet consumes
  `concat([z_t, x_src])` with `in_channels = 2·C_latent` (ADR-0014); the reward
  scoring `concat([x_src, tgt])` is the symmetric choice — both put `x_src` in view.
- **Concat is the minimal, zero-model-code change.** `RewardModel.forward` takes a
  single latent and is channel-count-agnostic; doubling `in_channels` to `2·C_latent`
  and having the caller concat changes no model code, no head, and (concat being
  channel-wise) does not change the PatchGAN spatial-collapse risk.
- **`x_src` is free at every consume site.** The rollout starts from `x_src`, so it
  is available at reward-training, probe, and GRPO-scoring time.
- **Rejected — unconditional (`in_channels = C_latent`):** zero new parameters but
  rewards copy-src — a broken reward for paired-GRPO.
- **Rejected — explicit `forward(latent, src=None)` kwarg:** more invasive, and
  inconsistent with the existing channel-agnostic single-arg `forward` and the UNet's
  caller-concat precedent.

## Consequences

- The reward contract is **conditional**: it consumes `(x_src, tgt)`, not `tgt`
  alone. The future Paired-GRPO must score `concat([x_src, z_K])` at both consume
  sites (the JiT-GRPO sites at `grpo.py:223/581` pass only `z_K` today). Because
  paired-GRPO does not yet exist, locking this now is zero-breakage.
- A distinct `RewardModel` config (`reward_model.in_channels = 2·C_latent`); the
  noise→data JiT reward keeps `in_channels = C_latent`.
- The discriminator's task is harder (learn src→tgt correspondence, not just
  realism) — accepted, because the unconditional alternative is broken.
