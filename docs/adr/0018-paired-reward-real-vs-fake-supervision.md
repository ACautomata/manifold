# Paired-JiT reward supervision — real tgt vs generated tgt (real-vs-fake), not corruption-level

> **Status: Superseded by [ADR-0034](0034-one-realism-reward-both-grpo-policies-delete-condition-aware.md).**
> The real-vs-fake paired reward and its pipeline are deleted; both GRPO policies share
> the unconditional realism reward (ADR-0009/0010). Kept as decision history.

The paired-JiT reward model is trained on **real-vs-fake preference pairs**: the
**winner** is the real target latent `x_tgt` (the VAE encode of the target volume),
the **loser** is the paired-JiT model's full src→tgt generation
(`sample_paired_latent_flow(x_src, …)`). This departs from the JiT reward model,
whose winner/loser are **both generated** (partial-denoise rollouts) labeled by input
corruption level `t` (ADR-0009/0010) — real latents there are only the seed for
corruption, never the positive. The generated-end **probe** is retained as the
ADR-0009 "measure-the-gap" insurance.

## Why

- **The paired generator is the imperfect current policy, so real/fake is a real
  signal.** The frozen paired-JiT UNet (the GRPO starting policy) sits around
  PSNR ≈ 26 dB — its src→tgt translations are genuinely separable from the real
  `x_tgt`, so a PatchGAN can learn a non-trivial real/fake boundary. In the JiT
  setting the frozen denoiser's partial reconstructions are near-clean, so a real
  latent and a reconstruction are nearly identical — real/fake would carry no signal,
  which is why JiT manufactured a within-fake gradient via corruption level instead.
  That trick is unneeded for paired.
- **Real tgt is the natural positive for translation fidelity.** The reward must
  score "is this a faithful target for this source?" — the real `x_tgt` is the
  ground-truth answer; the model's generation is the candidate. This is the standard
  GAN-as-reward framing, and it reuses `PairedLatentDataset` directly (it already
  emits `{src_latent, tgt_latent, src_label, tgt_label, spacing}`).
- **The probe is kept, not the corruption-level training.** ADR-0009's lesson
  transfers in *form* (a generated-end probe measures whether the reward ranks within
  the all-fake regime GRPO consumes) even as the *training* supervision changes. If
  the probe lags `val/pair_acc`, the documented fallback (mix within-fake pairs into
  training) is held in reserve.
- **Rejected — corruption-level (mirror JiT exactly):** both halves partial-paired
  rollouts ordered by `t`. Guarantees a within-fake gradient by construction, but
  discards the real-tgt-as-positive spec and needs a new partial-paired-rollout
  primitive for *training* (not just the probe). Real/fake already carries signal,
  so the extra machinery is unjustified for the primary objective.

## Consequences

- The reward's winner is a real latent and its loser is a generated latent — both in
  scaled latent space; the generated tgt is never VAE-decoded/re-encoded (that
  destroys the real-vs-generated signal the reward measures).
- `val/pair_acc` (real > generated) and `val/roc_auc` will saturate near 1.0 (a
  PatchGAN trivially separates a flawed generation from a real latent) — they are
  **diagnosis-only**. The within-fake probe is the load-bearing metric.
- The supervision is orthogonal to whether the discriminator conditions on `x_src`
  (ADR-0019) and to the offline precompute cadence (ADR-0020) — those are separate
  decisions.
