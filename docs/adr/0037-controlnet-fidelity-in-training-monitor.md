---
status: accepted
---

# ControlNet paired fidelity is also monitored in-training — observe-only PSNR/SSIM on a fixed paired subset

ADR-0036 scoped 3D PSNR/SSIM to an **offline before/after eval**, "never during training" —
but only *for the comparison use-case* ("rejected for the comparison use-case"). That left the
supervised ControlNet stage with a single latent surrogate (`val/x0_mae`) and the ControlNet-GRPO
stage with a single fidelity-blind scalar (`val/mean_reward`): no live read on whether the policy
is actually learning to translate toward the real `tgt`. We decide: paired fidelity is **also**
observed **in-training**, via a generative validation callback on the supervised ControlNet path
that, each gated validation epoch, rolls a **fixed paired subset** (a small fixed set of paired
`(src, tgt)` subjects under fixed initial noise — the fixed-sample-validation ethos) through the
module's own `sample()` (Heun `controlnet_rollout`) → VAE decode → per-sample min-max to `[0,1]`
→ MONAI 3D PSNR/SSIM, and logs `val/psnr` / `val/ssim`. It is **observe-only**: never a checkpoint
monitor (that stays `val/x0_mae`) and never a loss term.

## Why

- **Same metric, same口径 as the offline comparison.** The monitor runs the *full* Heun rollout
  and the *same* normalization + MONAI metrics as ADR-0036's offline pass, so the in-training
  curve and the before/after number are directly comparable — a cheaper single-step x0-decode
  proxy was rejected (below).
- **The architecture already anticipated it.** The callback registry names its generative family
  "FID, future PSNR/SSIM" — this is that planned second generative callback, mounted exactly like
  `FIDSpec` (a registered spec declaring `logged_metrics`, built through `CallbackContext`).
- **Observe-only matches the fidelity philosophy.** ADR-0034 keeps the realism reward
  fidelity-blind; we likewise keep fidelity out of the optimization — a screen, not a gradient.
  Checkpoint selection stays on the existing `val/x0_mae`.
- **Not subject to the FID ban.** The ControlNet-GRPO path forbids FID because FID's
  *unconditional* rollout ignores the ControlNet — a "constant frozen-base metric." Paired
  fidelity rolls `controlnet_rollout`, which *uses* the ControlNet, so that rationale does not
  apply. This is the basis for the GRPO-stage extension (a deliberate follow-up, below).

## Considered options (rejected)

- **Single-step x0-decode proxy** (decode the `LatentX0MAE` pred/target latents, no rollout):
  rejected — it measures one-step denoising reconstruction, not the deployed full-generation
  translation; exposure bias can look healthy while the rollout is not, and it diverges from the
  offline口径.
- **Drive checkpoint selection / early-stop with PSNR/SSIM:** rejected for now — observe-only
  first; the signal must be watched before it is trusted to select.
- **Shard the subset across ranks (FID-style):** rejected at the small fixed subset size — every
  rank redundantly evaluates the same fixed subset (DDP-synchronized weights + identical input ⇒
  identical result), so reduction is just Lightning's torchmetrics sync and the whole FID
  sharding / error-rendezvous machinery is unnecessary. Revisit only if the subset grows.

## Consequences

- **Supervised stage (this ADR).** A new registered spec + callback; it reuses the VAE-only
  `VaeStage` (the VAE-staging path extracted from, and now composed by, `VramStage`: VAE CPU→GPU
  for the decode, then back to CPU), `LatentDecoder`, `min_max_to_unit`, and
  `PairedFidelityMetrics`. The `controlnet.num_inference_steps` config knob (default 15 ⇒ 29 UNet
  evals) already exists for this "validation rollout"; a FID-style `every_n_epochs` gate throttles
  cadence. Fills the currently-`None` `CallbackContext.inference_recipe` on the supervised path.
- **DDP.** All ranks run it (ADR-0025); the fixed redundant subset makes collective-count
  invariance (ADR-0030) trivial — no custom `all_reduce` beyond Lightning's metric logging.
- **GRPO stage (deliberate follow-up, not in this change).** Extending the monitor to the
  ControlNet-GRPO path means releasing the blanket `forbidden_callbacks={"fid"}` for *this*
  callback specifically (its rationale does not cover paired fidelity) and keeping the VAE on GPU
  for the decode. That is a separate ticket/ADR; the supervised stage lands first.
- **ADR-0036 unchanged.** The offline before/after comparison stays the apples-to-apples pairing
  of record; this monitor is a complementary per-epoch drift screen, not a replacement.
