# Reward via partial-reconstruction labels — and measuring the train/inference gap with a generated-end probe

The GRPO reward model is trained on **preference pairs built from partial
reconstructions**: noise a clean latent to a flow-time `t`, denoise back to clean
with the frozen JiT denoiser, and label by corruption level — the **winner** is
lightly corrupted (`t_w ~ U[0.5, 1)`, near-clean), the **loser** heavily corrupted
(`t_l ~ U[0, 0.5)`, near-noise). The discriminator learns a realism/fidelity score
from this contrast (a GAN-as-reward). But GRPO scores **full generations** —
rollouts that start at pure noise (`t = 0`) and live entirely in the generated
regime, never near the clean data the winner half of a training pair approximates.
So the reward is *calibrated* on partial reconstructions but *consumed* on full
generations: a deliberate train/inference distribution gap.

This gap is accepted, not closed, and is **measured** rather than assumed: a
mandatory **generated-end validation probe** — a held-out pair set where *both*
samples are drawn from `t ∈ [0, 0.5)` and ordered by `t` — directly tests whether
the reward ranks quality *within the all-generated regime*, not just real-vs-
generated. Pairwise accuracy on reconstruction pairs alone could look healthy while
the reward is flat (uninformative) on actual generations; the probe catches that.

## Why

- **A usable, cheap label.** Reconstruction error to the clean latent is a
  ground-truth quality proxy that needs no human annotation; the corruption level
  `t` controls it continuously (per-sample `t` spans a smooth spectrum, so the
  discriminator learns a smooth realism function rather than a binary real/fake
  boundary). Both halves are denoised with the same step budget, so quality
  differences come from corruption level, not step count.
- **The gap is inherent to GAN-as-reward, not a bug.** The reward must be trained
  on *something* that distinguishes realistic from unrealistic latents.
  Reconstructions (near-clean) are the realistic anchor; the alternative — labelling
  full generations by their own quality — has no ground truth without a second
  scorer. Partial reconstruction is the cheapest signal that calibrates a usable
  realism gradient, at the cost of a regime shift at inference.
- **Measure the gap; fix only if it bites.** The probe makes the gap a *measured*
  quantity. If `val/gen_pair_acc` tracks reconstruction `val/pair_acc`, the reward
  generalizes to the generated regime and no further action is needed. If it lags,
  a deferred fallback (mix generated-regime `t ∈ [0, 0.5]` pairs into training, or
  a `t` dead-zone around 0.5 to reduce label noise) is held in reserve.

## Consequences

- Reward training optimizes a realism score whose training labels (partial
  reconstructions) are not drawn from the inference distribution (full
  generations). This is intentional; the generated-end probe is the load-bearing
  check that it still works.
- A second validation metric (`val/gen_pair_acc`, precomputed and reused across
  epochs — the denoiser is frozen, so probe pairs are constant) is mandatory, not
  optional, alongside the primary reconstruction pairwise accuracy and ROC-AUC.
- Precision/F1 are not reported: a fixed threshold is scale-arbitrary under
  Bradley–Terry (only `r_w − r_l` is constrained, not absolute `r`), so
  threshold-based metrics would be noise. Pairwise accuracy + ROC-AUC are
  threshold-free.
- The methodology is reward-specific; GRPO policy optimization (which *consumes*
  this reward) is out of scope for the reward-model effort and is tracked
  separately.
