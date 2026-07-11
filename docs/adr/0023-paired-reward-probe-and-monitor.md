# Paired reward generated-end probe — partial-paired rollout; monitor val/gen_pair_acc

> **Paired analogue of the [ADR-0009](0009-reward-partial-reconstruction-labels-and-generated-end-probe.md)
> generated-end probe.** The probe is the only metric that tests graded within-fake
> ranking — what GRPO consumes (it ranks among all-fake policy siblings). Under
> real-vs-fake supervision (ADR-0018), `val/pair_acc` (real>fake) and `val/roc_auc`
> saturate near 1.0 and are diagnosis-only.

The paired reward ships a **generated-end probe** built on a new
`partial_paired_rollout`: both samples start from
`z = scheduler.add_noise(x_tgt, x_src, t_start)` at `t_start ∈ [0, 0.5)`, rolled to
`t = 1` by the paired Heun; **winner = higher `t`** (less translation → closer to real
tgt → higher quality). The checkpoint monitors `val/gen_pair_acc` (`mode=max`),
mirroring the JiT reward (ADR-0010).

## Why

- **The probe is mandatory — without it the reward is blind to GRPO.** `val/pair_acc`
  saturates; only an all-fake probe tests whether the reward ranks within-fake quality.
- **Translation-progress is the faithful quality axis.** Starting closer to tgt (higher
  `t`) is easier → output closer to real tgt. This directly measures translation
  fidelity (what GRPO optimizes), unlike a `num_steps` (integration-precision) axis.
- **`partial_paired_rollout` is ~90% reuse.** It is `sample_paired_latent_flow` with two
  changes: start `z = add_noise(x_tgt, x_src, t_start)` (not `x_src`), and per-sample
  `(B,)` `t` from `set_timesteps_partial` (not scalar `nodes[i]`). `add_noise` and
  `set_timesteps_partial` are endpoint-agnostic → reusable for the src/tgt transport.
- **Cap `t_start ∈ [0, 0.5)`.** High `t` starts near real `x_tgt` → probe samples leak
  toward the positive class → the probe degenerates to real-vs-fake. `[0, 0.5)` keeps
  probe samples genuinely fake (matches the JiT probe range).
- **`(B,)`-t through the paired summed-label pathway is the one risk — gated by a
  parity test.** `sample_paired_latent_flow` passes only scalar `t`;
  `partial_denoise_rollout` passes `(B,)` `t` through the JiT single-label path. The
  time_embed and class_embed pathways are independent, so `(B,)`-t + summed labels
  *should* work, but it is an untested combination. A small parity test (scalar-t vs
  `(B,)`-t rollout equality) gates the primitive; if it surprisingly fails, fall back to
  a `num_steps` axis (fewer Heun steps = worse fake; reuses
  `sample_paired_latent_flow`, no parity risk).
- **`inference_mode` is fine for the probe** (correcting an over-cautious note): all
  rollouts are precomputed to disk (ADR-0020) and loaded as clean tensors (the flag does
  not survive the disk round-trip), and the probe is scored forward-only in validation.
  The `no_grad` requirement applies only to the future paired-GRPO (which backprops
  through the rollout) — out of scope for the reward.
- **Monitor `val/gen_pair_acc`** (`mode=max`). It is the metric we care about
  (within-fake ranking) and the one that will not trivially saturate (training does not
  directly optimize it). If it plateaus low, that is the ADR-0009 escalation signal
  (mix fake-vs-fake pairs into training — the ADR-0018 reserve). The DDP fallback (drop
  the monitor, keep `save_last` + `save_top_k=1`) mirrors the JiT reward.

## Consequences

- New `partial_paired_rollout` in `paired_sampler.py` (ADR-0005 single-source-of-truth);
  the probe path constructs `PartialFlowMatchHeunScheduler` explicitly (only the subclass
  has `set_timesteps_partial`).
- The probe is precomputed once over paired-val subjects (slow-EMA generator frozen ⇒
  static) and reused across epochs — exactly the JiT probe lifecycle.
- `_build_checkpoint` monitors `val/gen_pair_acc` (single-GPU); DDP multi-GPU drops the
  monitor (rank-local selection unreliable), keeping `save_last` + `save_top_k=1`.
