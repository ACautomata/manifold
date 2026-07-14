# GRPO Module: `training_step` override, multi-step inner loop, raw / no-EMA

> **Partial supersede (2026-07-06): the "No KL for v1" decision below was reversed by
> [ADR-0015](0015-grpo-kl-anchor-and-bounded-reward.md).** v1 hacked in one epoch
> (`val/mean_reward` −13.75 → 3370, `val/fid` 3.98 → 12.50), proving the FID-only screen
> insufficient against an unbounded raw-logit PatchGAN reward. ADR-0015 adds a KL anchor +
> a bounded reward. The rest of this ADR (training_step override, multi-step inner loop,
> no-EMA, raw arm, val/fid selection) stands.

The GRPO policy learner is a **`GRPOModule(spt.Module)`** that overrides
**`training_step`** (not `forward` — a deliberate deviation from the `RewardModule` seam,
justified by GRPO's multi-term, multi-step structure), holds the **trainable JiT UNet**
(the policy), a **frozen `RewardModel`** (unregistered, like the reward Module holds its
denoiser), and a stateless `FlowMatchGRPOScheduler`. Per `training_step`: a no-grad
singular-branch rollout (ADR-0011) fills a buffer; then an **inner loop over the
`eta_step_list`** runs one PPO update each — recompute `new_log_prob` under grad, clipped
surrogate loss, `manual_backward`, grad-clip, `opt.step`, scheduler, `zero_grad`, so the
ratio drifts off 1 and the clip **binds** (real trust region, Granular-faithful). **No EMA
during GRPO**: GRPO inherits the raw / no-EMA arm from JiT (ADR-0006; JiT itself attaches no
EMA callback, EMA having been removed codebase-wide); resume **raw**, select and export
**raw**. Validation reuses JiT's fixed-sample FID unchanged, monitored on `val/fid`;
`val/mean_reward` is logged as the progress signal.

## Why

- **Single-step would degenerate the clip to a no-op.** One `opt.step` per
  `training_step` ⇒ every term recomputes `new_log_prob` at the rollout weights ⇒ ratio = 1
  ⇒ `clip(1) = 1` never binds ⇒ the method collapses to group-normalized REINFORCE with no
  PPO trust region. The multi-step inner loop is what keeps the clip load-bearing — and it
  costs ~the same compute (exactly N×G grad UNet evals either way; (c) just does N cheap
  `opt.step`s instead of 1).
- **Override `training_step`, not `forward`.** `RewardModule` fit spt's "forward → one loss
  → one `manual_backward` → one `opt.step`" by returning a single BT loss. GRPO needs N
  `opt.step`s per rollout, so the single-loss seam cannot hold. Overriding `training_step`
  is still fully within `spt.Module` (Trainer, checkpointing, callbacks all
  reused); only the per-step optimizer loop is customized (~5-line replication of spt's
  grad-clip / `opt.step` / scheduler / `zero_grad`).
- **Memory stays at one live grad eval.** Inside each inner step, the per-term (or
  per-`eta_step`) `manual_backward` releases each term's graph as it returns, so peak
  autograd memory is one UNet-forward's activations — not N×G — at any group size. This is
  the property that lets `G` and `eta_step_list` be tuned later without an OOM wall.
- **No EMA under RL.** GRPO inherits the raw / no-EMA arm from JiT (ADR-0006); JiT itself
  attaches no EMA callback (EMA was removed codebase-wide), so there are no supervised-decay
  shadows to maintain or swap. The earlier rationale was that the JiT double-EMA decays
  (0.9999 / 0.9996) were RL-laggy (over RL steps the slow shadow sat at `0.9999^N ≈ 1`,
  reflecting no RL progress and worthless for eval) and wasted ~7 GB across two full UNet
  copies; that argument is now historical, since JiT never held those shadows in the current
  codebase. The JiT FID callback needs no EMA-swap (it already evaluates raw), matching
  Granular's single-model design.
- **Resume and select raw.** ADR-0006 deploys the raw arm; post-training anything else
  optimizes a non-deployed distribution. Monitor `val/fid` on raw (mode=min) so a
  reward-hacked checkpoint (high reward, high FID) is never selected.
- **No KL for v1.** The binding clip + LR + FID are the trust region; KL is the documented
  first escalation if reward rises without FID gain. Granular runs without KL; the medical /
  spoofable-reward risk is covered operationally by FID selection rather than preemptively
  by a `β·KL` term (Simplicity First; KL is cheap to add later — one frozen ref UNet + one
  buffer field + one loss term). **Reversed 2026-07-06 by
  [ADR-0015](0015-grpo-kl-anchor-and-bounded-reward.md): v1 hacked without it.**

## Considered options (rejected)

- **Override `forward` only, single aggregated loss, one `opt.step`** (the `RewardModule`
  seam). Rejected: makes the PPO clip a no-op (REINFORCE); also retains all N×G grad
  activations in one backward (OOM risk at large `G·N`).
- **Override `training_step` with one aggregated `opt.step`** (single-step). Rejected: same
  clip-no-op degeneration as above; loses the trust region (c) was chosen to provide.
- **Maintain JiT EMA during GRPO.** Rejected (now moot): the historical argument was
  RL-laggy useless shadows + ~7 GB, but JiT itself attaches no EMA callback (EMA removed
  codebase-wide), so there are no shadows to maintain; GRPO inherits JiT's raw arm directly
  (ADR-0006).
- **Resume / select a non-raw arm.** Rejected: ADR-0006 deploys raw; would post-train a
  non-deployed distribution. (Historically the slow-EMA arm; now moot, JiT carrying no EMA
  arm.)
- **KL-to-reference from v1.** Rejected as a v1 default: low reversal cost, so building it
  preemptively violates Simplicity First; FID selection already screens for hacking.
  **Adopted 2026-07-06 by [ADR-0015](0015-grpo-kl-anchor-and-bounded-reward.md)** (with a
  bounded reward) after v1 hacked.

## Consequences

- `GRPOModule(spt.Module)` owns: the trainable UNet (registered, the only params passed to
  `configure_optimizers`), the frozen `RewardModel` (unregistered via
  `object.__setattr__`, `eval` + `requires_grad_(False)`, device-moved in `on_fit_start`),
  the `FlowMatchGRPOScheduler`, and the GRPO knobs (`G`, `eta_step_list`, `η`, `clip_range`,
  `lr`, `adv_clip_max`). It overrides `training_step` and `configure_optimizers`; it
  attaches no EMA callback.
- The rollout buffer per `(i,k)` stores `(z_k, t_k, t_{k+1}, z_{k+1}, old_log_prob, A_{i,k})`;
  `new_log_prob` is recomputed under grad in the inner loop (`mean_new` via
  `scheduler.sde_step_mean(UNet(z_k,t_k), z_k, t_k, t_{k+1})`), so no policy output is
  buffered. `std` depends only on `t` ⇒ recomputable, not stored.
- Validation = JiT's fixed-sample FID (re-seeded noise, `sample_latent_flow` on the current
  UNet — the deployed Heun sampler, not the SDE — decode, unbiased FID), monitored on
  `val/fid` for checkpoint selection; `val/mean_reward` (PatchGAN on the same generated
  samples) logged as progress. Hacking alarm = reward↑ without FID↓ ⇒ escalate to KL.
- Launch gates: the ns15 reward model must finish and clear `val/gen_pair_acc > 0.8`; a tiny
  config (`G=2`, short `eta_step_list`, `num_steps=4`) is measured for it/s + peak memory on
  the target cluster before sizing the real `G`/`eta_step_list`/`n_epochs` (mirrors ADR-0010).
- v1 defaults: `num_steps=15`, `G=8`, `eta_step_list=[0..7]`, `η=0.7`, `clip_range=1e-4`,
  `lr=1e-6`, `adv_clip_max=5.0`. New `configs/train/config_grpo.yaml` + `manifold-train-grpo`
  console script (`manifold.training.grpo_cli:main`), paralleling the reward CLI.
