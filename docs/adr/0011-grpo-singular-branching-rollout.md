# GRPO policy rollout: singular branching on a Heun anchor, scheduler-inherited SDE

The JiT x0-denoiser checkpoint is post-trained with **GRPO** against the frozen
PatchGAN reward model. We adopt **Granular-GRPO's singular-branching rollout**
(arXiv 2510.01982), not the vanilla trajectory-level GRPO the SDE derivation
literally describes: one deterministic Heun **anchor** per group, one stochastic
**SDE step** branched off a single anchor node, and a deterministic Heun **suffix**
to the terminal latent the reward scores. The SDE transition math lives in a new
**`FlowMatchGRPOScheduler(FlowMatchHeunDiscreteScheduler)`** (ADR-0008 pattern),
exposed as math-only `sde_step_mean(x_θ, z, t, t_next) → (mean, std)`; the caller
draws the sample and owns the log-prob/replay. The diffusion is the equimarginal
reverse-SDE `σ_t ∝ η·sqrt((1−t)/t)` (η default 0.7), clamped at the noise end.

## Why

- **Vanilla trajectory GRPO is infeasible on 3D latents, not just suboptimal.** Every
  step stochastic ⇒ the new-log-prob recompute threads grad through the *entire*
  15-step two-eval Heun rollout (≈30 UNet evals) × `G` siblings — ~360 UNet-forward
  activations live in the autograd graph per update, almost certainly OOMing a single
  Blackwell on `[4,64,64,32]` latents. Singular branching puts **one** UNet eval under
  grad per branch (the perturbed step); the anchor and suffix are `no_grad`. The
  algorithmic choice is forced by compute.
- **Reward the deployed distribution.** The reward is scored on the terminal `z_K`, so
  `z_K` must come from the sampler JiT deploys — the two-eval Heun (ADR-0002/0005). An
  all-Euler rollout (Granular-faithful; FLUX deploys single-eval Euler) would optimize
  the *Euler* trajectory distribution while we infer with Heun → a train/inference
  sampler gap and RL gains that may not transfer. The Heun anchor + Heun suffix (no_grad)
  cost no grad memory; only the single perturbed step is Euler-Maruyama under grad.
- **Scheduler owns transition math, not policy concerns.** `sde_step_mean` returns the
  Gaussian `(mean, std)`; the Module draws `z_{k+1}=mean+std·ξ`, stores the buffer, and
  computes old/new log-probs. This matches ADR-0002's "scheduler = reverse-step math,
  caller drives evals/RNG" and CONTEXT.md's "Scheduler owns no training concern," and
  makes old/new replay trivially correct (re-eval the UNet under grad at the stored
  `z_k`; evaluate the density at the fixed `z_{k+1}`). The drift reuses `euler_step`'s
  velocity `v_θ=(x_θ−z)/(1−t)` — the SDE mean is *Euler update + Langevin correction* —
  so transport/Heun math is inherited, never reimplemented.
- **Both endpoint blowups reuse existing clamps.** The clean-end `(x_θ−z)/(1−t)` is the
  velocity already clamped by `euler_step` (start node) / `heun_correct` (`t_eps=0.05`).
  The noise-end `(σ²_t/2t)·x_θ` divergence is clipped via `t_safe=max(t, 1/n)`,
  mirroring Granular's `torch.where(σ==1, σ_max, σ)`. No new numerical regime.

## Considered options (rejected)

- **Vanilla trajectory GRPO** (the derivation's literal form: G full stochastic
  trajectories, terminal reward broadcast across all step ratios). Rejected: infeasible
  memory (above), and it is not what the reference repo does.
- **All-Euler rollout** (Granular-faithful end-to-end). Rejected: train/inference
  sampler gap — RL optimizes a different distribution than the one JiT ships.
- **Two-eval Heun-SDE perturbed step** (Heun-integrate the SDE for a higher-order drift).
  Rejected: doubles the grad UNet evals per branch for negligible gain, and deviates
  from singular branching's minimal-perturbation intent.
- **Free-function SDE step** (not on the scheduler). Rejected: violates the
  scheduler-owns-transport invariant (ADR-0001/0008) and the "single source of truth"
  for the reverse step (ADR-0005).
- **Custom vanishing `σ_t = η·sqrt(t(1−t))`** (no noise-end blowup). Rejected: not the
  equimarginal SDE — loses the score-based log-prob interpretation and the transport
  geometry; Granular's equimarginal choice is de-risked and only one knob (η).

## Consequences

- New scheduler `FlowMatchGRPOScheduler(FlowMatchHeunDiscreteScheduler)` adds
  `sde_step_mean` + the `σ_t` schedule (η, `t_safe`) and inherits `add_noise`,
  `set_timesteps` (anchor grid), `euler_step`, `heun_correct`. The GRPO anchor uses the
  parent's full `linspace(0,1,n+1)` grid — it is **not** the partial `set_timesteps_partial`
  scheduler (that owns reward-training's per-sample `t_start`).
- Per training step: **one** shared anchor rollout (same initial noise ⇒ identical across
  the `G` siblings) under `no_grad`; then for each perturbed step `k` and each sibling,
  one SDE branch (no_grad, draw `z_{k+1}`, old log-prob), one Heun suffix (no_grad), one
  reward; group-normalize the `G` rewards per step → `A_{i,k}`. Grad recompute is only the
  perturbed-step UNet evals.
- The transition-policy and group definitions live in CONTEXT.md ("GRPO policy learning");
  the spt.Module integration (single-update aggregation vs inner loop), KL/clip, validation
  metric, and resume arm are tracked separately (this ADR covers only the rollout spine).
