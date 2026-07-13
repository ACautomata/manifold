# Paired GRPO (G2RPO) — Brownian-bridge singular-branch GRPO on the paired src→tgt flow

The trained Paired JiT UNet is post-trained with **Granular-GRPO** (singular
branching, arXiv 2510.01982) against the frozen paired reward (ADR-0018–0023), over a
**data-to-data Brownian bridge** `Z_t = (1−t)·X_src + t·X_tgt + √(η·t·(1−t))·ε`. GRPO
requires a stochastic policy with a computable Gaussian transition density, so
ADR-0013's deterministic transport must be made stochastic — but with diffusion that
**vanishes at both data endpoints** (`σ²=η·t(1−t)`), not the equimarginal
`σ_t ∝ η·√((1−t)/t)` of ADR-0011 (which *diverges* at the noise end and is wrong when
`t=0` is data). The bridge SDE `dZ_t = (x̂₁_θ − Z_t)/(1−t) dt + √η·dW_t` is a *forward*
Doob h-transform pinned at `Z_1 = x̂₁_θ`; its x-pred drift is exactly the euler velocity
`(x̂₁−z)/(1−t)` with **no Langevin/score term**. **This supersedes ADR-0013's rejection
of the stochastic bridge FOR THE RL REGIME ONLY** — supervised Paired JiT stays
deterministic; the bridge is training-only exploration, never deployed (inference reuses
`PairedLatentFlowPipeline` + the deterministic Heun, raw arm). Sibling of the noise→data
GRPO (ADR-0011/0012/0015), consuming the paired reward. Design grilled + math
adversarially verified (4 independent re-derivations, 0 errors).

## Why

- **The bridge is forced by GRPO, not chosen for capacity.** ADR-0013 rejected the
  stochastic bridge for *supervised* Paired JiT (BraTS co-registered is one-to-one;
  determinism enables reproducible PSNR). GRPO inverts that: it needs a stochastic
  transition to define a per-step Gaussian policy `π_θ(z_{k+1}|z_k) = N(μ_θ, σ²_k)` whose
  log-prob and KL are computable. The bridge is the minimal such transport that keeps
  both endpoints as data. The bridge is *exploration-only*; deployment and validation
  use the deterministic Heun, so ADR-0013's reproducible-PSNR benefit is preserved at
  val/deploy.
- **Forward bridge, no Langevin term (§5 of the derivation).** The policy SDE is a
  forward Doob h-transform pinned at the predicted target `x̂₁_θ`; its drift is exactly
  `(x̂₁_θ − z)/(1−t)`. The equimarginal GRPO scheduler (ADR-0011) adds a
  `(σ²_t / 2t)·x_θ` Langevin correction because it *time-reverses* a noise→data marginal;
  the bridge does not reverse, so the score is absorbed into the pin and there is no
  correction. Verified end-to-end (integrating-factor + Itô isometry, Fokker–Planck,
  variation-of-constants).
- **§7 exact transition, not §6 Euler.** Both share the same mean = `euler_step` output
  (the drift is linear in Z *and* `β(t)=1/(1−t)` — an algebraic identity; verified
  float64-exact, err 8.9e-16); they differ only in std. §7 `σ²=η·Δt·(1−t_{k+1})/(1−t_k)`
  vanishes at the endpoint (terminal μ→x̂₁, stable) where §6 `σ²=η·Δt` leaves residual
  terminal noise. §7 is finite everywhere ⇒ no `t_safe` clamp (the bridge has no
  noise-end blowup). Consensus of all verify agents + branch + numerics.
- **σ is θ-independent ⇒ the entire `grpo.py` spine reuses verbatim.** `σ_k` depends only
  on `(t_k, t_{k+1}, η)`, never on θ, so the policy and frozen-reference transitions
  share equal variance and the diagonal-Gaussian KL collapses to `0.5·‖μ_θ − μ_ref‖²/σ²`
  (trace + log-det cancel). `gaussian_log_prob`, `_transition_kl`, `group_advantage`,
  `clipped_surrogate_loss`, and the multi-step PPO inner loop carry over unchanged
  (ADR-0011/0012/0015). The bridge scheduler is one method off `euler_step`.
- **Pure-RL, no auxiliary x₀-MSE.** Unlike noise→data GRPO, paired data *has* ground-truth
  `x_tgt`, but the bridge pins `Z_1→x̂₁_θ` (generative), and G2RPO's purpose is to push
  *past* the supervised PSNR ceiling via the realism reward — an aux-MSE would anchor
  back to that ceiling. The KL-to-init anchor is the first anti-hacking escalation
  (ADR-0015); aux-MSE is the documented second. Training batch =
  `{x_src, src_label, tgt_label, spacing}` (tgt volume unused at train; mirrors GRPO's
  conditioning-only batch).
- **Raw arm, inverting ADR-0021 for this stage.** Init = slow-EMA paired UNet
  (`load_frozen_paired_generator`, ADR-0021 — the smoothest arm, degrades most gracefully
  under bridge perturbation); train no-EMA (ADR-0012 — RL shadows are laggy + ~7 GB);
  resume/select/export **raw** (ADR-0006/0012). The published paired arm is thus no longer
  slow-EMA for the G2RPO stage — a deliberate asymmetry surfaced here.
- **η-ramp, not static.** The paired UNet was trained on a *zero-noise* deterministic
  transport (unlike the JiT UNet, trained on noisy inputs), so a static-high η shocks it
  (off-manifold suffix inputs → degraded x̂₁ → contaminated reward — off the grad path).
  An η-ramp (`η_min ≈ 0.1–0.2 → η_max ≈ 0.7` over the first ~0.3 epochs, then hold) bounds
  both the reward-spread OOD and the suffix-init OOD early. Ramp-and-hold, not a permanent
  cut (ADR-0015 rejected static η reduction as the *fix*; the ramp is a transient warm-up).
- **`val/psnr` selection, not `val/fid`.** Paired has ground-truth `x_tgt`, so PSNR is the
  honest, reproducible (deterministic Heun) goal metric — selection on `val/psnr` (max) +
  `val/ssim ≥ 0.9` guardrail, `val/mean_reward` as progress. This differs from ADR-0012
  (`val/fid` for noise→data, which has no GT) and reuses `PairedPSNRSSIMCallback`; no FID
  triple needed for v1.
- **Bridge-noise reward-ranking is a hard launch gate.** The paired reward was trained
  real-vs-DETERMINISTIC-fake (ADR-0020); G2RPO needs it to rank fake-vs-bridge-noised-fake
  (what `group_advantage` consumes). If near-random on the spread, G2RPO silently
  random-walks (signal-death, not a 3370 logit blowup — singular branching bounds the
  perturbation to ~0.2 std). A pre-launch probe (init UNet @ η_max, reward ranking vs
  PSNR-to-x_tgt surrogate, acc > 0.6) gates the launch; on failure, retrain the reward with
  bridge-noised fakes (augment the ADR-0020 fake cache) before G2RPO.

## Considered options (rejected)

- **Reverse bridge with a score term** (time-reverse the bridge, add `(g²/2)·s_t`).
  Rejected: the forward Doob h-transform pinned at `x̂₁_θ` already gives the exact drift
  with no score (§5); reversing would need a separate score model and gains nothing.
  Documented here so a future reader does not "fix" the missing Langevin term.
- **§6 Euler transition** (`σ²=η·Δt`). Rejected as default: leaves residual terminal noise
  (`z_K ≠ x̂₁`) and is not closed-form-exact (constant diffusion has a closed-form
  transition, so EM has no excuse). Kept as the flat-exploration variant only if
  `eta_step_list` is ever extended toward the terminal (§7's var→0 would collapse the KL
  there; guard with `max(1−t_{k+1}, t_eps)` or switch such steps to §6).
- **Static η=0.7 (no ramp).** Rejected: the paired UNet's zero-noise training makes the
  early shock a real `z_K`-quality risk; the ramp is cheap insurance. (Probe-first was
  considered; the ramp was chosen as the v1 default to de-risk the first launch — the
  probe still runs to tune the schedule.)
- **Auxiliary x₀-MSE on `x_tgt`.** Rejected for v1: anchors to the PSNR ceiling G2RPO
  exists to escape; redundant with the KL anchor; MSE/reward can conflict. Second
  escalation only.
- **Reuse `singular_branch_rollout` (no fork).** Rejected: the paired UNet signature
  differs structurally (`cat([z, x_src]) + class_labels_src/class_labels_tgt`), so the
  rollout, the inner-loop `_new_log_prob`, and the `_transition_kl` reference forward must
  thread `x_src` + summed embedding through every eval. A fork
  (`singular_branch_rollout_paired`) is the largest diff in the feature.
- **`val/fid` selection** (ADR-0012's noise→data choice). Rejected: paired has ground-truth
  `x_tgt`, so PSNR is the honest reproducible goal metric; FID is unnecessary for v1.
- **Subclass the equimarginal `FlowMatchGRPOScheduler`.** Rejected: its `_sigma_t`/`t_safe`
  are dead code for the bridge (constant `g²=η`, no noise-end blowup) and would cargo-cult
  a meaningless clamp. Subclass `FlowMatchHeunDiscreteScheduler` directly.
- **CFG / `guidance_scale` in `PairedGRPOModule.sample`.** Rejected (dropped): paired
  conditioning is `cat([z_t, x_src])` — there is no unconditional path to guide against.
- **`_real_inputs` stub** (as noise→data GRPO has). Rejected: G2RPO has no pure-noise mode
  (`x_src` is always data); data comes from `PairedLatentDataset` directly.

## Consequences

- **New scheduler** `FlowMatchBridgeGRPOScheduler(FlowMatchHeunDiscreteScheduler)`
  (`schedulers/scheduling_flow_match_bridge_grpo.py`) adds one method
  `sde_step_mean(x̂₁, z, t, t_next) → (mean, std)`: `mean = euler_step(x̂₁, z, t, t_next)[0]`
  (NO Langevin), `std = √(η·Δt·(1−t_next)/(1−t))` (§7); no `t_safe` clamp. Inherits
  `add_noise`/`set_timesteps`/`euler_step`/`heun_correct` verbatim.
- **Forked rollout** `singular_branch_rollout_paired` (`modules/paired_grpo.py`) threads
  `cat([z, x_src]) + embed(src)+embed(tgt)` through five sites (anchor, perturbed-step
  rollout eval, suffix, inner-loop `_new_log_prob` re-eval, `_transition_kl` reference).
  **Reward site (silent trap):** `reward_model(cat([x_src.repeat_interleave(G, dim=0), z_K]))`
  — `x_src` MUST be G-expanded before concat or siblings pair with the wrong source
  (corrupted advantage, no error).
- **New Module** `PairedGRPOModule(spt.Module)` overrides `training_step` (no-grad bridge
  rollout → multi-step PPO inner loop with the clip binding; KL anchor `kl_coef` + tanh
  reward bound carried over from ADR-0015). No EMA, no CFG. Validation via
  `PairedPSNRSSIMCallback` (deterministic Heun) + `val/mean_reward`; checkpoint monitors
  `val/psnr` (max), `val/ssim ≥ 0.9` guardrail. Resume/select/export raw.
- **CLI** `manifold-train-paired-grpo` (`training/paired_grpo_cli.py`) +
  `config_paired_grpo.yaml` (`num_steps=8`, `eta_step_list=[0..3]`, η-ramp,
  `kl_coef = β·D` — per-element scale, see below — `reward_temp` recalibrated to the paired
  PatchGAN). Init via `load_frozen_paired_generator` (slow-EMA),
  `reference_policy = deepcopy(policy)`, reward from the `manifold-train-paired-reward`
  `.ckpt`.
- **KL scale caveat (per-element).** `_transition_kl` returns the per-element MEAN (joint
  KL = D × it), so `kl_coef` is on the per-element scale — effective joint-KL weight =
  `kl_coef/D`. To get a textbook β anchor set `kl_coef = β·D`. The closed form is valid
  ONLY while σ stays θ-independent (if a θ-dependent σ is ever introduced, the trace +
  log-det terms would not cancel and the closed form would silently break — flag for the
  maintainer).
- **Launch gate** (mirror ADR-0011/0012, cost-ordered): (1) HARD bridge-noise reward-ranking
  probe @ η_max on init UNet (acc > 0.6 vs PSNR-to-x_tgt; fail ⇒ retrain reward with
  bridge-noised fakes); (2) η-ramp x̂₁-drift probe; (3) tiny-config it/s + peak-GPU (size
  G / η_max / eta_step_list / n_epochs); (4) `val/psnr` at start ≈ supervised ceiling; (5)
  ADR-0015 semantics (`val/mean_reward` in tanh range, not 3370). All pass before the real
  `--max-epochs`.
- **ADR-0013's bridge rejection is superseded for the RL regime only.** Supervised Paired
  JiT (ADR-0013) stays deterministic; the bridge lives in the G2RPO training path and never
  reaches the inference `PairedLatentFlowPipeline` (whose native ckpt carries the base
  `FlowMatchHeunDiscreteScheduler` config, not the bridge scheduler). ADR-0021's
  slow-EMA-as-published-arm is inverted for the G2RPO stage (raw arm); surfaced here.
