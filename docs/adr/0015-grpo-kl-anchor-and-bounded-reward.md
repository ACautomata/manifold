# GRPO v2: KL anchor + bounded reward (reverse ADR-0012's KL=0)

The v1 GRPO recipe (ADR-0011/0012) hacked in one epoch on euler (2026-07-05..06):
`val/mean_reward` rose **−13.75 → 3370** while `val/fid` worsened **3.98 → 12.50** (3.1×).
Empirical calibration proved this is **not** a numerical artifact — `RewardModel.forward`
is a raw PatchGAN patch-logit mean (no sigmoid; Bradley–Terry calibrated only *differences*,
so the absolute scale is unbounded). On **real** val/train latents it outputs −9.7~−11.2 ±8,
range **[−21, +26]**; the JiT baseline (−13.75) is in-distribution, but the policy's 3370 is
**~130× beyond the real-data maximum** — an adversarial OOD exploit of the discriminator,
corroborated by the FID blow-up. ADR-0012's "FID selection screens for hacking" was right
that selection rejects a hacked checkpoint, but wrong that it prevents a wholly-hacking run:
the baseline (val/fid 3.98) is not a saved candidate, so the run would ship a worse-FID
policy than the JiT arm it started from.

Root cause: **unbounded reward + no KL anchor + η=0.7.** The group-normalized advantage
(±`adv_clip_max`) bounds the *per-step* signal but not the *cumulative climb direction* —
~16,000 tiny clipped updates push the policy up the raw-reward gradient into OOD latents.

## Why

- **Add a KL-to-pretrained-policy anchor.** The GRPO transition is
  `π_θ(z_{k+1}|z_k) = N(μ_θ, σ²_t·Δt·I)` where `μ_θ = sde_step_mean(UNet_θ(z_k), …)` and
  `σ_t = η·sqrt((1−t)/t)` depends **only on `t`, not on θ**. So the reference transition
  (from a frozen deepcopy of the pretrained policy) shares the same variance, and the
  diagonal-Gaussian KL collapses to the closed form `0.5·‖μ_θ − μ_ref‖² / σ²` (the trace and
  log-det terms cancel for equal variances). This is computed in the inner loop alongside
  `new_log_prob` (`mean_ref` from one frozen, no-grad reference forward at the stored `z_k`;
  grad flows through `μ_θ` only) and added to each step's clipped-surrogate loss as
  `kl_coef · kl.mean()` — a separate term **outside** the PPO ratio clip, as in standard
  RLHF. The reference UNet is held frozen + unregistered (`object.__setattr__`,
  `eval` + `requires_grad_(False)`, device-moved in `on_fit_start`), mirroring the reward
  invariant — off `parameters()`/`state_dict()`/optimizer/DDP. Cost: +1 no-grad reference
  forward (batch `B`) per `eta` step ⇒ ~5% on top of the rollout; memory: +180M params.
- **Bound the reward with a monotone `tanh`.** `RewardModel` stays the raw scorer (its
  documented contract; the BT premise); the bound is applied at the GRPO consumer:
  `_bound_reward(r) = tanh(r / reward_temp)` for both the rollout's group-reward and the
  validation reward (so `val/mean_reward` tracks the training signal). `tanh` is **monotone**
  ⇒ distinct sibling rewards stay distinct (the group signal survives; a hard `clamp` would
  collapse them) and **bounded** ⇒ an OOD extreme saturates near ±1 instead of dominating
  the advantage. `reward_temp` ≈ the real-data reward std (~8 from calibration) spreads the
  in-distribution range across tanh's active region.
- **Both, not one.** The KL is the primary anchor (prevents leaving the real distribution);
  the bound is a belt-and-suspenders cap on the reward's worst-case OOD excursion. The
  re-measure decides whether either suffices alone (e.g. if KL alone holds FID, the bound
  can be dropped to `none` to recover the full in-distribution signal).
- **Backward-compat defaults.** `GRPOModule` defaults `reference_policy=None`,
  `kl_coef=0.0`, `reward_bound="none"`, `reward_temp=8.0` ⇒ the v1 raw-reward / no-KL
  behavior is unchanged when the knobs are unset. The fix is opt-in via
  `configs/train/config_grpo.yaml` (`kl_coef: 0.1`, `reward_bound: "tanh"`,
  `reward_temp: 8.0`). `_transition_kl` returns `None` when the reference is absent or
  `kl_coef ≤ 0`, so the inner loop adds nothing in the v1 path.

## Considered options (rejected)

- **KL anchor only (no bound).** Rejected as the sole fix: with a truly unbounded reward the
  policy may still find high-reward regions inside the KL budget (the KL penalizes drift but
  does not cap the reward gradient's magnitude at OOD). Kept the bound as a second screen.
- **Bounded reward only (no KL).** Rejected as the sole fix: `tanh` saturates the OOD
  extreme but, without an anchor, the policy can still drift to maximize the bounded reward
  at FID's expense within the (compressed) signal. Kept the KL as the primary anchor.
- **Lower η only (0.7 → ~0.2).** Rejected: a band-aid that slows exploration but does not
  fix the unbounded-reward / no-anchor root cause; the climb would resume, just slower.
- **Hard `clamp` instead of `tanh`.** Rejected: non-monotone at the edges ⇒ distinct sibling
  rewards collapse to the clamp bounds ⇒ the group advantage loses resolution; also a dead
  gradient beyond the threshold. `tanh` is monotone and smoothly saturating.
- **`z-score` the reward against the real distribution.** Rejected: `(r − real_mean)/real_std`
  is still unbounded (maps 3370 to ~423, not capped); it recenters but does not bound. `tanh`
  bounds.
- **Re-train the reward model / image-space reward.** Deferred (out of scope for v2): the
  existing PatchGAN reward is kept; v2 only bounds and anchors against it.

## Consequences

- `GRPOModule.__init__` gains `reference_policy`, `kl_coef`, `reward_bound`, `reward_temp`
  (all backward-compat defaults). It holds the frozen reference UNet unregistered
  (`reference_unet`), mirroring `reward_model`; `on_fit_start` device-moves it. `_new_log_prob`
  returns `(log_prob, mean_new, std_new)` so the KL reuses the policy's grad-bearing mean
  (no second policy forward). `_transition_kl(step, mean_new, std_new, …) → Tensor | None`
  computes the equal-variance KL or short-circuits to `None`. `_bound_reward` applies the
  `tanh`/identity. `singular_branch_rollout` gains an optional `reward_transform` (applied
  to the reshaped rewards before `group_advantage`).
- `GRPOInputs` gains `reference_policy`; `_real_inputs` deepcopies the pretrained policy
  (`copy.deepcopy(pipe.unet)`) BEFORE any GRPO update so the anchor is the bit-identical
  starting arm. `main` reads `kl_coef`/`reward_bound`/`reward_temp` via `opt(gcfg, …)` and
  adds a `--limit-train-batches` debug knob (plumbed via `build_trainer(extra_kwargs=…)`) for
  the fast re-measure.
- `configs/train/config_grpo.yaml` commits `kl_coef: 0.1`, `reward_bound: "tanh"`,
  `reward_temp: 8.0`. β=0.1 is a starting point; the re-measure tunes it.
- Launch gate: a `--limit-train-batches 50 --max-epochs 1` run must show
  `val/mean_reward` in tanh range (~−0.9, **not** 3370) AND moving (signal not killed) AND
  `val/fid` ≈ 3.98 (not climbing). If signal dead ⇒ raise `reward_temp` / drop the bound; if
  FID climbs ⇒ raise `kl_coef`. Only then the full `--max-epochs 10` run.
- ADR-0012's KL=0 decision is reversed here; its other decisions (training_step override,
  multi-step inner loop, no-EMA, raw arm, val/fid selection) stand.
