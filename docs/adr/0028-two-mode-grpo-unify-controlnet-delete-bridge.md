# Two-mode GRPO — unify noise→data and ControlNet into one Module; delete the bridge

`GRPOModule` becomes a **single** module supporting two modes:

- **Mode-1 (no ControlNet):** train the UNet — the current behavior (ADR-0011/0012/0015).
- **Mode-2 (ControlNet):** freeze the base UNet, train **only the ControlNet**
  (init from the supervised checkpoint, ADR-0027), against the condition-aware reward.

The paired-GRPO Brownian bridge (ADR-0024, superseded), its `FlowMatchBridgeGRPOScheduler`,
and `singular_branch_rollout_paired` are **deleted**. Mode-2 reuses the **noise→data
equimarginal** `FlowMatchGRPOScheduler` (ADR-0011) and the `grpo.py` spine **verbatim** —
the transition σ is θ-independent whether θ is the UNet or the ControlNet (verified,
adversarially). The KL-anchor reference policy is `deepcopy(base + ControlNet)` at Mode-2
launch.

## Why

- **User decision: GRPO supports two modes (with/without ControlNet).** With ControlNet
  the transport is noise→data (not the data↔data bridge), so the entire paired-GRPO
  bridge machinery — the bridge scheduler, the forked `singular_branch_rollout_paired`,
  the `concat([z, x_src])` UNet signature threaded through five eval sites — is obsolete.
  Mode-2 collapses into the regular `grpo.py` with a ControlNet branch. One module, one
  scheduler, one rollout spine.
- **The spine reuses verbatim (math, adversarially verified).** With the base frozen
  (θ_base fixed) and the ControlNet trainable (θ_ctrl), the singular-branch reverse-SDE
  drift `b_θ(z, x_src)` is a deterministic function of θ_ctrl, so the transition
  `π_θ(z_{k+1}|z_k) = N(mean_θ, σ²_k)` is still a diagonal Gaussian — `gaussian_log_prob`,
  the ratio, and `clipped_surrogate_loss` apply unchanged. `σ_k` depends only on
  `(t_k, t_{k+1}, η)` (equimarginal `σ_t ∝ η·√((1−t)/t)`), never on θ, so the policy and
  frozen-reference transitions share equal variance and the KL collapses to
  `0.5·‖μ_θ − μ_ref‖²/σ²` (ADR-0024's closed form) — `_transition_kl`, `group_advantage`,
  and the multi-step PPO inner loop carry over unchanged. Only the **x_θ source** differs:
  `base(z)` (Mode-1) vs `base(z) + controlnet_residual(z, x_src)` (Mode-2).
- **KL-anchor correctness requires the reference to match the policy structure.** The
  reference policy is `deepcopy(base + ControlNet)` at Mode-2 launch — copying only the
  base would leave the reference's drift ControlNet-free and the KL structurally wrong.
  (Mirror of ADR-0015's anchor, parameterized by θ_ctrl.)
- **v1 drops CFG (user decision).** The drift is `base + controlnet_residual` (no
  guidance scale); this matches ADR-0024's no-CFG stance and ADR-0011's train=deploy
  principle. CFG is mathematically free here (it only scales the residual by `1+w` and σ
  stays θ-independent), but it adds a hyperparameter and rollout/reward/inference
  consistency burden — deferred to v2 if translation authority is weak.

## Considered options (rejected)

- **Keep a separate `paired_grpo.py`/`PairedGRPOModule` (fork):** rejected — it exists
  only for the bridge transport, which is gone; forking would duplicate the spine the
  math says reuses verbatim.
- **CFG during Mode-2 training:** rejected for v1 (above); the spine stays simplest.
- **Subclass `FlowMatchGRPOScheduler` into a ControlNet variant:** rejected — the
  equimarginal scheduler is transport-only and already correct for Mode-2; a subclass
  would cargo-cult dead knobs.

## Consequences

- `modules/grpo.py` gains `controlnet: Optional[ControlNet3DConditionModel]` and
  `freeze_unet: bool`. The singular-branch rollout and the inner-loop `_new_log_prob` /
  `_transition_kl` evals call a **unified x_θ source**: `base(z)` in Mode-1,
  `base(z, residuals=controlnet(z, x_src))` in Mode-2. `configure_optimizers` returns
  `Adam(unet.parameters())` (Mode-1) or `Adam(controlnet.parameters())` (Mode-2). The
  reward site scores `z_K` (Mode-1) or `concat([x_src, z_K])` (Mode-2, condition-aware).
- `training/grpo_cli.py` gains `--grpo-mode {1,2}`; Mode-2 loads the frozen base +
  supervised ControlNet (ADR-0027 `load_frozen_controlnet_generator`) + the
  condition-aware reward; the KL reference is `deepcopy(base + controlnet)`.
- **DELETE:** `modules/paired_grpo.py`, `training/paired_grpo_cli.py`,
  `schedulers/scheduling_flow_match_bridge_grpo.py`, `configs/train/config_paired_grpo.yaml`,
  the `manifold-train-paired-grpo` entry point, and the paired-GRPO tests.
- The bridge-noise launch gate (ADR-0024) demotes to an **optional sanity probe** — the
  supervised init is already a real translator, so the cold-start signal-death risk that
  made it blocking is gone.
- σ stays θ-independent: if a θ-dependent σ is ever introduced, the trace + log-det
  terms in the KL would not cancel and the closed form would silently break (flag for the
  maintainer — carried over from ADR-0024).
