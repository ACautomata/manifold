# Device ownership — Lightning owns in-Trainer frozen arms (register + dual-exclude); a DevicePolicy owns pre-Trainer staging

The device-ownership problem splits in two, and the two halves get **different**
owners because they live on opposite sides of `Trainer.fit`:

- **A1 — frozen arms inside `Trainer.fit`** (`GRPOModule.reward_model`,
  `reference_unet` / `reference_controlnet`, the Mode-2 frozen base `unet`;
  `RewardModule.denoiser`; `ControlNetLatentFlowModule.unet`) become **normal
  registered `nn.Module` children**, so Lightning's automatic `.to(device)` moves
  them. The `object.__setattr__` registration bypass and the `on_fit_start`
  manual `.to(self.device)` moves are **deleted**.
- **A2 — data-pipeline models outside `Trainer.fit`** (`_real_inputs` rollouts,
  fake-cache builders, VAE cache-warm) **cannot** be Lightning-managed: they run
  before the process group exists, when `self.device` is still `cuda:0`. They
  are routed through a centralized **`DevicePolicy`** that resolves
  `cuda:{LOCAL_RANK}` from the launcher environment.

Off-optimizer and off-checkpoint — the two properties the `object.__setattr__`
bypass was protecting — are preserved explicitly after registration: the
optimizer is built over only the trainable arm's parameters, and
`state_dict()` / `load_state_dict()` are overridden to strip the frozen-arm
keys. This is phase A of the four-point architecture refactor (issue #157).

## Why

- **The central tension is real and self-confessed in the code.** A frozen arm
  must stay off the optimizer, off the checkpoint, and (today) off DDP — which
  is exactly what `object.__setattr__` buys by hiding it from `nn.Module`
  registration. But Lightning's automatic `.to(device)` only visits *registered*
  children, so the same bypass forces every module to re-move its frozen arms by
  hand in `on_fit_start` (the "automatic `.to(device)` skips them" safety-net
  comment in `grpo.py`). Registration + explicit dual-exclusion resolves the
  tension instead of working around it, and is the only design that actually
  delivers the brief's "let Lightning own device placement".
- **A1 and A2 are different problems (timing), not one.** The PR #156
  cuda:0-under-DDP deadlock class comes from **pre-Trainer** staging
  (`_real_inputs`, fake-cache, VAE warm), where no process group exists and
  `LOCAL_RANK` can only be read from the launcher environment. The `on_fit_start`
  manual moves (A1) are a separate, post-DDP concern. A single mechanism cannot
  cover both — a post-PG policy is impossible for A2, and Lightning cannot reach
  A2 at all.
- **Overriding `state_dict()` (not just the Lightning checkpoint hooks) keeps the
  existing tests green.** The repo's frozen-arm tests assert absence directly on
  the module: `assert "reward_model" not in mod.state_dict()` (test_grpo.py:351),
  `"reference_unet" not in mod.state_dict()` (:458), `not any(k.startswith("unet.")
  ...)` (test_grpo.py:1006, test_controlnet_module_training.py:92). Filtering only
  in `on_save_checkpoint`/`on_load_checkpoint` would not affect these direct calls
  and would force a test rewrite; overriding `state_dict()`/`load_state_dict()`
  themselves makes the registered-but-excluded invariant hold at the source, so
  the `state_dict()` assertions pass unchanged. (The `parameters()` disjointness
  checks **cannot** pass unchanged — registration includes the frozen arm in
  `parameters()` regardless of `requires_grad`; those assertions are updated, see
  Consequences.)
- **DDP keeps the default `broadcast_buffers=True`.** Registering frozen arms does
  not require disabling buffer broadcast: `requires_grad=False` keeps their
  gradients off the DDP sync, and they stay in `eval()` (see *frozen-arm
  mode-management*) so their buffers never update — broadcasting identical buffers
  is harmless. A global `broadcast_buffers=False` would instead harm the
  **trainable** `RewardModel` (MONAI PatchGAN, BATCH norm), letting its running
  stats diverge per rank under data-parallel batches. The off-DDP invariant the
  `object.__setattr__` bypass bought is preserved by `requires_grad=False` +
  `eval()`, not by a global DDP flag.
- **The A2 fix is a real, pre-existing bug, not speculative.** `device_policy.py`
  already exists (it was the PR #156 fix) but has **zero importers** — an orphan.
  `paired_reward_cli.main()` never calls `set_device(LOCAL_RANK)`, so its
  `_real_inputs` stages the generator/controlnet on `cuda:0` on every rank
  (paired_reward_cli.py:346-351), reproducing the exact contention class PR #156
  fixed for `grpo_cli`/`reward_cli`. Wiring the existing policy and closing this
  gap is corrective, not new abstraction.

## Considered options (rejected)

- **FrozenArm holder / adapter (centralized manual move).** Wrap each frozen arm
  in a plain-Python holder that owns a single `.to(device)` path called from
  `on_fit_start`. Rejected: it keeps the move **manual** inside a Lightning hook,
  so it under-delivers "let Lightning own device placement" (AC1); it breaks
  Mode-1/Mode-2 parity by making `self.unet` a raw module in Mode-1 and a holder
  in Mode-2; and it scopes out A2, leaving the cuda:0 class to the per-CLI
  patches. (Adversarial verdict: does not survive.)
- **A single post-PG `DevicePolicy` for both A1 and A2.** Rejected as a **timing
  contradiction**: A2 stages models before `Trainer.fit`, so a policy that reads
  `LOCAL_RANK` "after Lightning assigns it" either falls back to `cuda:0` during
  `_real_inputs` (reproducing PR #156) or reads the env var pre-fit (violating
  the post-PG premise). It also concedes Lightning does not automate the frozen
  arms, failing the brief. (Adversarial verdict: does not survive.)
- **Filter frozen keys only in `on_save_checkpoint`/`on_load_checkpoint`, leaving
  `state_dict()` untouched.** Rejected: the existing tests call `mod.state_dict()`
  directly and assert the frozen arms are absent; hook-only filtering does not
  affect those calls and would force a broad test rewrite. Overriding
  `state_dict()`/`load_state_dict()` is strictly stronger and keeps tests green.
- **Defer A1 entirely, do only A2.** Rejected: it leaves the `object.__setattr__`
  + `on_fit_start` manual-move scatter across three modules in place and does not
  address the brief's #1 point for the in-Trainer models.

## Consequences

- **Register the frozen arms** in `modules/grpo.py`, `modules/reward.py`, and
  `modules/controlnet_latent_flow.py` as normal `nn.Module` children; delete the
  `object.__setattr__` calls and the `on_fit_start` manual `.to(self.device)`
  moves. Lightning's `.to(device)` (run after DDP setup assigns `LOCAL_RANK`)
  places them per-rank — the cuda:0 class is not reintroduced for A1 (AC3).
- **Bare tensor probes stay manual.** `RewardModule.val_probe` /
  `PairedRewardModule.val_probe` are plain tensors, not `nn.Module`s, so they
  cannot be registered; they are moved explicitly and kept off the checkpoint by
  the same `state_dict()` filter. This is a documented exception, not a gap.
- **Off-optimizer (unchanged mechanism, now load-bearing).**
  `configure_optimizers` returns Adam over only the trainable arm's parameters
  (`unet` in Mode-1, `controlnet` in Mode-2 / ControlNet module, the reward head
  in the reward modules); frozen arms keep `requires_grad=False` as a second
  safety layer.
- **Off-checkpoint via `state_dict()`/`load_state_dict()` override.** The frozen
  names are declared once per module (a frozen-at-init set such as
  `{"reward_model", "reference_unet", "reference_controlnet"}` plus `"unet"` in
  Mode-2) and stripped from the emitted `state_dict`. Resume uses **strict loading
  with a frozen-arm allowlist**, NOT `load_state_dict(strict=False)`: a blanket
  `strict=False` would also hide a missing *trainable* key (an incomplete or
  mode-mismatched `.ckpt` short of policy/ControlNet/reward-head weights) and
  silently resume on random or stale weights, corrupting the experiment. Strict
  load therefore raises on any missing/unexpected key **except** the declared
  frozen-arm prefixes; only those are tolerated as absent (the frozen arms are
  rebuilt fresh each launch — the reference policy is a `deepcopy` at Mode-2
  launch per ADR-0028; the reward model is reloaded from its `.ckpt`), matching
  ADR-0006's native-export semantics. The allowlist is the same
  `_frozen_arm_names` set the `state_dict()` filter uses. Existing `state_dict()`
  assertions pass unchanged (the override strips frozen keys). The `parameters()`
  disjointness assertions (`test_mode2_freezes_base_and_optimizes_controlnet_only`,
  `test_base_is_frozen_and_unregistered`) are **updated, not preserved**:
  registration makes `module.parameters()` recursively include the frozen arm
  regardless of `requires_grad`, so the assertion moves from *"frozen params absent
  from `module.parameters()`"* to *"frozen params present in `module.parameters()`
  but `requires_grad=False`, absent from the optimizer param groups, and absent
  from `state_dict()`"*. The off-optimizer guarantee is the optimizer-level filter
  + `requires_grad=False`, not `parameters()` membership.
- **DDP:** keep the default `broadcast_buffers=True` (NOT `False`). A global
  `broadcast_buffers=False` would also stop broadcasting the **trainable**
  `RewardModel`'s BatchNorm running stats (MONAI PatchGAN defaults to BATCH norm),
  letting them diverge per rank under data-parallel batches and leaving the saved
  checkpoint with rank-local statistics. The frozen arms need no special
  buffer-broadcast flag: `requires_grad=False` keeps their gradients off the sync,
  and they stay in `eval()` (see *frozen-arm mode-management* below) so their
  buffers do not update — broadcasting identical buffers is harmless.
  `find_unused_parameters=True` is unchanged.
- **DDP init broadcasts the registered frozen-arm parameters (a documented cost).**
  Registration has a second DDP consequence beyond buffer broadcast: at
  `DistributedDataParallel` **construction**, every registered parameter —
  including the `requires_grad=False` frozen arms — is synchronized from rank 0
  (the `params_to_ignore` / `static_graph` knobs address gradient sync, not this
  init-time broadcast). This is the off-DDP invariant the `object.__setattr__`
  bypass bought, and registering the arms gives it up: the multi-gigabyte
  denoiser / reward / reference arms are broadcast once at startup. It is
  acceptable (the arms are rank-identical by construction, so the broadcast is
  redundant-but-correct, and it is a one-time startup cost — not per-step), but
  **GRPO pays it several times** (base + reward + reference unet + reference
  controlnet), adding multi-gigabyte startup communication that can compound the
  DDP-init timeout pressure. If that bites, the mitigation is to construct the
  frozen arms on each rank from the same source (so they are byte-identical and
  the broadcast is a no-op semantically) rather than adding a DDP exclusion hack;
  the cost is documented here, not silently incurred.
- **A2 — land and wire `DevicePolicy`.** A `training/device_policy.py`
  implementing per-rank `cuda:{LOCAL_RANK}` resolution exists locally (written for
  the PR #156 fix) but was **never committed** — it is absent from `HEAD` and from
  this PR, so it is *created/landed* as part of A (not "wired into" as if already
  present), becoming the single place that resolves `cuda:{LOCAL_RANK}` for the
  genuinely pre-PG staging — `_real_inputs` and the fake-cache builders. (The
  **VAE cache-warm is NOT** an A2 concern: it runs inside
  `LatentWarmDataModule.setup()` *after* Lightning initializes the process group,
  which is what enables the `i % world == rank` sharding — moving it pre-PG would
  make every rank see `world=1` and redundantly encode the full cache, regressing
  the multi-hour cold-start ADR-0017 removed. `DevicePolicy` may be used inside
  `setup()` for device selection only, but the warm stays in the DataModule
  lifecycle.)
  **Close the `paired_reward_cli` cuda:0 gap.** Remove the module-level bare
  functions `get_device_policy` / `reset_device_policy` (project OOP rule: only
  console `main` may be a module-level function) in favor of a constructed
  object. The policy resolves **device only**; dtype/AMP stay in the rollout
  helpers and autocast contexts.
- **Rollout helpers unchanged.** `partial_denoise.py`, `sampler.py`,
  `controlnet_sampler.py`, and `grpo._real_inputs` keep inferring
  `device = next(unet.parameters()).device` and casting tensors to it; with the
  arms registered, Lightning has already placed the models, so no signature
  change (AC4/AC5).
- **FID eval VRAM staging is not touched here.** `fid_callback.py`'s
  `_stage_eval_on_device` / `_restore_eval_to_cpu` are absorbed by the
  `VramStage` helper in ADR-0030 (phase C); A does not duplicate it (AC7).
- **Frozen-arm mode-management (the registered-arm cost).** Registering the arms
  makes them part of the module tree, so Lightning's `module.train()` recurses
  into them and flips them back to training mode — an `eval()` set at construction
  does **not** persist. A frozen arm in training mode would let its BatchNorm
  running stats drift during rollout/reward evaluation, corrupting the
  supposedly-fixed function (in `RewardModule`/`ControlNetLatentFlowModule` the
  frozen denoiser/base; in GRPO the first rollout would update the frozen
  reward/reference buffers). The Module overrides `train(mode)` to re-apply
  `eval()` to the frozen arms after `super().train(mode)` (or applies eval in
  `on_train_epoch_start`). This is exactly the cost the `object.__setattr__` bypass
  avoided by keeping them out of the tree.
- **AMP/precision watch-item.** Registering the frozen reward model means its
  forward may be autocast under mixed precision; the existing `.float()` rescues
  in `grpo.py` must be verified to still hold. Verified during implementation, not
  assumed.
- **Tests.** Existing frozen-arm `state_dict()` / optimizer / backward-only tests
  are unchanged; the `parameters()` disjointness tests are **updated** as described
  above (registered ⇒ present in `parameters()`, but `requires_grad=False` +
  off-optimizer + off-`state_dict()`). New tests assert the registered-but-excluded
  invariant, the frozen-arm `eval()` persistence across `module.train()` (the
  mode-management), and add DDP device assertions via `LOCAL_RANK` env-var
  simulation plus a real `torchrun` smoke for the `_real_inputs` placement (the PR
  #156 regression class).

## Out of scope (deferred)

- **Mode-2 GRPO native export path.** `export_to_native` (ADR-0006) today exports
  a plain JiT checkpoint or a supervised ControlNet stage-1 checkpoint; a Mode-2
  GRPO `.ckpt` contains only `controlnet.*` and its frozen base must come from
  the native dir. This ADR deliberately leaves `export_to_native` untouched
  (resume is handled by strict load + the frozen-arm allowlist (see Consequences)
  + fresh rebuild); the Mode-2 export branch is future work.
- **Restructuring the CLIs to defer staging into `DataModule.setup()`.** The
  A2 models are staged pre-Trainer by design; moving them into the DataModule
  lifecycle would be a structural rewrite and is not required once `DevicePolicy`
  centralizes the `LOCAL_RANK` resolution.
