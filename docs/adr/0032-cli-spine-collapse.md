# CLI spine collapse — a composed TrainingSpine over the registry; five thin shells; one checkpoint owner

The five training CLIs each hand-roll the same spine — assemble a callback list,
build a `ModelCheckpoint`, call `build_trainer`, `trainer.fit`. E collapses that
repetition into **one composed `TrainingSpine` object** (`src/manifold/training/core.py`)
that owns callback-name merging → `registry.resolve/build` → checkpoint
injection → `build_trainer` → `fit`. (Seeding stays in each **shell** — it must
run *before* module construction, which sets the initial weights, and before
`_real_inputs` generates the validation pairs / probe; moving it into the spine,
which receives an already-assembled module, would make initial weights and
precomputed stochastic inputs vary across nominally-identical seeded runs.) The
five `run_*` functions shrink to **thin shells** that seed, build their own
`module` + `datamodule` (the genuinely different `_real_inputs` paths stay put)
and delegate to `TrainingSpine.run`.

This is phase E of the four-point architecture refactor (issue #157). It is the
last structural piece, and it consumes the prior three: it is the single caller
of the ADR-0029 registry, it threads ADR-0031's DDP-strategy decision (keep
default `broadcast_buffers=True`; the frozen-arm isolation is `requires_grad=False`
+ `eval()`, not a global flag) into `build_trainer`, and it is the one place that
has to change (not five) when a new callback lands.

## Why

- **The spine is duplicated five ways, and the duplication is the bug surface.**
  Each CLI repeats the same assemble→checkpoint→trainer→fit sequence with only
  four real differences: the Module class + constructor args, the checkpoint
  monitor metric (`val/fid` / the generated-end probe / ControlNet native-export /
  `val/mean_reward`), the datamodule / `_real_inputs` assembly, and which of
  `TrainLossLogger` / `LatentX0MAE` / `FIDCallback` are in the list. A shared
  `TrainingSpine` makes a new callback (ADR-0029) or the FID hardening (ADR-0030)
  a one-place change instead of a five-CLI sweep.
- **The registry already needs a single caller.** ADR-0029 defines a two-phase
  resolve/build against a typed `CallbackContext`, but nothing in the tree calls
  it yet — the registry is unbuilt. The spine is where that handoff lives:
  `TrainingSpine.run` is the **only** caller of `registry.resolve/build`, the one
  place `CallbackContext` is populated, and the one place the dynamic default
  callback **name** set is derived per CLI (it is dynamic — JiT depends on
  `bundle.has_val` / `enable_fid`, GRPO Mode-2 suppresses FID).
- **ModelCheckpoint gets one owner (EC9).** Today `_build_checkpoint` is copied
  per CLI and imported by tests; the registry (ADR-0029) already registered
  `ModelCheckpoint` as `name="checkpoint"` with monitor validation. E resolves the
  duplication by making `CheckpointSpec` the **sole** owner — the per-CLI
  `_build_checkpoint` helpers are deleted, and the seven test sites that imported
  them (see the rejected shim option) are updated to assert on `CheckpointSpec` /
  `trainer.callbacks` membership instead. This is the trade-off that makes the
  collapse a real net reduction rather than a wash.
- **The DDP strategy has one construction site.** ADR-0031 keeps the default
  `broadcast_buffers=True` (the frozen-arm isolation is `requires_grad=False` +
  `eval()`, not a global flag), so the spine's only DDP concern is preserving
  `find_unused_parameters=True` (the `test_ddp_warm.py` source-string assertion) in
  `build_trainer`, with the single-GPU `"auto"` path untouched.

## Considered options (rejected)

- **A `TrainingRunner` base class that the five CLIs subclass:** rejected —
  violates the project OOP rule (composition over inheritance; `pl.Callback` /
  `spt.Module` / `nn.Module` are the only sanctioned bases, and "a training
  runner" is not one of them). The shared behavior is a composed `TrainingSpine`
  object the CLIs hold, not a base they extend.
- **Collapse the `_real_inputs` data-assembly paths into a unified abstraction:**
  rejected (EC5). The five data paths are genuinely different — JiT warm cache,
  ControlNet paired triple, GRPO Mode-1/Mode-2 rollout inputs, reward pairs. A
  fake-unified data interface would cost more than it saved and obscure real
  differences. Each CLI keeps its own `_real_inputs` / `_warm_*` and hands the
  spine only the assembled `module` + `datamodule`.
- **Keep `_build_checkpoint` as a deprecated re-export shim:** rejected (EC9). It
  would preserve the four test imports and the `reward-*.ckpt` glob dependency
  unchanged, but it leaves `ModelCheckpoint` with two owners (the registry spec
  and the shim) and erases the net-line reduction that is the point of the
  collapse. The shim is the "wash" outcome EC10 warns against; deleting it and
  updating the tests is the cleaner break. The full set of test sites importing
  `_build_checkpoint` (a repo-wide `rg "_build_checkpoint" tests`) is seven:
  `tests/ddp.py` three call sites (`:161`, `:295`, `:520`),
  `tests/test_paired_reward.py:623`, `tests/test_reward.py:588`,
  `tests/test_ddp_detection.py:167`, and `tests/test_training_cli.py:349` — all
  seven are migrated, not only the DDP/reward sites.
- **Fold `run_grpo_measurement` into the spine:** rejected (EC8). It is not a
  training spine — it is the #59 launch-gate measurement harness that *calls*
  `run_grpo_training`. It stays as-is and continues to call the `run_grpo_training`
  shell; its signature is not part of the collapse.
- **Two status-quo "no-collapse" angles (a spine object with no spine; a
  checkpoint-in-registry design with no caller):** both surfaced as effectively
  null designs during adversarial design — they fail EC4 (no registry caller),
  EC6 (no single DDP-strategy site), EC7 (five seams, not one), and EC10 (no
  reduction) outright and were not adopted.

## Consequences

- **New `src/manifold/training/core.py`** holds `@dataclass TrainingSpine` (a
  composed object, not a base class) whose `.run(...)` performs the full sequence.
  The object holds a `CallbackRegistry`; `run` takes the assembled `module`,
  `datamodule`, a `CallbackContext`, the `cfg`, a `default_callback_names` list,
  the CLI `callback_names_override` (separate from the YAML `callbacks:` list),
  the monitor / filename metadata, and the multi-GPU flag. No module-level bare
  functions except the existing console `main`s (OOP rule).
- **The five `run_*` are thin shells** — their existing positional / test-seam
  signature is preserved (`run_training(module, bundle=_DataBundle(...))`,
  `run_grpo_training(module, inputs=GRPOInputs(...))`, plus the
  `main(argv, data_provider=...)` injection seam), and each gains
  **backward-compatible keyword args** (`callback_names=None`, `cfg=None`) that
  forward the YAML `callbacks:` block and the `--callbacks` CLI override through
  to `TrainingSpine.run` (defaulting to `None` ⇒ the legacy hand-assembled set, so
  existing tests that call `run_*` without them are unchanged — `main` is the one
  caller that populates them). Each builds its module + datamodule via its own
  `_real_inputs`, derives its dynamic default callback **name** set and monitor
  metric, and delegates to `TrainingSpine.run`. They stay module-level functions
  (grandfathered public API; the OOP rule does not force rewriting existing seams
  into classes).
- **`TrainingSpine.run` is the single registry caller.** It applies the ADR-0029
  merge order — defaults → YAML `callbacks:` → CLI `--callbacks` **replace** —
  then `registry.resolve(names, cfg)` (fail-fast on unknown name/knob),
  populates `CallbackContext`, and `registry.build(specs, ctx)`. The default name
  set is derived in each shell (e.g. add `"fid"` only when `enable_fid`).
- **Mode-2 FID is rejected post-merge, not only at default-derivation.** GRPO
  Mode-2 deliberately suppresses FID (`grpo_cli.py:153-165`): `GRPOModule.sample()`
  ignores the trainable ControlNet, so a Mode-2 FID is a **constant frozen-base
  metric** that would select meaningless checkpoints. Suppressing it only while
  deriving defaults is insufficient — a YAML `callbacks:` or `--callbacks`
  override re-adding `"fid"` (or a checkpoint monitor on `val/fid`) would silently
  re-enable it. So after the merge, the spine (or `FIDSpec.build` against the
  Mode-2 context) **force-removes `"fid"` and rejects a `val/fid` monitor** for
  Mode-2, with a loud log. The override is honored for every other callback; FID
  is the one Mode-2-forbidden name.
- **`CallbackContext` gains optional `real_latents` and `feature_net` fields.**
  GRPO's FID reference lives in `GRPOInputs.real_latents` (`grpo_cli.py:72`), not
  in its conditioning-only datamodule; the shell sets `ctx.real_latents =
  inputs.real_latents`. JiT/cli leaves it `None` so `FIDSpec.build` falls back to
  `real_latents_source=ctx.datamodule` (ADR-0017 / F5 laziness preserved).
  Separately, the CPU / integration test seam injects a tiny **direct** feature
  network (`run_training(feature_net=...)`, `GRPOInputs.feature_net`) rather than
  invoking the production backbone factory; `ctx.feature_net` carries that
  already-built network so `FIDSpec.build` forwards it verbatim alongside the
  optional `feature_net_factory` (ADR-0030). Both default to `None`.
- **`CheckpointSpec` is the sole `ModelCheckpoint` owner.** Monitor / mode /
  `save_top_k` / filename metadata are passed by the shell and injected into
  `cfg.checkpoint` before `registry.resolve`; the spec validates the monitor metric
  is in the resolved callbacks' logged metrics **union the module's declared
  metrics** (ADR-0029 — reward / paired-reward / GRPO-without-FID monitors
  `val/gen_pair_acc` / `val/mean_reward` are module-logged, not callback-logged).
  **The reward / paired-reward DDP fallback is preserved** (mirrors the JiT
  fallback): those monitors are rank-0-local (`val/gen_pair_acc` is computed on
  rank 0's generated-end probe shard), so under `is_multi_gpu` the shell passes
  `monitor_metric=None` — yielding an unmonitored `save_last` + `save_top_k=1`
  checkpoint instead of monitoring a rank-local metric. The two existing tests
  (`tests/test_reward.py:586`, `tests/test_paired_reward.py:619`) assert exactly
  `multi.monitor is None` and stay authoritative; single-GPU keeps
  `monitor=val/gen_pair_acc`. The seven test sites that imported
  `_build_checkpoint` (see the rejected shim option) are rewritten to assert on
  `CheckpointSpec` or `trainer.callbacks` membership, and the `reward-*.ckpt`
  glob dependency is updated to the registry-specified name.
- **Seeding stays in each shell's `main`, before module construction.** The
  preserved `run_training(module, ...)` / `run_grpo_training(module, ...)`
  signatures receive an **already-constructed** module, so `run_*` / the spine
  cannot own the initial seed — `pl.seed_everything` must run in `main`
  (`cli.py:356`, `grpo_cli.py:147/305`, …) **before** the module factory sets
  initial weights and before `_real_inputs` generates the validation pairs /
  probe, or initial weights and precomputed stochastic inputs would vary across
  nominally-identical seeded runs. `run_*` may reseed for direct callers, but the
  authoritative seed is in `main`.
- **`build_trainer`** keeps constructing
  `DDPStrategy(find_unused_parameters=True)` under `is_multi_gpu` with the default
  `broadcast_buffers=True` (ADR-0031 — the frozen-arm isolation is
  `requires_grad=False` + `eval()`, not a global buffer flag); the single-GPU
  `"auto"` path is untouched. The `find_unused_parameters=True` source string is
  preserved (`test_ddp_warm.py:169` still passes).
- **`fid_eval` → `fid` rename** (ADR-0029 namespace alignment) lands with E:
  `tests/test_training_cli.py:465` and `configs/train/config_rflow_jit.yaml:40`
  are updated.
- **The pre-fit probe mutations** (`module.probe_batch_size`, `module.set_val_probe`
  in reward / paired-reward) stay in their shells — they are mode-specific
  Module setup, not spine concerns.
- **Behavior preservation.** Same callbacks built (now via `CallbackSpec.build`),
  same monitors, same DDP strategy, same CPU-smoke and `run_*` test seams. The
  collapse is a refactor; the new single-seam tests assert on `TrainingSpine.run`.

## Out of scope (deferred)

- **The `device_policy.py` orphan and the `paired_reward_cli` cuda:0 gap** stay in
  ADR-0031's A2 (pre-Trainer staging); E does not touch pre-`Trainer.fit` model
  placement.
- **Rewriting the existing module-level `_real_inputs` / `_warm_*` / `_inference_recipe`
  helpers into objects** — they are grandfathered; the OOP rule governs new modules
  (`core.py`), not a forced rewrite of every existing helper.
- **Checkpoint-resume callback-compatibility validation across a `.ckpt` resume**
  — unchanged from ADR-0029 (deferred).
