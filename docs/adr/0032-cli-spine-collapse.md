# CLI spine collapse — a composed TrainingSpine over the registry; five thin shells; one checkpoint owner

The five training CLIs each hand-roll the same spine — assemble a callback list,
build a `ModelCheckpoint`, call `build_trainer`, `trainer.fit`. E collapses that
repetition into **one composed `TrainingSpine` object** (`src/manifold/training/core.py`)
that owns seeding → callback-name merging → `registry.resolve/build` → checkpoint
injection → `build_trainer` → `fit`. The five `run_*` functions shrink to **thin
shells** that build their own `module` + `datamodule` (the genuinely different
`_real_inputs` paths stay put) and delegate to `TrainingSpine.run`.

This is phase E of the four-point architecture refactor (issue #157). It is the
last structural piece, and it consumes the prior three: it is the single caller
of the ADR-0029 registry, it threads ADR-0031's `broadcast_buffers=False` through
`build_trainer`, and it is the one place that has to change (not five) when a new
callback lands.

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
  `_build_checkpoint` helpers are deleted, and the few tests that imported them
  are updated to assert on `CheckpointSpec` / `trainer.callbacks` membership
  instead. This is the trade-off that makes the collapse a real net reduction
  rather than a wash.
- **`broadcast_buffers=False` (ADR-0031) needs one DDP-construction site.**
  ADR-0031 decided frozen-arm registration implies `broadcast_buffers=False`.
  The spine threads it through `build_trainer` once, preserving the
  `find_unused_parameters=True` source string that `test_ddp_warm.py` asserts, and
  leaving the single-GPU `"auto"` path untouched.

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
  updating the four tests is the cleaner break. (The tests are: `tests/ddp.py`
  three `_build_checkpoint` call sites, `tests/test_paired_reward.py:623`,
  `tests/test_reward.py:588`.)
- **Fold `run_grpo_measurement` into the spine:** rejected (EC8). It is not a
  training spine — it is the #59 launch-gate measurement harness that *calls*
  `run_grpo_training`. It stays as-is and continues to call the `run_grpo_training`
  shell; its signature is not part of the collapse.
- **Two status-quo "no-collapse" angles (a spine object with no spine; a
  checkpoint-in-registry design with no caller):** both surfaced as effectively
  null designs during adversarial design — they fail EC4 (no registry caller),
  EC6 (no `broadcast_buffers`), EC7 (five seams, not one), and EC10 (no reduction)
  outright and were not adopted.

## Consequences

- **New `src/manifold/training/core.py`** holds `@dataclass TrainingSpine` (a
  composed object, not a base class) whose `.run(...)` performs the full sequence.
  The object holds a `CallbackRegistry`; `run` takes the assembled `module`,
  `datamodule`, a `CallbackContext`, the `cfg`, a `default_callback_names` list,
  the CLI `callback_names_override` (separate from the YAML `callbacks:` list),
  and the monitor / filename metadata. No module-level bare functions except the
  existing console `main`s (OOP rule).
- **The five `run_*` are thin shells** (signatures unchanged — `run_training(module,
  bundle=_DataBundle(...))`, `run_grpo_training(module, inputs=GRPOInputs(...))`,
  etc., plus the `main(argv, data_provider=...)` injection seam). Each builds its
  module + datamodule via its own `_real_inputs`, derives its dynamic default
  callback **name** set and monitor metric, and delegates to `TrainingSpine.run`.
  They stay module-level functions (grandfathered public API; the OOP rule does
  not force rewriting existing seams into classes).
- **`TrainingSpine.run` is the single registry caller.** It applies the ADR-0029
  merge order — defaults → YAML `callbacks:` → CLI `--callbacks` **replace** —
  then `registry.resolve(names, cfg)` (fail-fast on unknown name/knob),
  populates `CallbackContext`, and `registry.build(specs, ctx)`. The default name
  set is derived in each shell (e.g. add `"fid"` only when `enable_fid`).
- **`CallbackContext` gains an optional `real_latents` field.** GRPO's FID
  reference lives in `GRPOInputs.real_latents` (`grpo_cli.py:72`), not in its
  conditioning-only datamodule; the shell sets `ctx.real_latents =
  inputs.real_latents`. JiT/cli leaves it `None` so `FIDSpec.build` falls back to
  `real_latents_source=ctx.datamodule` (ADR-0017 / F5 laziness preserved).
- **`CheckpointSpec` is the sole `ModelCheckpoint` owner.** Monitor / mode /
  `save_top_k` / filename metadata are passed by the shell and injected into
  `cfg.checkpoint` before `registry.resolve`; the spec validates the monitor metric
  is in the resolved set's logged metrics (the absent-vs-disabled distinction of
  ADR-0029). The four tests that imported `_build_checkpoint` are rewritten to
  assert on `CheckpointSpec` or `trainer.callbacks` membership, and the
  `reward-*.ckpt` glob dependency is updated to the registry-specified name.
- **`build_trainer` gains `ddp_broadcast_buffers: bool = False`** and constructs
  `DDPStrategy(find_unused_parameters=True, broadcast_buffers=…)` under
  `is_multi_gpu`; the single-GPU `"auto"` path is untouched. The
  `find_unused_parameters=True` source string is preserved (`test_ddp_warm.py:169`
  still passes); a new assertion covers `broadcast_buffers=False`.
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
