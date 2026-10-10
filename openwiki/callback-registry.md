---
type: Reference
title: Callback registry and training spine
description: CallbackRegistry two-phase resolve/build, the spec contract, post-resolve monitor validation, and TrainingSpine.run as the single caller that composed the five training CLIs (ADR-0029 + ADR-0032).
tags: [callbacks, registry, training-spine, ADR-0029, ADR-0032, ADR-0037]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-10T14:21:58.700Z
sources:
  - id: openwiki-source-45e03b8e93e2bb8574e8832b
    resource: repo://docs/adr/0029-callback-registry.md
  - id: openwiki-source-d27066d757bcec132494314d
    resource: repo://docs/adr/0032-cli-spine-collapse.md
  - id: openwiki-source-1f077c4de341ab7dab070abb
    resource: repo://docs/adr/0037-controlnet-fidelity-in-training-monitor.md
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-812f4876e72062b85443a7c2
    resource: repo://src/manifold/training/callbacks/__init__.py
  - id: openwiki-source-cf5e8257aac5d633b1ffd363
    resource: repo://src/manifold/training/callbacks/checkpoint.py
  - id: openwiki-source-47bfda18a8686819d04dc2de
    resource: repo://src/manifold/training/callbacks/context.py
  - id: openwiki-source-b7630a11a03e932cc7e14edc
    resource: repo://src/manifold/training/callbacks/fid.py
  - id: openwiki-source-4a976a922e9a1286c789e272
    resource: repo://src/manifold/training/callbacks/paired_fidelity.py
  - id: openwiki-source-bf73652a42104a8f9ef7e65e
    resource: repo://src/manifold/training/callbacks/registry.py
  - id: openwiki-source-de92ff3ec6c03380f90b6d65
    resource: repo://src/manifold/training/callbacks/train_loss.py
  - id: openwiki-source-cb9c9c415ab8026b88012d9e
    resource: repo://src/manifold/training/cli.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
  - id: openwiki-source-be99bd216fffaa3e1d9212c0
    resource: repo://src/manifold/training/core.py
  - id: openwiki-source-4bdb3113f633b5c6a86fd01d
    resource: repo://src/manifold/training/grpo_cli.py
  - id: openwiki-source-b4f781de2a8cfd17bbaf7eef
    resource: repo://src/manifold/training/reward_cli.py
  - id: openwiki-source-36db33c03ac89b676295a74d
    resource: repo://tests/test_callback_registry.py
  - id: openwiki-source-4e3bcc3726ac2935c40ee848
    resource: repo://tests/test_paired_fidelity_callback.py
generated: { by: "openwiki/0.7.2", at: "2026-10-10T14:21:58.700Z" }
---

# Callback registry and training spine

`CallbackRegistry` (ADR-0029) is the typed name → spec dispatcher that replaced
the ad-hoc `_build_callbacks` / `_build_checkpoint` clones that used to live in
every training CLI. `TrainingSpine` (ADR-0032) is the **single caller** of that
registry: it owns the `assemble → resolve → build → validate → trainer → fit`
sequence, plus the post-merge `forbidden_callbacks` / `forbidden_monitors` guards
that let a shell drop a callback no YAML or `--callbacks` override can re-enable.
Every training CLI now ends with `spine.run(...)`; the CLI's only responsibility
left is seeding, building the module + datamodule, and computing its dynamic
default callback-name set.

This page documents the spec contract, the two-phase construction, the
post-resolve monitor validation, and the spine's merge order — the load-bearing
pieces of any callback addition, knob rename, "drop FID on the GRPO ControlNet
path" change, or paired-fidelity-monitor extension (the supervised ControlNet
stage already ships with `PairedFidelitySpec`, ADR-0037 / issue #238; the
ControlNet-GRPO stage is a deliberate follow-up).

## Spec contract

Every callback registered with the registry is a `@dataclass` class with two
things (`src/manifold/training/callbacks/registry.py`):

1. **Knobs as dataclass fields** (with defaults). The field set is the
   allow-list used by `CallbackRegistry.resolve` to reject unknown knobs with a
   loud `ValueError` before any `pl.Callback` is built.
2. **A `build(self, ctx: CallbackContext) -> pl.Callback` method** that takes
   the runtime-objects bag and returns a constructed Lightning callback.

Specs match the `CallbackSpec` Protocol structurally — no inheritance
(`manifold.training.callbacks.registry.CallbackSpec`). A spec may optionally
expose:

- `logged_metrics: frozenset[str]` — the metrics the built callback logs. Used
  by `validate_monitor` so a `CheckpointSpec.monitor_metric` resolves against the
  full set of metric emitters, not just the registry ones. Declared as a
  `ClassVar` (e.g. `FIDSpec`, `PairedFidelitySpec`) so the field is **not** a
  config knob and does not enter `resolve`'s knob set.
- `monitor_metric: str | None` — the special-case field `validate_monitor`
  scans for. Only `CheckpointSpec` declares it.

## Two-phase construction

ADR-0029 splits spec instantiation from callback construction because
generative callbacks (FID, paired-fidelity) need runtime objects (the VAE,
inference recipe, and feature network) that do not exist at config resolution.
ADR-0037 adopts the same construction shape for the supervised ControlNet
`val/psnr` / `val/ssim` monitor — `PairedFidelitySpec` is registered and ships
in `controlnet_cli.run_controlnet_training`'s default callback-name set.

- **`CallbackRegistry.resolve(names, cfg)` (config-time)** — validates the
  requested name list and the per-name knob dict, then returns constructed spec
  instances in `names` order. Unknown name → `KeyError` with the registered
  set; unknown knob for a known name → `ValueError` with the allowed set.
  The list is **rank-symmetric** by contract (every DDP rank gets the same CLI
  args from `torchrun`).
- **`CallbackRegistry.build(specs, ctx)` (fit-prep)** — injects the runtime
  `CallbackContext` (the module, VAE, datamodule, inference recipe, model dir,
  seed, optional lazy `feature_net_factory`, optional `real_latents`) and
  returns the constructed `pl.Callback` list.

A callback that needs neither a knob nor a runtime object (e.g. `TrainLossSpec`)
is fine with empty fields and an empty `ctx`; the registry does not enforce a
minimum surface.

## Post-resolve monitor validation

`CallbackRegistry.validate_monitor(specs, module, extra_callbacks=None)` runs
**after** resolve/build, **before** the trainer is built. The contract:

- If no checkpoint spec is present, or its `monitor_metric` is `None`
  (the unmonitored periodic / last path), validation is a no-op — absence is the
  intended fallback when no held-out validation is wired.
- Otherwise the `monitor_metric` must be present in the union of:
  - every spec's `logged_metrics` (the registry path),
  - the `module.logged_metrics` attribute (the Module-declared path: GRPO's
    `val/mean_reward`, the reward module's `val/gen_pair_acc`),
  - every `extra_callbacks` member's `logged_metrics` (the hand-appended path:
    `LatentX0MAE`'s `val/x0_mae`, see `src/manifold/training/metrics.py`).

A `monitor_metric` that is logged by nobody raises `ValueError` with the
available set — so Lightning does not error mid-fit on a never-logged monitor.
The supervised ControlNet monitor `val/x0_mae` validates through the
`extra_callbacks` union (the shell appends `LatentX0MAE()` after the registry
callbacks); the observe-only `val/psnr` / `val/ssim` validates through the
registry path (`PairedFidelitySpec.logged_metrics`), but the checkpoint's own
`monitor_metric` keeps `val/x0_mae` so the monitor never displaces selection.

This validation is the safety net behind the GRPO ControlNet fid guard (see
below) and behind the `tests/test_callback_registry.py` monitor tests.

## Built-in specs

| Spec | Knobs (defaults) | Built callback | `logged_metrics` |
|---|---|---|---|
| `TrainLossSpec` | (none) | `pl.Callback` logging `train/loss_epoch` | `frozenset({"train/loss_epoch"})` |
| `FIDSpec` | `num_synth=16`, `every_n_epochs=1`, `center_slices_ratio=0.5`, `cov_ridge=1e-6` | `FIDCallback` (rank-strided, lazy feature-net) | `frozenset({"val/fid"})` |
| `PairedFidelitySpec` | `subset_size=8`, `every_n_epochs=1`, `num_inference_steps=None` (recipe-primary), `seed=0` | `PairedFidelityCallback` (observe-only ControlNet Heun rollout + VAE decode + `PairedFidelityMetrics`) | `frozenset({"val/psnr", "val/ssim"})` |
| `CheckpointSpec` | `monitor_metric`, `save_top_k`, `save_last`, `every_n_epochs`, `mode`, `filename` | `pl.callbacks.ModelCheckpoint` (monitored vs unmonitored two branches) | n/a (`validate_monitor` reads `monitor_metric` defensively) |

`CheckpointSpec.build` reproduces the prior `_build_checkpoint` two-branch
construction: `monitor_metric=None` keeps `save_top_k=1` plus `last.ckpt` at
the `every_n_epochs` cadence (the JiT production fallback); the monitored branch
tracks `monitor_metric` top-`k` plus last. The `filename` default picks
`unet3d-{epoch:03d}-{step}-{monitor:.3f}` for the monitored path and
`unet3d-{epoch:03d}-{step}` for the unmonitored path
(`src/manifold/training/callbacks/checkpoint.py`); the per-CLI shells override
this template with their own prefix (`reward-{...}`, `controlnet-{...}`,
`grpo-{...}`) computed against the **effective** (post-merge) monitor so a
YAML `monitor_metric` override does not leave a stale filename template.

`PairedFidelitySpec.build` is **recipe-primary** for the rollout step count:
`num_inference_steps=None` ⇒ read from `ctx.inference_recipe["num_inference_steps"]`,
which the supervised `controlnet_cli` fills from the existing
`controlnet.num_inference_steps` knob (issue #239). An explicit per-callback
knob overrides the recipe; with neither, the callback's own default (15)
applies.

## TrainingSpine.run — the merge order

`TrainingSpine.run` is a single method, parameterized by named arguments rather
than a per-shell subclass (composition, not inheritance — the project's OOP
rule). The merge order is:

1. Start from `default_names` (the per-shell dynamic default callback list,
   derived by the CLI from the resolved mode — e.g. the JiT path starts with
   `["train_loss"]` then appends `"fid"` when `enable_fid and val_enabled` and
   appends `"checkpoint"` always; the reward shell is
   `["checkpoint"]`; the GRPO shell is `["checkpoint"]` then appends `"fid"`
   when `fid_active`; the supervised ControlNet shell is
   `["train_loss", "checkpoint", "paired_fidelity"]` and appends
   `LatentX0MAE` via `extra_callbacks`).
2. `callback_cfg` knob dicts are applied to whatever specs resolve from those
   names.
3. `callback_names_override` **replaces** the name list entirely (the CLI
   `--callbacks` flag). A user can drop or add names here.
4. **`forbidden_callbacks` is applied AFTER the merge.** Each name in the map
   is force-removed from the merged list with a `rank_zero_info` log line. This
   is the load-bearing guard: a YAML knob or `--callbacks` override cannot
   re-enable a forbidden callback. The GRPO ControlNet path passes
   `forbidden_callbacks={"fid": "constant frozen-base metric"}` so the FID
   callback is dropped for that policy even if a stale recipe lists it.
5. **`forbidden_monitors` rejects a checkpoint `monitor_metric` BEFORE
   resolution.** The same GRPO ControlNet path passes
   `forbidden_monitors={"val/fid": ...}` so any checkpoint `monitor_metric:
   val/fid` raises `ValueError` instead of silently resuming on a constant
   metric.
6. Resolve → build → extend with `extra_callbacks` →
   `validate_monitor` against the union of registry specs, the module's
   `logged_metrics`, and the extra callbacks' `logged_metrics` → assemble
   `pl.Trainer` → `trainer.fit`.

```mermaid
flowchart TD
    A["default_names, per-shell dynamic list"] --> B["apply callback_cfg, per-name knob dict"]
    B --> C{"callback_names_override present?"}
    C -- yes --> D["replace name list"]
    C -- no --> E["keep merged list"]
    D --> F{"forbidden_callbacks match any name?"}
    E --> F
    F -- yes --> G["log and remove forbidden callback"]
    F -- no --> H{"forbidden_monitors match monitor_metric?"}
    G --> H
    H -- yes --> X1["raise ValueError"]
    H -- no --> I["registry.resolve, validate names and knobs"]
    I --> J["registry.build, inject CallbackContext"]
    J --> K["extend with extra callbacks such as LatentX0MAE"]
    K --> L["validate_monitor against logged metrics"]
    L --> M{"ModelCheckpoint in list?"}
    M -- no --> X2["raise ValueError, no ModelCheckpoint"]
    M -- yes --> N["build trainer with callbacks"]
    N --> O["trainer.fit"]
```
*Figure: `TrainingSpine.run` merge order — defaults → knobs → `--callbacks` override → `forbidden_callbacks` / `forbidden_monitors` → resolve → build → `extra_callbacks` → `validate_monitor` → fit.*

If no `ModelCheckpoint` survives the merge (an override dropped it),
`TrainingSpine.run` raises `ValueError("no ModelCheckpoint in resolved list")`
rather than letting `next(...)` raise `StopIteration` deeper in the call
(`tests/test_callback_registry.py::test_training_spine_fails_fast_without_checkpoint`,
codex #170 P2).

The forbidden-callbacks guard is the documented externalization of the
former "GRPO ControlNet fid" special case: it lives generically in the spine,
not in any GRPO vocabulary
(`tests/test_callback_registry.py::test_training_spine_forbidden_callbacks_force_removed_post_merge`,
ADR-0032).

## Change guidance

- **Adding a callback:** add a `@dataclass` spec under
  `src/manifold/training/callbacks/`, expose it from
  `src/manifold/training/callbacks/__init__.py`, and register it in each CLI's
  `spine.registry.register(...)` block. If the callback logs a monitored
  metric, declare it on `logged_metrics: ClassVar[frozenset[str]]` so
  `validate_monitor` accepts a `CheckpointSpec.monitor_metric` that points at
  it (the `FIDSpec` / `PairedFidelitySpec` pattern). Compose `CallbackContext`
  if the spec needs a runtime object not already in the bag.
- **Renaming a knob:** rename the dataclass field. `resolve` will start
  rejecting the old name in any recipe that still uses it — that loud
  `ValueError` is the contract's signal that all recipe sites need an update.
- **Adding a knob with a non-default:** same as above; recipes that omit it
  inherit the new default.
- **Dropping a callback for one policy:** prefer `forbidden_callbacks` over
  hard-coding the absent name into `default_names`. The forbidden map is the
  loud, post-merge, override-resistant guard. Pair it with
  `forbidden_monitors` when a *monitor* (not just a callback) must be
  unreachable.
- **Adding a new monitored metric:** make sure the emitting side declares it
  either as a spec's `logged_metrics`, the module's `logged_metrics`, or an
  `extra_callbacks` member's `logged_metrics`. Otherwise
  `validate_monitor` rejects any checkpoint that monitors it.
- **Extending `PairedFidelitySpec` to the ControlNet-GRPO stage (deliberate
  follow-up, not yet active):** the blanket `forbidden_callbacks={"fid"}` /
  `forbidden_monitors={"val/fid"}` on the GRPO ControlNet path does **not**
  cover paired fidelity — paired rollout uses the trainable ControlNet, so
  the unconditional-FID rationale does not apply (ADR-0037). Releasing the
  blanket for this callback alone, threading the held VAE on GPU for the
  decode, and registering the spec on the GRPO shell is the lifecycle for
  that change. The supervised stage ships first; the GRPO extension is a
  separate ticket with its own tests.

## Source map

- Spec contract + registry: `src/manifold/training/callbacks/registry.py`
- Runtime objects bag: `src/manifold/training/callbacks/context.py`
- Built-in specs: `src/manifold/training/callbacks/{train_loss,fid,checkpoint,paired_fidelity}.py`
- Built callback for the paired spec: `src/manifold/metrics/paired_callback.py`
- Spec barrel: `src/manifold/training/callbacks/__init__.py`
- Spine implementation: `src/manifold/training/core.py`
- CLI callers:
  - `src/manifold/training/cli.py` (JiT — registers `train_loss` / `fid` /
    `checkpoint`)
  - `src/manifold/training/reward_cli.py` (registers `train_loss` /
    `checkpoint`; default `["checkpoint"]`)
  - `src/manifold/training/grpo_cli.py` (registers `train_loss` / `fid` /
    `checkpoint`; uses `forbidden_callbacks` / `forbidden_monitors` on the
    ControlNet policy path)
  - `src/manifold/training/controlnet_cli.py` (registers `train_loss` /
    `checkpoint` / `paired_fidelity`; default
    `["train_loss", "checkpoint", "paired_fidelity"]`; appends
    `LatentX0MAE` via `extra_callbacks`)
- Trainer builder (called by the spine): `src/manifold/training/trainer.py`

## Focused tests

- `tests/test_callback_registry.py::test_resolve_rejects_unknown_name`,
  `test_resolve_rejects_unknown_knob` — config-time validation
- `tests/test_callback_registry.py::test_validate_monitor_rejects_orphan`,
  `test_validate_monitor_accepts_module_declared_metrics`,
  `test_validate_monitor_accepts_extra_callback_metrics`,
  `test_validate_monitor_fails_when_extra_callback_omits_metric` — post-resolve
  monitor validation paths
- `tests/test_callback_registry.py::test_paired_fidelity_spec_*`,
  `test_validate_monitor_accepts_psnr_and_ssim`,
  `test_paired_fidelity_spec_keeps_checkpoint_monitor_on_x0_mae`,
  `test_paired_fidelity_spec_reads_num_inference_steps_from_recipe` —
  the observe-only paired spec contract
- `tests/test_callback_registry.py::test_training_spine_fails_fast_without_checkpoint`
  — codex #170 P2
- `tests/test_callback_registry.py::test_training_spine_forbidden_callbacks_force_removed_post_merge`,
  `test_training_spine_forbidden_monitor_raises` — ADR-0032 forbidden guards
- `tests/test_paired_fidelity_callback.py` — wiring smoke for the observe-only
  in-training paired-fidelity monitor (decode → min-max → score → log)
- `tests/test_training_cli.py::test_*` covering CLI × spine integration

## Minimal validation

```bash
pytest tests/test_callback_registry.py -q
```

For a knob rename that touches one spec, also run that CLI's focused tests
(reward → `test_reward_cli.py`, JiT → `test_training_cli.py`, GRPO →
`test_grpo_cli.py` if present, ControlNet → `test_controlnet_cli.py`). For the
paired-fidelity monitor, also run `tests/test_paired_fidelity_callback.py` (the
unit-at-hook wiring smoke) and `tests/test_paired_fidelity.py` (the metric
identity / known-PSNR contract).
