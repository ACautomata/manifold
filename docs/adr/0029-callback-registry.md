# Callback registry — Protocol specs, two-phase resolve/build, registry-driven callbacks

A single `CallbackRegistry` maps a callback **name** to a `CallbackSpec` (a
`@dataclass` implementing a `build(ctx) -> pl.Callback` Protocol). The five training
CLIs stop hand-assembling callback lists; a `callbacks:` config list and a
`--callbacks` CLI flag both build the list through the registry, so config and CLI
never diverge. Construction is **two-phase** — `resolve` (config-time: validate names
+ knobs, fail-fast) and `build` (fit-prep: inject runtime objects via a typed
`CallbackContext`) — because generative callbacks (`FIDCallback`, future PSNR/SSIM)
need runtime objects (`module`, `vae`, `datamodule`, `inference_recipe`,
`feature_net_factory`) that do not exist at config resolution time.

This is phase B of the four-point architecture refactor (issue #157) — the
structural prerequisite that makes the FID split (C) and the CLI-spine collapse (E)
one-place changes instead of five-CLI sweeps.

## Why

- **Callbacks are hand-wired per CLI, with no config/CLI path.** Each of
  `cli.py`, `controlnet_cli.py`, `grpo_cli.py`, `paired_reward_cli.py`,
  `reward_cli.py` builds its own callback list inside its `run_training`/`run_*`.
  Only `FIDCallback` has config-threaded knobs (via a hardcoded `_dict_subset` key
  list). There is no way to add a callback from the YAML recipe or the command line.
  The registry is one interface consumed by all five CLIs — adding a callback
  becomes one spec + one `register` call, reachable from every training mode.
- **Deferred construction forces two phases (C1).** `FIDCallback.__init__` needs
  `module`, `vae`, `real_latents_source`, and the six `_inference_recipe`-derived
  params (`latent_shape`, `spacing`, `modality`, `num_inference_steps`,
  `guidance_scale`, `cfg_interval`). The first three are runtime objects built in
  `main()`; the last six are derived from the composed config. None exist at config
  resolution. A single-phase "name → instance" registry cannot construct FID; the
  resolve/build split with a typed `CallbackContext` carrying both runtime objects
  **and** `inference_recipe` is what makes deferred construction tractable.
- **Design-it-twice + adversarial verification picked the shape.** Three designs
  (deferred-factory context-bag; `@dataclass`-spec with `OmegaConf.structured`;
  ConfigMixin mirror) were broken against a ten-constraint gauntlet (deferred
  construction, five-CLI spine, the OOP rule, config/CLI parity, checkpoint-monitor
  coupling, DDP collective symmetry, ADR-0017 laziness, the single test seam,
  behavior preservation, the `build_trainer` boundary). Only a Protocol-spec +
  typed-context + `to_container`-binding hybrid survived without a structural
  rewrite; the rejected options are recorded below so a future review does not
  re-suggest them.

## Considered options (rejected)

- **ConfigMixin for callback knob declaration:** rejected. ConfigMixin is the
  **persisted component** contract (ADR-0004 — `register_to_config` /
  `from_config` round-trip into `config.json`). Callbacks are ephemeral — they are
  never saved as components — so ConfigMixin's persistence machinery is a concept
  mismatch and an artificial two-class split (a "config" ConfigMixin that never
  instantiates, plus the real Lightning callback). The surviving `configmixin`
  design flagged this as its own headshot weakness.
- **`OmegaConf.structured` / structured-`@dataclass` config binding:** rejected.
  The repo uses OmegaConf uniformly as `OmegaConf.to_container(block)` → `**kwargs`
  (see `config/builder.py::_block_kwargs`, `config/loader.py`); there is **no**
  `OmegaConf.structured` / structured-config precedent anywhere. Introducing it for
  callbacks alone fragments the config style. Specs stay `@dataclass` for typed
  in-code declaration, but bind via the existing `to_container` →
  filter-to-`dataclasses.fields` → construct path (unknown knob fails fast).
- **Bare callable factories at module top-level:** rejected — violates the project
  OOP rule (no module-level bare functions except the console `main`). Factories
  that are callable *instances* would satisfy the letter of the rule, but a spec
  class with typed config fields + a `build(ctx)` method is cleaner, type-checkable,
  and self-describing for the knob set.
- **A DDP rank-symmetry collective guard** (rank-0 broadcasts the resolved
  callback-name list hash; mismatch fails fast before any business collective):
  rejected. DDP launches (`torchrun` / Lightning spawn) give every rank an
  identical command line, so a rank-asymmetric `--callbacks` requires unusual
  manual scripting of a never-observed failure mode (the project has never had a
  real callback-asymmetry deadlock — ADR-0025's all-reduce validation is symmetric
  by construction; cf. the "8-DCU slow ≠ deadlock" discipline). A guard would add a
  collective for a speculative risk, against the simplicity bias. The constraint is
  documented instead: the resolved callback list **must** be rank-symmetric.
- **Dual-path `run_training`** (accept either pre-built callback instances *or*
  callback names): rejected as unnecessary. The existing tests already drive the
  `run_training` seam — `run_training(module=, bundle=_DataBundle(...))` builds
  callbacks internally and the integration tests assert on `trainer.callbacks`
  membership; the low-level callback/DDP tests construct the callback classes
  directly and bypass the registry entirely. So the registry lives **inside**
  `run_training` (replacing the hand-assembly), its signature is unchanged, and
  behavior preservation (C9) holds by construction — no second code path.

## Consequences

- **New package `src/manifold/training/callbacks/`:** `registry.py`
  (`CallbackRegistry`), `context.py` (the `CallbackContext` dataclass), and one
  module per spec (`train_loss.py`, `x0_mae.py`, `fid.py`, `checkpoint.py`, …).
  Package name without underscores (per the OOP rule); module filenames may use
  underscores.
- **`CallbackSpec` Protocol** — `build(ctx: CallbackContext) -> pl.Callback`, with
  each spec a `@dataclass` carrying its typed config fields + defaults. Composition,
  not inheritance (`pl.Callback` remains the sanctioned framework base class).
- **`CallbackContext`** — typed dataclass carrying `module`, `vae`, `datamodule`,
  `inference_recipe`, `feature_net_factory`, `model_dir`, `seed` (and CLI-specific
  runtime objects). Populated by `run_training` as the runtime objects materialize.
  This is the seam's payload — the honest scope of "what the CLI contributes beyond
  a Module factory and a default callback-name list" (the brief's per-CLI default
  list is **dynamic**, derived from bundle flags, not a static YAML list).
- **Runtime build order** (inside `run_training`): derive the default callback
  **name** set from bundle flags (`has_val`, `enable_fid`) → apply `callbacks:`
  YAML → apply `--callbacks` CLI (**CLI replaces the YAML list**, mirroring
  `OmegaConf.merge`'s list-replace semantics for dotlist parity) →
  `registry.resolve(names, cfg)` (fail-fast on unknown name / unknown knob) →
  assemble `CallbackContext` → `registry.build(specs, ctx)` → monitor validation →
  `build_trainer(callbacks=…)` → `Trainer.fit`.
- **`ModelCheckpoint` is a registered callback** (`name="checkpoint"`), not a
  special case. Its knobs declare the **full existing checkpoint config surface**
  — `monitor_metric` / `save_top_k` / `save_last` / `every_n_epochs` / `mode` /
  `filename` (the existing JiT/ControlNet recipes already use `save_last`, and
  ADR-0032 injects `filename` metadata, so every existing and injected field is
  declared — strict unknown-knob validation does not reject the live configs).
  Its one special behavior is a **post-resolve monitor validation**: the
  `monitor_metric` must be in the resolved set's logged-metrics **union the
  training module's declared metrics**. The union is required because reward /
  paired-reward / GRPO-without-FID monitors (`val/gen_pair_acc`,
  `val/mean_reward`) are logged by the `RewardModule` / `GRPOModule` directly, not
  by any resolved callback — a callbacks-only validation would reject these valid
  default monitors. Absence in both sets fails fast. The runtime
  graceful-degradation path is preserved — when `FIDCallback`'s backbone fails it
  logs `val/fid = +inf` and `mode='min'` falls through to `save_last` (the
  absent-vs-disabled distinction: a missing monitored callback is an error; a
  present-but-disabled one is the intended fallback).
- **Lazy `real_latents` preserved (ADR-0017 / F5).** `FIDSpec.build` passes
  `real_latents=None`, `real_latents_source=ctx.datamodule`; the post-build
  `fid._real_latents_source = datamodule` mutation in `run_training` is removed.
- **Config namespace alignment:** the `fid_eval` YAML block is renamed to `fid` so
  the callback name and its config namespace are identical everywhere (no
  `config_prefix` indirection). The rename carries a **legacy-override
  translation**: the JiT CLI today reads `fid_eval.save_top_k` as a
  higher-precedence `checkpoint.save_top_k` override (`cli.py:414`). After the
  rename, a user's `fid.save_top_k` would fail `FIDSpec`'s strict unknown-knob
  validation (`save_top_k` is a checkpoint knob, not an FID one). So the shell
  translates `fid.save_top_k` → `checkpoint.save_top_k` **before** registry
  resolution (or keeps a documented alias); existing experiment overrides must not
  stop launching.
- **`build_trainer` boundary unchanged.** The invariant framework callbacks
  (`spt.ModuleRegistryCallback`, `MetricsPlotCallback`) stay auto-appended by
  `build_trainer`; the registry owns only the project metric/lifecycle callbacks.
  No namespace collision — framework callbacks are not registry entries.
- **Tests.** CLI integration tests are unchanged (same bundle → same callback set →
  same `trainer.callbacks` assertions). Low-level callback/DDP tests are unchanged
  (they construct the callback classes directly and do not touch the registry). New
  single-seam tests assert config/CLI parity and the fail-fast cases (unknown name,
  unknown knob, absent monitor metric).

## Out of scope (deferred)

- **Resume / checkpoint callback-compatibility validation** (callbacks changed
  across a `.ckpt` resume) — not addressed; future work.
- **PSNR / SSIM specs** are slots only — they materialize when a paired path needs
  them (ADR-0028 deleted the paired bridge; the source `psnr_ssim_callback.py` is
  gone, only the stale bytecode cache remains). Each lands as its own
  single-responsibility callback, never folded into a neighbor.
- **The FID mega-callback split (candidate C)** and **the CLI-spine collapse
  (candidate E)** are separate phases that this registry enables; they get their
  own ADRs if they surface load-bearing decisions.
