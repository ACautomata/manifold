# FID callback split — metric granularity, composable helpers, collective-count hardening; materializer deferred

`FIDCallback` stays **one callback owning one metric** (`val/fid` + the per-plane
`val/fid_{xy,yz,zx}`). The ~18 KB mega-callback body shrinks by delegating its
pipeline stages to **composable helpers** (objects, not callbacks, not registered):
`FixedSampleRollout`, `LatentDecoder`, `FeatureExtractor`,
`SufficientStatsReducer`, `VramStage`. Single-responsibility is achieved at
**metric granularity**, not pipeline-stage granularity. Alongside the split, the
callback **hardens a pre-existing DDP collective fragility** (no `except` around
the per-plane `all_reduce`s; staging outside the `try`; unguarded mid-loop feature
extraction) by enforcing a **collective-count invariant**: any rank-local
exception during the staged phase `all_reduce`s an error flag so every rank takes
the same abort branch together, instead of diverging into a deadlock. A shared
`FixedSampleMaterializer` + decoded-image cache (the would-be efficiency win for
multiple generative metrics) is **deferred** — PSNR/SSIM do not exist yet
(ADR-0028 deleted the paired path), so building it now is speculative.

This is phase C of the four-point architecture refactor (issue #157), building on
the callback registry (ADR-0029): `FIDCallback` is registered as `FIDSpec`, and
the helpers are composable objects its `build(ctx)` constructs.

## Why

- **The mega-callback bundles ~7 concerns behind one `on_validation_epoch_end`.**
  Generation, VAE decode, 2.5D feature extraction, the per-plane sufficient-stats
  `all_reduce`, the Fréchet score, and the VRAM stage/restore all live in one
  method. The brief's complaint is that this is too coarse — PSNR/SSIM, when they
  return, would land in the same blob.
- **Metric granularity, not stage granularity (the central decision).** A
  pipeline-stage split — `FixedSampleRollout` → `Decode` → `FeatureExtract` →
  `FrechetScore` as separate callbacks communicating via `trainer.callback_state`
  — is **structurally broken**: it fractures the symmetric DDP collectives (the
  per-plane `all_reduce`, the disable-flag `all_reduce`, the `_feat_dim` probe)
  across callbacks, so one rank can skip or reorder a collective while others
  block in it (ADR-0025's symmetry is load-bearing here). Single-responsibility is
  the goal; the unit is **one metric per callback**, with the pipeline stages as
  reusable helpers invoked within one call stack (no inter-callback channel, no
  stale-state risk).
- **The split is the moment to fix the latent collective fragility (the highest-
  value part of C).** Adversarial review surfaced that the current
  `on_validation_epoch_end` is a `try:/finally:` with **no `except`**; the
  `_stage_eval_on_device()` call sits **outside** the `try`; and the
  `_synth_moments` loop's `get_features_2p5d` is **unguarded**. Any rank-local
  exception (a mid-loop OOM, a non-factory staging failure) therefore propagates
  after the `finally` restore without entering the disable-flag `all_reduce`, so
  the failing rank exits while the others block in the next collective — a
  deadlock. It has never bitten (rank-divergent exceptions are rare; the "8-DCU
  slow ≠ deadlock" discipline holds), but it is real. The split rewrites exactly
  this code, so the collective-count invariant is added now: generalize the
  existing disable-flag `all_reduce` (which already covers backbone-factory
  failure) into an **error-flag `all_reduce`** covering any exception in the
  staged phase, so all ranks abort together (collective fail-fast, consistent
  with the simplicity-bias "fail fast when anything unexpected happens"; strictly
  better than a latent deadlock). Happy-path `val/fid` is unchanged.
- **PSNR/SSIM are slots, not a commitment.** Their source was deleted with the
  paired bridge (ADR-0028); only a stale bytecode cache remains. Each will land
  as its own single-responsibility callback reusing `FixedSampleRollout` +
  `LatentDecoder`; whether they share generation via a `FixedSampleMaterializer`
  is decided when they actually exist (see "Out of scope").

## Considered options (rejected)

- **Pipeline-stage-per-callback** (`rollout`/`decode`/`extract`/`score` as
  separate callbacks over `trainer.callback_state`): rejected as **structurally
  fatal**. It splits the symmetric per-plane `all_reduce`, the disable-flag
  `all_reduce`, and the `_feat_dim` probe across callbacks whose execution order
  and partial-failure semantics Lightning does not guarantee — rank-asymmetric
  entry into a collective deadlocks (ADR-0025). Single-responsibility is achieved
  at metric granularity instead; the stages become helpers.
- **Build the `FixedSampleMaterializer` + decoded-image cache now** (generate+decode
  once per epoch, shared by FID/PSNR/SSIM): rejected as **premature** (YAGNI).
  PSNR/SSIM do not exist; the only generative metric today is FID. Building the
  materializer now would also reintroduce the inter-callback data channel + ordering
  dependency that metric-granularity deliberately avoids, and leaves the decoded-image
  device/VRAM question (CC14/CC23) unspecified. Deferred to when a second generative
  metric actually arrives.
- **Change the per-plane skip semantics** (require all three planes, or log a
  `val/fid_partial_planes` warning when a plane has global `n<2` and returns
  `None`): rejected as a **metric-definition change** outside a structure
  refactor. The current behavior (mean of the available planes, silently skipping
  `None`) is documented in place; altering it would change `val/fid` values and
  break the behavior-preservation contract.
- **Drop `val/fid_{xy,yz,zx}` to "really" have one logged key:** rejected — the
  per-plane breakdown is one responsibility (FID with per-plane decomposition;
  the whole pipeline exists to produce it). Splitting them would multiply
  generation cost 4× for no locality gain.

## Consequences

- **New composable helpers** under `src/manifold/metrics/` (or a `fid/` subpackage):
  `FixedSampleRollout` (rank-strided seeded generation — `seed + i` for
  `i % world == rank`, preserving the Fixed-sample re-seed-every-epoch mechanism),
  `LatentDecoder` (VAE decode with the `norm_float16` handling and float32-on-GPU
  semantics), `FeatureExtractor` (the three-plane 2.5D extraction via the injectable
  `feature_net`), `SufficientStatsReducer` (the symmetric per-plane `all_reduce` of
  sufficient stats to global `(μ, Σ, n)` — ADR-0025; zero stats for an empty shard so
  the collective cannot deadlock), and `VramStage` (a context manager: `__enter__`
  snapshots the VAE CPU state **before** moving VAE+`feature_net` to GPU and
  lazily builds the backbone fail-safe; `__exit__` unconditionally restores both to
  CPU — the `_eval_staged`-flag + `finally` pattern, encapsulated). `__enter__`
  carries its own cleanup-on-error path: Python does **not** call `__exit__` when
  `__enter__` raises, so a failure after the VAE has moved to GPU (an OOM, a
  backbone build or feature-dim probe error) must restore from within `__enter__`
  before re-raising — otherwise the VAE occupies training VRAM for the rest of the
  run. The `with VramStage(...)` site is the single entry point, so this is local
  to the helper.
- **Helpers are objects, not module-level bare functions** (project OOP rule).
  `FixedSampleRollout`/`LatentDecoder`/`FeatureExtractor`/`SufficientStatsReducer`
  are stateless callable objects; `VramStage` is necessarily stateful across
  `__enter__`/`__exit__`. They are **not registered** as callbacks — `FIDCallback`
  alone sequences them in one `on_validation_epoch_end` call stack, so data flows
  as local variables (no inter-callback channel).
- **`FIDCallback` keeps its current constructor** (low-level DDP tests in
  `tests/ddp.py`, `tests/test_ddp_metrics.py` construct it directly with the
  runtime objects). `FIDSpec` (an `@dataclass` `CallbackSpec` per ADR-0029, knobs
  `num_synth` / `every_n_epochs` / `center_slices_ratio` / `cov_ridge` / `seed`)
  is an **additional** build path whose `build(ctx)` extracts the runtime objects
  from `CallbackContext` and calls the same constructor. The lazy `real_latents`
  pull (ADR-0017 / F5) is preserved — `build` passes `real_latents=None`,
  `real_latents_source=ctx.datamodule`.
- **Collective-count invariant (the hardening).** The error-flag `all_reduce` is a
  **rendezvous before each reduction-bearing phase**, not an exception-path
  afterthought. Naively wrapping the staged phase in `try/except` and
  `all_reduce`-ing an error flag on a rank-local exception is **insufficient**: the
  failing rank would enter the error-flag collective while healthy ranks continue
  into `SufficientStatsReducer`'s per-plane moment reductions — collectives in a
  different order, still a hang. So the staged phase is structured as: (1) run the
  fallible rank-local work (staging, real/synth feature extraction) under `try`; on
  any exception set a local error flag; (2) **before** entering any
  `all_reduce`-bearing reduction, `all_reduce(MAX)` the error flag so **every rank
  agrees** whether to take the abort branch (`val/fid=+inf` / skip) or proceed into
  the reductions together. This makes the count and order of collectives identical
  on every rank in every path. It generalizes the existing disable-flag
  `all_reduce` (backbone-factory failure) to **any** exception. The
  `SufficientStatsReducer` stays symmetric (every rank enters one `all_reduce` per
  plane, zero stats for an empty shard). Collective-count invariance — every rank
  enters the same collectives in the same order in every code path — becomes a
  testable property.
- **`val/fid` behavior is preserved.** Same unbiased-Frechet math, same all-reduce,
  same fixed samples, same per-plane breakdown. The split is a refactor, not a
  metric change (verified by the existing integration + DDP tests passing unchanged).
- **Tests.** Existing CLI integration tests and the low-level FID DDP tests are
  unchanged (they construct `FIDCallback` directly / assert `trainer.callbacks`
  membership; the injectable `feature_net` fake is still the test seam). **New**
  unit tests cover each helper (rollout seeding, decode dtype, reducer symmetry,
  `VramStage` restore-on-exception) and a **new** test asserts the collective-count
  invariant under a forced rank-local exception (the hardening). The "no new seams"
  phrasing in the original spec is amended: helpers are stateful objects and get
  their own unit tests; the single **integration** seam (`run_training`) is
  unchanged.

## Out of scope (deferred)

- **`FixedSampleMaterializer` + decoded-image cache** — deferred until a second
  generative metric exists. When PSNR/SSIM return, decide then whether they share
  one generation (accepting an inter-callback channel + ordering dependency +
  decoded-image device/VRAM policy) or each reuse the helpers independently.
- **PSNR/SSIM callbacks** themselves — slots only (ADR-0028 deleted the paired
  path). Each lands as its own single-responsibility callback reusing
  `FixedSampleRollout` + `LatentDecoder`; never folded into `FIDCallback` or each
  other.
- **Per-plane skip / `every_n_epochs` cache-staleness / world-size-stability
  semantics** — documented as the current behavior (the real reference is
  immutable; the cache is reused across skip epochs; rank-striding assumes stable
  `world_size`), not changed. A future ADR can revisit if a metric-definition
  change is wanted.
- **Checkpoint-resume callback-list compatibility** — unchanged from ADR-0029
  (deferred); `FIDCallback` remains one callback, so its resume footprint is
  unchanged by C.
