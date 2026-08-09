# DevicePolicy — the single owner of the per-rank pre-PG CUDA device (ADR-0031 §A2 landed)

ADR-0031 split the device-ownership problem in two: **A1** (frozen arms inside
`Trainer.fit`) is owned by Lightning via `nn.Module` registration + dual-exclude;
**A2** (data-pipeline models staged *before* the process group exists — `_real_inputs`
rollouts, the fake-cache builders, the post-PG VAE cache-warm) cannot be
Lightning-managed, because at that moment `self.device` is still `cuda:0` and
`LOCAL_RANK` is only readable from the launcher environment. ADR-0031 *specified* a
centralized `DevicePolicy` for A2 but **never landed it** — its own text contradicts
itself on whether the module already existed (see *ADR-0031 contradiction* below).
This ADR lands it: `DevicePolicy` is a constructed object that answers **only** "which
GPU does this rank use", nothing else. The pre-PG device decision moves out of the
open-coded `set_device` twins and the bare `cuda:0` literals and into one place.

## The contract (three methods, one seam)

`training/device_policy.py` — a constructed object (the repo OOP rule forbids free
functions; A2's policy is a `Policy` role). Construction is **side-effect free**:
`__init__` snapshots the launcher's `LOCAL_RANK` env var (missing → `0`) and touches
neither CUDA nor the process group. The snapshot is taken at construction; post-PG
callers do **not** re-resolve via `dist.get_rank()`.

- **`pin() -> torch.device`** — the pre-PG call for the three training shells. Performs
  the one-time `torch.cuda.set_device(local_rank)` side effect and returns
  `cuda:{local_rank}`. The out-of-range guard (`local_rank >= device_count`) skips
  `set_device` (the twin's tolerance semantics); when CUDA is unavailable it returns
  `cpu` untouched.
- **`device() -> torch.device`** — the side-effect-free read of the same device, for
  read-only / debug paths (never calls `set_device`).
- **`warm_device(fallback) -> torch.device`** — the post-PG VAE-warm device
  resolution. `torch.distributed` is imported lazily *inside the method body* so
  importing this module never touches the process group; under an initialized PG with a
  CUDA `fallback` it returns `cuda:{local_rank}`, otherwise the `fallback` unchanged.

The staged-onto-device *how* (`.to(device).eval()` + `requires_grad_(False)`) is
explicitly **not** DevicePolicy's job — that is the FrozenArm staging chain (ADR-0031
A1), and the two candidates stay decoupled so each can be deleted independently.

## Why

- **The A2 fix is a real, pre-existing bug, not new abstraction.** The controlnet
  supervised shell resolved a *bare* `torch.device("cuda" if torch.cuda.is_available()
  else "cpu")` — i.e. `cuda:0` — and staged its frozen base UNet + the fresh ControlNet
  onto it *before* `LOCAL_RANK` was knowable. Under a torchrun launch every rank piled
  onto GPU 0, reproducing the exact PR #156 cuda:0 contention / DDP-init-timeout class
  ADR-0031 §A2 promised to close. Routing the real path through `DevicePolicy().pin()`
  (per-rank `cuda:{LOCAL_RANK}`) closes the gap; the `data_provider` CPU smoke stays
  `DevicePolicy`-free (`torch.device("cpu")`) and transparent to the existing smokes.
- **The grpo + reward shells carried a character-identical pre-PG `set_device` twin.**
  Same process-level side effect, same "why not cuda:0" comment, copy-pasted across two
  shells with no shared seam. Routing both through `pin()` is de-duplication, not a
  behavior change — the out-of-range guard and the no-CUDA fallback are preserved.
- **The post-PG VAE-warm device resolution was a misplaced free function.**
  `latent_pipeline.resolve_warm_device` was a module-level free function (a direct
  violation of the repo OOP rule: only console `main` may be a module-level function).
  Its two call sites — the JiT `_warm_data` warm and the controlnet `_real_inputs`
  warm — both move to `DevicePolicy.warm_device(fallback)`, and the free function is
  **deleted** (this ADR's close-out). Behavior is byte-identical for every production
  launch (a torchrun launch always sets `LOCAL_RANK`).
- **One fact source for "this rank's GPU".** Both the pre-PG `pin` and the post-PG
  `warm_device` resolve `LOCAL_RANK` from the same construction-time snapshot, so the
  per-rank device has exactly one owner instead of being re-derived from
  env-var / `dist.get_rank()` at each site.

## Considered options (rejected)

- **A single post-PG `DevicePolicy` for both A1 and A2.** Rejected as a *timing
  contradiction* (already argued in ADR-0031): A2 stages models before `Trainer.fit`, so
  a policy that reads `LOCAL_RANK` "after Lightning assigns it" either falls back to
  `cuda:0` during `_real_inputs` (reproducing PR #156) or reads the env var pre-fit
  (violating the post-PG premise). A2 needs the env-var snapshot; A1 needs Lightning
  registration. One mechanism cannot cover both.
- **Fold the FrozenArm staging chain into `DevicePolicy`.** Rejected: it would couple
  two independent refactor candidates (the frozen-arm staging chain is ADR-0031 A1's
  territory) and bloat the policy past its single "which device" responsibility. The
  staging chain stays where it is; `DevicePolicy` returns a device, the caller stages.
- **Re-resolve the rank post-PG via `dist.get_rank()` inside `warm_device`.** Rejected:
  the deleted free function fell back to `dist.get_rank()` only when `LOCAL_RANK` was
  unset — an edge case unreachable in production (a torchrun launch always sets it).
  The construction-time snapshot is stable across the policy's lifetime and avoids
  touching the process group needlessly; the unreachable unset-`LOCAL_RANK`+PG-up edge
  simplifies to the snapshot's `0`, matching the no-post-PG-re-resolution contract.
- **Put `DevicePolicy` on the CPU smoke (`data_provider`) path.** Rejected: the smoke
  must stay pure-CPU, GPU-free, and launcher-env-free. `DevicePolicy` is constructed
  only on the real path; the smoke stays `torch.device("cpu")`, so the existing CLI
  smokes are untouched.

## Consequences

- **`grpo_cli` / `reward_cli`** (`main`): the character-identical `set_device` twin is
  deleted; `device = DevicePolicy().pin()`. Behavior (out-of-range guard, no-CUDA
  fallback, the comment's intent) preserved verbatim. (Landed T2 / #205.)
- **`controlnet_cli`** (`main`): the bare `cuda:0` device is replaced by
  `device = DevicePolicy().pin()` on the real path — **the core bug fix**. The
  `_real_inputs` warm closure moves off `resolve_warm_device` onto
  `DevicePolicy().warm_device(device)`. The `data_provider` CPU smoke stays
  `torch.device("cpu")`. (Pin landed T3 / #206; warm migration landed T5 / #208.)
- **`cli` (JiT)** (`_warm_data`): pre-PG stages nothing on GPU (the VAE is built on CPU;
  the warm is deferred to `DataModule.setup()`). Its warm closure uses
  `DevicePolicy().warm_device(device)`. (Landed T4 / #207.)
- **`export_cli`** is **out of scope** — CPU, no `Trainer`, not a DDP training shell. No
  `DevicePolicy`, no DDP/`Trainer` concepts introduced for it.
- **`Trainer(accelerator="auto", devices=<int>)` explicitness is unchanged.**
  `DevicePolicy` resolves the pre-PG *staging* device; Lightning still owns the
  in-`Trainer` device selection. The two are not coupled.
- **`latent_pipeline.resolve_warm_device` is deleted**; the module-level `os` / `dist`
  imports are retained (still used by other functions). The JiT + controlnet warm call
  sites are the only consumers; after migration there are zero runtime references (the
  remaining `resolve_warm_device` mentions are docstring provenance notes and the
  regression-guard assertion strings that prevent the old name from returning).
- **Smoke transparency.** No existing CLI smoke test is modified: `DevicePolicy` is
  constructed only on the real path, so the `data_provider`-injected CPU smokes see no
  `DevicePolicy`, no GPU, and no launcher env.
- **Tests.** `DevicePolicy` is a directly unit-testable object (env-var / `dist`
  monkeypatch, no real GPU): `pin` calls `set_device` exactly once with the out-of-range
  guard; `device` is side-effect-free; `warm_device` reproduces the deleted free
  function's PG-gated semantics. The three shells' wiring is locked by source-level
  guards (`inspect.getsource`): grpo/reward/controlnet `main` call `pin()`; controlnet
  no longer resolves a bare cuda literal; the JiT + controlnet warm closures use
  `DevicePolicy.warm_device` and not the deleted free function. A 2-rank `torchrun` smoke
  asserts the controlnet `_real_inputs` stages onto the per-rank device (the PR #156
  regression class — ADR-0031 required it and it had never landed).

## ADR-0031 contradiction (clarified)

ADR-0031's §A2 text is internally contradictory: one passage calls `device_policy.py`
an *"existing orphan (zero importers)"* (the PR #156 fix that was never wired), while a
later passage says it was *"created/landed as part of A"* (absent from `HEAD`). The
truth is the second one: **`device_policy.py` was never committed before this ADR.**
This ADR creates the object, lands the three-method contract, wires the three shells,
and deletes the misplaced `resolve_warm_device` free function — so the doc and the code
agree. A pointer note is added at the top of ADR-0031 (append-only; its merged body is
unchanged).

## Out of scope (deferred)

- **The frozen-arm staging chain** (`.to(device).eval()` + `requires_grad_(False)`) —
  ADR-0031 A1's territory; deliberately decoupled from `DevicePolicy`.
- **`Trainer(accelerator/devices/strategy)` explicitness** — the spine-collapse /
  `TrainingSpine` candidate family (ADR-0032), not the per-rank staging decision.
- **The export shell** — CPU, no `Trainer`, not a DDP training shell.
