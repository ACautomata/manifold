---
type: Reference
title: Frozen arms and per-rank device policy
description: FrozenArmMixin (register + dual-exclude off the optimizer / checkpoint) and DevicePolicy (the per-rank CUDA device decision that replaced resolve_warm_device and the pre-PG set_device twin).
tags: [frozen-arm, device-policy, ADR-0031, ADR-0035]
verified:
  - by: openwiki/0.9.0
    at: 2026-11-13T16:00:22.072Z
sources:
  - id: openwiki-source-39c3295efc089133e87a9c80
    resource: repo://CONTEXT.md
  - id: openwiki-source-bdb26e011dbdfd0846537627
    resource: repo://docs/adr/0031-device-ownership.md
  - id: openwiki-source-0b3288e3dba4430c1bb2668c
    resource: repo://docs/adr/0035-devicepolicy-per-rank-device.md
  - id: openwiki-source-aefad772862dded74740a681
    resource: repo://src/manifold/modules/controlnet_latent_flow.py
  - id: openwiki-source-487431944da97854e8cb6d33
    resource: repo://src/manifold/modules/frozen_arm.py
  - id: openwiki-source-ffd9f2f79ca81385e197ae7f
    resource: repo://src/manifold/modules/grpo.py
  - id: openwiki-source-82594a5d307aebca75251b76
    resource: repo://src/manifold/modules/reward.py
  - id: openwiki-source-cb9c9c415ab8026b88012d9e
    resource: repo://src/manifold/training/cli.py
  - id: openwiki-source-28b61e3219922e44e25b13ad
    resource: repo://src/manifold/training/controlnet_cli.py
  - id: openwiki-source-deca8f8f4a3d2dbbba47ac16
    resource: repo://src/manifold/training/device_policy.py
  - id: openwiki-source-4bdb3113f633b5c6a86fd01d
    resource: repo://src/manifold/training/grpo_cli.py
  - id: openwiki-source-b4f781de2a8cfd17bbaf7eef
    resource: repo://src/manifold/training/reward_cli.py
  - id: openwiki-source-c6b41726b73d1e18b82404a3
    resource: repo://src/manifold/training/trainer.py
  - id: openwiki-source-7131b0d80871bbec1bff1bff
    resource: repo://tests/test_controlnet_cli.py
  - id: openwiki-source-95fa8f50a78d587065883999
    resource: repo://tests/test_controlnet_module_training.py
  - id: openwiki-source-6db58ef83ad3e6a34ce75117
    resource: repo://tests/test_ddp_warm.py
  - id: openwiki-source-54aad16b0d9e880d1e1286dc
    resource: repo://tests/test_device_policy.py
  - id: openwiki-source-9c5f114a4443114e6c506f09
    resource: repo://tests/test_frozen_arm_mixin.py
  - id: openwiki-source-d4a3306929ca8168b983ec89
    resource: repo://tests/test_grpo.py
  - id: openwiki-source-f1dc960767df2c21c8576b02
    resource: repo://tests/test_reward.py
generated: { by: "openwiki/0.9.0", at: "2026-11-13T16:00:22.072Z" }
---

# Frozen arms and per-rank device policy

Two mechanisms own the cross-cutting "what is frozen / which GPU does it land on"
concern that every training CLI has to get right:

- **`FrozenArmMixin`** (`src/manifold/modules/frozen_arm.py`, ADR-0031 A1) — the
  shared "register + dual-exclude" implementation that keeps a frozen arm off
  the optimizer and off the checkpoint while letting Lightning own its device
  placement through the standard submodule machinery. Used by `GRPOModule` (the
  frozen `unet` base on the ControlNet-policy path, the frozen `reward_model`,
  and the KL-anchor arms `reference_unet` / `reference_controlnet`), by
  `ControlNetLatentFlowModule` (the frozen `unet` base), and by `RewardModule`
  (the frozen `denoiser`).
- **`DevicePolicy`** (`src/manifold/training/device_policy.py`, ADR-0035) — the
  per-rank CUDA device decision. Replaces both the duplicated pre-PG
  `set_device` twin that lived in the GRPO and reward training `main()`s and the
  `manifold.data.latent_pipeline.resolve_warm_device` free function used by the
  post-PG VAE warm. Three methods: `pin` (one `set_device` side effect pre-PG),
  `device` (side-effect-free read), `warm_device` (post-PG per-rank CUDA
  resolution, byte-identical to the former free function).

The two are paired concerns (one decides what is frozen, the other decides where
the GPU lands) and share an ADR-0035 test pattern: source-level guards via
`inspect.getsource` keep the retired free function / twin / mixin bypass from
sneaking back in.

Related pages: [Architecture and source map](architecture.md) (where the hosts
and shells sit), [Callback registry and training spine](callback-registry.md)
(the `TrainingSpine` that owns the fit), [Domain vocabulary and
decisions](domain-vocabulary-and-decisions.md) (the canonical *frozen arm* /
*DevicePolicy* terms and the ADR trail), [Quickstart](quickstart.md) (task
routing).

## FrozenArmMixin: register + dual-exclude

### Why this exists (ADR-0031 A1)

The previous scheme kept a frozen base UNet *unregistered* on the host via
`object.__setattr__`, hiding it from the module tree entirely. That worked for
the off-optimizer and off-checkpoint invariants but broke Lightning's automatic
`.to(device)`, forcing a manual staging step in `on_fit_start` and leaving the
UNet-policy and ControlNet-policy GRPO forks out of sync. ADR-0031 chose the
opposite direction: **register** the frozen arm as a normal `nn.Module`
submodule so Lightning owns its device placement, and **dual-exclude** it from
the optimizer and the checkpoint through targeted overrides.

### The mixin contract

```python
class XModule(FrozenArmMixin, spt.Module):
    def __init__(self):
        super().__init__()
        self._register_frozen_arm("frozen_name", arm)
        self.trainable = ...
```

Declare the host as `class XModule(FrozenArmMixin, spt.Module)` so the mixin
precedes `spt.Module`. The three dunder overrides chain to `nn.Module` via
cooperative `super()` because neither `spt.Module` nor
`pl.LightningModule` overrides `state_dict` / `load_state_dict` / `train`
(verified in `tests/test_frozen_arm_mixin.py`). The production MRO is exercised
by the three host modules' existing tests; the mixin's own tests use a minimal
`nn.Module`-based `_Host` to exercise the cooperative-`super()` contract
without dragging in Lightning's forward/optim machinery.

`_register_frozen_arm(name, arm)` does the uniform prep **once at construction**:

- `arm.eval()`
- `arm.parameters()` → `requires_grad_(False)`
- `setattr(self, name, arm)` (registers as a normal submodule)
- `self._frozen_arm_names |= {name}` (the arm set is fixed at init)

That single call is the only place the "register + dual-exclude" prep is
applied — adding a frozen arm cannot forget the freeze. The name set is a
`frozenset` class attribute by default (`frozenset()`), so a host with no frozen
arms degrades to a no-op instead of raising `AttributeError`.

```mermaid
flowchart TD
    Host["Host __init__"] --> Register["_register_frozen_arm name, arm"]
    Register --> Eval["arm.eval()"]
    Eval --> NoGrad["every param requires_grad False"]
    NoGrad --> Sub["setattr registers the arm as a normal submodule"]
    Sub --> Names["name added to _frozen_arm_names"]
    Names --> Place["Lightning .to(device) places the arm per rank"]
    Place --> Exclude["Dual exclude"]
    Exclude --> Opt["configure_optimizers selects the trainable arm only"]
    Exclude --> Ckpt["state_dict strips the frozen prefixes in place"]
    Exclude --> Load["load_state_dict strict on trainable keys, frozen allow-listed"]
    Exclude --> Mode["train re-applies eval to the frozen arms"]
```
*Figure: the register + dual-exclude chain — one registration call at construction, then four independent exclusions (optimizer, checkpoint emit, checkpoint load, train mode).*

### The three dunder overrides

- **`state_dict(destination, prefix='', keep_vars=False, ...)`** — strips the
  registered frozen arms **in place on the shared `destination`** (matching the
  arms under the caller-supplied `prefix`). The filter must mutate, not build a
  fresh dict, because a parent/wrapper that drives the recursion reads the
  `destination` it passed in (DDP's `module.state_dict(destination=..., prefix='')`
  discards the child's return value). Prefix-awareness is what stops a
  `module.frozen.*` key from leaking under the DDP wrapper prefix.
- **`load_state_dict(state_dict, strict=True, ...)`** — strict on the
  **trainable** keys, lenient only on the frozen arms. The checkpoint never
  carries frozen-arm weights (`state_dict` strips them); the arms are rebuilt
  fresh each launch (the reward from its own `.ckpt`, the reference policy via
  `deepcopy`, the ControlNet-policy base from the native export). The frozen
  arms being absent from the checkpoint is the one tolerated mismatch. A
  missing or unexpected **trainable** key still raises with the standard
  `RuntimeError("missing")` / `RuntimeError("unexpected")` (no blanket
  `strict=False`, which would resume on stale or random weights). Stray frozen
  keys left in a stale pre-refactor checkpoint are stripped before the load,
  and the checkpoint's `_metadata` is carried over.
- **`train(mode=True)`** — re-applies `eval()` on every registered frozen arm
  after `super().train(mode)`. Registration makes Lightning's recursive
  `train(True)` flip the arms to training mode; the override re-freezes them so
  BatchNorm running stats cannot drift during rollout / reward evaluation.

The mixin **does not own optimizer arm-selection**: "which arm is optimized"
stays in each host's `configure_optimizers` (GRPO → unet/controlnet, reward →
`reward_model`, ControlNet → `controlnet`). The mixin guarantees only the
`requires_grad=False` layer; the host's optimizer simply never selects the
frozen arms.

### Which host freezes what

| Host module | Frozen arms (registered + dual-excluded) | Trainable arm |
|---|---|---|
| `GRPOModule` (`src/manifold/modules/grpo.py`) | `reward_model` (always); `unet` — the frozen base, **only** on the ControlNet-policy path; `reference_unet` + `reference_controlnet` — the KL anchor, when a `reference_policy` is supplied | `unet` (UNet policy) or `controlnet` (ControlNet policy) |
| `RewardModule` (`src/manifold/modules/reward.py`) | `denoiser` — the frozen JiT x0-denoiser, when one is supplied | `reward_model` (the discriminator) |
| `ControlNetLatentFlowModule` (`src/manifold/modules/controlnet_latent_flow.py`) | `unet` — the frozen base UNet | `controlnet` |

Two naming traps worth stating explicitly:

- In `GRPOModule` the attribute name `unet` is **frozen** on the ControlNet
  policy path and **trainable** on the UNet-policy path (where it is a plain
  assignment, never a registered frozen arm). The frozen-arm set is derived
  from `controlnet is not None`, not from a constructor argument.
- `RewardModule`'s frozen arm is named **`denoiser`**, not `reference_unet` —
  `reference_unet` / `reference_controlnet` are GRPO's KL-anchor arms.

### MRO detail

Because the production host is `FrozenArmMixin, spt.Module` and the dunders
chain through `nn.Module`, the `_Host` test fixture uses the same ordering with
`nn.Module` as the base. This keeps the mixin's behavioral contract tests
independent of Lightning's forward / configure_optimizers machinery while
remaining a faithful exercise of the cooperative-`super()` chain.

### Acceptance matrix (FrozenArmMixin)

The mixin's behavior contract is locked down by `tests/test_frozen_arm_mixin.py`:

| Behavior | Test |
|---|---|
| Registration applies the uniform frozen-arm prep (eval + requires_grad=False) and adds to `_frozen_arm_names` | `test_register_frozen_arm_freezes_and_evals` |
| `state_dict()` strips the frozen arm in place on the shared `destination` | `test_state_dict_strips_frozen_arm_in_place` |
| `state_dict()` strips the frozen arm through the DDP wrapper prefix | `test_state_dict_strips_frozen_arm_through_wrapper_prefix` |
| `load_state_dict` strict on trainable, lenient on the frozen arm | `test_load_state_dict_strict_except_frozen_allowlist` |
| `module.train()` keeps the frozen arm in eval | `test_train_keeps_frozen_arm_in_eval` |
| Backward reaches the trainable arm; the frozen arm carries no grad | `test_backward_only_touches_trainable_arm` |
| The host's `configure_optimizers` wires the trainable arm only | `test_configure_optimizers_excludes_frozen_arm` |

The host-level proofs of "registered but dual-excluded" on the production MRO:

| Host | Proof | Test |
|---|---|---|
| ControlNet supervised | the base is present in `parameters()` (so Lightning owns device placement) but `requires_grad=False`, absent from the optimizer groups, and absent from `state_dict()` | `tests/test_controlnet_module_training.py::test_base_is_registered_but_dual_excluded` |
| GRPO | the reward + reference arms are registered, frozen, off-optimizer, off-checkpoint; the UNet-policy `unet` stays trainable | `tests/test_grpo.py::test_frozen_arms_registered_but_dual_excluded` |
| GRPO | frozen arms do not leak through a wrapper prefix (`module.*`) | `tests/test_grpo.py::test_state_dict_strips_frozen_arms_through_wrapper_prefix` |
| GRPO | strict load with the frozen allow-list; `train()` re-freezes the reward | `tests/test_grpo.py::test_load_state_dict_strict_except_frozen_allowlist`, `::test_train_keeps_frozen_arms_in_eval` |
| Reward | the frozen `denoiser` is in `parameters()` yet carries no grad and is stripped from the checkpoint | `tests/test_reward.py::test_module_backward_updates_discriminator_only` |

### The registered-arm cost (DDP)

Registration buys Lightning-owned placement but gives up one invariant the old
`object.__setattr__` bypass had for free: at `DistributedDataParallel`
construction every registered parameter — including the `requires_grad=False`
frozen arms — is broadcast from rank 0. The arms are rank-identical by
construction, so the broadcast is redundant-but-correct and one-time, but GRPO
pays it several times (base + reward + reference UNet + reference ControlNet),
which compounds the DDP-init timeout pressure. The documented mitigation is to
build the arms identically on every rank, not a DDP exclusion hack. DDP keeps
the default `broadcast_buffers=True` (a global `False` would let the *trainable*
reward model's BatchNorm stats diverge per rank), and `find_unused_parameters=True`
is unchanged.

### Known stale wording in `CONTEXT.md`

`CONTEXT.md` still describes the reward Module's frozen denoiser and the GRPO
Module's frozen reward model as **unregistered** ("holds the frozen JiT
x0-denoiser (unregistered, so off the checkpoint/optimizer)" and "frozen Reward
Model (unregistered, like the reward Module holds its denoiser)"). That wording
predates ADR-0031 A1: the implemented mixin **registers** both arms as normal
submodules and then dual-excludes them. Do not copy the stale "unregistered"
phrasing — the same file's *Frozen arm* entry already describes the
register + dual-exclude convention correctly, and the wiki tracks the drift in
its [stale-marker list](domain-vocabulary-and-decisions.md#stale-markers-in-the-domain-docs).
The same pre-ADR-0031 qualifier survives in source comments — "held
unregistered" in `src/manifold/training/export.py`,
`src/manifold/training/export_cli.py`,
`src/manifold/training/controlnet_cli.py`, and
`src/manifold/training/grpo_cli.py` — so read those as pre-ADR-0031 wording
too, not as a description of the current mechanism.

## DevicePolicy — the per-rank CUDA decision

### Why this exists (ADR-0035)

Three pre-existing call sites all duplicated the same answer to "which GPU does
this rank use":

1. The pre-PG `set_device` twin, character-identical across `grpo_cli.main`
   and `reward_cli.main` (issue #205, ADR-0035 T2).
2. The controlnet CLI's pre-PG `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
   that resolved to `cuda:0` for every DDP rank — the source of the PR #156
   `cuda:0` race / DDP-init-timeout category ADR-0031 §A2 promised to close
   (issue #206, ADR-0035 T3).
3. The `manifold.data.latent_pipeline.resolve_warm_device` free function used
   by the JiT `_warm_data` and the controlnet `_real_inputs` VAE warm closures
   (issues #207 / #208, ADR-0035 T4 / T5).

ADR-0035 collapses all three into a single `DevicePolicy` class with three
methods. The behavior is byte-identical to the prior code paths; the change is
de-duplication. `DevicePolicy` answers **only** *which* device — the
"how" of staging a model onto it (`.to(device).eval()` + `requires_grad_(False)`)
stays with the caller (the frozen-arm staging chain of ADR-0031 A1), and the two
stay decoupled so either can be deleted independently.

### Construction

```python
policy = DevicePolicy()  # side-effect free: snapshots LOCAL_RANK (missing -> 0)
```

`__init__` only reads `os.environ.get("LOCAL_RANK")` and stores the result. It
touches neither CUDA nor the process group. The snapshot is taken **once at
construction** — post-PG callers do **not** re-resolve via `dist.get_rank()`.
A `torchrun` launch always sets `LOCAL_RANK`; a single-process launch does not,
and the default 0 is correct in that case.

### `pin` — the one-time pre-PG side effect

```python
device = policy.pin()
```

This is the call the GRPO, reward, and controlnet training shells' `main()`
makes **before** Lightning initializes the process group:

- If `torch.cuda.is_available()`:
  - if `local_rank < torch.cuda.device_count()` → `torch.cuda.set_device(local_rank)` (the one side effect), then return `torch.device(f"cuda:{local_rank}")`.
  - else (out-of-range, the "twin guard") → **skip** `set_device`, return `cuda:{local_rank}`.
- Else → return `torch.device("cpu")`, no side effects.

The out-of-range guard mirrors the prior twin: if `LOCAL_RANK` exceeds the
device count (e.g. a misconfigured `CUDA_VISIBLE_DEVICES`), the rank still
gets the per-rank device value but does not call `set_device` (which would
raise). This is the "twice-bitten" guard: never silently fall back, never
raise on a misconfigured but recoverable launch.

```mermaid
stateDiagram-v2
    [*] --> Constructed: __init__ snapshots LOCAL_RANK, default 0
    Constructed --> Pinned: pin()
    Constructed --> ReadOnly: device()
    Pinned --> ReadOnly: device()
    ReadOnly --> ReadOnly: device()
    state Pinned {
        [*] --> CheckCuda
        CheckCuda --> CudaAvailable: torch.cuda.is_available()
        CheckCuda --> CpuFallback: no CUDA
        CudaAvailable --> InRange: local_rank below device_count
        CudaAvailable --> OutOfRange: local_rank at or above device_count
        InRange --> SetDeviceAndReturn: set_device(local_rank), return cuda local rank
        OutOfRange --> ReturnSkipSetDevice: skip set_device, return cuda local rank
        CpuFallback --> ReturnCpu: return cpu, no side effects
    }
    state ReadOnly {
        [*] --> CheckCuda2
        CheckCuda2 --> CudaRead: torch.cuda.is_available()
        CheckCuda2 --> CpuRead: no CUDA
        CudaRead --> ReturnCuda: return cuda local rank
        CpuRead --> ReturnCpu2: return cpu
    }
```
*Figure: `DevicePolicy` state transitions — `pin()` performs the one-time pre-PG `set_device` (with the twin's out-of-range guard), `device()` is the side-effect-free read. The post-PG `warm_device` resolution is described in prose below because it is driven by the process group's liveness rather than by the policy's own state.*

### `device` — side-effect-free read

```python
device = policy.device()
```

Returns the same `cuda:{local_rank}` (or `cpu`) without calling `set_device`.
It exists for read-only / debug paths where the caller does not want the side
effect. Note that no production shell calls it today — `pin()` and
`warm_device()` are the two methods with call sites — so it is contract-only
surface exercised by the unit tests.

### `warm_device` — post-PG VAE warm resolution

```python
device = policy.warm_device(launch_time_fallback)
```

This replaces `manifold.data.latent_pipeline.resolve_warm_device` (now
deleted). The semantics are byte-identical:

- `import torch.distributed as dist` happens **inside the method body**, so
  importing the module never touches the process group
  (`test_warm_device_dist_import_is_lazy` guards the contract).
- If `fallback.type == "cuda"` and `dist.is_initialized()` → return
  `cuda:{local_rank}` (the per-rank device from the construction-time
  snapshot, not a `dist.get_rank()` re-resolution).
- Otherwise → return the `fallback` unchanged (CPU fallback stays CPU;
  single-process fallback stays whatever the launch-time capture was).

There is **no `is_available()` check**: the gate is "CUDA fallback under a
live PG", so a CUDA fallback under a live PG resolves to `cuda:{local_rank}`
even when CUDA is unavailable on the box. This matches the former
`resolve_warm_device` semantics — verified by
`test_warm_device_pg_up_without_cuda_still_resolves_cuda_fallback`.

The launch-time `fallback` is captured in the shell's `main()` **before** the
PG is initialized, so under DDP it is the default `cuda:0` (the rank's
`CUDA_VISIBLE_DEVICES`-exposed index 0). After the PG is up (inside
`DataModule.setup()`), `LOCAL_RANK` names the rank's actual GPU and
`warm_device` returns it.

### CLI wiring

| Shell | Pre-PG device decision | Post-PG warm device |
|---|---|---|
| `src/manifold/training/grpo_cli.py` (`main`) | `DevicePolicy().pin()` on the real path (before `_real_inputs` runs the policy rollout on the GPU) | — (no VAE warm) |
| `src/manifold/training/reward_cli.py` (`main`) | `DevicePolicy().pin()` on the real path (before `_real_inputs` runs the frozen-denoiser rollout) | — (no VAE warm) |
| `src/manifold/training/controlnet_cli.py` (`main`) | `DevicePolicy().pin()` on the real path (before `_real_inputs` stages the frozen base + the fresh ControlNet) | `DevicePolicy().warm_device(device)` inside `_real_inputs`'s `warm_fn` |
| `src/manifold/training/cli.py` (JiT, `main` + `_warm_data`) | none — the launch-time device stays `torch.device("cuda" if torch.cuda.is_available() else "cpu")` and the VAE is built on CPU pre-PG, so nothing is staged on a GPU before the PG exists | `DevicePolicy().warm_device(device)` inside `_warm_data`'s `warm_fn` |

Two seams run through all of these:

- The **CPU smoke seam**: when a `data_provider` is injected, each shell takes
  `device = torch.device("cpu")` and never constructs a `DevicePolicy` — the
  smoke stays GPU-free and launcher-env-free.
- The **staging `how` is not the policy's job**: the controlnet `_real_inputs`
  still does `base.to(device).eval()` + `requires_grad_(False)` itself; the
  policy only names the device.

`src/manifold/training/export_cli.py` is deliberately out of scope (CPU-only, no
`Trainer`, no DDP shell).

### Acceptance matrix (DevicePolicy)

The behavior contract is locked down by `tests/test_device_policy.py`:

| Behavior | Test |
|---|---|
| `pin()` under LOCAL_RANK=k + CUDA available returns `cuda:k` and calls `set_device(k)` exactly once | `test_pin_returns_cuda_local_rank_and_sets_device_exactly_once` |
| Out-of-range LOCAL_RANK → no `set_device`, still returns `cuda:k` | `test_pin_out_of_range_local_rank_skips_set_device` |
| No CUDA → `cpu`, zero side effects | `test_pin_cpu_unavailable_returns_cpu_no_side_effects` |
| LOCAL_RANK unset → defaults to 0 | `test_pin_local_rank_unset_defaults_to_zero` |
| `device()` returns the same value as `pin()` but never calls `set_device` | `test_device_returns_cuda_local_rank_side_effect_free` |
| `device()` without CUDA → `cpu` | `test_device_cpu_unavailable_returns_cpu` |
| `warm_device` under live PG + CUDA fallback → `cuda:3` | `test_warm_device_local_rank_under_ddp` |
| `warm_device` non-CUDA fallback returned unchanged under DDP | `test_warm_device_non_cuda_fallback_returned_unchanged` |
| `warm_device` single-process fallback returned unchanged | `test_warm_device_single_process_fallback_returned_unchanged` |
| `warm_device` PG-up + CUDA-fallback + no CUDA still resolves (no `is_available` gate) | `test_warm_device_pg_up_without_cuda_still_resolves_cuda_fallback` |
| `import torch.distributed` is lazy (inside the method body, not at module level) | `test_warm_device_dist_import_is_lazy` |

The source-level guards keep the prior call patterns from sneaking back in:

| Guard | File | Locks |
|---|---|---|
| `test_grpo_cli_main_pins_via_device_policy_not_inline_set_device` | `tests/test_device_policy.py` | `grpo_cli.main` constructs a `DevicePolicy`, calls `.pin()`, and no longer inlines `set_device` |
| `test_reward_cli_main_pins_via_device_policy_not_inline_set_device` | `tests/test_device_policy.py` | same for `reward_cli.main` |
| `test_controlnet_cli_main_pins_via_device_policy_not_bare_cuda` | `tests/test_device_policy.py` | `controlnet_cli.main` pins through the policy and no longer resolves a bare `torch.device("cuda"...)` |
| `test_p1_warm_fn_uses_local_rank_device_not_launch_device` | `tests/test_ddp_warm.py` | the JiT `_warm_data` warm closure uses `DevicePolicy.warm_device`, not the deleted free function, and the VAE is still built on CPU pre-PG |
| `test_p1_controlnet_warm_uses_device_policy_not_resolve_warm_device` | `tests/test_ddp_warm.py` | the controlnet `_real_inputs` warm closure uses `DevicePolicy.warm_device`, not the deleted free function |

The end-to-end lock is a 2-rank spawn smoke
(`tests/test_controlnet_cli.py::test_main_real_path_pins_each_rank_to_local_gpu`):
`controlnet_cli.main`'s real path is driven under `torchrun`-style spawn with
`_real_inputs` stubbed to surface the device it receives, and the two ranks must
**disagree** on it (`cuda:0` vs `cuda:1`) — a bare `cuda:0` resolution would
collapse both onto GPU 0 (the PR #156 class).

## Source map

- Mixin implementation: `src/manifold/modules/frozen_arm.py`
- Mixin barrel export: `src/manifold/modules/__init__.py` (re-exports
  `FrozenArmMixin`)
- Hosts that consume the mixin: `src/manifold/modules/grpo.py`,
  `src/manifold/modules/controlnet_latent_flow.py`, `src/manifold/modules/reward.py`
- Device policy implementation: `src/manifold/training/device_policy.py`
- CLI shells that call `pin()`: `src/manifold/training/grpo_cli.py`,
  `src/manifold/training/reward_cli.py`,
  `src/manifold/training/controlnet_cli.py`
- VAE-warm closures that call `warm_device()`:
  `src/manifold/training/cli.py::_warm_data`,
  `src/manifold/training/controlnet_cli.py::_real_inputs`
- Retired free function: `manifold.data.latent_pipeline.resolve_warm_device`
  (deleted; only docstring provenance notes and the regression-guard assertion
  strings remain)
- Decision records: `docs/adr/0031-device-ownership.md` (A1 frozen arms, A2
  pre-Trainer staging), `docs/adr/0035-devicepolicy-per-rank-device.md`

## Focused tests

```bash
pytest tests/test_frozen_arm_mixin.py tests/test_device_policy.py -q
```

For a host wiring change (adding a frozen arm to a new module, switching a
host to `FrozenArmMixin`), also run the host's own tests:

- GRPO hosts: `pytest tests/test_grpo.py -q`
- ControlNet hosts: `pytest tests/test_controlnet_module_training.py -q`
- Reward hosts: `pytest tests/test_reward.py -q`

For a CLI shell change to the `pin()` / `warm_device()` wiring, also run:

```bash
pytest tests/test_ddp_warm.py tests/test_training_cli.py tests/test_controlnet_cli.py -q
```

The `inspect.getsource`-style guards in `tests/test_device_policy.py` and
`tests/test_ddp_warm.py` are the load-bearing acceptance: any regression that
re-introduces the inline `set_device` twin, the bare `torch.device("cuda")` in
the controlnet CLI, or the `resolve_warm_device` free function will fail those
tests loudly.
