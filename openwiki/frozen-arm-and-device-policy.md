---
type: Reference
title: Frozen arms and per-rank device policy
description: FrozenArmMixin (register + dual-exclude off the optimizer / checkpoint, Lightning owns placement) and DevicePolicy (the per-rank CUDA device decision that replaced resolve_warm_device and the pre-PG set_device twin).
tags: [frozen-arm, device-policy, ADR-0031, ADR-0035]
verified:
  - by: openwiki/0.7.2
    at: 2026-10-31T14:25:43.516Z
sources:
  - id: openwiki-source-bdb26e011dbdfd0846537627
    resource: repo://docs/adr/0031-device-ownership.md
  - id: openwiki-source-0b3288e3dba4430c1bb2668c
    resource: repo://docs/adr/0035-devicepolicy-per-rank-device.md
  - id: openwiki-source-f6308550d1ec17a2ebed8f58
    resource: repo://src/manifold/metrics/paired_callback.py
  - id: openwiki-source-a4a6e4b18ed70c8f8195546e
    resource: repo://src/manifold/metrics/vae_stage.py
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
generated: { by: "openwiki/0.7.2", at: "2026-10-31T14:25:43.516Z" }
---

# Frozen arms and per-rank device policy

Two cross-cutting mechanisms own the "what is frozen / which GPU does it land on"
concern that every training CLI has to get right:

- **`FrozenArmMixin`** (`src/manifold/modules/frozen_arm.py`, ADR-0031 A1) — the
  shared "register + dual-exclude" implementation that keeps a frozen arm off
  the optimizer and off the checkpoint while letting Lightning own its device
  placement through the standard submodule machinery. Used by `GRPOModule`, by
  `ControlNetLatentFlowModule`, and by `RewardModule`.
- **`DevicePolicy`** (`src/manifold/training/device_policy.py`, ADR-0035) — the
  per-rank CUDA device decision. Replaces both the duplicated pre-PG
  `set_device` twin that lived in the GRPO / reward training `main()`s and the
  `manifold.data.latent_pipeline.resolve_warm_device` free function used by the
  post-PG VAE warm. Three methods: `pin` (one `set_device` side effect pre-PG),
  `device` (side-effect-free read), `warm_device` (post-PG per-rank CUDA
  resolution, byte-identical to the former free function).

The two are paired concerns (one decides what is frozen, the other decides where
the GPU lands) and share an ADR-0035 test pattern: source-level guards via
`inspect.getsource` keep the retired free function / twin / mixin bypass from
sneaking back in.

### Who owns which device decision

Device ownership is deliberately split across several owners, because the calls
happen on opposite sides of the process-group boundary:

| Decision | Owner | When |
|---|---|---|
| Where a *frozen arm* (in-`Trainer`) lives | Lightning — the arm is a registered `nn.Module`, so the automatic `.to(device)` places it per rank | inside `Trainer.fit` |
| Where a *pre-`Trainer` model* is staged (policy rollout, ControlNet base) | `DevicePolicy.pin()` — the launcher's `LOCAL_RANK` snapshot | pre-PG, in the shell's `main()` |
| Where the *VAE cache-warm* runs | `DevicePolicy.warm_device(fallback)` | post-PG, inside `DataModule.setup()` |
| Where the *VAE* lives during a validation decode | the module's own device, via `VaeStage` (`src/manifold/metrics/vae_stage.py`) | post-PG, inside the callback |

The last row is the boundary to keep in mind: `DevicePolicy` owns **only** the
per-rank launcher decision. The paired-fidelity monitor
(`src/manifold/metrics/paired_callback.py`) does **not** go through
`DevicePolicy` — it stages the VAE through `VaeStage` onto the module's own
device (`next(self._module.unet.parameters()).device`), because by then the
module has already been placed by Lightning and the VAE is simply following it
onto the same GPU for the decode, then restored to CPU.

## FrozenArmMixin: register + dual-exclude

### Why this exists (ADR-0031 A1)

The previous scheme kept a frozen base UNet *unregistered* on the host via
`object.__setattr__`, hiding it from the module tree entirely. That worked for
the off-optimizer and off-checkpoint invariants but broke Lightning's automatic
`.to(device)`, forcing a manual staging step in `on_fit_start` and leaving
Mode-1 (supervised) and Mode-2 (GRPO ControlNet) forks out of sync. ADR-0031
chose the opposite direction: **register** the frozen arm as a normal
`nn.Module` submodule so Lightning owns its device placement, and **dual-exclude**
it from the optimizer and the checkpoint through targeted overrides.

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
applied — adding a frozen arm cannot forget the freeze. The arm-name set lives
on the mixin as a class-level `frozenset` default, so a host with no frozen
arms degrades to a no-op rather than `AttributeError`.

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
  carries frozen-arm weights (state_dict strips them); the arms are rebuilt
  fresh each launch (the reward from its own `.ckpt`, the reference policy via
  `deepcopy`, the ControlNet-policy base from the native export). Stray frozen
  keys carried by a stale pre-refactor checkpoint are stripped from the
  incoming dict before the load. The frozen arms being absent from the
  checkpoint is the one tolerated mismatch. A missing or unexpected
  **trainable** key still raises with the standard `RuntimeError("missing")` /
  `RuntimeError("unexpected")` (no blanket `strict=False`, which would resume on
  stale or random weights).
- **`train(mode=True)`** — re-applies `eval()` on every registered frozen arm
  after `super().train(mode)`. Registration makes Lightning's recursive
  `train(True)` flip the arms to training mode; the override re-freezes them so
  BatchNorm running stats cannot drift during rollout / reward evaluation.

The mixin **does not own optimizer arm-selection**: "which arm is optimized"
stays in each host's `configure_optimizers` (GRPO → unet/controlnet, reward →
`reward_model`, ControlNet → `controlnet`). The mixin guarantees only the
`requires_grad=False` layer; the host's optimizer simply never selects the
frozen arms. `configure_optimizers` excludes frozen arms by construction (see
the host tests: `tests/test_grpo.py`, `tests/test_controlnet_module_training.py`,
`tests/test_reward.py`).

### Which host freezes which arm

| Host | Frozen arms (`_frozen_arm_names`) | Optimized arm |
|---|---|---|
| `GRPOModule` (`src/manifold/modules/grpo.py`) | `reward_model` (always), `unet` (the ControlNet-policy base — a derived invariant of `controlnet is not None`), `reference_unet` + `reference_controlnet` (the KL anchor, when a reference policy is passed) | the UNet (UNet policy) or the ControlNet (ControlNet policy) |
| `RewardModule` (`src/manifold/modules/reward.py`) | `denoiser` (the frozen JiT x0-denoiser held for the online rollout) | `reward_model` (the discriminator) |
| `ControlNetLatentFlowModule` (`src/manifold/modules/controlnet_latent_flow.py`) | `unet` (the frozen base) | `controlnet` |

In the UNet-policy path the GRPO `unet` is the *trainable* policy and is
deliberately never registered as a frozen arm; the `None` reference arms stay
plain attributes, so they contribute no keys to `state_dict()` and no params to
`parameters()`.

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

The host-level proof that "the base is registered but dual-excluded" lives at
`tests/test_controlnet_module_training.py::test_base_is_registered_but_dual_excluded`
(ADR-0031 A1 acceptance): the base is present in `parameters()` (so Lightning
owns device placement) but `requires_grad=False`, absent from the optimizer
groups, and absent from `state_dict()`. The `ControlNetLatentFlowModule` host
test exercises the same contract on the production `spt.Module` MRO.

| Host-level behavior | Test |
|---|---|
| GRPO frozen arms registered but dual-excluded (reward + reference; the UNet policy stays trainable) | `tests/test_grpo.py::test_frozen_arms_registered_but_dual_excluded` |
| GRPO `state_dict()` strips in place on the shared destination / through a DDP-style `module.` prefix / under a nested `grpo.` prefix | `tests/test_grpo.py::test_state_dict_filters_destination_in_place`, `test_state_dict_strips_frozen_arms_through_wrapper_prefix`, `test_state_dict_strips_frozen_arms_when_wrapped` |
| GRPO strict load with the frozen-arm allow-list | `tests/test_grpo.py::test_load_state_dict_strict_except_frozen_allowlist` |
| GRPO `train()` keeps the frozen reward in eval | `tests/test_grpo.py::test_train_keeps_frozen_arms_in_eval` |
| ControlNet base registered but dual-excluded; optimizer wires the ControlNet only | `tests/test_controlnet_module_training.py::test_base_is_registered_but_dual_excluded`, `test_configure_optimizers_is_adam_over_controlnet_only` |
| Backward reaches the ControlNet, not the frozen base | `tests/test_controlnet_module_training.py::test_backward_routes_grad_to_controlnet_not_base` |
| Reward module holds the frozen denoiser registered + dual-excluded | `tests/test_reward.py::test_module_backward_updates_discriminator_only` |
| Lightning's automatic `.to(device)` places the registered frozen denoiser (the manual `on_fit_start` move is gone) | `tests/test_reward.py::test_run_reward_training_writes_ckpt_and_logs_metrics` |

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
de-duplication. (ADR-0031 §A2 had *specified* a centralized policy but never
landed one; ADR-0035 creates the object and deletes the free function.)

### Construction

```python
policy = DevicePolicy()  # side-effect free: snapshots LOCAL_RANK (missing -> 0)
```

`__init__` only reads `os.environ.get("LOCAL_RANK")` and stores the
result. It touches neither CUDA nor the process group. The snapshot is taken
**once at construction** — post-PG callers do **not** re-resolve via
`dist.get_rank()`. A `torchrun` launch always sets `LOCAL_RANK`; a single-process
launch does not, and the default 0 is correct in that case.

### `pin` — the one-time pre-PG side effect

```python
device = policy.pin()
```

This is the call the training shells that stage a model **before** Lightning
initializes the process group make (`grpo_cli`, `reward_cli`, `controlnet_cli`):

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
    Constructed --> WarmResolved: warm_device(fallback)
    Pinned --> ReadOnly: device()
    ReadOnly --> ReadOnly: device()
    Pinned --> WarmResolved: warm_device(fallback)
    ReadOnly --> WarmResolved: warm_device(fallback)
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
    state WarmResolved {
        [*] --> CheckFallback
        CheckFallback --> PGLiveCuda: cuda fallback and PG initialized
        CheckFallback --> KeepFallback: any other case
        PGLiveCuda --> ReturnCudaRank: return cuda local rank
        KeepFallback --> ReturnFallback: return fallback unchanged
    }
```
*Figure: `DevicePolicy` state transitions — `pin()` performs the one-time pre-PG `set_device` (with the twin's out-of-range guard), `device()` is the side-effect-free read, and `warm_device()` resolves the post-PG VAE-warm device from the fallback plus the PG state.*

### `device` — side-effect-free read

```python
device = policy.device()
```

Returns the same `cuda:{local_rank}` (or `cpu`) without calling `set_device`.
Use this for read-only / debug paths where the caller does not want the side
effect (most code does want `pin`).

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

### Where the three methods are called

| Shell | Call | Notes |
|---|---|---|
| `src/manifold/training/grpo_cli.py::main` | `device = DevicePolicy().pin()` | real path only; the `data_provider` CPU smoke stays `torch.device("cpu")` |
| `src/manifold/training/reward_cli.py::main` | `device = DevicePolicy().pin()` | same CPU-smoke seam |
| `src/manifold/training/controlnet_cli.py::main` | `device = DevicePolicy().pin()` | same CPU-smoke seam; replaces the bare `cuda:0` resolution |
| `src/manifold/training/cli.py::_warm_data` (JiT) | `warm_device = DevicePolicy().warm_device(device)` | the warm runs inside `DataModule.setup()` (post-PG) |
| `src/manifold/training/controlnet_cli.py::_real_inputs` | `warm_device = DevicePolicy().warm_device(device)` | mirrors the JiT warm |

The JiT shell keeps its launch-time `device` as the bare
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`; that is safe
because it stages **nothing** on the GPU pre-PG — the VAE is built on CPU
(`build_encode_pipeline(cfg, device=torch.device("cpu"))`) and the warm
re-stages it inside `warm_fn` on the per-rank device. The controlnet CLI has
no such latitude: it stages the frozen base UNet and the fresh ControlNet
before `trainer.fit`, which is exactly the PR #156 `cuda:0` pile-up.

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

The **source-level guards** keep the prior call patterns from sneaking back in.
They live in two files:

| Guard | File |
|---|---|
| `test_grpo_cli_main_pins_via_device_policy_not_inline_set_device` | `tests/test_device_policy.py` |
| `test_reward_cli_main_pins_via_device_policy_not_inline_set_device` | `tests/test_device_policy.py` |
| `test_controlnet_cli_main_pins_via_device_policy_not_bare_cuda` | `tests/test_device_policy.py` |
| `test_p1_warm_fn_uses_local_rank_device_not_launch_device` | `tests/test_ddp_warm.py` |
| `test_p1_controlnet_warm_uses_device_policy_not_resolve_warm_device` | `tests/test_ddp_warm.py` |

The two `test_p1_*` guards are the *warm* counterparts of the `pin` guards: they
assert that the JiT `_warm_data` and the controlnet `_real_inputs` closures
build the VAE on CPU pre-PG, rebuild the encode_fn on the per-rank device, and
no longer mention `resolve_warm_device`.

The end-to-end per-rank proof is a 2-rank DDP smoke:
`tests/test_controlnet_cli.py::test_main_real_path_pins_each_rank_to_local_gpu`
drives `controlnet_cli.main` on the real branch under spawn with `_real_inputs`
stubbed, and asserts the two ranks receive **different** devices (`cuda:0` vs
`cuda:1`) — the PR #156 regression lock (a bare `cuda:0` resolution would
collapse both ranks onto GPU 0).

## Retired patterns — never reintroduce

| Retired pattern | Status | Guard |
|---|---|---|
| `manifold.data.latent_pipeline.resolve_warm_device` free function | deleted (ADR-0035 close-out); no runtime call site remains — only docstring provenance notes | `tests/test_ddp_warm.py::test_p1_*` |
| The pre-PG inline `set_device` twin in `grpo_cli.main` / `reward_cli.main` | deleted; both route through `DevicePolicy().pin()` | `tests/test_device_policy.py::test_*_cli_main_pins_via_device_policy_not_inline_set_device` |
| The bare `torch.device("cuda" if torch.cuda.is_available() else "cpu")` in `controlnet_cli.main` | deleted on the real path; the CPU smoke keeps `torch.device("cpu")` | `tests/test_device_policy.py::test_controlnet_cli_main_pins_via_device_policy_not_bare_cuda` |
| The `object.__setattr__` registration bypass for frozen arms | retired by ADR-0031 A1; arms are normal registered submodules | `tests/test_frozen_arm_mixin.py`, `tests/test_grpo.py::test_frozen_arms_registered_but_dual_excluded` |
| A hand-rolled per-host `state_dict` / `load_state_dict` override | the mixin is the single owner of the dual-exclude contract | `tests/test_frozen_arm_mixin.py` |

## Source map

- Mixin implementation: `src/manifold/modules/frozen_arm.py`
- Mixin barrel export: `src/manifold/modules/__init__.py` (re-exports
  `FrozenArmMixin`)
- Hosts that consume the mixin: `src/manifold/modules/grpo.py`,
  `src/manifold/modules/controlnet_latent_flow.py`, `src/manifold/modules/reward.py`
- Device policy implementation: `src/manifold/training/device_policy.py`
  (imported by the shells as `from .device_policy import DevicePolicy`; it is not
  re-exported by the `manifold.training` barrel)
- CLI shells that call `pin()`: `src/manifold/training/grpo_cli.py`,
  `src/manifold/training/reward_cli.py`, `src/manifold/training/controlnet_cli.py`
- VAE-warm closures that call `warm_device()`:
  `src/manifold/training/cli.py::_warm_data`,
  `src/manifold/training/controlnet_cli.py::_real_inputs`
- VAE staging for validation decodes (a different owner):
  `src/manifold/metrics/vae_stage.py`, `src/manifold/metrics/paired_callback.py`

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

- `pytest tests/test_ddp_warm.py tests/test_training_cli.py tests/test_controlnet_cli.py -q`

The `inspect.getsource`-style guards in `tests/test_device_policy.py` and
`tests/test_ddp_warm.py` are the load-bearing acceptance: any regression that
re-introduces the inline `set_device` twin, the bare `torch.device("cuda")` in
the controlnet CLI, or the `resolve_warm_device` free function will fail those
tests loudly.
