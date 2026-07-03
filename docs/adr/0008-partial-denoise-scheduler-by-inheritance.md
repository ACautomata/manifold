# A second scheduler for partial per-sample denoise — by inheritance, never a forked transport

The reward model (GRPO) builds preference pairs by noising a clean latent to a
**per-sample** flow-time `t_start` and denoising back to clean. The JiT scheduler
plus its shared rollout `sample_latent_flow` cannot express this: they start at
pure noise (`set_timesteps = linspace(0,1,n+1)`) on one batch-wide grid, and
`euler_step` / `heun_correct` take scalar `float(t)`. We add a
`PartialFlowMatchHeunScheduler(FlowMatchHeunDiscreteScheduler)` that **inherits**
`add_noise` (the transport — already `(B,)`-capable) and the Heun step math
verbatim, and adds only `set_timesteps_partial(t_start: (B,) Tensor, num_steps)`
producing a per-sample `(B, n+1)` grid. The parent's `euler_step` /
`heun_correct` are generalized **backward-compatibly** to accept `(B,)`
`t`/`t_next` (scalar is the degenerate special case; JiT callers are unchanged).
The rollout loop itself lives in `modules/partial_denoise.py`, mirroring
`sample_latent_flow` — ADR-0005 keeps rollouts module-owned and scheduler-delegated.

**Hard constraint:** the new scheduler *delegates* transport and Heun, it never
*reimplements* them. The UNet was trained on JiT's transport
`z = t·x + (1−t)·e`, so a forked transport would feed it out-of-distribution
noise and the denoised pairs — hence the reward signal — would be meaningless;
forking either piece also violates the ADR-0001/0002 single-source-of-truth.

## Why

- **Single source of truth.** Transport and Heun math stay in one place
  (ADR-0001/0002); only the grid (per-sample, partial range `t_start → 1`) is new.
- **Isolation over in-place generalization.** Subclassing leaves the JiT
  scheduler's callers (the train `Module` and the inference `Pipeline`) byte-untouched,
  so the parity ADR-0005 migrated and verified cannot regress — the reward feature's
  blast radius stays inside the reward feature.
- **OOD safety is structural.** Delegating `add_noise` guarantees the pairs' noised
  starts match the UNet's training distribution without a test to enforce it.

## Consequences

- A second scheduler class exists and is reward-specific; a future reader seeing two
  `FlowMatchHeun*Scheduler` classes should land here.
- `FlowMatchHeunDiscreteScheduler.euler_step` / `heun_correct` widen to
  tensor-or-float `t`/`t_next` (backward compatible; the scalar path is the special case).
- A new `modules/partial_denoise.py` rollout primitive is the reward analogue of
  `sample_latent_flow`, calling the new scheduler's grid + the (shared) Heun steps.
