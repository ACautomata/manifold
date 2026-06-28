# Diffusers-style layout with own base classes; x0-only JiT; orthodox scheduler split

Manifold mimics the diffusers four-part layout (models / schedulers / training
module / pipeline) and naming, but with manifold-defined lightweight base classes
— it does **not** subclass `diffusers.{ModelMixin,SchedulerMixin,DiffusionPipeline}`.
`diffusers` stays a utility-only dependency (LR schedulers, training helpers), as
it already is in `hope`. Backbones stay MONAI MAISI (wrapped, not reimplemented).
Only the JiT (x0-denoiser) formulation is migrated; the legacy velocity RFlow,
RM/GRPO/MeanFlow, LoRA, and FID are out of scope. The Scheduler owns transport +
reverse-step only; the `spt.Module` owns logit-normal `t`-sampling, the
`(1−t)⁻²` loss weight, and the MSE.

## Considered options

- **Subclass the `diffusers` library** (free `from_pretrained`/`save_pretrained`
  + Hub integration). Rejected: MONAI MAISI 3D medical models fight diffusers'
  2D/text-conditioned `ModelMixin` assumptions, and it would structurally entangle
  three frameworks (monai + spt + diffusers). Mimicking the layout gets the
  mental model without the coupling.
- **Keep both velocity + x0 formulations** (hope's switchable Formulation).
  Rejected: "JiT" strictly means x0; x0-only removes hope's "Formulation must
  switch as one unit or generation silently corrupts" invariant, which is what
  permits the clean orthogonal split below.
- **Cohesive Formulation scheduler** (scheduler also owns `t`-distribution +
  loss weight; module becomes a thin shell). Rejected: that pushes training
  concerns into the diffusers-scheduler role, deviating from "diffusers-style."
  With only one formulation, the cohesion invariant is moot, so each concern
  lands in its idiomatic home: module = training, scheduler = transport/sampling.

## Consequences

- No free HF-Hub `from_pretrained`/config persistence — checkpoint and config
  plumbing is manifold's own (see ADR-0003 for `scale_factor` + checkpoint shape).
- Scheduler ↔ Module share the transport by construction (same object owns
  `add_noise` and `step`); the module must call `scheduler.add_noise` rather than
  re-deriving the noising, or the two can drift.
