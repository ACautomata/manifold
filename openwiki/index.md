---
okf_version: "0.2"
---

# Files

- [Architecture and Source Map](architecture.md) - Component boundaries, the training-orchestration and metrics layers, the Experiment-config vs Component-config split, the native-artifact persistence boundary, and where to look in source for each concern.
- [Callback registry and training spine](callback-registry.md) - CallbackRegistry's two-phase resolve/build, the spec contract, the post-resolve monitor validation, the built-in spec table (including the observe-only paired-fidelity spec), and TrainingSpine.run as the single caller that assembles every training CLI's callbacks (ADR-0029 + ADR-0032 + ADR-0037).
- [Domain concepts and decision index](concepts.md) - The repository's domain vocabulary mapped to its owning component and source symbol, the anti-terms CONTEXT.md explicitly rejects, the superseded concepts kept only as searchable history, and a one-line index of ADR-0001 through ADR-0037 so a change can find its governing decision.
- [Before/after GRPO evaluation](evaluation.md) - Shipped workflow and source map for manifold-eval, the same-noise before/after GRPO comparison, 3D paired-fidelity scoring, slice-grid output, and the self-contained comparison page builder, plus the shipped observe-only in-training paired-fidelity monitor that shares the same metric and normalization.
- [Frozen arms and per-rank device policy](frozen-arm-and-device-policy.md) - FrozenArmMixin (register + dual-exclude off the optimizer / checkpoint, Lightning owns placement) and DevicePolicy (the per-rank CUDA device decision that replaced resolve_warm_device and the pre-PG set_device twin).
- [Operations and Testing](operations-and-testing.md) - Setup, validation behavior, distributed metrics, runbook cautions, and focused test commands for Manifold.
- [Quickstart](quickstart.md) - Routing entry for the Manifold wiki; what the wiki covers, how it is organized, and where to go next for each change area.
- [Key Workflows](workflows.md) - JiT, supervised ControlNet translator, reward/GRPO training stages, before/after evaluation, inference, checkpoints, and export.
