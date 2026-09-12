---
okf_version: "0.2"
---

# Files

- [Architecture and Source Map](architecture.md) - Component boundaries, data/config layers, evaluation/reporting boundaries, domain vocabulary, and where to look in source.
- [Callback registry and training spine](callback-registry.md) - CallbackRegistry two-phase resolve/build, the spec contract, monitor validation, and TrainingSpine as the single caller that composed the five training CLIs (ADR-0029 + ADR-0032).
- [Before/after GRPO evaluation](evaluation.md) - Shipped workflow and source map for manifold-eval, the same-noise before/after GRPO comparison, 3D paired-fidelity scoring, slice-grid output, and the self-contained comparison page builder. Also records the in-training paired-fidelity monitor (supervised ControlNet stage shipped; ControlNet-GRPO stage is the deliberate follow-up).
- [Frozen arms and per-rank device policy](frozen-arm-and-device-policy.md) - FrozenArmMixin (register + dual-exclude off the optimizer / checkpoint) and DevicePolicy (the per-rank CUDA device decision that replaced resolve_warm_device and the pre-PG set_device twin).
- [Operations and Testing](operations-and-testing.md) - Setup, validation behavior, distributed metrics, runbook cautions, deadlock vs slow-validation diagnostic, and focused test commands for Manifold.
- [Quickstart](quickstart.md) - Routing entry for the Manifold wiki; what the wiki covers, how it is organized, and where to go next for each change area.
- [Key Workflows](workflows.md) - JiT, supervised ControlNet translator, reward/GRPO training stages, before/after evaluation, inference, checkpoints, and export.
