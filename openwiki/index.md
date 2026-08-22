---
okf_version: "0.1"
---

# Files

- [Architecture and Source Map](architecture.md) - Component boundaries, data/config layers, evaluation/reporting boundaries, domain vocabulary, and where to look in source.
- [Callback registry and training spine](callback-registry.md) - CallbackRegistry two-phase resolve/build, the active paired-fidelity spec, and TrainingSpine as the single training callback composition point (ADR-0029, ADR-0032, ADR-0037).
- [Before/after GRPO evaluation](evaluation.md) - Runtime and source map for the active in-training paired-fidelity monitor and the shipped manifold-eval same-noise before/after workflow, including 3D PSNR/SSIM, slice grids, and the self-contained comparison page builder.
- [Frozen arms and per-rank device policy](frozen-arm-and-device-policy.md) - FrozenArmMixin (register + dual-exclude off the optimizer / checkpoint) and DevicePolicy (the per-rank CUDA device decision that replaced resolve_warm_device and the pre-PG set_device twin).
- [Operations and Testing](operations-and-testing.md) - Setup, validation behavior, distributed metrics, runbook cautions, and focused test commands for Manifold.
- [Quickstart](quickstart.md) - Routing entry for the Manifold wiki; what the wiki covers, how it is organized, and where to go next for each change area.
- [Key Workflows](workflows.md) - JiT, supervised ControlNet translator, reward/GRPO training stages, before/after evaluation, inference, checkpoints, and export.
