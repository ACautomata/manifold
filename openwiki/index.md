---
okf_version: "0.2"
---

# Files

- [Architecture and Source Map](architecture.md) - Component boundaries, runtime flows, configuration and persistence contracts, and where to look in source — including the shipped observe-only in-training paired-fidelity monitor for the supervised ControlNet stage.
- [Callback registry and training spine](callback-registry.md) - CallbackRegistry two-phase resolve/build, the spec contract, and TrainingSpine as the single caller that composes the training CLIs (ADR-0029 + ADR-0032) — including the shipped observe-only paired-fidelity spec registered by the supervised ControlNet shell (ADR-0037).
- [Domain vocabulary and decision records](domain-vocabulary-and-decisions.md) - Maps the repository's canonical glossary (CONTEXT.md) onto the source symbols that implement each term, indexes the ADR decision trail by subsystem, and records how work is proposed, triaged, and reconciled against existing decisions.
- [Before/after GRPO evaluation](evaluation.md) - Shipped workflow and source map for manifold-eval, the same-noise before/after GRPO comparison, 3D paired-fidelity scoring, slice-grid output, and the self-contained comparison page builder — plus the shipped observe-only in-training paired-fidelity monitor (ADR-0037) that logs val/psnr and val/ssim on a fixed paired subset.
- [Frozen arms and per-rank device policy](frozen-arm-and-device-policy.md) - FrozenArmMixin (register + dual-exclude off the optimizer / checkpoint) and DevicePolicy (the per-rank CUDA device decision that replaced resolve_warm_device and the pre-PG set_device twin).
- [Operations and testing](operations-and-testing.md) - Standard checks, the focused test matrix, the distributed-validation contract, DDP deadlock diagnostics, the before/after eval runbook, and validation/checkpoint cautions for Manifold — including the shipped observe-only in-training paired-fidelity monitor (val/psnr / val/ssim) and its DDP and cost properties.
- [Quickstart](quickstart.md) - Routing entry for the Manifold wiki; what the wiki covers, how it is organized, and where to go next for each change area.
- [Key Workflows](workflows.md) - JiT, supervised ControlNet translator, reward/GRPO training stages, before/after evaluation, inference, checkpoints, and export.
