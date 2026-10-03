---
okf_version: "0.2"
---

# Files

- [Architecture and Source Map](architecture.md) - Repository-wide component model (Model / Scheduler / Module / Pipeline + training orchestration, metrics, eval), runtime flows for the JiT generator / supervised ControlNet / reward+GRPO / before-after eval, configuration-vs-persistence split, and global change guidance.
- [Callback registry and training spine](callback-registry.md) - CallbackRegistry two-phase resolve/build, the spec contract, monitor validation, TrainingSpine.run merge order with forbidden_callbacks / forbidden_monitors guards, and the now-active PairedFidelitySpec (ADR-0037).
- [Before/after GRPO evaluation](evaluation.md) - Shipped manifold-eval workflow (export after → load both pipelines → same-noise generation → decode → paired PSNR/SSIM → 2.5D slice grids), public manifold.eval API, the library-only ComparisonPageBuilder, and the now-active observe-only PairedFidelityCallback / PairedFidelitySpec that mirrors the offline metric into a per-epoch supervised ControlNet monitor (ADR-0037).
- [Frozen arms and per-rank device policy](frozen-arm-and-device-policy.md) - FrozenArmMixin (register + dual-exclude off the optimizer / checkpoint) and DevicePolicy (the per-rank CUDA device decision that replaced resolve_warm_device and the pre-PG set_device twin).
- [Operations and Testing](operations-and-testing.md) - Setup, validation behavior, distributed metrics, runbook cautions, and focused test commands for Manifold.
- [Quickstart](quickstart.md) - Routing entry for the Manifold wiki; what the wiki covers, how it is organized, and where to go next for each change area.
- [Key Workflows](workflows.md) - JiT, supervised ControlNet translator, reward/GRPO training stages, before/after evaluation, inference, checkpoints, and export.
