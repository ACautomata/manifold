---
okf_version: "0.2"
---

# Files

- [Architecture and Source Map](architecture.md) - Component boundaries, data/config layers, evaluation/reporting boundaries, the runtime flows (JiT, supervised ControlNet, in-training paired-fidelity monitor, reward/GRPO, before/after eval), and where to look in source.
- [Callback registry and training spine](callback-registry.md) - CallbackRegistry's typed name-to-spec dispatch, the two-phase resolve/build split, the CallbackContext runtime bag, post-resolve monitor validation, the built-in spec catalog, and TrainingSpine.run's merge order with its post-merge forbidden-callback / forbidden-monitor guards (ADR-0029, ADR-0032, ADR-0037).
- [Before/after GRPO evaluation](evaluation.md) - Shipped workflow and source map for manifold-eval, the same-noise before/after GRPO comparison, 3D paired-fidelity scoring, slice-grid output, and the self-contained comparison page builder, plus the shared metric and normalization contract with the observe-only in-training paired-fidelity monitor.
- [Frozen arms and per-rank device policy](frozen-arm-and-device-policy.md) - How the repo decides what is frozen and which GPU a rank uses — FrozenArmMixin (register a frozen arm, then dual-exclude it from the optimizer and the checkpoint while Lightning owns its device placement) and DevicePolicy (the single per-rank CUDA device decision that replaced the duplicated set_device twins and the latent_pipeline resolve_warm_device free function).
- [Operations and Testing](operations-and-testing.md) - Standard checks, the focused-test matrix, the distributed-validation metric contract, the before/after and distributed-validation runbooks, deadlock diagnosis, DDP failure modes, and validation/checkpoint cautions for Manifold.
- [In-training Paired-Fidelity Monitor](paired-fidelity-monitor.md) - The shipped observe-only paired-fidelity monitor (ADR-0037) for the supervised ControlNet stage - PairedFidelityCallback and PairedFidelitySpec logging val/psnr and val/ssim from a fixed paired subset under fixed seeded noise, its knob contract, DDP redundant-evaluation design, and its comparability contract with the offline before/after eval.
- [Quickstart](quickstart.md) - Routing entry for the Manifold wiki; what the wiki covers, how it is organized, and where to go next for each change area.
- [Key Workflows](workflows.md) - JiT, supervised ControlNet translator, reward/GRPO training stages, before/after evaluation, inference, checkpoints, and export.
