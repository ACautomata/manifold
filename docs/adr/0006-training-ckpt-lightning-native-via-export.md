# Training checkpoints are Lightning `.ckpt`; the native dir comes from a separate export

manifold's persistence contract is the native per-component directory
(`Pipeline.from_pretrained` / `save_pretrained`, ADR-0003). hope trained with a
bespoke `LegacyCheckpoint` callback that wrote its own flat format every epoch.
We do neither during training: training writes a stock Lightning
`ModelCheckpoint` `.ckpt`, and the native per-component inference dir is produced
by a **separate export** step.

The `ModelCheckpoint` monitors `val/fid` (`mode='min'`, `save_top_k`, `save_last`,
`save_on_train_epoch_end`), writes full training state (UNet + optimizer +
LR-schedule + epoch) so `trainer.fit(ckpt_path=…)` resumes cleanly. The export
loads a `.ckpt` and bakes the **raw UNet weights** as the inference UNet, and
writes the native dir so `Pipeline.from_pretrained` can load it. The monitor
(`val/fid`, the raw-optimizer arm) and the export (raw weights) are deliberately
aligned: the exported "best" checkpoint is best for the weights that are actually
published.

## Note on EMA removal (2026-07-14)

EMA training was removed 2026-07-14, accepting the reversal of the paired-JiT
+9dB fast-EMA finding (#113).

## Why

- **Framework-standard checkpointing.** `ModelCheckpoint` gives metric monitoring,
  top-k best-by-FID selection, last-ckpt, and resume for free — no bespoke
  callback to maintain (hope's `LegacyCheckpoint` is ~500 lines including a
  Pareto frontier we don't need).
- **Respects the train/infer boundary (ADR-0005).** Training never instantiates
  the `LatentFlowPipeline`; the native format is the inference side's concern,
  reached only through the export bridge.
- **Resume is automatic.** Lightning persists optimizer/scheduler state in the
  `.ckpt`; the bespoke flat format had to hand-wire the `ema` key.

## Consequences

- A trained checkpoint is **not** immediately Pipeline-loadable — run the export
  first. This mirrors the shape of the now-retired hope→native converter
  (ADR-0007); export is now the sole checkpoint → inference path.
- Best-by-FID monitoring is meaningful only when FID actually logs each epoch,
  i.e. **single-GPU** (FID is rank-0-only, the hope invariant); under DDP the
  checkpoint falls back to `save_last` + `every_n_epochs` without a monitor.
