# Training checkpoints are Lightning `.ckpt`; the native dir comes from a separate export

manifold's persistence contract is the native per-component directory
(`Pipeline.from_pretrained` / `save_pretrained`, ADR-0003). hope trained with a
bespoke `LegacyCheckpoint` callback that wrote its own flat format every epoch.
We do neither during training: training writes a stock Lightning
`ModelCheckpoint` `.ckpt`, and the native per-component inference dir is produced
by a **separate export** step.

The `ModelCheckpoint` monitors `val/fid_raw` (`mode='min'`, `save_top_k`, `save_last`,
`save_on_train_epoch_end`), writes full training state (UNet + optimizer +
LR-schedule + epoch + EMA callback state) so `trainer.fit(ckpt_path=…)` resumes
cleanly, and the `EMACallback` shadows are captured via its callback `state_dict`.
The export loads a `.ckpt` and bakes the **raw UNet weights** as the inference
UNet by default, and writes the native dir so `Pipeline.from_pretrained` can load
it. The monitor (`val/fid_raw`, the raw-optimizer arm) and the export default
(raw weights) are deliberately aligned: the exported "best" checkpoint is best
for the weights that are actually published. `prefer_ema=True` bakes the slowest
EMA shadow instead (EMA-selection lives in `slowest_shadow_index`; the hope→native
converter that previously shared it is retired — ADR-0007) for runs where the
0.9999 EMA has converged (warm-start / long horizon, as hope trains).

## Note on the monitor/export policy (2026-07)

`val/fid_avg` (slow-EMA, the published model under hope's policy) is still logged
alongside `val/fid_raw`, but is no longer what selects or weights the "best"
checkpoint. On short from-scratch runs the 0.9999 EMA lags the raw model (at ep7
of the GLI BraTS run: raw FID ≈ 5.8 vs slow-EMA ≈ 21.5 — the shadow is still
~61% init at `0.9999^5008`), so monitoring/publishing the EMA undersells the
model and its per-plane quality oscillates epoch-to-epoch. hope avoids this by
warm-starting + training 200–1000 epochs; manifold's default monitor/export now
tracks the raw model until manifold matches that regime, at which point
`prefer_ema=True` (and monitoring `val/fid_avg`) restores the EMA-publish policy.

## Why

- **Framework-standard checkpointing.** `ModelCheckpoint` gives metric monitoring,
  top-k best-by-FID selection, last-ckpt, and resume for free — no bespoke
  callback to maintain (hope's `LegacyCheckpoint` is ~500 lines including a
  Pareto frontier we don't need).
- **Respects the train/infer boundary (ADR-0005).** Training never instantiates
  the `LatentFlowPipeline`; the native format is the inference side's concern,
  reached only through the export bridge.
- **Resume + EMA are automatic.** Lightning persists optimizer/scheduler/EMA
  state in the `.ckpt`; the bespoke flat format had to hand-wire the `ema` key.

## Consequences

- A trained checkpoint is **not** immediately Pipeline-loadable — run the export
  first. This mirrors the shape of the now-retired hope→native converter
  (ADR-0007); export is now the sole checkpoint → inference path.
- Best-by-FID monitoring is meaningful only when FID actually logs each epoch,
  i.e. **single-GPU** (FID is rank-0-only, the hope invariant); under DDP the
  checkpoint falls back to `save_last` + `every_n_epochs` without a monitor.
