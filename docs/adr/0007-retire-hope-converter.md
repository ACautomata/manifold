# Retire the hope→native converter; migration complete

The one-shot hope→native converter (ADR-0003) — `convert_hope_checkpoint` and its
helpers — is removed. The migration it existed for is finished: no hope-trained
flat checkpoints remain to ingest, so `export_to_native` (ADR-0006) becomes the
sole checkpoint → inference path and the inference Pipeline's interface narrows to
native-only.

## Why now

- **No hope flat checkpoints left to consume.** JiT training (#29–#31) writes a
  stock Lightning `.ckpt`, exported to the native per-component dir via
  `export_to_native`. The frozen VAE `autoencoder_v1.pt` is loaded by the data
  stack (`latent_pipeline.build_vae`) — a separate path, untouched by this retire.
- **The converter was the largest hope artifact left in `src/`.** It sat *inside*
  the inference Pipeline Implementation and was exported from the Pipeline
  package's interface, smearing a foreign-format conversion concern across the
  inference module's name.
- **Its EMA-selection logic was duplicated.** `convert_hope_checkpoint` and
  `export_to_native` each hand-rolled the "slowest-decay EMA shadow = the
  inference UNet" argmax, with diverging fallbacks (`len−1` vs `None`). Removing
  the converter collapses three sites to two; EMA has since been removed entirely
  (2026-07-14), so both the converter and the EMA-selection logic are gone, and
  `export_to_native` now always bakes the raw optimizer weights.

## Considered options

- **Keep the converter, relocate it to `checkpoints/hope.py`.** Rejected: there is
  nothing left to ingest. Keeping it perpetuates a duplicated EMA policy and
  leaves migration glue in the package's interface for a format that no longer
  reaches the load path.
- **Retire the converter but keep `_reject_if_hope_flat` as a hope-agnostic
  peek.** Rejected: a `.pt` passed to `from_pretrained` already hits a generic
  `FileNotFoundError(... missing model_index.json)`. A targeted peek is speculative
  UX that fails the deletion test, and its old message pointed at the now-deleted
  converter.
- **Retire only the public API, keep the helpers private.** Rejected: the helpers
  (`_select_inference_weights`, `_load_vae_weights`, `_reject_if_hope_flat`) had no
  caller but the converter; partial retirement leaves orphans.

## Consequences

- `LatentFlowPipeline.from_pretrained` loads only native per-component dirs; the
  `convert_hope_checkpoint` export, the `scripts/convert_hope_checkpoint.py` CLI,
  the `--warm-start` hope-flat branch, and the converter test block are gone.
- Warm-start (`_load_warm_start`) accepts a Lightning `.ckpt` or a bare state dict.
- EMA has been removed entirely (2026-07-14); `training/ema.py` and
  `slowest_shadow_index` are deleted, and `export_to_native` always bakes the raw
  optimizer weights (no slowest-shadow argmax).
- A future hope-format `.pt` would require a fresh converter — acceptable, since
  none exist and new runs emit Lightning `.ckpt`.
- This **supersedes the converter portion of ADR-0003**. ADR-0003's other
  decisions — `scale_factor` owned by the VAE, the native per-component format —
  still stand.
