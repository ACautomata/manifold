# Paired JiT conditioning — concat the source latent at the input, sum the contrast labels in the embedding pathway; train from scratch

The Paired JiT UNet conditions on the source by **concatenation**: the input is
`[z_t, x_src_latent]` along channels, so `in_channels = 2·C_latent` (the MAISI
class is unchanged — only the config value). This puts `x_src` in view at *every*
`t`, letting the model disambiguate the mid-`t` mix (ADR-0013's consequence). The
translation direction is encoded by **summing** the contrast embeddings,
`embed(src_contrast) + embed(tgt_contrast)`, through the existing class-label
pathway — one model then learns every contrast direction (any-to-any pairing). The
UNet is **trained from scratch** (no warm-start from the noise→data JiT UNet).

## Why

- **Mid-`t` disambiguation.** At `t ≈ 0.5`, `z_t` is a 50/50 mix and
  `x_tgt = (z_t − (1−t)·x_src)/t` is unsolvable without `x_src` in view. A
  label-only model (conditioning through `z_t` + summed labels) collapses to the
  conditional mean — blurry outputs. This is structurally unlike noise→data JiT,
  where the `t = 0` half is a *known simple* distribution (Gaussian), so the model
  needs only the data prior. Concat is the standard paired-I2I flow-matching fix.
- **Summed labels reuse the embedding table and enable any-to-any.** The direction
  is a sum of two existing rows, not a fresh row per `(src, tgt)` pair, so the
  class-embedding table does not blow up combinatorially and a single model serves
  all contrast directions.
- **Train from scratch removed the only counter-argument.** Warm-starting from the
  noise→data JiT UNet was the sole reason to keep `in_channels` unchanged (which
  label-only conditioning and cross-attention preserve). With warm-start off the
  table, concat — the safe/standard choice — costs nothing.
- **Rejected — cross-attention via `context`:** keeps the backbone's `in_channels`
  identical (warm-start friendly) but needs a *new source-encoder* subnetwork to
  turn a dense 3D volume into a token sequence, and cross-attention on a dense
  volume is a less-proven conditioning for I2I.
- **Rejected — FiLM per-block:** backbone-reusable but injects modulation heads
  into every MAISI resblock — invasive wrapper surgery that violates ADR-0001's
  "wrap MONAI, never reimplement."
- **Rejected — label-only:** minimal and zero new parameters, but accepts the
  mid-`t` blur; rejected once warm-start (its only offsetting benefit) was off.

## Consequences

- A distinct UNet config (`in_channels = 2·C_latent`, `out_channels = C_latent`);
  the noise→data JiT UNet's input conv does **not** initialize it (random init,
  train from scratch). The embedding table and conv stack are fresh.
- The class-label pathway receives a summed embedding; `num_class_embeds` must
  cover the per-contrast label space (rows are shared across directions, not a
  Cartesian `|src|·|tgt|` table).
- Any-to-any pairing: per subject, the ordered contrast pairs are enumerated as
  dataset items; latents are cached per (subject, contrast) and **shared** across
  the pairs that reference them (no 12× duplication).
