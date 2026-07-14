# Paired reward generator — load via a paired native export bridge, bake the raw arm

The frozen paired-JiT generator (whose src→tgt rollout produces the reward's fake
losers) is loaded from a **paired native export** via a new
`load_frozen_paired_generator` — a sibling of `load_frozen_denoiser`
(`reward_pairs.py:90`) that calls `PairedLatentFlowPipeline.from_pretrained`. The
export bakes the **raw optimizer weights** — the arm paired's checkpoint selection
monitors as `val/psnr` — matching the JiT reward, which also bakes raw (aligned with
`val/fid`).

## Why

- **The paired native export does not yet exist.** `manifold-train-paired` writes only
  Lightning `.ckpt`; `export_to_native` builds a `LatentFlowPipeline`. A paired export
  bridge (parameterize the pipeline class, or a sibling `export_paired_checkpoint`) is
  needed — and it is **not throwaway**: paired *inference* reaches the Pipeline via
  native export too (ADR-0006). The raw-bake machinery (`_bake_backbone`, the
  `_STATE_PREFIX` constant) is
  MAISI-backbone-keyed → reusable verbatim (the paired UNet wraps the same backbone,
  only `in_channels = 2·C_latent`).
- **`from_pretrained` gives all three requirements in one load.** The reward needs the
  frozen UNet + scheduler + `vae.scaling_factor` (to scale src latents into the UNet's
  training space). The native dir carries all three; a direct-`.ckpt` load would need a
  separate VAE for the scale.
- **Raw arm = "the paired model."** Paired's best checkpoint is selected on `val/psnr`
  measured on the raw optimizer arm (`paired_cli.py:185`), so the published generator
  *is* the raw arm. The reward's fakes must come from the same weights whose quality
  `val/psnr` reflects, and from what paired-GRPO will start from. This matches the JiT
  reward (raw, per `val/fid`) — each bakes the arm its own checkpoint selection monitors
  (EMA training was removed; ADR-0006).
- **Rejected — direct `.ckpt` load:** ad-hoc, no reusable native artifact, scale-factor
  plumbed separately.

## Consequences

- New `load_frozen_paired_generator(native_dir)` returns `(unet, scheduler,
  scaling_factor)`; the scheduler is the **base** `FlowMatchHeunDiscreteScheduler` (the
  loser is a full 0→1 rollout) — NOT re-instantiated as Partial (unlike
  `load_frozen_denoiser`); only the probe path constructs the Partial subclass.
- A paired export bridge is a new one-shot artifact producer (sibling of
  `export_to_native`, or `scripts/export_paired_checkpoint.py`); the bake is always
  raw (no `prefer_ema` knob — EMA training was removed, ADR-0006).
- The export must run once to produce the native dir before reward training; resume
  re-reads it (the reward checkpoint holds no generator — ADR-0020).
