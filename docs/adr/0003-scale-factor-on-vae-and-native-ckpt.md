# scale_factor owned by the VAE; native per-component checkpoints with a hope converter

`scaling_factor` is a buffer+config on the VAE wrapper: `encode` returns **scaled**
latents and `decode` undoes the scaling internally, so the Module and Pipeline never
reference it. This absorbs hope's scattered `latent * scale_factor` (in the module)
and `ReconModel`'s `z / scale_factor` (in decode) into one place, matching
diffusers' `AutoencoderKL.scaling_factor` placement.

Checkpoints are a native manifold **per-component** format, read/written by
`Pipeline.from_pretrained` / `save_pretrained`. Old hope flat checkpoints
`{unet_state_dict, scale_factor, ema}` are **not** read by `from_pretrained`; they
come over through a one-shot converter script
(`scripts/convert_hope_checkpoint.py`) that maps `unet_state_dict → unet` and
`scale_factor → vae.scaling_factor`.

## Considered options

- **scale_factor on the Module** (hope's training-side placement, with a second copy
  on the decode path). Rejected: scatters one number across train + decode, and the
  VAE — the thing that produced the latents — wouldn't know its own scale.
- **Keep hope's flat format on save** (`{unet_state_dict, scale_factor, ...}`).
  Rejected: fights the VAE-owns-scale ownership (scale sits at the ckpt top level
  but belongs to the VAE) and flattens a now-component-structured model back into a
  bag of tensors.
- **`from_pretrained` reads hope ckpts directly** (back-compat). Rejected in favour
  of a clean native format + explicit converter: keeps the load path honest about
  what manifold's component structure is, and isolates hope-specific shape quirks in
  one script instead of inside the Pipeline.

## Data-stack estimation (addendum)

The data stack derives `scaling_factor` for from-scratch training: it warms a cache
of **unscaled** latents via `AutoencoderKL.encode_raw` (the unscaled affordance),
estimates `1/std(z)` over the full cache, sets `vae.scaling_factor`, and multiplies
by it at `LatentDataset.__getitem__` so the Module receives scaled latents. The
Module and Pipeline still never reference `scaling_factor`; only the data stack
reads the VAE's own property. The public `encode` contract (returns scaled latents)
is unchanged — `encode_raw` is an internal affordance for estimation. At inference,
`scaling_factor` comes from the converted checkpoint, not re-estimated.

Considered: estimate-then-encode-scaled (estimate from a subset, then re-encode the
full cache scaled — fully clean but two encode passes, deviation from hope's
over-full-cache estimate) and config-supplied (no estimation — rejected: abandons
the latent-prep flow and gives no derivation for a new dataset).
