# Paired JiT — the same transport as noise→data JiT, with a data latent at the t=0 endpoint

For paired src→tgt translation, Paired JiT reuses the existing
`FlowMatchHeunDiscreteScheduler` transport `z = t·x + (1−t)·e` **verbatim**, passing
the source latent as `e` — so `z_t = t·x_tgt + (1−t)·x_src` is a straight
interpolation between two *data* latents. The network predicts `x_tgt`
(`prediction_type = "sample"`, unchanged); the Heun rollout starts from
`z_0 = x_src_latent` instead of Gaussian noise. No new scheduler, no new integrator,
no new loss — the existing `(1−t)⁻²`-weighted x0-MSE is reused.

**Hard constraint:** the scheduler, the velocity derivation `v = (x0 − z)/(1−t)`,
the two-evaluation Heun (`euler_step` / `heun_correct`), and the `t_eps` endpoint
clamp are all reused *as-is*. The transport is endpoint-agnostic — the noise→data
JiT just happens to put Gaussian noise at `t = 0`; Paired JiT puts a data latent
there instead. Forking any of these would violate ADR-0001's single-source-of-truth
and gain nothing.

## Why

- **Single source of truth (ADR-0001).** The transport `z = t·(t=1 endpoint) +
  (1−t)·(t=0 endpoint)` is endpoint-agnostic; the velocity derivation is consistent
  for it (recovers `x_tgt − x_src` exactly), and the only singularity (`t → 1`) is
  already clamped by `t_eps`. There is nothing transport-level to add.
- **Rejected — stochastic bridge** (`z = (1−t)·x_src + t·x_tgt + σ(t)·ε`): would
  require a new `σ(t)` schedule and a noise-injecting integrator, and BraTS
  cross-modality within a subject is a deterministic (co-registered) map, so the
  one-to-many capacity buys little.
- **Rejected — velocity-prediction rewrite** (predict `v = x_tgt − x_src`): cleanest
  math (no singularity) but discards the x0-Heun primitive, breaks
  `prediction_type = "sample"` (ADR-0001/0002), and gives the *same* deterministic
  behavior the transport-reuse path already delivers.
- **Deterministic given x_src** (no noise injection) ⇒ validation PSNR/SSIM is
  reproducible without the per-epoch noise re-seeding the noise→data FID needs.

## Consequences

- A `PairedLatentFlowModule` + `PairedLatentFlowPipeline` + a start-from-src sampler
  primitive mirror the noise→data triple (ADR-0005); the **scheduler class is
  shared, not duplicated** — a future reader seeing one `FlowMatchHeun*Scheduler`
  serving both noise→data and src→tgt should land here.
- The rollout's stochastic input is now a *data latent* (`x_src`), not Gaussian
  noise — so `add_noise`'s `noise` argument is semantically "the `t = 0` endpoint,"
  not noise. The naming smell is accepted to avoid a transport fork.
- Because `t = 0` is data, the mid-`t` interpolation `z_t` is a mix of two data
  manifolds; the model cannot recover `x_tgt` from `z_t` alone at `t ≈ 0.5`. That
  disambiguation is handled by conditioning (ADR-0014), not by the transport.
- The `(1−t)⁻²` loss weight is reused: for this transport it is the velocity-MSE
  (`((x_tgt − x̂_tgt)/(1−t))²`), as it is for noise→data.
