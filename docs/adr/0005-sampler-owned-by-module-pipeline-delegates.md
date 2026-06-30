# The x0 Heun sampler is owned by the Module; the Pipeline delegates

ADR-0002 placed the true two-evaluation Heun reverse step in the **Pipeline**.
During-training validation (the unbiased-FID callback) must generate volumes, but
the `LatentFlowPipeline` is the **inference path** and training must not import or
instantiate it — generation in training is delivered by the `spt.Module`. We
therefore move the rollout's *home*: a shared primitive
`sample_latent_flow(unet, scheduler, …)` is the single source of truth for the x0
Heun rollout; `LatentFlowModule.sample()` calls it (in-training generation goes
through the module, which calls the scheduler's `set_timesteps` / `euler_step` /
`heun_correct` directly), and `LatentFlowPipeline.sample_latent` delegates to the
same primitive (inference stays a thin packaging over it).

The two-evaluation Heun **mechanism** is unchanged (still the predictor+corrector
pair ADR-0002 specifies); only its home moves, and `validate_against_hope.py`'s
`<1e-3` parity is preserved because both paths call the same primitive.

## Why

- **Single source of truth.** One rollout, not two copies that can drift and
  silently break the parity a working sampler was migrated for.
- **Train/infer boundary.** Training (the module + Lightning callbacks) never
  reaches for the Pipeline; the Pipeline stays the inference-only object.
- **EMA is transparent.** `module.sample()` uses `self.unet`; the `EMACallback`
  swaps EMA weights into `module.unet` in place, so generation automatically
  reflects the EMA model with no extra wiring.

## Consequences

- `LatentFlowPipeline.sample_latent` becomes a thin delegate; `LatentFlowModule`
  gains a `sample()` generative API alongside `forward` (train step).
- This **supersedes** ADR-0002's "rollout owned by the Pipeline" placement; the
  scheduler's two-phase reverse API (`euler_step` / `heun_correct`) is unchanged.
