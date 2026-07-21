# Paired reward pairs are precomputed offline (inverts ADR-0010)

> **Status: Superseded by [ADR-0034](0034-one-realism-reward-both-grpo-policies-delete-condition-aware.md).**
> The paired-reward offline fake cache is deleted with the pipeline. Kept as decision
> history.

> **Inverts [ADR-0010](0010-online-rollout-in-the-loop-preference-pairs.md)** for the
> paired reward. ADR-0010's online rollout-in-the-loop (fresh `t` + fresh noise each
> fit step, for unbounded pair diversity) transfers **no** value to paired, because
> the paired rollout is deterministic. Paired reward pairs are precomputed **once**
> before training.

The paired reward trains on `{real tgt, generated tgt}` pairs precomputed by a single
pass of the frozen generator over the paired cache. Each generated tgt is
`sample_paired_latent_flow(unet, scheduler, x_src, …)` rolled once; fakes are written
to a **disk fake-cache** (the `roll → cache → train` analogue of the latent cache).
The Paired Reward Module holds **no** generator: `fit` consumes precomputed pairs and
is structurally identical to the JiT reward's `validate` path.

## Why

- **The paired rollout is deterministic — online re-rolling is pure waste.**
  `sample_paired_latent_flow` has no RNG (`z = x_src`, deterministic `set_timesteps`,
  `inference_mode`; `paired_sampler.py`). Re-rolling the fake each fit step yields a
  byte-identical fake at ~`E×` compute (`E` = epochs) with zero diversity gain.
  ADR-0010's diversity source (fresh `t_a,t_b ~ U[0,1)` + fresh `randn`) has no
  paired analog.
- **Memorization is identical online vs offline.** Determinism ⇒ the train set is
  the fixed finite set `{(real_tgt_i, gen(src_i))}`; re-rolling adds no new pair. So
  "online ⇒ no memorization" — ADR-0010's other leg — does not hold for paired either.
  The only honest generalization signal is a held-out-**subject** split, not the
  cadence.
- **Offline drops the generator from the Module.** No unregistered denoiser, no
  `on_fit_start` device move, no scheduler in the Module — a real simplification, and
  `fit` reduces to scoring precomputed pairs (the JiT `validate` path).
- **The fake-cache mirrors the existing latent-cache idiom.** The paired pipeline
  already runs `encode → latent-cache → train`; the reward adds
  `roll → fake-cache → train`. Real tgt is already in the latent cache (no re-store);
  only the fake (~`2 MB × |pairs|`) is cached to disk, because it cannot fit in RAM.

## Consequences

- The Paired Reward Module holds only the `RewardModel` (no frozen generator); `fit`
  and `validate` both consume `{winner, loser}` pair batches — `fit` adds the
  Bradley–Terry backward, `validate` adds metrics.
- A one-time fake-cache build step precedes training (run the frozen generator over
  train + val + probe src latents once). Resume re-reads the cache; the checkpoint
  holds no generator.
- `num_steps` (the rollout Heun budget) is a **one-time precompute cost**, not a
  per-step-per-epoch cost — a larger budget is affordable offline (cf. JiT's
  `precompute_num_steps=4` vs train `num_steps=2`).
