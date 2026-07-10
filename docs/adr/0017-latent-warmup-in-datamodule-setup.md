---
status: proposed
---

# Latent cache warm-up moves to `DataModule.setup()` for PG-initialized sharding

> **Status: proposed (decided 2026-07-10).** This records the decision; the
> implementation is **not yet landed** (planned PR #2). Today the warm still runs
> in `main()` before `trainer.fit`, exactly as the defect describes.

The DDP sharding machinery in `LatentDataset.warm_cache` — each rank encodes its
`i % world == rank` shard, all ranks `barrier()`, then every rank loads the full
set — is **dead code in production**. The JiT CLI warms inside `main()` *before*
`trainer.fit` (`cli.py:276` → `_warm_data` → `warm_latent_pipeline` at
`cli.py:349`), when Lightning has not yet initialized the process group
(`dist.is_initialized()` is `False`), and never passes `rank`/`world`
(`warm_latent_pipeline` defaults to `world=1` at `latent_pipeline.py:183`, so
`sharded = world > 1` is `False` at `latent_dataset.py:127` — the sharded branch
is unreachable). The Paired CLI has the same shape via
`PairedLatentDataset.warm_cache` (called from `paired_cli.py` without `rank`/
`world`). The **certain** harm is world× redundant VAE encode on every cold start
(the ~2.7h cold-start cost). A **secondary, theoretical** risk is cold-cache
`scale_factor` diverging across ranks — but `estimate_scale_factor` is
deterministic over the first *n* warmed latents (`latent_dataset.py:268`), so
static inspection establishes redundant compute, not proven divergent values; the
divergence would only arise from non-deterministic encode (e.g. cuDNN), and is
expected to be negligible.

**Decision:** move the VAE-encode warm into a Lightning **`DataModule.setup()`**
— the post-PG-init, per-rank hook — for the two CLIs that encode
(`manifold-train`, `manifold-train-paired`). There `dist.is_initialized()` is
true, the barriers fire, the already-written + tested `i % world == rank`
sharding activates: one writer per cache file and ~8× faster cold start. The
Module is sized from `len(vol_ds)` (the source volume count, known before warm)
instead of `len(latent_ds)`; the seeded `val_latents` subset is built inside
`setup()` and exposed to the FID callback. `manifold-train-reward` and
`manifold-train-grpo` are **unaffected by this bug** — they load pre-computed
`.pt` latents (`reward_cli.py:315`, `grpo_cli.py:477`) and do not VAE-encode
training data (GRPO still loads a VAE, but only for FID decode/latent scaling at
`grpo_cli.py:430`).

## Why

- **`setup()` is the Lightning-blessed post-PG hook.** Lightning initializes the
  process group inside `trainer.fit` before calling `setup()`, so the sharding
  works without any manual PG juggling — the one structural reason the warm was
  unreachable disappears.
- **It activates code we already wrote and tested.** The barrier symmetry,
  strided coverage, atomic writes, and resume idempotency were verified correct
  *as written*; they simply never executed. Moving the call site is a small
  change that turns them on.
- **Cold-start time is the felt harm.** New-cluster / new-dataset runs are the
  cold-cache case; the 8× warm speed-up is felt there. (Rank-identical
  `scale_factor` is a domain invariant either way — redundant encode today still
  converges to identical values; sharding makes it faster, not more correct,
  absent non-deterministic ops.)

## Considered options (rejected)

- **Early manual `init_process_group("nccl")` in `main()`** + read `rank`/`world`
  from `dist`. Rejected: smallest diff, but pre-initializing the PG before
  Lightning is the low-level cleverness that bites later (Lightning's launcher
  semantics are an abstraction worth keeping intact). Chose the idiomatic hook.
- **Rank-0-only encode via `prepare_data()`** (single owner, no shard math).
  Rejected: no warm parallelism — cold start stays ~2.7h wall — and still a
  DataModule change, so it buys little for the same refactor cost.

## Consequences (target, once PR #2 lands)

- The JiT / Paired CLIs construct the Module from `len(vol_ds)` (not the warmed
  `latent_ds`), and `val_latents` is produced inside `setup()` (so the FID
  callback reads it lazily rather than receiving it pre-warm in `run_training`).
- `warm_latent_pipeline` / `warm_cache` read `rank`/`world` from `dist` when the
  PG is initialized (falling back to `0`/`1`) instead of trusting caller args;
  the CLI call sites drop those args. (Today the caller-arg signatures still
  exist at `latent_pipeline.py:183` / `latent_dataset.py:113`.)
- No deadlock risk: the barriers are symmetric and unconditional within the
  sharded branch, and the only way to reach them is the post-PG `setup()` path.
