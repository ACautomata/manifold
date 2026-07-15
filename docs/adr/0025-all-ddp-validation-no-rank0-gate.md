---
status: accepted
---

# All validation metrics are fully DDP (no rank-0-only gate)

> **Status: accepted (decided 2026-07-15).** Supersedes the ADR-0016 rank-0-only
> revert (PR #115). Every validation metric - FID, PSNR/SSIM, `val/mean_reward` -
> runs on all ranks over each rank's `DistributedSampler` shard and is reduced to a
> global value. The rank-0-only gates are removed; the `ModelCheckpoint` monitors
> (`val/fid`, `val/psnr`, `val/mean_reward`) stay on under multi-GPU.

The ADR-0016 rank-0-only revert was a workaround for a DCU/DTK device-side stall under
8-way concurrent full-volume MAISI VAE decode (py-spy: all ranks frozen in
`sliding_window_inference -> _conv_forward`). It made `val/psnr`/`val/ssim` a
rank-0-shard estimate and dropped the PSNR/FID/mean_reward checkpoint monitors under
DDP. This ADR undoes that: validation is fully distributed again.

## Decision

- **All ranks decode / generate / score** their own val shard; no `is_global_zero`
  early-return in `PairedPSNRSSIMCallback._gated`, `FIDCallback._gated`, or the
  GRPO / Paired-GRPO `validation_step`.
- **Per-metric global reduction:**
  - PSNR/SSIM: per-volume `(psnr_sum, ssim_sum, count)` `all_reduce`'d in
    `on_validation_epoch_end` -> true global mean (a cheap `(sum, count)` reduce).
  - FID: per-plane **sufficient statistics** `(sum_x, sum_xxT, n)` `all_reduce`'d to
    global moments, then `frechet_from_moments` runs the exact unbiased Fréchet math
    on every rank - **no feature-matrix gather** (the "large effort" ADR-0016 rejected
    was a feature gather, not a `(sum, stat)` reduce). `features_to_sufficient_stats` +
    `moments_from_sufficient_stats` recover the global `(mu, sigma)` exactly. Symmetric:
    every rank enters one `all_reduce` per plane (zero stats for an empty shard, sized by
    a probed `_feat_dim`, so the collective cannot deadlock).
  - `val/mean_reward`: `sync_dist=True` (the val dataloader is evenly sharded, so the
    synced epoch mean is the global mean - mirroring the `reward.py val/pair_acc` convention).
- **Rank-strided generation:** FID synth seeds `seed + i` for `i % world == rank`, so
  the global synth set is the union across ranks, not `world x` rank-0. (PSNR/SSIM and
  GRPO val need no special seeding - the `DistributedSampler` already shards their data.)
- **Monitors re-enabled under DDP:** `val/fid` (JiT), `val/psnr` (Paired), and the GRPO
  `val/fid` / `val/mean_reward` monitors stay on under multi-GPU (the metrics are now
  global). The `is_multi_gpu` / `not multi_gpu` monitor guards are removed from
  `cli.py`, `paired_cli.py`, `grpo_cli.py`, `paired_grpo_cli.py`.

## Why

- **The deadlock workaround discarded the global mean.** `val/psnr`/`val/ssim` became a
  rank-0-shard estimate (~64 samples) and best-by-PSNR selection fell back to
  `save_last` under DDP - the metric the autoresearch sweeps select on was silently
  coarsened. A `(sum, count)` reduce is cheap and exact; the rank-0-only gate traded
  correctness for a workaround.
- **FID distribution is not "large effort."** ADR-0016 rejected distributed FID as
  "cross-rank generation + feature gather." Generation already runs per-rank (strided);
  the Fréchet math needs only `(mu, sigma, n)` per set, recoverable from
  `(sum_x, sum_xxT, n)` sufficient statistics with a small all_reduce (a `D x D` matrix
  per plane, ~16 MB over HCCL) - not a feature gather.
- **One rule, four sites.** A single policy ("no rank-0-only gate anywhere") is simpler
  to enforce and test than "rank-0-only except FID/except PSNR/except mean_reward."

## The deadlock (honest caveat - probe-pending)

The stall-avoidance rests on the VAE `num_splits`/`save_mem` config (`config_network.yaml`
autoencoder block: `num_splits=4, dim_split=1, save_mem=true`, present since 2026-06-28,
commit `3070d30` - it block-splits the decoder convs via `MaisiConvolution.forward`,
matching the UNet block which runs healthy under 8-DDP training). **This is unverified on
the DCU:** that config was already active when the 2026-07-14 deadlock was diagnosed, and
the MAISI relief path it activates (`autoencoderkl_maisi.py:214-233`: `non_blocking` H2D
+ per-block `gc.collect` + `_empty_cuda_cache`) is itself one of the ranked root-cause
candidates (the 0.4 "memory-relief transfer stall"). Static analysis is self-contradicting
on whether `num_splits=4` is the fix or part of the cause; only the sugon probe (never
run) resolves it.

**The sugon probe gates reliance on this ADR:** 8-DCU `--max-epochs 1
check_val_every_n_epoch=1 val_subset_size=4`, py-spy all 8 ranks through one val epoch.
If the pass completes (all 8 ranks decode via `num_splits=4`), the ADR holds. If it
deadlocks, fall back (in order): (1) serialized-GPU decode (ranks decode one-at-a-time
past a barrier - removes the concurrency trigger, still yields a global mean); (2) CPU
decode (`sw_device='cpu'`, the knob the encode path already uses). Both keep the all-DDP
metric contract; only the decode strategy changes. The architecture keeps the decode
strategy as the one swappable seam if a fallback is needed.

## Consequences

- `val/psnr`, `val/ssim`, `val/fid`, `val/mean_reward` are global means under DDP
  (logged identically on every rank); best-by-metric checkpoint selection works under
  multi-GPU again.
- `frechet_distance_unbiased` delegates to `frechet_from_moments` (the math is
  unchanged; the split enables the sufficient-stats path). `features_to_sufficient_stats` /
  `moments_from_sufficient_stats` are the distributed-FID helpers.
- The `feature_net` is now built on every rank (every rank extracts features for its
  shard) - the rank-0-only lazy build is gone. (~100 MB x N RadImageNet loads; acceptable
  on the training hosts.)
- **Out of scope:** the `reward_cli` / `paired_reward_cli` `val/pair_acc` monitor guards
  still drop under DDP (pre-existing; `val/pair_acc` was already global via `sync_dist`/
  `all_gather` in `reward.py` - its monitor drop is a separate inconsistency, not a
  rank-0-only gate this ADR clears).
