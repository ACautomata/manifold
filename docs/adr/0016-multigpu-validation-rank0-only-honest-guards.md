---
status: proposed
---

# Multi-GPU validation stays rank-0-only; guards made honest, not distributed

> **Status: proposed (decided 2026-07-10).** This records the decision from a
> DDP-correctness audit; the implementation is **not yet landed** (planned PR #1).
> The "Consequences" below describe the *target* state, not today's code.

> **Amendment (2026-07-10): distributed PSNR.** The PSNR/SSIM callback is no
> longer rank-0-only: every rank decodes its own `DistributedSampler` shard of
> the val set and `all_gather`'s the per-volume `(psnr_sum, ssim_sum, count)`, so
> `val/psnr` / `val/ssim` are the **global** mean over the full val set under DDP,
> and the paired `ModelCheckpoint(monitor="val/psnr")` stays on under multi-GPU.
> PSNR's aggregation is a cheap `(sum, count)` reduce (like `val/x0_mae`), NOT
> the "large-effort cross-rank generation + feature-gather" this ADR rejected -
> the two were conflated in the original decision. The premise "the runs are
> short enough to tolerate the stall" is also revisited: an 8-DCU paired-JiT run
> showed 89% validation idle (rank 0 decoding its shard while 7 ranks block at
> the epoch-end metric barrier). **FID stays rank-0-only** - its Frechet
> distance needs a feature-matrix gather (not a `(sum, count)` reduction), so the
> "large effort" argument still holds for FID.

FID / PSNR / SSIM generation is rank-0-only (the ADR-0006 consequence). A DDP
audit found two real defects in *how* that policy is reached today:

1. **Monitor guard diverges from the actual DDP decision on the `devices="auto"`
   path — JiT and Paired only.** `cli.py` (`multi_gpu = isinstance(devices, int)
   and devices > 1` at :144; `monitor_fid = enable_fid and not multi_gpu` at :170)
   and `paired_cli.py` (:145, :148) miss the `"auto"` branch that `build_trainer`
   honors (`trainer.py:76`). So `manifold-train` / `manifold-train-paired` with
   the default `-g 1` on an N>1-GPU host sends `devices="auto"` → `build_trainer`
   enables N-way DDP, but `multi_gpu` stays `False` → the checkpoint monitors a
   **rank-0-only** metric (non-root ranks may warn "monitor metric not found").
   **GRPO does NOT have this bug** — `grpo_cli.py:161` already mirrors
   `build_trainer`, and its `main` passes `1` (not `"auto"`) on `-g 1`
   (`grpo_cli.py:354`).
2. **Cheap scalar metrics are logged rank-local, no `sync_dist`.**
   `val/mean_reward` (`grpo.py:562`), `train/loss_epoch` (`metrics.py:49`), and
   `val/x0_mae` (`metrics.py:82`) each report the local-rank shard mean under
   DDP, not the global mean. (GRPO `validation_step` also runs the full
   rollout+reward on every rank and discards N−1 — `grpo.py:540-563`.) The repo
   already has the fix pattern: `reward.py:305` logs `val/pair_acc` with
   `sync_dist=True`, and the pooled ROC-AUC is `all_gather`'d before logging
   (`reward.py:330,344`).

**Decision:** keep the rank-0-only policy — do **not** build distributed FID/PSNR
(cross-rank generation + all-gather features). Instead make it honest: the monitor
guard matches the trainer's actual strategy; `-g 1` means exactly one device
(multi-GPU requires explicit `-g N`); the cheap linear metrics (`train/loss_epoch`, `val/x0_mae`) migrate
to `torchmetrics.MeanMetric(weight=batch_size)` for the **true sample-weighted
global mean** (amendment below: naive `sync_dist=True` yields a mean-of-per-rank-
means); GRPO `train/loss` (already logged with `batch_size=B`) gets `sync_dist=True`;
a rank-0-shard scope label for `val/psnr`/`val/ssim`/`val/mean_reward`, whose
generation stays rank-0-only); GRPO
`validation_step` generation is gated to `is_global_zero`. Best-checkpoint
selection under multi-GPU stays `save_last` + `every_n_epochs`, with best-by-FID
decided offline via export.

## Why

- **The offline workflow already covers selection.** The deployment path is
  `export_to_native` + offline eval (`scripts/eval_paired_step_sweep.py`,
  `scripts/diag_*`); in-training FID selection is not relied upon under
  multi-GPU. Distributed FID is large effort (distributed generation + feature
  gather + small-sample-bias-corrected math) for low marginal value.
- **`MeanMetric(weight=batch_size)` is the true sample-weighted global mean.**
  The repo already uses `sync_dist=True` for exact cases (`reward.py`), but the
  `train/loss_epoch` / `val/x0_mae` callbacks accumulated `_sum/_n` with no
  `batch_size` - naive `sync_dist` would give a mean-of-per-rank-means, not the
  global mean. `MeanMetric` reduces `sum(loss*B)/sum(B)` across ranks (codex
  M6=PARTIAL; see the amendment above).
- **The `-g 1 → "auto"` default was the root of defect (1).** Defaulting to
  exactly one device removes the dangerous auto→DDP path entirely; multi-GPU
  becomes an explicit, deliberate launch.

## Considered options (rejected)

- **Distributed FID/PSNR** (cross-rank generation + all-gather). Rejected: the
  offline workflow already covers best-selection; not worth the complexity now.
- **Async / offline eval decoupling** (rank-0 generation taken out of the
  synchronous validation loop so N−1 ranks don't stall). Deferred: the stall is
  documented and the runs are short enough to tolerate; revisit if multi-GPU
  long-horizon runs make the idle time material.

## Consequences (target, once PR #1 lands)

- Under multi-GPU the JiT **FID** `ModelCheckpoint` monitor is dropped (`val/fid` is
  rank-0-only); rely on `save_last` + `every_n_epochs` + offline export. The paired
  `val/psnr` monitor stays on (distributed - see the amendment). This is
  ADR-0006's stated consequence, **actually enforced** once the JiT/Paired guard
  matches `build_trainer` (today the `auto` path still lets the monitor linger).
- Logged `val/mean_reward`, `train/loss_epoch`, `val/x0_mae` become global means
  under DDP; `val/psnr`/`val/ssim` are now **global** means too (distributed decode
  + `all_gather` of per-volume sums, per the amendment). Only `val/fid` remains
  rank-0-shard-scoped (Frechet aggregation needs a feature gather).
- GRPO `validation_step` generation runs rank-0-only (today it runs on every rank).
