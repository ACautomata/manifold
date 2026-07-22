# Online rollout-in-the-loop preference-pair generation (full-range ordered)

The reward model is trained on **preference pairs generated fresh each training
step**, not on a static precomputed cache. The Reward Module holds the **frozen
JiT x0-denoiser**; each fit step, for each clean latent it samples two flow-times
`t_a, t_b ~ U[0, 1)`, noises the clean latent to each via the Scheduler transport,
partial-denoises both with the frozen denoiser, and labels by **input corruption
level** — the **winner** is the larger-`t` (less-corrupted) sample, the **loser**
the smaller-`t`. Because both `t`'s are drawn from the **same** `[0, 1)`
distribution and merely ordered, the same latent can be a winner in one pair and a
loser in another: the winner-`t` and loser-`t` distributions **overlap**, destroying
the single-threshold shortcut. The frozen denoiser produces **no gradients**; the
discriminator scores the detached rollout outputs under Bradley–Terry.

This supersedes the offline *train*-pair path of ADR-0009 (whose disjoint
`[0.5, 1) / [0, 0.5)` halves let a single global clean-ness threshold separate
every pair — `val/pair_acc` saturated at 0.997 in epoch 0 and never moved, and the
reward was uninformative for ranking fully generated latents).

## Why

- **Pairs are always fresh, never memorized.** The offline generator built a static
  16 GB pair set the discriminator could memorize. Rolling fresh each step makes
  the pair set effectively unbounded — the discriminator must learn a *relative*
  realism function, not a fixed lookup.
- **The de-saturation property is structural.** Drawing both `t`'s from the full
  `[0, 1)` range and ordering by `max`/`min` makes the winner and loser corruption
  distributions overlap (a loser-`t` in one pair can exceed a winner-`t` in
  another). No single threshold separates them, so the trivial shortcut the offline
  disjoint split enabled is impossible by construction.
- **The label is cheap and annotation-free.** Ordering by input `t` (corruption
  level) needs no measured output quality — both halves are denoised with the same
  frozen denoiser under the same Heun step budget, so quality differences come from
  corruption level, not step count.
- **Validation stays static and held-out.** The denoiser is frozen ⇒ the val pairs
  (full-range, mirroring the train distribution) and the `[0, 0.5)` generated-end
  probe are precomputed **once** at startup over held-out subjects and reused
  across epochs — validation measures generalization, not memorization, and is not
  re-rolled wastefully every epoch.

## Decisions

- **Frozen-denoiser ownership — unregistered, frozen, device-moved manually.** The
  denoiser is attached via `object.__setattr__(module, "denoiser", unet)` (plain
  assignment would auto-register it into `_modules`, leaking its 3.6 GB into
  `state_dict`, `parameters`, the optimizer, and DDP replication). It is therefore
  absent from the checkpoint and the optimizer; `requires_grad_(False)` + `eval()`;
  moved to the device in `on_fit_start` (the bypass also hides it from Lightning's
  automatic `.to(device)`). Resume re-reads it from `--native-dir` — the checkpoint
  holds no denoiser.
- **Rollout primitive under `no_grad` (not `inference_mode`).** The fit-step rollout
  output is the discriminator's input, where backward must save it; PyTorch forbids
  saving an `inference_mode` tensor for backward (`detach`/`clone` do not clear the
  flag). `no_grad` keeps the no-activation-retention while yielding a backward-safe
  tensor. Parity with `sample_latent_flow` (ADR-0008) is preserved — both contexts
  run identical math; only the tensor flag differs.
- **One combined `[2B]` winner-first rollout call.** Both halves are noised and
  partial-denoised in a single `partial_denoise_rollout` with
  `clean_2b = cat(clean, clean)`, `t_start = cat(winner_t, loser_t)`, and per-sample
  `spacing`/`modality` duplicated to `2B`. A batch-size assertion in the rollout
  surfaces a mismatched combined batch as a clear error (not a MAISI-internal
  crash); the reward split in `_score_pair` matches the winner-first order.
- **The `t`-guard is a difficulty knob, not numerical safety.** `torch.rand` is
  half-open `[0, 1)`, so `t == 1` (the only true NaN source — the step-start
  denominator `1 − t` vanishes) is precluded. On the uniform partial grid
  `v1·dt` cancels the `1/(1−t)` denominator, so the update is bounded by
  `(x0 − z)` independent of `t_start`; an optional `winner_t_guard` caps the
  winner's start to make pairs harder — it is not a load-bearing safety cap.
- **Checkpoint monitors `val/gen_pair_acc`.** The generated-end probe (both samples
  `t ∈ [0, 0.5)`, ordered by `t`) is the GRPO-regime metric — ranking within the
  all-generated regime the policy operates in. `val/pair_acc` and `val/roc_auc`
  remain logged for diagnosis. The DDP multi-GPU fallback (drop the rank-local
  monitor, keep `save_last` + `save_top_k = 1`) is preserved.
- **Rollout-cost budget — the per-step train `num_steps` is the lever.** Every-step
  rollout at `num_steps = 4` × 50 epochs is ~50× the offline one-time rollout.
  Mitigation: a smaller per-step train `num_steps` (2 ⇒ 3 Heun evals/step vs 4 ⇒ 7),
  with `num_steps = 4` reserved for the one-time val/probe precompute. Chunking the
  rollout bounds **peak memory, not wall-clock** (same FLOPs). Throughput is
  measured on the target cluster before committing to the full schedule (issue #52).

## Consequences

- The Reward Module now holds the frozen denoiser (unregistered) + the frozen
  Partial Flow-Match Heun Scheduler + the train `num_steps` + the `t`-guard.
  `forward("fit")` consumes a clean-latent batch (`{latent, spacing, label}`);
  `forward("validate")` still consumes precomputed `{winner, loser}` pairs. `forward`
  branches on `stage` and fails loudly on a mismatched batch.
- Training reads the warmed latent cache via a clean-latent dataset (scale-on-read)
  over **train subjects only** — the held-out-subject split is enforced at the
  dataloader, not only at val-precompute (no leakage).
- The offline pair-generation functions `generate_reward_pairs` /
  `generate_generated_end_probe` are retained for the one-time val/probe
  precompute and offline inspection; the offline *train*-pair path is
  superseded. (The standalone `scripts/generate_reward_pairs.py` wrapper was
  retired in ADR-0033; these functions live in `src/manifold/data/reward_pairs.py`.)
  A new `generate_full_range_val_pairs` mirrors the train distribution for
  validation.
- ADR-0009's preference-pair *definition* changes (disjoint halves → full-range
  ordered); ADR-0009's generated-end-probe rationale (measure the train/inference
  gap) is unchanged and is now the selection metric.
