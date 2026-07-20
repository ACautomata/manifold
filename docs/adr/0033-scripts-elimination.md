# scripts/ elimination — one migrated entry point (export), one retired script (pair-gen), the dead diagnostics removed from git

The brief's "eliminate all scripts/" resolves, after ground-truth verification, to
**one live migration**, **one retirement**, and **cleaning the git-tracked dead
diagnostic**:

- **`scripts/export_checkpoint.py` → `src/manifold/training/export_cli.py`** as the
  entry point **`manifold-export`**. The heavy logic already lives in
  `manifold.training.export.export_to_native`; the new module is a thin CLI shell.
- **`scripts/generate_reward_pairs.py` is retired**, not migrated. It is the
  superseded offline reward-pair workflow (the live path is the online fit-step
  rollout in `modules/reward.py`); making it a first-class entry point would gild
  deprecated code.
- **`scripts/diag_brain_mask_psnr.py` is removed from git** — it is dead against
  current checkpoints (EMA was removed in PR #114; the script reads
  `callbacks['DoubleEMACallback']` that new `.ckpt`s no longer carry). The other
  two diagnostics were never in git. `scripts/` is deleted.

The brief's literal "replace with Lightning.callback" does **not** apply here:
export is a post-hoc ops step and pair-gen is data-prep — neither runs inside a
training loop, so neither is a Lightning callback. The actionable directive is
"eliminate scripts/ in favor of proper entry points", not "convert ops to
callbacks". This is phase D of the four-point architecture refactor (issue #157).

## Why

- **The scope is far smaller than the five-file inventory suggests (verification).**
  Reading the files and the git state reveals: only `export_checkpoint.py` is an
  unambiguous live entry point; `generate_reward_pairs.py` is a documented older
  workflow superseded by online rollout (`openwiki/workflows.md:52`); and of the
  three EMA diagnostics, only `diag_brain_mask_psnr.py` is even in `HEAD`
  (`diag_raw_rollout.py` and `eval_paired_step_sweep.py` are listed in
  `.git/info/exclude` with hardcoded gauss `/data72` paths and were never
  committed). Treating all five uniformly — as the three design angles did —
  would have over-migrated a deprecated script and assumed git history that does
  not exist for two of the files.
- **Retiring pair-gen is the simpler choice (DC9), not a loss.** The live
  reward→GRPO chain uses the online rollout in `modules/reward.py`; the offline
  pair generator has no current consumer. Promoting it to a `manifold-gen-*`
  entry would add a first-class command for code nothing calls, against the
  simplicity bias. Git preserves it.
- **`export_checkpoint` is the one real migration, and it is mechanical.** The
  script already imports `export_to_native` and the `manifold.config` builders;
  it is argparse + the `_ROOT` `sys.path` hack + glue. Moving it next to
  `export.py`, dropping the `sys.path` hack (the editable install puts `src/` on
  the path), and registering the entry is behavior-preserving.
- **The brief's "Lightning.callback" must be reconciled honestly (DC4).** A
  callback runs at training-loop hooks (`on_batch_end`, `on_epoch_end`, …).
  Export loads a `.ckpt`, bakes the inference weights, and writes a native dir —
  no `Trainer` is involved. Force-fitting it as a callback would be an
  anti-pattern. (The deleted EMA diagnostics *were* callback diagnostics — they
  read `DoubleEMACallback` state — which is precisely why they died when the
  callback was removed.)

## Considered options (rejected)

- **Migrate both live scripts to flat entry points** (`manifold-export` +
  `manifold-gen-reward-pairs`): rejected for pair-gen — it is a superseded
  workflow with no consumer; migration would promote deprecated code (DC9).
  Export is migrated alone.
- **Archive the dead diagnostics under `scripts/_archive/`** (honoring the
  `validate_against_hope.py` precedent): rejected as a **category error**. The
  precedent (ADR-0005) archived `validate_against_hope.py` because its logic was
  *migrated into `src/`* and the archive preserved a `<1e-3` parity **proof** —
  archive semantically means "superseded-and-migrated". The EMA diagnostics are
  genuinely dead (never migrated, cannot run); archiving them would mislabel
  dead code as migrated. Two of the three were never in git anyway.
- **A single `manifold-ops` umbrella with subcommands** (`manifold-ops export
  …`): rejected (DC7). It fragments the flat `manifold-train-*` convention, and
  for one ops command a router is pure indirection tax.
- **Add a `data_provider` kwarg to `export_cli.main` for signature parity with
  the five training CLIs:** rejected. There is no warm-data step; the kwarg would
  be a dead parameter. The existing test seam for export is `argv` + tiny
  `tmp_path` artifacts, not `data_provider` injection (DC5).
- **Co-locate `export_cli` under a new `src/manifold/ops/` package:** rejected.
  All existing CLIs live under `training/`; `export_cli` sits beside `export.py`
  (whose functions it shells). No third top-level location.

## Consequences

- **New `src/manifold/training/export_cli.py`** — `def main(argv: list[str] |
  None = None) -> int` plus argparse and the `if __name__ == "__main__"` block,
  matching the training-CLI convention except for the (unused) `data_provider`
  kwarg. The intra-`main` **lazy imports** of the heavy pipeline/controlnet
  classes (`LatentFlowPipeline`, `ControlNetLatentFlowPipeline`,
  `build_controlnet`, `load_vae`) are **preserved** — they keep cold-start cheap
  and must not be hoisted to module top level.
- **Entry point `manifold-export = "manifold.training.export_cli:main"`**
  registered in `pyproject.toml`. The two debt comments
  (`pyproject.toml:57`, `:60`) are removed; the five `manifold-train-*` entries
  are untouched.
- **`scripts/generate_reward_pairs.py` is deleted** (retired, not migrated).
  Git preserves it for anyone who needs the offline pair workflow.
- **`scripts/diag_brain_mask_psnr.py` is deleted from `HEAD`.**
  `diag_raw_rollout.py` and `eval_paired_step_sweep.py` were never tracked (local
  `.git/info/exclude` entries; hardcoded gauss paths) and are left to local
  cleanup — they are not a repo concern.
- **`scripts/` directory is removed** entirely, including the tracked-adjacent
  `__pycache__` — **except** `validate_against_hope.py`, which is *relocated, not
  deleted*. It is the retained sampler-parity proof that `src/manifold/modules/sampler.py:9`
  and ADR-0005 reference; deleting `_archive/` wholesale would leave both pointing
  at a nonexistent file and discard exactly the artifact this ADR distinguishes
  from the dead diagnostics. It moves out of `scripts/` to a non-executable home
  (e.g. `tests/parity/validate_against_hope.py` — it is a verification script, not
  a runtime tool), and the two references are updated to the new path. (The
  category-error argument above applies to the *dead EMA diagnostics* — genuinely
  dead, never migrated — not to this still-referenced parity proof.)
- **Tests are updated**, not rewritten. The `sys.path.insert(0, …/scripts)` +
  `import export_checkpoint as cli` sites (in `tests/test_training_cli.py`) become
  `import manifold.training.export_cli as cli` — importing the **module**, not the
  function, so the existing `cli.main(...)` call seam is preserved unchanged
  (importing `main as cli` would make `cli` a function and break those calls with
  `AttributeError`). In
  `tests/test_reward_pairs.py` **only the script-entry test** is deleted
  (`test_generate_reward_pairs_script_end_to_end`, which `sys.path`-imports the
  retired script); the rest of that file tests still-live production behavior in
  `src/manifold/data/reward_pairs.py` (`generate_reward_pairs`,
  `generate_generated_end_probe`, save/load round-trip, frozen-denoiser load,
  accelerator placement — consumed by `reward_cli` / `grpo_cli`) and stays. The
  injection seam
  (`argv` + `tmp_path`) is unchanged.
- **Deploy cutover (the load-bearing operational risk).** `export_checkpoint.py`
  is a live deploy path — the reward→GRPO task chain and the sugon/gauss/euler
  runbooks invoke `python scripts/export_checkpoint.py` by exact path. Renaming to
  `manifold-export` silently breaks every runbook and the running chain unless
  each target re-runs `pip install -e . --no-deps` (new `[project.scripts]`
  entries only appear after reinstall — per the "python -m vs entry-point"
  discipline, a missing entry on a DCU can exit 0 with only two-three log lines).
  The cutover order matters: deleting `scripts/export_checkpoint.py` while the
  runbooks still call that path leaves every export job failing in the gap
  (reinstall adds `manifold-export` but does not revive the old `python` path), so
  the order is **(1) update every runbook / chain invocation to `manifold-export`
  + reinstall on each cluster, (2) verify the new path works end-to-end on a
  cluster, (3) only then delete `scripts/export_checkpoint.py`**. Until step 3,
  the old script and the new entry coexist (the script is a thin shell over the
  same `export_to_native`, so both produce identical output) — a compatibility
  overlap, not a shim to maintain long-term.
- **ADR-0016 is revised.** Its `:80-82` frames the diag scripts as the *active*
  multi-GPU offline FID-selection deployment path; that statement is stale —
  `val/fid` now runs a per-plane symmetric `all_reduce` (ADR-0025) inside the fit,
  so in-training FID selection already covers multi-GPU. Deleting the diag
  scripts does not open a selection gap; the ADR text is corrected alongside D.
- **OpenWiki is regenerated, not hand-edited.** `openwiki/workflows.md` and
  `operations-and-testing.md` embed `scripts/` paths; per the repo convention
  (CLAUDE.md), the source is updated and the scheduled Action regenerates the
  pages. No manual wiki edits.

## Out of scope (deferred)

- **A second ops entry point** (e.g. an offline-eval CLI reviving a modernized
  `eval_paired_step_sweep`). If a genuine multi-GPU offline-selection need
  resurfaces that the in-fit `val/fid` does not cover, it lands as its own
  single-responsibility CLI then — not pre-built now.
- **Rewriting the retired pair-gen logic** into the online rollout path — it is
  already there (`modules/reward.py`); the offline script is simply dropped.
