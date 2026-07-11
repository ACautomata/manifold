# Paired reward subject split — 2-way, reusing the paired train/val partition

The paired reward trains on **paired-train subjects** and validates on **paired-val
subjects** — the SAME partition paired training holds out, resolved via the **paired**
split mechanism (`_train_val_manifests` / native `val_data_base_dir` /
`split_brats_pair_manifest`), NOT JiT reward's `partition_subjects`/`_subject_id`.

## Why

- **paired-val subjects are already generator-unseen.** Paired training holds out
  paired-val subjects, so the frozen generator never trained on them — reward fakes on
  paired-val subjects are honest (unseen). A 3-way carve adds no honesty; it only makes
  reward-val fakes off-distribution for the generator (GRPO's fakes come from the
  generator's training-subject space).
- **Generalization for this reward = held-out subjects, not held-out generator.** The
  generator is frozen and singular; the reward's generalization question is "does the
  real/fake separability hold on unseen subjects," which the 2-way val answers directly.
- **The two split mechanisms must not be mixed.** The paired split derives subject ids
  from src paths / manifests; JiT's `partition_subjects` derives them by `rsplit('-')`
  on the cache filename. They can silently disagree → subject leakage or
  mis-partition. The paired reward reuses the paired mechanism (and the existing
  `paired_train` cache, `cache_tag='paired_train'`, disjoint sample_ids → free disk
  hits, no write conflict).
- **Rejected — 3-way:** no added honesty (paired-val is already generator-unseen), and
  val fakes off-distribution for the generator.

## Consequences

- The reward data builder reuses `_train_val_manifests` + `PairedLatentDataset` over the
  paired cache (train + val subjects), NOT `CleanLatentDataset` / `partition_subjects`.
- The held-out-subject split is the only honest generalization signal (memorization is
  identical online vs offline — ADR-0020 — so cadence is not a lever).
