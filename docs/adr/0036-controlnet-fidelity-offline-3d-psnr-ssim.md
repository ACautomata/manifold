# ControlNet translation fidelity is offline 3D PSNR/SSIM against the real target, not FID

The before/after-GRPO comparison needs a quantitative quality metric for **each**
policy. The JiT policy already has one: `val/fid` is computed by the same FID callback
on both sides (base JiT = 17.32, GRPO-JiT ≈ 14.79), so that comparison is apples-to-
apples out of the box. The ControlNet policy has **no scalar recorded on both sides**:
supervised ControlNet logs only `val/x0_mae` (no FID — `config_controlnet_supervised.yaml`,
checkpoint monitor `val/x0_mae` in `controlnet_cli.py`), and the ControlNet-GRPO path
**forcibly disables FID** (`grpo_cli.py`: `fid_active=False` plus `forbidden_callbacks`/
`forbidden_monitors`), logging only `val/mean_reward`. We decide: ControlNet translation
fidelity is measured by **3D PSNR + 3D SSIM of the generated target volume against the
real target volume**, computed in an **offline before/after eval**, not during training.

## Why

- **FID is the wrong axis for paired translation.** It is a reference-free *realism*
  estimator over an unconditional noise→data rollout that ignores the ControlNet —
  ADR-0034 already notes `val/fid` is skipped for the ControlNet policy as "a constant
  frozen-base metric." It says nothing about fidelity to the real `tgt`.
- **The realism reward is fidelity-blind by construction** (ADR-0034: "zero fidelity
  gradient"), so it cannot measure translation fidelity either.
- **3D PSNR/SSIM are the standard full-reference translation metrics** and directly
  answer "does the generated `tgt` match the real `tgt`" — exactly the fidelity signal
  ADR-0034 deferred to a manual "visual check." This ADR supplies that screen
  quantitatively.
- **Why offline, not a training callback.** ADR-0029 imagined PSNR/SSIM as *generative
  callbacks* alongside `FIDCallback`. We deliberately deviate: the ControlNet-GRPO path
  forbids in-training FID-style validation, and training **never persists samples**
  (FID decodes in-memory only; no NIfTI / slice-grid / logger image is written anywhere
  in `src/`). An offline pass that generates from the supervised and the GRPO checkpoint
  under **identical noise + conditioning** is the only route to an apples-to-apples
  number — and the same pass furnishes the visual slice grids.
- **Normalization + reuse.** Both generated and real `tgt` are VAE-decoded and then
  per-sample min-max normalized to `[0,1]` (the Pipeline's published-output convention,
  `min_max_to_unit` in `pipelines/pipeline_utils.py`), so `data_range = 1.0`.
  We reuse MONAI `PSNRMetric(max_val=1.0)` / `SSIMMetric(spatial_dims=3, data_range=1.0)`
  — never hand-rolled, matching the RewardModel-wraps-MONAI ethos.

## Considered options (rejected)

- **Report before-`x0_mae` and after-`mean_reward` separately:** rejected — different
  quantities on different scales; not a comparison.
- **Latent-space `x0_mae` as the headline:** rejected — latent L1 is a training
  surrogate, less interpretable than image-space PSNR/SSIM. It stays a training monitor
  (`val/x0_mae`), not the reported quality metric.
- **Enable FID on the ControlNet-GRPO path:** rejected — it measures frozen-base
  realism, not translation fidelity, and the code forbids it deliberately.
- **Reward-scored before/after:** rejected — fidelity-blind (above).
- **In-training PSNR/SSIM callback (ADR-0029's original framing):** rejected for the
  comparison use-case — it cannot run on the FID-forbidden ControlNet-GRPO path, and an
  offline pass decoupled from training is what makes the before/after pairing exact.

## Consequences

- **Implement** MONAI-backed 3D PSNR/SSIM under `src/manifold/metrics/`, and an offline
  before/after eval entry (a new console entry per the `manifold-<verb>` convention):
  export the after-ckpt via `manifold-export --pipeline controlnet --base-native-dir
  <jit_export>` (ADR-0006), generate before/after under fixed noise + conditioning, score
  PSNR/SSIM, and render 2.5D slice grids (before | after | real `tgt`). Implementation
  pending.
- **Training is unchanged.** GRPO still logs only `val/mean_reward`; supervised
  ControlNet still logs `val/x0_mae`. The fidelity screen is an evaluation-time concern,
  layered on top — it closes the gap ADR-0034 left open without touching the trainer.
- **Scope.** PSNR/SSIM is specific to the paired policy (it needs a real `tgt`); the
  unconditional JiT keeps its comparable `val/fid` + `val/mean_reward` and gets a
  same-noise before/after visual grid (no per-sample ground truth exists for it).
