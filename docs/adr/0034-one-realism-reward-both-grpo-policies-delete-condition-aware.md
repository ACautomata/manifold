# One realism reward for both GRPO policies — delete the condition-aware paired reward

The unified `GRPOModule` trains **either** the UNet (the JiT x0-denoiser) **or** a
ControlNet on the frozen base UNet — and both policies are scored by the **same**
reward: the mode-agnostic `RewardModel` (a PatchGAN discriminator) scoring the terminal
latent `z_K` unconditionally, `in_channels = C_latent`, trained on partial-denoise
corruption-level pairs (ADR-0009/0010). The **condition-aware paired reward**
(ADR-0019 — `concat([x_src, tgt])`, `2·C_latent`, real-vs-fake) and its entire pipeline
(`PairedRewardModule`, `paired_reward_cli`, `paired_reward_pairs`, the offline fake
cache, the probe) are **deleted**. ADR-0018–0023 are superseded; ADR-0028 is amended.

## Why

- **The reward scores the backbone's noise→data output, not the ControlNet.** `z_K` is
  the terminal latent of the Heun rollout — the base UNet's denoising prediction
  (`base(z) + controlnet_residual(z, x_src)`); the ControlNet only injects residuals and
  is never scored directly (`controlnet_sampler.py:95-110`). The reward is trained on
  **partial-denoise** samples with **no `x_src` anywhere** — its discrimination axis is
  noise→data corruption / denoising quality (`reward.py:244-287`, ADR-0009/0010). It is
  mode-agnostic by construction, so one reward serves both policies.
- **ADR-0019's copy-src premise died with the Paired-JiT transport.** ADR-0019 required
  condition-awareness to defeat the copy-src identity shortcut of Paired JiT
  (ADR-0013/0014), where `x_src` was the `t = 0` transport endpoint and copy-src was the
  trivial "output the endpoint" minimum. ADR-0028 deleted that transport; in the
  ControlNet regime `x_src` is a **control signal**, not an endpoint
  (`controlnet_sampler.py:6,125-131` — the rollout starts from noise). The failure
  ADR-0019 was designed against no longer exists.
- **Translation fidelity is not the reward's job.** For the ControlNet policy, fidelity
  is enforced by the ControlNet conditioning on `x_src` + the supervised init
  (`ControlNetLatentFlowModule`, MSE on `x_tgt`, ADR-0027) + the KL anchor
  (`deepcopy(base + controlnet)`, ADR-0015). The reward supplies only realism — a
  refinement signal that ranks among `x_src`-conditioned group siblings.
- **This codifies the code reality.** The GRPO reward site already scored `z_K`
  unconditionally in both cases (`grpo.py:260-263` — `reward_fn(z_K)` /
  `reward_model(z_K)`); a `2·C` condition-aware reward would in fact crash on the
  `C`-channel `z_K`. ADR-0028/0019's "condition-aware concat for Mode-2" was never
  wired. This ADR reconciles the docs with the code.

## Considered options (rejected)

- **Keep the condition-aware paired reward for the ControlNet policy (ADR-0019):**
  rejected — its premise (Paired-JiT copy-src) is gone with the transport, and it is
  redundant with the ControlNet's own `x_src` conditioning (ADR-0019 itself noted it
  "mirrors the paired UNet's own conditioning"). It was also never consumed by GRPO.
- **Make both policies condition-aware:** rejected — the UNet policy is unconditional
  noise→data generation; there is no `x_src` to concat. Condition-awareness is
  inapplicable to the UNet policy.
- **Accept the shared reward but keep the paired-reward pipeline "for later":** rejected
  — an orphaned `2·C` reward pipeline plus six ADRs is long-term debt; if
  condition-awareness is ever needed again it is a small, well-understood re-add.

## Consequences

- **Fidelity is delegated to conditioning + init + anchor, not the reward.** Accepted
  residual risk: a realism-only reward carries zero fidelity gradient; a residual
  copy-src basin (the ControlNet learning to route `x_src → z_K`) is possible but
  non-trivial and is resisted by the supervised init + KL anchor. Monitor via
  `val/mean_reward` + `val/fid` + translation PSNR / visual check; if copy-src appears,
  reintroduce condition-awareness (the rejected option above).
- **DELETE** the paired-reward pipeline: `modules/paired_reward.py`,
  `training/paired_reward_cli.py`, `data/paired_reward_pairs.py`,
  `configs/train/config_paired_reward.yaml`, the `manifold-train-paired-reward` entry
  point, and the paired-reward tests.
- **RELOCATE** two shared symbols the ControlNet path still needs:
  `load_frozen_controlnet_generator` → `training/controlnet_inputs.py` (consumed by
  `grpo_cli`); `_train_val_manifests` → `data/paired_manifests.py` (consumed by both
  `controlnet_cli` and `grpo_cli`).
- **Amend ADR-0028:** the "Mode-1/Mode-2" vocabulary is dropped — `GRPOModule` trains
  whichever policy it is given, inferred from whether `controlnet` is present in the
  inputs (no `--grpo-mode` flag); the reward consequence is corrected to "both policies
  score `z_K` with the shared unconditional realism reward." The unification (one module,
  spine reuses verbatim, bridge deleted) stands.
- **Supersede ADR-0018, 0019, 0020, 0021, 0022, 0023** (0021's generator loader and
  0022's subject-split helper survive, relocated as above).
