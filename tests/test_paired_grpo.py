"""G2RPO (Paired GRPO, Brownian-bridge) tests: scheduler + forked rollout + Module
+ CLI (#103).

External-behavior seams (PRD #102, mirroring tests/test_grpo.py + the paired stack):
the bridge ``sde_step_mean`` mean equals the inherited ``euler_step`` (the §5
no-Langevin collapse) and its std is the §7 exact transition (vanishing at the
terminal, θ-independent ⇒ the equal-variance KL holds); the forked rollout's anchor
at η=0 reproduces ``sample_paired_latent_flow`` (the deployed Paired JiT sampler),
threads ``cat([z, x_src])`` through every UNet eval, and G-expands ``x_src`` BEFORE
the reward concat (D9); the Module's backward touches the policy UNet ONLY (the
frozen reward + reference are unregistered); the KL anchor is ~0 at init and grows
with drift; and a G2RPO run completes end-to-end on toy injected policy + frozen
reward via the CLI smoke, writing a checkpoint and logging ``val/mean_reward``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from manifold import (
    FlowMatchBridgeGRPOScheduler,
    FlowMatchHeunDiscreteScheduler,
    RewardModel,
)
from manifold.modules import sample_paired_latent_flow, singular_branch_rollout_paired
from manifold.modules.grpo import clipped_surrogate_loss, gaussian_log_prob, group_advantage

#: A tiny latent shape + paired RewardModel config (in_channels = 2·C_latent = 8).
#: ``paired_unet`` (conftest) is UNet3DConditionModel(in_channels=8, out_channels=4).
_LAT = (4, 8, 8, 8)
_RM_KW = dict(spatial_dims=3, in_channels=8, channels=8, num_layers_d=1)


def _reward_model() -> RewardModel:
    torch.manual_seed(0)
    return RewardModel(**_RM_KW)


# -- FlowMatchBridgeGRPOScheduler.sde_step_mean (the §5/§7 bridge transition) -----


def test_sde_step_mean_equals_euler_step_no_langevin(paired_unet):
    """mean == euler_step(x̂₁, z, t, t_next)[0] exactly — the §5 no-Langevin collapse.

    The forward Doob h-transform drift is *exactly* the euler velocity (the score is
    absorbed into the pin); the equimarginal GRPO scheduler adds a ``(σ²/2t)·x_θ``
    Langevin correction because it time-reverses, the bridge does not. So the bridge
    mean is byte-identical to the inherited ``euler_step`` output (#103 acceptance).
    """
    sched = FlowMatchBridgeGRPOScheduler(eta=0.7)
    sched.set_timesteps(8)
    torch.manual_seed(0)
    z = torch.randn(2, *_LAT)
    x0 = torch.randn(2, *_LAT)
    t, t_next = 0.25, 0.375  # an interior node pair
    mean, std = sched.sde_step_mean(x0, z, t, t_next)
    euler_mean, _ = sched.euler_step(x0, z, t, t_next)
    assert torch.equal(mean, euler_mean), "bridge mean must equal euler_step (no Langevin term)"


def test_sde_step_mean_std_formula_matches_section7():
    """std == sqrt(η·Δt·(1−t_next)/(1−t)) — the §7 exact transition (closed-form)."""
    sched = FlowMatchBridgeGRPOScheduler(eta=0.7)
    sched.set_timesteps(8)
    z = torch.randn(1, *_LAT)
    x0 = torch.randn(1, *_LAT)
    for t, t_next in [(0.0, 0.125), (0.25, 0.375), (0.5, 0.625)]:
        _, std = sched.sde_step_mean(x0, z, t, t_next)
        dt = t_next - t
        expected = math.sqrt(0.7 * dt * (1.0 - t_next) / (1.0 - t))
        assert math.isclose(float(std), expected, rel_tol=1e-12), (t, t_next, float(std), expected)


def test_sde_step_mean_std_vanishes_at_terminal():
    """std → 0 as t_next → 1 (σ² = η·Δt·(1−t_next)/(1−t) → 0; z_K → x̂₁, stable)."""
    sched = FlowMatchBridgeGRPOScheduler(eta=0.7)
    sched.set_timesteps(8)
    z = torch.randn(1, *_LAT)
    x0 = torch.randn(1, *_LAT)
    _, std = sched.sde_step_mean(x0, z, 0.875, 1.0)
    assert float(std) == 0.0, f"std must vanish at the terminal (t_next=1), got {float(std)}"


def test_sde_step_mean_std_is_theta_independent():
    """std is unchanged across different x̂₁ inputs ⇒ equal-variance KL closed form holds.

    The bridge σ depends only on (t, t_next, η), never on θ — so the policy and
    frozen-reference transitions share variance and the diagonal-Gaussian KL collapses
    to 0.5·‖μ_θ − μ_ref‖²/σ² (trace + log-det cancel). A θ-dependent σ would silently
    break the closed form (#103 acceptance).
    """
    sched = FlowMatchBridgeGRPOScheduler(eta=0.7)
    sched.set_timesteps(8)
    z = torch.randn(1, *_LAT)
    _, std = sched.sde_step_mean(torch.randn(1, *_LAT), z, 0.25, 0.375)
    for _ in range(3):
        _, std_other = sched.sde_step_mean(torch.randn(1, *_LAT) * 100.0, z, 0.25, 0.375)
        assert math.isclose(float(std), float(std_other), rel_tol=1e-15), "std must not depend on x̂₁"


def test_sde_step_mean_matches_closed_form_bridge_transition():
    """The full (mean, std) matches the manually-derived exact bridge transition.

    Integrating-factor solution of dZ = (x̂₁−Z)/(1−t) dt + √η dW (pinned at Z_1=x̂₁):
    mean = [(1−t_next)·Z + Δt·x̂₁]/(1−t); std² = η·Δt·(1−t_next)/(1−t). The mean is
    algebraically identical to euler_step (verified separately above); this checks the
    closed-form mean directly against the formula, end-to-end (#103 acceptance: the
    tests encode the adversarially-verified math).
    """
    sched = FlowMatchBridgeGRPOScheduler(eta=0.7)
    sched.set_timesteps(8)
    torch.manual_seed(0)
    z = torch.randn(2, *_LAT)
    x0 = torch.randn(2, *_LAT)
    t, t_next = 0.375, 0.5
    mean, std = sched.sde_step_mean(x0, z, t, t_next)
    dt = t_next - t
    expected_mean = ((1.0 - t_next) * z + dt * x0) / (1.0 - t)
    expected_std = math.sqrt(0.7 * dt * (1.0 - t_next) / (1.0 - t))
    assert torch.allclose(mean.float(), expected_mean, atol=1e-6)
    assert math.isclose(float(std), expected_std, rel_tol=1e-12)


# -- singular_branch_rollout_paired (the forked 5-site rollout) ------------------


def test_rollout_anchor_eta_zero_matches_sample_paired_latent_flow(paired_unet):
    """η=0, last-step branch ⇒ terminal z_K == sample_paired_latent_flow (no fork).

    The anchor runs the same two-eval Heun from x_src as the deployed Paired JiT
    sampler; with η=0 the single bridge SDE step reduces to the euler advance (the
    §5 mean) and the suffix is empty for the last step. So the terminal latent is
    bit-identical to ``sample_paired_latent_flow`` over the same grid — the
    anchor-parity guard (#103 acceptance: the anchor uses the existing Heun primitive).
    η=0 ⇒ std·ξ = 0, so all G siblings are identical; sibling 0 is the reference.
    """
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    n = 4
    bridge = FlowMatchBridgeGRPOScheduler(eta=0.0)
    base = FlowMatchHeunDiscreteScheduler()
    buf = singular_branch_rollout_paired(
        paired_unet, bridge, _reward_model(), x_src, [1.0, 1.0, 1.0], 1, 2,
        G=2, eta_step_list=[n - 1], num_steps=n,
    )
    z_K = buf[0]["z_kplus1"][:, 0]
    ref = sample_paired_latent_flow(
        paired_unet, base, x_src, [1.0, 1.0, 1.0], 1, 2, num_inference_steps=n
    )
    assert torch.equal(z_K, ref)


def test_rollout_buffer_shapes_carry_the_group(paired_unet):
    """Buffer per step: z_k (B,...), z_{k+1} (B,G,...), old_log_prob / advantage (B,G)."""
    torch.manual_seed(0)
    B, G, n = 2, 3, 4
    x_src = torch.randn(B, *_LAT)
    buf = singular_branch_rollout_paired(
        paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.7), _reward_model(), x_src,
        [1.0, 1.0, 1.0], 1, 2, G=G, eta_step_list=[0, 1], num_steps=n,
    )
    assert len(buf) == 2
    entry = buf[0]
    assert entry["z_k"].shape == (B, *_LAT)
    assert entry["z_kplus1"].shape == (B, G, *_LAT)
    assert entry["old_log_prob"].shape == (B, G)
    assert entry["advantage"].shape == (B, G)
    assert entry["rewards"].shape == (B, G)
    assert entry["t_k"] < entry["t_next"]


def test_rollout_outputs_are_detached_no_grad(paired_unet):
    """The rollout is fully no_grad — buffer tensors carry no autograd graph."""
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    buf = singular_branch_rollout_paired(
        paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.7), _reward_model(), x_src,
        [1.0, 1.0, 1.0], 1, 2, G=3, eta_step_list=[0], num_steps=4,
    )
    e = buf[0]
    for key in ("z_k", "z_kplus1", "old_log_prob", "advantage", "rewards"):
        assert not e[key].requires_grad, f"{key} must be detached (no_grad rollout)"
        assert e[key].grad_fn is None


def test_rollout_siblings_diverge_with_noise(paired_unet):
    """With η>0 the G siblings get distinct terminal rewards (the bridge draw explores)."""
    torch.manual_seed(0)
    x_src = torch.randn(1, *_LAT)
    buf = singular_branch_rollout_paired(
        paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.9), _reward_model(), x_src,
        [1.0, 1.0, 1.0], 1, 2, G=4, eta_step_list=[0], num_steps=3,
    )
    rewards = buf[0]["rewards"]  # (1, G=4)
    assert rewards.unique().numel() > 1


def test_rollout_runs_for_each_perturbed_step(paired_unet):
    """A 2-element eta_step_list yields a 2-entry buffer at the right grid nodes."""
    torch.manual_seed(0)
    x_src = torch.randn(1, *_LAT)
    buf = singular_branch_rollout_paired(
        paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.5), _reward_model(), x_src,
        [1.0, 1.0, 1.0], 1, 2, G=2, eta_step_list=[0, 2], num_steps=4,
    )
    assert [e["t_k"] for e in buf] == [0.0, 0.5]  # nodes[0]=0, nodes[2]=0.5 on n=4


def test_rollout_rejects_terminal_eta_step(paired_unet):
    """eta_step_list max must be < num_steps (a perturbed step needs a suffix node)."""
    x_src = torch.randn(1, *_LAT)
    with pytest.raises(ValueError, match="must be < num_steps"):
        singular_branch_rollout_paired(
            paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.5), _reward_model(), x_src,
            [1.0, 1.0, 1.0], 1, 2, G=2, eta_step_list=[4], num_steps=4,
        )


def test_rollout_rejects_terminal_step_with_eta_positive(paired_unet):
    """The §7 std collapses (σ²→0) at the terminal step with η>0 ⇒ reject it.

    At k=num_steps−1, t_next=1.0 ⇒ std²=η·Δt·(1−1)/(1−t)=0 ⇒ all G siblings identical
    ⇒ zero advantage ⇒ a silently wasted inner step (no gradient, no error). The guard
    rejects this for η>0. η=0 (the anchor-parity debug case) is exempt — std=0
    everywhere by construction there, so the terminal step is allowed."""
    x_src = torch.randn(1, *_LAT)
    with pytest.raises(ValueError, match="var-collapse terminal"):
        singular_branch_rollout_paired(
            paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.5), _reward_model(), x_src,
            [1.0, 1.0, 1.0], 1, 2, G=2, eta_step_list=[3], num_steps=4,  # k=3=num_steps−1
        )


def test_rollout_allows_terminal_step_at_eta_zero(paired_unet):
    """η=0 is the anchor-parity debug case — the terminal step is allowed (std=0 everywhere)."""
    torch.manual_seed(0)
    x_src = torch.randn(1, *_LAT)
    buf = singular_branch_rollout_paired(
        paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.0), _reward_model(), x_src,
        [1.0, 1.0, 1.0], 1, 2, G=2, eta_step_list=[3], num_steps=4,  # terminal, η=0
    )
    assert len(buf) == 1


def test_rollout_reward_pairs_each_sibling_with_correct_source(paired_unet):
    """D9 (silent trap): each sibling's reward pairs with its OWN source.

    The reward scores ``cat([x_src_bg, z_K])`` where ``x_src_bg =
    x_src.repeat_interleave(G, dim=0)`` — x_src MUST be G-expanded BEFORE the concat
    (flat index b·G+g = sibling g of b). A wrong expansion order (e.g. ``repeat``
    instead of ``repeat_interleave``) pairs siblings with the wrong source ⇒ corrupted
    advantage, no error. This spies on the reward input and asserts the source half of
    each row matches ``x_src[row // G]`` (#103 acceptance).
    """
    B, G = 2, 3
    torch.manual_seed(0)
    # Distinct per-source constants so a mis-pairing is detectable.
    x_src = torch.stack([torch.full(_LAT, 1.0), torch.full(_LAT, -1.0)])
    captured: list[torch.Tensor] = []
    base_reward = _reward_model()

    class _SpyReward(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            captured.append(x.detach().clone())
            return self.base(x)

    spy = _SpyReward(base_reward).eval()
    singular_branch_rollout_paired(
        paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.7), spy, x_src,
        [1.0, 1.0, 1.0], 1, 2, G=G, eta_step_list=[0], num_steps=4,
    )
    assert len(captured) == 1
    reward_input = captured[0]  # (B·G, 2·C, ...)
    assert reward_input.shape[0] == B * G
    C = _LAT[0]
    for i in range(B * G):
        # The source half of row i must equal x_src[i // G] (repeat_interleave order).
        assert torch.equal(reward_input[i, :C], x_src[i // G]), (
            f"D9 violation: reward row {i} (sibling {i % G} of batch {i // G}) paired "
            f"with the wrong source."
        )


def test_rollout_threads_source_concat_through_every_unet_eval(paired_unet):
    """Every rollout UNet eval sees cat([z, x_src]) (2·C channels) + src/tgt labels.

    The 3 rollout eval sites — anchor, perturbed-step, suffix — all thread the source
    concat. A missed site (e.g. the suffix reusing the (B,) source) would silently feed
    the UNet a mis-shaped or wrong-batch input. This spies on the UNet and asserts every
    call carries the doubled channel count and both contrast labels (#103 acceptance).
    """
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    calls: list[tuple[int, bool, bool]] = []

    class _SpyUnet(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, sample, timestep, spacing, class_labels_src=None,
                    class_labels_tgt=None, **kw):
            calls.append((int(sample.shape[1]), class_labels_src is not None,
                          class_labels_tgt is not None))
            return self.base(sample=sample, timestep=timestep, spacing=spacing,
                             class_labels_src=class_labels_src, class_labels_tgt=class_labels_tgt)

    spy = _SpyUnet(paired_unet)
    singular_branch_rollout_paired(
        spy, FlowMatchBridgeGRPOScheduler(eta=0.7), _reward_model(), x_src,
        [1.0, 1.0, 1.0], 1, 2, G=2, eta_step_list=[0, 1], num_steps=4,
    )
    assert calls, "expected UNet evals during the rollout"
    two_c = 2 * _LAT[0]
    for ch, has_src, has_tgt in calls:
        assert ch == two_c, f"every UNet eval must see cat([z, x_src]) ({two_c} ch), got {ch}"
        assert has_src and has_tgt, "every UNet eval must carry src + tgt contrast labels"


# -- PairedGRPOModule (the policy learner) ---------------------------------------


def _module(latent_shape=_LAT, **kw):
    """A tiny PairedGRPOModule: a fresh trainable paired policy UNet + tiny frozen reward."""
    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule

    torch.manual_seed(0)
    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    return PairedGRPOModule(
        policy, _reward_model(), FlowMatchBridgeGRPOScheduler(eta=0.5),
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3, **kw
    )


def test_module_backward_updates_unet_only():
    """backward populates UNet (policy) grads; the frozen reward is unregistered.

    The Module HOLDS the frozen paired reward (unregistered via object.__setattr__) — so
    it is absent from parameters()/state_dict()/optimizer, and backward only touches the
    policy UNet (#103 acceptance: the exclusion invariant)."""
    mod = _module()
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    buf = singular_branch_rollout_paired(
        mod.unet, mod.scheduler, mod.reward_model, x_src, [1.0, 1.0, 1.0], 1, 2,
        G=2, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    step = buf[0]
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    src_labels = torch.tensor([1, 1], dtype=torch.long)
    tgt_labels = torch.tensor([2, 2], dtype=torch.long)
    new_lp, _mean_new, _std_new = mod._new_log_prob(step, spacing_t, src_labels, tgt_labels, x_src)
    loss = clipped_surrogate_loss(new_lp, step["old_log_prob"], step["advantage"], mod.clip_range)
    loss.backward()
    unet_params = list(mod.unet.parameters())
    assert unet_params and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in unet_params
    )
    assert all(p.grad is None for p in mod.reward_model.parameters())
    assert "reward_model" not in mod.state_dict()
    opt = mod.configure_optimizers()["optimizer"]
    opt_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert opt_ids == {id(p) for p in unet_params}
    assert opt_ids.isdisjoint({id(p) for p in mod.reward_model.parameters()})


def test_module_advantage_group_normalized_in_buffer():
    """The buffer's advantage is the group-normalized (R−mean)/std over the G siblings.

    Asserts the stored advantage equals ``group_advantage(rewards)`` computed from the
    buffer's OWN rewards — the wired invariant (the rollout group-normalizes its
    terminal rewards). The group-normalization math itself (zero-mean over G before the
    clip) is covered by ``test_group_advantage_is_group_normalized_over_G`` in
    test_grpo.py (shared code); the tiny-CPU PatchGAN can score near-degenerate sibling
    rewards (std ≈ 1e-8), where a zero-mean assertion would be fp-noise-amplified.
    """
    mod = _module()
    torch.manual_seed(0)
    x_src = torch.randn(1, *_LAT)
    buf = singular_branch_rollout_paired(
        mod.unet, mod.scheduler, mod.reward_model, x_src, [1.0, 1.0, 1.0], 1, 2,
        G=4, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    A = buf[0]["advantage"]  # (1, G=4)
    expected = group_advantage(buf[0]["rewards"], adv_clip_max=mod.adv_clip_max)
    assert torch.allclose(A, expected, atol=1e-6), "buffer advantage must be group_advantage(rewards)"


def test_module_rejects_degenerate_group_size():
    """G < 2 raises: torch.std (Bessel) needs ≥2 siblings, else NaN advantage → NaN grads."""
    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule

    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    with pytest.raises(ValueError, match="G must be >= 2"):
        PairedGRPOModule(policy, _reward_model(), FlowMatchBridgeGRPOScheduler(), G=1)


def test_module_rejects_empty_eta_step_list():
    """An empty eta_step_list ⇒ a silent no-op training (no optimizer step); reject it."""
    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule

    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    with pytest.raises(ValueError, match="eta_step_list must be non-empty"):
        PairedGRPOModule(
            policy, _reward_model(), FlowMatchBridgeGRPOScheduler(),
            G=2, eta_step_list=[], num_steps=3,
        )


def test_module_rejects_kl_coef_without_reference():
    """kl_coef>0 with no reference_policy is a silent dead KL anchor; fail fast.

    Without a reference, ``_transition_kl`` short-circuits to None and the KL term is
    never added — the policy would reward-hack with no indication the regularizer is
    off. The guard makes a forgotten reference (e.g. #104's _real_inputs) crash loudly
    instead of silently degrading the launch."""
    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule

    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    with pytest.raises(ValueError, match="requires a reference_policy"):
        PairedGRPOModule(
            policy, _reward_model(), FlowMatchBridgeGRPOScheduler(),
            G=2, eta_step_list=(0,), num_steps=3, kl_coef=0.1,  # reference_policy omitted
        )


# -- v2: bounded reward + KL anchor (ADR-0015; carried over) ---------------------


def test_bounded_reward_is_monotonic_and_bounded():
    """_bound_reward: 'none' is identity; 'tanh' maps into (−1, 1) monotonically (v2)."""
    mod_none = _module(reward_bound="none")
    r = torch.linspace(-30.0, 30.0, 13)
    assert torch.equal(mod_none._bound_reward(r), r), "'none' must be the identity"

    mod_tanh = _module(reward_bound="tanh", reward_temp=8.0)
    b = mod_tanh._bound_reward(r)
    assert (b.abs() < 1.0).all(), "tanh bound must map into the open interval (−1, 1)"
    assert (b.diff() >= -1e-7).all(), "tanh bound must be monotonic"
    assert mod_tanh._bound_reward(torch.tensor([3370.0])).item() > 0.999999


def test_kl_is_zero_at_init_and_grows_with_drift():
    """The KL anchor is ~0 at init (policy==reference) and >0 once the policy drifts.

    The equal-variance per-transition KL ``0.5·‖μ_θ − μ_ref‖²/σ²`` (the bridge σ is
    θ-independent ⇒ the two transitions share variance) must (a) read ~0 while the
    trainable policy still equals its frozen reference deepcopy, (b) turn positive once
    the policy weights move, (c) flow gradient to the policy ONLY, and (d) keep the
    reference off the checkpoint/optimizer.
    """
    import copy as _copy

    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule

    torch.manual_seed(0)
    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    reference = _copy.deepcopy(policy)
    mod = PairedGRPOModule(
        policy, _reward_model(), FlowMatchBridgeGRPOScheduler(eta=0.5),
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3,
        reference_policy=reference, kl_coef=0.1,
    )
    torch.manual_seed(1)
    x_src = torch.randn(2, *_LAT)
    buf = singular_branch_rollout_paired(
        mod.unet, mod.scheduler, mod.reward_model, x_src, [1.0, 1.0, 1.0], 1, 2,
        G=2, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    step = buf[0]
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    src_labels = torch.tensor([1, 1], dtype=torch.long)
    tgt_labels = torch.tensor([2, 2], dtype=torch.long)

    # (a) at init μ_θ == μ_ref ⇒ KL ≈ 0.
    new_lp, mean_new, std_new = mod._new_log_prob(step, spacing_t, src_labels, tgt_labels, x_src)
    kl0 = mod._transition_kl(step, mean_new, std_new, spacing_t, src_labels, tgt_labels, x_src)
    assert kl0 is not None and kl0.shape == (2,)
    assert kl0.abs().max().item() < 1e-6, f"KL must be ~0 at init (policy==reference), got {kl0}"

    # (b) drift the policy ⇒ μ_θ diverges from μ_ref ⇒ KL > 0 + grad reaches policy only.
    with torch.no_grad():
        for p in mod.unet.parameters():
            p.add_(0.5 * torch.randn_like(p))
    _, mean_new2, std_new2 = mod._new_log_prob(step, spacing_t, src_labels, tgt_labels, x_src)
    kl = mod._transition_kl(step, mean_new2, std_new2, spacing_t, src_labels, tgt_labels, x_src)
    assert kl is not None
    (0.1 * kl.mean()).backward()
    assert kl.mean().item() > 1e-8, "KL must grow after the policy drifts off the reference"
    # (c) grad flows to the policy UNet, NEVER to the frozen reference.
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in mod.unet.parameters())
    assert all(p.grad is None for p in mod.reference_unet.parameters()), "reference must stay frozen"
    # (d) the reference is unregistered: off the checkpoint + the optimizer.
    assert "reference_unet" not in mod.state_dict()
    opt_ids = {id(p) for p in mod.configure_optimizers()["optimizer"].param_groups[0]["params"]}
    assert opt_ids.isdisjoint({id(p) for p in mod.reference_unet.parameters()})


def test_kl_coef_zero_yields_no_kl_term():
    """kl_coef=0 (the backward-compat default) ⇒ _transition_kl returns None (no KL)."""
    import copy as _copy

    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule

    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    mod = PairedGRPOModule(
        policy, _reward_model(), FlowMatchBridgeGRPOScheduler(eta=0.5),
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3,
        reference_policy=_copy.deepcopy(policy), kl_coef=0.0,
    )
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    buf = singular_branch_rollout_paired(
        mod.unet, mod.scheduler, mod.reward_model, x_src, [1.0, 1.0, 1.0], 1, 2,
        G=2, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    step = buf[0]
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    src_labels = torch.tensor([1, 1], dtype=torch.long)
    tgt_labels = torch.tensor([2, 2], dtype=torch.long)
    new_lp, mean_new, std_new = mod._new_log_prob(step, spacing_t, src_labels, tgt_labels, x_src)
    assert mod._transition_kl(step, mean_new, std_new, spacing_t, src_labels, tgt_labels, x_src) is None, (
        "kl_coef=0 must short-circuit the KL term"
    )


def test_module_inner_loop_and_reference_thread_source_concat():
    """The 2 module eval sites (``_new_log_prob`` + the ``_transition_kl`` reference)
    also receive cat([z, x_src]) (2·C channels) + src/tgt labels — closing the 5-site
    coverage gap (the rollout test covers the other 3 sites).

    A future refactor that drops x_src from either call would silently feed the UNet a
    4-channel input where it expects 8 → a crash today (in_channels=2·C_latent is
    structurally enforced), but this test pins the contract explicitly so the intent
    survives a future in_channels change."""
    import copy as _copy

    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule

    torch.manual_seed(0)
    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    mod = PairedGRPOModule(
        policy, _reward_model(), FlowMatchBridgeGRPOScheduler(eta=0.5),
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3,
        reference_policy=_copy.deepcopy(policy), kl_coef=0.1,
    )
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    buf = singular_branch_rollout_paired(
        mod.unet, mod.scheduler, mod.reward_model, x_src, [1.0, 1.0, 1.0], 1, 2,
        G=2, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    step = buf[0]
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    src_labels = torch.tensor([1, 1], dtype=torch.long)
    tgt_labels = torch.tensor([2, 2], dtype=torch.long)

    # Spy on both the policy UNet (site 4: _new_log_prob) + the reference UNet
    # (site 5: _transition_kl) — wrap forward to record the concat + labels.
    class _SpyUnet(nn.Module):
        def __init__(self, base, calls):
            super().__init__()
            self.base = base
            self.calls = calls

        def forward(self, sample, timestep, spacing, class_labels_src=None,
                    class_labels_tgt=None, **kw):
            self.calls.append((int(sample.shape[1]), class_labels_src is not None,
                               class_labels_tgt is not None))
            return self.base(sample=sample, timestep=timestep, spacing=spacing,
                             class_labels_src=class_labels_src, class_labels_tgt=class_labels_tgt)

    policy_calls: list[tuple[int, bool, bool]] = []
    ref_calls: list[tuple[int, bool, bool]] = []
    mod.unet = _SpyUnet(mod.unet, policy_calls)
    object.__setattr__(mod, "reference_unet", _SpyUnet(mod.reference_unet, ref_calls))

    new_lp, mean_new, std_new = mod._new_log_prob(step, spacing_t, src_labels, tgt_labels, x_src)
    assert new_lp.shape == (2, mod.G)
    kl = mod._transition_kl(step, mean_new, std_new, spacing_t, src_labels, tgt_labels, x_src)
    assert kl is not None  # kl_coef=0.1 + reference present ⇒ the KL term is active

    two_c = 2 * _LAT[0]
    assert policy_calls, "_new_log_prob must eval the policy UNet"
    assert ref_calls, "_transition_kl must eval the reference UNet"
    for ch, has_src, has_tgt in policy_calls + ref_calls:
        assert ch == two_c, f"module eval must see cat([z, x_src]) ({two_c} ch), got {ch}"
        assert has_src and has_tgt, "module eval must carry src + tgt contrast labels"


# -- multi-step PPO inner loop (the clip must bind) ------------------------------


def test_multi_step_inner_loop_ratio_drifts_off_one_so_clip_binds(tmp_path):
    """From the 2nd inner step on the ratio drifts off 1 → the clip binds (ADR-0012).

    The multi-step inner loop is load-bearing: at step 0 the policy is unchanged since
    rollout, so ``new_log_prob`` == ``old`` ⇒ ``ratio == 1`` (no clip). After the first
    ``opt.step()`` the weights moved, so step ≥1 recomputes a ratio off 1 — past the
    tight ``clip_range`` the clipped branch freezes the gradient. The large LR makes one
    Adam step move the ratio past ``clip_range`` demonstratively.
    """
    import manifold.modules.paired_grpo as pgmod

    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule
    from manifold.training.paired_grpo_cli import PairedGRPOInputs, run_paired_grpo_training

    torch.manual_seed(0)
    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    inputs = PairedGRPOInputs(
        policy=policy, reward_model=_reward_model(),
        scheduler=FlowMatchBridgeGRPOScheduler(eta=0.5),
        train_ds=_ToyPairedDS(), val_ds=_ToyPairedDS(),
    )
    module = PairedGRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0, 1), num_steps=3, lr=1e-2,  # large LR → visible drift
    )
    captured: list[tuple[float, bool]] = []
    real_loss = pgmod.clipped_surrogate_loss

    def spy(new_lp, old_lp, advantage, clip_range):
        with torch.no_grad():
            ratio = torch.exp(new_lp - old_lp)
            eps = float(clip_range)
            clipped_ratio = ratio.clamp(1.0 - eps, 1.0 + eps)
            binds = (clipped_ratio * advantage) < (ratio * advantage)
            captured.append((float((ratio - 1.0).abs().mean()), bool(binds.any())))
        return real_loss(new_lp, old_lp, advantage, clip_range)

    pgmod.clipped_surrogate_loss = spy
    try:
        run_paired_grpo_training(
            module=module, inputs=inputs, model_dir=str(tmp_path),
            max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
        )
    finally:
        pgmod.clipped_surrogate_loss = real_loss

    assert len(captured) >= 2, "expected ≥2 inner steps (eta_step_list=(0,1))"
    assert captured[0][0] < 1e-9, f"step-0 ratio must be 1 (unchanged weights), got {captured[0][0]}"
    assert not captured[0][1], "step-0 clip must be idle (ratio == 1)"
    assert captured[1][1], (
        f"step-1 clip must bind (ratio drifted past clip_range={module.clip_range} after opt.step); "
        f"drift={captured[1][0]}"
    )
    assert captured[1][0] > captured[0][0], "ratio drift must increase after the first opt.step"


# -- CLI smoke (the end-to-end seam) --------------------------------------------


class _ToyPairedDS(Dataset):
    """A tiny source-latent dataset (train/val): emits {src_latent, src/tgt_label, spacing}.

    G2RPO is pure-RL — the bridge starts from x_src (data); the target volume is unused
    at train. Fixed-seed per-item source latents so the smoke is reproducible.
    """

    def __init__(self, n: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.src = torch.randn(n, *_LAT, generator=g)

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, i):
        return {
            "src_latent": self.src[i],
            "src_label": torch.tensor(1, dtype=torch.long),
            "tgt_label": torch.tensor(2, dtype=torch.long),
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
        }


def _inputs():
    """The injection-seam bundle: tiny trainable paired policy + tiny frozen reward.

    Supplies a frozen deepcopy reference_policy so the committed-recipe CLI smoke
    (which reads kl_coef=0.1 from config_paired_grpo.yaml) runs with the KL anchor
    actually on — mirroring the #104 real path. """
    import copy as _copy

    from manifold import UNet3DConditionModel
    from manifold.training.paired_grpo_cli import PairedGRPOInputs

    torch.manual_seed(0)
    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    return PairedGRPOInputs(
        policy=policy, reward_model=_reward_model(),
        scheduler=FlowMatchBridgeGRPOScheduler(eta=0.5),
        train_ds=_ToyPairedDS(), val_ds=_ToyPairedDS(),
        reference_policy=_copy.deepcopy(policy),
    )


def _run(tmp_path, **kw):
    from manifold.modules import PairedGRPOModule
    from manifold.training.paired_grpo_cli import run_paired_grpo_training

    inputs = _inputs()
    module = PairedGRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3,
    )
    return run_paired_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2, **kw,
    )


def test_run_paired_grpo_training_writes_ckpt_and_logs_mean_reward(tmp_path):
    """run_paired_grpo_training: fit + validate → checkpoint + finite val/mean_reward."""
    trainer, ckpt = _run(tmp_path)
    metrics = trainer.callback_metrics
    assert "val/mean_reward" in metrics
    assert torch.isfinite(metrics["val/mean_reward"])
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()
    # on_fit_start moved the unregistered frozen reward onto the module device.
    assert next(trainer.model.reward_model.parameters()).device == trainer.model.device


def test_run_paired_grpo_training_multi_step_inner_loop_runs(tmp_path):
    """A 2-entry eta_step_list completes fit (the multi-step inner loop iterates)."""
    from manifold.modules import PairedGRPOModule
    from manifold.training.paired_grpo_cli import run_paired_grpo_training

    inputs = _inputs()
    module = PairedGRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0, 1), num_steps=3, lr=1e-3,
    )
    trainer, _ = run_paired_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
    )
    assert torch.isfinite(trainer.callback_metrics["val/mean_reward"])


def test_paired_grpo_module_sample_is_deployed_heun_not_the_bridge(paired_unet):
    """Module.sample is the deployed two-eval Heun — bridge η does NOT leak into generation.

    Validation / the PSNR callback must measure the deterministic distribution Paired
    JiT ships, so ``sample`` delegates to ``sample_paired_latent_flow`` (the deployed
    Heun), NOT the bridge SDE. The bridge scheduler's η only affects ``sde_step_mean``;
    the inherited ``euler_step``/``heun_correct`` are η-agnostic. So same x_src + the
    bridge(η=0.9) scheduler produces a latent bit-identical to the plain Heun scheduler.
    """
    from manifold.modules import PairedGRPOModule

    mod = PairedGRPOModule(
        paired_unet, _reward_model(), FlowMatchBridgeGRPOScheduler(eta=0.9),
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3,
    )
    torch.manual_seed(0)
    x_src = torch.randn(1, *_LAT)
    out = mod.sample(x_src, [1.0, 1.0, 1.0], 1, 2, num_inference_steps=3)
    ref = sample_paired_latent_flow(
        paired_unet, FlowMatchHeunDiscreteScheduler(), x_src, [1.0, 1.0, 1.0], 1, 2,
        num_inference_steps=3,
    )
    assert torch.equal(out, ref), "η-agnostic deployed Heun — no bridge SDE leak into generation"


def test_run_paired_grpo_training_attaches_no_ema_callback(tmp_path):
    """G2RPO runs WITHOUT DoubleEMACallback (ADR-0012; inverts ADR-0021 for this stage).

    The double-EMA's supervised-decay shadows are useless under RL and hold ~7 GB the
    rollout needs; deployment / validation evaluate the raw policy. The checkpoint is the
    only callback attached by the tracer (#105 adds the PSNR callback)."""
    from manifold.modules import PairedGRPOModule
    from manifold.training import DoubleEMACallback
    from manifold.training.paired_grpo_cli import run_paired_grpo_training

    inputs = _inputs()
    module = PairedGRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3,
    )
    trainer, _ = run_paired_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
    )
    assert not any(isinstance(c, DoubleEMACallback) for c in trainer.callbacks), (
        "G2RPO must NOT attach DoubleEMACallback — the shadows are useless under RL."
    )


def test_run_paired_grpo_measurement_reports_it_per_s(tmp_path):
    """run_paired_grpo_measurement times a fit + reports it/s (peak GPU is 0 off-CUDA)."""
    from manifold.modules import PairedGRPOModule
    from manifold.training.paired_grpo_cli import run_paired_grpo_measurement

    inputs = _inputs()
    module = PairedGRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0,), num_steps=3, lr=1e-3,
    )
    it_per_s, peak, elapsed = run_paired_grpo_measurement(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        devices=1, accelerator="cpu", batch_size=2,
    )
    assert it_per_s > 0, f"it/s must be a positive real, got {it_per_s}"
    assert elapsed > 0
    assert peak == 0  # off-CUDA: torch.cuda.max_memory_allocated() is 0


_TINY_NETWORK_YAML = "spatial_dims: 3\nlatent_channels: 4\n"


def _write_tiny_configs(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(_TINY_NETWORK_YAML)
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
        + "val_subset_size: 4\n"
    )
    train = tmp_path / "train.yaml"
    train.write_text(
        "paired_grpo_train: {batch_size: 2, lr: 1.0e-3, n_epochs: 1, num_steps: 3, "
        "G: 2, eta: 0.5, clip_range: 1.0e-4, adv_clip_max: 5.0, "
        "eta_step_list: [0], kl_coef: 0.0, reward_bound: none, reward_temp: 8.0, "
        "latent_shape: [4, 8, 8, 8]}\n"
    )
    return str(env), str(train), str(net)


def test_main_runs_end_to_end_with_fake_data(tmp_path):
    """main(): argparse -> compose -> build -> fit -> ckpt (the fake-data injection seam)."""
    from manifold.training.paired_grpo_cli import main as pg_main

    env, train, net = _write_tiny_configs(tmp_path)
    rc = pg_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=lambda cfg, device: _inputs(),
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_main_native_reward_default_none_and_validated(tmp_path):
    """--native-dir/--reward-path default None and are required without a data_provider."""
    from manifold.training.paired_grpo_cli import main as pg_main

    env, train, net = _write_tiny_configs(tmp_path)
    with pytest.raises(ValueError, match="native-dir"):
        pg_main(["-e", env, "-c", train, "-t", net, "-g", "1"])
    rc = pg_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=lambda cfg, device: _inputs(),
    )
    assert rc == 0


def test_main_uses_committed_default_paired_grpo_recipe(tmp_path):
    """main() with NO -c (argparse default) resolves the committed config_paired_grpo.yaml."""
    from manifold.training.paired_grpo_cli import main as pg_main

    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
        + "val_subset_size: 4\n"
    )
    net = "configs/network/config_network.yaml"
    rc = pg_main(
        ["-e", str(env), "-t", net, "-g", "1", "--max-epochs", "1",
         "paired_grpo_train.G=2", "paired_grpo_train.num_steps=3",
         "paired_grpo_train.eta_step_list=[0]",
         "paired_grpo_train.latent_shape=[4,8,8,8]"],
        data_provider=lambda cfg, device: _inputs(),
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


# -- full v1 budget (#104): the committed G=8 / [0..3] / num_steps=8 feasibility --


def test_full_v1_budget_rollout_is_finite_across_trajectory(paired_unet):
    """G=8, eta_step_list=[0..3], num_steps=8: every buffer entry finite.

    The committed v1 budget walks the first half of the 8-step grid. The acceptance: no
    NaN/Inf anywhere in the trajectory — every z_kplus1, log-prob, advantage, and reward
    is finite (the §7 std is finite everywhere on [0..3]; no terminal collapse)."""
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    buf = singular_branch_rollout_paired(
        paired_unet, FlowMatchBridgeGRPOScheduler(eta=0.7), _reward_model(), x_src,
        [1.0, 1.0, 1.0], 1, 2, G=8, eta_step_list=[0, 1, 2, 3], num_steps=8,
    )
    assert len(buf) == 4
    for e in buf:
        for key in ("z_kplus1", "old_log_prob", "advantage", "rewards"):
            assert torch.isfinite(e[key]).all(), f"{key} has NaN/Inf in the full-budget rollout"
        assert e["t_k"] < e["t_next"]


def test_full_v1_budget_runs_end_to_end(tmp_path):
    """The committed v1 budget (G=8, [0..3], num_steps=8) completes a fit (#104)."""
    from manifold import UNet3DConditionModel
    from manifold.modules import PairedGRPOModule
    from manifold.training.paired_grpo_cli import PairedGRPOInputs, run_paired_grpo_training

    torch.manual_seed(0)
    policy = UNet3DConditionModel(
        in_channels=8, out_channels=4, num_class_embeds=4, include_spacing_input=True
    )
    inputs = PairedGRPOInputs(
        policy=policy, reward_model=_reward_model(),
        scheduler=FlowMatchBridgeGRPOScheduler(eta=0.7),
        train_ds=_ToyPairedDS(), val_ds=_ToyPairedDS(),
    )
    module = PairedGRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=8, eta_step_list=(0, 1, 2, 3), num_steps=8, lr=1e-3,
    )
    trainer, ckpt = run_paired_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
    )
    assert torch.isfinite(trainer.callback_metrics["val/mean_reward"])
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()
