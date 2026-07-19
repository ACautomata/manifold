"""GRPO policy-learning tests: singular-branch rollout + GRPOModule + CLI (#56).

External-behavior seams (PRD #55 "Testing Decisions", mirroring the reward stack):
the singular-branch rollout degenerates to the deployed ``sample_latent_flow`` at
η=0 (the anchor uses the same Heun primitive — no fork); the buffer shapes carry
the ``(B, G)`` group; the rollout is fully ``no_grad`` (the grad eval lives in the
inner loop); the clipped-surrogate loss is finite and its gradient pushes toward
higher-advantage transitions; backward touches the policy UNet ONLY (the frozen
``RewardModel`` is unregistered — off the checkpoint/optimizer); and a GRPO run
completes end-to-end on toy injected policy + frozen reward via the CLI smoke,
writing a checkpoint and logging ``val/mean_reward``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from manifold import (
    ControlNet3DConditionModel,
    FlowMatchGRPOScheduler,
    FlowMatchHeunDiscreteScheduler,
    RewardModel,
    UNet3DConditionModel,
)
from manifold.modules import sample_latent_flow, singular_branch_rollout
from manifold.modules.grpo import (
    GRPOModule,
    clipped_surrogate_loss,
    gaussian_log_prob,
    group_advantage,
)

#: A tiny latent shape + RewardModel config that survives the PatchGAN strided
#: convs on CPU (mirrors tests/test_reward.py).
_LAT = (4, 8, 8, 8)
_RM_KW = dict(spatial_dims=3, in_channels=4, channels=8, num_layers_d=1)


def _reward_model() -> RewardModel:
    torch.manual_seed(0)
    return RewardModel(**_RM_KW)


# -- group_advantage + gaussian_log_prob (the GRPO objective pieces) -----------


def test_group_advantage_is_group_normalized_over_G():
    """A = (R − mean R)/(std R + ε) over the G siblings; zero mean per group."""
    R = torch.tensor([[1.0, 2.0, 3.0, 4.0], [10.0, -2.0, 0.0, 4.0]])  # (B=2, G=4)
    A = group_advantage(R, adv_clip_max=5.0)
    mean_per_group = R.mean(dim=1, keepdim=True)
    std_per_group = R.std(dim=1, keepdim=True)
    expected = (R - mean_per_group) / (std_per_group + 1e-8)
    assert torch.allclose(A, expected, atol=1e-6)
    # Group-normalized ⇒ zero mean over G (before the clip; these are unclipped here).
    assert torch.allclose(A.mean(dim=1), torch.zeros(2), atol=1e-6)


def test_group_advantage_clips_to_adv_clip_max():
    """An outlier-driven advantage is clipped to ±adv_clip_max (the stabilizer)."""
    R = torch.tensor([[0.0, 0.0, 0.0, 1e6]])  # one huge outlier ⇒ raw adv huge
    A = group_advantage(R, adv_clip_max=3.0)
    assert A.max().item() <= 3.0 + 1e-5
    assert A.min().item() >= -3.0 - 1e-5


def test_gaussian_log_prob_matches_manual_mean_reduced():
    """log N(z; mean, std²) mean-reduced over non-batch dims → (B, G)."""
    torch.manual_seed(0)
    B, G = 2, 3
    mean = torch.randn(B, *_LAT)
    std = 0.5
    z = mean.unsqueeze(1) + std * torch.randn(B, G, *_LAT)  # (B, G, ...)
    lp = gaussian_log_prob(z, mean, std)
    assert lp.shape == (B, G)
    # Manual: mean-reduce the per-element log-density over (C, D, H, W).
    import math

    diff = z - mean.unsqueeze(1)
    elem = -0.5 * (diff.pow(2) / std ** 2 + math.log(2 * math.pi * std ** 2))
    expected = elem.flatten(start_dim=2).mean(dim=2)
    assert torch.allclose(lp, expected, atol=1e-6)


# -- singular_branch_rollout -------------------------------------------------


def _soft_policy() -> nn.Module:
    """A NON-identity fake policy UNet (x0 = 0.5·z) so the rollout provably moves.

    Carries a dummy param so it mimics a real module (the rollout reads the device
    off ``next(unet.parameters())``). Frozen + eval by the rollout (no_grad).
    """

    class _Soft(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.ones(3))

        def forward(self, sample, timestep, spacing, class_labels=None, **kw):
            return 0.5 * sample

    return _Soft()


def test_rollout_anchor_eta_zero_matches_sample_latent_flow(unet):
    """η=0, last-step branch ⇒ the terminal z_K == sample_latent_flow (no fork).

    The anchor runs the same two-eval Heun as the deployed sampler; with η=0 the
    single SDE step reduces to the Euler advance (== the Heun final-step Euler at
    t_next=1), and the suffix is empty for the last step. So the terminal latent is
    bit-identical to ``sample_latent_flow`` over the same grid — the anchor-parity
    guard (#56 acceptance: the anchor uses the existing Heun primitive, not a fork).
    η=0 ⇒ std·ξ = 0, so all G siblings are identical; sibling 0 is the reference.
    """
    torch.manual_seed(0)
    noise = torch.randn(2, *_LAT)
    n = 4
    grpo = FlowMatchGRPOScheduler(eta=0.0)
    full = FlowMatchHeunDiscreteScheduler()
    buf = singular_branch_rollout(
        unet, grpo, _reward_model(), noise, [1.0, 1.0, 1.0], 1,
        G=2, eta_step_list=[n - 1], num_steps=n,
    )
    # Last perturbed step ⇒ empty suffix ⇒ z_{k+1} IS the terminal z_K (B, G, ...).
    # η=0 ⇒ all siblings identical; sibling 0 == the deployed-sampler terminal.
    z_K = buf[0]["z_kplus1"][:, 0]
    ref = sample_latent_flow(unet, full, noise, [1.0, 1.0, 1.0], 1, num_inference_steps=n)
    assert torch.equal(z_K, ref)


def test_rollout_buffer_shapes_carry_the_group(unet):
    """Buffer per step: z_k (B,...), z_{k+1} (B,G,...), old_log_prob / advantage (B,G)."""
    torch.manual_seed(0)
    B, G, n = 2, 3, 4
    noise = torch.randn(B, *_LAT)
    buf = singular_branch_rollout(
        unet, FlowMatchGRPOScheduler(eta=0.7), _reward_model(), noise,
        [1.0, 1.0, 1.0], 1, G=G, eta_step_list=[0, 1], num_steps=n,
    )
    assert len(buf) == 2  # one entry per perturbed step
    entry = buf[0]
    assert entry["z_k"].shape == (B, *_LAT)
    assert entry["z_kplus1"].shape == (B, G, *_LAT)
    assert entry["old_log_prob"].shape == (B, G)
    assert entry["advantage"].shape == (B, G)
    assert entry["rewards"].shape == (B, G)
    assert entry["t_k"] < entry["t_next"]


def test_rollout_outputs_are_detached_no_grad(unet):
    """The rollout is fully no_grad — buffer tensors carry no autograd graph.

    The grad eval lives in the inner loop (recompute new_log_prob under grad); the
    anchor + suffix + reward must NOT retain a graph (the 3D-memory invariant,
    ADR-0011/0012)."""
    torch.manual_seed(0)
    noise = torch.randn(2, *_LAT)
    buf = singular_branch_rollout(
        unet, FlowMatchGRPOScheduler(eta=0.7), _reward_model(), noise,
        [1.0, 1.0, 1.0], 1, G=3, eta_step_list=[0], num_steps=4,
    )
    e = buf[0]
    for key in ("z_k", "z_kplus1", "old_log_prob", "advantage", "rewards"):
        assert not e[key].requires_grad, f"{key} must be detached (no_grad rollout)"
        assert e[key].grad_fn is None


def test_rollout_siblings_diverge_with_noise(unet):
    """With η>0 the G siblings get distinct terminal rewards (the SDE draw explores).

    With η=0 all siblings are identical (std=0 → same z_{k+1}); with η>0 the ξ draw
    spreads them — the exploration singular branching provides (#56 acceptance)."""
    torch.manual_seed(0)
    noise = torch.randn(1, *_LAT)
    buf = singular_branch_rollout(
        unet, FlowMatchGRPOScheduler(eta=0.9), _reward_model(), noise,
        [1.0, 1.0, 1.0], 1, G=4, eta_step_list=[0], num_steps=3,
    )
    rewards = buf[0]["rewards"]  # (1, G=4)
    assert rewards.unique().numel() > 1  # siblings are NOT all identical


def test_rollout_runs_for_each_perturbed_step(unet):
    """A 2-element eta_step_list yields a 2-entry buffer (one rollout per step)."""
    torch.manual_seed(0)
    noise = torch.randn(1, *_LAT)
    buf = singular_branch_rollout(
        unet, FlowMatchGRPOScheduler(eta=0.5), _reward_model(), noise,
        [1.0, 1.0, 1.0], 1, G=2, eta_step_list=[0, 2], num_steps=4,
    )
    assert [e["t_k"] for e in buf] == [0.0, 0.5]  # nodes[0]=0, nodes[2]=0.5 on n=4


# -- clipped-surrogate loss (the PPO objective piece) ------------------------


def test_clipped_surrogate_loss_gradient_direction():
    """L = -mean(min(r·A, clip(r)·A)); descent raises new_log_prob for A>0, lowers for A<0.

    The load-bearing PPO property: near r=1 (no clip bound), ∂L/∂new = -r·A — so
    descent increases the transition log-prob for positive-advantage siblings and
    decreases it for negative-advantage ones (pushing the policy toward the
    higher-reward transition), #56 acceptance.
    """
    old = torch.zeros(3)
    new_pos = torch.zeros(3, requires_grad=True)
    clipped_surrogate_loss(new_pos, old, torch.tensor([1.0, 1.0, 1.0]), clip_range=1e-4).backward()
    assert (new_pos.grad < 0).all()  # A > 0 ⇒ descent raises new_log_prob

    new_neg = torch.zeros(3, requires_grad=True)
    clipped_surrogate_loss(new_neg, old, torch.tensor([-1.0, -1.0, -1.0]), clip_range=1e-4).backward()
    assert (new_neg.grad > 0).all()  # A < 0 ⇒ descent lowers new_log_prob


def test_clipped_surrogate_loss_clips_ratio_beyond_bound():
    """When r exceeds 1+ε (A>0) the surrogate uses clip(r)·A → ∂L/∂new = 0 (the trust region).

    A ratio driven far off 1 is frozen out by the clip — the PPO trust region the
    multi-step inner loop keeps load-bearing (a single aggregated step would make
    r=1 always and this clip a no-op), #56 acceptance.
    """
    eps = 1e-4
    old = torch.zeros(1)
    new = torch.tensor([5.0], requires_grad=True)  # r = e^5 ≈ 148 >> 1+ε
    loss = clipped_surrogate_loss(new, old, torch.tensor([1.0]), clip_range=eps)
    assert torch.isfinite(loss)
    loss.backward()
    assert new.grad.abs().item() < 1e-8  # clipped branch ⇒ no gradient flows


def test_multi_step_inner_loop_ratio_drifts_off_one_so_clip_binds(tmp_path):
    """From the 2nd inner step on the ratio drifts off 1 → the clip binds (#57).

    The multi-step inner loop is load-bearing, not cosmetic: at step 0 the policy
    is unchanged since rollout, so the re-evaluated ``new_log_prob`` == ``old``,
    giving ``ratio == 1`` (no clip). After the first ``opt.step()`` the weights
    moved, so step ≥1 recomputes a ratio off 1 — past the tight ``clip_range`` the
    clipped branch freezes the gradient (a real PPO trust region). A single
    aggregated step would leave ``ratio == 1`` always and the clip a permanent
    no-op (→ REINFORCE). The large LR makes one Adam step move the ratio past
    ``clip_range`` demonstratively; the assertion is the mechanism, not the
    production LR (1e-6). Verifies #57 acceptance: the clip binds, verified by a
    test (not just asserted in prose).
    """
    import manifold.modules.grpo as grpo_mod

    from manifold import UNet3DConditionModel
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import GRPOInputs, run_grpo_training

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    inputs = GRPOInputs(
        policy=policy, reward_model=_reward_model(), scheduler=FlowMatchGRPOScheduler(eta=0.5),
        train_ds=_ToyCondDS(), val_ds=_ToyCondDS(), latent_shape=_LAT,
    )
    module = GRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0, 1), num_steps=3, latent_shape=_LAT, lr=1e-2,  # large LR → visible drift
    )
    #: Per inner step: (mean|r−1|, clip_active?). Captured by spying on the loss.
    captured: list[tuple[float, bool]] = []
    real_loss = grpo_mod.clipped_surrogate_loss

    def spy(new_lp, old_lp, advantage, clip_range):
        with torch.no_grad():
            ratio = torch.exp(new_lp - old_lp)
            eps = float(clip_range)
            clipped_ratio = ratio.clamp(1.0 - eps, 1.0 + eps)
            # PPO binds where the clipped term is the min — i.e. the gradient is
            # frozen — which happens ONLY on the side selected by the advantage sign
            # (ratio>1+eps for A>0; ratio<1-eps for A<0). Checking ratio bounds alone
            # would mark the clip "active" even when every element still uses the
            # unclipped term, so the test could pass without the trust region ever
            # engaging (codex #61 P2). This is the exact binding condition.
            binds = (clipped_ratio * advantage) < (ratio * advantage)
            captured.append((float((ratio - 1.0).abs().mean()), bool(binds.any())))
        return real_loss(new_lp, old_lp, advantage, clip_range)

    # Patch the module global training_step looks up (NOT the test's local import,
    # which bound a separate reference). Safe per-test: pytest workers are separate
    # processes, and the finally restores it even on exception within a worker.
    grpo_mod.clipped_surrogate_loss = spy
    try:
        run_grpo_training(
            module=module, inputs=inputs, model_dir=str(tmp_path),
            max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
        )
    finally:
        grpo_mod.clipped_surrogate_loss = real_loss

    # Step 0: weights unchanged since rollout ⇒ ratio == 1 ⇒ no drift, clip idle.
    assert len(captured) >= 2, "expected ≥2 inner steps (eta_step_list=(0,1))"
    assert captured[0][0] < 1e-9, f"step-0 ratio must be 1 (unchanged weights), got drift {captured[0][0]}"
    assert not captured[0][1], "step-0 clip must be idle (ratio == 1)"
    # Step 1: opt.step() moved the policy ⇒ ratio off 1, past clip_range ⇒ clip binds.
    assert captured[1][1], (
        f"step-1 clip must bind (ratio drifted past clip_range={module.clip_range} after opt.step); "
        f"drift={captured[1][0]}"
    )
    assert captured[1][0] > captured[0][0], "ratio drift must increase after the first opt.step"


# -- GRPOModule (the policy learner) -----------------------------------------


def _module(latent_shape=_LAT, **kw):
    """A tiny GRPOModule: a fresh trainable policy UNet + tiny frozen RewardModel."""
    from manifold import UNet3DConditionModel
    from manifold.modules import GRPOModule

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    return GRPOModule(
        policy, _reward_model(), FlowMatchGRPOScheduler(eta=0.5),
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=latent_shape, lr=1e-3, **kw
    )


def test_module_backward_updates_unet_only():
    """backward populates UNet (policy) grads; the frozen RewardModel is unregistered.

    The Module HOLDS the frozen reward (unregistered via object.__setattr__) — so it
    is absent from parameters()/state_dict()/optimizer, and backward only touches the
    policy UNet (#56 acceptance: the exclusion invariant, mirroring RewardModule)."""
    mod = _module()
    torch.manual_seed(0)
    noise = torch.randn(2, *_LAT)
    buf = singular_branch_rollout(
        mod.unet, mod.scheduler, mod.reward_model, noise, [1.0, 1.0, 1.0], 1,
        G=2, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    step = buf[0]
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    class_labels = torch.tensor([1, 1], dtype=torch.long)
    # Recompute new_log_prob under grad (the inner-loop grad eval) + loss + backward.
    new_lp, _mean_new, _std_new = mod._new_log_prob(step, spacing_t, class_labels)
    loss = clipped_surrogate_loss(new_lp, step["old_log_prob"], step["advantage"], mod.clip_range)
    loss.backward()
    # Policy UNet: every param got a finite grad.
    unet_params = list(mod.unet.parameters())
    assert unet_params and all(p.grad is not None and torch.isfinite(p.grad).all() for p in unet_params)
    # Frozen reward: held but UNREGISTERED — grads None, off the optimizer/checkpoint.
    assert all(p.grad is None for p in mod.reward_model.parameters())
    assert "reward_model" not in mod.state_dict()
    opt = mod.configure_optimizers()["optimizer"]
    opt_ids = {id(p) for p in opt.param_groups[0]["params"]}
    assert opt_ids == {id(p) for p in unet_params}
    assert opt_ids.isdisjoint({id(p) for p in mod.reward_model.parameters()})


def test_module_advantage_group_normalized_in_buffer():
    """The buffer's advantage is the group-normalized (R−mean)/std over the G siblings."""
    mod = _module()
    torch.manual_seed(0)
    noise = torch.randn(1, *_LAT)
    buf = singular_branch_rollout(
        mod.unet, mod.scheduler, mod.reward_model, noise, [1.0, 1.0, 1.0], 1,
        G=4, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    A = buf[0]["advantage"]  # (1, G=4)
    # Group-normalized ⇒ zero mean over G (before the clip; this case is unclipped).
    assert torch.allclose(A.mean(dim=1), torch.zeros(1), atol=1e-6)


def test_module_rejects_degenerate_group_size():
    """G < 2 raises: torch.std (Bessel) needs ≥2 siblings, else NaN advantage → NaN grads."""
    from manifold import UNet3DConditionModel
    from manifold.modules import GRPOModule

    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    with pytest.raises(ValueError, match="G must be >= 2"):
        GRPOModule(policy, _reward_model(), FlowMatchGRPOScheduler(), G=1, latent_shape=_LAT)


# -- v2: bounded reward + KL anchor (ADR-0015; the v1 reward-hacking fix) -----


def test_bounded_reward_is_monotonic_and_bounded():
    """_bound_reward: 'none' is identity; 'tanh' maps into (−1, 1) monotonically (v2).

    The bound caps the unbounded PatchGAN logit so the policy cannot profit from OOD
    latents (ADR-0015). It must be MONOTONIC so distinct sibling rewards stay distinct
    (a hard clamp would collapse them and break the group signal), and bounded so an OOD
    extreme (the v1 raw 3370) saturates instead of dominating the advantage.
    """
    mod_none = _module(reward_bound="none")
    r = torch.linspace(-30.0, 30.0, 13)  # spans the real-data range [−21, +26] and beyond
    assert torch.equal(mod_none._bound_reward(r), r), "'none' must be the identity"

    mod_tanh = _module(reward_bound="tanh", reward_temp=8.0)
    b = mod_tanh._bound_reward(r)
    assert (b.abs() < 1.0).all(), "tanh bound must map into the open interval (−1, 1)"
    # Monotonic non-decreasing ⇒ a sorted input yields a sorted output (siblings distinct).
    assert (b.diff() >= -1e-7).all(), "tanh bound must be monotonic"
    # The v1 hacking magnitude saturates near +1 instead of dominating (the cap).
    assert mod_tanh._bound_reward(torch.tensor([3370.0])).item() > 0.999999


def test_kl_is_zero_at_init_and_grows_with_drift():
    """The KL anchor is ~0 at init (policy==reference) and >0 once the policy drifts (v2).

    The equal-variance per-transition KL ``0.5·‖μ_θ − μ_ref‖²/σ²`` (ADR-0015; σ_t depends
    only on t ⇒ the two transitions share variance) must (a) read ~0 while the trainable
    policy still equals its frozen reference deepcopy, (b) turn positive once the policy
    weights move, (c) flow gradient to the policy ONLY (the reference is frozen +
    unregistered, mirroring the reward invariant), and (d) keep the reference off the
    checkpoint/optimizer.
    """
    import copy as _copy

    from manifold import UNet3DConditionModel
    from manifold.modules import GRPOModule

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    reference = _copy.deepcopy(policy)
    mod = GRPOModule(
        policy, _reward_model(), FlowMatchGRPOScheduler(eta=0.5),
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=_LAT, lr=1e-3,
        reference_policy=reference, kl_coef=0.1,
    )
    torch.manual_seed(1)
    noise = torch.randn(2, *_LAT)
    buf = singular_branch_rollout(
        mod.unet, mod.scheduler, mod.reward_model, noise, [1.0, 1.0, 1.0], 1,
        G=2, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    step = buf[0]
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    class_labels = torch.tensor([1, 1], dtype=torch.long)

    # (a) at init μ_θ == μ_ref ⇒ KL ≈ 0.
    new_lp, mean_new, std_new = mod._new_log_prob(step, spacing_t, class_labels)
    kl0 = mod._transition_kl(step, mean_new, std_new, spacing_t, class_labels)
    assert kl0 is not None and kl0.shape == (2,)
    assert kl0.abs().max().item() < 1e-6, f"KL must be ~0 at init (policy==reference), got {kl0}"

    # (b) drift the policy ⇒ μ_θ diverges from μ_ref ⇒ KL > 0 + grad reaches the policy only.
    with torch.no_grad():
        for p in mod.unet.parameters():
            p.add_(0.5 * torch.randn_like(p))
    _, mean_new2, std_new2 = mod._new_log_prob(step, spacing_t, class_labels)
    kl = mod._transition_kl(step, mean_new2, std_new2, spacing_t, class_labels)
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
    """kl_coef=0 (the backward-compat default) ⇒ _transition_kl returns None (no KL).

    Guards that the v2 KL term is opt-in: with the default kl_coef=0.0 the inner-loop loss
    is exactly the v1 clipped surrogate (no reference forward, no KL), so all v1 tests and
    the locked-recipe default behavior are preserved.
    """
    import copy as _copy

    from manifold import UNet3DConditionModel
    from manifold.modules import GRPOModule

    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    # Reference provided BUT kl_coef=0 ⇒ the anchor is dormant.
    mod = GRPOModule(
        policy, _reward_model(), FlowMatchGRPOScheduler(eta=0.5),
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=_LAT, lr=1e-3,
        reference_policy=_copy.deepcopy(policy), kl_coef=0.0,
    )
    torch.manual_seed(0)
    noise = torch.randn(2, *_LAT)
    buf = singular_branch_rollout(
        mod.unet, mod.scheduler, mod.reward_model, noise, [1.0, 1.0, 1.0], 1,
        G=2, eta_step_list=[0], num_steps=3, adv_clip_max=mod.adv_clip_max,
    )
    step = buf[0]
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    class_labels = torch.tensor([1, 1], dtype=torch.long)
    new_lp, mean_new, std_new = mod._new_log_prob(step, spacing_t, class_labels)
    assert mod._transition_kl(step, mean_new, std_new, spacing_t, class_labels) is None, (
        "kl_coef=0 must short-circuit the KL term (backward-compat with v1)"
    )


# -- CLI smoke (the end-to-end seam) -----------------------------------------


class _ToyCondDS(Dataset):
    """A tiny conditioning dataset (train/val): emits {spacing, label} (GRPO is generative)."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"spacing": torch.tensor([1.0, 1.0, 1.0]), "label": torch.tensor(1, dtype=torch.long)}


def _inputs():
    """The injection-seam bundle: tiny trainable policy + tiny frozen reward + toy conditioning."""
    from manifold import UNet3DConditionModel
    from manifold.training.grpo_cli import GRPOInputs

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    return GRPOInputs(
        policy=policy, reward_model=_reward_model(), scheduler=FlowMatchGRPOScheduler(eta=0.5),
        train_ds=_ToyCondDS(), val_ds=_ToyCondDS(), latent_shape=_LAT,
    )


def _run(tmp_path, **kw):
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import run_grpo_training

    inputs = _inputs()
    module = GRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=inputs.latent_shape, lr=1e-3,
    )
    return run_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2, **kw,
    )


def test_run_grpo_training_writes_ckpt_and_logs_mean_reward(tmp_path):
    """run_grpo_training: fit + validate → checkpoint + finite val/mean_reward logged."""
    trainer, ckpt = _run(tmp_path)
    metrics = trainer.callback_metrics
    assert "val/mean_reward" in metrics
    assert torch.isfinite(metrics["val/mean_reward"])
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()
    # on_fit_start moved the unregistered frozen reward onto the module device.
    assert next(trainer.model.reward_model.parameters()).device == trainer.model.device


def test_run_grpo_training_multi_step_inner_loop_runs(tmp_path):
    """A 2-entry eta_step_list completes fit (the multi-step inner loop iterates, ADR-0012)."""
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import run_grpo_training

    inputs = _inputs()
    module = GRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0, 1), num_steps=3, latent_shape=inputs.latent_shape, lr=1e-3,
    )
    trainer, _ = run_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
    )
    assert torch.isfinite(trainer.callback_metrics["val/mean_reward"])


class _FakeFeatureNet(nn.Module):
    """Deterministic 2D-plane → feature: flatten + a fixed linear (no RNG).

    Mirrors tests/test_fid.py._FakeFeatureNet (not importable across test modules).
    The VAE decodes an 8×8×8 latent to a 16×16×16 image ⇒ 16×16 = 256-d planes;
    the net keeps the first 64 dims (≥64 ⇒ no pad).
    """

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 6, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.linspace(0.01, 0.06, self.proj.weight.numel()).reshape_as(self.proj.weight)
            )

    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        b = plane.shape[0]
        flat = plane.reshape(b, -1)[:, :64]
        if flat.shape[1] < 64:
            flat = torch.nn.functional.pad(flat, (0, 64 - flat.shape[1]))
        return self.proj(flat)


def test_run_grpo_training_with_fid_logs_val_fid_and_selects_on_it(tmp_path):
    """The FID triple ⇒ FIDCallback logs val/fid; ckpt monitors val/fid (#58).

    The anti-reward-hacking screen: when ``GRPOInputs`` carries ``vae`` +
    ``real_latents`` + ``feature_net``, ``run_grpo_training`` attaches
    ``FIDCallback`` which generates via the deployed Heun
    (``GRPOModule.sample``) and logs ``val/fid``; the checkpoint switches its monitor
    to ``val/fid`` (mode ``min``) — a higher-reward-but-higher-FID checkpoint is NOT
    selected. The PatchGAN ``val/mean_reward`` stays logged (validation_step's own
    generation pass — the RL progress signal). No new metric code: the existing JiT
    FIDCallback + the unbiased Fréchet primitive are reused verbatim.
    """
    from manifold import AutoencoderKL, UNet3DConditionModel
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import GRPOInputs, run_grpo_training

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    inputs = GRPOInputs(
        policy=policy, reward_model=_reward_model(), scheduler=FlowMatchGRPOScheduler(eta=0.5),
        train_ds=_ToyCondDS(), val_ds=_ToyCondDS(), latent_shape=_LAT,
        vae=AutoencoderKL(scaling_factor=0.5),
        real_latents=torch.randn(6, *_LAT),
        feature_net=_FakeFeatureNet(),
    )
    module = GRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=_LAT, lr=1e-3,
    )
    trainer, ckpt = run_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
        num_synth=3, cov_ridge=1e-2,
    )
    metrics = trainer.callback_metrics
    assert "val/fid" in metrics, "FIDCallback must log val/fid"
    assert torch.isfinite(metrics["val/fid"])
    assert "val/mean_reward" in metrics, "the PatchGAN progress signal stays logged"
    assert torch.isfinite(metrics["val/mean_reward"])
    # The selection metric flips to val/fid (min) — the anti-reward-hacking screen.
    assert ckpt.monitor == "val/fid"
    assert ckpt.mode == "min"


def test_run_grpo_training_monitor_override_derives_correct_mode(tmp_path):
    """Overriding monitor_metric=val/mean_reward keeps mode=max even with FID attached (codex #62).

    The mode auto-derives from the FINAL monitor_metric, not from fid_active: a caller
    who keeps the FID triple but selects back on the reward signal must get mode=max
    (else the checkpoint would select the LOWEST reward — a silent footgun).
    """
    from manifold import AutoencoderKL, UNet3DConditionModel
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import GRPOInputs, run_grpo_training

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    inputs = GRPOInputs(
        policy=policy, reward_model=_reward_model(), scheduler=FlowMatchGRPOScheduler(eta=0.5),
        train_ds=_ToyCondDS(), val_ds=_ToyCondDS(), latent_shape=_LAT,
        vae=AutoencoderKL(scaling_factor=0.5),
        real_latents=torch.randn(6, *_LAT),
        feature_net=_FakeFeatureNet(),
    )
    module = GRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=_LAT, lr=1e-3,
    )
    _, ckpt = run_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
        num_synth=3, cov_ridge=1e-2,
        monitor_metric="val/mean_reward",  # override back to the reward signal; mode=None
    )
    assert ckpt.monitor == "val/mean_reward"
    assert ckpt.mode == "max", "overriding monitor_metric must derive mode from the metric, not fid_active"


def test_grpo_module_sample_is_deployed_heun_not_the_sde(unet):
    """GRPOModule.sample is the deployed two-eval Heun — η does NOT leak into val gen (#58).

    Validation must measure the distribution JiT ships, so ``sample`` delegates to
    the shared ``sample_latent_flow`` primitive (the deployed Heun), NOT the rollout
    SDE. The GRPO scheduler's η knob only affects ``sde_step_mean`` (the exploration
    step); the inherited ``euler_step``/``heun_correct`` that ``sample_latent_flow``
    calls are η-agnostic. So same noise + the GRPO(η=0.9) scheduler produces a latent
    bit-identical to the plain Heun scheduler — the parity guard that the FID
    callback's generation path is the deployed sampler.
    """
    from manifold import FlowMatchHeunDiscreteScheduler
    from manifold.modules import GRPOModule

    mod = GRPOModule(
        unet, _reward_model(), FlowMatchGRPOScheduler(eta=0.9),
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=_LAT, lr=1e-3,
    )
    spacing = [1.0, 1.0, 1.0]
    out = mod.sample((1, *_LAT), spacing, 1, 3, generator=torch.Generator().manual_seed(0))
    # Same noise rebuilt from the same seed; reference runs the PLAIN Heun scheduler.
    noise = torch.randn(1, *_LAT, generator=torch.Generator().manual_seed(0))
    ref = sample_latent_flow(
        unet, FlowMatchHeunDiscreteScheduler(), noise, spacing, 1, num_inference_steps=3
    )
    assert torch.equal(out, ref)  # η-agnostic deployed Heun — no SDE leak into val gen


# -- launch readiness (#59): no EMA + measurement harness -------------------


def test_run_grpo_measurement_reports_it_per_s(tmp_path):
    """run_grpo_measurement times a fit + reports it/s (peak GPU is 0 off-CUDA) (#59).

    The launch-gate harness: sizes G / eta_step_list / n_epochs by measuring the real
    budget's throughput + peak GPU memory on the target cluster. Off-CUDA the peak
    memory read is 0 (the read is GPU-only); the it/s + elapsed are real on any host.
    """
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import run_grpo_measurement

    inputs = _inputs()
    module = GRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=2, eta_step_list=(0,), num_steps=3, latent_shape=inputs.latent_shape, lr=1e-3,
    )
    it_per_s, peak, elapsed = run_grpo_measurement(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        devices=1, accelerator="cpu", batch_size=2,
    )
    assert it_per_s > 0, f"it/s must be a positive real, got {it_per_s}"  # nan fails >
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
        "grpo_train: {batch_size: 2, lr: 1.0e-3, n_epochs: 1, num_steps: 3, "
        "G: 2, eta: 0.5, clip_range: 1.0e-4, adv_clip_max: 5.0, "
        "eta_step_list: [0], latent_shape: [4, 8, 8, 8], "
        "kl_coef: 0.0, reward_bound: none, reward_temp: 8.0}\n"
    )
    return str(env), str(train), str(net)


def test_main_runs_end_to_end_with_fake_data(tmp_path):
    """main(): argparse -> compose -> build -> fit -> ckpt (the fake-data injection seam)."""
    from manifold.training.grpo_cli import main as grpo_main

    env, train, net = _write_tiny_configs(tmp_path)
    rc = grpo_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=lambda cfg, device: _inputs(),
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_main_native_reward_default_none_and_validated(tmp_path):
    """--native-dir/--reward-path default None and are required without a data_provider."""
    from manifold.training.grpo_cli import main as grpo_main

    env, train, net = _write_tiny_configs(tmp_path)
    with pytest.raises(ValueError, match="native-dir"):
        grpo_main(["-e", env, "-c", train, "-t", net, "-g", "1"])
    # With a data_provider the missing args are NOT required (smoke seam intact).
    rc = grpo_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=lambda cfg, device: _inputs(),
    )
    assert rc == 0


def test_main_uses_committed_default_grpo_recipe(tmp_path):
    """main() with NO -c (argparse default) resolves the committed config_grpo.yaml."""
    from manifold.training.grpo_cli import main as grpo_main

    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
        + "val_subset_size: 4\n"
    )
    net = "configs/network/config_network.yaml"
    rc = grpo_main(
        ["-e", str(env), "-t", net, "-g", "1", "--max-epochs", "1",
         "grpo_train.G=2", "grpo_train.num_steps=3", "grpo_train.eta_step_list=[0]",
         "grpo_train.latent_shape=[4,8,8,8]"],
        data_provider=lambda cfg, device: _inputs(),
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


# -- full v1 rollout budget (#57) --------------------------------------------
#
# The numerics (σ_t equimarginal, noise-end clamp 1/n, clean-end t_eps, per-sample
# (B,) parity) ship in #56 with their own tests. #57's remaining acceptance is the
# full-v1-budget feasibility: the real G=8 / eta_step_list=[0..7] / num_steps=15
# rollout is finite across the whole trajectory (no NaN/Inf from either blowup) and
# runs end-to-end via per-term manual_backward (peak autograd = one UNet-forward).


def test_full_v1_budget_rollout_is_finite_across_trajectory(unet):
    """G=8, eta_step_list=[0..7], num_steps=15: every buffer entry finite (#57).

    The full noisy-half budget walks the whole grid, so it exercises both the
    noise-end σ_t blowup (node 0, t=0 → clamped to 1/n) and the clean-end
    1/(1−t) blowup (node 14→15, t_next=1 → the Heun final-step Euler). The
    acceptance: no NaN/Inf anywhere in the trajectory — every z_kplus1, log-prob,
    advantage, and reward is finite. (CPU cannot measure GPU OOM, but finiteness
    across the trajectory is the numerics half of the budget acceptance; the
    end-to-end fit below is the per-term-manual_backward half.)
    """
    torch.manual_seed(0)
    noise = torch.randn(2, *_LAT)
    buf = singular_branch_rollout(
        unet, FlowMatchGRPOScheduler(eta=0.7), _reward_model(), noise,
        [1.0, 1.0, 1.0], 1, G=8, eta_step_list=list(range(8)), num_steps=15,
    )
    assert len(buf) == 8  # one entry per perturbed step in [0..7]
    for e in buf:
        for key in ("z_kplus1", "old_log_prob", "advantage", "rewards"):
            assert torch.isfinite(e[key]).all(), f"{key} has NaN/Inf in the full-budget rollout"
        assert e["t_k"] < e["t_next"]


def test_full_v1_budget_runs_end_to_end(tmp_path):
    """The real v1 budget (G=8, [0..7], num_steps=15) completes a fit without OOM (#57).

    End-to-end via ``run_grpo_training``: the per-term ``manual_backward`` frees each
    inner step's graph before the next, so peak autograd memory is one UNet-forward
    at G·N = 8·15 (the 3D-feasibility invariant, ADR-0011/0012). CPU cannot measure
    GPU OOM, but completing the fit (finite loss + checkpoint written) is the
    tractability signal the budget pass requires.
    """
    from manifold import UNet3DConditionModel
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import GRPOInputs, run_grpo_training

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    inputs = GRPOInputs(
        policy=policy, reward_model=_reward_model(), scheduler=FlowMatchGRPOScheduler(eta=0.7),
        train_ds=_ToyCondDS(), val_ds=_ToyCondDS(), latent_shape=_LAT,
    )
    module = GRPOModule(
        inputs.policy, inputs.reward_model, inputs.scheduler,
        G=8, eta_step_list=tuple(range(8)), num_steps=15, latent_shape=_LAT, lr=1e-3,
    )
    trainer, ckpt = run_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
    )
    assert torch.isfinite(trainer.callback_metrics["val/mean_reward"])
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()


def test_codex116_val_noise_seeded_by_rank_and_batch():
    """codex #116 P2 (Comment 5): GRPO ``validation_step`` offsets the validation-noise
    generator seed by rank + batch, so under DDP each rank's val shard is a distinct
    draw (not a duplicate of rank 0). Verified by source inspection."""
    import inspect

    from manifold.modules import grpo

    src = inspect.getsource(grpo.GRPOModule.validation_step)
    assert "torch.Generator" in src, "validation_step must seed a Generator (not plain randn)"
    assert "get_rank" in src, "generator seed must offset by rank (Comment 5)"
    assert "manual_seed" in src


def test_codex116_padding_mask_uses_global_sum_count():
    """GRPO excludes padded rewards then logs one globally reduced sum/count ratio."""
    import inspect
    from manifold.modules import grpo
    step = inspect.getsource(grpo.GRPOModule.validation_step)
    end = inspect.getsource(grpo.GRPOModule.on_validation_epoch_end)
    assert "_is_padding" in step
    assert "_val_reward_sum" in step and "_val_reward_count" in step
    assert "all_reduce" in end and "ReduceOp.SUM" in end
    assert "self.log(" not in step


# ============================================================================
# Mode-2 (ControlNet on the frozen base) — ADR-0028 / issue #138
# ============================================================================
#
# The two-mode unification: Mode-2 freezes the base UNet, trains the ControlNet
# against the frozen reward on z_K (unconditional — the ControlNet's conditional
# fidelity is driven by the policy x0, not the reward input), and reuses the SAME
# spine (transition log-prob / KL anchor / advantage / clipped surrogate /
# singular-branch rollout).
# The Mode-2 perturbed-step backward is autograd-safe ONLY because the base
# wrapper's out-of-place residual forward (ADR-0026 corrected hazard) neutralizes
# MONAI's in-place residual adds — these tests pin that contract.


def _mode2_base() -> UNet3DConditionModel:
    """A tiny base UNet with the zero-init output conv re-initialized.

    MONAI MAISI zero-initializes the final output projection, so at init the base
    output is identically zero and the ControlNet's residual-injection effect on the
    output is masked. Re-initializing the all-zero ``out`` params (emulating a
    pretrained base) lets the tests exercise the full base-output→ControlNet path.
    """
    torch.manual_seed(0)
    base = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in base.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return base


def _mode2_controlnet(base) -> ControlNet3DConditionModel:
    torch.manual_seed(1)
    cn = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    cn.load_base_encoder_weights(base)
    return cn


def _mode2_reward() -> RewardModel:
    """Single-latent reward (in_channels = C_latent) scoring z_K unconditionally.

    Mode-2 scores the terminal latent ``z_K`` only (``reward(z_K)``) — the same
    single-latent reward Mode-1 uses, NOT a 2·C condition-aware concat.
    """
    torch.manual_seed(0)
    return RewardModel(spatial_dims=3, in_channels=4, channels=8, num_layers_d=1)


def _mode2_batch() -> dict:
    return {
        "src_latent": torch.randn(2, 4, 16, 16, 8),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
        "src_label": torch.tensor([1, 0]),
        "tgt_label": torch.tensor([2, 3]),
    }


def _mode2_module(base, controlnet, reward, *, reference=None, kl_coef=0.0, G=4):
    return GRPOModule(
        base,
        reward,
        FlowMatchGRPOScheduler(eta=0.5),
        G=G,
        eta_step_list=[0, 1],
        num_steps=4,
        latent_shape=(4, 16, 16, 8),
        reference_policy=reference,
        kl_coef=kl_coef,
        controlnet=controlnet,
        freeze_unet=True,
        lr=1e-5,
    )


def _run_manual_training_step(module, batch):
    """Drive one training_step outside a Trainer (no-op optimizer so grads persist)."""
    opt = module.configure_optimizers()["optimizer"]

    class _NoStepOpt:
        def step(self):
            pass

        def zero_grad(self, set_to_none=True):
            pass

    module.optimizers = lambda: _NoStepOpt()
    module.lr_schedulers = lambda: None
    module.manual_backward = lambda loss: loss.backward()
    module.log = lambda *a, **k: None
    return module.training_step(dict(batch), 0), opt


def test_mode2_requires_freeze_unet():
    """Mode-2 (controlnet set) without freeze_unet=True is a construction error."""
    base = _mode2_base()
    cn = _mode2_controlnet(base)
    with pytest.raises(ValueError, match="freeze_unet"):
        GRPOModule(
            base, _mode2_reward(), FlowMatchGRPOScheduler(eta=0.5),
            controlnet=cn, freeze_unet=False, latent_shape=(4, 16, 16, 8),
        )


def test_mode2_freezes_base_and_optimizes_controlnet_only():
    """Mode-2: base frozen + unregistered; optimizer wires ControlNet params only."""
    base = _mode2_base()
    cn = _mode2_controlnet(base)
    module = _mode2_module(base, cn, _mode2_reward())

    # Base frozen + held unregistered (off parameters/state_dict/checkpoint).
    assert not any(p.requires_grad for p in base.parameters())
    opt_param_ids = {id(p) for p in module.parameters()}
    assert not ({id(p) for p in base.parameters()} & opt_param_ids)
    assert {id(p) for p in cn.parameters()} <= opt_param_ids
    assert not any(k.startswith("unet.") for k in module.state_dict())
    assert any("controlnet" in k for k in module.state_dict())

    # Optimizer wires ONLY the ControlNet params.
    opt = module.configure_optimizers()["optimizer"]
    opt_params = {p for g in opt.param_groups for p in g["params"]}
    assert opt_params == set(cn.parameters())
    assert not (opt_params & set(base.parameters()))


def test_mode2_kl_anchor_is_base_plus_controlnet_pair():
    """Mode-2 reference_policy is a (base, controlnet) pair (ADR-0015 anchor)."""
    import copy

    base = _mode2_base()
    cn = _mode2_controlnet(base)
    ref = (copy.deepcopy(base), copy.deepcopy(cn))
    module = _mode2_module(base, cn, _mode2_reward(), reference=ref, kl_coef=0.1)
    assert module.reference_unet is not None
    assert module.reference_controlnet is not None
    # Both frozen.
    assert not any(p.requires_grad for p in module.reference_unet.parameters())
    assert not any(p.requires_grad for p in module.reference_controlnet.parameters())


def test_mode2_kl_zero_at_init_when_anchor_matches_policy():
    """The Mode-2 KL term is ~0 at init when the anchor is a deepcopy of the policy."""
    import copy

    base = _mode2_base()
    cn = _mode2_controlnet(base)
    ref = (copy.deepcopy(base), copy.deepcopy(cn))
    module = _mode2_module(base, cn, _mode2_reward(), reference=ref, kl_coef=0.1)

    # A stored-step dict mimicking the rollout buffer at one perturbed step.
    module.scheduler.set_timesteps(4, device="cpu")  # sde_step_mean needs the grid
    z_k = torch.randn(2, 4, 16, 16, 8)
    spacing_t = torch.tensor([1.0, 1.0, 1.0])
    cond = (
        torch.randn(2, 4, 16, 16, 8),
        torch.tensor([1, 0]),
        torch.tensor([2, 3]),
    )
    step = {"z_k": z_k, "t_k": 0.25, "t_next": 0.5, "z_kplus1": torch.randn(2, 4, 4, 16, 16, 8)}
    x_src, src_labels, tgt_labels = cond
    x0 = module._controlnet_forward(cn, z_k, step["t_k"], x_src, spacing_t, src_labels, tgt_labels)
    mean_new, std_new = module.scheduler.sde_step_mean(x0, z_k, step["t_k"], step["t_next"])
    kl = module._transition_kl(step, mean_new, std_new, spacing_t, tgt_labels, cond)
    assert kl is not None
    # Anchor == policy ⇒ μ_θ == μ_ref ⇒ KL ≈ 0.
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)


def test_mode2_training_step_backward_through_perturbed_step():
    """The Mode-2 inner-loop backward flows base-output→ControlNet (hazard neutralized).

    This is the load-bearing Mode-2 test: the perturbed-step grad re-eval runs the
    frozen base forward WITH ControlNet residuals injected; the backward must reach
    the ControlNet WITHOUT the MONAI in-place-residual autograd error (the base
    wrapper's out-of-place ``_forward_with_residuals`` is what makes this safe), and
    must NOT touch the frozen base. Grad magnitudes may underflow to ~0 (the tiny
    clip_range + zero-conv gating), so assert on grad PRESENCE for the ControlNet
    and ABSENCE for the base.
    """
    base = _mode2_base()
    cn = _mode2_controlnet(base)
    module = _mode2_module(base, cn, _mode2_reward())
    out, _ = _run_manual_training_step(module, _mode2_batch())

    assert torch.isfinite(out["loss"])
    n_cn_grad = sum(1 for p in cn.parameters() if p.grad is not None)
    assert n_cn_grad > 0, "backward did not reach the ControlNet"
    base_grads = [p.grad for p in base.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in base_grads), "frozen base received grad"


def test_mode2_optimizer_step_updates_controlnet_not_base():
    """A real opt.step() moves ControlNet params; the frozen base never moves."""
    base = _mode2_base()
    cn = _mode2_controlnet(base)
    module = _mode2_module(base, cn, _mode2_reward())
    base_before = [p.detach().clone() for p in base.parameters()]

    out, _ = _run_manual_training_step(module, _mode2_batch())
    assert torch.isfinite(out["loss"])
    # The frozen base never moved (no grad, never stepped).
    for before, after in zip(base_before, base.parameters()):
        assert torch.equal(before, after)


def test_mode2_conditioning_reads_src_tgt_labels():
    """Mode-2 _conditioning returns (x_src, src_labels, tgt_labels) as cond."""
    base = _mode2_base()
    cn = _mode2_controlnet(base)
    module = _mode2_module(base, cn, _mode2_reward())
    spacing_t, class_labels, B, cond = module._conditioning(_mode2_batch())
    assert B == 2
    assert cond is not None
    x_src, src_labels, tgt_labels = cond
    assert x_src.shape == (2, 4, 16, 16, 8)
    # class_labels is the per-sample tgt_label (the base's own modality embedding).
    assert torch.equal(class_labels, torch.tensor([2, 3]))
    assert torch.equal(src_labels, torch.tensor([1, 0]))
    assert torch.equal(tgt_labels, torch.tensor([2, 3]))


# -- Mode-2 CLI entry point (#141): --grpo-mode 2 on the GRPO entry point --------


class _ToyPairedCondDS(Dataset):
    """A tiny PAIRED conditioning dataset for Mode-2 (train/val).

    Emits ``{spacing, src_latent, src_label, tgt_label}`` — the ControlNet's control
    signal + translation direction. GRPO samples the group noise; the paired batch
    carries only the ControlNet condition (the rollout's stochastic input).
    """

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "src_latent": torch.randn(4, 16, 16, 8),
            "src_label": torch.tensor(1, dtype=torch.long),
            "tgt_label": torch.tensor(2, dtype=torch.long),
        }


def _mode2_inputs():
    """Mode-2 injection-seam bundle: frozen base + trainable ControlNet + reward + paired conditioning."""
    from manifold.training.grpo_cli import GRPOInputs

    base = _mode2_base()
    cn = _mode2_controlnet(base)
    return GRPOInputs(
        policy=base,
        reward_model=_mode2_reward(),
        scheduler=FlowMatchGRPOScheduler(eta=0.5),
        train_ds=_ToyPairedCondDS(),
        val_ds=_ToyPairedCondDS(),
        latent_shape=(4, 16, 16, 8),
        controlnet=cn,
    )


def test_main_runs_end_to_end_mode2(tmp_path):
    """main() ``--grpo-mode 2``: the ControlNet path runs end-to-end via the CLI seam.

    The Mode-2 entry-point acceptance (#141): ``--grpo-mode 2`` freezes the base
    UNet and trains the ControlNet against the frozen reward on z_K. The
    data_provider injects a frozen base + trainable ControlNet + paired conditioning;
    main wires ``controlnet`` / ``freeze_unet`` into GRPOModule (the --grpo-mode
    flag's load-bearing path) and writes a checkpoint.
    """
    from manifold.training.grpo_cli import main as grpo_main

    env, train, net = _write_tiny_configs(tmp_path)
    rc = grpo_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1",
         "--grpo-mode", "2", "grpo_train.latent_shape=[4,16,16,8]"],
        data_provider=lambda cfg, device: _mode2_inputs(),
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_main_mode2_requires_controlnet(tmp_path):
    """``--grpo-mode 2`` without a ControlNet (Mode-1 inputs) fails fast at main.

    The Mode-2 guard: ``--grpo-mode 2`` requires ``GRPOInputs.controlnet`` (the data
    provider / real inputs must supply it); Mode-1 inputs have none.
    """
    from manifold.training.grpo_cli import main as grpo_main

    env, train, net = _write_tiny_configs(tmp_path)
    with pytest.raises(ValueError, match="ControlNet"):
        grpo_main(
            ["-e", env, "-c", train, "-t", net, "-g", "1", "--grpo-mode", "2"],
            data_provider=lambda cfg, device: _inputs(),  # Mode-1 inputs — no controlnet
        )


def test_run_grpo_training_mode2_skips_unconditional_fid(tmp_path):
    """Regression (codex #142): Mode-2 skips the unconditional FID and monitors
    ``val/mean_reward`` even when the FID triple is present.

    The base UNet is frozen and only the ControlNet trains, but FIDCallback's
    unconditional ``module.sample()`` rollout ignores the ControlNet — so val/fid
    would be a CONSTANT frozen-base metric, independent of the learned policy.
    ``run_grpo_training`` must not attach it in Mode-2.
    """
    from manifold.modules import GRPOModule
    from manifold.training.grpo_cli import GRPOInputs, run_grpo_training

    class _FakeVAE:  # FID triple present — would force fid_active=True in Mode-1
        pass

    base = _mode2_base()
    cn = _mode2_controlnet(base)
    inputs = GRPOInputs(
        policy=base,
        reward_model=_mode2_reward(),
        scheduler=FlowMatchGRPOScheduler(eta=0.5),
        train_ds=_ToyPairedCondDS(),
        val_ds=_ToyPairedCondDS(),
        latent_shape=(4, 16, 16, 8),
        controlnet=cn,
        vae=_FakeVAE(),
        real_latents=torch.randn(2, 4, 16, 16, 8),
        feature_net=object(),
    )
    module = GRPOModule(
        base, inputs.reward_model, FlowMatchGRPOScheduler(eta=0.5),
        G=2, eta_step_list=[0], num_steps=3, latent_shape=(4, 16, 16, 8),
        controlnet=cn, freeze_unet=True, lr=1e-5,
    )
    trainer, ckpt = run_grpo_training(
        module=module, inputs=inputs, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
    )
    from manifold.metrics import FIDCallback

    assert not any(isinstance(c, FIDCallback) for c in trainer.callbacks), (
        "FIDCallback attached in Mode-2 — a constant frozen-base metric"
    )
    assert ckpt.monitor == "val/mean_reward"


# -- the real _real_inputs_mode2 CLI path (#143) ------------------------------
#
# Mode-2's real-data path loads the supervised ControlNet export (frozen base +
# ControlNet) and resolves the paired conditioning split, then constructs a
# GRPOInputs with controlnet set so main() reaches run_grpo_training without the
# "requires a ControlNet" guard. The real BraTS + VAE path is cluster-only; these
# tests fake the manifest / split / warmed cache / reward ckpt (the same seam
# paired_reward_cli's real-inputs test uses).


def _save_mode2_reward_ckpt(path, *, in_channels=4) -> None:
    """Write a minimal reward ckpt (state_dict keys prefixed ``reward_model.``).

    Mode-2's reward scores the terminal latent ``z_K`` unconditionally
    (``in_channels = C_latent``, the same single-latent reward Mode-1 uses) — NOT the
    2·C condition-aware paired reward. _real_inputs_mode2 strips the ``reward_model.``
    Lightning prefix.
    """
    rm = RewardModel(spatial_dims=3, in_channels=in_channels, channels=8, num_layers_d=1)
    torch.save({"state_dict": {f"reward_model.{k}": v for k, v in rm.state_dict().items()}}, str(path))


def _fake_mode2_manifests(n_train=4, n_val=2):
    train = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(n_train)
    ]
    val = [
        {"src": f"/v/s{i}-t1n.nii.gz", "tgt": f"/v/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(n_val)
    ]
    return train, val


class _FakeMode2PairedDS(Dataset):
    """A warmed ``PairedLatentDataset`` stand-in emitting the paired conditioning keys.

    Serves scaled latents + labels + spacing; carries a settable ``scaling_factor``
    sentinel so the test asserts _real_inputs_mode2 overwrote it with the export's
    (ADR-0021 scale-consistency). ``warm_cache`` is a no-op (the fake is pre-warmed).
    """

    def __init__(self, n):
        self._n = n
        self.scaling_factor = None  # sentinel: _real_inputs_mode2 must set this
        torch.manual_seed(0)
        self._lat = torch.randn(n, 4, 16, 16, 8)

    def warm_cache(self, *a, **k):
        return None

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return {
            "src_latent": self._lat[i],
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "src_label": torch.tensor(0, dtype=torch.long),
            "tgt_label": torch.tensor(1, dtype=torch.long),
        }


def _mode2_cfg(tmp_path):
    import omegaconf

    return omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "latent_cache_dir": str(tmp_path / "cache"),
            "diffusion_unet_inference": {"dim": [128, 128, 64]},
            "autoencoder": {"num_channels": [8, 8]},
            "latent_channels": 4,
            "reward_model": {"spatial_dims": 3, "channels": 8, "num_layers_d": 1, "norm": "BATCH"},
            "grpo": {"cache_tag": "paired_train", "val_fraction": 0.0},
            "grpo_train": {"latent_shape": [4, 16, 16, 8], "eta": 0.7},
            "random_seed": 0,
        }
    )


def test_real_inputs_mode2_loads_controlnet_and_builds_paired_conditioning(tmp_path, monkeypatch):
    """_real_inputs_mode2: frozen base + trainable ControlNet + paired conditioning (#143).

    Loads the supervised ControlNet export via load_frozen_controlnet_generator,
    keeps the base frozen, unfreezes ONLY the ControlNet, resolves the paired
    train/val split, and returns GRPOInputs.controlnet — so main() --grpo-mode 2
    reaches run_grpo_training without the "requires a ControlNet" guard.
    """
    from tests.test_paired_reward_real import _save_controlnet_export

    from manifold.data import paired_brats as pb
    from manifold.data import paired_latent_dataset as pld_mod
    from manifold.training import grpo_cli, paired_reward_cli

    _save_controlnet_export(tmp_path / "native")
    _save_mode2_reward_ckpt(tmp_path / "reward.ckpt", in_channels=4)

    train_manifest, val_manifest = _fake_mode2_manifests()
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest + val_manifest)
    monkeypatch.setattr(
        paired_reward_cli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, val_manifest)
    )

    # Fake the warmed paired cache: two instances (train then val), pre-warmed.
    built = []
    fake_train, fake_val = _FakeMode2PairedDS(4), _FakeMode2PairedDS(2)
    _queue = [fake_val, fake_train]  # first build pops train, second pops val

    class _FakePLD:
        def __init__(self, vol_ds, encode_fn=None, cache_dir=None, cache_tag=None):
            self._ds = _queue.pop()
            self.scaling_factor = None  # sentinel
            built.append(self)

        def warm_cache(self, *a, **k):
            return None

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, i):
            return self._ds[i]

    monkeypatch.setattr(pld_mod, "PairedLatentDataset", _FakePLD)

    cfg = _mode2_cfg(tmp_path)
    inputs = grpo_cli._real_inputs_mode2(
        cfg, str(tmp_path / "native"), str(tmp_path / "reward.ckpt"),
        str(tmp_path / "cache"), torch.device("cpu"),
    )

    # The ControlNet is supplied (main() won't trip the "requires a ControlNet" guard).
    assert inputs.controlnet is not None
    # Base frozen; the ControlNet is the ONLY trainable arm.
    assert not any(p.requires_grad for p in inputs.policy.parameters())
    assert any(p.requires_grad for p in inputs.controlnet.parameters())
    # The reward is frozen and scores z_K unconditionally (in_channels = C_latent).
    assert not any(p.requires_grad for p in inputs.reward_model.parameters())
    # Paired conditioning datasets (train 4 / val 2).
    assert len(inputs.train_ds) == 4
    assert len(inputs.val_ds) == 2
    item = inputs.train_ds[0]
    for key in ("src_latent", "spacing", "src_label", "tgt_label"):
        assert key in item, f"paired conditioning batch missing {key}"
    # ADR-0021: both datasets' scaling_factor set to the export's (0.5 in the helper).
    assert len(built) == 2
    assert all(ds.scaling_factor == 0.5 for ds in built), (
        "_real_inputs_mode2 must set ds.scaling_factor = the export's scaling_factor (ADR-0021)"
    )
    # No FID triple in Mode-2 (the unconditional FID is a constant frozen-base metric).
    assert inputs.vae is None and inputs.real_latents is None and inputs.feature_net is None
    # The KL anchor (ADR-0015): a (base, controlnet) pair snapshot, so the recipe's
    # kl_coef does not silently disable (codex #151 P1). Weight-matched to the loaded
    # arms (a pre-update deepcopy; the GRPOModule freezes + unregisters it).
    assert inputs.reference_policy is not None
    ref_base, ref_cn = inputs.reference_policy
    for k, v in inputs.policy.state_dict().items():
        assert torch.equal(ref_base.state_dict()[k], v)
    for k, v in inputs.controlnet.state_dict().items():
        assert torch.equal(ref_cn.state_dict()[k], v)


def test_real_inputs_mode2_raises_on_no_val_split(tmp_path, monkeypatch):
    """No held-out val split -> clear ValueError (train never reused as val)."""
    from tests.test_paired_reward_real import _save_controlnet_export

    from manifold.data import paired_brats as pb
    from manifold.training import grpo_cli, paired_reward_cli

    _save_controlnet_export(tmp_path / "native")
    _save_mode2_reward_ckpt(tmp_path / "reward.ckpt", in_channels=4)

    train_manifest, _ = _fake_mode2_manifests()
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest)
    monkeypatch.setattr(
        paired_reward_cli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, [])
    )

    cfg = _mode2_cfg(tmp_path)
    with pytest.raises(ValueError, match="val split"):
        grpo_cli._real_inputs_mode2(
            cfg, str(tmp_path / "native"), str(tmp_path / "reward.ckpt"),
            str(tmp_path / "cache"), torch.device("cpu"),
        )


def test_real_inputs_mode2_rejects_condition_aware_reward_ckpt(tmp_path, monkeypatch):
    """A 2·C condition-aware paired-reward ckpt fails fast with a readable error.

    Mode-2 scores z_K unconditionally (in_channels = C_latent = 4). Passing a
    2·C paired-reward ckpt (in_channels = 8, from manifold-train-paired-reward)
    must raise a clear ValueError BEFORE load_state_dict's cryptic shape error
    (codex #151: the z_K-only reward is an intentional design decision; the check
    turns the real 2·C-ckpt incompatibility into an actionable message).
    """
    from tests.test_paired_reward_real import _save_controlnet_export

    from manifold.data import paired_brats as pb
    from manifold.training import grpo_cli, paired_reward_cli

    _save_controlnet_export(tmp_path / "native")
    # 2·C condition-aware ckpt: in_channels = 8 = 2 * C_latent(4).
    _save_mode2_reward_ckpt(tmp_path / "reward.ckpt", in_channels=8)

    train_manifest, val_manifest = _fake_mode2_manifests()
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest + val_manifest)
    monkeypatch.setattr(
        paired_reward_cli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, val_manifest)
    )

    cfg = _mode2_cfg(tmp_path)
    with pytest.raises(ValueError, match="in_channels=8"):
        grpo_cli._real_inputs_mode2(
            cfg, str(tmp_path / "native"), str(tmp_path / "reward.ckpt"),
            str(tmp_path / "cache"), torch.device("cpu"),
        )


def test_main_mode2_real_path_dispatches_and_builds_controlnet_module(tmp_path, monkeypatch):
    """main() --grpo-mode 2 dispatches to _real_inputs_mode2 and reaches fit (no guard).

    Exercises the full main() wiring (arg dispatch → _real_inputs_mode2 → GRPOModule
    with controlnet + freeze_unet → run_grpo_training) with the paired cache faked.
    """
    from tests.test_paired_reward_real import _save_controlnet_export

    from manifold.data import paired_brats as pb
    from manifold.data import paired_latent_dataset as pld_mod
    from manifold.training import grpo_cli, paired_reward_cli

    env, train, net = _write_tiny_configs(tmp_path)
    # Mode-2 requires data_base_dir + the paired/reward config blocks; the tiny configs
    # already carry model_dir. Point data_base_dir at a throwaway dir and add the
    # network-side blocks _real_inputs_mode2 reads (inference dim / autoencoder divisor /
    # reward_model arch / grpo.cache_tag).
    import omegaconf
    extra_env = omegaconf.OmegaConf.create({"data_base_dir": str(tmp_path)})
    env_cfg = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.load(env), extra_env)
    omegaconf.OmegaConf.save(env_cfg, env)
    extra_net = omegaconf.OmegaConf.create(
        {
            "diffusion_unet_inference": {"dim": [128, 128, 64]},
            "autoencoder": {"num_channels": [8, 8]},
            "reward_model": {"spatial_dims": 3, "channels": 8, "num_layers_d": 1, "norm": "BATCH"},
            "grpo": {"cache_tag": "paired_train", "val_fraction": 0.0},
        }
    )
    net_cfg = omegaconf.OmegaConf.merge(omegaconf.OmegaConf.load(net), extra_net)
    omegaconf.OmegaConf.save(net_cfg, net)

    _save_controlnet_export(tmp_path / "native")
    _save_mode2_reward_ckpt(tmp_path / "reward.ckpt", in_channels=4)

    train_manifest, val_manifest = _fake_mode2_manifests()
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest + val_manifest)
    monkeypatch.setattr(
        paired_reward_cli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, val_manifest)
    )
    fake_train, fake_val = _FakeMode2PairedDS(4), _FakeMode2PairedDS(2)
    _queue = [fake_val, fake_train]

    class _FakePLD:
        def __init__(self, vol_ds, encode_fn=None, cache_dir=None, cache_tag=None):
            self._ds = _queue.pop()
            self.scaling_factor = None

        def warm_cache(self, *a, **k):
            return None

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, i):
            return self._ds[i]

    monkeypatch.setattr(pld_mod, "PairedLatentDataset", _FakePLD)

    rc = grpo_cli.main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1",
         "--grpo-mode", "2", "grpo_train.latent_shape=[4,16,16,8]",
         "--native-dir", str(tmp_path / "native"),
         "--reward-path", str(tmp_path / "reward.ckpt"),
         "--latents-dir", str(tmp_path / "cache")],
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


# -- codex #151 regression guards ----------------------------------------------


def test_mode2_recipe_resolves_inference_dim():
    """The committed Mode-2 recipe defines diffusion_unet_inference.dim (codex #151 P1).

    _real_inputs_mode2 reads cfg.diffusion_unet_inference.dim directly (load-bearing
    for the paired cache's target_dim / divisor); the recipe must define it so the
    documented launch does not raise ConfigAttributeError.
    """
    from manifold.config import load_config

    cfg = load_config(
        "configs/env/environment.yaml",
        "configs/train/config_controlnet_grpo.yaml",
        "configs/network/config_network.yaml",
    )
    assert tuple(int(d) for d in cfg.diffusion_unet_inference.dim) == (256, 256, 128)


def test_real_inputs_mode2_rejects_stale_cache_shape(tmp_path, monkeypatch):
    """A paired cache warmed at a different target_dim -> fail-fast ValueError (codex #151 P2).

    The cache key is sample_id + tag (NOT target_dim), so a stale cache silently
    reuses wrong-shape src latents. _real_inputs_mode2 validates every unique
    latent's spatial shape against the recipe's target_dim / divisor and raises.
    """
    from tests.test_paired_reward_real import _save_controlnet_export

    from manifold.data import paired_brats as pb
    from manifold.data import paired_latent_dataset as pld_mod
    from manifold.training import grpo_cli, paired_reward_cli

    _save_controlnet_export(tmp_path / "native")
    _save_mode2_reward_ckpt(tmp_path / "reward.ckpt", in_channels=4)

    train_manifest, val_manifest = _fake_mode2_manifests()
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest + val_manifest)
    monkeypatch.setattr(
        paired_reward_cli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, val_manifest)
    )

    # A fake PairedLatentDataset exposing the real cache interface (source + raw_latent)
    # but serving a WRONG-shape latent. The recipe target_dim [128,128,64] / divisor 4
    # (autoencoder [64,128,256] -> div 4) expects ceil -> (32,32,16); serve (16,16,8).
    class _FakePLDStale:
        class source:
            @staticmethod
            def unique_sample_ids():
                return ["s0", "s1", "s2", "s3"]

        def __init__(self, vol_ds, encode_fn=None, cache_dir=None, cache_tag=None):
            self.scaling_factor = None

        def warm_cache(self, *a, **k):
            return None

        def __len__(self):
            return 4

        def raw_latent(self, sid):
            return torch.randn(4, 16, 16, 8)  # WRONG spatial (16,16,8) != (32,32,16)

    monkeypatch.setattr(pld_mod, "PairedLatentDataset", _FakePLDStale)

    cfg = _mode2_cfg(tmp_path)  # dim [128,128,64]; autoencoder [8,8] -> divisor 2 -> (64,64,32)
    with pytest.raises(ValueError, match="Cached paired latent"):
        grpo_cli._real_inputs_mode2(
            cfg, str(tmp_path / "native"), str(tmp_path / "reward.ckpt"),
            str(tmp_path / "cache"), torch.device("cpu"),
        )
