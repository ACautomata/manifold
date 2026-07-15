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

from manifold import FlowMatchGRPOScheduler, FlowMatchHeunDiscreteScheduler, RewardModel
from manifold.modules import sample_latent_flow, singular_branch_rollout
from manifold.modules.grpo import clipped_surrogate_loss, gaussian_log_prob, group_advantage

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
