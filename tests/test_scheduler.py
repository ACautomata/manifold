"""Scheduler unit tests: transport + the true two-evaluation Heun reverse step.

Asserts the math: the transport
``z = t·x + (1−t)·e``; the predictor's step-start denominator unclamped; the
corrector's endpoint denominator clamped at ``t_eps``; and that a Heun rollout
evaluates the model twice per non-final step (the defining property of a true
Heun, ADR-0002 — not a single-call multistep scheme).
"""

from __future__ import annotations

import math

import torch

from manifold import FlowMatchHeunDiscreteScheduler


def test_add_noise_is_transport_scalar_and_batch():
    sched = FlowMatchHeunDiscreteScheduler()
    x = torch.randn(2, 4, 4, 4, 4)
    e = torch.randn(2, 4, 4, 4, 4)
    # Scalar t: pure transport.
    assert torch.allclose(sched.add_noise(x, e, 0.0), e)
    assert torch.allclose(sched.add_noise(x, e, 1.0), x)
    assert torch.allclose(sched.add_noise(x, e, 0.25), 0.25 * x + 0.75 * e)
    # Per-sample (B,) t: each sample interpolated by its own t.
    t = torch.tensor([0.0, 1.0])
    out = sched.add_noise(x, e, t)
    assert torch.allclose(out[0], e[0])
    assert torch.allclose(out[1], x[1])


def test_set_timesteps_grid_is_zero_to_one():
    sched = FlowMatchHeunDiscreteScheduler()
    nodes = sched.set_timesteps(4)
    assert nodes.shape == (5,)
    assert nodes[0].item() == 0.0
    assert nodes[-1].item() == 1.0
    assert sched.timesteps is nodes


def test_prediction_type_is_sample():
    assert FlowMatchHeunDiscreteScheduler().prediction_type == "sample"


def test_euler_step_formula_unclamped_denominator():
    """v1 = (x0 − z) / (1 − t) [unclamped]; z_euler = z + v1·(t_next − t)."""
    sched = FlowMatchHeunDiscreteScheduler()
    z = torch.randn(1, 4, 4, 4, 4)
    x0 = torch.randn(1, 4, 4, 4, 4)
    t, t_next = 0.3, 0.55
    z_euler, v1 = sched.euler_step(x0, z, t, t_next)
    expected_v1 = (x0 - z) / (1.0 - t)
    expected_z = z + expected_v1 * (t_next - t)
    assert torch.allclose(v1, expected_v1)
    assert torch.allclose(z_euler, expected_z)


def test_heun_correct_formula_clamped_denominator():
    """v2 = (x0 − z_euler) / max(1 − t_next, t_eps); trapezoidal average."""
    sched = FlowMatchHeunDiscreteScheduler(t_eps=0.05)
    z = torch.randn(1, 4, 4, 4, 4)
    z_euler = torch.randn(1, 4, 4, 4, 4)
    x0_2 = torch.randn(1, 4, 4, 4, 4)
    v1 = torch.randn(1, 4, 4, 4, 4)
    t, t_next = 0.7, 1.0  # t_next = 1 exercises the t_eps clamp
    out = sched.heun_correct(x0_2, z, z_euler, v1, t, t_next)
    denom = max(1.0 - t_next, 0.05)
    v2 = (x0_2 - z_euler) / denom
    expected = z + 0.5 * (v1 + v2) * (t_next - t)
    assert torch.allclose(out, expected)


def test_euler_step_per_sample_matches_scalar_loop():
    """A (B,) t divides each sample by its own (1 − t) — equals per-sample scalar.

    The per-sample generalization (ADR-0008) must equal the scalar formula applied
    sample-by-sample; this is the load-bearing parity guard for the reverse step.
    """
    sched = FlowMatchHeunDiscreteScheduler()
    z = torch.randn(3, 4, 4, 4, 4)
    x0 = torch.randn(3, 4, 4, 4, 4)
    t = torch.tensor([0.2, 0.5, 0.8])
    t_next = torch.tensor([0.45, 0.9, 1.0])  # last sample hits the endpoint clamp
    z_euler_b, v1_b = sched.euler_step(x0, z, t, t_next)
    # Scalar formula, one sample at a time:
    for b in range(3):
        v1 = (x0[b] - z[b]) / (1.0 - float(t[b]))
        z_e = z[b] + v1 * (float(t_next[b]) - float(t[b]))
        assert torch.allclose(v1_b[b], v1)
        assert torch.allclose(z_euler_b[b], z_e)


def test_heun_correct_per_sample_matches_scalar_loop():
    """The endpoint clamp is elementwise over the (B,) batch (ADR-0008)."""
    sched = FlowMatchHeunDiscreteScheduler(t_eps=0.05)
    z = torch.randn(3, 4, 4, 4, 4)
    z_euler = torch.randn(3, 4, 4, 4, 4)
    x0_2 = torch.randn(3, 4, 4, 4, 4)
    v1 = torch.randn(3, 4, 4, 4, 4)
    t = torch.tensor([0.2, 0.5, 0.8])
    t_next = torch.tensor([0.45, 0.9, 1.0])  # last sample's 1 − t_next < t_eps
    out_b = sched.heun_correct(x0_2, z, z_euler, v1, t, t_next)
    for b in range(3):
        denom = max(1.0 - float(t_next[b]), 0.05)
        v2 = (x0_2[b] - z_euler[b]) / denom
        expected = z[b] + 0.5 * (v1[b] + v2) * (float(t_next[b]) - float(t[b]))
        assert torch.allclose(out_b[b], expected)


def test_scalar_path_byte_identical_to_pre_change():
    """The scalar (python-float) fast path is unchanged by the (B,) generalization.

    Pre-#41 the steps divided by a python ``float`` denom/dt; a scalar ``t`` still
    does (``_step_t`` returns a float), so the outputs are bit-identical to the old
    arithmetic — the load-bearing parity guard for JiT train/inference (ADR-0001/2).
    """
    sched = FlowMatchHeunDiscreteScheduler(t_eps=0.05)
    torch.manual_seed(1)
    z, x0 = torch.randn(1, 4, 4, 4, 4), torch.randn(1, 4, 4, 4, 4)
    z_euler = torch.randn(1, 4, 4, 4, 4)
    v1 = torch.randn(1, 4, 4, 4, 4)
    t, t_next = 0.3, 1.0  # t_next = 1 exercises the clamp on the scalar path too

    z_e, v1_out = sched.euler_step(x0, z, t, t_next)
    out = sched.heun_correct(x0, z, z_euler, v1, t, t_next)

    # The literal pre-#41 arithmetic (python-float denom + dt).
    denom = 1.0 - float(t)
    v1_old = (x0.float() - z.float()) / denom
    z_e_old = z.float() + v1_old * (float(t_next) - float(t))
    denom_c = max(1.0 - float(t_next), 0.05)
    v2_old = (x0.float() - z_euler.float()) / denom_c
    out_old = z.float() + 0.5 * (v1 + v2_old) * (float(t_next) - float(t))

    assert torch.equal(v1_out, v1_old)
    assert torch.equal(z_e, z_e_old.to(z_e.dtype))
    assert torch.equal(out, out_old.to(out.dtype))


def test_true_heun_evaluates_model_twice_per_nonfinal_step(unet, vae):
    """A true Heun needs the model at z_t AND at the Euler-advanced point.

    For ``n`` steps that is ``2·(n−1) + 1`` evaluations (the final step is Euler,
    a single eval, since the endpoint t=1 is singular). A single-call multistep
    scheme would use ``n`` — this test rules that out (ADR-0002).
    """
    from manifold import LatentFlowPipeline

    counts = {"n": 0}
    real_forward = unet.forward

    def counting_forward(*args, **kwargs):
        counts["n"] += 1
        return real_forward(*args, **kwargs)

    unet.forward = counting_forward  # type: ignore[method-assign]
    try:
        pipe = LatentFlowPipeline(unet, vae, FlowMatchHeunDiscreteScheduler())
        n = 5
        pipe(
            (1, 4, 4, 4, 4),
            spacing=[1.0, 1.0, 1.0],
            modality=1,
            num_inference_steps=n,
            generator=torch.Generator().manual_seed(0),
        )
    finally:
        unet.forward = real_forward  # type: ignore[method-assign]
    assert counts["n"] == 2 * (n - 1) + 1


# -- FlowMatchGRPOScheduler: inherited transport/Heun + sde_step_mean (#56) ----


def _grpo(eta: float = 0.7, n: int = 8):
    """A FlowMatchGRPOScheduler with its grid set (the 1/n noise-end floor needs it)."""
    from manifold import FlowMatchGRPOScheduler

    s = FlowMatchGRPOScheduler(eta=eta)
    s.set_timesteps(n)
    return s


def test_grpo_scheduler_subclasses_and_inherits_verbatim():
    """GRPO scheduler IS-A Heun scheduler; transport + Heun are inherited, never forked.

    The ADR-0008 / ADR-0011 invariant: ``add_noise`` / ``euler_step`` / ``heun_correct``
    are the parent's methods (a fork would feed the policy out-of-distribution noise and
    break the single source of truth). ``sde_step_mean`` is the only addition.
    """
    from manifold import FlowMatchGRPOScheduler

    assert issubclass(FlowMatchGRPOScheduler, FlowMatchHeunDiscreteScheduler)
    s = FlowMatchGRPOScheduler()
    assert FlowMatchGRPOScheduler.add_noise is FlowMatchHeunDiscreteScheduler.add_noise
    assert FlowMatchGRPOScheduler.euler_step is FlowMatchHeunDiscreteScheduler.euler_step
    assert FlowMatchGRPOScheduler.heun_correct is FlowMatchHeunDiscreteScheduler.heun_correct
    assert hasattr(s, "sde_step_mean")  # the one addition
    # The inherited transport still works unchanged.
    x = torch.randn(2, 4, 4, 4, 4)
    e = torch.randn(2, 4, 4, 4, 4)
    assert torch.allclose(s.add_noise(x, e, 0.25), 0.25 * x + 0.75 * e)


def test_grpo_scheduler_set_timesteps_is_full_zero_to_one_anchor_grid():
    """set_timesteps is the JiT anchor grid linspace(0,1,n+1) — not the partial per-sample grid."""
    s = _grpo(n=4)
    assert torch.equal(s.timesteps, torch.linspace(0.0, 1.0, 5))
    assert s.num_inference_steps == 4  # the 1/n noise-end floor


def test_grpo_scheduler_sde_mean_eta_zero_equals_euler_mean():
    """η→0: the Langevin term (σ²/2t)·x_θ vanishes → SDE mean == euler_step's z_euler.

    The parity guard (#56): sde_step_mean must reduce to the deterministic Euler
    advance as η→0, so a η=0 rollout is the deterministic Heun anchor.
    """
    s = _grpo(eta=0.0)
    z = torch.randn(2, 4, 4, 4, 4)
    x0 = torch.randn(2, 4, 4, 4, 4)
    t, t_next = 0.3, 0.55
    z_euler, _ = s.euler_step(x0, z, t, t_next)
    mean, std = s.sde_step_mean(x0, z, t, t_next)
    assert torch.allclose(mean, z_euler)
    assert float(std) == 0.0  # σ_t ∝ η → 0


def test_grpo_scheduler_sde_mean_and_std_formula():
    """mean = z + Δt·(v_θ + (σ²/2t)·x_θ); std = σ_t·√Δt (the equimarginal reverse-SDE)."""
    eta = 0.7
    s = _grpo(eta=eta, n=8)  # 1/n = 0.125
    z = torch.randn(1, 4, 4, 4, 4)
    x0 = torch.randn(1, 4, 4, 4, 4)
    t, t_next = 0.4, 0.5  # both > 1/n → no clamp, isolates the formula
    mean, std = s.sde_step_mean(x0, z, t, t_next)
    dt = t_next - t
    sigma = eta * math.sqrt((1.0 - t) / t)
    v1 = (x0 - z) / (1.0 - t)
    expected_mean = z + dt * (v1 + (sigma ** 2) / (2.0 * t) * x0)
    assert torch.allclose(mean, expected_mean, atol=1e-6)
    assert abs(float(std) - sigma * math.sqrt(dt)) < 1e-6


def test_grpo_scheduler_noise_end_clamp_is_finite():
    """At t=0 the σ_t blowup is clamped by t_safe=max(t,1/n) — finite, not inf/NaN.

    The clamp mirrors Granular's σ==1 → σ_max (#56 endpoint-stability guard). With
    n=4 the floor is 1/4, so t=0 evaluates σ at t_safe=0.25.
    """
    eta = 0.7
    s = _grpo(eta=eta, n=4)  # floor 0.25
    z = torch.randn(1, 4, 4, 4, 4)
    x0 = torch.randn(1, 4, 4, 4, 4)
    mean, std = s.sde_step_mean(x0, z, 0.0, 0.25)  # t=0 → clamped to 0.25
    assert torch.isfinite(mean).all()
    assert math.isfinite(float(std))
    sigma_at_floor = eta * math.sqrt((1.0 - 0.25) / 0.25)
    assert abs(float(std) - sigma_at_floor * math.sqrt(0.25)) < 1e-6


def test_grpo_scheduler_sde_mean_per_sample_matches_scalar_loop():
    """A (B,) t gives per-sample mean/std equal to the scalar formula sample-by-sample.

    The (B,) parity guard (mirrors euler_step's): each sample advances by its own
    dt and divides by its own (1−t) / clamps its own t_safe.
    """
    eta = 0.6
    s = _grpo(eta=eta, n=10)  # floor 0.1; all t's below > floor
    z = torch.randn(3, 4, 4, 4, 4)
    x0 = torch.randn(3, 4, 4, 4, 4)
    t = torch.tensor([0.2, 0.5, 0.8])
    t_next = torch.tensor([0.3, 0.6, 0.9])
    mean_b, std_b = s.sde_step_mean(x0, z, t, t_next)
    for b in range(3):
        m_s, std_s = s.sde_step_mean(x0[b : b + 1], z[b : b + 1], float(t[b]), float(t_next[b]))
        assert torch.allclose(mean_b[b], m_s[0], atol=1e-6)
        assert abs(float(std_b[b]) - float(std_s)) < 1e-6


def test_grpo_scheduler_sde_mean_requires_set_timesteps():
    """sde_step_mean without set_timesteps raises clearly (the 1/n floor needs the grid)."""
    from manifold import FlowMatchGRPOScheduler

    s = FlowMatchGRPOScheduler(eta=0.7)
    z = torch.randn(1, 4, 4, 4, 4)
    x0 = torch.randn(1, 4, 4, 4, 4)
    with __import__("pytest").raises(RuntimeError, match="set_timesteps"):
        s.sde_step_mean(x0, z, 0.3, 0.5)
