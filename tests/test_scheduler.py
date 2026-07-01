"""Scheduler unit tests: transport + the true two-evaluation Heun reverse step.

Asserts the math: the transport
``z = t·x + (1−t)·e``; the predictor's step-start denominator unclamped; the
corrector's endpoint denominator clamped at ``t_eps``; and that a Heun rollout
evaluates the model twice per non-final step (the defining property of a true
Heun, ADR-0002 — not a single-call multistep scheme).
"""

from __future__ import annotations

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
