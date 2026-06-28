"""Training module tests (Seam 2, issue #6).

``LatentFlowModule.forward(batch, "fit")`` returns a finite scalar loss whose
``.backward()`` updates UNet parameters; the noised latent comes from
``scheduler.add_noise`` (not a re-derived transport); the loss is the
``(1 − t)⁻²``-weighted MSE; and the module holds no ``scale_factor``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from manifold import (
    FlowMatchHeunDiscreteScheduler,
    LatentFlowModule,
    UNet3DConditionModel,
)


def _trainable_unet() -> UNet3DConditionModel:
    """A tiny UNet with MAISI's zero-init output conv re-initialized.

    MONAI MAISI zero-initializes the final output projection (standard for
    diffusion models), so at init the output is identically zero and only that
    projection receives gradient. Re-initializing the all-zero parameters lets
    the test exercise the FULL backprop path (all 115 params) — verifying the
    module wires the loss to the UNet, not just the last layer.
    """
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in unet.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return unet


@pytest.fixture
def unet_trainable():
    return _trainable_unet()


@pytest.fixture
def module(unet_trainable):
    return LatentFlowModule(unet_trainable, FlowMatchHeunDiscreteScheduler())


@pytest.fixture
def batch():
    # Latent is already scaled (ADR-0003); the module never touches scale_factor.
    return {
        "latent": torch.randn(1, 4, 4, 4, 4),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
        "label": torch.tensor([2]),
    }


def test_forward_returns_finite_scalar_loss(module, batch):
    out = module(batch, "fit")
    assert set(out.keys()) == {"loss"}
    assert out["loss"].dim() == 0  # scalar
    assert torch.isfinite(out["loss"])


def test_validation_stage_exposes_pred_and_target(module, batch):
    out = module(batch, "validate")
    assert {"loss", "pred", "target"} <= set(out.keys())
    assert out["pred"].shape == batch["latent"].shape
    assert out["target"].shape == batch["latent"].shape


def test_backward_updates_unet_parameters(module, batch, unet_trainable):
    out = module(batch, "fit")
    out["loss"].backward()
    grads = [p.grad for p in unet_trainable.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    # The full UNet is in the graph — not just the output projection.
    with_grad = sum(
        1 for p in unet_trainable.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for _ in unet_trainable.parameters())
    assert with_grad > total // 2

    before = [p.detach().clone() for p in unet_trainable.parameters()]
    torch.optim.SGD(unet_trainable.parameters(), lr=0.1).step()
    after = list(unet_trainable.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_module_uses_scheduler_add_noise(module, batch, monkeypatch):
    """The noised latent must come from scheduler.add_noise, not a re-derived transport (ADR-0001)."""
    called = {"n": 0}
    real = module.scheduler.add_noise

    def spy(*args, **kwargs):
        called["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module.scheduler, "add_noise", spy)
    module(batch, "fit")
    assert called["n"] == 1


def test_module_holds_no_scale_factor(module):
    assert not hasattr(module, "scale_factor")
    assert "scale_factor" not in dict(module.named_buffers())


class _ZeroUNet(nn.Module):
    """UNet stand-in returning zeros, to pin the loss formula independent of the network."""

    def forward(self, sample, timestep, spacing, class_labels=None, context=None):
        return torch.zeros_like(sample)


def test_loss_is_inverse_t_weighted_mse(monkeypatch):
    """loss = mean(((latent − x0_pred) / max(1 − t, t_eps))²) with t logit-normal.

    With x0_pred ≡ 0 this reduces to ``mean((latent / max(1 − t, t_eps))²)``,
    evaluated here at a fixed ``t`` and fixed noise so the formula is exact.
    """
    unet = _ZeroUNet()
    scheduler = FlowMatchHeunDiscreteScheduler(t_eps=0.1)
    module = LatentFlowModule(unet, scheduler, t_eps=0.1)
    module._sample_timesteps = lambda batch_size, device: torch.full(  # noqa: ANN001
        (batch_size,), 0.25, device=device
    )

    torch.manual_seed(0)
    latent = torch.randn(1, 4, 4, 4, 4)
    batch = {"latent": latent, "spacing": torch.tensor([1.0, 1.0, 1.0]), "label": torch.tensor([2])}
    loss = module(batch, "fit")["loss"]

    t = 0.25
    weight = max(1.0 - t, 0.1)
    expected = ((latent.float() - 0.0) / weight).pow(2).mean()
    assert torch.allclose(loss, expected)
