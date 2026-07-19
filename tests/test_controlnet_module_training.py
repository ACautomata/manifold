"""ControlNet supervised module training tests (issue #130 / ADR-0027).

The supervised-module seam (mirrors ``tests/test_module_training.py``):
``ControlNetLatentFlowModule.forward(batch, "fit")`` returns a finite scalar loss
whose ``.backward()`` reaches the **ControlNet** params and NOT the frozen base; one
training step drives the loss down; the loss is the ``(1 − t)⁻²``-weighted
velocity-MSE (with an optional L1 knob); the optimizer wires only the ControlNet;
``sample()`` delegates to the shared ``controlnet_rollout`` primitive. Raw arm — no
EMA (ADR-0006).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from manifold import (
    ControlNet3DConditionModel,
    FlowMatchHeunDiscreteScheduler,
    UNet3DConditionModel,
)
from manifold.modules import ControlNetLatentFlowModule


def _frozen_base() -> UNet3DConditionModel:
    """A tiny base UNet with the zero-init output conv re-initialized.

    MONAI MAISI zero-initializes the final output projection, so at init the base
    output is identically zero and the ControlNet's residual-injection effect on the
    output is masked. Re-initializing the all-zero ``out`` params (emulating a
    warm-started / pretrained base) lets the test exercise the full
    base-output→ControlNet backprop path.
    """
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in unet.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return unet


def _controlnet(base: UNet3DConditionModel) -> ControlNet3DConditionModel:
    torch.manual_seed(1)
    controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    controlnet.load_base_encoder_weights(base)
    return controlnet


@pytest.fixture
def base():
    return _frozen_base()


@pytest.fixture
def module(base):
    controlnet = _controlnet(base)
    return ControlNetLatentFlowModule(base, controlnet, FlowMatchHeunDiscreteScheduler())


@pytest.fixture
def batch():
    # Latents are already scaled (ADR-0003); the module never touches scale_factor.
    return {
        "src_latent": torch.randn(2, 4, 8, 8, 4),
        "tgt_latent": torch.randn(2, 4, 8, 8, 4),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
        "src_label": torch.tensor([1, 0]),
        "tgt_label": torch.tensor([2, 3]),
    }


def test_forward_returns_finite_scalar_loss(module, batch):
    out = module(batch, "fit")
    assert set(out.keys()) == {"loss"}
    assert out["loss"].dim() == 0
    assert torch.isfinite(out["loss"])


def test_validation_stage_exposes_pred_and_target(module, batch):
    out = module(batch, "validate")
    assert {"loss", "pred", "target"} <= set(out.keys())
    assert out["pred"].shape == batch["tgt_latent"].shape
    assert out["target"].shape == batch["tgt_latent"].shape


def test_base_is_frozen_and_unregistered(module, base):
    """The base is off the optimizer/checkpoint; the ControlNet is the only arm."""
    opt_params = {id(p) for p in module.parameters()}
    assert not ({id(p) for p in base.parameters()} & opt_params)
    assert {id(p) for p in module.controlnet.parameters()} <= opt_params
    assert not any(k.startswith("unet.") for k in module.state_dict())
    assert any("controlnet" in k for k in module.state_dict())


def test_backward_routes_grad_to_controlnet_not_base(module, batch, base):
    """loss through the base output reaches the ControlNet, not the frozen base."""
    out = module(batch, "fit")
    out["loss"].backward()
    cn_grads = [p.grad for p in module.controlnet.parameters() if p.grad is not None]
    assert cn_grads and any(g.abs().sum() > 0 for g in cn_grads)
    base_grads = [p.grad for p in base.parameters() if p.grad is not None]
    assert all(g.abs().sum() == 0 for g in base_grads)


def test_module_uses_scheduler_add_noise(module, batch, monkeypatch):
    """The noised latent must come from scheduler.add_noise (ADR-0001)."""
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


def test_configure_optimizers_is_adam_over_controlnet_only(module, base):
    """Adam over the ControlNet params only + a step-interval cosine-with-warmup."""
    config = module.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.Adam)
    opt_params = {p for group in config["optimizer"].param_groups for p in group["params"]}
    assert opt_params == set(module.controlnet.parameters())
    assert not (opt_params & set(base.parameters()))
    assert config["lr_scheduler"]["interval"] == "step"


def test_optimizer_step_descends_loss_and_updates_controlnet(base, batch):
    """A manual forward → backward → step reduces loss and updates ControlNet params.

    ``t`` and the add-noise RNG are pinned so two forwards at the same weights are
    comparable (a one-step descent is otherwise unobservable under resampled ``t``).
    """
    controlnet = _controlnet(base)
    fresh = ControlNetLatentFlowModule(
        base, controlnet, FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0
    )
    fresh._sample_timesteps = lambda bs, dev: torch.full((bs,), 0.5, device=dev)  # pin t
    optimizer = fresh.configure_optimizers()["optimizer"]

    def loss_at():
        torch.manual_seed(123)  # fix the add_noise sample so forwards are comparable
        return fresh(batch, "fit")["loss"].item()

    before = [p.detach().clone() for p in controlnet.parameters()]
    loss0 = loss_at()
    for _ in range(5):
        torch.manual_seed(123)
        loss = fresh(batch, "fit")["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss1 = loss_at()

    after = list(controlnet.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert loss1 < loss0
    # The frozen base never moved.
    assert all(p.grad is None for p in base.parameters())


def test_optional_l1_knob_changes_loss(base, batch):
    """The optional L1 term (default off) adds to the loss when ``l1_weight > 0``."""
    controlnet = _controlnet(base)
    plain = ControlNetLatentFlowModule(base, controlnet, FlowMatchHeunDiscreteScheduler())
    plain._sample_timesteps = lambda bs, dev: torch.full((bs,), 0.5, device=dev)

    controlnet2 = _controlnet(base)
    with_l1 = ControlNetLatentFlowModule(
        base, controlnet2, FlowMatchHeunDiscreteScheduler(), l1_weight=0.5
    )
    with_l1._sample_timesteps = lambda bs, dev: torch.full((bs,), 0.5, device=dev)

    torch.manual_seed(7)
    l_plain = plain(batch, "fit")["loss"].item()
    torch.manual_seed(7)
    l_l1 = with_l1(batch, "fit")["loss"].item()
    assert l_l1 > l_plain  # the L1 term is nonnegative and added


def test_grad_norm_hook_stashes_amp_corrected_value(module, batch):
    """after_manual_backward stashes the ControlNet grad norm; off-GPU scale is 1.0."""
    out = module(batch, "fit")
    out["loss"].backward()
    assert module._last_grad_norm is None
    module.after_manual_backward()

    expected = torch.sqrt(
        sum((p.grad.detach().float() ** 2).sum() for p in module.controlnet.parameters())
    )
    assert module._last_grad_norm is not None
    assert module._last_grad_norm == pytest.approx(float(expected))
    assert module._amp_scale() == 1.0


# -- sample() delegates to the shared controlnet_rollout primitive (ADR-0005) --


def test_module_sample_equals_pipeline_rollout(base):
    """Module.sample and the pipeline share one rollout primitive (bit-identical)."""
    from manifold import AutoencoderKL, ControlNetLatentFlowPipeline

    controlnet = _controlnet(base)
    scheduler = FlowMatchHeunDiscreteScheduler()
    module = ControlNetLatentFlowModule(base, controlnet, scheduler)
    vae = AutoencoderKL(scaling_factor=0.5)
    pipeline = ControlNetLatentFlowPipeline(base, controlnet, vae, scheduler)

    noise = torch.randn(1, 4, 8, 8, 4)
    src = torch.randn(1, 4, 8, 8, 4)
    spacing = [1.0, 1.0, 1.0]
    latent_module = module.sample(noise, src, spacing, 1, 2, num_inference_steps=4)
    latent_pipe = pipeline.sample_latent(noise, src, spacing, 1, 2, num_inference_steps=4)
    assert torch.equal(latent_module, latent_pipe)
