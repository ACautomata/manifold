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


# -- Slice A (issue #24): the shared x0 Heun rollout -------------------------
# Module.sample == Pipeline.sample_latent under the same generator (parity), and
# the Module's generation never imports/instantiates the inference Pipeline.


def test_module_sample_equals_pipeline_sample_latent(unet, scheduler):
    """Module.sample and Pipeline.sample_latent delegate to one primitive (ADR-0005).

    Sharing the SAME unet + scheduler between the module and the pipeline, the
    two generation paths produce a bit-identical latent under the same generator.
    """
    from manifold import AutoencoderKL, LatentFlowPipeline, LatentFlowModule

    vae = AutoencoderKL(scaling_factor=0.5)  # the decode half; not used by sample_latent
    module = LatentFlowModule(unet, scheduler)
    pipeline = LatentFlowPipeline(unet, vae, scheduler)  # shares unet + scheduler

    args = dict(
        target_shape=(1, 4, 4, 4, 4),
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=4,
        guidance_scale=1.5,
        cfg_interval=(0.1, 1.0),
    )
    latent_module = module.sample(generator=torch.Generator().manual_seed(7), **args)
    latent_pipe = pipeline.sample_latent(generator=torch.Generator().manual_seed(7), **args)
    assert latent_module.shape == (1, 4, 4, 4, 4)
    assert torch.equal(latent_module, latent_pipe)


def test_module_sample_finite_and_no_cfg_path(unet, scheduler):
    """Module.sample returns a finite latent; the no-CFG path (scale=1.0) runs."""
    from manifold import LatentFlowModule

    module = LatentFlowModule(unet, scheduler)
    latent = module.sample(
        (1, 4, 4, 4, 4),
        spacing=[1.0, 1.0, 1.0],
        modality=1,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(0),
    )
    assert latent.shape == (1, 4, 4, 4, 4)
    assert torch.isfinite(latent).all()


def test_module_sample_never_reaches_pipeline(unet, scheduler):
    """In-training generation never imports or instantiates LatentFlowPipeline (ADR-0005)."""
    import sys

    from manifold import LatentFlowModule

    module = LatentFlowModule(unet, scheduler)
    # Drop any cached pipeline module so the assertion below is meaningful.
    sys.modules.pop("manifold.pipelines.latent_flow", None)
    module.sample(
        (1, 4, 4, 4, 4),
        spacing=[1.0, 1.0, 1.0],
        modality=0,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert "manifold.pipelines.latent_flow" not in sys.modules


# -- Slice B (issue #25): optimizer + grad-norm + LR-schedule wiring ----------


def test_configure_optimizers_is_adam_plus_cosine_step(unet_trainable):
    """Adam over all UNet params + a step-interval cosine-with-warmup scheduler."""
    from manifold import LatentFlowModule

    module = LatentFlowModule(
        unet_trainable,
        FlowMatchHeunDiscreteScheduler(),
        lr=1e-4,
        lr_warmup_steps=2,
        num_train_examples=8,
        train_batch_size=2,
        n_epochs=3,
    )
    config = module.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.Adam)
    # All UNet params are in the optimizer's param groups.
    opt_params = {p for group in config["optimizer"].param_groups for p in group["params"]}
    assert opt_params == set(unet_trainable.parameters())
    sched_cfg = config["lr_scheduler"]
    assert sched_cfg["interval"] == "step"
    # 8 examples / (2 batch * 1 world) = 4 steps/epoch * 3 epochs = 12 total.
    assert module._total_optimizer_steps() == 12


def test_optimizer_step_descends_loss_and_updates_params(unet_trainable, monkeypatch):
    """A manual forward -> backward -> step reduces loss and updates UNet params.

    ``t`` (and the add-noise RNG) are pinned so two forwards at the same weights
    are comparable — otherwise the logit-normal ``t`` resample makes loss0/loss1
    measure different points and a one-step descent is unobservable.
    """
    from manifold import LatentFlowModule

    fresh = LatentFlowModule(
        unet_trainable, FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0
    )
    fresh._sample_timesteps = lambda bs, dev: torch.full(  # pin t
        (bs,), 0.5, device=dev
    )
    optimizer = fresh.configure_optimizers()["optimizer"]
    batch = {
        "latent": torch.randn(1, 4, 4, 4, 4),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
        "label": torch.tensor([2]),
    }

    def loss_at():
        torch.manual_seed(123)  # fix the add_noise sample so forwards are comparable
        return fresh(batch, "fit")["loss"].item()

    before = [p.detach().clone() for p in unet_trainable.parameters()]
    loss0 = loss_at()
    for _ in range(5):
        torch.manual_seed(123)
        loss = fresh(batch, "fit")["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss1 = loss_at()

    after = list(unet_trainable.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
    assert loss1 < loss0


def test_grad_norm_hook_stashes_amp_corrected_value(module, batch):
    """after_manual_backward stashes the grad norm; off-GPU scale is 1.0."""
    out = module(batch, "fit")
    out["loss"].backward()
    assert module._last_grad_norm is None  # not set until the hook runs
    module.after_manual_backward()

    expected = torch.sqrt(
        sum((p.grad.detach().float() ** 2).sum() for p in module.unet.parameters())
    )
    assert module._last_grad_norm is not None
    assert module._last_grad_norm == pytest.approx(float(expected))
    # No Trainer / off-GPU → the AMP scale is exactly 1.0.
    assert module._amp_scale() == 1.0


