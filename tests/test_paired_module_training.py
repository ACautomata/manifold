"""Paired JiT Module tests (Seam 1, issue #66).

``PairedLatentFlowModule.forward(batch, "fit")`` returns a finite scalar loss
whose ``.backward()`` updates UNet parameters; the interpolated latent comes from
``scheduler.add_noise(x_tgt, x_src, t)`` (the transport interpolates: ``z`` at
``t=0`` ≡ ``x_src``, at ``t=1`` ≡ ``x_tgt`` — ADR-0013); the UNet input is
``concat([z_t, x_src])`` with ``in_channels = 2·C_latent`` (ADR-0014); the
conditioning is the summed ``embed(src)+embed(tgt)`` label; and the loss is the
``(1 − t)⁻²``-weighted x0-MSE on ``x_tgt``, →0 when the UNet predicts ``x_tgt``
exactly. Mock UNets pin the formula independent of MONAI; a real trainable UNet
exercises the full backprop path.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from manifold import (
    FlowMatchHeunDiscreteScheduler,
    PairedLatentFlowModule,
    UNet3DConditionModel,
)

#: The latent channel count ``C_latent`` of the tiny fixtures (matches the VAE).
C_LATENT = 4


def _trainable_paired_unet() -> UNet3DConditionModel:
    """A tiny paired UNet (``in_channels = 2·C_latent``) with MAISI's zero-init
    output conv re-initialized.

    MONAI MAISI zero-initializes the final output projection (standard for
    diffusion models), so at init the output is identically zero and only that
    projection receives gradient. Re-initializing the all-zero parameters lets the
    test exercise the FULL backprop path — verifying the module wires the loss to
    the UNet, not just the last layer.
    """
    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    for p in unet.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return unet


@pytest.fixture
def paired_unet_trainable() -> UNet3DConditionModel:
    return _trainable_paired_unet()


@pytest.fixture
def paired_module(paired_unet_trainable, paired_scheduler) -> PairedLatentFlowModule:
    return PairedLatentFlowModule(paired_unet_trainable, paired_scheduler)


@pytest.fixture
def paired_batch():
    # Both latents already scaled (ADR-0003); the module never touches scale_factor.
    return {
        "src_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "tgt_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }


def test_forward_returns_finite_scalar_loss(paired_module, paired_batch):
    out = paired_module(paired_batch, "fit")
    assert set(out.keys()) == {"loss"}
    assert out["loss"].dim() == 0  # scalar
    assert torch.isfinite(out["loss"])


def test_validation_stage_exposes_pred_and_target(paired_module, paired_batch):
    out = paired_module(paired_batch, "validate")
    assert {"loss", "pred", "target"} <= set(out.keys())
    assert out["pred"].shape == paired_batch["tgt_latent"].shape
    assert out["target"].shape == paired_batch["tgt_latent"].shape


def test_paired_transport_interpolates_endpoints(paired_scheduler):
    """z_t = t·x_tgt + (1−t)·x_src: z at t=0 ≡ x_src, z at t=1 ≡ x_tgt (ADR-0013)."""
    x_src = torch.randn(1, C_LATENT, 4, 4, 4)
    x_tgt = torch.randn(1, C_LATENT, 4, 4, 4)
    z0 = paired_scheduler.add_noise(x_tgt, x_src, torch.tensor(0.0))
    z1 = paired_scheduler.add_noise(x_tgt, x_src, torch.tensor(1.0))
    assert torch.allclose(z0, x_src)
    assert torch.allclose(z1, x_tgt)


def test_module_uses_scheduler_add_noise(paired_module, paired_batch, monkeypatch):
    """The interpolated latent must come from scheduler.add_noise, not a re-derived
    transport (ADR-0001 — single source of truth, shared with the noise→data JiT)."""
    called = {"n": 0}
    real = paired_module.scheduler.add_noise

    def spy(*args, **kwargs):
        called["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(paired_module.scheduler, "add_noise", spy)
    paired_module(paired_batch, "fit")
    assert called["n"] == 1


def test_unet_input_is_concat_of_zt_and_x_src(paired_module, paired_batch, monkeypatch):
    """The UNet sees concat([z_t, x_src]) along channels → in_channels = 2·C_latent
    (ADR-0014). Spies on the UNet forward to pin the input channel count."""
    seen = {}

    real_forward = paired_module.unet.forward

    def spy(sample, *args, **kwargs):
        seen["in_channels"] = sample.shape[1]
        return real_forward(sample, *args, **kwargs)

    monkeypatch.setattr(paired_module.unet, "forward", spy)
    paired_module(paired_batch, "fit")
    assert seen["in_channels"] == 2 * C_LATENT


def test_module_holds_no_scale_factor(paired_module):
    """The module owns no scale_factor; the VAE does (ADR-0003)."""
    assert not hasattr(paired_module, "scale_factor")
    assert "scale_factor" not in dict(paired_module.named_buffers())


class _ZeroPairedUNet(nn.Module):
    """UNet stand-in returning zeros, to pin the loss formula independent of the net.

    Accepts both the noise→data (``class_labels``) and Paired JiT
    (``class_labels_src``/``class_labels_tgt``) signatures.
    """

    def __init__(self, out_channels: int = C_LATENT):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, sample, timestep, spacing, class_labels=None, context=None, *,
                class_labels_src=None, class_labels_tgt=None):
        return torch.zeros(sample.shape[0], self.out_channels, *sample.shape[2:])


def test_loss_is_inverse_t_weighted_mse_on_tgt(monkeypatch):
    """loss = mean(((x_tgt − x0_pred) / max(1 − t, t_eps))²) with t logit-normal.

    With x0_pred ≡ 0 this reduces to ``mean((x_tgt / max(1 − t, t_eps))²)``,
    evaluated here at a fixed ``t`` and fixed src/tgt so the formula is exact.
    """
    unet = _ZeroPairedUNet()
    scheduler = FlowMatchHeunDiscreteScheduler(t_eps=0.1)
    module = PairedLatentFlowModule(unet, scheduler, t_eps=0.1)
    module._sample_timesteps = lambda batch_size, device: torch.full(  # pin t
        (batch_size,), 0.25, device=device
    )

    torch.manual_seed(0)
    x_src = torch.randn(1, C_LATENT, 4, 4, 4)
    x_tgt = torch.randn(1, C_LATENT, 4, 4, 4)
    batch = {
        "src_latent": x_src,
        "tgt_latent": x_tgt,
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }
    loss = module(batch, "fit")["loss"]

    weight = max(1.0 - 0.25, 0.1)
    expected = ((x_tgt.float() - 0.0) / weight).pow(2).mean()
    assert torch.allclose(loss, expected)


def test_loss_uniform_is_plain_mse_on_tgt():
    """loss_weight="uniform": loss = mean((x_tgt − x0_pred)²) — no (1−t)⁻² divisor.

    With x0_pred ≡ 0 this reduces to ``mean(x_tgt²)``, independent of ``t`` (the
    velocity weight is dropped). Pins the regime fix (ADR-0013/0014 addendum):
    the model is no longer free to satisfy high-``t`` by copying while starving
    the low-``t`` transport the rollout depends on.
    """
    unet = _ZeroPairedUNet()
    scheduler = FlowMatchHeunDiscreteScheduler(t_eps=0.1)
    module = PairedLatentFlowModule(unet, scheduler, t_eps=0.1, loss_weight="uniform")
    module._sample_timesteps = lambda batch_size, device: torch.full(  # pin t
        (batch_size,), 0.25, device=device
    )

    torch.manual_seed(0)
    x_src = torch.randn(1, C_LATENT, 4, 4, 4)
    x_tgt = torch.randn(1, C_LATENT, 4, 4, 4)
    batch = {
        "src_latent": x_src,
        "tgt_latent": x_tgt,
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }
    loss = module(batch, "fit")["loss"]
    expected = x_tgt.float().pow(2).mean()
    assert torch.allclose(loss, expected)


def test_invalid_loss_weight_raises():
    """An unknown loss_weight is rejected at construction (fails fast)."""
    unet = _ZeroPairedUNet()
    with pytest.raises(ValueError, match="loss_weight"):
        PairedLatentFlowModule(unet, FlowMatchHeunDiscreteScheduler(), loss_weight="bogus")


class _TargetUNet(nn.Module):
    """UNet stand-in that always predicts the target latent ``x_tgt`` exactly.

    Used to pin ``loss → 0`` (Seam 1) and the oracle rollout ``z → x_tgt``
    (Seam 2). Carries a dummy parameter so ``next(unet.parameters())`` resolves in
    the shared sampler primitive.
    """

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, sample, timestep, spacing, class_labels=None, context=None, *,
                class_labels_src=None, class_labels_tgt=None):
        return self.target


def test_loss_zero_when_unet_predicts_tgt_exactly():
    """When the UNet predicts x_tgt exactly, the (1−t)⁻²-weighted MSE is 0."""
    torch.manual_seed(0)
    x_src = torch.randn(1, C_LATENT, 4, 4, 4)
    x_tgt = torch.randn(1, C_LATENT, 4, 4, 4)
    unet = _TargetUNet(x_tgt)
    module = PairedLatentFlowModule(unet, FlowMatchHeunDiscreteScheduler())
    module._sample_timesteps = lambda batch_size, device: torch.full(
        (batch_size,), 0.5, device=device
    )
    batch = {
        "src_latent": x_src,
        "tgt_latent": x_tgt,
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }
    loss = module(batch, "fit")["loss"]
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_backward_updates_unet_parameters(paired_module, paired_batch, paired_unet_trainable):
    out = paired_module(paired_batch, "fit")
    out["loss"].backward()
    grads = [p.grad for p in paired_unet_trainable.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    # The full UNet is in the graph — not just the output projection.
    with_grad = sum(
        1 for p in paired_unet_trainable.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for _ in paired_unet_trainable.parameters())
    assert with_grad > total // 2


def test_backward_updates_unet_parameters_under_uniform_loss(paired_unet_trainable, paired_scheduler):
    """loss_weight="uniform" still backprops through the full UNet — the uniform
    branch (plain MSE, no (1−t)⁻² division) keeps x0_pred in the graph, so the
    regime fix is trainable end-to-end, not just evaluatable."""
    module = PairedLatentFlowModule(paired_unet_trainable, paired_scheduler, loss_weight="uniform")
    batch = {
        "src_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "tgt_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }
    module(batch, "fit")["loss"].backward()
    grads = [p.grad for p in paired_unet_trainable.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    with_grad = sum(
        1 for p in paired_unet_trainable.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total = sum(1 for _ in paired_unet_trainable.parameters())
    assert with_grad > total // 2


def test_swap_grad_reaches_class_embedding_rows(paired_unet_trainable, paired_scheduler):
    """The _PinnedClassEmbedding swap must not detach the autograd path: backward
    through the paired forward must populate ``class_embedding.weight`` at exactly
    the looked-up rows (src=0, tgt=1) and nowhere else.

    This pins the riskiest mechanism in the slice (ADR-0014 wiring): ``cond`` is
    computed from the real ``nn.Embedding`` rows before the swap and returned by the
    stand-in without ``detach()``, so gradients still reach the embedding table.
    """
    module = PairedLatentFlowModule(paired_unet_trainable, paired_scheduler)
    batch = {
        "src_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "tgt_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }
    paired_unet_trainable.zero_grad()
    module(batch, "fit")["loss"].backward()
    grad = paired_unet_trainable.unet.class_embedding.weight.grad
    assert grad is not None
    assert grad[0].abs().sum() > 0  # src row (looked up for cond)
    assert grad[1].abs().sum() > 0  # tgt row (looked up for cond)
    assert grad[2].abs().sum() == 0  # unreferenced row untouched
    assert grad[3].abs().sum() == 0  # unreferenced row untouched


def test_configure_optimizers_is_adam_over_unet(paired_unet_trainable, paired_scheduler):
    """Adam over all paired-UNet params (the cosine schedule + grad-norm hook land in Slice 4)."""
    module = PairedLatentFlowModule(paired_unet_trainable, paired_scheduler)
    config = module.configure_optimizers()
    assert isinstance(config["optimizer"], torch.optim.Adam)
    opt_params = {p for group in config["optimizer"].param_groups for p in group["params"]}
    assert opt_params == set(paired_unet_trainable.parameters())


def test_configure_optimizers_scales_peak_lr_paired(paired_unet_trainable, paired_scheduler):
    """Paired configure_optimizers sets Adam's LR to the scaled peak (mirrors LatentFlowModule)."""
    # train_batch_size=2, ref=8, world=1 → peak = base × √(2/8) = base × 0.5.
    module = PairedLatentFlowModule(
        paired_unet_trainable,
        paired_scheduler,
        lr=1e-4,
        lr_ref_batch_size=8,
        lr_scale_rule="sqrt",
        num_train_examples=8,
        train_batch_size=2,
    )
    optimizer = module.configure_optimizers()["optimizer"]
    # param_groups[0]["lr"] is the scheduled (post-warmup-ratio) lr; the peak Adam
    # was built with lives in optimizer.defaults["lr"].
    assert optimizer.defaults["lr"] == pytest.approx(5e-5)


def test_configure_optimizers_no_scaling_when_batch_unknown_ddp_paired(
    paired_unet_trainable, paired_scheduler, monkeypatch
):
    """Unknown batch under DDP must not scale the peak (paired mirror of the latent guard)."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 8)
    module = PairedLatentFlowModule(
        paired_unet_trainable,
        paired_scheduler,
        lr=1e-4,
        lr_ref_batch_size=8,
        lr_scale_rule="sqrt",
        train_batch_size=None,
    )
    optimizer = module.configure_optimizers()["optimizer"]
    assert optimizer.defaults["lr"] == pytest.approx(1e-4)


def test_summed_label_conditioning_injected(paired_unet):
    """The paired path injects embed(src)+embed(tgt) at the backbone's injection point.

    Equality trick: set ``embedding.weight[L] = embed(src)+embed(tgt)`` for some
    index L, then the paired call (src, tgt) must equal a single-label call with
    ``class_labels=L`` — proving the summed embedding is injected exactly.
    """
    torch.manual_seed(0)
    sample = torch.randn(1, 2 * C_LATENT, 4, 4, 4)  # concat([z_t, x_src])
    timestep = torch.tensor([0.5])
    spacing = torch.tensor([1.0, 1.0, 1.0])
    src, tgt = torch.tensor([0]), torch.tensor([1])
    table = paired_unet.unet.class_embedding.weight
    L = 3
    with torch.no_grad():
        table[L] = table[src.item()] + table[tgt.item()]
    paired_out = paired_unet(
        sample=sample, timestep=timestep, spacing=spacing,
        class_labels_src=src, class_labels_tgt=tgt,
    )
    single_out = paired_unet(
        sample=sample, timestep=timestep, spacing=spacing, class_labels=torch.tensor([L]),
    )
    assert torch.allclose(paired_out, single_out, atol=1e-6)


def test_paired_labels_must_be_passed_together(paired_unet):
    """Passing only one of (src, tgt) raises — they must arrive as a pair."""
    sample = torch.randn(1, 2 * C_LATENT, 4, 4, 4)
    spacing = torch.tensor([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="together"):
        paired_unet(sample=sample, timestep=0.5, spacing=spacing,
                    class_labels_src=torch.tensor([0]))


def _paired_unet_with_offset(offset: int) -> UNet3DConditionModel:
    """A paired UNet (in_channels = 2*C_latent) with a direction offset set."""
    torch.manual_seed(0)
    return UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
        paired_direction_offset=offset,
    )


def test_paired_direction_offset_breaks_ab_symmetry():
    """With paired_direction_offset>0, A->B and B->A use distinct target rows,
    so their summed conditions differ (the A<->B symmetry is broken). With
    offset=0 they are identical (the symmetric ADR-0014 behaviour)."""
    def cond_rows(unet, src, tgt):
        table = unet.unet.class_embedding.weight
        off = unet.paired_direction_offset
        return table[src] + table[tgt + off]

    off0 = _paired_unet_with_offset(0)
    off2 = _paired_unet_with_offset(2)
    # offset 0: A->B (rows 0,1) == B->A (rows 1,0) - symmetric sum
    assert torch.allclose(cond_rows(off0, 0, 1), cond_rows(off0, 1, 0), atol=1e-6)
    # offset 2: A->B (rows 0,3) != B->A (rows 1,2) - symmetry broken
    assert not torch.allclose(cond_rows(off2, 0, 1), cond_rows(off2, 1, 0), atol=1e-6)


def test_paired_direction_offset_injects_shifted_target():
    """A non-zero offset injects embed(src) + embed(tgt + offset): the paired call
    with (src, tgt) equals a single-label call whose precomputed row is the
    shifted sum (the offset-aware analogue of test_summed_label_conditioning_injected)."""
    unet = _paired_unet_with_offset(offset=2)
    torch.manual_seed(0)
    sample = torch.randn(1, 2 * C_LATENT, 4, 4, 4)
    timestep = torch.tensor([0.5])
    spacing = torch.tensor([1.0, 1.0, 1.0])
    src, tgt = torch.tensor([0]), torch.tensor([1])
    table = unet.unet.class_embedding.weight
    L = 3  # distinct from src(0) and tgt+offset(3) so the injection is non-trivial
    with torch.no_grad():
        table[L] = table[src.item()] + table[tgt.item() + unet.paired_direction_offset]
    paired_out = unet(
        sample=sample, timestep=timestep, spacing=spacing,
        class_labels_src=src, class_labels_tgt=tgt,
    )
    single_out = unet(
        sample=sample, timestep=timestep, spacing=spacing, class_labels=torch.tensor([L]),
    )
    assert torch.allclose(paired_out, single_out, atol=1e-6)


def test_paired_direction_offset_out_of_range_raises():
    """A shifted target row outside the embedding table fails fast (no silent wrap)."""
    unet = _paired_unet_with_offset(offset=4)  # num_class_embeds=4, tgt=1 -> row 5
    sample = torch.randn(1, 2 * C_LATENT, 4, 4, 4)
    spacing = torch.tensor([1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="out of range"):
        unet(sample=sample, timestep=0.5, spacing=spacing,
             class_labels_src=torch.tensor([0]), class_labels_tgt=torch.tensor([1]))


def test_noise_to_data_path_unchanged(paired_unet):
    """A single ``class_labels`` (the noise→data signature) still forwards — the
    paired extension is purely additive and backward-compatible (ADR-0014 wiring)."""
    sample = torch.randn(1, 2 * C_LATENT, 4, 4, 4)
    spacing = torch.tensor([1.0, 1.0, 1.0])
    out = paired_unet(sample=sample, timestep=0.5, spacing=spacing, class_labels=torch.tensor([2]))
    assert out.shape == (1, C_LATENT, 4, 4, 4)
    assert torch.isfinite(out).all()


# -- t_sampler: the data2data-adapted flow-time sampler ------------------------


def _timestep_module(**kwargs) -> PairedLatentFlowModule:
    """A module built for ``_sample_timesteps`` tests only (the UNet is unused)."""
    return PairedLatentFlowModule(_ZeroPairedUNet(), FlowMatchHeunDiscreteScheduler(), **kwargs)


def test_invalid_t_sampler_raises():
    """An unknown t_sampler is rejected at construction (fails fast)."""
    with pytest.raises(ValueError, match="t_sampler"):
        _timestep_module(t_sampler="bogus")


def test_invalid_t_range_raises():
    """t_min >= t_max (or out of [0, 1]) is rejected at construction (fails fast)."""
    with pytest.raises(ValueError, match="t_min"):
        _timestep_module(t_sampler="uniform", t_min=0.8, t_max=0.2)
    with pytest.raises(ValueError, match="t_min"):
        _timestep_module(t_sampler="uniform", t_min=-0.1, t_max=0.5)
    with pytest.raises(ValueError, match="t_min"):
        _timestep_module(t_sampler="uniform", t_min=0.5, t_max=1.1)


def test_default_t_sampler_is_logit_normal():
    """Omitting t_sampler keeps the inherited logit-normal sampler (backward-compat)."""
    module = _timestep_module()
    assert module.t_sampler == "logit_normal"
    torch.manual_seed(0)
    t = module._sample_timesteps(4096, torch.device("cpu"))
    assert t.min() >= 0.0 and t.max() <= 1.0
    # logit-normal with p_mean=-0.8 centers near sigmoid(-0.8) ≈ 0.31 (low-t biased).
    assert module.p_mean == pytest.approx(-0.8)
    expected_center = torch.sigmoid(torch.tensor(module.p_mean)).item()
    assert t.mean().item() == pytest.approx(expected_center, abs=0.05)


def test_uniform_t_sampler_covers_range():
    """t_sampler="uniform" draws t ~ U[t_min, t_max], centered at the midpoint."""
    t_min, t_max = 0.1, 0.9
    module = _timestep_module(t_sampler="uniform", t_min=t_min, t_max=t_max)
    torch.manual_seed(0)
    t = module._sample_timesteps(4096, torch.device("cpu"))
    assert t.min() >= t_min and t.max() <= t_max
    assert t.mean().item() == pytest.approx((t_min + t_max) / 2, abs=0.03)


def test_bridge_t_sampler_lands_on_uniform_grid():
    """t_sampler="bridge" quantizes t onto the 1000-node uniform grid over [t_min, t_max]."""
    t_min, t_max = 0.0, 1.0
    module = _timestep_module(t_sampler="bridge", t_min=t_min, t_max=t_max)
    torch.manual_seed(0)
    t = module._sample_timesteps(4096, torch.device("cpu"))
    assert t.min() >= t_min and t.max() <= t_max
    # Every draw equals (to fp tolerance) a node of the linspace grid over [t_min, t_max].
    grid = torch.linspace(t_min, t_max, 1000)
    # distance from each sample to its nearest grid node is ~0
    nearest = (t.unsqueeze(1) - grid.unsqueeze(0)).abs().min(dim=1).values
    assert nearest.max().item() < 1e-5


def test_uniform_and_bridge_ignore_p_mean():
    """The data2data samplers are driven by t_min/t_max, not the logit-normal p_mean."""
    module = _timestep_module(t_sampler="uniform", t_min=0.2, t_max=0.8, p_mean=5.0)
    torch.manual_seed(0)
    t = module._sample_timesteps(4096, torch.device("cpu"))
    # p_mean=5.0 would center a logit-normal near 1.0; uniform ignores it → midpoint 0.5.
    assert t.mean().item() == pytest.approx(0.5, abs=0.03)

