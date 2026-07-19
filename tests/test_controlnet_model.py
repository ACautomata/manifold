"""ControlNet model-layer regression tests (issue #130 / ADR-0026).

Locks the :class:`~manifold.ControlNet3DConditionModel` semantics independent of
any pipeline, protecting the MONAI residual contract:

- **zero-conv** ⇒ all emitted residuals are zero (safe warm-start = no-op at init);
- **warm-start** — ``load_base_encoder_weights`` copies the base encoder submodules;
- **injection** — non-zero residuals change the base's *pre-out* feature (the wrapped
  MONAI backbone's ``out`` conv is ``zero_module``-initialized, so an untrained base
  emits an all-zero output that would mask an injection effect in the model output —
  assert on the pre-``out`` feature instead, the documented hazard); and
- **gradient routing** — ControlNet params receive grad, base params do not (the
  base input ``z_t`` is a no-grad data tensor in the supervised regime).

Mirrors the wrapper-level assertion style of ``tests/test_persistence.py``.
"""

from __future__ import annotations

import torch

from manifold import ControlNet3DConditionModel, UNet3DConditionModel


def _tiny_unet() -> UNet3DConditionModel:
    torch.manual_seed(0)
    return UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)


def _tiny_controlnet(base: UNet3DConditionModel) -> ControlNet3DConditionModel:
    torch.manual_seed(1)
    return ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)


def _inputs(batch: int = 1):
    c, d, h, w = 4, 8, 8, 4
    z = torch.randn(batch, c, d, h, w)
    x_src = torch.randn(batch, c, d, h, w)
    spacing = torch.tensor([1.0, 1.0, 1.0])
    src = torch.full((batch,), 1, dtype=torch.long)
    tgt = torch.full((batch,), 2, dtype=torch.long)
    return z, x_src, spacing, src, tgt


# -- zero-conv ⇒ zero residuals (no-op at init) ------------------------------


def test_zero_conv_init_emits_zero_residuals():
    base = _tiny_unet()
    controlnet = _tiny_controlnet(base)
    controlnet.load_base_encoder_weights(base)
    z, x_src, spacing, src, tgt = _inputs()
    down_res, mid_res = controlnet(
        sample=z, controlnet_cond=x_src, timestep=0.5, spacing=spacing,
        class_labels_src=src, class_labels_tgt=tgt,
    )
    assert len(down_res) == len(controlnet.controlnet_down_zero_convs)
    for r in down_res:
        assert r.abs().max().item() == 0.0
    assert mid_res.abs().max().item() == 0.0


def test_warm_start_copies_base_encoder_weights():
    base = _tiny_unet()
    controlnet = _tiny_controlnet(base)
    # Perturb construction init so the copy is observable (not a no-op match).
    with torch.no_grad():
        for p in controlnet.conv_in.parameters():
            p.add_(0.5)
    controlnet.load_base_encoder_weights(base)
    for cn_p, base_p in zip(controlnet.conv_in.parameters(), base.unet.conv_in.parameters()):
        assert torch.equal(cn_p, base_p)
    for cn_p, base_p in zip(
        controlnet.middle_block.parameters(), base.unet.middle_block.parameters()
    ):
        assert torch.equal(cn_p, base_p)
    # The zero-convs + direction MLP keep construction init (not copied from base).
    for zc in list(controlnet.controlnet_down_zero_convs) + [controlnet.controlnet_mid_zero_conv]:
        for p in zc.parameters():
            assert p.abs().max().item() == 0.0


def test_zero_residual_injection_matches_plain_base():
    base = _tiny_unet()
    controlnet = _tiny_controlnet(base)
    controlnet.load_base_encoder_weights(base)
    z, x_src, spacing, src, tgt = _inputs()
    down_res, mid_res = controlnet(
        sample=z, controlnet_cond=x_src, timestep=0.5, spacing=spacing,
        class_labels_src=src, class_labels_tgt=tgt,
    )
    with torch.no_grad():
        out_injected = base(
            sample=z, timestep=0.5, spacing=spacing, class_labels=tgt,
            down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
        )
        out_plain = base(sample=z, timestep=0.5, spacing=spacing, class_labels=tgt)
    assert torch.equal(out_injected, out_plain)


# -- injection reaches the backbone (assert on pre-out feature) ---------------


def _pre_out_feature(base: UNet3DConditionModel, **fwd_kwargs) -> torch.Tensor:
    """Capture the backbone's input to its ``out`` conv (the pre-out feature).

    The wrapped MONAI backbone's ``out`` is ``zero_module``-init, so an untrained
    base's *output* is all-zero and would mask any residual-injection effect.
    Hooking the ``out`` conv's input lets the injection effect be measured on the
    feature the residuals actually perturb.
    """
    captured: dict[str, torch.Tensor] = {}
    handle = base.unet.out.register_forward_pre_hook(
        lambda module, args: captured.setdefault("feat", args[0].detach())
    )
    try:
        base(**fwd_kwargs)
    finally:
        handle.remove()
    return captured["feat"]


def test_nonzero_residuals_change_pre_out_feature():
    base = _tiny_unet()
    controlnet = _tiny_controlnet(base)
    controlnet.load_base_encoder_weights(base)
    z, x_src, spacing, src, tgt = _inputs()

    # Make the zero-convs non-zero so the residuals are non-trivial.
    with torch.no_grad():
        for zc in list(controlnet.controlnet_down_zero_convs) + [
            controlnet.controlnet_mid_zero_conv
        ]:
            for p in zc.parameters():
                p.add_(0.1 * torch.randn_like(p))

    down_res, mid_res = controlnet(
        sample=z, controlnet_cond=x_src, timestep=0.5, spacing=spacing,
        class_labels_src=src, class_labels_tgt=tgt,
    )
    assert any(r.abs().max().item() > 0 for r in down_res) or mid_res.abs().max().item() > 0

    with torch.no_grad():
        feat_injected = _pre_out_feature(
            base, sample=z, timestep=0.5, spacing=spacing, class_labels=tgt,
            down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
        )
        feat_plain = _pre_out_feature(
            base, sample=z, timestep=0.5, spacing=spacing, class_labels=tgt
        )
    shift = (feat_injected - feat_plain).abs().max().item()
    assert shift > 0.0


def test_residual_forward_is_bit_identical_to_monai_native():
    """The wrapper's out-of-place residual forward == MONAI's native forward args.

    The out-of-place re-implementation must not change the forward output — only
    the autograd-safety of the backward differs. Compare the wrapper's residual
    path against the backbone called directly with its native forward args.
    """
    base = _tiny_unet()
    controlnet = _tiny_controlnet(base)
    controlnet.load_base_encoder_weights(base)
    z, x_src, spacing, src, tgt = _inputs()
    with torch.no_grad():
        for zc in list(controlnet.controlnet_down_zero_convs) + [
            controlnet.controlnet_mid_zero_conv
        ]:
            for p in zc.parameters():
                p.add_(0.1 * torch.randn_like(p))

    down_res, mid_res = controlnet(
        sample=z, controlnet_cond=x_src, timestep=0.5, spacing=spacing,
        class_labels_src=src, class_labels_tgt=tgt,
    )
    with torch.no_grad():
        out_wrapper = base(
            sample=z, timestep=0.5, spacing=spacing, class_labels=tgt,
            down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
        )
        # MONAI native forward (in-place adds) — safe under no_grad.
        out_native = base.unet(
            x=z,
            timesteps=base._scaled_timesteps(0.5, z.shape[0], z.device, z.dtype),
            class_labels=tgt,
            spacing_tensor=base._batched_spacing(spacing, z.shape[0]),
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )
    assert torch.equal(out_wrapper, out_native)


# -- gradient routing through the base output (the supervised path) ------------


def test_supervised_backward_routes_grad_to_controlnet_not_base():
    """loss through the base output → grad reaches the ControlNet, not the frozen base.

    This is the real supervised-stage path: the ControlNet's residuals are injected
    into the frozen base, the base emits ``x0_pred``, and the ``(1−t)⁻²``-weighted
    MSE against ``x_tgt`` back-propagates through the base output to the ControlNet.
    It exercises the out-of-place residual forward (MONAI's native in-place adds
    would raise an autograd version error here — ADR-0026's hazard).
    """
    base = _tiny_unet()
    # Warm-start the base 'out' conv off zero so the output is a meaningful
    # function of the injected residuals (dodges the zero-init out hazard).
    with torch.no_grad():
        for p in base.unet.out.parameters():
            p.add_(0.1 * torch.randn_like(p))
    # Freeze the base (the supervised regime): no base param may receive grad.
    for p in base.parameters():
        p.requires_grad_(False)

    controlnet = _tiny_controlnet(base)
    controlnet.load_base_encoder_weights(base)
    z, x_src, spacing, src, tgt = _inputs()

    down_res, mid_res = controlnet(
        sample=z, controlnet_cond=x_src, timestep=0.5, spacing=spacing,
        class_labels_src=src, class_labels_tgt=tgt,
    )
    x0_pred = base(
        sample=z, timestep=0.5, spacing=spacing, class_labels=tgt,
        down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res,
    )
    target = torch.randn_like(x0_pred)
    loss = (x0_pred.float() - target.float()).pow(2).mean()
    loss.backward()  # must not raise (out-of-place residual forward)

    cn_grads = [p.grad for p in controlnet.parameters() if p.grad is not None]
    assert len(cn_grads) > 0
    assert any(g.abs().max().item() > 0 for g in cn_grads)
    base_grads = [p.grad for p in base.parameters() if p.grad is not None]
    assert all(g.abs().max().item() == 0.0 for g in base_grads)
