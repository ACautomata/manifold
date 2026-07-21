"""Callback registry (ADR-0029) tests: fail-fast resolve + two-phase build.

Issue #159 acceptance: ``CallbackRegistry.resolve`` fails fast on an unknown
callback name and on an unknown knob for a known name; ``build`` yields the
constructed callback via the registered spec.

Issue #160 acceptance: ``FIDSpec`` / ``CheckpointSpec`` build their callbacks
through the registry; ``validate_monitor`` fails fast on a monitored checkpoint
whose metric is neither callback-logged nor module-declared, and an explicit
``monitor_metric=None`` yields an unmonitored checkpoint without error.
"""

from dataclasses import dataclass

import pytest

from manifold.metrics import FIDCallback
from manifold.training.callbacks import (
    CallbackContext,
    CallbackRegistry,
    CheckpointSpec,
    FIDSpec,
    TrainLossSpec,
)
from manifold.training.metrics import TrainLossLogger


def _ctx():
    """A CallbackContext for build (TrainLossSpec reads none of its fields)."""
    return CallbackContext(
        module=None,
        vae=None,
        datamodule=None,
        inference_recipe=None,
        model_dir=".",
        seed=0,
    )


def _fid_ctx(real_latents=None):
    """A CallbackContext carrying a tiny generation recipe for FIDSpec.build."""
    recipe = {
        "latent_shape": (1, 4, 4, 4, 4),
        "spacing": [1.0, 1.0, 1.0],
        "modality": 1,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "cfg_interval": None,
    }
    datamodule = object()  # the JiT lazy real_latents_source (an opaque handle).
    return CallbackContext(
        module=None, vae=None, datamodule=datamodule, inference_recipe=recipe,
        model_dir=".", seed=0, real_latents=real_latents,
    ), datamodule


def _registry():
    reg = CallbackRegistry()
    reg.register("train_loss", TrainLossSpec)
    reg.register("fid", FIDSpec)
    reg.register("checkpoint", CheckpointSpec)
    return reg


def test_resolve_fails_fast_on_unknown_name():
    """An unregistered name fails fast at resolve (config-time), not at build."""
    reg = _registry()
    with pytest.raises(KeyError, match="Unknown callback name"):
        reg.resolve(["bogus"])


def test_resolve_fails_fast_on_unknown_knob_for_known_name():
    """An unknown knob for a known name fails fast at resolve.

    ``TrainLossSpec`` declares no knobs, so any knob is unknown — this is the
    live JiT path exercised by ``cli.run_training``.
    """
    reg = _registry()
    with pytest.raises(ValueError, match="Unknown knob"):
        reg.resolve(["train_loss"], cfg={"train_loss": {"bogus_knob": 1}})


def test_unknown_knob_against_fielded_spec():
    """A fielded spec accepts its knobs and fails fast on unknown ones.

    The stronger form of the unknown-knob check: a spec with real config knobs
    resolves the known ones and rejects the rest (the case the field-less
    ``TrainLossSpec`` cannot itself exercise).
    """

    @dataclass(frozen=True)
    class _FieldedSpec:
        a: int = 0
        b: str = "x"

        def build(self, ctx):
            ...

    reg = CallbackRegistry()
    reg.register("fake", _FieldedSpec)
    [spec] = reg.resolve(["fake"], cfg={"fake": {"a": 7}})
    assert spec.a == 7 and spec.b == "x"
    with pytest.raises(ValueError, match="Unknown knob"):
        reg.resolve(["fake"], cfg={"fake": {"c": 1}})


def test_build_returns_train_loss_logger():
    """resolve + build yields a TrainLossLogger via the registered spec."""
    reg = _registry()
    [cb] = reg.build(reg.resolve(["train_loss"]), _ctx())
    assert isinstance(cb, TrainLossLogger)


# -- issue #160: FIDSpec / CheckpointSpec / validate_monitor --------------------


def test_fid_spec_builds_fid_callback_with_knobs_and_lazy_pull():
    """FIDSpec.build threads its knobs + the generation recipe into FIDCallback,
    preserving the JiT lazy real-latents pull (real_latents=None -> datamodule)."""
    ctx, datamodule = _fid_ctx(real_latents=None)
    reg = _registry()
    [spec] = reg.resolve(["fid"], cfg={"fid": {"num_synth": 2, "cov_ridge": 1.0e-2}})
    [cb] = reg.build([spec], ctx)
    assert isinstance(cb, FIDCallback)
    assert cb.num_synth == 2 and cb.cov_ridge == 1.0e-2
    assert cb.latent_shape == (1, 4, 4, 4, 4)
    assert cb.real_latents is None
    assert cb._real_latents_source is datamodule  # JiT lazy fallback (F5)


def test_fid_spec_grpo_supplied_real_latents_wins():
    """When ctx.real_latents is set (GRPO), the callback uses it directly."""
    ctx, _ = _fid_ctx(real_latents=object())
    reg = _registry()
    [cb] = reg.build(reg.resolve(["fid"]), ctx)
    assert cb.real_latents is ctx.real_latents


def test_checkpoint_spec_declares_full_knob_surface():
    """The checkpoint config surface is all overridable via the registry."""
    reg = _registry()
    [spec] = reg.resolve(
        ["checkpoint"],
        cfg={"checkpoint": {"monitor_metric": "val/fid", "save_top_k": 5, "mode": "min"}},
    )
    assert spec.monitor_metric == "val/fid" and spec.save_top_k == 5
    with pytest.raises(ValueError, match="Unknown knob"):
        reg.resolve(["checkpoint"], cfg={"checkpoint": {"not_a_knob": 1}})


def test_validate_monitor_passes_when_metric_is_callback_logged():
    """A monitored checkpoint on a callback-logged metric (val/fid) is valid."""
    reg = _registry()
    specs = reg.resolve(["fid", "checkpoint"], cfg={"checkpoint": {"monitor_metric": "val/fid"}})
    reg.validate_monitor(specs, module=None)  # no module metrics; val/fid from FIDSpec.


def test_validate_monitor_fails_on_unlogged_metric():
    """A monitored metric absent from callbacks AND the module fails fast."""
    reg = _registry()
    specs = reg.resolve(["checkpoint"], cfg={"checkpoint": {"monitor_metric": "val/bogus"}})
    with pytest.raises(ValueError, match="monitor_metric 'val/bogus'"):
        reg.validate_monitor(specs, module=None)


def test_validate_monitor_passes_on_module_declared_metric():
    """A metric the module declares (not any callback) is a valid monitor — the
    reward/GRPO case (val/gen_pair_acc / val/mean_reward logged by the Module)."""
    module = type("M", (), {"logged_metrics": {"val/gen_pair_acc"}})()
    reg = _registry()
    specs = reg.resolve(["checkpoint"], cfg={"checkpoint": {"monitor_metric": "val/gen_pair_acc"}})
    reg.validate_monitor(specs, module)


def test_validate_monitor_none_bypasses_validation():
    """monitor_metric=None yields an unmonitored checkpoint without error, even
    with no callback logging anything (the no-held-out-val production path)."""
    reg = _registry()
    specs = reg.resolve(["checkpoint"], cfg={"checkpoint": {"monitor_metric": None}})
    reg.validate_monitor(specs, module=None)  # must not raise.


def test_validate_monitor_accepts_extra_callback_logged_metric():
    """A monitored metric an extra (non-registry) callback logs is valid without
    the shell mutating module.logged_metrics — the LatentX0MAE/val/x0_mae case
    (ADR-0029). ``LatentX0MAE`` declares ``logged_metrics={"val/x0_mae"}``.
    """
    from manifold.training.metrics import LatentX0MAE

    reg = _registry()
    specs = reg.resolve(["checkpoint"], cfg={"checkpoint": {"monitor_metric": "val/x0_mae"}})
    # No module.logged_metrics — the metric comes from the extra callback.
    reg.validate_monitor(specs, module=None, extra_callbacks=[LatentX0MAE()])


def test_validate_monitor_fails_when_extra_callback_omits_metric():
    """Without the extra callback that logs the metric, validation fails — the
    flip side of the test above (proves the extra-callback path is load-bearing)."""
    reg = _registry()
    specs = reg.resolve(["checkpoint"], cfg={"checkpoint": {"monitor_metric": "val/x0_mae"}})
    with pytest.raises(ValueError, match="monitor_metric 'val/x0_mae'"):
        reg.validate_monitor(specs, module=None, extra_callbacks=None)
