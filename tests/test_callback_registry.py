"""Callback registry (ADR-0029) tests: fail-fast resolve + two-phase build.

Issue #159 acceptance: ``CallbackRegistry.resolve`` fails fast on an unknown
callback name and on an unknown knob for a known name; ``build`` yields the
constructed callback via the registered spec.
"""

from dataclasses import dataclass

import pytest

from manifold.training.callbacks import (
    CallbackContext,
    CallbackRegistry,
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


def _registry():
    reg = CallbackRegistry()
    reg.register("train_loss", TrainLossSpec)
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
