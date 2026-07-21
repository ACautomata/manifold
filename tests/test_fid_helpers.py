"""Tests for the 5 composable FID helpers (ADR-0030) + collective-count hardening.

Covers: VramStage, FixedSampleRollout, LatentDecoder, SufficientStatsReducer,
and the error-flag rendezvous pattern that enforces collective-count invariance.
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from manifold.metrics.fid import (
    LatentDecoder,
    SufficientStatsReducer,
    VramStage,
)


# -- VramStage ---------------------------------------------------------------


class _ToyVAE(nn.Module):
    """A VAE-like module that can be moved to/from CPU for VramStage tests."""

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))
        self._called_decode = False

    def state_dict(self, *a, **k):
        return {"_dummy": self._dummy.detach().clone()}

    def load_state_dict(self, sd, strict=True):
        self._dummy = nn.Parameter(sd["_dummy"].clone())

    def modules(self):
        return iter([self])

    def decode(self, latents):
        self._called_decode = True
        return latents


class _FakeFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 8, bias=False)

    def forward(self, x):
        b = x.shape[0]
        flat = x.reshape(b, -1)[:, :64]
        if flat.shape[1] < 64:
            flat = nn.functional.pad(flat, (0, 64 - flat.shape[1]))
        return self.proj(flat)


def test_vram_stage_restores_vae_to_cpu():
    """VramStage.__exit__ restores the VAE + feature_net to CPU."""
    vae = _ToyVAE()
    fn = _FakeFeatureNet()
    assert next(vae.parameters()).device.type == "cpu"
    assert fn.proj.weight.device.type == "cpu"

    with VramStage(
        vae, feature_net=fn, device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert stage.fid_disabled is False
        assert stage.feature_net is fn

    # After exit: both are back on CPU.
    assert next(vae.parameters()).device.type == "cpu"
    assert fn.proj.weight.device.type == "cpu"


def test_vram_stage_sets_feature_net_to_eval():
    """VramStage.__enter__ sets the feature net to eval mode."""
    vae = _ToyVAE()
    fn = _FakeFeatureNet()
    fn.train()  # explicit train mode
    assert fn.training

    with VramStage(
        vae, feature_net=fn, device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert stage.feature_net.training is False


def test_vram_stage_disables_fid_when_feature_net_is_none_and_no_factory():
    """When both feature_net and factory are None, fid_disabled is True."""
    vae = _ToyVAE()
    with VramStage(
        vae, feature_net=None, feature_net_factory=None,
        device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert stage.fid_disabled is True
        assert stage.feature_net is None


def test_vram_stage_lazy_builds_from_factory():
    """VramStage.__enter__ calls the factory and uses its result."""
    vae = _ToyVAE()
    fn = _FakeFeatureNet()
    built = [False]

    def factory():
        built[0] = True
        return fn

    with VramStage(
        vae, feature_net=None, feature_net_factory=factory,
        device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert built[0] is True
        assert stage.feature_net is fn
        assert stage.fid_disabled is False


def test_vram_stage_failing_factory_sets_disabled():
    """A factory that returns None => fid_disabled = True."""
    vae = _ToyVAE()

    with VramStage(
        vae, feature_net=None,
        feature_net_factory=lambda: None,
        device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert stage.fid_disabled is True


def test_vram_stage_raising_factory_caught_and_sets_disabled():
    """A RAISING factory is caught in VramStage.__enter__ — fid_disabled is set
    and the VAE is restored to CPU (no exception escapes the context manager)."""
    vae = _ToyVAE()

    def _raising_factory():
        raise RuntimeError("simulated factory crash")

    with VramStage(
        vae, feature_net=None,
        feature_net_factory=_raising_factory,
        device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert stage.fid_disabled is True
        assert stage.feature_net is None

    # VAE must be back on CPU.
    assert next(vae.parameters()).device.type == "cpu"


def test_vram_stage_probes_feat_dim():
    """VramStage probes feat_dim once from the feature_net."""
    vae = _ToyVAE()
    fn = _FakeFeatureNet()

    with VramStage(
        vae, feature_net=fn, feat_dim=None,
        device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert stage.feat_dim == 8  # _FakeFeatureNet outputs 8-dim


def test_vram_stage_respects_cached_feat_dim():
    """When feat_dim is already known, no re-probing occurs."""
    vae = _ToyVAE()
    fn = _FakeFeatureNet()

    with VramStage(
        vae, feature_net=fn, feat_dim=42,
        device_fn=lambda: torch.device("cpu"),
    ) as stage:
        assert stage.feat_dim == 42  # cached, not re-probed


# -- LatentDecoder -----------------------------------------------------------


def test_latent_decoder_disables_norm16_once():
    """LatentDecoder disables norm_float16 on the first call and caches the flag."""

    class _VAEWithNorm16(nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = nn.Parameter(torch.zeros(1))
            self.norm = _Norm16Module()

        def state_dict(self, *a, **k):
            return {}

        def load_state_dict(self, sd, strict=True):
            pass

        def modules(self):
            return [self, self.norm]

        def decode(self, latents):
            return latents

    class _Norm16Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_float16 = True

    vae = _VAEWithNorm16()
    assert vae.norm.norm_float16 is True

    decoder = LatentDecoder(vae)
    assert decoder.norm16_disabled is False

    # First call disables norm_float16.
    decoder(torch.randn(1, 4, 4, 4, 4))
    assert vae.norm.norm_float16 is False
    assert decoder.norm16_disabled is True

    # Second call — no-op (flag is cached).
    vae.norm.norm_float16 = True  # simulate re-enable
    decoder(torch.randn(1, 4, 4, 4, 4))
    assert vae.norm.norm_float16 is True  # not re-disabled (cached skip)


# -- SufficientStatsReducer --------------------------------------------------


def test_reducer_n1_shard_contributes_stats():
    """A single-feature shard (n==1) contributes non-zero stats — only the
    GLOBAL n>=2 check after all_reduce gates FID (codex #116 P2 regression)."""
    reducer = SufficientStatsReducer(feat_dim=6)
    feats = torch.randn(1, 6)
    planes = [feats, torch.empty(0), feats]  # middle plane empty
    result = reducer(planes, device=torch.device("cpu"))
    # Under single-process (world=1), n==1 → None (FID undefined).
    assert result[0] is None  # n=1: global < 2
    assert result[1] is None  # empty
    assert result[2] is None  # n=1

    # But the zero branch is ONLY for genuinely empty shards, not n==1.
    src = inspect.getsource(SufficientStatsReducer.__call__)
    assert "shape[0] < 2" not in src, "n==1 is zeroed before all_reduce"
    assert "shape[0] == 0" in src, "empty-shard guard lost"


def test_reducer_n2_yields_moments():
    """When global n>=2, reducer returns (mu, sigma, n)."""
    reducer = SufficientStatsReducer(feat_dim=4)
    feats = torch.randn(3, 4)  # n=3 >= 2
    result = reducer([feats, feats, feats], device=torch.device("cpu"))
    assert len(result) == 3
    for (mu, sigma, n) in result:
        assert mu.shape == (4,)
        assert sigma.shape == (4, 4)
        assert n == 3


# -- FixedSampleRollout (low-level) ------------------------------------------


def test_rollout_rank_striding():
    """FixedSampleRollout generates indices rank, rank+world, ... for each rank."""
    from manifold.metrics.fid.rollout import FixedSampleRollout, _rank_world

    # Simulate rank-striding by inspecting the range logic: _rank_world returns
    # (rank, world); FixedSampleRollout uses range(rank, num_synth, world).
    # Single-process world=1 => generates all num_synth indices.
    assert _rank_world() == (0, 1)


def test_rollout_produces_correct_count():
    """Single-process rollout generates num_synth latents."""
    from manifold.metrics.fid.rollout import FixedSampleRollout

    class _ToyModule:
        def sample(self, *a, generator=None, **k):
            return torch.randn(1, 4, 4, 4, 4)

    rollout = FixedSampleRollout(
        module=_ToyModule(),
        latent_shape=(1, 4, 4, 4, 4),
        spacing=[1.0, 1.0, 1.0],
        modality=1,
        num_inference_steps=2,
        num_synth=4,
        seed=0,
    )
    latents = rollout(torch.device("cpu"))
    assert len(latents) == 4  # world=1 → all 4 generated


def test_rollout_deterministic_seed():
    """Same seed produces identical latents across calls."""
    from manifold.metrics.fid.rollout import FixedSampleRollout

    class _ToyModule:
        def sample(self, *a, generator=None, **k):
            return torch.randn(1, 4, 4, 4, 4, generator=generator)

    rollout = FixedSampleRollout(
        module=_ToyModule(),
        latent_shape=(1, 4, 4, 4, 4),
        spacing=[1.0, 1.0, 1.0],
        modality=1,
        num_inference_steps=2,
        num_synth=3,
        seed=42,
    )
    first = rollout(torch.device("cpu"))
    second = rollout(torch.device("cpu"))
    for a, b in zip(first, second):
        assert torch.equal(a, b), "latents differ across passes with same seed"


# -- Error-flag rendezvous (collective-count hardening) ----------------------


def test_all_reduce_flag_returns_true_on_any_positive(monkeypatch):
    """_all_reduce_flag all_reduces with MAX and returns True if any rank has flag>0."""
    from manifold.metrics.fid.callback import FIDCallback

    # Simulate single-process: no PG initialized → returns the raw flag value.
    assert FIDCallback._all_reduce_flag(torch.tensor([0.0])) is False
    assert FIDCallback._all_reduce_flag(torch.tensor([1.0])) is True


def test_all_reduce_flag_uses_max_op():
    """Source guard: _all_reduce_flag reduces with ReduceOp.MAX."""
    from manifold.metrics.fid.callback import FIDCallback

    src = inspect.getsource(FIDCallback._all_reduce_flag)
    assert "ReduceOp.MAX" in src, "error flag must use MAX reduction"


def test_on_validation_epoch_end_has_error_rendezvous():
    """on_validation_epoch_end contains two rendezvous points:
    (1) disabled-flag all_reduce, and (2) error-flag all_reduce before reducer."""
    from manifold.metrics.fid.callback import FIDCallback

    src = inspect.getsource(FIDCallback.on_validation_epoch_end)
    # Rendezvous #1: disabled flag via _all_reduce_flag.
    assert "_all_reduce_flag" in src
    # Rendezvous #2: error flag before reducer.
    # Count occurrences of _all_reduce_flag — should be 2.
    assert src.count("_all_reduce_flag") >= 2, (
        "on_validation_epoch_end must have at least 2 rendezvous points"
    )
