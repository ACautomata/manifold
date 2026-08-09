"""DevicePolicy unit tests (ADR-0035, issue #204).

External-behavior assertions only (PRD "Testing Decisions"): ``pin`` is the
one-time pre-PG side effect (set_device, exactly once, with the twin's
out-of-range guard), ``device`` is the side-effect-free read, and
``warm_device`` reproduces the former ``resolve_warm_device`` semantics
(PG-initialized + CUDA fallback -> ``cuda:{local_rank}``, else the fallback
unchanged). All CUDA calls are monkeypatched — no real GPU required.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.distributed as dist

from manifold.training.device_policy import DevicePolicy


@pytest.fixture()
def policy(monkeypatch):
    """A DevicePolicy with a deterministic LOCAL_RANK snapshot."""
    monkeypatch.setenv("LOCAL_RANK", "3")
    return DevicePolicy()


def test_pin_returns_cuda_local_rank_and_sets_device_exactly_once(policy, monkeypatch):
    """``pin()`` under LOCAL_RANK=k + CUDA available returns ``cuda:k`` and calls
    ``set_device(k)`` exactly once (the one pre-PG side effect)."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(i))

    assert policy.pin() == torch.device("cuda:3")
    assert calls == [3], f"set_device called {calls} (expected exactly [3])"


def test_pin_out_of_range_local_rank_skips_set_device(policy, monkeypatch):
    """The twin's guard semantics: LOCAL_RANK >= device_count -> NO set_device
    (the rank still gets its cuda:{local_rank} device value)."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(i))

    assert policy.pin() == torch.device("cuda:3")
    assert calls == [], "set_device called for an out-of-range LOCAL_RANK"


def test_pin_cpu_unavailable_returns_cpu_no_side_effects(policy, monkeypatch):
    """No CUDA -> ``cpu``, zero side effects (no set_device, no device_count)."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(i))

    assert policy.pin() == torch.device("cpu")
    assert calls == []


def test_pin_local_rank_unset_defaults_to_zero(monkeypatch):
    """LOCAL_RANK missing -> the snapshot defaults to 0 (cuda:0, the twin's
    os.environ.get("LOCAL_RANK", 0) semantics)."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(i))

    assert DevicePolicy().pin() == torch.device("cuda:0")
    assert calls == [0]


def test_device_returns_cuda_local_rank_side_effect_free(policy, monkeypatch):
    """``device()`` returns the same value as ``pin()`` but never calls
    ``set_device`` (read-only / debug paths)."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "set_device", lambda i: calls.append(i))

    assert policy.device() == torch.device("cuda:3")
    assert calls == [], "device() must have no set_device side effect"


def test_device_cpu_unavailable_returns_cpu(policy, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert policy.device() == torch.device("cpu")


def test_warm_device_local_rank_under_ddp(policy, monkeypatch):
    """PG initialized + CUDA fallback -> ``cuda:3`` (post-PG VAE warm uses the
    per-rank local device, the former ``resolve_warm_device`` P1 behavior)."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    assert policy.warm_device(torch.device("cuda")) == torch.device("cuda:3")


def test_warm_device_non_cuda_fallback_returned_unchanged(policy, monkeypatch):
    """A non-CUDA fallback is returned unchanged even under DDP (warm stays CPU)."""
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    assert policy.warm_device(torch.device("cpu")) == torch.device("cpu")


def test_warm_device_single_process_fallback_returned_unchanged(policy, monkeypatch):
    """PG NOT initialized (single-process) -> the fallback unchanged."""
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    assert policy.warm_device(torch.device("cuda")) == torch.device("cuda")
    assert policy.warm_device(torch.device("cpu")) == torch.device("cpu")


def test_warm_device_pg_up_without_cuda_still_resolves_cuda_fallback(policy, monkeypatch):
    """Byte-identical to the former ``resolve_warm_device``: the gate is
    ``fallback.type == "cuda" and dist.is_initialized()`` — there is NO
    ``is_available`` check, so a CUDA fallback under a live PG resolves to
    ``cuda:{local_rank}`` even when CUDA is unavailable on this box."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)

    assert policy.warm_device(torch.device("cuda")) == torch.device("cuda:3")
    assert policy.warm_device(torch.device("cpu")) == torch.device("cpu")


def test_warm_device_dist_import_is_lazy():
    """The module import must NOT import ``torch.distributed`` (importing the
    module never touches the process group): the lazy import lives inside the
    method body, and no top-level ``import torch.distributed`` exists."""
    from manifold.training import device_policy

    src = inspect.getsource(device_policy)
    assert "import torch.distributed" not in src.split("class DevicePolicy")[0], (
        "module-level import of torch.distributed (must be lazy, inside the method)"
    )
    assert "import torch.distributed" in inspect.getsource(
        device_policy.DevicePolicy.warm_device
    ), "warm_device lost its lazy torch.distributed import"


# -- Shell wiring guards (ADR-0035 T2, issue #205) ------------------------------
#
# The grpo + reward shells used to carry a character-identical pre-PG
# ``set_device`` twin. #205 deletes both twins and routes them through
# ``DevicePolicy.pin()`` (de-duplication, not a behavior change). Verified by
# source: each ``main`` constructs ``DevicePolicy`` and calls ``.pin()``, and no
# longer inlines the ``set_device`` block (the test_f3_* getsource pattern).


def test_grpo_cli_main_pins_via_device_policy_not_inline_set_device():
    """grpo ``main`` pins through ``DevicePolicy.pin()`` and no longer holds the
    pre-PG ``set_device`` twin (ADR-0035 T2)."""
    from manifold.training import grpo_cli

    src = inspect.getsource(grpo_cli.main)
    assert "DevicePolicy" in src, "grpo_cli.main does not construct DevicePolicy (T2)"
    assert ".pin()" in src, "grpo_cli.main does not call DevicePolicy.pin() (T2)"
    assert "set_device" not in src, "grpo_cli.main still inlines the set_device twin (T2)"


def test_reward_cli_main_pins_via_device_policy_not_inline_set_device():
    """reward ``main`` pins through ``DevicePolicy.pin()`` and no longer holds the
    pre-PG ``set_device`` twin (ADR-0035 T2)."""
    from manifold.training import reward_cli

    src = inspect.getsource(reward_cli.main)
    assert "DevicePolicy" in src, "reward_cli.main does not construct DevicePolicy (T2)"
    assert ".pin()" in src, "reward_cli.main does not call DevicePolicy.pin() (T2)"
    assert "set_device" not in src, "reward_cli.main still inlines the set_device twin (T2)"


# -- Shell wiring guards (ADR-0035 T3, issue #206) ------------------------------
#
# controlnet's shell used to resolve a BARE cuda device
# (``torch.device("cuda" if torch.cuda.is_available() else "cpu")`` -> cuda:0), so
# under DDP every rank staged its frozen base + ControlNet onto GPU 0 — the PR #156
# cuda:0 race / DDP-init-timeout category ADR-0031 §A2 promised to close. #206 routes
# the real path through ``DevicePolicy.pin()`` (per-rank cuda:{LOCAL_RANK}); the
# data_provider CPU smoke stays DevicePolicy-free (transparent). Verified by source.


def test_controlnet_cli_main_pins_via_device_policy_not_bare_cuda():
    """controlnet ``main`` pins through ``DevicePolicy.pin()`` on the real path and
    no longer resolves the bare cuda device that collapsed every DDP rank onto GPU 0
    (ADR-0035 T3 / PR #156)."""
    from manifold.training import controlnet_cli

    src = inspect.getsource(controlnet_cli.main)
    assert "DevicePolicy" in src, "controlnet_cli.main does not construct DevicePolicy (T3)"
    assert ".pin()" in src, "controlnet_cli.main does not call DevicePolicy.pin() (T3)"
    # The bare cuda device resolution (every DDP rank onto GPU 0 — PR #156) is gone:
    # main no longer constructs a cuda device directly; the device comes solely from
    # DevicePolicy.pin() on the real path (and torch.device("cpu") on the smoke path).
    assert 'torch.device("cuda"' not in src, (
        "controlnet_cli.main still resolves a bare cuda device (T3 / PR #156 regression)"
    )
