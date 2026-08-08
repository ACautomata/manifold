"""Behavior-contract tests for :class:`manifold.modules.FrozenArmMixin` (ADR-0031 A1).

The mixin is a pure-expand deliverable (issue #198): it lands the shared
"register + dual-exclude" mechanism WITHOUT wiring any real host module. With no
host wired, the spec's "indirect coverage via the host modules' existing tests"
gives zero coverage here, so these tests exercise the mixin through a minimal
fixture host that mirrors the real shape (``FrozenArmMixin`` precedes an
``nn.Module``-derived base, one frozen arm + one trainable arm, the host owns its
own ``configure_optimizers``).

They assert ONLY external behavior — the same contract the GRPOModule tests
(``test_frozen_arms_registered_but_dual_excluded`` et al.) assert on a real host:
state_dict contents, optimizer param groups, ``requires_grad``, eval-after-train,
wrapper-prefix stripping, strict-load parity. No mixin-internal structure.

``FrozenArmMixin``'s three dunders chain to ``torch.nn.Module`` (verified: neither
``spt.Module`` nor ``pl.LightningModule`` overrides ``state_dict`` /
``load_state_dict`` / ``train``), so an ``nn.Module`` base faithfully exercises the
cooperative-``super()`` contract and keeps the backward test free of Lightning's
forward/optim machinery. The production ``spt.Module`` MRO is covered by the three
host modules' existing tests (untouched here).
"""

from collections import OrderedDict

import pytest
import torch
from torch import nn

from manifold.modules import FrozenArmMixin


class _Host(FrozenArmMixin, nn.Module):
    """Minimal host exercising ``FrozenArmMixin``'s dunder contract.

    One frozen arm + one trainable arm; the host owns its own
    ``configure_optimizers`` (the mixin does NOT take over arm selection). ``forward``
    lets a backward pass prove grad reaches the trainable arm only.
    """

    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(4, 3)
        self._register_frozen_arm("frozen", nn.Linear(4, 3))

    def forward(self, x):
        return self.trainable(x).sum() + self.frozen(x).sum()

    def configure_optimizers(self):
        return {"optimizer": torch.optim.Adam(self.trainable.parameters(), lr=1e-3)}


def _host():
    return _Host()


def test_register_frozen_arm_freezes_and_evals():
    """Registration applies the uniform frozen-arm prep + declares the arm set.

    The frozen arm is eval + requires_grad=False and present in parameters()
    (registered, so Lightning's automatic .to(device) owns it); the trainable arm
    stays in training mode with grad. ``_frozen_arm_names`` is the declared set
    (fixed at construction).
    """
    host = _host()

    assert host._frozen_arm_names == frozenset({"frozen"})

    module_param_ids = {id(p) for p in host.parameters()}
    assert {id(p) for p in host.frozen.parameters()} <= module_param_ids
    assert not host.frozen.training, "frozen arm must be eval at registration"
    assert not any(p.requires_grad for p in host.frozen.parameters())
    assert host.trainable.training, "trainable arm follows the module mode"
    assert any(p.requires_grad for p in host.trainable.parameters())


def test_state_dict_strips_frozen_arm_in_place():
    """state_dict() strips the frozen arm IN PLACE on the shared destination.

    super().state_dict() writes the frozen arm into the caller-supplied
    destination; a parent/wrapper drives the recursion and reads destination (the
    return value IS that object), so the filter must mutate it, not build a fresh
    dict — otherwise the multi-GB arm leaks into the checkpoint.
    """
    host = _host()
    destination = OrderedDict()
    returned = host.state_dict(destination=destination)

    assert returned is destination, "state_dict() must return the shared destination"
    assert not any(k.split(".", 1)[0] == "frozen" for k in destination), "frozen arm leaked"
    assert any(k.startswith("trainable.") for k in destination), "trainable arm must remain"


def test_state_dict_strips_frozen_arm_through_wrapper_prefix():
    """state_dict() strips the frozen arm when collected through a wrapper (DDP).

    PyTorch recurses via module.state_dict(destination=..., prefix='module.') and
    DISCARDS the child's return value, so the override must (a) recognize the
    prefixed keys (``module.frozen.*``) and (b) mutate the shared destination in
    place — else the arm leaks under the wrapper prefix.
    """
    host = _host()

    class _Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    keys = set(_Wrapper(host).state_dict())
    leaked = [k for k in keys if k.startswith("module.frozen.")]
    assert not leaked, f"frozen arm leaked through wrapper state_dict: {leaked[:3]}"
    assert any(k.startswith("module.trainable.") for k in keys), "trainable arm must remain"


def test_load_state_dict_strict_except_frozen_allowlist():
    """load_state_dict is strict on trainable keys, lenient only on the frozen arm.

    A checkpoint (state_dict() strips the frozen arm) round-trips into a fresh host
    with strict=True — the arm's absence is the one tolerated mismatch. A missing or
    unexpected TRAINABLE key raises instead of silently resuming on stale/random
    weights (no blanket strict=False).
    """
    host = _host()
    sd = host.state_dict()  # trainable.* only (frozen arm stripped)

    fresh = _host()
    fresh.load_state_dict(sd)  # strict=True (default) — frozen arm absent → allowlisted

    a_trainable_key = next(k for k in sd if k.startswith("trainable."))
    short = {k: v for k, v in sd.items() if k != a_trainable_key}
    with pytest.raises(RuntimeError, match="missing"):
        fresh.load_state_dict(short)

    extra = {**sd, "trainable.NOT_A_REAL_PARAM": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="unexpected"):
        fresh.load_state_dict(extra)


def test_train_keeps_frozen_arm_in_eval():
    """module.train() re-freezes the registered frozen arm to eval.

    Registration makes nn.Module.train() recurse into the frozen arm and flip it to
    training mode; the train() override re-applies eval() so its BatchNorm running
    stats cannot drift. The trainable arm follows the module mode.
    """
    host = _host()
    host.train(True)
    assert not host.frozen.training, "frozen arm must stay eval after train(True)"
    assert host.trainable.training, "trainable arm follows the module mode"
    host.train(False)
    assert not host.frozen.training
    assert not host.trainable.training


def test_backward_only_touches_trainable_arm():
    """backward reaches the trainable arm; the frozen arm (requires_grad=False) does not.

    The off-optimizer + no-grad invariant at the autograd level: grad flows into the
    trainable arm only; the frozen arm's params carry no grad.
    """
    host = _host()
    host(torch.randn(2, 4)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in host.trainable.parameters())
    assert all(p.grad is None for p in host.frozen.parameters())


def test_configure_optimizers_excludes_frozen_arm():
    """The host's optimizer wires the trainable arm only — the mixin does not own this.

    configure_optimizers stays the host's concern (the spec: the mixin does NOT take
    over arm selection); the frozen arm is off the optimizer at the param-group level.
    """
    host = _host()
    opt = host.configure_optimizers()["optimizer"]
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    assert opt_ids == {id(p) for p in host.trainable.parameters()}
    assert opt_ids.isdisjoint({id(p) for p in host.frozen.parameters()})
