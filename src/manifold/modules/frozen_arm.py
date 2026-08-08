"""Shared "register + dual-exclude" mechanism for frozen arms (ADR-0031 A1).

This module is the shared implementation of the ADR-0031 A1 convention — NOT the
wrapper-adapter/holder direction that ADR-0031 explicitly rejected (that path kept
a manual ``.to(device)``, broke the Mode-1/Mode-2 parity, and did not cover A2).
A frozen arm is a plain registered ``nn.Module`` submodule (so Lightning's
automatic ``.to(device)`` places it), kept off the optimizer
(``requires_grad=False`` + the host's trainable-arm-only ``configure_optimizers``)
and off the checkpoint (the ``state_dict()`` override), and held in ``eval()``
across ``module.train()`` (the ``train()`` override).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrozenArmMixin:
    """The shared "register + dual-exclude" mechanism for frozen arms (ADR-0031 A1).

    This is the shared implementation of ADR-0031 A1's "register + dual-exclude"
    convention — *not* the wrapper-adapter/holder direction ADR-0031 rejected. A
    host declares its frozen arms once at construction (via
    :meth:`_register_frozen_arm`); the mixin then carries the invariant's
    mechanics — strip the arms off the checkpoint, strict-load the trainable keys
    with the arms as an explicit allow-list, and re-freeze ``eval()`` on the arms
    after ``module.train()`` — so a new frozen arm needs only a name, not a third
    copy of the three dunder overrides.

    The mixin does NOT own optimizer arm-selection: "which arm is optimized" stays
    in each host's ``configure_optimizers`` (GRPO → unet/controlnet, reward →
    reward_model, controlnet → controlnet). The mixin guarantees only the
    ``requires_grad=False`` layer; the host's optimizer simply never selects the
    frozen arms.

    MRO: declare the host as ``class XModule(FrozenArmMixin, spt.Module)`` so this
    mixin precedes ``spt.Module`` and the three dunder overrides chain to
    ``nn.Module`` via cooperative ``super()`` (neither ``spt.Module`` nor
    ``pl.LightningModule`` overrides ``state_dict`` / ``load_state_dict`` /
    ``train``, so ``super()`` resolves straight to ``nn.Module``).
    """

    #: The registered submodule prefixes kept off the optimizer + checkpoint
    #: (ADR-0031 dual-exclude). Declared once at construction via
    #: :meth:`_register_frozen_arm` (the arm set is fixed at init) and read by
    #: :meth:`state_dict` (strip) and :meth:`load_state_dict` (the strict-load
    #: allow-list). A class default so a host with no frozen arms degrades to a
    #: no-op rather than ``AttributeError``.
    _frozen_arm_names: frozenset[str] = frozenset()

    def _register_frozen_arm(self, name: str, arm: nn.Module) -> None:
        """Freeze + register a frozen arm and add it to ``_frozen_arm_names``.

        Applies the uniform frozen-arm prep (``eval()`` + ``requires_grad_(False)``)
        and registers ``arm`` as a normal submodule (so Lightning owns its device
        placement via the automatic ``.to(device)``). Call this in ``__init__`` for
        each frozen arm INSTEAD of a bare ``self.<name> = arm`` — it is the single
        place the "register + dual-exclude" prep is applied, so adding a frozen arm
        cannot forget the freeze. The arm set is fixed at construction; this is for
        init-time use only.
        """
        arm = arm.eval()
        for p in arm.parameters():
            p.requires_grad_(False)
        setattr(self, name, arm)
        # ``frozenset`` is not a Module/Parameter/Buffer, so nn.Module.__setattr__
        # stores it as a plain attribute (no object.__setattr__ bypass — the very
        # pattern ADR-0031 retires).
        self._frozen_arm_names = self._frozen_arm_names | frozenset({name})

    def _is_frozen_key(self, key: str) -> bool:
        """Whether a ``state_dict`` key belongs to a registered frozen arm.

        Matched on the key's top-level segment (the prefix before the first ``.``),
        so ``reward_model.conv.weight`` matches the ``reward_model`` arm. Non-frozen
        keys (a host's trainable arms) return ``False`` — they stay on the checkpoint
        and in the optimizer.
        """
        return key.split(".", 1)[0] in self._frozen_arm_names

    def state_dict(self, destination=None, prefix="", keep_vars=False, **kwargs):
        """Strip the registered frozen arms — they are rebuilt fresh each launch.

        The frozen arms stay off the checkpoint: each is reloaded from its own source
        at launch (the reward from its own ``.ckpt``, a reference policy via
        ``deepcopy``, a ControlNet-policy base from the native export). Registering
        the arms (so Lightning owns their device placement) would otherwise leak
        their multi-GB weights into the checkpoint; this override restores the
        off-checkpoint invariant at the source, so direct ``mod.state_dict()`` calls
        see them stripped (ADR-0031).

        The filter mutates the shared ``destination`` IN PLACE (and matches the arms
        under the caller-supplied ``prefix``): ``super().state_dict()`` writes every
        frozen-arm tensor into ``destination``, and a parent/wrapper that drives the
        recursion reads ``destination`` (the return value is the same object), so
        building a fresh dict would leave the multi-GB arms behind. Prefix-awareness
        makes a host nested under another ``nn.Module`` —
        ``state_dict(prefix="module.")`` (DDP) — strip them too.
        """
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs)
        frozen_prefixes = tuple(f"{prefix}{name}." for name in self._frozen_arm_names)
        for key in [k for k in destination if k.startswith(frozen_prefixes)]:
            del destination[key]
        return destination

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Strict load over the TRAINABLE keys; frozen arms are an explicit allow-list.

        The checkpoint never carries frozen-arm weights (:meth:`state_dict` strips
        them); the arms are rebuilt fresh each launch (ADR-0031). Strip any stray
        frozen keys a stale pre-refactor checkpoint may carry too, then enforce
        strict parity on the trainable keys ONLY — a missing/unexpected TRAINABLE
        key (an incomplete or mode-mismatched ``.ckpt``) surfaces loudly, NOT
        silently via a blanket ``strict=False`` (which would also hide a missing
        trainable key and resume on random or stale weights, corrupting the
        experiment). The frozen arms being absent from the checkpoint is the one
        tolerated mismatch.
        """
        incoming = type(state_dict)({k: v for k, v in state_dict.items() if not self._is_frozen_key(k)})
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None and hasattr(incoming, "__dict__"):
            incoming._metadata = metadata
        result = super().load_state_dict(incoming, strict=False, **kwargs)
        # super() reports the registered frozen arms as missing (present in the
        # module, absent in the incoming) — that is the allow-listed tolerance.
        # Anything else is a real trainable-key mismatch and (when strict) must raise.
        bad_missing = [k for k in result.missing_keys if not self._is_frozen_key(k)]
        bad_unexpected = [k for k in result.unexpected_keys if not self._is_frozen_key(k)]
        if strict and (bad_missing or bad_unexpected):
            raise RuntimeError(
                f"Error(s) loading state_dict for {type(self).__name__} (frozen arms "
                f"{sorted(self._frozen_arm_names)} allow-listed): missing "
                f"{len(bad_missing)} trainable key(s) {bad_missing[:5]}; unexpected "
                f"{len(bad_unexpected)} key(s) {bad_unexpected[:5]}."
            )
        return result

    def train(self, mode: bool = True):
        """Re-freeze ``eval()`` on the registered frozen arms after ``super().train()``.

        Registration makes Lightning's ``module.train(mode)`` recurse into the frozen
        arms and flip them to training mode — an ``eval()`` set at construction does
        NOT persist. A frozen arm in training mode would let its BatchNorm running
        stats drift during rollout / reward evaluation, corrupting the supposedly-fixed
        function. This override re-applies ``eval()`` to every present frozen arm after
        the recursive call (the mode-management cost registration buys; ADR-0031).
        """
        result = super().train(mode)
        for name in self._frozen_arm_names:
            arm = getattr(self, name, None)
            if arm is not None:
                arm.eval()
        return result
