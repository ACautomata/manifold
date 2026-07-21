"""The callback registry: name → spec, two-phase resolve/build (ADR-0029).

A :class:`CallbackRegistry` maps a callback **name** to a spec dataclass class.
Specs are ``@dataclass`` types whose fields ARE the callback's config knobs
(with defaults). Construction is **two-phase**:

- :meth:`CallbackRegistry.resolve` (config-time): validate the requested names
  and their knob dicts, then construct the spec instances — **fail-fast** on an
  unknown name (``KeyError``) or an unknown knob for a known name
  (``ValueError``).
- :meth:`CallbackRegistry.build` (fit-prep): inject the runtime
  :class:`CallbackContext` and return the constructed ``pl.Callback`` list.

The split exists because generative callbacks (FID, future PSNR/SSIM) need
runtime objects that do not exist at config resolution. The tracer-bullet
(issue #159) wires only ``TrainLossLogger`` (no knobs); the remaining callbacks
migrate behind the registry in follow-on issues.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Protocol

import lightning.pytorch as pl

from manifold.training.callbacks.context import CallbackContext


class CallbackSpec(Protocol):
    """The spec contract: a typed callback declaration + its config knobs.

    Each concrete spec is a ``@dataclass``: its fields are the callback's config
    knobs (with defaults). :meth:`build` receives the runtime
    :class:`CallbackContext` and returns the constructed ``pl.Callback``. Specs
    match this Protocol structurally (composition, not subclassing — the OOP
    rule): a concrete spec declares its knobs as dataclass fields and a
    ``build`` method, without inheriting anything.

    A spec whose callback logs a monitored metric MAY declare a
    ``logged_metrics: frozenset[str]`` (e.g. ``FIDSpec`` logs ``val/fid``); the
    checkpoint spec declares a ``monitor_metric`` knob instead. Neither is part
    of the required Protocol — :meth:`CallbackRegistry.validate_monitor` reads
    them defensively via ``getattr``.
    """

    def build(self, ctx: CallbackContext) -> pl.Callback:
        """Construct the callback, injecting runtime objects from *ctx*."""
        ...


class CallbackRegistry:
    """Maps callback names to spec dataclasses; two-phase resolve/build."""

    def __init__(self) -> None:
        self._specs: dict[str, type] = {}

    def register(self, name: str, spec_cls: type) -> None:
        """Map *name* to a spec dataclass *spec_cls*.

        Raises:
            TypeError: if *spec_cls* is not a ``@dataclass`` class (specs carry
                their knobs as dataclass fields, so unknown-knob validation has
                a field set to check against).
        """
        if not (isinstance(spec_cls, type) and is_dataclass(spec_cls)):
            raise TypeError(
                f"spec must be a @dataclass class, got {spec_cls!r} "
                f"({type(spec_cls).__name__})"
            )
        self._specs[name] = spec_cls

    def resolve(self, names: list[str], cfg: dict | None = None) -> list[CallbackSpec]:
        """Validate *names* + per-name knob dicts; return constructed spec instances.

        Args:
            names: the callback names to resolve. The resolved list **must** be
                rank-symmetric — every DDP rank requests the identical list
                (``torchrun`` / Lightning spawn hand every rank the same CLI, so
                an asymmetric list is a never-observed manual-scripting failure).
            cfg: optional ``{name: {knob: value}}`` override map.

        Returns:
            The constructed spec instances, in *names* order.

        Raises:
            KeyError: on an unknown callback *name*.
            ValueError: on an unknown *knob* for a known name.
        """
        overrides = cfg or {}
        resolved = []
        for name in names:
            spec_cls = self._specs.get(name)
            if spec_cls is None:
                raise KeyError(
                    f"Unknown callback name {name!r}; "
                    f"registered: {sorted(self._specs) or '(none)'}"
                )
            allowed = {f.name for f in fields(spec_cls)}
            knobs = overrides.get(name, {}) or {}
            unknown = set(knobs) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown knob(s) for callback {name!r}: {sorted(unknown)}; "
                    f"allowed: {sorted(allowed) or '(none)'}"
                )
            resolved.append(spec_cls(**{k: knobs[k] for k in knobs if k in allowed}))
        return resolved

    def build(self, specs: list[CallbackSpec], ctx: CallbackContext) -> list[pl.Callback]:
        """Construct ``pl.Callback`` instances from *specs*, injecting *ctx*."""
        return [spec.build(ctx) for spec in specs]

    def validate_monitor(
        self, specs: list[CallbackSpec], module: Any, extra_callbacks: list | None = None
    ) -> None:
        """Post-resolve monitor validation (ADR-0029): the checkpoint spec's
        ``monitor_metric`` must be logged by some resolved callback **or**
        declared by the training *module*.

        The module side of the union covers reward / paired-reward / GRPO
        monitors (``val/gen_pair_acc``, ``val/mean_reward``) that the Module
        logs directly rather than through any resolved callback. A module
        declares them via a ``logged_metrics`` attribute (absent -> empty, the
        JiT case where every val/* metric comes from a callback). An explicit
        ``monitor_metric=None`` (the checkpoint's unmonitored periodic / last
        path) bypasses validation — absence is the intended fallback, not a
        missing-but-expected monitor.

        *extra_callbacks* (non-registry callbacks like the hand-appended
        ``LatentX0MAE``) are scanned for a ``logged_metrics`` attribute too, so a
        checkpoint monitoring a metric an extra callback emits validates without
        the shell having to mutate ``module.logged_metrics``.

        Raises:
            ValueError: if ``monitor_metric`` is set but is neither
                callback-logged nor module-declared (Lightning would otherwise
                error mid-fit on a never-logged monitor).
        """
        ckpt = next((s for s in specs if hasattr(s, "monitor_metric")), None)
        if ckpt is None or ckpt.monitor_metric is None:
            return  # no checkpoint spec, or the unmonitored path — nothing to check.
        logged: set[str] = set()
        for spec in specs:
            logged |= set(getattr(spec, "logged_metrics", frozenset()))
        logged |= set(getattr(module, "logged_metrics", frozenset()))
        for cb in extra_callbacks or []:
            logged |= set(getattr(cb, "logged_metrics", frozenset()))
        if ckpt.monitor_metric not in logged:
            raise ValueError(
                f"checkpoint monitor_metric {ckpt.monitor_metric!r} is logged by no "
                f"resolved callback and is not declared by the module "
                f"({type(module).__name__}). Available: {sorted(logged) or '(none)'}."
            )
