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
from typing import Protocol

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
