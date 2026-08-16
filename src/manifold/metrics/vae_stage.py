"""VaeStage context manager — stage/restore the VAE only, for an eval decode.

During training the VAE lives on CPU to free VRAM for the UNet. During a
validation decode the UNet is idle, so the VAE is moved to the device for a faster
decode, then restored to CPU. ``VaeStage`` encapsulates that VAE-only cycle,
**decoupled from the feature_net / fid-disabled concern** that
:class:`~manifold.metrics.fid.VramStage` adds for FID (ADR-0037 — the paired-fidelity
monitor stages the VAE for the decode and nothing else)::

    with VaeStage(vae, device_fn=device) as stage:
        ...  # decode on the device
    # the VAE is back on CPU here

It is the single VAE-staging implementation: :class:`~manifold.metrics.fid.VramStage`
composes it, so the paired-fidelity monitor and FID share one snapshot/move/restore
path that cannot drift.

``__enter__`` snapshots the VAE CPU state before moving it. On any exception during
``__enter__``, the VAE is restored to CPU before re-raising — Python does NOT call
``__exit__`` when ``__enter__`` raises, so cleanup must be inlined.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


class VaeStage:
    """Context manager: stage the VAE to a device for a decode, restore to CPU on exit.

    Args:
        vae: the held frozen VAE.
        device_fn: ``() -> torch.device`` resolving the target device at entry time
            (the module / UNet's device).

    Attributes (populated after ``__enter__``):
        device: the resolved device the VAE was staged to.
    """

    def __init__(self, vae: nn.Module, *, device_fn: Callable[[], torch.device]) -> None:
        self.vae = vae
        self._device_fn = device_fn
        self.device: torch.device | None = None
        self._staged: bool = False
        self._vae_cpu_state: dict[str, torch.Tensor] | None = None

    def __enter__(self) -> "VaeStage":
        """Stage the VAE to the device.

        On any exception during ``vae.to(device)``, the VAE is restored to CPU before
        re-raising — Python does not call ``__exit__`` when ``__enter__`` raises, so
        cleanup must be inlined here.
        """
        device = self._device_fn()
        self.device = device
        # Snapshot VAE CPU state BEFORE moving it (so a partial move can be undone by
        # load_state_dict into a fresh .to("cpu") VAE).
        self._vae_cpu_state = {k: v.detach().clone() for k, v in self.vae.state_dict().items()}
        # Staged flag BEFORE the fallible move: if ``vae.to(device)`` fails partway
        # (e.g., staging OOM), ``_restore_to_cpu()`` moves any already-moved
        # parameters back instead of being a no-op (codex #171 P2).
        self._staged = True
        try:
            self.vae.to(device)
            return self
        except Exception:
            # Cleanup on error during __enter__: restore the VAE to CPU so it does not
            # occupy training VRAM for the rest of the run.
            self._restore_to_cpu()
            raise

    def __exit__(self, *exc_info) -> None:
        """Restore the VAE to CPU unconditionally."""
        self._restore_to_cpu()
        return None

    def _restore_to_cpu(self) -> None:
        """Return the VAE to CPU (free VRAM for training)."""
        if self._staged:
            self.vae.to("cpu")
            if self._vae_cpu_state is not None:
                self.vae.load_state_dict(self._vae_cpu_state)
            self._staged = False
