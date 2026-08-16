"""VramStage context manager — stage/restore VAE + feature_net for the FID phase.

During training the VAE lives on CPU to free VRAM for the UNet. During validation
the UNet is idle, so the VAE + feature_net are moved to GPU for faster decode +
feature extraction. ``VramStage`` encapsulates that staging/restore cycle::

    with VramStage(vae, feature_net=fn, feature_net_factory=factory,
                   device_fn=device) as stage:
        if stage.fid_disabled:
            ...  # backbone absent -> skip FID
        # stage.feature_net is the resolved feature net (built from factory if needed)
        # stage.feat_dim is the probed feature dimension
        ...  # decode, extract features, reduce, log FID
    # VAE + feature_net are back on CPU here

The VAE stage/restore cycle is delegated to a composed
:class:`~manifold.metrics.vae_stage.VaeStage` (the single VAE-staging implementation);
``VramStage`` adds only the FID-specific concerns on top — the lazy feature_net
build (fail-safe), eval-mode set, and the ``feat_dim`` probe. ``__enter__`` does the
VAE move first (via ``VaeStage``, which snapshots the CPU state and restores on its
own failure), then the feature_net work; on any exception during the feature_net
work the composed ``VaeStage`` is unwound before re-raising — Python does NOT call
``__exit__`` when ``__enter__`` raises, so cleanup must be inlined.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from manifold.metrics.vae_stage import VaeStage

try:
    from lightning.pytorch.utilities.rank_zero import rank_zero_info
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    from pytorch_lightning.utilities.rank_zero import rank_zero_info  # type: ignore


class VramStage:
    """Context manager: stage VAE + feature_net to GPU, restore to CPU on exit.

    Args:
        vae: the held frozen VAE.
        feature_net: direct feature network (test seam); None means use the factory.
        feature_net_factory: lazy fail-safe factory ``() -> Module | None``.
        device_fn: ``() -> torch.device`` resolving the UNet's device at entry time.
        feat_dim: cached feature dim; probes once if None.

    Attributes (populated after ``__enter__``):
        feature_net: the resolved feature network (built from factory if needed).
        feat_dim: the probed feature dimension.
        fid_disabled: True if the feature_net is absent (both direct and factory are None / failed).
    """

    def __init__(
        self,
        vae: nn.Module,
        *,
        feature_net: nn.Module | None = None,
        feature_net_factory: Callable[[], nn.Module | None] | None = None,
        device_fn: Callable[[], torch.device],
        feat_dim: int | None = None,
    ) -> None:
        # The VAE stage/restore cycle is composed (the single VAE-staging path); this
        # stage adds only the feature_net / fid-disabled concern on top (ADR-0037).
        self._vae_stage = VaeStage(vae, device_fn=device_fn)
        self._feature_net = feature_net
        self._feature_net_factory = feature_net_factory
        self.feat_dim = feat_dim
        self.fid_disabled: bool = False

    @property
    def vae(self) -> nn.Module:
        return self._vae_stage.vae

    @property
    def _staged(self) -> bool:
        """Whether the VAE is staged (delegates to the composed ``VaeStage``).

        ``FIDCallback`` reads this to decide whether a manual ``__exit__`` is needed
        after a stage error (the error-rendezvous pattern), so it stays de-facto
        public surface.
        """
        return self._vae_stage._staged

    @property
    def feature_net(self) -> nn.Module | None:
        return self._feature_net

    def __enter__(self) -> "VramStage":
        """Stage the VAE + feature_net to GPU; lazy-build the feature_net.

        The VAE moves first (via the composed ``VaeStage``, which snapshots its CPU
        state and restores on its own failure). On any exception during the
        subsequent feature_net work, the composed ``VaeStage`` is unwound (VAE back
        to CPU) before re-raising — Python does not call ``__exit__`` when
        ``__enter__`` raises, so cleanup must be inlined here.
        """
        self._vae_stage.__enter__()
        device = self._vae_stage.device
        try:
            # Lazy feature_net build (fail-safe): a raising factory (bad/corrupt
            # cache, version mismatch) is caught -> feature_net stays None ->
            # FID is skipped gracefully.
            if self._feature_net is None and self._feature_net_factory is not None:
                try:
                    self._feature_net = self._feature_net_factory()
                except Exception:  # pragma: no cover - backbone load failure
                    rank_zero_info("RadImageNet backbone build failed; FID will be skipped.", exc_info=True)
                    self._feature_net = None

            if self._feature_net is None:
                self.fid_disabled = True
                return self

            self._feature_net.to(device)
            # eval so BatchNorm uses fixed running stats (RadImageNet ResNet50
            # is BN-based). In train mode every forward updates them, so the
            # raw arm would inherit stats drifted by the real/slow arms — and
            # since the raw arm is the checkpoint monitor, that contamination
            # would distort selection.
            self._feature_net.eval()

            # Probe the feature dim once (deterministic across ranks).
            if self.feat_dim is None:
                with torch.no_grad():
                    self.feat_dim = int(self._feature_net(
                        torch.zeros(1, 1, 64, 64, device=device)
                    ).shape[1])

            return self
        except Exception:
            # Cleanup on error during __enter__: restore the VAE to CPU (via the
            # composed VaeStage) so it does not occupy training VRAM for the rest
            # of the run.
            self._restore_to_cpu()
            raise

    def __exit__(self, *exc_info) -> None:
        """Restore VAE + feature_net to CPU unconditionally."""
        self._restore_to_cpu()
        return None

    def _restore_to_cpu(self) -> None:
        """Return VAE + feature_net to CPU (free VRAM for training)."""
        if self._vae_stage._staged:
            self._vae_stage._restore_to_cpu()
            if self._feature_net is not None:
                self._feature_net.to("cpu")
