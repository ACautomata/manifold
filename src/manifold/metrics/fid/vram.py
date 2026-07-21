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

``__enter__`` snapshots the VAE CPU state before moving it, lazy-builds the
feature_net from the factory (fail-safe), sets it to eval, and probes
``_feat_dim``. On any exception during ``__enter__``, the VAE is restored to CPU
before re-raising — Python does NOT call ``__exit__`` when ``__enter__`` raises,
so cleanup must be inlined.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

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
        self.vae = vae
        self._feature_net = feature_net
        self._feature_net_factory = feature_net_factory
        self._device_fn = device_fn
        self.feat_dim = feat_dim
        self.fid_disabled: bool = False
        self._staged: bool = False
        self._vae_cpu_state: dict[str, torch.Tensor] | None = None

    @property
    def feature_net(self) -> nn.Module | None:
        return self._feature_net

    def __enter__(self) -> "VramStage":
        """Stage the VAE + feature_net to GPU; lazy-build the feature_net.

        On any exception after ``vae.to(device)``, the VAE is restored to CPU
        before re-raising — Python does not call ``__exit__`` when ``__enter__``
        raises, so cleanup must be inlined here.
        """
        device = self._device_fn()
        # Snapshot VAE CPU state BEFORE moving it (so a partial move can be
        # undone by load_state_dict into a fresh .to("cpu") VAE).
        self._vae_cpu_state = {k: v.detach().clone() for k, v in self.vae.state_dict().items()}
        try:
            self.vae.to(device)
            # Mark staged BEFORE any further work that might raise — the finally
            # in on_validation_epoch_end calls _restore_eval_to_cpu only when
            # _eval_staged is True. A skip-path return before this flag would
            # leave the full VAE resident on the training GPU for the rest of
            # the run (the VRAM pressure the skip is meant to avoid).
            self._staged = True

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
            # Cleanup on error during __enter__: restore VAE to CPU so it
            # does not occupy training VRAM for the rest of the run.
            self._restore_to_cpu()
            raise

    def __exit__(self, *exc_info) -> None:
        """Restore VAE + feature_net to CPU unconditionally."""
        self._restore_to_cpu()
        return None

    def _restore_to_cpu(self) -> None:
        """Return VAE + feature_net to CPU (free VRAM for training)."""
        if self._staged:
            self.vae.to("cpu")
            if self._vae_cpu_state is not None:
                self.vae.load_state_dict(self._vae_cpu_state)
            if self._feature_net is not None:
                self._feature_net.to("cpu")
            self._staged = False
