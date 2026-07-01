"""Double-EMA callback (JiT's ``0.9999`` slow / ``0.9996`` fast).

Maintains two exponential-moving-average shadows of the UNet parameters,
updated after every optimizer step on every rank (DDP all-reduces the params
first, so the shadows stay in sync with no extra collective). Generation and
validation sample from the **slow** (largest-decay) shadow — the published EMA
model — via :meth:`DoubleEMACallback.swap_in`, which copies the slow shadow into
``module.unet`` in place; :meth:`restore` puts the raw optimizer weights back.

Because the Module's :meth:`~manifold.modules.LatentFlowModule.sample` shares
``self.unet`` with training, the in-place swap is seen by ``sample()`` with no
extra wiring (ADR-0005). The shadows are captured in the Lightning checkpoint
through the callback's :meth:`state_dict` / :meth:`load_state_dict`, so a resumed
run continues the EMA history instead of re-snapshotting from raw weights.

Shadows cover the UNet **parameters** (the trainable set; buffers are not EMA'd —
the MAISI UNet is GroupNorm-based). The EMA lives in a Lightning callback.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore


def slowest_shadow_index(decays: Sequence[float]) -> int:
    """Index of the largest-decay EMA shadow — the published (inference) model.

    The shadow with the slowest (largest) decay is the generation/eval copy baked
    as the inference UNet. Owned here so training (``_EMAShadows`` swap-in) and
    the ``.ckpt`` → native export bridge cite one policy for which shadow is
    published (ADR-0006).
    """
    return max(range(len(decays)), key=lambda i: decays[i])


class _EMAShadows:
    """One EMA shadow per decay over a fixed set of named parameters.

    Each shadow is a ``{name: tensor}`` dict mirroring the UNet's parameters;
    all start as a clone of the current weights. :meth:`update` is the standard
    in-place EMA ``ema ← decay·ema + (1 − decay)·p``.
    """

    def __init__(self, named_params: Iterable[tuple[str, Tensor]], decays: tuple[float, ...]):
        self.decays = tuple(float(d) for d in decays)
        params = {n: p.detach().clone() for n, p in named_params}
        self._names = list(params.keys())
        self.shadows: list[dict[str, Tensor]] = [
            {n: params[n].clone() for n in self._names} for _ in self.decays
        ]

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def slow_index(self) -> int:
        """Index of the largest-decay shadow (the published EMA model)."""
        return slowest_shadow_index(self.decays)

    @torch.no_grad()
    def update(self, named_params: Iterable[tuple[str, Tensor]]) -> None:
        for name, p in named_params:
            p_data = p.detach()
            for decay, shadow in zip(self.decays, self.shadows):
                shadow[name].mul_(decay).add_(p_data, alpha=1.0 - decay)

    def state(self) -> dict:
        return {"decays": list(self.decays), "shadows": self.shadows, "names": list(self._names)}

    def load_state(self, state: dict) -> None:
        self.decays = tuple(float(d) for d in state["decays"])
        self._names = list(state["names"])
        self.shadows = state["shadows"]


class DoubleEMACallback(pl.Callback):
    """Maintain double EMA shadows of ``module.unet``; swap the slow one for eval.

    Args:
        module: the :class:`~manifold.modules.LatentFlowModule` whose UNet params
            are shadowed (read once at construction).
        decays: ``(slow, fast)`` EMA decays. JiT's published ``(0.9999, 0.9996)``;
            the largest decay is the **slow** shadow used for generation/eval.
    """

    def __init__(self, module, decays: tuple[float, float] = (0.9999, 0.9996)):
        super().__init__()
        self._shadows = _EMAShadows(module.unet.named_parameters(), tuple(decays))
        #: Raw optimizer weights snapshotted by ``swap_in`` for ``restore``.
        self._saved: dict[str, Tensor] | None = None

    @property
    def decays(self) -> tuple[float, ...]:
        return self._shadows.decays

    @property
    def slow_index(self) -> int:
        return self._shadows.slow_index

    @torch.no_grad()
    def update(self, module) -> None:
        """Advance every shadow one EMA step from the current UNet params."""
        self._shadows.update(module.unet.named_parameters())

    @torch.no_grad()
    def swap_in(self, module) -> None:
        """Copy the **slow** shadow into ``module.unet`` (snapshot the raw first).

        Generation/validation run on the published EMA model. The raw optimizer
        weights are stashed so :meth:`restore` can put them back for the next
        optimizer step.
        """
        if self._saved is not None:
            raise RuntimeError("EMA swap_in called twice without restore().")
        slow = self._shadows.shadows[self._shadows.slow_index]
        self._saved = {n: p.detach().clone() for n, p in module.unet.named_parameters()}
        for n, p in module.unet.named_parameters():
            p.copy_(slow[n])

    @torch.no_grad()
    def restore(self, module) -> None:
        """Restore the raw optimizer weights snapshotted by :meth:`swap_in`."""
        if self._saved is None:
            return
        for n, p in module.unet.named_parameters():
            p.copy_(self._saved[n])
        self._saved = None

    # -- Lightning hooks ------------------------------------------------------

    def on_fit_start(self, trainer, module) -> None:
        # The shadows are plain tensors (not module buffers), so Lightning's
        # device move does not relocate them — follow the module's device here so
        # the in-place EMA update and swap stay on one device (CPU/GPU/MPS).
        device = next(module.unet.parameters()).device
        self._shadows.shadows = [
            {n: t.to(device) for n, t in shadow.items()} for shadow in self._shadows.shadows
        ]

    def on_train_batch_end(self, trainer, module, *args, **kwargs) -> None:
        # After spt's manual-optimization training_step (accumulation=1 → one
        # optimizer step per batch), the params are post-step; advance the EMA.
        self.update(module)

    # -- resume (Lightning captures these in the callback checkpoint state) ---

    def state_dict(self) -> dict:
        return self._shadows.state()

    def load_state_dict(self, state: dict) -> None:
        self._shadows.load_state(state)
