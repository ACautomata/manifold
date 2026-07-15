"""Build a Lightning ``Trainer`` for JiT latent-flow training.

A thin factory building a Lightning ``Trainer`` for JiT latent-flow training:
AMP via ``precision`` (``16-mixed`` on CUDA else ``32-true``), DDP with
``find_unused_parameters=True`` (the class-embedding row may be untouched when
labels are dropped), the spt :class:`~stable_pretraining.callbacks.ModuleRegistryCallback`
appended **once** so spt-side ``log()`` reaches the logger, and CSV +
TensorBoard loggers under the model dir.

The callbacks the trainer needs beyond spt's own (the
train-metrics callbacks) are passed in by the caller; :func:`build_trainer` only
owns the invariant wiring (precision, strategy, registry callback, loggers).
"""

from __future__ import annotations

import lightning.pytorch as pl
import stable_pretraining as spt
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import CSVLogger

from ..metrics.metric_plot_callback import MetricsPlotCallback

try:  # TensorBoard is optional — only added when the package is importable.
    from lightning.pytorch.loggers import TensorBoardLogger

    import tensorboard  # noqa: F401  (probes the runtime dep the logger needs)
except ImportError:  # pragma: no cover — TB absent on CPU CI / minimal installs
    TensorBoardLogger = None  # type: ignore[assignment]


def _tensorboard_available() -> bool:
    return TensorBoardLogger is not None


def _have_registry(callbacks) -> bool:
    return any(isinstance(c, spt.callbacks.ModuleRegistryCallback) for c in callbacks)


def _have_metrics_plot(callbacks) -> bool:
    return any(isinstance(c, MetricsPlotCallback) for c in callbacks)


def build_trainer(
    *,
    max_epochs: int,
    callbacks: list | None = None,
    model_dir: str,
    devices: int | str = "auto",
    accelerator: str = "auto",
    precision: str | None = None,
    log_every_n_steps: int = 1,
    limit_val_batches: int | float | None = None,
    extra_kwargs: dict | None = None,
) -> Trainer:
    """Construct the Lightning ``Trainer`` for JiT latent-flow training.

    Args:
        max_epochs: the training epoch horizon.
        callbacks: project callbacks (train-metrics, FID, ModelCheckpoint).
            The spt ``ModuleRegistryCallback`` is appended once if absent so
            spt-side ``log()`` reaches the logger.
        model_dir: checkpoint / log output dir; CSV + TensorBoard log under it.
        devices / accelerator: Lightning device selection (default ``"auto"``).
        precision: ``None`` → ``"16-mixed"`` on CUDA else ``"32-true"``.
        limit_val_batches: cap on validation batches (bounds the cheap x0-MAE
            loop's time). Passed through to the Trainer.
        extra_kwargs: escape hatch for additional ``Trainer(...`` kwargs.
    """
    callbacks = list(callbacks or [])
    if not _have_registry(callbacks):
        # Appended once so spt's own logging (registry/online metrics) reaches
        # the logger — the project callbacks below are independent of it.
        callbacks.append(spt.callbacks.ModuleRegistryCallback())
    if not _have_metrics_plot(callbacks):
        # Metrics line-chart PNG under model_dir, re-rendered every epoch +
        # at fit end (crash-robust on remote DCU). No-op if matplotlib is
        # absent — the callback swallows the ImportError into a warning.
        callbacks.append(MetricsPlotCallback())

    if precision is None:
        precision = "16-mixed" if __cuda_available() else "32-true"

    # DDP with find_unused_parameters: the class-embedding row may receive no
    # gradient when labels are dropped, which strict DDP rejects. Single-device
    # runs use the default strategy. ``is_multi_gpu`` is the shared predicate -
    # the CLIs' checkpoint monitor guard calls it too so monitor/strategy agree.
    if is_multi_gpu(devices):
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    logger: list = [CSVLogger(save_dir=model_dir, name="csv")]
    if _tensorboard_available():
        logger.append(TensorBoardLogger(save_dir=model_dir, name="tb"))

    kwargs: dict = dict(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        strategy=strategy,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=log_every_n_steps,
    )
    if limit_val_batches is not None:
        kwargs["limit_val_batches"] = limit_val_batches
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return Trainer(**kwargs)


def __cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def __cuda_device_count() -> int:
    import torch

    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def is_multi_gpu(devices: int | str) -> bool:
    """Whether a ``devices`` selection will spawn DDP (the shared monitor guard).

    Mirrors :func:`build_trainer`'s strategy decision exactly, so the checkpoint
    monitor guard and the trainer agree on whether DDP is in effect: an explicit
    ``int > 1`` OR ``"auto"`` on a host with ``> 1`` visible CUDA devices. The
    ``"auto"`` branch reuses the ``is_available()``-guarded
    :func:`__cuda_device_count` (a bare ``device_count()`` would diverge under
    ``CUDA_VISIBLE_DEVICES`` / on a CUDA-less host). Single-GPU (``1`` or
    ``"auto"`` on a 1-GPU box) returns ``False`` -> selection stays on.
    """
    return (isinstance(devices, int) and devices > 1) or (
        devices == "auto" and __cuda_device_count() > 1
    )
