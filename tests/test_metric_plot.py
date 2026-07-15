"""MetricsPlotCallback tests (#119).

Drives the render pipeline via a real Lightning CSVLogger writing a sparse
metrics.csv under a tmp model_dir, plus a lightweight FakeTrainer (no GPU, no
fit) so the callback hooks run directly. Covers: multi-prefix render + atomic
write (no ``.tmp`` leftover), the rank-0 guard, inf-sentinel filtering, the
no-CSV / no-logger no-ops, single-point visibility, and per-epoch overwrite.
"""

from __future__ import annotations

import builtins
import logging
import os
from types import SimpleNamespace

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pytest
import torch
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, TensorDataset

from manifold.metrics import MetricsPlotCallback
from manifold.metrics.metric_plot_callback import _prefix
from manifold.training.trainer import build_trainer


def _trainer(loggers, is_global_zero=True):
    """A minimal stand-in exposing exactly what the callback reads off the Trainer."""
    return SimpleNamespace(
        loggers=loggers,
        is_global_zero=is_global_zero,
        default_root_dir=".",
    )


def _write_csv(logger, rows):
    """Log sparse metric rows (each at its own step) and flush once to disk.

    CSVLogger buffers in memory and flushes on ``save()``; one flush after all
    rows writes the true sparse layout (one column per metric key, empty cells
    where a metric was not logged at that step).
    """
    for step, metrics in rows:
        logger.log_metrics(metrics, step=step)
    logger.save()


def _is_valid_png(path):
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n" and os.path.getsize(path) > 100


@pytest.fixture
def csv_logger(tmp_path):
    return CSVLogger(save_dir=str(tmp_path), name="csv", version=0)


def test_renders_all_prefixes_and_is_atomic(tmp_path, csv_logger):
    _write_csv(csv_logger, [
        (0, {"epoch": 0, "train/loss_epoch": 1.0}),
        (1, {"epoch": 1, "train/loss_epoch": 0.8, "val/psnr": 20.0}),
        (2, {"epoch": 2, "val/psnr": 22.5}),
    ])
    cb = MetricsPlotCallback()
    cb.on_fit_end(_trainer([csv_logger]), module=None)

    png = tmp_path / "metrics.png"
    assert png.exists()
    assert _is_valid_png(str(png))
    assert not (tmp_path / "metrics.png.tmp").exists()  # atomic: no leftover


def test_rank_guard_skips_non_zero(tmp_path, csv_logger):
    _write_csv(csv_logger, [(0, {"epoch": 0, "train/loss_epoch": 1.0})])
    cb = MetricsPlotCallback()
    cb.on_fit_end(_trainer([csv_logger], is_global_zero=False), module=None)
    assert not (tmp_path / "metrics.png").exists()


def test_inf_sentinel_filtered(tmp_path, csv_logger):
    # FIDCallback logs val/fid = +inf when its backbone is disabled — the value
    # must be dropped (not plotted) so it cannot collapse axis autoscaling.
    _write_csv(csv_logger, [
        (0, {"epoch": 0, "val/fid": float("inf")}),
        (1, {"epoch": 1, "val/fid": 3.2}),
    ])
    cb = MetricsPlotCallback()
    cb.on_fit_end(_trainer([csv_logger]), module=None)
    assert _is_valid_png(str(tmp_path / "metrics.png"))


def test_no_csv_is_noop(tmp_path, csv_logger):
    # Logger present but no metrics written yet -> no metrics.csv on disk.
    cb = MetricsPlotCallback()
    cb.on_fit_end(_trainer([csv_logger]), module=None)
    assert not (tmp_path / "metrics.png").exists()


def test_single_point_visible(tmp_path, csv_logger):
    _write_csv(csv_logger, [(0, {"epoch": 0, "val/psnr": 21.0})])
    cb = MetricsPlotCallback()
    cb.on_fit_end(_trainer([csv_logger]), module=None)
    assert _is_valid_png(str(tmp_path / "metrics.png"))


def test_per_epoch_render_overwrites(tmp_path, csv_logger):
    _write_csv(csv_logger, [(0, {"epoch": 0, "train/loss_epoch": 1.0})])
    cb = MetricsPlotCallback()
    cb.on_train_epoch_end(_trainer([csv_logger]), module=None)
    png = tmp_path / "metrics.png"
    assert png.exists()
    # A second epoch appends data and overwrites the same path.
    _write_csv(csv_logger, [(1, {"epoch": 1, "train/loss_epoch": 0.5})])
    cb.on_train_epoch_end(_trainer([csv_logger]), module=None)
    assert _is_valid_png(str(png))


def test_no_csvlogger_is_noop(tmp_path):
    cb = MetricsPlotCallback()
    cb.on_fit_end(_trainer([]), module=None)
    assert not (tmp_path / "metrics.png").exists()


def test_build_trainer_registers_once(tmp_path):
    trainer = build_trainer(
        max_epochs=1,
        model_dir=str(tmp_path / "auto"),
        devices=1,
        accelerator="cpu",
    )
    assert sum(isinstance(cb, MetricsPlotCallback) for cb in trainer.callbacks) == 1

    existing = MetricsPlotCallback(filename="custom.png")
    trainer = build_trainer(
        max_epochs=1,
        model_dir=str(tmp_path / "existing"),
        callbacks=[existing],
        devices=1,
        accelerator="cpu",
    )
    plots = [cb for cb in trainer.callbacks if isinstance(cb, MetricsPlotCallback)]
    assert plots == [existing]


class _TinyModule(pl.LightningModule):
    """One-parameter module for the real Trainer.fit lifecycle test."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch, batch_idx):
        loss = (self.weight * batch[0]).mean().square()
        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_real_trainer_fit_writes_plot(tmp_path):
    logger = CSVLogger(save_dir=str(tmp_path), name="csv", version=0)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=logger,
        callbacks=[MetricsPlotCallback()],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_TinyModule(), DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1))
    assert _is_valid_png(str(tmp_path / "metrics.png"))


def test_read_series_edge_cases(tmp_path):
    path = tmp_path / "edge.csv"
    path.write_text(
        "step,lr,val/psnr,val/ssim\n"
        "9007199254740993,0.01,,nan\n"
        "5,,22.0,0.9\n"
        "3,,21.0,inf\n"
    )
    series = MetricsPlotCallback._read_series(str(path))
    assert series == {
        "lr": [(9007199254740993, 0.01)],
        "val/psnr": [(5, 22.0), (3, 21.0)],
        "val/ssim": [(5, 0.9)],
    }
    assert _prefix("lr") == "(root)"


def test_build_fig_has_expected_lines_and_closes_on_error(monkeypatch):
    series = {
        "lr": [(1, 0.01)],
        "train/loss": [(1, 1.0)],
        "val/psnr": [(2, 22.0), (1, 20.0)],  # non-monotonic input
        "val/ssim": [(1, 0.8)],
    }
    fig = MetricsPlotCallback._build_fig(series, plt)
    try:
        axes = [ax for ax in fig.axes if ax.get_visible()]
        # Root metric ("lr") gets its own axis, ordered first by _prefix.
        assert [ax.get_title() for ax in axes] == ["lr", "train/loss", "val/psnr", "val/ssim"]
        # Non-monotonic input is sorted to ascending step on both axes.
        psnr = axes[2].lines[0]
        assert list(psnr.get_xdata()) == [1, 2]
        assert list(psnr.get_ydata()) == [20.0, 22.0]
        assert all(ax.get_xlabel() == "step" for ax in axes)
    finally:
        plt.close(fig)

    before = set(plt.get_fignums())
    original_subplots = plt.subplots

    def failing_subplots(*args, **kwargs):
        failed_fig, axes = original_subplots(*args, **kwargs)
        monkeypatch.setattr(axes.flat[0], "plot", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        return failed_fig, axes

    monkeypatch.setattr(plt, "subplots", failing_subplots)
    with pytest.raises(RuntimeError, match="boom"):
        MetricsPlotCallback._build_fig({"train/loss": [(1, 1.0)]}, plt)
    assert set(plt.get_fignums()) == before


def test_missing_matplotlib_warns_once(tmp_path, csv_logger, monkeypatch, caplog):
    _write_csv(csv_logger, [(0, {"epoch": 0, "train/loss": 1.0})])
    original_import = builtins.__import__

    def missing_matplotlib(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("blocked for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_matplotlib)
    cb = MetricsPlotCallback()
    caplog.set_level(logging.WARNING)
    cb.on_train_epoch_end(_trainer([csv_logger]), module=None)
    cb.on_fit_end(_trainer([csv_logger]), module=None)
    messages = [r.message for r in caplog.records if "matplotlib is unavailable" in r.message]
    assert len(messages) == 1
    assert not (tmp_path / "metrics.png").exists()
