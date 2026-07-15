"""Post-epoch metrics line-chart callback.

Renders all logged scalar metrics from the CSVLogger's ``metrics.csv`` into one
PNG line chart under the model dir, re-rendered every epoch AND at fit end.
Registered by default in :func:`manifold.training.build_trainer`, so every
train CLI gets it.

Design (vetted by a parallel recon of the Lightning source + adversarial review):

- **Crash-robust, not just final.** The chart is re-rendered every
  ``on_train_epoch_end`` AND at ``on_fit_end``. Training on remote DCU clusters
  frequently hangs or is SIGKILL'd, so a single ``on_fit_end`` render is
  unreliable — re-rendering each epoch means the last-completed epoch's PNG
  survives a crash.
- **Intermediate renders lag one epoch on epoch-level metrics.** Lightning
  flushes epoch-level metrics (``val/*``, ``train/loss_epoch``) at
  ``fit_loop.on_advance_end`` (~line 497), AFTER all ``on_train_epoch_end``
  hooks return (per-step train metrics ARE flushed within the batch loop). So a
  chart rendered inside ``on_train_epoch_end`` is one epoch stale on precisely
  the metrics most worth watching. The ``on_fit_end`` render is the
  authoritative complete picture (every epoch's flush is already persisted by
  then). This lag is documented, not "fixed": merging ``trainer.callback_metrics``
  would double-count stale val values under validate-every-N-epochs.
- **Atomic write.** The PNG is written to ``<png>.tmp`` then ``os.replace``'d
  onto ``<png>`` (same-filesystem atomic rename). A hard kill mid-render leaves
  the previous good PNG, not a truncated file — an in-place overwrite would
  defeat the crash-robustness rationale above.
- **DDP-safe.** ``on_train_epoch_end`` / ``on_fit_end`` fire on ALL ranks
  (Lightning has no rank gate there), so the render is guarded by
  ``trainer.is_global_zero`` — only rank 0 reads the CSV and writes the PNG.
- **Headless + lazy import.** matplotlib is imported with the ``Agg`` backend
  INSIDE the render try/except, AFTER the rank guard: non-zero ranks never pay
  the import, and a missing matplotlib (it is a declared dep, but the sugon DCU
  cluster installs with ``--no-deps``) degrades to a one-shot warning rather
  than aborting fit.
- **Non-fatal.** The whole render is wrapped in try/except; any failure logs a
  warning and returns. Plotting must never interrupt training.
- **No pandas.** The sparse CSV (``epoch, step, <one col per metric>``, empty
  cells where a metric was not logged) is parsed with stdlib :mod:`csv` — pandas
  would be an undeclared heavy transitive dep for a trivial parse. Only FINITE
  values are plotted (``math.isfinite``), so the ``val/fid = +inf`` sentinel
  that :class:`~manifold.metrics.FIDCallback` logs when its backbone is disabled
  cannot collapse matplotlib's axis autoscale.
- **Small multiples, one metric per axis.** FID, PSNR, SSIM, loss, and
  learning rate have incompatible scales; sharing a y-axis would flatten the
  smaller-valued series, while a dual axis would be ambiguous. Each finite metric
  therefore gets its own subplot, ordered by prefix + name. A single-series plot
  needs no legend — its full metric key is the title. The line uses the validated
  default blue ``#2a78d6``, a 2 px stroke, and 8 px markers; gridlines stay
  recessive. Zero finite data points → the write is skipped (no blank PNG).
"""

from __future__ import annotations

import csv
import logging
import math
import os

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore[assignment]

from lightning.pytorch.loggers import CSVLogger as _CSVLogger

_log = logging.getLogger(__name__)

#: The CSVLogger's on-disk metrics file (``<log_dir>/metrics.csv``).
_METRICS_FILE = "metrics.csv"


def _prefix(metric: str) -> str:
    """The subplot group for a metric key: its ``xxx/`` head, or ``"(root)"``.

    ``"val/psnr"`` → ``"val/"``; ``"train/loss_epoch"`` → ``"train/"``; a bare
    ``"lr"`` → ``"(root)"``.
    """
    return metric.split("/", 1)[0] + "/" if "/" in metric else "(root)"


class MetricsPlotCallback(pl.Callback):
    """Render logged metrics to a line-chart PNG each epoch + at fit end.

    Args:
        filename: PNG filename under the CSVLogger's ``save_dir`` (the model
            dir). Default ``"metrics.png"``.
        dpi: render resolution.
    """

    def __init__(self, *, filename: str = "metrics.png", dpi: int = 150) -> None:
        super().__init__()
        self._filename = filename
        self._dpi = dpi
        self._matplotlib_unavailable = False

    # -- hooks -----------------------------------------------------------------

    def on_train_epoch_end(self, trainer, module) -> None:  # noqa: D401
        # Intermediate, crash-survivable render (lags one epoch on epoch-level
        # metrics — see module docstring). Cheap relative to an epoch.
        self._render(trainer)

    def on_fit_end(self, trainer, module) -> None:  # noqa: D401
        # Authoritative final render — every epoch's flush is already persisted.
        self._render(trainer)

    # -- render pipeline -------------------------------------------------------

    def _render(self, trainer) -> None:
        # Hooks fire on ALL DDP ranks (no rank gate in on_train_epoch_end /
        # on_fit_end); only rank 0 may read + write to avoid an 8-way file race.
        if not getattr(trainer, "is_global_zero", True):
            return
        if self._matplotlib_unavailable:
            return

        plt = None
        fig = None
        try:
            try:
                import matplotlib

                matplotlib.use("Agg")  # headless — remote DCU servers have no display.
                import matplotlib.pyplot as plt
            except ImportError:
                # A sugon deployment may use ``pip install -e . --no-deps``.
                # Warn once — retrying the same absent import every epoch only
                # floods the training log; unrelated render failures still retry.
                self._matplotlib_unavailable = True
                _log.warning(
                    "MetricsPlotCallback: matplotlib is unavailable; plotting disabled.",
                    exc_info=True,
                )
                return

            csv_path, out_dir = self._resolve_paths(trainer)
            if csv_path is None or not os.path.isfile(csv_path):
                return  # no metrics yet (fit ended before epoch 1) or no CSVLogger.
            series = self._read_series(csv_path)
            if not series:
                return  # zero finite data points — skip the write (no blank PNG).

            fig = self._build_fig(series, plt)
            tmp_path = os.path.join(out_dir, self._filename + ".tmp")
            final_path = os.path.join(out_dir, self._filename)
            # format="png" explicitly: the .tmp suffix is not a recognized image
            # extension, so matplotlib cannot infer the format from it.
            fig.savefig(tmp_path, format="png", dpi=self._dpi, bbox_inches="tight")
            os.replace(tmp_path, final_path)  # atomic rename on the same filesystem.
        except Exception:
            _log.warning("MetricsPlotCallback: render failed; plot not updated.", exc_info=True)
        finally:
            # Always release the figure (a long sugon run would otherwise leak).
            if fig is not None and plt is not None:
                try:
                    plt.close(fig)
                except Exception:  # pragma: no cover — close never matters post-failure
                    pass

    def _resolve_paths(self, trainer) -> tuple[str | None, str]:
        """The live CSVLogger's metrics.csv + the dir to write the PNG into.

        Discovers the CSVLogger from ``trainer.loggers`` rather than globbing a
        ``version_X`` dir: a resumed run creates a new ``version_{N+1}``, and
        under DDP the version-resolution race can leave an empty dir — only the
        live logger's own ``log_dir`` is guaranteed to be the one being written.
        The PNG goes at the logger's ``save_dir`` (== ``model_dir``), a stable
        path with no version-hunt, mirroring how ``ModelCheckpoint`` exposes
        ``last.ckpt``.
        """
        loggers = getattr(trainer, "loggers", []) or []
        csv_logger = next((lg for lg in loggers if isinstance(lg, _CSVLogger)), None)
        if csv_logger is None:
            return None, getattr(trainer, "default_root_dir", ".")
        return os.path.join(csv_logger.log_dir, _METRICS_FILE), csv_logger.save_dir

    @staticmethod
    def _read_series(csv_path: str) -> dict[str, list[tuple[int, float]]]:
        """Parse the sparse CSV into ``{metric: [(step, value), ...]}``.

        Skips empty cells AND non-finite values (NaN, ±inf) — the latter so the
        ``val/fid = +inf`` sentinel (logged when the FID backbone is disabled)
        cannot collapse matplotlib's axis autoscale.
        """
        series: dict[str, list[tuple[int, float]]] = {}
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                step_field = row.get("step")
                if step_field in (None, ""):
                    continue
                try:
                    step = int(step_field)
                except ValueError:
                    continue
                for key, raw in row.items():
                    if key in (None, "step", "epoch") or raw in (None, ""):
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        continue
                    series.setdefault(key, []).append((step, value))
        return series

    @staticmethod
    def _build_fig(series, plt):
        """One small-multiple subplot per metric (incompatible scales stay separate).

        FID, PSNR, SSIM, loss, and learning rate must not share a y-axis: the
        smaller-valued series would become unreadable. Each subplot has one line,
        so its full metric-key title replaces a redundant legend. ``markersize=4``
        means an 8 px marker diameter; the 2 px line follows the static-chart mark
        spec and keeps single early points visible.
        """
        keys = sorted(series, key=lambda key: (_prefix(key), key))
        ncols = min(len(keys), 3)
        nrows = max(1, math.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
        try:
            for ax, key in zip(axes.flat, keys):
                pts = sorted(series[key])
                xs = [s for s, _ in pts]
                ys = [v for _, v in pts]
                ax.plot(
                    xs,
                    ys,
                    color="#2a78d6",
                    marker="o",
                    markersize=4,
                    linewidth=2,
                )
                ax.set_title(key)
                ax.set_xlabel("step")
                ax.grid(True, color="#deddda", linewidth=0.6, alpha=0.7)
            for ax in list(axes.flat)[len(keys):]:
                ax.set_visible(False)
            fig.tight_layout()
            return fig
        except BaseException:
            # ``fig = self._build_fig(...)`` in the caller assigns only AFTER
            # this function returns. If construction fails after subplots(), the
            # outer finally cannot see the partial figure — close it here before
            # preserving the original exception (including KeyboardInterrupt).
            plt.close(fig)
            raise
