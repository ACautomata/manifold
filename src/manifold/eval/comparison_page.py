"""The before/after-GRPO comparison Artifact page builder (issue #230, ADR-0036).

A local, presentation-only tool layered on the #228 eval run's written outputs:
the ``metrics.json`` (provenance +, on the paired policy, before/after 3D
PSNR/SSIM), the slice-grid PNGs (recorded as basenames in the JSON), and each
training run's CSVLogger ``metrics.csv`` (the FID / reward curves). It assembles
them into **one self-contained HTML page** — every image embedded as a PNG data
URI, no external references, responsive — so the comparison is shareable as-is.

The page opens by stating *why the two policies use different metrics* (the JiT
policy is unconditional — reference-free FID + the reward trajectory; the
ControlNet policy is paired — full-reference 3D PSNR/SSIM), so the numbers are
not misread. When no ControlNet eval is given yet (GRPO-ControlNet S4 pending),
its section renders as a clearly-marked slot — the builder re-runs unchanged
once that eval exists.

Deliberately **not** a console entry (ADR-0033): assembling the page is a local
presentation step over pulled-home artifacts, not a runbook operation.
"""

from __future__ import annotations

import base64
import html
import json
import os
from io import BytesIO
from typing import NamedTuple

from ..metrics.metric_plot_callback import MetricsPlotCallback
from .before_after import _METRICS_FILE

#: The CSVLogger metric columns the curves/stats consume.
_FID_KEY = "val/fid"
_REWARD_KEY = "val/mean_reward"

#: Categorical slots 1/2 of the reference palette — a validated adjacent pair
#: (CVD ΔE 24.7 on the light surface); identity is carried by the legend too.
_BASE_COLOR = "#2a78d6"
_GRPO_COLOR = "#eb6834"

_DEFAULT_TITLE = "Manifold before/after-GRPO comparison"


class JitComparison(NamedTuple):
    """The JiT policy's page inputs.

    ``eval_dir`` is the #228 run dir (``metrics.json`` + slice-grid PNGs);
    ``before_csv`` / ``after_csv`` are the base and GRPO training runs'
    ``metrics.csv`` (the ``val/fid`` and ``val/mean_reward`` curves).
    """

    eval_dir: str
    before_csv: str | None
    after_csv: str | None


class ControlNetComparison(NamedTuple):
    """The ControlNet policy's page inputs: its #228 eval run dir alone.

    PSNR/SSIM come from the run's ``metrics.json``; the training CSVs carry no
    comparable scalar on both sides (ADR-0036), so none are read.
    """

    eval_dir: str


class ComparisonPageBuilder:
    """Assemble the shareable before/after-GRPO page from the eval artifacts.

    Args:
        dpi: curve render resolution (the slice grids arrive as finished PNGs).
    """

    def __init__(self, *, dpi: int = 150) -> None:
        self._dpi = int(dpi)

    # -- public seam ----------------------------------------------------------

    def build(
        self,
        *,
        jit: JitComparison,
        controlnet: ControlNetComparison | None = None,
        title: str = _DEFAULT_TITLE,
        out_path: str | None = None,
    ) -> str:
        """Build the self-contained page; return the HTML (and write ``out_path``).

        ``controlnet=None`` renders the clearly-marked pending slot (S4 not yet
        run); pass the eval dir once it exists and the section fills in.
        """
        page = "\n".join(
            [
                self._document_open(title),
                self._jit_section(jit),
                self._controlnet_section(controlnet),
                self._document_close(),
            ]
        )
        if out_path is not None:
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w") as fh:
                fh.write(page)
            os.replace(tmp_path, out_path)  # atomic rename on the same filesystem.
        return page

    # -- inputs ----------------------------------------------------------------

    @staticmethod
    def _read_metrics(eval_dir: str) -> dict:
        """The eval run's metrics JSON (the #228 driver's written payload)."""
        path = os.path.join(eval_dir, _METRICS_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{eval_dir!r} is not a before/after eval run dir (missing "
                f"metrics.json) — point it at the manifold-eval --output dir."
            )
        with open(path) as fh:
            return json.load(fh)

    @staticmethod
    def _read_csv_series(csv_path: str | None) -> dict[str, list[tuple[int, float]]]:
        """All metrics' ``(step, value)`` points from one training metrics.csv.

        Reuses the CSVLogger-parse + non-finite-filter of
        :meth:`~manifold.metrics.MetricsPlotCallback._read_series` (the FID
        ``+inf`` sentinel must not reach the curve). A missing file yields
        ``{}`` — the page degrades by dropping that run's curves, never by
        failing the build. Each file is parsed once per build and both the
        stats and the curves read from that one parse.
        """
        if not csv_path or not os.path.isfile(csv_path):
            return {}
        return MetricsPlotCallback._read_series(csv_path)

    # -- curves ----------------------------------------------------------------

    def _curve_png(self, lines, title: str) -> bytes:
        """Render one line chart to PNG bytes: 2 px lines, 8 px markers, hairline grid."""
        import matplotlib

        matplotlib.use("Agg")  # headless — the builder runs where slices were rendered.
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=self._dpi)
        try:
            for label, color, pts in lines:
                xs, ys = zip(*sorted(pts))
                ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4, label=label)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.grid(True, color="#e1e0d9", linewidth=0.6, alpha=0.7)
            if len(lines) > 1:
                ax.legend()  # two series need the identity channel; one names itself.
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            return buf.getvalue()
        finally:
            plt.close(fig)

    @staticmethod
    def _data_uri(png: bytes) -> str:
        """PNG bytes as a self-contained ``<img>`` source."""
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

    def _fid_figure(self, before_pts, after_pts) -> str:
        """The FID training curve: base vs GRPO, one axis (both are val/fid)."""
        if not before_pts and not after_pts:
            return ""
        lines = [
            entry
            for entry in (
                ("base (pre-GRPO)", _BASE_COLOR, before_pts),
                ("GRPO", _GRPO_COLOR, after_pts),
            )
            if entry[2]
        ]
        return self._figure(
            self._data_uri(self._curve_png(lines, _FID_KEY)),
            "Unbiased FID (val/fid, lower is better) — the same FID callback + "
            "real-latent cache on both sides.",
        )

    def _reward_figure(self, reward_pts) -> str:
        """The GRPO reward trajectory (only the GRPO run logs it).

        Orange like the GRPO line in the FID figure — color follows the entity
        across the page's charts, so "GRPO" stays one color throughout.
        """
        if not reward_pts:
            return ""
        return self._figure(
            self._data_uri(self._curve_png([("GRPO", _GRPO_COLOR, reward_pts)], _REWARD_KEY)),
            "Mean realism reward during GRPO training (val/mean_reward).",
        )

    # -- sections ----------------------------------------------------------

    def _jit_section(self, jit: JitComparison) -> str:
        metrics = self._read_metrics(jit.eval_dir)
        before_fid_pts = self._read_csv_series(jit.before_csv).get(_FID_KEY, [])
        after_series = self._read_csv_series(jit.after_csv)
        after_fid_pts = after_series.get(_FID_KEY, [])
        reward_pts = after_series.get(_REWARD_KEY, [])
        stats = []
        before_fid = [v for _, v in before_fid_pts]
        after_fid = [v for _, v in after_fid_pts]
        if before_fid and after_fid:
            stats.append(f"best val/fid: {min(before_fid):.2f} → {min(after_fid):.2f}")
        elif before_fid or after_fid:
            stats.append(f"best val/fid: {min(before_fid or after_fid):.2f}")
        if len(reward_pts) >= 2:
            # Ordered by step, matching the curve render — the stat and the chart
            # must agree on which endpoints "first → last" means.
            ordered = sorted(reward_pts)
            stats.append(f"{_REWARD_KEY}: {ordered[0][1]:.2f} → {ordered[-1][1]:.2f}")
        stats.append(
            f"same initial noise (seed={metrics['seed']}), "
            f"{metrics['num_inference_steps']} Heun steps, "
            f"{metrics['num_samples']} sample pair(s)"
        )
        return (
            '<section id="jit">\n<h2>JiT x0-denoiser — unconditional generation</h2>\n'
            + self._stats_list(stats)
            + "\n"
            + self._fid_figure(before_fid_pts, after_fid_pts)
            + self._reward_figure(reward_pts)
            + self._grid_figures(metrics, jit.eval_dir, "Same-initial-noise before|after "
                              "2.5D slice grids (rows xy/yz/zx) — any difference is "
                              "attributable to GRPO, not sampling noise.")
            + "</section>\n"
        )

    def _controlnet_section(self, controlnet: ControlNetComparison | None) -> str:
        if controlnet is None:
            return (
                '<section id="controlnet">\n<h2>ControlNet — paired MRI translation</h2>\n'
                '<div class="slot"><strong>Pending:</strong> the GRPO-ControlNet (S4) '
                "run has not completed, so this section holds its slot. Once its eval "
                "exists, re-run the page builder with "
                "<code>controlnet=ControlNetComparison(eval_dir=…)</code> — no "
                "rewrite. It will carry the offline 3D PSNR/SSIM (generated target vs "
                "real target) and the before|after|real-target slice grids.</div>\n"
                "</section>\n"
            )
        metrics = self._read_metrics(controlnet.eval_dir)
        before, after = metrics["before"], metrics["after"]
        stats = [
            f"3D PSNR: {before['psnr']:.2f} → {after['psnr']:.2f}",
            f"3D SSIM: {before['ssim']:.4f} → {after['ssim']:.4f}",
            f"same initial noise + conditioning (seed={metrics['seed']}), "
            f"{metrics['num_inference_steps']} Heun steps, "
            f"{metrics['num_samples']} pair(s)",
        ]
        return (
            '<section id="controlnet">\n<h2>ControlNet — paired MRI translation</h2>\n'
            + self._stats_list(stats)
            + "\n"
            + self._grid_figures(
                metrics,
                controlnet.eval_dir,
                "Same-noise, same-conditioning before|after|real-target 2.5D slice "
                "grids — the real target is the fidelity reference column.",
            )
            + "</section>\n"
        )

    def _grid_figures(self, metrics: dict, eval_dir: str, caption: str) -> str:
        """Every slice-grid PNG the metrics JSON records, embedded as data URIs."""
        figures = []
        for basename in metrics.get("grids", []):
            with open(os.path.join(eval_dir, basename), "rb") as fh:
                figures.append(self._figure(self._data_uri(fh.read()), caption))
        return "".join(figures)

    # -- HTML chrome --------------------------------------------------------

    @staticmethod
    def _figure(src: str, caption: str) -> str:
        return (
            f'<figure>\n<img src="{src}" alt="{html.escape(caption)}">\n'
            f"<figcaption>{html.escape(caption)}</figcaption>\n</figure>\n"
        )

    @staticmethod
    def _stats_list(stats: list[str]) -> str:
        items = "".join(f"<li>{html.escape(s)}</li>" for s in stats)
        return f'<ul class="stats">\n{items}</ul>'

    def _document_open(self, title: str) -> str:
        return (
            "<!DOCTYPE html>\n<html>\n<head>\n"
            '<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f"<title>{html.escape(title)}</title>\n<style>\n{self._css()}</style>\n"
            "</head>\n<body>\n<main>\n"
            f"<h1>{html.escape(title)}</h1>\n"
            "<p class=\"lede\">Why the two policies use different metrics: the "
            "<strong>JiT</strong> policy generates unconditionally — no per-sample "
            "ground truth exists — so its quality is measured reference-free, by "
            "<strong>Unbiased FID</strong> (<code>val/fid</code>, the same FID "
            "callback and real-latent cache on both sides, lower is better) plus the "
            "GRPO <strong>reward trajectory</strong> (<code>val/mean_reward</code>). "
            "The <strong>ControlNet</strong> policy is a paired translation — a real "
            "target exists per source — so its quality is measured by full-reference "
            "<strong>3D PSNR / SSIM</strong> of the generated target against the real "
            "target (ADR-0036); FID would score the frozen base's realism, not the "
            "translation. Before/after samples everywhere are generated under "
            "identical initial noise and conditioning.</p>\n"
        )

    @staticmethod
    def _document_close() -> str:
        return (
            "<footer>Assembled by manifold.eval.ComparisonPageBuilder from the "
            "before/after eval outputs (metrics.json + slice grids) and the training "
            "runs' metrics.csv. Re-runnable when GRPO-ControlNet (S4) completes.</footer>\n"
            "</main>\n</body>\n</html>\n"
        )

    @staticmethod
    def _css() -> str:
        return (
            "* { box-sizing: border-box; }\n"
            "body { margin: 0; background: #f9f9f7; color: #0b0b0b; "
            'font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }\n'
            "main { max-width: 1080px; margin: 0 auto; padding: 28px 20px 56px; }\n"
            "h1 { font-size: 24px; margin: 0 0 10px; }\n"
            "h2 { font-size: 18px; margin: 0 0 12px; }\n"
            ".lede { color: #52514e; line-height: 1.55; max-width: 72ch; }\n"
            "section { background: #fcfcfb; border: 1px solid rgba(11, 11, 11, 0.10); "
            "border-radius: 10px; padding: 20px 22px; margin-top: 28px; }\n"
            "figure { margin: 14px 0 0; }\n"
            "img { max-width: 100%; height: auto; border-radius: 6px; "
            "border: 1px solid rgba(11, 11, 11, 0.10); }\n"
            "figcaption { color: #52514e; font-size: 13px; margin-top: 6px; line-height: 1.45; }\n"
            "ul.stats { list-style: none; display: flex; flex-wrap: wrap; "
            "gap: 10px 28px; margin: 0; padding: 0; }\n"
            "ul.stats li { font-size: 14px; }\n"
            ".slot { border: 1px dashed #c3c2b7; border-radius: 8px; "
            "padding: 16px 18px; color: #52514e; line-height: 1.5; }\n"
            "code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; "
            "font-size: 0.92em; }\n"
            "footer { color: #898781; font-size: 12px; margin-top: 32px; }\n"
        )
