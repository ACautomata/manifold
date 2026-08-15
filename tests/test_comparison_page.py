"""ComparisonPageBuilder tests (#230, Seam 3 — ADR-0036).

Assembles the before/after-GRPO comparison page from tiny fixtures — a metrics
JSON + slice-grid PNGs (the #228 driver's outputs) + sparse ``metrics.csv``
files (the training runs') — on CPU, with no GPU and no real checkpoints.
Asserts external behaviour at the builder's public seam: the page's sections,
its metric values, embedded data-URI images, and the absence of any external
reference (self-contained by construction).
"""

from __future__ import annotations

import base64
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless — fixtures render tiny PNGs.
import matplotlib.pyplot as plt
import pytest

from manifold.eval import ComparisonPageBuilder, ControlNetComparison, JitComparison


def _tiny_png(path: str) -> bytes:
    """Write a small valid grayscale PNG; return its bytes (for data-URI checks)."""
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.imshow([[0.1, 0.9], [0.9, 0.1]], cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_axis_off()
    try:
        fig.savefig(path, format="png")
    finally:
        plt.close(fig)
    with open(path, "rb") as fh:
        return fh.read()


def _jit_eval_dir(tmp_path, *, policy: str = "jit", num_samples: int = 2) -> tuple[str, list[bytes]]:
    """A fake #228 eval run dir: metrics.json + one slice-grid PNG per sample."""
    eval_dir = tmp_path / f"eval_{policy}"
    eval_dir.mkdir()
    payloads = [_tiny_png(str(eval_dir / f"slice_grid_{i}.png")) for i in range(num_samples)]
    metrics = {
        "policy": policy,
        "seed": 7,
        "num_inference_steps": 15,
        "num_samples": num_samples,
        "grids": [f"slice_grid_{i}.png" for i in range(num_samples)],
    }
    if policy == "controlnet":
        metrics["before"] = {"psnr": 18.25, "ssim": 0.7123}
        metrics["after"] = {"psnr": 21.5, "ssim": 0.8046}
    (eval_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return str(eval_dir), payloads


def _write_csv(path, rows: list[dict]) -> str:
    """A sparse CSVLogger-style metrics.csv: one column per key, empty cells."""
    keys: list[str] = ["epoch", "step"]
    for row in rows:
        keys += [k for k in row if k not in keys]
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def _before_after_csvs(tmp_path):
    """The JiT sides' training CSVs: base logs val/fid; GRPO adds mean_reward."""
    before_csv = _write_csv(
        tmp_path / "before.csv",
        [
            {"epoch": 0, "step": 10, "val/fid": 17.32},
            {"epoch": 1, "step": 20, "val/fid": 18.0},
        ],
    )
    after_csv = _write_csv(
        tmp_path / "after.csv",
        [
            {"epoch": 0, "step": 10, "val/fid": 17.9, "val/mean_reward": -0.83},
            {"epoch": 1, "step": 20, "val/fid": 14.79, "val/mean_reward": -0.39},
        ],
    )
    return before_csv, after_csv


def _assert_no_external_references(html: str) -> None:
    assert "<script" not in html
    assert "<link" not in html
    assert "http://" not in html and "https://" not in html


def _img_srcs(html: str) -> list[str]:
    return re.findall(r'<img[^>]*\bsrc="([^"]*)"', html)


def test_jit_page_is_self_contained(tmp_path):
    eval_dir, png_payloads = _jit_eval_dir(tmp_path)
    before_csv, after_csv = _before_after_csvs(tmp_path)
    html = ComparisonPageBuilder().build(
        jit=JitComparison(eval_dir=eval_dir, before_csv=before_csv, after_csv=after_csv),
    )

    assert '<meta name="viewport"' in html  # responsive
    _assert_no_external_references(html)
    srcs = _img_srcs(html)
    assert len(srcs) >= len(png_payloads)  # >= the slice grids
    assert all(src.startswith("data:image/png;base64,") for src in srcs)
    # Every slice-grid PNG is embedded verbatim as a data URI.
    for payload in png_payloads:
        assert base64.b64encode(payload).decode() in html


def test_page_opener_explains_metric_choice(tmp_path):
    eval_dir, _ = _jit_eval_dir(tmp_path)
    before_csv, after_csv = _before_after_csvs(tmp_path)
    html = ComparisonPageBuilder().build(
        jit=JitComparison(eval_dir=eval_dir, before_csv=before_csv, after_csv=after_csv),
    )
    opener = html.split("<section", 1)[0]
    assert "FID" in opener and "PSNR" in opener and "SSIM" in opener
    assert "ControlNet" in opener and "JiT" in opener


def test_jit_section_shows_curves_and_grids(tmp_path):
    eval_dir, _ = _jit_eval_dir(tmp_path)
    before_csv, after_csv = _before_after_csvs(tmp_path)
    html = ComparisonPageBuilder().build(
        jit=JitComparison(eval_dir=eval_dir, before_csv=before_csv, after_csv=after_csv),
    )
    assert 'id="jit"' in html
    # The best val/fid of each side (independent truth: min of the CSV columns).
    assert "17.32" in html and "14.79" in html
    # The GRPO reward trajectory's endpoints.
    assert "-0.83" in html and "-0.39" in html
    # The same-noise pairing provenance is stated.
    assert "seed=7" in html
    # Curves are embedded images too (beyond the two slice grids).
    assert len(_img_srcs(html)) >= 4


def test_controlnet_section_filled(tmp_path):
    eval_dir, png_payloads = _jit_eval_dir(tmp_path, policy="controlnet", num_samples=1)
    jit_dir, _ = _jit_eval_dir(tmp_path)
    before_csv, after_csv = _before_after_csvs(tmp_path)
    html = ComparisonPageBuilder().build(
        jit=JitComparison(eval_dir=jit_dir, before_csv=before_csv, after_csv=after_csv),
        controlnet=ControlNetComparison(eval_dir=eval_dir),
    )
    assert 'id="controlnet"' in html
    assert "21.50" in html and "0.8046" in html  # after PSNR / SSIM (formatted)
    assert "18.25" in html and "0.7123" in html  # before PSNR / SSIM
    for payload in png_payloads:
        assert base64.b64encode(payload).decode() in html


def test_controlnet_slot_marked_when_absent(tmp_path):
    jit_dir, _ = _jit_eval_dir(tmp_path)
    before_csv, after_csv = _before_after_csvs(tmp_path)
    html = ComparisonPageBuilder().build(
        jit=JitComparison(eval_dir=jit_dir, before_csv=before_csv, after_csv=after_csv),
        controlnet=None,
    )
    controlnet_section = html.split('id="controlnet"', 1)[1]
    assert "pending" in controlnet_section.lower()


def test_missing_columns_skip_curves_page_still_complete(tmp_path):
    eval_dir, _ = _jit_eval_dir(tmp_path)
    # Neither CSV carries val/fid or val/mean_reward — no curves, but the page
    # (sections + grids + opener) is still fully assembled.
    before_csv = _write_csv(tmp_path / "b.csv", [{"epoch": 0, "step": 1, "train/loss": 1.0}])
    after_csv = _write_csv(tmp_path / "a.csv", [{"epoch": 0, "step": 1, "train/loss": 0.5}])
    html = ComparisonPageBuilder().build(
        jit=JitComparison(eval_dir=eval_dir, before_csv=before_csv, after_csv=after_csv),
    )
    assert 'id="jit"' in html
    assert len(_img_srcs(html)) == 2  # only the two slice grids


def test_build_writes_out_path(tmp_path):
    eval_dir, _ = _jit_eval_dir(tmp_path)
    before_csv, after_csv = _before_after_csvs(tmp_path)
    out = tmp_path / "page.html"
    html = ComparisonPageBuilder().build(
        jit=JitComparison(eval_dir=eval_dir, before_csv=before_csv, after_csv=after_csv),
        title="GRPO before/after",
        out_path=str(out),
    )
    assert out.is_file()
    assert out.read_text() == html
    assert "<title>GRPO before/after</title>" in html
