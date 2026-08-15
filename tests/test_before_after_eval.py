"""Before/after eval-driver core tests (issue #228, ADR-0036).

The highest seam (Seam 2): inject tiny ``LatentFlowPipeline`` /
``ControlNetLatentFlowPipeline`` objects, feed a fixed seed, and assert the
**same-noise pairing invariant** plus well-formed outputs (valid slice-grid PNG(s)
+ a metrics JSON) — all on CPU with tiny components (prior art:
``test_pipeline_inference.py`` / ``test_controlnet_pipeline_inference.py`` for the
tiny pipelines, ``test_metric_plot.py`` for PNG validity).

The strongest *behavioral* proof of the pairing invariant: inject the **same**
pipeline object as both before and after. Identical initial noise + identical
conditioning + identical weights ⇒ the before and after volumes must be equal; any
divergence means the driver let the noise (or conditioning) drift between sides.
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch
import torch.nn as nn

from manifold import (
    AutoencoderKL,
    ControlNet3DConditionModel,
    ControlNetLatentFlowPipeline,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowPipeline,
    UNet3DConditionModel,
)
from manifold.eval import BeforeAfterEval
from manifold.metrics import PairedFidelityMetrics

# Tiny shapes consistent with the default 2-level VAE (latent divisor 2). The
# unconditional (JiT) grid needs no SSIM, so a small 4³ latent (→ 8³ image) is fine.
JIT_LATENT_SHAPE = (1, 4, 4, 4, 4)
# The paired pass scores 3D SSIM (MONAI default win_size=11), so the decoded image
# must be ≥ 11 voxels per spatial dim: latent 8³ → image 16³ clears the floor.
PAIRED_LATENT_SHAPE = (1, 4, 8, 8, 8)
PAIRED_IMAGE_SHAPE = (1, 1, 16, 16, 16)


def _jit_pipeline() -> LatentFlowPipeline:
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    vae = AutoencoderKL(scaling_factor=0.5)
    return LatentFlowPipeline(unet, vae, FlowMatchHeunDiscreteScheduler())


def _controlnet_pipeline() -> ControlNetLatentFlowPipeline:
    """A tiny base + ControlNet with the zero-init output conv re-initialized.

    Mirrors ``tests/test_controlnet_pipeline_inference._frozen_base`` — MONAI MAISI
    zero-initializes the final projection, so re-initializing it lets the full
    base→ControlNet path run end-to-end.
    """
    torch.manual_seed(0)
    base = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in base.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    torch.manual_seed(1)
    controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    controlnet.load_base_encoder_weights(base)
    vae = AutoencoderKL(scaling_factor=0.5)
    return ControlNetLatentFlowPipeline(base, controlnet, vae, FlowMatchHeunDiscreteScheduler())


# -- unconditional (JiT): same-noise pairing ---------------------------------


def _is_valid_png(path):
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n" and os.path.getsize(path) > 100


def test_unconditional_same_pipeline_gives_identical_before_after(tmp_path):
    """The fairness invariant: same pipeline + same seed ⇒ before == after.

    Injecting one pipeline as both sides isolates the driver's seeding: if the
    before/after noise or conditioning drifted, the two volumes would differ even
    though the weights are identical.
    """
    pipe = _jit_pipeline()
    result = BeforeAfterEval().run_unconditional(
        pipe,
        pipe,
        target_shape=JIT_LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert torch.allclose(result.before, result.after)


def test_unconditional_writes_valid_slice_grid_png(tmp_path):
    """One 2.5D three-plane before|after grid PNG per sample, well-formed."""
    result = BeforeAfterEval().run_unconditional(
        _jit_pipeline(),
        _jit_pipeline(),
        target_shape=JIT_LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert len(result.grids) == JIT_LATENT_SHAPE[0]  # one grid per sample (batch 1)
    for grid in result.grids:
        assert _is_valid_png(grid)


def test_unconditional_writes_metrics_json(tmp_path):
    """The unconditional pass writes a well-formed metrics JSON (provenance only —
    no fidelity scalars, which are the paired policy's)."""
    result = BeforeAfterEval().run_unconditional(
        _jit_pipeline(),
        _jit_pipeline(),
        target_shape=JIT_LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text())
    assert payload["policy"] == "jit"
    assert payload["seed"] == 0
    assert payload["num_inference_steps"] == 3
    assert payload["num_samples"] == JIT_LATENT_SHAPE[0]
    # Grids are recorded as portable basenames (resolved relative to the JSON's dir).
    assert payload["grids"] == [os.path.basename(g) for g in result.grids]
    assert "psnr" not in payload and "ssim" not in payload
    assert result.metrics == payload


# -- paired (ControlNet): same-noise + real-target reference ------------------


def _paired_inputs():
    """A fixed src control signal + real target latent (distinct seeds ⇒ distinct)."""
    src = torch.randn(*PAIRED_LATENT_SHAPE, generator=torch.Generator().manual_seed(10))
    real = torch.randn(*PAIRED_LATENT_SHAPE, generator=torch.Generator().manual_seed(20))
    return src, real


def test_paired_same_pipeline_gives_identical_before_after(tmp_path):
    """The fairness invariant holds for the paired policy too: same pipeline + same
    seed ⇒ before == after (the driver passes one shared noise tensor to both)."""
    pipe = _controlnet_pipeline()
    src, real = _paired_inputs()
    result = BeforeAfterEval().run_paired(
        pipe,
        pipe,
        noise_shape=PAIRED_LATENT_SHAPE,
        src_latent=src,
        real_tgt_latent=real,
        spacing=[1.0, 1.0, 1.0],
        src_label=1,
        tgt_label=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert torch.allclose(result.before, result.after)


def test_paired_decodes_real_target_to_unit(tmp_path):
    """The real target latent is decoded through the same VAE + min-max to [0, 1]."""
    pipe = _controlnet_pipeline()
    src, real = _paired_inputs()
    result = BeforeAfterEval().run_paired(
        pipe,
        pipe,
        noise_shape=PAIRED_LATENT_SHAPE,
        src_latent=src,
        real_tgt_latent=real,
        spacing=[1.0, 1.0, 1.0],
        src_label=1,
        tgt_label=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert result.real is not None
    assert result.real.shape == PAIRED_IMAGE_SHAPE
    assert result.real.min().item() >= -1e-5
    assert result.real.max().item() <= 1.0 + 1e-5


def test_paired_writes_valid_slice_grid_png(tmp_path):
    """The paired pass writes a well-formed 2.5D grid PNG per sample."""
    pipe = _controlnet_pipeline()
    src, real = _paired_inputs()
    result = BeforeAfterEval().run_paired(
        pipe,
        pipe,
        noise_shape=PAIRED_LATENT_SHAPE,
        src_latent=src,
        real_tgt_latent=real,
        spacing=[1.0, 1.0, 1.0],
        src_label=1,
        tgt_label=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert len(result.grids) == PAIRED_LATENT_SHAPE[0]
    for grid in result.grids:
        assert _is_valid_png(grid)


def test_paired_grid_has_before_after_real_columns(tmp_path):
    """The paired grid is before|after|real — the real target is the reference column.

    A recording grid double (the injected render seam) captures exactly what the
    driver passes, so this pins the column order and that the third column is the
    decoded real target — not an implementation detail.
    """

    class _SpyGrid:
        def __init__(self):
            self.calls = []

        def render(self, volumes, labels, out_path):
            self.calls.append((list(volumes), list(labels)))
            return out_path

    spy = _SpyGrid()
    pipe = _controlnet_pipeline()
    src, real = _paired_inputs()
    result = BeforeAfterEval(grid=spy).run_paired(
        pipe,
        pipe,
        noise_shape=PAIRED_LATENT_SHAPE,
        src_latent=src,
        real_tgt_latent=real,
        spacing=[1.0, 1.0, 1.0],
        src_label=1,
        tgt_label=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    assert len(spy.calls) == PAIRED_LATENT_SHAPE[0]
    volumes, labels = spy.calls[0]
    assert labels == ["before", "after", "real"]
    assert len(volumes) == 3
    assert torch.allclose(volumes[2], result.real[0])


def test_paired_scores_psnr_ssim_against_real(tmp_path):
    """The paired pass scores 3D PSNR/SSIM of each side vs the real target.

    The metric is the source of truth: the JSON's after-PSNR/SSIM must equal an
    independent ``PairedFidelityMetrics`` call on the returned volumes, pinning that
    the driver wires the generated after-volume and the real target into the scorer
    (not some other pair).
    """
    pipe = _controlnet_pipeline()
    src, real = _paired_inputs()
    result = BeforeAfterEval().run_paired(
        pipe,
        pipe,
        noise_shape=PAIRED_LATENT_SHAPE,
        src_latent=src,
        real_tgt_latent=real,
        spacing=[1.0, 1.0, 1.0],
        src_label=1,
        tgt_label=2,
        num_inference_steps=3,
        seed=0,
        out_dir=str(tmp_path),
    )
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["policy"] == "controlnet"
    assert result.metrics == payload
    for side, vol in (("before", result.before), ("after", result.after)):
        expected = PairedFidelityMetrics()(vol, result.real)
        assert payload[side]["psnr"] == pytest.approx(expected.psnr)
        assert payload[side]["ssim"] == pytest.approx(expected.ssim)
        assert math.isfinite(payload[side]["psnr"])
        assert -1.0 <= payload[side]["ssim"] <= 1.0


# -- reproducibility (the seed makes the before/after pairing re-runnable) -----


def test_unconditional_rerun_same_seed_reproduces_before(tmp_path):
    """Re-running with the same seed reproduces the before output (the comparison is
    re-runnable — the tool regenerates the ControlNet section after S4, #226)."""
    pipe = _jit_pipeline()
    kwargs = dict(
        target_shape=JIT_LATENT_SHAPE,
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=3,
        seed=7,
    )
    r1 = BeforeAfterEval().run_unconditional(pipe, pipe, out_dir=str(tmp_path / "a"), **kwargs)
    r2 = BeforeAfterEval().run_unconditional(pipe, pipe, out_dir=str(tmp_path / "b"), **kwargs)
    assert torch.allclose(r1.before, r2.before)


def test_paired_rerun_same_seed_reproduces_before(tmp_path):
    pipe = _controlnet_pipeline()
    src, real = _paired_inputs()
    kwargs = dict(
        noise_shape=PAIRED_LATENT_SHAPE,
        src_latent=src,
        real_tgt_latent=real,
        spacing=[1.0, 1.0, 1.0],
        src_label=1,
        tgt_label=2,
        num_inference_steps=3,
        seed=7,
    )
    r1 = BeforeAfterEval().run_paired(pipe, pipe, out_dir=str(tmp_path / "a"), **kwargs)
    r2 = BeforeAfterEval().run_paired(pipe, pipe, out_dir=str(tmp_path / "b"), **kwargs)
    assert torch.allclose(r1.before, r2.before)
