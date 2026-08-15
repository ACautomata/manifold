"""The injectable before/after-GRPO eval-driver core (issue #228, ADR-0036).

Policy-agnostic across the JiT (unconditional) and ControlNet (paired) policies.
Given a before pipeline and an after pipeline plus a fixed seed + conditioning, the
driver owns the seeding so both sides see **identical initial noise** — the fairness
invariant that makes any before/after difference attributable to GRPO and not to
sampling stochasticity:

- the JiT pipeline builds its start noise from a ``generator`` — the driver hands
  each side a fresh generator seeded identically; and
- the ControlNet pipeline takes the start noise directly — the driver draws one
  noise tensor and passes the *same* tensor to both sides.

Both sides are then decoded via the frozen VAE (the FID-eval float32
:class:`~manifold.metrics.LatentDecoder` mechanism) and per-sample min-max
normalized to ``[0, 1]`` (:func:`~manifold.pipelines.pipeline_utils.min_max_to_unit`,
the published-output convention), so the before/after/real volumes are directly
comparable and the paired-fidelity metric can assume ``data_range = 1.0``.

The driver is dependency-injected: the two pipelines are passed per call and the
scoring / grid collaborators are injectable at construction, so it runs on tiny
components on CPU (no GPU, no real checkpoints) — mirroring the FID
factory-injection seam.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple, Sequence

import torch
from torch import Tensor

from ..metrics import LatentDecoder, PairedFidelityMetrics
from ..pipelines.pipeline_utils import min_max_to_unit
from .slice_grid import SliceGrid

#: The metrics-JSON filename written under ``out_dir``.
_METRICS_FILE = "metrics.json"


class BeforeAfterResult(NamedTuple):
    """The outcome of one before/after pass.

    ``before`` / ``after`` are the decoded ``[B, C, D, H, W]`` volumes in ``[0, 1]``;
    ``real`` is the decoded real target (paired policy only, else ``None``);
    ``grids`` are the written slice-grid PNG paths (one per sample); ``metrics`` is
    the metrics-JSON payload (provenance for the unconditional policy, plus
    ``psnr``/``ssim`` for the paired one).
    """

    before: Tensor
    after: Tensor
    real: Tensor | None
    grids: list[str]
    metrics: dict


class BeforeAfterEval:
    """Before/after-GRPO eval driver: same-noise generate → decode → grid + metrics.

    Args:
        fidelity: the paired-fidelity metric (default
            :class:`~manifold.metrics.PairedFidelityMetrics`). Used only by the
            paired policy; injectable so tests can substitute a tiny/fake scorer.
        grid: the slice-grid renderer (default :class:`SliceGrid`). Injectable so
            tests can observe or substitute the render step.
    """

    def __init__(
        self,
        *,
        fidelity: PairedFidelityMetrics | None = None,
        grid: SliceGrid | None = None,
    ) -> None:
        self._fidelity = fidelity if fidelity is not None else PairedFidelityMetrics()
        self._grid = grid if grid is not None else SliceGrid()

    # -- shared decode --------------------------------------------------------

    @staticmethod
    def _decode_normalize(latent: Tensor, vae) -> Tensor:
        """VAE-decode a latent and min-max normalize to ``[0, 1]``.

        A fresh :class:`~manifold.metrics.LatentDecoder` per call (its
        ``norm_float16`` disable is idempotent) gives a uniform FID-eval float32
        decode for the before/after/real volumes alike. ``vae.eval()`` mirrors the
        pipelines' own decode path (eval-mode decode, no dropout/BN drift).
        """
        vae.eval()
        with torch.inference_mode():
            vol = LatentDecoder(vae)(latent)
        return min_max_to_unit(vol)

    @staticmethod
    def _device_dtype(pipeline) -> tuple[torch.device, torch.dtype]:
        """The pipeline UNet's (device, dtype) — where generation noise must live."""
        param = next(pipeline.unet.parameters())
        return param.device, param.dtype

    # -- unconditional (JiT) pass --------------------------------------------

    def run_unconditional(
        self,
        before,
        after,
        *,
        target_shape: Sequence[int],
        spacing: Tensor | Sequence[float],
        modality: int,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        cfg_interval: tuple[float, float] | None = None,
        seed: int,
        out_dir: str,
    ) -> BeforeAfterResult:
        """Generate before/after under identical noise and decode both to ``[0, 1]``.

        The driver owns the seeding: each side gets a fresh
        ``torch.Generator(...).manual_seed(seed)`` on the UNet's device, so the two
        start noises are identical. Same noise + same conditioning + (possibly
        different) weights ⇒ any output difference is attributable to the weights.
        """
        device, _ = self._device_dtype(before)
        before_latent = before.sample_latent(
            target_shape,
            spacing,
            modality,
            num_inference_steps,
            guidance_scale=guidance_scale,
            cfg_interval=cfg_interval,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        after_latent = after.sample_latent(
            target_shape,
            spacing,
            modality,
            num_inference_steps,
            guidance_scale=guidance_scale,
            cfg_interval=cfg_interval,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        before_vol = self._decode_normalize(before_latent, before.vae)
        after_vol = self._decode_normalize(after_latent, after.vae)
        grids = self._render_grids([before_vol, after_vol], ["before", "after"], out_dir)
        metrics = self._write_metrics(
            {
                "policy": "jit",
                "seed": int(seed),
                "num_inference_steps": int(num_inference_steps),
                "num_samples": int(before_vol.shape[0]),
            },
            grids,
            out_dir,
        )
        return BeforeAfterResult(
            before=before_vol, after=after_vol, real=None, grids=grids, metrics=metrics
        )

    # -- paired (ControlNet) pass ----------------------------------------------

    def run_paired(
        self,
        before,
        after,
        *,
        noise_shape: Sequence[int],
        src_latent: Tensor,
        real_tgt_latent: Tensor,
        spacing: Tensor | Sequence[float],
        src_label: int | Tensor,
        tgt_label: int | Tensor,
        num_inference_steps: int,
        seed: int,
        out_dir: str,
    ) -> BeforeAfterResult:
        """Generate before/after under one shared noise tensor; decode + score vs real.

        The driver draws the start noise once (``torch.randn(..., generator=
        torch.Generator(device).manual_seed(seed))``) and passes the *same* tensor to
        both pipelines' ``sample_latent``, so the pairing holds noise constant. The
        real target latent (from the paired cache) is decoded through the same frozen
        VAE + min-max, so all three volumes share the ``[0, 1]`` convention and the
        fidelity metric can assume ``data_range = 1.0``.
        """
        device, dtype = self._device_dtype(before)
        noise = torch.randn(
            tuple(noise_shape),
            generator=torch.Generator(device=device).manual_seed(seed),
            device=device,
            dtype=dtype,
        )
        before_latent = before.sample_latent(
            noise, src_latent, spacing, src_label, tgt_label, num_inference_steps
        )
        after_latent = after.sample_latent(
            noise, src_latent, spacing, src_label, tgt_label, num_inference_steps
        )
        before_vol = self._decode_normalize(before_latent, before.vae)
        after_vol = self._decode_normalize(after_latent, after.vae)
        real_vol = self._decode_normalize(real_tgt_latent, before.vae)
        grids = self._render_grids(
            [before_vol, after_vol, real_vol], ["before", "after", "real"], out_dir
        )
        before_scores = self._fidelity(before_vol, real_vol)
        after_scores = self._fidelity(after_vol, real_vol)
        metrics = self._write_metrics(
            {
                "policy": "controlnet",
                "seed": int(seed),
                "num_inference_steps": int(num_inference_steps),
                "num_samples": int(before_vol.shape[0]),
                "before": {"psnr": before_scores.psnr, "ssim": before_scores.ssim},
                "after": {"psnr": after_scores.psnr, "ssim": after_scores.ssim},
            },
            grids,
            out_dir,
        )
        return BeforeAfterResult(
            before=before_vol, after=after_vol, real=real_vol, grids=grids, metrics=metrics
        )

    # -- shared render ---------------------------------------------------------

    @staticmethod
    def _write_metrics(metrics: dict, grids: list[str], out_dir: str) -> dict:
        """Write the metrics JSON (with grid basenames) under ``out_dir``; return it.

        Grid paths are recorded as basenames so the Artifact page builder (#230) can
        resolve them relative to the JSON's own directory regardless of where the
        eval ran.
        """
        payload = {**metrics, "grids": [os.path.basename(g) for g in grids]}
        with open(os.path.join(out_dir, _METRICS_FILE), "w") as f:
            json.dump(payload, f, indent=2)
        return payload

    def _render_grids(
        self, columns: list[Tensor], labels: list[str], out_dir: str
    ) -> list[str]:
        """Write one before|after(|real) slice-grid PNG per sample; return the paths.

        ``columns`` are the per-policy decoded volumes ``[B, C, D, H, W]`` sharing a
        batch; sample ``i`` of each column forms one grid (so a before/after pair is
        always the same sample index).
        """
        os.makedirs(out_dir, exist_ok=True)
        batch = columns[0].shape[0]
        grids: list[str] = []
        for i in range(batch):
            path = os.path.join(out_dir, f"slice_grid_{i}.png")
            self._grid.render([col[i] for col in columns], labels, path)
            grids.append(path)
        return grids
