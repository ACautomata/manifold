"""Per-epoch pixel-space 3D PSNR/SSIM validation for Paired JiT (issue #68).

Mirrors :class:`~manifold.metrics.FIDCallback`'s lifecycle (VAE staged to GPU
around validation + restored to CPU after, rank-0 gate under DDP, float32 decode
with ``norm_float16`` disabled) but evaluates Paired JiT's deterministic src→tgt
rollout instead of unconditional generation, and aggregates per-sample pixel
fidelity rather than a Fréchet distance.

Because the Paired JiT transport is deterministic given ``x_src`` (ADR-0013 — the
``t = 0`` endpoint is a data latent, not Gaussian noise; the rollout has no
stochastic input), the prediction is reproducible across epochs with **no
per-epoch noise re-seeding** — only the fixed validation subset (the val split
itself). This is the key difference from the noise→data FID callback, which
re-seeds its generation noise every epoch to keep the synthetic arm a
deterministic function of the model: here the determinism is structural, so the
metric tracks only model changes between epochs.

For each validation batch the callback:

1. runs the start-from-src Heun rollout via
   :meth:`~manifold.pipelines.PairedLatentFlowPipeline.sample_latent` — the
   shared rollout primitive (ADR-0005) wired through the live module UNet, so
   training updates are visible at validation;
2. decodes both the predicted and the true ``x_tgt`` latent through the held
   frozen VAE (staged to the UNet's device) — pixel space, not latent space
   (the VAE owns ``scaling_factor`` and undoes the scaling internally,
   ADR-0003);
3. computes full-volume 3D PSNR and SSIM per sample with
   ``data_range = target[max − min]`` and averages over the val set; logs
   ``val/psnr`` and ``val/ssim``.

Best-checkpoint selection is configurable on either metric via a stock
Lightning ``ModelCheckpoint(monitor="val/psnr" | "val/ssim")`` — the same pattern
the noise→data stack uses for ``val/fid_raw`` (see ``training/cli.py``).
This callback only *logs* the metrics, so wiring selection needs no trainer
change here (the paired trainer, Slice 4, will pass the monitor name through).

Single-GPU / rank-0 only: like FIDCallback, a multi-minute decode loop would
deadlock the other ranks at an NCCL collective, so under DDP the callback warns
loudly and skips (the paired trainer falls back to ``save_last`` +
``every_n_epochs``, mirroring the noise→data DDP path).
"""

from __future__ import annotations

import logging
import math

import torch
from torchmetrics.functional import structural_similarity_index_measure

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore

_log = logging.getLogger(__name__)


class PairedPSNRSSIMCallback(pl.Callback):
    """Per-epoch pixel-space 3D PSNR/SSIM over the fixed Paired JiT val subset.

    Args:
        pipeline: the :class:`~manifold.pipelines.PairedLatentFlowPipeline`
            carrying the LIVE module UNet (by reference), the held frozen VAE,
            and the scheduler. Constructed by the caller as
            ``PairedLatentFlowPipeline(module.unet, vae, module.scheduler)`` so
            optimizer updates to the UNet are visible at validation. Only
            :meth:`~manifold.pipelines.PairedLatentFlowPipeline.sample_latent`
            (the shared rollout primitive — ADR-0005) and ``pipeline.vae``
            (staged around validation) are touched.
        num_inference_steps: Heun integration steps over ``t: 0 → 1``.
        every_n_epochs: run cadence (1 = every validation epoch).
        ema_callback: optional :class:`~manifold.training.DoubleEMACallback`. When
            provided, the slow-EMA shadow is swapped into ``module.unet`` around
            each rollout so the reported PSNR/SSIM reflects the published EMA
            model (mirrors :class:`FIDCallback`'s slow arm). ``None`` reports on
            the raw optimizer weights (e.g. a no-EMA regime or a raw-arm monitor).
    """

    def __init__(
        self,
        *,
        pipeline,
        num_inference_steps: int,
        every_n_epochs: int = 1,
        ema_callback=None,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.num_inference_steps = int(num_inference_steps)
        self.every_n_epochs = int(every_n_epochs)
        self.ema_callback = ema_callback
        self._active = False
        self._eval_staged = False
        self._norm16_disabled = False
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._count = 0

    # -- gate + staging (mirror FIDCallback) ---------------------------------

    def _gated(self, trainer) -> bool:
        """Rank-0 + cadence gate; warn loudly under DDP and skip otherwise.

        Identical policy to FIDCallback: the decode + rollout loop is
        single-GPU only, so under DDP the non-rank-0 processes must skip and
        must NOT block on an NCCL collective here.
        """
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        if world > 1:
            _log.warning(
                "PairedPSNRSSIMCallback: running only on rank 0 (world_size=%d). The "
                "other ranks skip the PSNR/SSIM decode loop and must NOT block on an "
                "NCCL collective here — single-GPU is the supported config.", world,
            )
            if not trainer.is_global_zero:
                return False
        epoch = trainer.current_epoch
        if self.every_n_epochs <= 1 or (epoch % self.every_n_epochs == 0):
            return True
        return False

    def _device(self):
        return next(self.pipeline.unet.parameters()).device

    def _stage_eval_on_device(self) -> None:
        """Stage the VAE onto the UNet's (GPU) device for the validation phase.

        Mirrors FIDCallback: ``warm_latent_pipeline`` leaves the VAE on CPU to
        free VRAM for UNet *training*; during validation the UNet is idle, so
        decoding on GPU is feasible and far faster than CPU. The VAE is returned
        to CPU once the phase ends. The pre-stage CPU state_dict is cloned so the
        restore is bit-identical (defensive — ``.to`` should be lossless, but the
        clone guarantees it).
        """
        if not self._eval_staged:
            self._vae_cpu_state = {
                k: v.detach().clone() for k, v in self.pipeline.vae.state_dict().items()
            }
            self.pipeline.vae.to(self._device())
            self._eval_staged = True

    def _restore_eval_to_cpu(self) -> None:
        """Return the VAE to CPU after the validation phase (free VRAM for training)."""
        if self._eval_staged:
            self.pipeline.vae.to("cpu")
            if hasattr(self, "_vae_cpu_state"):
                self.pipeline.vae.load_state_dict(self._vae_cpu_state)
            self._eval_staged = False

    def _eval_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode in float32 on the staged device, ``norm_float16`` disabled.

        Mirrors FIDCallback: ``norm_float16`` makes MaisiGroupNorm3D cast its
        output to float16 unconditionally, so a downstream float32 conv raises a
        bias-type mismatch unless an outer autocast reconciles them — which a
        validation hook cannot rely on. PSNR/SSIM are *evaluation* metrics, so
        float32 decode is both robust and more correct than half precision. The
        latent is moved to the VAE's device; the VAE undoes ``scaling_factor``
        internally (ADR-0003), so the caller passes already-scaled latents.
        """
        if not self._norm16_disabled:
            for m in self.pipeline.vae.modules():
                if hasattr(m, "norm_float16"):
                    m.norm_float16 = False
            self._norm16_disabled = True
        vae_device = next(self.pipeline.vae.parameters()).device
        return self.pipeline.vae.decode(latents.float().to(vae_device))

    # -- per-batch metric ----------------------------------------------------

    def _batch_metrics(
        self, pred_vol: torch.Tensor, tgt_vol: torch.Tensor
    ) -> tuple[float, float, int]:
        """Per-sample full-volume 3D PSNR + SSIM over a decoded batch.

        ``data_range = target[max − min]`` is per-sample (the standard PSNR/SSIM
        convention for medical images whose intensity range is not normalized to
        ``[0, 1]``). PSNR is ``10·log10(data_range² / mse)`` — self-contained
        (clearer than torchmetrics' PSNR for the per-sample ``data_range`` and
        identical in value). SSIM uses torchmetrics'
        :func:`structural_similarity_index_measure`, which runs a **true 3D SSIM**
        on the ``[1,C,D,H,W]`` volume (``is_3d = preds.ndim == 5`` →
        ``_gaussian_kernel_3d`` + ``F.conv3d`` with 3D reflection padding), so it
        captures volumetric (not slice-wise) structural similarity.

        Samples with a degenerate target (zero data range) or an exact
        reconstruction (``mse == 0`` → PSNR ``+inf``) are skipped — they never
        arise outside synthetic fixtures and would otherwise poison the running
        mean. Returns ``(psnr_sum, ssim_sum, n_valid)``; the caller keeps a
        running sum + count to average over the whole val set.
        """
        psnr_sum = 0.0
        ssim_sum = 0.0
        n = 0
        for i in range(pred_vol.shape[0]):
            p = pred_vol[i : i + 1].float()
            t = tgt_vol[i : i + 1].float()
            data_range = float(t.max() - t.min())
            if data_range <= 0.0:
                continue  # constant target — PSNR/SSIM undefined
            mse = float((p - t).pow(2).mean())
            if mse == 0.0:
                continue  # exact reconstruction → PSNR +inf; never arises in practice
            psnr_sum += 10.0 * math.log10((data_range * data_range) / mse)
            ssim_sum += float(structural_similarity_index_measure(p, t, data_range=data_range))
            n += 1
        return psnr_sum, ssim_sum, n

    # -- Lightning hooks -----------------------------------------------------

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._active = self._gated(trainer)
        if not self._active:
            return
        self._stage_eval_on_device()
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._count = 0

    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        if not self._active:
            return
        # The shared rollout primitive takes scalar contrast labels and broadcasts
        # them across the batch via ``torch.full``. Paired JiT v1 validation is
        # single-direction (one src→tgt per run, ADR-0014), so every sample in a
        # val batch shares labels and the first sample's labels are authoritative.
        src_label = int(batch["src_label"].reshape(-1)[0].item())
        tgt_label = int(batch["tgt_label"].reshape(-1)[0].item())
        # Swap the slow-EMA shadow in around the rollout so the reported metric
        # reflects the published EMA model (mirrors FIDCallback's slow arm).
        # ``pipeline.unet`` is ``pl_module.unet`` (the pipeline is built over the
        # module's UNet by reference), so swapping the module swaps the rollout.
        if self.ema_callback is not None:
            self.ema_callback.swap_in(pl_module)
        try:
            pred_latent = self.pipeline.sample_latent(
                batch["src_latent"],
                batch["spacing"],
                src_label,
                tgt_label,
                self.num_inference_steps,
            )
        finally:
            if self.ema_callback is not None:
                self.ema_callback.restore(pl_module)
        pred_vol = self._eval_decode(pred_latent)
        tgt_vol = self._eval_decode(batch["tgt_latent"])
        psnr_sum, ssim_sum, n = self._batch_metrics(pred_vol, tgt_vol)
        self._psnr_sum += psnr_sum
        self._ssim_sum += ssim_sum
        self._count += n

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._active:
            return
        try:
            if self._count > 0:
                pl_module.log("val/psnr", self._psnr_sum / self._count)
                pl_module.log("val/ssim", self._ssim_sum / self._count)
        finally:
            self._restore_eval_to_cpu()
            self._active = False


__all__ = ["PairedPSNRSSIMCallback"]
