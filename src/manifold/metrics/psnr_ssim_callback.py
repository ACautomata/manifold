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
the noise→data stack uses for ``val/fid`` (see ``training/cli.py``).
This callback only *logs* the metrics, so wiring selection needs no trainer
change here (the paired trainer, Slice 4, will pass the monitor name through).

Distributed under DDP: **rank-0-only** (mirrors :class:`~manifold.metrics.FIDCallback`; the ADR-0016 "distributed PSNR" amendment is reverted). The per-batch full-volume 3D MAISI VAE decode (``monai sliding_window_inference``) is heavy, and running it concurrently on all DDP ranks deadlocks the DCU/DTK vendor runtime inside the decode conv (a device-side stall in ``_conv_forward``, not an NCCL collective - observed on sugon 8-DCU; single-DCU and 8-NVIDIA run the identical callback clean). The amendment's "no per-batch collective -> no deadlock" premise is falsified: the freeze is in the *local* conv, not the (now-removed) epoch-end ``all_gather``. So under DDP only rank 0 decodes its ``DistributedSampler`` shard and logs ``val/psnr`` / ``val/ssim`` as a rank-0-shard estimate (like ``val/fid``); the other ranks skip and log nothing (the ``ModelCheckpoint(monitor="val/psnr")`` "monitor not found" note on non-root ranks is benign - selection runs on rank 0).
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
    """

    def __init__(
        self,
        *,
        pipeline,
        num_inference_steps: int,
        every_n_epochs: int = 1,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.num_inference_steps = int(num_inference_steps)
        self.every_n_epochs = int(every_n_epochs)
        self._active = False
        self._eval_staged = False
        self._norm16_disabled = False
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._count = 0

    # -- gate + staging (mirror FIDCallback) ---------------------------------

    def _gated(self, trainer) -> bool:
        """Rank-0 + cadence gate; warn loudly under DDP and skip otherwise.

        Mirrors :class:`~manifold.metrics.FIDCallback`: the per-batch full-volume 3D
        MAISI VAE decode (``monai sliding_window_inference``) is heavy, and running it
        concurrently on all DDP ranks deadlocks the DCU/DTK vendor runtime inside the
        decode conv (a device-side stall, not a collective). The ADR-0016 "distributed
        PSNR" amendment's "no per-batch collective -> no deadlock" premise is
        falsified by repeated sugon 8-DCU hangs (single-DCU and 8-NVIDIA run the same
        callback clean). So under DDP only rank 0 decodes its ``DistributedSampler``
        shard and logs ``val/psnr`` / ``val/ssim`` as a rank-0-shard estimate (like
        ``val/fid``); the other ranks skip and must NOT block on a collective here.
        """
        world = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world = torch.distributed.get_world_size()
        if world > 1:
            if not trainer.is_global_zero:
                return False
            _log.warning(
                "PairedPSNRSSIMCallback: running only on rank 0 (world_size=%d). The "
                "other ranks skip the per-batch VAE decode - a concurrent 8-rank "
                "full-volume decode deadlocks the DCU runtime. val/psnr / val/ssim are "
                "a rank-0-shard estimate (not a global mean); non-root ranks do not log "
                "them (the ModelCheckpoint 'monitor not found' note on those ranks is "
                "benign).",
                world,
            )
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
            # ``eval()`` mirrors the inference pipeline + FIDCallback: today the
            # MAISI VAE is all GroupNorm with no Dropout (so train/eval decode are
            # bit-identical), but a future BatchNorm/Dropout layer would silently
            # drift the decoded volumes if decode ran in train mode.
            self.pipeline.vae.eval()
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

        Inputs are the RAW float32 VAE decodes of pred and tgt (C2): both passed
        through the same frozen VAE, so they share one image space and the
        comparison is true pixel fidelity — per-volume gain/offset/contrast errors
        are visible. ``data_range`` is the raw target's ``[max − min]`` (standard
        for float medical data with no fixed intensity ceiling). PSNR is
        ``10·log10(data_range² / mse)``. SSIM uses torchmetrics'
        :func:`structural_similarity_index_measure` (true **3D SSIM** on the
        ``[1,C,D,H,W]`` volume — ``is_3d`` → ``_gaussian_kernel_3d`` +
        ``F.conv3d`` with 3D reflection padding).

        Samples with a degenerate target (zero data range) are skipped —
        ``data_range <= 0`` means a constant target where PSNR/SSIM are
        undefined. Returns ``(psnr_sum, ssim_sum, n_valid)``; the caller keeps a
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
            # Bit-exact pred == tgt → mse == 0 → PSNR is +inf. Cap at 100 dB and
            # SSIM = 1.0 so the checkpoint monitor sees a finite value (guards
            # math.log10(0) too). Affine errors (gain/offset) are NOT collapsed
            # any more — they raise mse and lower PSNR, as a fidelity metric should.
            if mse == 0.0:
                psnr_sum += 100.0
                ssim_sum += 1.0
                n += 1
                continue
            psnr = 10.0 * math.log10((data_range * data_range) / mse)
            psnr_sum += min(psnr, 100.0)
            ssim_sum += float(structural_similarity_index_measure(p, t, data_range=data_range))
            n += 1
        return psnr_sum, ssim_sum, n

    # -- Lightning hooks -----------------------------------------------------

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._active = self._gated(trainer)
        if not self._active:
            return
        self._stage_eval_on_device()
        # Defragment the caching allocator before the heavy per-batch full-volume 3D
        # decode loop. Under DDP rank 0 also holds the UNet + optimizer + gradient
        # buckets + HCCL buffers, so a fragmented cache is the leading residual way the
        # multi-GB image-resolution decode can still device-stall even after the
        # rank-0-only revert (single-DCU, with none of that overhead, decodes clean).
        # Runs on rank 0 only (the gate above returned on non-root ranks); no-op on CPU.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._psnr_sum = 0.0
        self._ssim_sum = 0.0
        self._count = 0

    @torch.no_grad()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx: int = 0
    ) -> None:
        if not self._active:
            return
        # Per-sample contrast labels (C1): a Paired JiT val batch mixes all 12
        # within-subject contrast directions (``build_brats_pair_manifest`` emits
        # every ordered src→tgt pair), so each sample must be conditioned on its
        # OWN (src, tgt) pair. The earlier scalar ``[0]`` collapsed the whole batch
        # to sample 0's direction, conditioning 7/8 of an 8-sample val batch on the
        # wrong translation. ``batch["src_label"]`` / ``["tgt_label"]`` are ``[B]``
        # long tensors (the same tensors training consumes); the shared rollout
        # forwards them per-sample.
        # ``pipeline.unet`` is ``pl_module.unet`` (the pipeline is built over the
        # module's UNet by reference), so the rollout samples the live (raw)
        # optimizer weights directly - no EMA swap (EMA training was removed).
        pred_latent = self.pipeline.sample_latent(
            batch["src_latent"],
            batch["spacing"],
            batch["src_label"],
            batch["tgt_label"],
            self.num_inference_steps,
        )
        pred_vol = self._eval_decode(pred_latent)
        tgt_vol = self._eval_decode(batch["tgt_latent"])
        # PSNR/SSIM on the RAW float32 decodes (C2): pred and tgt both pass through
        # the same frozen VAE (which undoes ``scaling_factor`` internally,
        # ADR-0003), so they already share one image space — comparing them directly
        # is true pixel fidelity. The earlier per-volume ``_minmax_to_unit`` on EACH
        # side applied different affine maps (pred's own min/max vs tgt's), making
        # PSNR/SSIM invariant to per-volume gain+offset — blind to exactly the
        # brightness/contrast errors a contrast-translation model must be penalized
        # for. ``data_range`` is read from the raw target inside ``_batch_metrics``.
        psnr_sum, ssim_sum, n = self._batch_metrics(pred_vol, tgt_vol)
        self._psnr_sum += psnr_sum
        self._ssim_sum += ssim_sum
        self._count += n

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._active:
            return
        try:
            # Rank-0-only (ADR-0016 amendment reverted): only rank 0 ran the decode +
            # accumulate, so the per-volume sums are already the rank-0-shard totals.
            # The cross-rank ``all_gather`` is removed - it was the amendment's
            # mechanism, and the hang is in the *local* decode conv, not the gather
            # (py-spy: all ranks frozen in ``_conv_forward``). Rank 0 logs
            # ``val/psnr`` / ``val/ssim`` as a rank-0-shard estimate; the other ranks
            # skipped (``_active`` False) and log nothing. No collective is called, so
            # non-root ranks cannot block here (they sync implicitly at the next DDP
            # training-step gradient all-reduce - the FIDCallback contract).
            count = float(self._count)
            if count > 0:
                pl_module.log("val/psnr", self._psnr_sum / count)
                pl_module.log("val/ssim", self._ssim_sum / count)
        finally:
            self._restore_eval_to_cpu()
            self._active = False



__all__ = ["PairedPSNRSSIMCallback"]
