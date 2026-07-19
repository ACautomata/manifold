"""Per-epoch pixel-space 3D PSNR/SSIM validation for Paired JiT (issue #68).

Mirrors :class:`~manifold.metrics.FIDCallback`'s lifecycle (VAE staged to GPU
around validation + restored to CPU after, all-rank decode under DDP with a
global (sum, count) reduce, float32 decode
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
3. computes **brain-masked** 3D PSNR and SSIM per sample (skull-stripped BraTS
   background excluded — see :meth:`_batch_metrics`) with
   ``data_range = target[max − min]`` clamped to the ``[0, 1]`` VAE convention,
   and averages over the val set; logs ``val/psnr`` and ``val/ssim``.

Best-checkpoint selection is configurable on either metric via a stock
Lightning ``ModelCheckpoint(monitor="val/psnr" | "val/ssim")`` — the same pattern
the noise→data stack uses for ``val/fid`` (see ``training/cli.py``).
This callback only *logs* the metrics, so wiring selection needs no trainer
change here (the paired trainer, Slice 4, will pass the monitor name through).

Distributed under DDP: **all ranks decode**. Each rank decodes its own
``DistributedSampler`` shard and the per-volume ``(psnr_sum, ssim_sum, count)`` are
``all_reduce``'d in :meth:`on_validation_epoch_end` for the true global mean — the
ADR-0016 "distributed PSNR" intent, restored. The prior rank-0-only gate (PR #115)
was a workaround for a DCU/DTK device-side stall under 8-way concurrent full-volume
MAISI VAE decode; it is removed now that the VAE ``num_splits``/``save_mem`` config
addresses the stall (ADR-0025). That the decode no longer deadlocks under DDP is
empirical and **probe-pending on sugon 8-DCU**; until the probe clears, the
``ModelCheckpoint(monitor="val/psnr")`` stays on under multi-GPU (selection runs on
the now-global metric).
"""

from __future__ import annotations

import math

import torch
from torchmetrics.functional import structural_similarity_index_measure

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore


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
        """Cadence gate only. All ranks decode their own ``DistributedSampler`` shard
        and the per-volume sums are all-reduced in :meth:`on_validation_epoch_end` for
        the true global ``val/psnr`` / ``val/ssim`` — the ADR-0016 distributed-PSNR
        intent, restored now that the VAE decode no longer deadlocks under DDP (VAE
        ``num_splits`` config; ADR-0025). The prior rank-0-only gate was the
        DCU-deadlock workaround (PR #115); it is removed.
        """
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

    def _masked_ssim(self, p: torch.Tensor, t: torch.Tensor, brain: torch.Tensor, data_range: float) -> float:
        """3D SSIM restricted to the brain (skull-stripped background excluded).

        torchmetrics' ``structural_similarity_index_measure`` has no mask arg, so
        the background is excluded by multiplying **both** pred and tgt by the
        brain mask before SSIM: background voxels become 0 in both, so their
        (background↔brain) local windows collapse to the identical-constant SSIM
        of 1 while the brain structure drives the score below 1 only when pred
        mismatches. ``tgt == tgt`` still gives exactly 1.0, preserving the
        torchmetrics identity contract the tests pin. This is materially more
        background-robust than a full-volume SSIM (where ~80–90% zero voxels
        inflate the score), though its absolute value is not the classical
        full-volume SSIM.
        """
        m = brain.to(p.dtype)
        ssim = float(structural_similarity_index_measure(p * m, t * m, data_range=data_range))
        # Masked-window means can push SSIM marginally outside [0, 1]; clamp to the
        # bounded range the SSIM contract (and downstream tests/monitors) assume.
        return min(max(ssim, 0.0), 1.0)

    # -- per-batch metric ----------------------------------------------------

    def _batch_metrics(
        self, pred_vol: torch.Tensor, tgt_vol: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[float, float, int]:
        """Per-sample **brain-masked** 3D PSNR + SSIM over a decoded batch.

        Inputs are the RAW float32 VAE decodes of pred and tgt (C2): both passed
        through the same frozen VAE, so they share one image space and the
        comparison is true pixel fidelity — per-volume gain/offset/contrast errors
        are visible. BraTS is skull-stripped (~80–90% background zeros), so both
        metrics are computed over the **brain mask** (``tgt > 0``) rather than the
        full volume — otherwise the trivially-matched background dominates MSE and
        inflates the score toward the copy-src ceiling (~23.8 dB full-volume).
        ``data_range`` is the raw target's ``[max − min]`` (computed over the full
        volume so the masked metric stays comparable), clamped to ``>= 1.0`` to
        match the VAE's ``[0, 1]`` intensity convention (BraTS MR percentile
        normalization is ``clip=False`` so the bright tail can exceed 1.0). PSNR is
        ``10·log10(data_range² / mse_brain)``; SSIM is the brain-masked variant in
        :meth:`_masked_ssim`.

        Samples with a degenerate target (zero data range, or an empty brain mask)
        are skipped — PSNR/SSIM are undefined there. Returns ``(psnr_sum,
        ssim_sum, n_valid)``; the caller keeps a running sum + count to average
        over the whole val set.
        """
        psnr_sum = 0.0
        ssim_sum = 0.0
        n = 0
        for i in range(pred_vol.shape[0]):
            if valid_mask is not None and not bool(valid_mask[i]):
                continue
            p = pred_vol[i : i + 1].float()
            t = tgt_vol[i : i + 1].float()
            data_range = max(float(t.max() - t.min()), 1.0)  # fix #3: clamp to [0,1] VAE convention
            if data_range <= 0.0:
                continue  # constant target — PSNR/SSIM undefined
            brain = t > 0  # skull-stripped BraTS: restrict the metric to the brain
            if not bool(brain.any()):
                continue  # empty brain — masked MSE/SSIM undefined
            mse = float((p - t).pow(2)[brain].mean())
            # Bit-exact pred == tgt over the brain → mse == 0 → PSNR is +inf. Cap
            # at 100 dB and SSIM = 1.0 so the checkpoint monitor sees a finite
            # value (guards math.log10(0) too). Affine errors (gain/offset) are NOT
            # collapsed any more — they raise mse and lower PSNR, as a fidelity
            # metric should.
            if mse == 0.0:
                psnr_sum += 100.0
                ssim_sum += 1.0
                n += 1
                continue
            psnr = 10.0 * math.log10((data_range * data_range) / mse)
            psnr_sum += min(psnr, 100.0)
            ssim_sum += self._masked_ssim(p, t, brain, data_range)
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
        # for. ``data_range`` is read from the raw target inside ``_batch_metrics``
        # (clamped to >= 1.0, the [0, 1] VAE convention).
        valid_mask = ~batch.get(
            "_is_padding", torch.zeros(pred_vol.shape[0], dtype=torch.bool)
        ).bool()
        psnr_sum, ssim_sum, n = self._batch_metrics(pred_vol, tgt_vol, valid_mask)
        self._psnr_sum += psnr_sum
        self._ssim_sum += ssim_sum
        self._count += n

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._active:
            return
        try:
            # All ranks decoded their own shard (gate removed) and accumulated local
            # per-sample sums. all_reduce (psnr_sum, ssim_sum, count) for the true
            # global mean (ADR-0016's distributed-PSNR intent, restored). Symmetric —
            # every rank enters here with ``_active`` True on the same epoch — so the
            # collective cannot deadlock; the only thing that can hang is the per-batch
            # decode conv itself, which is gated on the VAE ``num_splits`` config + the
            # sugon probe (ADR-0025).
            agg = torch.tensor(
                [self._psnr_sum, self._ssim_sum, float(self._count)],
                device=self._device(),
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(agg, op=torch.distributed.ReduceOp.SUM)
            psnr_sum, ssim_sum, count = (float(x) for x in agg.tolist())
            if count > 0:
                pl_module.log("val/psnr", psnr_sum / count)
                pl_module.log("val/ssim", ssim_sum / count)
        finally:
            self._restore_eval_to_cpu()
            self._active = False



__all__ = ["PairedPSNRSSIMCallback"]
