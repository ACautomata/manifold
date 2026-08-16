"""In-training paired-fidelity monitor for supervised ControlNet (ADR-0037, issue #238).

An **observe-only** generative validation callback. On each gated validation epoch it
translates a small **fixed paired subset** under **fixed initial noise** through the
module's own full Heun ControlNet rollout (:meth:`ControlNetLatentFlowModule.sample`),
VAE-decodes the generated and the real target, per-sample min-max normalizes both to
``[0, 1]``, scores 3D PSNR + 3D SSIM with the existing
:class:`~manifold.metrics.PairedFidelityMetrics`, and logs ``val/psnr`` / ``val/ssim``.

Because it reuses the **same metric, same normalization, and same full-rollout
generation** as the offline before/after comparison (ADR-0036), the in-training curve
is directly comparable to the offline number — but it is **observe-only**: it never
drives checkpoint selection (that stays ``val/x0_mae``), never enters the loss, and
never touches the optimizer / EMA. Generation runs under ``inference_mode`` (and the
rollout itself is inference-mode), so no gradient is ever formed.

Reused, not rebuilt (ADR-0037 "no parallel machinery"): the VAE-only VRAM staging is
:class:`~manifold.metrics.vae_stage.VaeStage`, the decode is
:class:`~manifold.metrics.LatentDecoder`, the normalization is
:func:`~manifold.pipelines.pipeline_utils.min_max_to_unit`, and the metric is
:class:`~manifold.metrics.PairedFidelityMetrics` — each already tested.

DDP (ADR-0025): every rank runs the monitor on the **same** fixed subset redundantly
(DDP-synchronized weights + identical seeded noise + identical fixed input ⇒ identical
result), so the cross-rank reduction is just Lightning's torchmetrics sync on the
logged :class:`~torchmetrics.MeanMetric`. There is no sharding and no error-rendezvous
machinery — collective-count invariance (ADR-0030) rests on the redundant work being
identical on every rank. Revisit only if the subset grows (ADR-0037).
"""

from __future__ import annotations

import torch
import torchmetrics
from torch.utils.data import default_collate

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover — lightning is a hard dep via spt
    import pytorch_lightning as pl  # type: ignore

from manifold.metrics.fid.decoder import LatentDecoder
from manifold.metrics.paired import PairedFidelityMetrics
from manifold.metrics.vae_stage import VaeStage
from manifold.pipelines.pipeline_utils import min_max_to_unit

#: Private attribute names under which the two MeanMetrics attach to the module
#: (Lightning scans ``named_modules()`` for ``Metric`` subclasses, so a logged metric
#: must live on the module — mirroring ``LatentX0MAE``).
_PSNR_ATTR = "_manifold_val_psnr_mean"
_SSIM_ATTR = "_manifold_val_ssim_mean"


class PairedFidelityCallback(pl.Callback):
    """Observe-only in-training paired-fidelity monitor (``val/psnr`` / ``val/ssim``).

    Args:
        module: the :class:`~manifold.modules.ControlNetLatentFlowModule`; its
            :meth:`~manifold.modules.ControlNetLatentFlowModule.sample` runs the full
            Heun rollout, and its frozen ``unet`` marks the device.
        vae: the held frozen VAE; its ``.decode()`` decodes the generated and the real
            target latents (it undoes ``scaling_factor`` internally, ADR-0003).
        paired_data: the fixed-subset source — the paired validation dataset (sized +
            indexable → ``{src_latent, tgt_latent, src_label, tgt_label, spacing}``),
            or a datamodule exposing ``val_latent_ds``.
            Resolved **lazily at the first gated epoch** (F5, ADR-0017): the cold
            datamodule path replaces ``val_latent_ds`` in ``setup()`` post-PG, so the
            dataset is read at first use, never captured at build time.
        subset_size: number of fixed paired subjects scored per monitored epoch.
        every_n_epochs: run cadence (1 = every validation epoch).
        num_inference_steps: Heun integration steps for the rollout (the
            ``controlnet.num_inference_steps`` knob; default 15 ⇒ 29 UNet evals).
        seed: the generation-noise + subset-selection seed — re-seeded fresh each
            monitored epoch so only the model changes between epochs.
        fidelity: the paired-fidelity scorer (default
            :class:`~manifold.metrics.PairedFidelityMetrics`); injectable so tests can
            substitute a tiny/fake scorer.
    """

    #: Declares the metrics this callback logs so the registry's ``validate_monitor``
    #: accepts ``val/psnr`` / ``val/ssim`` as *validatable* monitors (ADR-0029) without
    #: switching checkpoint selection away from ``val/x0_mae`` (observe-only).
    logged_metrics: frozenset[str] = frozenset({"val/psnr", "val/ssim"})

    def __init__(
        self,
        *,
        module,
        vae,
        paired_data,
        subset_size: int = 8,
        every_n_epochs: int = 1,
        num_inference_steps: int = 15,
        seed: int = 0,
        fidelity: PairedFidelityMetrics | None = None,
    ):
        super().__init__()
        self._module = module
        self._vae = vae
        self._paired_source = paired_data
        self._fidelity = fidelity if fidelity is not None else PairedFidelityMetrics()
        self.subset_size = int(subset_size)
        self.every_n_epochs = int(every_n_epochs)
        self.num_inference_steps = int(num_inference_steps)
        self.seed = int(seed)
        #: The materialized fixed paired subset (collated batched tensors); cached so
        #: epoch-over-epoch change reflects the model, not a moving data sample.
        self._subset: dict | None = None

    # -- gating + device -------------------------------------------------------

    def _gated(self, trainer) -> bool:
        """Cadence gate only (mirrors ``FIDCallback._gated``). All ranks gate
        identically (``current_epoch`` and ``every_n_epochs`` are rank-symmetric), so
        the metric-sync collective stays symmetric."""
        epoch = trainer.current_epoch
        return bool(self.every_n_epochs <= 1 or epoch % self.every_n_epochs == 0)

    def _device(self) -> torch.device:
        """The module's device (the frozen base UNet's), where the VAE is staged."""
        return next(self._module.unet.parameters()).device

    # -- fixed paired subset (lazy, F5) ----------------------------------------

    def _paired_dataset(self):
        """Resolve the paired validation dataset lazily (F5): the source may be the
        dataset itself (warm / test path) or a datamodule exposing ``val_latent_ds``
        (the cold path, whose ``setup()`` replaces it post-PG — so resolve at first
        use, not at build)."""
        src = self._paired_source
        return getattr(src, "val_latent_ds", src)

    def _fixed_subset(self) -> dict:
        """Select + collate + cache the fixed paired subset.

        A seeded ``randperm`` prefix of ``subset_size`` (the fixed-sample-validation
        ethos): the same subjects are scored every monitored epoch. Collation stacks
        the per-sample dicts to ``src_latent``/``tgt_latent`` ``[N, C, D, H, W]``,
        ``src_label``/``tgt_label`` ``[N]`` long, and ``spacing`` ``[N, 3]``.
        """
        if self._subset is None:
            ds = self._paired_dataset()
            n = min(self.subset_size, len(ds))
            idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(self.seed))[:n].tolist()
            self._subset = default_collate([ds[i] for i in idx])
        return self._subset

    # -- metrics (module-attached, mirroring LatentX0MAE) -----------------------

    @staticmethod
    def _ensure_metrics(module) -> tuple[torchmetrics.MeanMetric, torchmetrics.MeanMetric]:
        """Attach (idempotently) the two MeanMetrics to *module*; return them.

        The metric must live on the module for Lightning to register + restore it.
        Reuses the same instance across epochs (state survives); moved to the module's
        device so the torchmetrics sync reduces on the right device.
        """
        if getattr(module, _PSNR_ATTR, None) is None:
            setattr(module, _PSNR_ATTR, torchmetrics.MeanMetric())
            setattr(module, _SSIM_ATTR, torchmetrics.MeanMetric())
        device = module.device if hasattr(module, "device") else torch.device("cpu")
        psnr, ssim = getattr(module, _PSNR_ATTR), getattr(module, _SSIM_ATTR)
        psnr.to(device)
        ssim.to(device)
        return psnr, ssim

    def on_fit_start(self, trainer, module) -> None:
        self._ensure_metrics(module)

    # -- Lightning hook ---------------------------------------------------------

    def on_validation_epoch_end(self, trainer, module) -> None:
        if not self._gated(trainer):
            return
        psnr_metric, ssim_metric = self._ensure_metrics(module)
        subset = self._fixed_subset()
        device = self._device()

        # Generate the translated target latent under fixed noise (a fresh Generator
        # each gated epoch, so sampling stochasticity is isolated from quality drift)
        # through the module's own full Heun rollout — observe-only (inference_mode).
        with torch.inference_mode():
            noise = torch.randn(
                tuple(subset["tgt_latent"].shape),
                generator=torch.Generator(device=device).manual_seed(self.seed),
                device=device,
                dtype=next(self._module.unet.parameters()).dtype,
            )
            generated = self._module.sample(
                noise,
                subset["src_latent"],
                subset["spacing"],
                subset["src_label"],
                subset["tgt_label"],
                num_inference_steps=self.num_inference_steps,
            )

        # Stage the VAE onto the device for the decode, then restore it to CPU (the
        # VAE-only staging path — no feature_net). Decode generated + real target
        # identically, then per-sample min-max to [0, 1] (data_range = 1.0).
        with VaeStage(self._vae, device_fn=self._device):
            self._vae.eval()
            decoder = LatentDecoder(self._vae)
            with torch.inference_mode():
                generated_vol = min_max_to_unit(decoder(generated))
                real_vol = min_max_to_unit(decoder(subset["tgt_latent"]))

        # Score with the existing paired-fidelity metric and log via torchmetrics so
        # Lightning handles the (trivial, redundant-subset) cross-rank reduction.
        # PSNR is +inf on identical volumes; that ceiling is surfaced honestly.
        scores = self._fidelity(generated_vol, real_vol)
        psnr_metric.reset()
        ssim_metric.reset()
        psnr_metric.update(torch.as_tensor(scores.psnr, dtype=torch.float32))
        ssim_metric.update(torch.as_tensor(scores.ssim, dtype=torch.float32))
        module.log("val/psnr", psnr_metric)
        module.log("val/ssim", ssim_metric)


__all__ = ["PairedFidelityCallback"]
