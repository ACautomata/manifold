"""Paired JiT training component: the src→tgt x0 objective as an ``spt.Module``.

A :class:`stable_pretraining.Module` (:class:`spt.Module`) implementing the Paired
JiT training objective (ADR-0013/0014). ``forward(batch, stage)`` samples flow-
times ``t`` from the logit-normal distribution, obtains the interpolated latent
from the **scheduler's** :meth:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler.add_noise`
as ``z_t = scheduler.add_noise(x_tgt, x_src, t) = t·x_tgt + (1−t)·x_src`` (the
transport is the scheduler's, never re-derived — ADR-0001, so train and inference
cannot drift; the scheduler class is shared unchanged, ADR-0013), runs the UNet on
``concat([z_t, x_src])`` with the summed-label condition ``embed(src)+embed(tgt)``
to predict ``x_tgt``, and forms the ``(1 − t)⁻²``-weighted x0-MSE against it.

The module owns **no** ``scale_factor`` (ADR-0003): both src and tgt latents in
the batch are already scaled (the VAE's ``encode`` scaled them; the data stack
estimates one ``scale_factor`` over src+tgt pooled). It wires the manual-
optimization training a Lightning ``Trainer`` needs to ``fit`` it: Adam over the
UNet + a cosine-with-warmup schedule (horizon in optimizer steps, divided by
world size), and an AMP-aware grad-norm hook (mirrors ``LatentFlowModule``). The
:meth:`sample` method delegates to the shared rollout primitive so in-training
generation (the PSNR/SSIM callback, Slice 3) and inference cannot drift.

The objective mirrors :class:`manifold.modules.LatentFlowModule`; only the
transport endpoints (data src at ``t = 0``, data tgt at ``t = 1``), the doubled
input (``concat``), and the summed-label conditioning differ.
"""

from __future__ import annotations

from typing import Any, Iterable

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.unet_3d_condition import UNet3DConditionModel
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from .latent_flow import cosine_with_warmup
from .paired_sampler import sample_paired_latent_flow

#: A Paired JiT training batch: scaled src + tgt latents, contrast labels, spacing.
#: The data stack (Slice 2) produces these; this module only consumes them.
PairedSampleDict = dict[str, Any]


class PairedLatentFlowModule(spt.Module):
    """Paired JiT src→tgt training module (``spt.Module``).

    Args:
        unet: the :class:`~manifold.UNet3DConditionModel` with
            ``in_channels = 2·C_latent`` (predicts x_tgt from ``concat([z_t, x_src])``).
        scheduler: the :class:`~manifold.FlowMatchHeunDiscreteScheduler`; its
            :meth:`add_noise` is the single source of truth for the transport
            (shared with the noise→data JiT, ADR-0013).
        p_mean / p_std: logit-normal timestep sampler parameters
            (``t ~ sigmoid(N(p_mean, p_std))``). Defaults are JiT's published
            values (``-0.8`` / ``0.8``).
        t_eps: floor on ``1 − t`` in the loss denominator (matches the sampler's
            endpoint clamp), avoiding the singularity as ``t → 1``.
        loss_weight: the x0-MSE weighting regime. ``"1mt_sq"`` (default, the JiT
            x0-denoiser velocity-MSE ``((x0−x_tgt)/(1−t))²``) or ``"uniform"``
            (plain ``(x0−x_tgt)²``). The ``(1−t)⁻²`` weight lets the model satisfy
            high-``t`` by copying ``z_t`` (correct there) while leaving the low-``t``
            transport — which the start-from-``x_src`` rollout depends on —
            under-trained (predicted velocity 0 → the rollout stalls at ``x_src``).
            ``"uniform"`` + low-``t``-biased sampling (``p_mean < 0``) forces genuine
            low-``t`` translation (ADR-0013/0014 addendum).
        lr: Adam learning rate (paired is trained from scratch, ADR-0014).
        lr_warmup_steps: cosine-schedule warmup, in optimizer steps.
        num_train_examples: ``len(train paired dataset)`` — together with
            ``train_batch_size`` / ``n_epochs`` it fixes the cosine horizon.
        train_batch_size: per-device (per-rank) batch size.
        n_epochs: the schedule's epoch horizon.
    """

    def __init__(
        self,
        unet: UNet3DConditionModel,
        scheduler: FlowMatchHeunDiscreteScheduler,
        *,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        t_eps: float = 0.05,
        loss_weight: str = "1mt_sq",
        lr: float = 1.0e-4,
        lr_warmup_steps: int = 1000,
        num_train_examples: int | None = None,
        train_batch_size: int | None = None,
        n_epochs: int = 1,
    ):
        # NOTE: forward is NOT passed to spt.Module — it would double-bind self.
        # Overriding forward directly is the supported pattern (mirrors LatentFlowModule).
        super().__init__(
            hparams={
                "p_mean": p_mean,
                "p_std": p_std,
                "t_eps": t_eps,
                "loss_weight": loss_weight,
                "lr": lr,
                "lr_warmup_steps": lr_warmup_steps,
            }
        )
        self.unet = unet
        self.scheduler = scheduler
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = float(t_eps)
        if loss_weight not in ("1mt_sq", "uniform"):
            raise ValueError(
                f"loss_weight must be '1mt_sq' or 'uniform', got {loss_weight!r}."
            )
        self.loss_weight = str(loss_weight)
        self.lr = float(lr)
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.num_train_examples = (
            None if num_train_examples is None else int(num_train_examples)
        )
        self.train_batch_size = None if train_batch_size is None else int(train_batch_size)
        self.n_epochs = int(n_epochs)
        #: Last AMP-corrected grad norm (stashed by ``after_manual_backward``) so
        #: it is unit-testable without a Trainer / logger.
        self._last_grad_norm: float | None = None

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Logit-normal ``t ~ sigmoid(N(p_mean, p_std))`` in ``(0, 1)``."""
        logits = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return torch.sigmoid(logits)

    def forward(self, batch: PairedSampleDict, stage: str) -> dict[str, Tensor]:
        """Paired JiT forward: the x0-MSE on the target latent.

        The interpolated latent comes from ``scheduler.add_noise(x_tgt, x_src, t)``
        (the transport is the scheduler's). The UNet input is ``concat([z_t, x_src])``
        and the loss is the x0-MSE on ``x_tgt``: ``(1 − t)⁻²``-weighted (velocity-MSE,
        the default) when ``loss_weight="1mt_sq"``, or plain MSE when
        ``loss_weight="uniform"`` (see :meth:`__init__`).
        """
        x_src = batch["src_latent"]  # already scaled (ADR-0003): no scale_factor here
        x_tgt = batch["tgt_latent"]  # already scaled (same VAE / scale_factor)
        device = x_src.device
        t = self._sample_timesteps(x_tgt.shape[0], device)  # (B,) in (0, 1)
        noisy = self.scheduler.add_noise(x_tgt, x_src, t)  # z_t = t·x_tgt + (1−t)·x_src

        sample = torch.cat([noisy, x_src], dim=1)  # in_channels = 2·C_latent (ADR-0014)
        x0_pred = self.unet(
            sample=sample,
            timestep=t,
            spacing=batch["spacing"],
            class_labels_src=batch["src_label"],
            class_labels_tgt=batch["tgt_label"],
        )

        if self.loss_weight == "uniform":
            # Plain x0-MSE — weight low- and high-``t`` equally. The ``"1mt_sq"``
            # velocity-MSE lets the model satisfy high-``t`` by copying ``z_t``
            # while leaving low-``t`` transport under-trained (predicted velocity 0
            # → the start-from-``x_src`` rollout stalls). Uniform MSE + low-``t``-
            # biased sampling forces genuine low-``t`` translation.
            loss = F.mse_loss(x0_pred.float(), x_tgt.float())
        else:
            # ``(1 − t)⁻²``-weighted x0-MSE == the velocity-MSE
            # ``v = (x0 − z)/(1 − t)``; the JiT x0-denoiser weight (the default).
            t_b = t.view(-1, *([1] * (x_tgt.ndim - 1)))  # broadcast over spatial dims
            weight = (1.0 - t_b).clamp(min=self.t_eps)
            loss = F.mse_loss(x0_pred.float() / weight, x_tgt.float() / weight)

        out: dict[str, Tensor] = {"loss": loss}
        if stage != "fit":
            out["pred"] = x0_pred.detach()  # the target-latent prediction
            out["target"] = x_tgt.detach()  # the target latent x_tgt
        return out

    def sample(
        self,
        src_latent,
        spacing,
        src_label: int,
        tgt_label: int,
        num_inference_steps: int,
    ) -> Tensor:
        """Translate ``src_latent`` → a predicted tgt latent (ADR-0005).

        In-training generation (the PSNR/SSIM callback) goes through the Module —
        never the inference Pipeline. This delegates to the shared
        :func:`~manifold.modules.sample_paired_latent_flow` primitive the Pipeline
        also uses, so the train and infer paths cannot drift.

        Because generation shares ``self.unet`` with training, an EMA shadow
        swapped into ``self.unet`` in place (the EMA callback, around eval) is
        seen here with no extra wiring — reported quality reflects the EMA model.
        """
        return sample_paired_latent_flow(
            self.unet,
            self.scheduler,
            src_latent,
            spacing,
            src_label,
            tgt_label,
            num_inference_steps=num_inference_steps,
        )

    # -- optimizer + grad-norm wiring (mirrors LatentFlowModule) ---------------

    def _total_optimizer_steps(self) -> int:
        """Cosine horizon in **optimizer steps**, divided by ``world_size``.

        ``steps_per_epoch = num_train_examples // (batch_size * world_size)`` and
        ``total = n_epochs * steps_per_epoch`` — so warmup/cosine are not slowed
        ``world_size``× under DDP. Returns a 1-step horizon when the dataset/batch
        are unset (unit tests / a quick smoke).
        """
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if self.num_train_examples and self.train_batch_size:
            denom = self.train_batch_size * world
            steps_per_epoch = max(1, self.num_train_examples // denom)
            return max(1, self.n_epochs * steps_per_epoch)
        return 1

    def configure_optimizers(self):
        """Adam over every UNet param + a step-interval cosine-with-warmup.

        Full fine-tune from scratch (ADR-0014 — no warm-start); the schedule
        horizon is in optimizer steps (see :meth:`_total_optimizer_steps`).
        """
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr)
        scheduler = cosine_with_warmup(
            optimizer, self.lr_warmup_steps, self._total_optimizer_steps()
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    def _trainer_or_none(self):
        """The attached Trainer, or ``None`` (the ``trainer`` property raises when unset)."""
        try:
            return self.trainer
        except RuntimeError:
            return None

    def _amp_scale(self) -> float:
        """The current AMP loss-scaler value, or ``1.0`` off-GPU / without a Trainer.

        Under ``16-mixed`` Lightning scales the loss before backward, so the raw
        gradient norm is inflated by this factor — dividing by it recovers the
        true magnitude.
        """
        trainer = self._trainer_or_none()
        if trainer is None:
            return 1.0
        plugin = getattr(trainer, "precision_plugin", None)
        scaler = getattr(plugin, "scaler", None) if plugin is not None else None
        if scaler is None:
            return 1.0
        scale = scaler.get_scale()
        return float(scale) if scale else 1.0

    def after_manual_backward(self):
        """AMP-corrected grad-norm hook (stashed + logged each step).

        Runs immediately after ``manual_backward`` (gradients populated, before
        the optimizer step). The raw L2 norm of the UNet gradients is divided by
        the AMP scale to recover the true gradient magnitude; the value is
        stashed on ``self._last_grad_norm`` (unit-testable without a Trainer) and
        logged as ``train/grad_norm`` (on step) when a Trainer is attached.
        """
        grad_norm = _grad_norm(self.unet.parameters()) / self._amp_scale()
        self._last_grad_norm = float(grad_norm)
        if self._trainer_or_none() is not None:
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> Tensor:
    """L2 norm of all populated ``.grad`` tensors (the UNet's, here)."""
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.zeros((), dtype=torch.float32)
    for g in grads:
        total = total + g.pow(2).sum()
    return total.sqrt()


__all__ = ["PairedLatentFlowModule", "PairedSampleDict"]
