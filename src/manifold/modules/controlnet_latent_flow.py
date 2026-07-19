"""ControlNet supervised training component (``spt.Module``) — ADR-0027.

A :class:`stable_pretraining.Module` implementing the **supervised ControlNet
stage**: the ControlNet is trained on the frozen noise→data JiT UNet to translate
``x_src`` → ``x_tgt`` before any GRPO. ``forward(batch, stage)`` samples flow-times
``t`` (logit-normal, the JiT sampler), obtains the noised target latent from the
**scheduler's** :meth:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler.add_noise`
(``z = t·x_tgt + (1−t)·ε`` — the canonical **noise→data** transport, never
re-derived, ADR-0001), runs the **frozen base UNet** with the ControlNet's residual
injections to predict ``x_tgt``, and forms the ``(1 − t)⁻²``-weighted velocity-MSE
against it (the same discipline the base was trained with). An optional L1 term on
the target prediction (default weight 0) preserves the L1+direction-offset lever
from the prior paired experiments (spec user story 7).

**Base frozen + unregistered; ControlNet is the only optimized arm.** The frozen
base UNet is held via ``object.__setattr__`` (bypassing ``nn.Module`` registration)
so it stays off ``parameters()`` / ``state_dict()`` / the optimizer / DDP — the
same pattern :class:`~manifold.modules.GRPOModule` uses to hold its frozen reward.
The ControlNet is a registered submodule (the only optimized + checkpointed arm).
The base receives only the target-contrast ``class_labels``; the ControlNet carries
the (src, tgt) direction conditioning. Gradient flows from the base output back to
the ControlNet through the base wrapper's out-of-place residual forward
(ADR-0026's corrected hazard — MONAI's native in-place adds would break this
backward).

The optimizer/schedule/grad-norm wiring **composes** the existing
:class:`~manifold.modules.LatentFlowModule` helpers (``scaled_peak_lr``,
``resolve_warmup_steps``, ``cosine_with_warmup``) rather than duplicating them
(project OOP rule); only the optimized parameter set differs (ControlNet, not the
UNet). No EMA: the raw optimizer arm is validated / selected / exported (ADR-0006;
spec's EMA mention corrected — the JiT supervised EMA it referenced was removed).
:meth:`sample` delegates to the shared
:func:`~manifold.modules.controlnet_rollout` primitive so validation (PSNR/SSIM)
and inference cannot drift (ADR-0005).
"""

from __future__ import annotations

from typing import Any, Iterable

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor

from ..models.controlnet_3d import ControlNet3DConditionModel
from ..models.unet_3d_condition import UNet3DConditionModel
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from .controlnet_sampler import controlnet_rollout
from .latent_flow import cosine_with_warmup, resolve_warmup_steps, scaled_peak_lr

#: A ControlNet supervised training batch: scaled src (control) + tgt (target)
#: latents, contrast labels, spacing. The paired data stack emits these; this
#: module consumes ``src_latent`` as the ControlNet condition and ``tgt_latent`` as
#: the noise→data transport's clean endpoint.
ControlNetSampleDict = dict[str, Any]


class ControlNetLatentFlowModule(spt.Module):
    """Supervised ControlNet training module (``spt.Module``).

    Args:
        unet: the **frozen** noise→data :class:`~manifold.UNet3DConditionModel`
            (predicts x0). Held unregistered — off the optimizer/checkpoint.
        controlnet: the trainable :class:`~manifold.ControlNet3DConditionModel`
            (the only optimized params); emits the residual injections.
        scheduler: the :class:`~manifold.FlowMatchHeunDiscreteScheduler`; its
            :meth:`add_noise` is the single source of truth for the transport.
        p_mean / p_std: logit-normal timestep sampler parameters
            (``t ~ sigmoid(N(p_mean, p_std))``; JiT's published defaults).
        t_eps: floor on ``1 − t`` in the loss denominator (matches the sampler's
            endpoint clamp), avoiding the singularity as ``t → 1``.
        l1_weight: optional L1 weight on the target prediction (default ``0.0`` —
            off). The L1+direction-offset lever from the prior paired experiments.
        lr: base Adam learning rate — the peak LR at ``lr_ref_batch_size``; the
            actual peak auto-scales with the effective batch
            (:func:`~manifold.modules.latent_flow.scaled_peak_lr`).
        lr_warmup_steps: cosine-schedule warmup, in optimizer steps (used only when
            ``lr_warmup_ratio`` is ``None``).
        lr_ref_batch_size: the effective batch size at which ``lr`` is the peak.
        lr_scale_rule: how the peak scales with the effective batch —
            ``"sqrt"`` (default), ``"linear"``, or ``"none"``.
        lr_warmup_ratio: optional warmup as a fraction of total optimizer steps.
        num_train_examples: ``len(train dataset)`` — fixes the cosine horizon with
            ``train_batch_size`` / ``n_epochs``.
        train_batch_size: per-device (per-rank) batch size.
        n_epochs: the schedule's epoch horizon.
    """

    def __init__(
        self,
        unet: UNet3DConditionModel,
        controlnet: ControlNet3DConditionModel,
        scheduler: FlowMatchHeunDiscreteScheduler,
        *,
        p_mean: float = -0.8,
        p_std: float = 0.8,
        t_eps: float = 0.05,
        l1_weight: float = 0.0,
        lr: float = 1.0e-4,
        lr_warmup_steps: int = 1000,
        lr_ref_batch_size: int = 8,
        lr_scale_rule: str = "sqrt",
        lr_warmup_ratio: float | None = None,
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
                "l1_weight": l1_weight,
                "lr": lr,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_ref_batch_size": lr_ref_batch_size,
                "lr_scale_rule": lr_scale_rule,
                "lr_warmup_ratio": lr_warmup_ratio,
            }
        )
        # The frozen base UNet, held UNregistered (object.__setattr__ bypasses
        # nn.Module registration) → absent from parameters()/state_dict()/optimizer/
        # DDP, moved to the device manually in on_fit_start (mirrors GRPOModule's
        # frozen reward). It is eval + grad-disabled: the supervised stage never
        # trains it.
        unet = unet.eval()
        for p in unet.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "unet", unet)

        # The trainable ControlNet — the only registered (optimized + checkpointed) arm.
        self.controlnet = controlnet
        self.scheduler = scheduler
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = float(t_eps)
        self.l1_weight = float(l1_weight)
        self.lr = float(lr)
        self.lr_warmup_steps = int(lr_warmup_steps)
        self.lr_ref_batch_size = int(lr_ref_batch_size)
        if lr_scale_rule not in ("sqrt", "linear", "none"):
            raise ValueError(
                f"lr_scale_rule must be 'sqrt' | 'linear' | 'none', got {lr_scale_rule!r}."
            )
        self.lr_scale_rule = str(lr_scale_rule)
        self.lr_warmup_ratio = None if lr_warmup_ratio is None else float(lr_warmup_ratio)
        self.num_train_examples = (
            None if num_train_examples is None else int(num_train_examples)
        )
        self.train_batch_size = None if train_batch_size is None else int(train_batch_size)
        self.n_epochs = int(n_epochs)
        #: Last AMP-corrected grad norm (stashed by ``after_manual_backward``).
        self._last_grad_norm: float | None = None

    def on_fit_start(self) -> None:
        """Move the unregistered frozen base to the device (Lightning's ``.to`` skips it)."""
        self.unet.to(self.device)

    def _sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Logit-normal ``t ~ sigmoid(N(p_mean, p_std))`` in ``(0, 1)`` (the JiT sampler)."""
        logits = torch.randn(batch_size, device=device) * self.p_std + self.p_mean
        return torch.sigmoid(logits)

    def forward(self, batch: ControlNetSampleDict, stage: str) -> dict[str, Tensor]:
        """ControlNet forward: ``(1 − t)⁻²``-weighted velocity-MSE on ``x_tgt``.

        The noised target latent comes from ``scheduler.add_noise(x_tgt, ε, t)``
        (the noise→data transport). The ControlNet emits the residual injections
        from ``(z_t, x_src, src/tgt labels)``; the frozen base consumes them to
        predict ``x_tgt``. The loss is ``mean(((x0 − x0_pred) / max(1 − t, t_eps))²)``
        plus an optional L1 term (``l1_weight``).
        """
        x_src = batch["src_latent"]  # the ControlNet control signal (already scaled)
        x_tgt = batch["tgt_latent"]  # the noise→data transport's clean endpoint (scaled)
        device = x_tgt.device
        noise = torch.randn_like(x_tgt)
        t = self._sample_timesteps(x_tgt.shape[0], device)  # (B,) in (0, 1)
        z_t = self.scheduler.add_noise(x_tgt, noise, t)  # z = t·x_tgt + (1−t)·ε

        spacing = batch["spacing"]
        src_labels = batch["src_label"]
        tgt_labels = batch["tgt_label"]

        # ControlNet residual injections (the grad-bearing arm); frozen base consumes
        # them through the out-of-place residual forward (ADR-0026's corrected hazard).
        down_res, mid_res = self.controlnet(
            sample=z_t,
            controlnet_cond=x_src,
            timestep=t,
            spacing=spacing,
            class_labels_src=src_labels,
            class_labels_tgt=tgt_labels,
        )
        x0_pred = self.unet(
            sample=z_t,
            timestep=t,
            spacing=spacing,
            class_labels=tgt_labels,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )

        # (1 − t)⁻²-weighted x0-MSE == the velocity-MSE the base was trained with.
        t_b = t.view(-1, *([1] * (x_tgt.ndim - 1)))  # broadcast over spatial dims
        weight = (1.0 - t_b).clamp(min=self.t_eps)
        loss = F.mse_loss(x0_pred.float() / weight, x_tgt.float() / weight)
        if self.l1_weight > 0.0:
            loss = loss + self.l1_weight * F.l1_loss(x0_pred.float(), x_tgt.float())

        out: dict[str, Tensor] = {"loss": loss}
        if stage != "fit":
            out["pred"] = x0_pred.detach()  # the target-latent prediction
            out["target"] = x_tgt.detach()  # the target latent x_tgt
        return out

    def sample(
        self,
        noise: Tensor,
        src_latent: Tensor,
        spacing,
        src_label: int | Tensor,
        tgt_label: int | Tensor,
        num_inference_steps: int,
    ) -> Tensor:
        """Generate a target latent from noise + the src control signal (ADR-0005).

        Validation generation (the PSNR/SSIM callback) goes through the Module and
        delegates to the shared :func:`~manifold.modules.controlnet_rollout`
        primitive the inference Pipeline also uses, so the train/validation and
        inference paths cannot drift. No EMA swap: the raw optimizer arm is sampled
        directly (ADR-0006).

        Args:
            noise: the pure-noise start latent ``[B, C_latent, D, H, W]`` (the
                noise→data transport's ``t = 0`` endpoint).
            src_latent: the source latent control signal ``[B, C_latent, D, H, W]``.
            spacing: raw voxel spacing ``[3]`` or ``[B, 3]``.
            src_label / tgt_label: the (src, tgt) contrast pair (scalar broadcast or
                ``[B]`` per-sample — a val batch mixes directions).
            num_inference_steps: Heun integration steps over ``t: 0 → 1``.

        Returns:
            The predicted target latent ``[B, C_latent, D, H, W]``.
        """
        return controlnet_rollout(
            self.unet,
            self.controlnet,
            self.scheduler,
            noise,
            src_latent,
            spacing,
            src_label,
            tgt_label,
            num_inference_steps=num_inference_steps,
        )

    # -- optimizer + grad-norm wiring (composes LatentFlowModule helpers) -------

    def _total_optimizer_steps(self) -> int:
        """Cosine horizon in **optimizer steps**, divided by ``world_size`` (mirrors
        :meth:`LatentFlowModule._total_optimizer_steps`)."""
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if self.num_train_examples and self.train_batch_size:
            denom = self.train_batch_size * world
            steps_per_epoch = max(1, self.num_train_examples // denom)
            return max(1, self.n_epochs * steps_per_epoch)
        return 1

    def configure_optimizers(self):
        """Adam over the **ControlNet params only** (the base is frozen/unregistered).

        The schedule horizon is in optimizer steps (``_total_optimizer_steps``); the
        Adam peak LR is auto-scaled from ``self.lr`` by the effective batch
        (:func:`scaled_peak_lr`) — composed from ``LatentFlowModule``, not duplicated.
        """
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if self.train_batch_size:
            peak_lr = scaled_peak_lr(
                self.lr,
                self.train_batch_size,
                ref_batch_size=self.lr_ref_batch_size,
                rule=self.lr_scale_rule,
                world_size=world,
            )
            eff_desc = f"{self.train_batch_size}/proc×{world} world = {self.train_batch_size * world}"
        else:
            # No per-device batch (unit-test smokes only) → effective batch unknown,
            # so do NOT scale: peak == base (mirrors LatentFlowModule).
            peak_lr = self.lr
            eff_desc = "unknown (train_batch_size=None) → no scaling, peak = base"
        total = self._total_optimizer_steps()
        warmup = resolve_warmup_steps(self.lr_warmup_steps, self.lr_warmup_ratio, total)
        rank_zero_info(
            "LR schedule: base=%.3e -> peak=%.3e (eff_batch: %s; rule=%s; "
            "warmup=%d/%d optimizer steps)",
            self.lr, peak_lr, eff_desc, self.lr_scale_rule, warmup, total,
        )
        optimizer = torch.optim.Adam(self.controlnet.parameters(), lr=peak_lr)
        scheduler = cosine_with_warmup(optimizer, warmup, total)
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
        """The current AMP loss-scaler value, or ``1.0`` off-GPU / without a Trainer."""
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
        """AMP-corrected grad-norm hook over the **ControlNet** grads (stashed + logged)."""
        grad_norm = _grad_norm(self.controlnet.parameters()) / self._amp_scale()
        self._last_grad_norm = float(grad_norm)
        if self._trainer_or_none() is not None:
            self.log("train/grad_norm", grad_norm, on_step=True, on_epoch=False, prog_bar=False)


def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> Tensor:
    """L2 norm of all populated ``.grad`` tensors (the ControlNet's, here)."""
    grads = [p.grad.detach().float() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    total = torch.zeros((), dtype=torch.float32)
    for g in grads:
        total = total + g.pow(2).sum()
    return total.sqrt()


__all__ = ["ControlNetLatentFlowModule", "ControlNetSampleDict"]
