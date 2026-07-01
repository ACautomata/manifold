"""JiT training component: the x0-denoiser objective as a stable-pretraining Module.

A :class:`stable_pretraining.Module` (:class:`spt.Module`) implementing the JiT
x0-denoiser training objective. ``forward(batch, stage)`` samples flow-times
``t`` from the logit-normal distribution, obtains the noised latent from the
**scheduler's** :meth:`~manifold.schedulers.FlowMatchHeunDiscreteScheduler.add_noise`
(the transport is the scheduler's, never re-derived — ADR-0001, so train and
inference cannot drift), runs the UNet to predict the clean latent x0, and forms
the ``(1 − t)⁻²``-weighted MSE against it.

The module owns **no** ``scale_factor`` (ADR-0003): latents in the batch are
already scaled (the VAE's ``encode`` scaled them). It also wires the manual-
optimization training a Lightning ``Trainer`` needs to ``fit`` it: Adam over the
UNet + a cosine-with-warmup schedule (horizon in optimizer steps, divided by
world size), and an AMP-aware grad-norm hook (issue #25).

The x0-denoiser forward: ``t = 1 → data`` matches the scheduler and the sampler.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.unet_3d_condition import UNet3DConditionModel
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from .sampler import sample_latent_flow

#: A training batch: a scaled latent + medical conditioning. The data stack
#: (deferred) produces these; this module only consumes them.
SampleDict = dict[str, Any]


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup → cosine-to-zero LR schedule (a ``LambdaLR``).

    Reproduces ``diffusers.get_scheduler("cosine_with_warmup", ...)`` exactly:
    linear ramp ``0 → lr`` over ``num_warmup_steps``, then a cosine decay
    ``lr → 0`` over the remaining ``num_training_steps − num_warmup_steps``.
    Implemented with a ``LambdaLR`` so manifold stays free of a ``diffusers``
    runtime dependency (ADR-0001's install-surface discipline).
    """
    warmup = max(0, int(num_warmup_steps))
    total = max(1, int(num_training_steps))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup:
            return float(current_step) / float(max(1, warmup))
        progress = float(current_step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LatentFlowModule(spt.Module):
    """JiT x0-denoiser training module (``spt.Module``).

    Args:
        unet: the :class:`~manifold.UNet3DConditionModel` (predicts x0).
        scheduler: the :class:`~manifold.FlowMatchHeunDiscreteScheduler`; its
            :meth:`add_noise` is the single source of truth for the transport.
        p_mean / p_std: logit-normal timestep sampler parameters
            (``t ~ sigmoid(N(p_mean, p_std))``). Defaults are JiT's published
            values (``-0.8`` / ``0.8``).
        t_eps: floor on ``1 − t`` in the loss denominator (matches the sampler's
            endpoint clamp), avoiding the singularity as ``t → 1``.
        include_modality: whether the UNet takes a class-label condition.
        lr: Adam learning rate (JiT recipe default ``1e-4``).
        lr_warmup_steps: cosine-schedule warmup, in optimizer steps.
        num_train_examples: ``len(train latent dataset)`` — together with
            ``train_batch_size`` / ``n_epochs`` it fixes the cosine horizon
            (``None`` → a 1-step horizon, used by unit tests without a real
            dataset).
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
        include_modality: bool = True,
        lr: float = 1.0e-4,
        lr_warmup_steps: int = 1000,
        num_train_examples: int | None = None,
        train_batch_size: int | None = None,
        n_epochs: int = 1,
    ):
        # NOTE: forward is NOT passed to spt.Module — it would double-bind self.
        # Overriding forward directly is the supported pattern.
        super().__init__(
            hparams={
                "p_mean": p_mean,
                "p_std": p_std,
                "t_eps": t_eps,
                "lr": lr,
                "lr_warmup_steps": lr_warmup_steps,
            }
        )
        self.unet = unet
        self.scheduler = scheduler
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        self.t_eps = float(t_eps)
        self.include_modality = bool(include_modality)
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

    def forward(self, batch: SampleDict, stage: str) -> dict[str, Tensor]:
        """JiT x0-denoiser forward: ``(1 − t)⁻²``-weighted MSE on the clean latent.

        The noised latent comes from ``scheduler.add_noise`` (the transport is the
        scheduler's). The loss is ``mean(((x0 − x0_pred) / max(1 − t, t_eps))²)``.
        """
        latent = batch["latent"]  # already scaled (ADR-0003): no scale_factor here
        device = latent.device
        noise = torch.randn_like(latent)
        t = self._sample_timesteps(latent.shape[0], device)  # (B,) in (0, 1)
        noisy = self.scheduler.add_noise(latent, noise, t)  # z = t·x + (1−t)·e

        spacing = batch["spacing"]
        kwargs: dict[str, Any] = {"spacing": spacing}
        if self.include_modality and "label" in batch:
            kwargs["class_labels"] = batch["label"]

        x0_pred = self.unet(sample=noisy, timestep=t, **kwargs)

        t_b = t.view(-1, *([1] * (latent.ndim - 1)))  # broadcast over spatial dims
        weight = (1.0 - t_b).clamp(min=self.t_eps)
        loss = F.mse_loss(x0_pred.float() / weight, latent.float() / weight)

        out: dict[str, Tensor] = {"loss": loss}
        if stage != "fit":
            out["pred"] = x0_pred.detach()  # the clean-latent prediction
            out["target"] = latent.detach()  # the clean latent x0
        return out

    def sample(
        self,
        target_shape,
        spacing,
        modality: int,
        num_inference_steps: int,
        *,
        guidance_scale: float = 1.0,
        cfg_interval: tuple[float, float] | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate a latent ``[B, C_latent, D, H, W]`` from pure noise (ADR-0005).

        In-training generation (the FID callback) goes through the Module — never
        the inference Pipeline. This delegates to the shared
        :func:`~manifold.modules.sample_latent_flow` primitive the Pipeline also
        uses, so the train and infer paths cannot drift.

        Because generation shares ``self.unet`` with training, an EMA shadow
        swapped into ``self.unet`` in place (the EMA callback, around eval) is
        seen here with no extra wiring — reported quality reflects the EMA model.

        Args mirror :meth:`~manifold.LatentFlowPipeline.sample_latent`: same
        generator + shape produces a bit-identical latent (parity).
        """
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        noise = torch.randn(target_shape, generator=generator, device=device, dtype=dtype)
        return sample_latent_flow(
            self.unet,
            self.scheduler,
            noise,
            spacing,
            modality,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            cfg_interval=cfg_interval,
        )

    # -- optimizer + grad-norm wiring (issue #25) ----------------------------

    def _total_optimizer_steps(self) -> int:
        """Cosine horizon in **optimizer steps**, divided by ``world_size``.

        ``steps_per_epoch = num_train_examples // (batch_size * world_size)`` and
        ``total = n_epochs * steps_per_epoch`` — so warmup/cosine are not slowed
        ``world_size``× under DDP (each rank sees ``1/world_size`` of the data
        per step). Returns a 1-step horizon when the dataset/batch are unset
        (unit tests / a quick smoke).
        """
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if self.num_train_examples and self.train_batch_size:
            denom = self.train_batch_size * world
            steps_per_epoch = max(1, self.num_train_examples // denom)
            return max(1, self.n_epochs * steps_per_epoch)
        return 1

    def configure_optimizers(self):
        """Adam over every UNet param + a step-interval cosine-with-warmup.

        Full fine-tune (no LoRA, no frozen subsets); the schedule horizon is in
        optimizer steps (see :meth:`_total_optimizer_steps`).
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
