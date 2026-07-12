"""``manifold-train-paired-grpo`` console entry + the testable G2RPO-training core.

The console entry (issue #103) composes the experiment config, builds the
:class:`PairedGRPOModule` (trainable Paired JiT policy UNet + frozen paired reward)
+ the data inputs (real paired policy + reward, or a fake via the ``data_provider``
injection seam for the CPU smoke), and calls ``Trainer.fit``. The G2RPO job is
standalone: decoupled from Paired JiT pretraining and from paired-reward training,
independently resumable.

The integration core :func:`run_paired_grpo_training` (Module + datamodule +
``ModelCheckpoint`` + ``build_trainer`` + ``fit``) is split out so a tiny CPU smoke
can drive it with a fake policy + toy source latents (the issue's testing seam)
instead of the real paired checkpoint + trained reward. The real-data launch path
(loading the slow-EMA paired UNet + the trained paired reward, wiring the paired
latent cache, the η-ramp, the raw-arm export) is gated on the bridge-noise
reward-ranking probe + a tiny-config measurement (#106); the ``data_provider`` seam
ships in #103, ``_real_inputs`` in #104.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import torch

try:
    import lightning.pytorch as pl
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl  # type: ignore

from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import opt
from ..data.datamodule import build_datamodule
from ..metrics import PairedPSNRSSIMCallback
from ..modules.paired_grpo import PairedGRPOModule
from ..pipelines.paired_latent_flow import PairedLatentFlowPipeline
from ..schedulers.scheduling_flow_match_bridge_grpo import FlowMatchBridgeGRPOScheduler
from .trainer import build_trainer, is_multi_gpu

_log = logging.getLogger(__name__)


@dataclass
class PairedGRPOInputs:
    """Module-construction + data inputs for one G2RPO run.

    ``policy`` / ``reward_model`` / ``scheduler`` go to the :class:`PairedGRPOModule`
    ctor; ``train_ds`` / ``val_ds`` emit the source latent + the contrast direction
    (``{src_latent, src_label, tgt_label, spacing}`` — pure-RL, the target volume is
    unused at train). The ``data_provider`` seam injects a fake policy + toy source
    latents for the CPU smoke; the real path (``_real_inputs``, #104) loads the
    slow-EMA paired UNet + the trained paired reward.
    """

    policy: Any
    reward_model: Any
    scheduler: Any
    train_ds: Any
    val_ds: Any
    reference_policy: Any = None  # the frozen KL anchor (ADR-0015); None ⇒ no KL (v1)
    vae: Any = None  # the frozen VAE for the PSNR/SSIM decode (#105); None ⇒ val/mean_reward only


class GuardedModelCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` that gates best-checkpoint selection on a guardrail metric.

    G2RPO selects on ``val/psnr`` (max) - the reproducible deterministic-Heun goal
    metric - BUT only among checkpoints whose ``val/ssim >= guardrail_min`` (the
    anti-artifact guardrail, ADR-0024). A high-``val/psnr``-but-low-``val/ssim``
    checkpoint (e.g. a reward-hacked or structurally-artifacted generation) is rejected
    from "best" selection: :meth:`check_monitor_top_k` returns False while the
    guardrail is unmet, so ``_save_monitor_checkpoint`` skips it (``save_last`` still
    keeps the latest weights for resume). When the guardrail is ``None`` the selection
    is the stock top-k (the #103 ``val/mean_reward`` path).

    The guardrail metric is read live from ``trainer.callback_metrics`` at the
    save-decision point (the PSNR callback logs ``val/ssim`` in the same validation
    epoch), so no metric plumbing is needed beyond attaching the PSNR callback.
    """

    def __init__(
        self, *args,
        guardrail_metric: str | None = None,
        guardrail_min: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.guardrail_metric = guardrail_metric
        self.guardrail_min = guardrail_min

    def check_monitor_top_k(self, trainer, current=None) -> bool:
        if current is None:
            return False
        if self.guardrail_metric is not None and self.guardrail_min is not None:
            gr = trainer.callback_metrics.get(self.guardrail_metric)
            # Reject non-finite (NaN/Inf) guardrail values: a NaN val/ssim (from an
            # unstable decode/SSIM) makes ``float(gr) < guardrail_min`` False, so a
            # NaN-ssim checkpoint would slip past the guardrail and be selected as "best".
            # Treat missing-or-non-finite as below the guardrail (codex #108).
            if gr is None or not torch.isfinite(gr) or float(gr) < float(self.guardrail_min):
                return False
        return super().check_monitor_top_k(trainer, current)


class EtaRampCallback(pl.Callback):
    """Ramps the bridge scheduler's ``eta`` from ``eta_min`` -> ``eta_max`` over the
    first ``ramp_fraction`` of total optimizer steps, then holds (ADR-0024 Q7).

    The paired UNet was trained on a **zero-noise** deterministic transport (unlike the
    JiT UNet, trained on noisy inputs), so a static-high ``eta`` shocks it (off-manifold
    suffix inputs -> degraded ``x_hat_1`` -> contaminated reward, off the grad path).
    The ramp bounds both the reward-spread OOD and the suffix-init OOD early; it is
    ramp-and-hold, not a permanent cut (ADR-0015 rejected static ``eta`` reduction as
    the *fix*; the ramp is a transient warm-up). The bridge rollout reads
    ``scheduler.eta`` at each ``training_step``, so updating it at
    ``on_train_batch_start`` (before the rollout) is the load-bearing point.
    """

    def __init__(
        self, scheduler: FlowMatchBridgeGRPOScheduler,
        eta_min: float, eta_max: float, ramp_fraction: float = 0.3,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.ramp_fraction = float(ramp_fraction)

    def eta_at(self, step: int, total_steps: int) -> float:
        """The ramped ``eta`` at ``step`` of ``total_steps`` (pure, testable).

        ``step`` / ``total_steps`` are BATCH counts (not optimizer steps): under manual
        optimization ``trainer.global_step`` counts ``opt.step()`` calls (= ``len(eta_step_list)``
        per batch), so driving the ramp off it would finish ``len(eta_step_list)``x too early.
        The callback passes ``batch_idx`` + ``estimated_stepping_batches`` (both batch counts)."""
        ramp_steps = max(1.0, self.ramp_fraction * float(total_steps))
        frac = min(1.0, float(step) / ramp_steps)
        return self.eta_min + (self.eta_max - self.eta_min) * frac

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # Batch-timeline (NOT global_step): estimated_stepping_batches is the BATCH count
        # and batch_idx is the batch index, so the ramp spans ramp_fraction of batches.
        # global_step counts opt steps (len(eta_step_list) per batch) -> would finish
        # Lx too early (codex #109 P2).
        total = int(trainer.estimated_stepping_batches) or 1
        self.scheduler.eta = self.eta_at(int(batch_idx), total)


def calibrate_reward_temp(reward_model, samples: torch.Tensor, *, batch_size: int = 8) -> float:
    """Score real ``cat([x_src, x_tgt])`` pairs through the paired reward; return the std.

    The tanh reward bound's ``reward_temp`` should be ``~`` the real-data reward std so
    the in-distribution range spreads across the tanh's soft-clip (ADR-0015 / ADR-0024).
    Calibration scores a sample of REAL condition-aware pairs (``[N, 2*C_latent, ...]``,
    built offline) + returns ``rewards.std()`` (Bessel). The paired PatchGAN scored real
    val/train latents at ``-9.7~-11.2 +- 8`` (range ``[-21, +26]``) in the noise->data
    calibration; the paired reward's scale is measured the same way at G2RPO startup.

    Runs under ``no_grad`` + ``eval`` (a forward-only measurement - the reward is frozen).
    """
    reward_model.eval()
    rewards = []
    with torch.no_grad():
        for s in range(0, int(samples.shape[0]), int(batch_size)):
            rewards.append(reward_model(samples[s : s + batch_size]).float())
    r = torch.cat(rewards)
    if r.numel() < 2:
        raise ValueError(
            f"need >=2 real reward samples to compute a std, got {r.numel()}; "
            "pass more calibration samples."
        )
    return float(r.std())


def _calibrate_reward_temp_from_val(module, val_ds, *, n: int = 16) -> None:
    """Measure ``reward_temp`` from real ``cat([x_src, x_tgt])`` val pairs; mutate the module.

    Pulls up to ``n`` real val pairs (each item's ``src_latent`` + ``tgt_latent`` are
    already scaled - ADR-0021), concatenates them into the condition-aware
    ``[N, 2*C_latent, ...]`` layout the paired reward scores, and sets
    ``module.reward_temp`` to the rewards' std (ADR-0015 / ADR-0024). A no-op when the
    val set has no tgt or the bound is not tanh - the config value stands.
    """
    if module.reward_bound != "tanh":
        return
    samples = []
    for i in range(min(n, len(val_ds))):
        it = val_ds[i]
        src = it.get("src_latent")
        tgt = it.get("tgt_latent")
        if src is None or tgt is None:
            break  # no tgt in this val set -> skip calibration (config value stands)
        samples.append(torch.cat([src, tgt], dim=0))
    if len(samples) < 2:
        _log.warning(
            "reward_temp calibration skipped (only %d val pairs with tgt); using the "
            "config reward_temp=%s.", len(samples), module.reward_temp,
        )
        return
    batch = torch.stack(samples).to(next(module.reward_model.parameters()).device)
    module.reward_temp = calibrate_reward_temp(module.reward_model, batch)
    _log.info("Calibrated reward_temp=%.4f from %d real val pairs (reward std).",
              module.reward_temp, len(samples))


# -- Slice 4 (#106): the launch gate ------------------------------------------
#
# A standalone bridge-noise reward-ranking probe (the HARD gate): the paired reward
# was trained real-vs-DETERMINISTIC-fake (ADR-0020); G2RPO scores bridge-noised fakes.
# If the reward can't rank them (acc ~ random), G2RPO silently random-walks (R1). The
# probe generates G bridge-branch siblings per source at eta_max, scores reward +
# PSNR-to-x_tgt, and checks the ranking agreement; acc > threshold passes, else the
# launch refuses (escalation: retrain the reward with bridge-noised fakes).


def _latent_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample latent-space PSNR (dB) - higher = closer to target.

    A quality surrogate for the probe (the honest metric is image-space PSNR via the
    PSNR callback, but the probe needs a cheap per-sibling score over many bridge draws).
    ``data_range`` is the per-sample target ``[max - min]``; a degenerate (constant)
    target returns +inf (the sibling ranking is undefined - excluded by the caller).
    """
    pred = pred.float()
    target = target.float()
    n = pred.shape[0]
    out = torch.full((n,), float("inf"))
    for i in range(n):
        t = target[i : i + 1]
        rng = float(t.max() - t.min())
        if rng <= 0.0:
            continue  # constant target - PSNR undefined
        mse = float((pred[i : i + 1] - t).pow(2).mean())
        if mse <= 0.0:
            out[i] = 100.0  # bit-identical -> cap at 100 dB
        else:
            out[i] = 10.0 * torch.log10(torch.tensor(rng * rng / mse))
    return out


@torch.no_grad()
def bridge_noise_reward_ranking_probe(
    unet, scheduler: FlowMatchBridgeGRPOScheduler, reward_model,
    x_src: torch.Tensor, x_tgt: torch.Tensor, spacing, src_label, tgt_label,
    *,
    G: int = 2, perturbed_step: int = 1, num_steps: int = 8,
    batch_size: int = 4,
) -> dict:
    """The bridge-noise reward-ranking launch gate (ADR-0024 R1 / #106).

    For each source: generate ``G`` bridge-branch siblings (one bridge SDE draw off the
    anchor at ``perturbed_step`` at ``eta_max``, deterministic Heun suffix to ``z_K``),
    score with the frozen paired reward, and compute a PSNR-to-``x_tgt`` quality
    surrogate per sibling. The ranking agreement (top-1 acc: does the reward's
    best-ranked sibling match the PSNR's best?) tells whether the reward can rank
    bridge-noised fakes - the signal ``group_advantage`` consumes. ``acc > threshold``
    (default 0.6) passes the gate; ``acc ~ 1/G`` (random) means G2RPO would silently
    random-walk -> refuse the launch + retrain the reward with bridge-noised fakes.

    Default ``G=2`` (pairwise preference, matching the paired reward's Bradley-Terry
    training - random baseline 0.5; the design's ``acc ~ 0.5 => fail``).

    Returns:
        ``{"acc", "n", "G", "perturbed_step", "eta"}`` - ``acc`` is the fraction of
        sources whose reward's top-1 sibling matches the PSNR's top-1.
    """
    import math as _math

    from ..modules.paired_grpo import _heun_rollout_paired, _paired_unet_call
    from ..modules.paired_sampler import _as_label_tensor

    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    src = x_src if isinstance(x_src, torch.Tensor) else torch.stack(list(x_src))
    tgt = x_tgt if isinstance(x_tgt, torch.Tensor) else torch.stack(list(x_tgt))
    if src.shape != tgt.shape:
        raise ValueError(f"x_src {tuple(src.shape)} and x_tgt {tuple(tgt.shape)} must match.")
    n = src.shape[0]
    spatial = src.shape[1:]
    spacing_t = torch.as_tensor(spacing, device=device)
    src_labels_full = _as_label_tensor(src_label, n, device)
    tgt_labels_full = _as_label_tensor(tgt_label, n, device)
    nodes = scheduler.set_timesteps(num_steps, device=device)  # (num_steps+1,)
    if perturbed_step >= num_steps - 1:
        raise ValueError(
            f"perturbed_step ({perturbed_step}) must be < num_steps-1 ({num_steps-1}) to "
            "avoid the §7 var-collapse terminal (the probe branches at this step)."
        )
    unet.eval()
    reward_model.eval()
    agree = 0
    scored = 0
    for s in range(0, n, batch_size):
        b = min(batch_size, n - s)
        src_b = src[s : s + b].to(device=device, dtype=dtype)
        tgt_b = tgt[s : s + b].to(device=device, dtype=dtype)
        spacing_b = spacing_t.repeat_interleave(b, dim=0) if spacing_t.dim() == 2 else spacing_t
        src_lab = src_labels_full[s : s + b] if isinstance(src_labels_full, torch.Tensor) else src_labels_full
        tgt_lab = tgt_labels_full[s : s + b] if isinstance(tgt_labels_full, torch.Tensor) else tgt_labels_full
        # Anchor rollout to the perturbed step (the deployed-Heun path from x_src).
        anchor_z = _heun_rollout_paired(
            unet, scheduler, src_b, src_b, nodes, spacing_b, src_lab, tgt_lab, 0, perturbed_step
        )
        z_k = anchor_z[perturbed_step]
        t_k = float(nodes[perturbed_step])
        t_next = float(nodes[perturbed_step + 1])
        # G bridge siblings off z_k at eta_max (the bridge SDE draw).
        x0 = _paired_unet_call(unet, z_k, src_b, t_k, spacing_b, src_lab, tgt_lab)
        mean_old, std_old = scheduler.sde_step_mean(x0, z_k, t_k, t_next)
        xi = torch.randn(b, G, *spatial, device=device, dtype=dtype)
        z_kplus1 = mean_old.unsqueeze(1) + float(std_old) * xi  # (b, G, *spatial)
        # Deterministic Heun suffix to the terminal z_K (thread x_src_bg).
        src_labels_bg = src_lab.repeat_interleave(G) if isinstance(src_lab, torch.Tensor) else src_lab
        tgt_labels_bg = tgt_lab.repeat_interleave(G) if isinstance(tgt_lab, torch.Tensor) else tgt_lab
        spacing_bg = spacing_b.repeat_interleave(G, dim=0) if spacing_b.dim() == 2 else spacing_b
        x_src_bg = src_b.repeat_interleave(G, dim=0)
        z_g = z_kplus1.reshape(b * G, *spatial)
        suffix = _heun_rollout_paired(
            unet, scheduler, z_g, x_src_bg, nodes, spacing_bg, src_labels_bg, tgt_labels_bg,
            perturbed_step + 1, num_steps,
        )
        z_K = suffix[-1]  # (b·G, *spatial)
        # Reward + PSNR-to-x_tgt per sibling, then per-source top-1 agreement.
        rewards = reward_model(torch.cat([x_src_bg, z_K], dim=1)).float().reshape(b, G)
        tgt_bg = src_b.repeat_interleave(G, dim=0)  # placeholder; replaced below
        tgt_bg = tgt_b.repeat_interleave(G, dim=0)
        psnr = _latent_psnr(z_K, tgt_bg).reshape(b, G)
        # Exclude sources whose PSNR is all-inf (constant tgt -> ranking undefined).
        valid = torch.isfinite(psnr).all(dim=1)
        if not valid.any():
            continue
        reward_top = rewards[valid].argmax(dim=1)
        psnr_top = psnr[valid].argmax(dim=1)
        agree += int((reward_top == psnr_top).sum())
        scored += int(valid.sum())
    if scored == 0:
        raise ValueError(
            "bridge-noise reward-ranking probe scored 0 sources (all val targets "
            "degenerate - constant volumes); pass non-constant x_tgt."
        )
    return {"acc": agree / scored, "n": scored, "G": G,
            "perturbed_step": int(perturbed_step), "eta": float(scheduler.eta)}


def _run_probe(module, inputs, cfg, *, n_probe: int = 16) -> float:
    """Pull real val (src, tgt) pairs + run the bridge-noise reward-ranking probe.

    The probe gate runs on the INIT policy (before any G2RPO update) at eta_max - the
    worst case for the reward's bridge-noised-fake ranking (the policy is the smooth
    slow-EMA arm; its bridge siblings are the fakes G2RPO would score). Returns the acc
    (fraction of sources whose reward's top-1 sibling matches PSNR's top-1).
    """
    gcfg = cfg.paired_grpo_train
    num_steps = int(opt(gcfg, "num_steps", 8))
    # Default the probe to a single mid first-half step (branch at step 1, away from the
    # var-collapse terminal) - the most informative single bridge-noise level.
    perturbed_step = int(opt(cfg, "paired_grpo.probe_step", 1))
    G = int(opt(cfg, "paired_grpo.probe_G", 2))
    # Pull (src, tgt) val pairs (the probe needs the GT target for the PSNR surrogate).
    srcs, tgts, spacings, src_labs, tgt_labs = [], [], [], [], []
    for i in range(min(n_probe, len(inputs.val_ds))):
        it = inputs.val_ds[i]
        if it.get("src_latent") is None or it.get("tgt_latent") is None:
            break
        srcs.append(it["src_latent"])
        tgts.append(it["tgt_latent"])
        spacings.append(torch.as_tensor(it["spacing"], dtype=torch.float32))
        src_labs.append(torch.as_tensor(it["src_label"], dtype=torch.long))
        tgt_labs.append(torch.as_tensor(it["tgt_label"], dtype=torch.long))
    if len(srcs) < 2:
        raise ValueError(
            "the probe needs >=2 val pairs with tgt_latent (the PSNR surrogate); the "
            "val_ds does not carry them. Run on the real path (val carries tgt for PSNR)."
        )
    x_src = torch.stack(srcs).to(module.device)
    x_tgt = torch.stack(tgts).to(module.device)
    spacing = torch.stack(spacings)
    res = bridge_noise_reward_ranking_probe(
        module.unet, module.scheduler, module.reward_model, x_src, x_tgt, spacing,
        torch.stack(src_labs), torch.stack(tgt_labs),
        G=G, perturbed_step=perturbed_step, num_steps=num_steps,
    )
    _log.info("bridge-noise probe: acc=%.3f (n=%d, G=%d, step=%d, eta=%.3f)",
              res["acc"], res["n"], res["G"], res["perturbed_step"], res["eta"])
    return res["acc"]


def _build_checkpoint(
    model_dir: str,
    *,
    monitor_metric: str = "val/mean_reward",
    mode: str = "max",
    save_top_k: int = 1,
    multi_gpu: bool = False,
    guardrail_metric: str | None = None,
    guardrail_min: float | None = None,
) -> ModelCheckpoint:
    """Stock Lightning ``ModelCheckpoint`` monitoring the G2RPO progress signal.

    #105 selects on ``val/psnr`` (mode ``max``) — the reproducible deterministic-Heun
    goal metric — when the PSNR callback is attached; #103 (the tracer) monitors
    ``val/mean_reward`` (mode ``max``) for the reward-only smoke. ``auto_insert_metric_name
    = False`` because the metric key contains a ``/``. ``save_last=True`` for resume.
    Under DDP the rank-local ``val/mean_reward`` is dropped (the PSNR callback's
    ``val/psnr`` is the global cross-rank selection metric, kept) — mirroring the
    noise→data GRPO checkpoint DDP fallback.
    """
    common = dict(
        dirpath=model_dir,
        save_top_k=save_top_k,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        save_weights_only=False,
    )
    if multi_gpu and monitor_metric == "val/mean_reward":
        # val/mean_reward is rank-0-only (validation_step gate); drop the monitor
        # under DDP (save_last + save_top_k=1 keep the latest). val/psnr is a global
        # cross-rank mean (the PSNR callback all_gathers) -> monitor stays on (#105).
        # Force save_top_k=1 (not the caller's): with no monitor there is no metric to
        # rank by, so keeping >1 is pointless disk waste - mirrors grpo_cli's DDP
        # fallback (codex #108).
        return ModelCheckpoint(filename="paired-grpo-{epoch:03d}", **{**common, "save_top_k": 1})
    ckwt = dict(
        filename=f"paired-grpo-{{epoch:03d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode=mode,
        **common,
    )
    if guardrail_metric is not None and guardrail_min is not None:
        return GuardedModelCheckpoint(
            guardrail_metric=guardrail_metric, guardrail_min=guardrail_min, **ckwt,
        )
    return ModelCheckpoint(**ckwt)


def run_paired_grpo_training(
    *,
    module: PairedGRPOModule,
    inputs: PairedGRPOInputs,
    model_dir: str,
    max_epochs: int,
    devices: int | str = "auto",
    accelerator: str = "auto",
    batch_size: int = 2,
    num_workers: int = 0,
    save_top_k: int = 1,
    monitor_metric: str | None = None,
    mode: str | None = None,
    limit_val_batches: int | float = 1.0,
    limit_train_batches: int | float | None = None,
    seed: int = 0,
    ckpt_path: str | None = None,
    ssim_guardrail: float | None = 0.9,
    psnr_num_inference_steps: int | None = None,
    eta_min: float | None = None,
    eta_ramp_fraction: float = 0.3,
) -> tuple[pl.Trainer, ModelCheckpoint]:
    """Assemble callbacks + ``Trainer`` and ``fit`` the G2RPO module (the core seam).

    When ``inputs.vae`` is set (#105), attaches
    :class:`~manifold.metrics.PairedPSNRSSIMCallback` (the deterministic-Heun
    src->tgt rollout + VAE decode, no EMA - G2RPO evaluates the raw policy) over the
    ``val_ds`` (which must then emit ``tgt_latent``), and selects the checkpoint on
    ``val/psnr`` (max) gated by ``val/ssim >= ssim_guardrail`` (a
    :class:`GuardedModelCheckpoint`). Otherwise monitors ``val/mean_reward`` (max),
    the #103 tracer default. ``val/mean_reward`` stays logged either way (the RL
    progress signal, rank-0-only).

    When ``eta_min`` is set (#104), attaches :class:`EtaRampCallback` warming the
    bridge ``eta`` from ``eta_min`` -> ``scheduler.eta`` (eta_max) over the first
    ``eta_ramp_fraction`` of total steps, then holds (ADR-0024 Q7 - the paired UNet
    is zero-noise-trained; a static-high eta shocks it). ``None`` (the CPU smoke) keeps
    a static eta.

    Args:
        inputs: the train/val source-latent datasets + the policy/reward/scheduler
            (+ the optional KL reference + the optional VAE for PSNR).
        monitor_metric / mode: ``None`` (the default) auto-selects ``val/psnr`` (max)
            when ``inputs.vae`` is set, else ``val/mean_reward`` (max). Override
            explicitly to force one.
        ssim_guardrail: the minimum ``val/ssim`` for a checkpoint to be eligible for
            "best" selection (ADR-0024); ``None`` disables the guardrail (stock
            top-k). Default 0.9.
        psnr_num_inference_steps: Heun steps for the PSNR rollout (defaults to
            ``module.num_steps`` - the deployed resolution).
        eta_min: the ramp start (ADR-0024 Q7); ``None`` disables the ramp.
        eta_ramp_fraction: fraction of total steps over which eta ramps.
        ckpt_path: optional warm-start / resume checkpoint passed to ``fit``.
    """
    pl.seed_everything(seed, workers=True)
    multi_gpu = is_multi_gpu(devices)
    psnr_active = inputs.vae is not None
    if monitor_metric is None:
        monitor_metric = "val/psnr" if psnr_active else "val/mean_reward"
    if mode is None:
        mode = "max"  # both val/psnr and val/mean_reward are mode=max
    guardrail = "val/ssim" if (psnr_active and ssim_guardrail is not None) else None
    ckpt = _build_checkpoint(
        model_dir, monitor_metric=monitor_metric, mode=mode, save_top_k=save_top_k,
        multi_gpu=multi_gpu, guardrail_metric=guardrail, guardrail_min=ssim_guardrail,
    )
    callbacks: list[pl.Callback] = [ckpt]
    if eta_min is not None:
        # The eta warm-up (ADR-0024 Q7): ramp eta_min -> eta_max (scheduler.eta) over
        # the first eta_ramp_fraction of steps, then hold. The rollout reads
        # scheduler.eta at training_step, so on_train_batch_start is the load-bearing
        # hook. ``eta_min`` is the ramp start; the scheduler's current ``eta`` is eta_max.
        callbacks.append(
            EtaRampCallback(
                module.scheduler, eta_min=float(eta_min), eta_max=float(module.scheduler.eta),
                ramp_fraction=float(eta_ramp_fraction),
            )
        )
    if psnr_active:
        pipeline = PairedLatentFlowPipeline(module.unet, inputs.vae, module.scheduler)
        psnr_steps = int(psnr_num_inference_steps) if psnr_num_inference_steps is not None else module.num_steps
        callbacks.append(
            PairedPSNRSSIMCallback(
                pipeline=pipeline,
                num_inference_steps=psnr_steps,
                ema_callback=None,  # G2RPO evaluates the raw policy (no EMA, ADR-0012)
            )
        )
    datamodule = build_datamodule(
        inputs.train_ds, batch_size=batch_size, val_dataset=inputs.val_ds, num_workers=num_workers
    )
    trainer = build_trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        model_dir=model_dir,
        devices=devices,
        accelerator=accelerator,
        limit_val_batches=limit_val_batches,
        extra_kwargs=(
            {"limit_train_batches": limit_train_batches}
            if limit_train_batches is not None
            else None
        ),
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
    return trainer, ckpt


# -- console entry -----------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manifold-train-paired-grpo",
        description="Post-train the Paired JiT UNet with Granular-GRPO (G2RPO) over the "
        "src→tgt Brownian bridge against the frozen paired reward (ADR-0024).",
    )
    parser.add_argument("-e", "--env", required=True, help="env config YAML (paths).")
    parser.add_argument(
        "-c", "--train", default="configs/train/config_paired_grpo.yaml",
        help="G2RPO recipe YAML.",
    )
    parser.add_argument(
        "-t", "--network", required=True, help="network construction YAML (latent metadata)."
    )
    parser.add_argument("-g", "--num-gpus", type=int, default=1, help="number of GPUs.")
    parser.add_argument("--max-epochs", type=int, default=None, help="override n_epochs.")
    parser.add_argument(
        "--native-dir",
        default=None,
        help="native paired export dir (the slow-EMA policy + VAE scale); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--reward-path",
        default=None,
        help="trained paired RewardModel checkpoint (.ckpt); required without --data-provider.",
    )
    parser.add_argument(
        "--latents-dir",
        default=None,
        help="paired latent cache dir (source latents + contrast labels); "
        "required without --data-provider.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="run a tiny-config measurement (it/s + peak GPU memory) and exit — the "
        "#106 launch gate; size G / eta_step_list / n_epochs before the full run.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run the bridge-noise reward-ranking launch gate (ADR-0024 R1) and exit: "
        "generate G bridge-branch siblings per source at eta_max, check the reward "
        "ranking agrees with a PSNR-to-x_tgt surrogate. Passes (rc=0) at acc >= "
        "--probe-threshold; fails (rc=1) when the reward can't rank bridge-noised "
        "fakes (escalation: retrain the reward with bridge-noised fakes).",
    )
    parser.add_argument(
        "--probe-threshold", type=float, default=0.6,
        help="bridge-noise reward-ranking acc threshold to pass --probe (default 0.6).",
    )
    parser.add_argument(
        "--resume", default=None, help="resume a Lightning .ckpt (trainer.fit(ckpt_path=...))."
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=None,
        help="cap train batches/epoch (a debug knob for the fast re-measure; the full "
        "run leaves it unset).",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style dotlist overrides.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, data_provider=None) -> int:
    """Console entry: compose config → build → ``run_paired_grpo_training``.

    ``data_provider`` is the injection seam for the CPU smoke test: a callable
    ``(cfg, device) -> PairedGRPOInputs`` returning a fake policy + toy source
    latents so the full ``main`` path runs without the real paired checkpoint +
    trained reward. The real path loads the slow-EMA paired UNet from ``--native-dir``
    and the paired reward from ``--reward-path`` (#104).
    """
    args = _parse_args(argv)

    from omegaconf import OmegaConf

    from ..config import load_config, merge_overrides, require_paths

    cfg = load_config(args.env, args.train, args.network)
    cfg = merge_overrides(cfg, {"num_gpus": args.num_gpus}, list(args.overrides))
    # G2RPO needs no VAE / image data — only the output model_dir is required.
    require_paths(cfg, keys=("model_dir",))
    OmegaConf.resolve(cfg)
    if getattr(cfg, "paired_grpo_train", None) is None:
        raise ValueError(
            "Config has no `paired_grpo_train` block — use the G2RPO recipe "
            "(-c configs/train/config_paired_grpo.yaml), not a Paired JiT train or "
            "reward config."
        )

    seed = int(opt(cfg, "random_seed", 0))
    pl.seed_everything(seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if data_provider is not None:
        inputs = data_provider(cfg, device)
    else:
        # --native-dir / --reward-path / --latents-dir are NOT argparse-required: that
        # would break the data_provider injection seam (the CPU smoke). Validate them
        # here, only on the real path.
        if not args.native_dir or not args.reward_path or not args.latents_dir:
            raise ValueError(
                "G2RPO needs --native-dir <native paired export (slow-EMA arm)>, "
                "--reward-path <trained paired RewardModel .ckpt>, and "
                "--latents-dir <paired latent cache> (or inject a data_provider for the smoke)."
            )
        inputs = _real_inputs(cfg, args.native_dir, args.reward_path, args.latents_dir, device)

    gcfg = cfg.paired_grpo_train
    module = PairedGRPOModule(
        inputs.policy,
        inputs.reward_model,
        inputs.scheduler if inputs.scheduler is not None
        else FlowMatchBridgeGRPOScheduler(eta=float(opt(gcfg, "eta", 0.7))),
        G=int(opt(gcfg, "G", 8)),
        eta_step_list=list(opt(gcfg, "eta_step_list", [0, 1, 2, 3])),
        clip_range=float(opt(gcfg, "clip_range", 1e-4)),
        lr=float(gcfg.lr),
        adv_clip_max=float(opt(gcfg, "adv_clip_max", 5.0)),
        num_steps=int(opt(gcfg, "num_steps", 8)),
        reference_policy=inputs.reference_policy,
        kl_coef=float(opt(gcfg, "kl_coef", 0.0)),
        reward_bound=str(opt(gcfg, "reward_bound", "none")),
        reward_temp=float(opt(gcfg, "reward_temp", 8.0)),
    )

    # Calibrate reward_temp from real val pairs at startup (ADR-0015/0024): the tanh
    # bound's temperature ~ the real-data reward std. The CPU smoke (data_provider)
    # uses the config value; the real path measures it.
    if data_provider is None:
        _calibrate_reward_temp_from_val(module, inputs.val_ds)

    if args.probe:
        # The bridge-noise reward-ranking launch gate (ADR-0024 R1): refuse the launch
        # if the reward can't rank bridge-noised fakes (G2RPO would silently random-walk).
        acc = _run_probe(module, inputs, cfg)
        threshold = float(args.probe_threshold)
        passed = acc >= threshold
        print(
            f"[manifold-train-paired-grpo] probe: bridge-noise reward-ranking acc={acc:.3f} "
            f"(G=...) | threshold={threshold:.2f} | {'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            _log.error(
                "G2RPO launch gate FAILED: the reward can't rank bridge-noised fakes "
                "(acc=%.3f < %.2f). G2RPO would silently random-walk (R1). Retrain the "
                "paired reward with bridge-noised fakes (build_paired_bridge_noised_fakes) "
                "before retrying.", acc, threshold,
            )
            return 1
        return 0

    if args.measure:
        # The #106 launch-gate measurement: a 1-epoch fit timing + peak GPU memory.
        # Forward eta_min so the measurement reflects the real (ramped) training path.
        it_per_s, peak, elapsed = run_paired_grpo_measurement(
            module=module,
            inputs=inputs,
            model_dir=str(cfg.model_dir),
            devices=args.num_gpus if args.num_gpus > 1 else 1,
            batch_size=int(gcfg.batch_size),
            seed=seed,
            eta_min=(float(opt(gcfg, "eta_min", 0.1)) if bool(opt(gcfg, "eta_ramp", True)) else None),
            eta_ramp_fraction=float(opt(gcfg, "eta_ramp_fraction", 0.3)),
        )
        print(
            f"[manifold-train-paired-grpo] measure: {it_per_s:.3f} it/s | "
            f"peak GPU {peak / 1e9:.2f} GB | {elapsed:.1f}s"
        )
        return 0

    run_paired_grpo_training(
        module=module,
        inputs=inputs,
        model_dir=str(cfg.model_dir),
        max_epochs=int(args.max_epochs or gcfg.n_epochs),
        devices=args.num_gpus if args.num_gpus > 1 else 1,
        batch_size=int(gcfg.batch_size),
        save_top_k=int(opt(cfg, "checkpoint.save_top_k", 1)),
        seed=seed,
        ckpt_path=args.resume,
        limit_train_batches=args.limit_train_batches,
        eta_min=(float(opt(gcfg, "eta_min", 0.1)) if bool(opt(gcfg, "eta_ramp", True)) else None),
        eta_ramp_fraction=float(opt(gcfg, "eta_ramp_fraction", 0.3)),
    )
    print(f"[manifold-train-paired-grpo] done; checkpoints under {cfg.model_dir}")
    return 0


def run_paired_grpo_measurement(
    *,
    module: PairedGRPOModule,
    inputs: PairedGRPOInputs,
    model_dir: str,
    devices: int | str = 1,
    accelerator: str = "auto",
    batch_size: int = 2,
    seed: int = 0,
    eta_min: float | None = None,
    eta_ramp_fraction: float = 0.3,
) -> tuple[float, int, float]:
    """Time a 1-epoch G2RPO fit + report it/s + peak GPU memory (the #106 launch gate).

    Sizes ``G`` / ``eta_step_list`` / ``n_epochs`` by measuring the real budget's
    throughput + peak GPU memory on the target cluster before committing to the full
    run. Returns ``(it_per_s, peak_gpu_bytes, elapsed_s)``. Peak memory is 0 off-CUDA
    (the read is GPU-only); a tiny ``--measure`` run on the cluster is the real signal.

    ``eta_min`` is forwarded to the fit so the measurement reflects the real training
    path (with the ramp), not a static-eta timing (codex #109 P2).
    """
    import time

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer, _ = run_paired_grpo_training(
        module=module,
        inputs=inputs,
        model_dir=model_dir,
        max_epochs=1,
        devices=devices,
        accelerator=accelerator,
        batch_size=batch_size,
        seed=seed,
        eta_min=eta_min,
        eta_ramp_fraction=eta_ramp_fraction,
    )
    elapsed = time.perf_counter() - start
    it_per_s = float(trainer.global_step) / elapsed if elapsed > 0 else float("nan")
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return it_per_s, peak, elapsed


def _real_inputs(
    cfg, native_dir: str, reward_path: str, latents_dir: str, device: torch.device
) -> PairedGRPOInputs:
    """Build the real G2RPO inputs from the slow-EMA paired UNet + the trained reward (#104).

    The slow-EMA Paired JiT UNet (ADR-0021 - the paired native export's arm, inverted
    for G2RPO: it is the **trainable** policy init, not a frozen generator) is the
    policy; a bit-identical frozen deep-copy is the KL reference (ADR-0015). The trained
    paired RewardModel (in_channels = 2*C_latent) is the frozen reward. The paired
    latent cache furnishes the source latents + contrast labels (train is pure-RL -
    no tgt; val carries tgt for PSNR). The VAE (for the PSNR decode, #105) and
    scaling_factor come from the native export.

    Mirrors paired_reward_cli._real_inputs (the closest prior art - it loads the
    paired generator + warms the paired cache) + grpo_cli._real_inputs (the
    reward-ckpt load + the KL-reference deepcopy). Launch is gated on the bridge-noise
    reward-ranking probe + a --measure run (#106) - not exercisable here (no real
    artifacts on the dev machine); the data_provider seam covers the CPU smoke.
    """
    import copy as _copy
    import os

    from ..config import autoencoder_divisor
    from ..data.paired_brats import build_brats_pair_manifest
    from ..data.paired_latent_dataset import PairedLatentDataset
    from ..data.paired_volume_dataset import PairedNiftiVolumeDataset
    from ..models.reward_model import RewardModel
    from ..pipelines.paired_latent_flow import PairedLatentFlowPipeline
    from .paired_cli import _train_val_manifests

    # 1. The slow-EMA paired UNet (trainable policy init) + VAE + base scheduler +
    #    scaling_factor, all from the paired native export (ADR-0021: the export baked
    #    the slow-EMA arm; G2RPO inverts it - the slow-EMA arm becomes the policy init,
    #    and the published arm is raw for this stage).
    pipe = PairedLatentFlowPipeline.from_pretrained(str(native_dir))
    policy = pipe.unet.to(device)
    for p in policy.parameters():  # G2RPO post-trains the policy (the reward is frozen).
        p.requires_grad_(True)
    vae = pipe.vae
    scaling_factor = float(vae.scaling_factor)
    base_sched_cfg = pipe.scheduler.config
    # The frozen KL anchor (ADR-0015): a bit-identical snapshot taken BEFORE any G2RPO
    # update (deepcopy, not a second from_pretrained). The Module freezes + unregisters it.
    reference_policy = _copy.deepcopy(policy)

    # 2. The bridge scheduler from the base scheduler's transport config + eta (the
    #    bridge is training-only; the native ckpt carries the base Heun config - #104 export).
    scheduler = FlowMatchBridgeGRPOScheduler(
        num_train_timesteps=int(base_sched_cfg.get("num_train_timesteps", 1000)),
        t_eps=float(base_sched_cfg.get("t_eps", 0.05)),
        eta=float(opt(cfg.paired_grpo_train, "eta", 0.7)),
    )

    # 3. The trained paired RewardModel (in_channels = 2*C_latent) from its
    #    PairedRewardModule Lightning checkpoint. Architecture from the network config;
    #    opt() falls back to RewardModel defaults if a non-standard network file omits it.
    reward_cfg = opt(cfg, "reward_model", {})
    latent_c = int(opt(cfg, "latent_channels", 4))
    reward_model = RewardModel(
        spatial_dims=int(opt(reward_cfg, "spatial_dims", 3)),
        # The paired reward scores condition-aware cat([x_src, z_K]) pairs -> in_channels
        # = 2*C_latent structurally (ADR-0019). The network config's
        # reward_model.in_channels is the JiT default (=latent_channels=C, NOT 2*C), so
        # the opt(..., 8) fallback was dead (codex #109 P1 - a 4-channel reward would
        # crash on the first 8-channel concat). Force 2*C_latent regardless of config
        # (mirrors paired_reward_cli; codex #96/#99 P1/P2).
        in_channels=2 * latent_c,
        channels=int(opt(reward_cfg, "channels", 64)),
        num_layers_d=int(opt(reward_cfg, "num_layers_d", 3)),
        norm=str(opt(reward_cfg, "norm", "BATCH")),
    )
    # weights_only=True (no arbitrary-code-execution risk). A PairedRewardModule ckpt
    # is state_dict (tensors) + ModelCheckpoint callback state (dicts of tensors/nums)
    # + optimizer_states (tensor dicts) - all allowlisted, so this never needs the
    # unsafe fallback. reward_path is the user's OWN trained reward (trusted); if a
    # future non-allowlisted global makes weights_only=True fail, surface the error
    # (never fall back to weights_only=False - that unpickles arbitrary objects).
    ckpt = torch.load(str(reward_path), map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    reward_sd = {k[len("reward_model."):]: v for k, v in state.items() if k.startswith("reward_model.")}
    if not reward_sd:
        raise ValueError(
            f"No 'reward_model.*' keys in {reward_path} - not a trained PairedRewardModule checkpoint."
        )
    reward_model.load_state_dict(reward_sd, strict=True)
    reward_model.eval().to(device)
    for p in reward_model.parameters():
        p.requires_grad_(False)

    # 4. The paired train/val split (ADR-0022): resolve via _train_val_manifests
    #    (val_data_base_dir / val_fraction), reuse the existing paired_train cache.
    inf_cfg = cfg.diffusion_unet_inference
    target_dim = tuple(int(d) for d in inf_cfg.dim)
    divisor = autoencoder_divisor(cfg)
    brats_dir = str(cfg.data_base_dir)
    manifest = build_brats_pair_manifest(brats_dir)
    if not manifest:
        raise FileNotFoundError(
            f"No paired BraTS volumes found under data_base_dir={brats_dir} "
            f"(need >=1 subject with all 4 contrasts)."
        )
    # _train_val_manifests reads ROOT cfg.val_fraction, but the G2RPO recipe defines the
    # held-out fraction under paired_grpo.val_fraction. Mirror the nested value to root
    # when the native-split DIRECTORY path is not taken (val_data_base_dir unset / not a
    # dir), else the val split resolves to 0 -> empty val -> the guard below raises
    # (codex #109 P2; mirrors paired_reward_cli._real_inputs codex #99/#100). Mirror ONLY
    # when the root key is absent: an explicit root override (CLI dotlist / a profile)
    # wins.
    val_dir = opt(cfg, "val_data_base_dir", None)
    if not (val_dir and os.path.isdir(str(val_dir))):
        from omegaconf import OmegaConf

        if OmegaConf.select(cfg, "val_fraction", default=None) is None:
            g2rpo_val_fraction = float(opt(cfg, "paired_grpo.val_fraction", 0.0))
            cfg = OmegaConf.merge(cfg, OmegaConf.create({"val_fraction": g2rpo_val_fraction}))

    train_manifest, val_manifest = _train_val_manifests(cfg, manifest)
    if not val_manifest:
        raise ValueError(
            "G2RPO needs a held-out val split (val_data_base_dir set, or val_fraction>0) "
            "for the PSNR selection metric; train data is never reused as val (ADR-0022)."
        )
    cache_dir = str(
        latents_dir
        or opt(cfg, "latent_cache_dir", os.path.join(str(cfg.model_dir), "paired_latent_cache"))
    )
    cache_tag = str(opt(cfg, "paired_grpo.cache_tag", "paired_train"))

    def _warm_ds(manifest_split):
        vol_ds = PairedNiftiVolumeDataset(manifest_split, target_dim=target_dim, divisor=divisor)
        ds = PairedLatentDataset(vol_ds, encode_fn=None, cache_dir=cache_dir, cache_tag=cache_tag)
        ds.warm_cache(device, logger=_log, show_progress=False)
        ds.scaling_factor = scaling_factor  # scale-on-read (ADR-0021: reuse verbatim)
        return ds

    train_latent_ds = _warm_ds(train_manifest)
    val_latent_ds = _warm_ds(val_manifest)

    # The paired_train cache is keyed by sample_id + cache_tag (NOT target_dim), so a
    # target_dim mismatch silently reuses stale wrong-shape latents. Validate EVERY
    # entry's spatial shape (both splits) against the config's target_dim / divisor and
    # fail fast (ceil division: PairedNiftiVolumeDataset zero-pads each spatial dim up
    # to a multiple of the divisor before encoding). Test fakes bypass this (no
    # raw_latent/source) - they don't model the encode (codex #109 P2; mirrors
    # paired_reward_cli codex #99/#100).
    if hasattr(train_latent_ds, "raw_latent") and hasattr(train_latent_ds, "source"):
        expected_spatial = tuple(-(-d // divisor) for d in target_dim)  # ceil(d / divisor)
        for split_name, ds in (("train", train_latent_ds), ("val", val_latent_ds)):
            for sid in ds.source.unique_sample_ids():
                cached_spatial = tuple(ds.raw_latent(sid).shape[1:])
                if cached_spatial != expected_spatial:
                    raise ValueError(
                        f"Cached paired latent ({split_name}, {sid}) spatial shape "
                        f"{cached_spatial} does not match the G2RPO config's "
                        f"target_dim={target_dim} / divisor={divisor} = {expected_spatial} "
                        f"(ceil). The paired_train cache was built with a different "
                        f"target_dim (or is a mixed/partial cache); point --latents-dir at "
                        f"a matching cache or re-warm it."
                    )

    # 5. Train is pure-RL (src-only: the bridge pins Z_1 -> x_hat_1; tgt unused at
    #    train); val carries tgt for the PSNR decode (#105). Both share the paired cache.
    class _TrainCondDS(torch.utils.data.Dataset):
        """Source-latent + direction only (G2RPO is pure-RL - tgt volume unused at train)."""

        def __init__(self, ds):
            self.ds = ds

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            it = self.ds[i]
            return {
                "src_latent": it["src_latent"],
                "src_label": it["src_label"],
                "tgt_label": it["tgt_label"],
                "spacing": it["spacing"],
            }

    _log.info(
        "G2RPO real inputs: %d train (src-only) / %d val (src+tgt) paired latents.",
        len(train_latent_ds), len(val_latent_ds),
    )
    return PairedGRPOInputs(
        policy=policy,
        reward_model=reward_model,
        scheduler=scheduler,
        train_ds=_TrainCondDS(train_latent_ds),
        val_ds=val_latent_ds,  # full (src+tgt) for the PSNR callback
        reference_policy=reference_policy,
        vae=vae,
    )

__all__ = [
    "EtaRampCallback",
    "GuardedModelCheckpoint",
    "PairedGRPOInputs",
    "calibrate_reward_temp",
    "main",
    "run_paired_grpo_measurement",
    "run_paired_grpo_training",
]
