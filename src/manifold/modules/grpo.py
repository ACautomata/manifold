"""The GRPO singular-branch rollout + objective pieces (ADR-0011 / ADR-0012).

One deterministic two-eval-Heun **anchor** per group (shared across the ``G``
siblings — same initial noise ⇒ identical trajectory); for each perturbed step,
exactly one stochastic SDE step branched off the anchor at ``z_k``; a deterministic
Heun **suffix** to the terminal latent ``z_K`` the frozen reward scores. The single
SDE step is the x-pred equimarginal reverse-time SDE of the JiT transport
(:meth:`~manifold.FlowMatchGRPOScheduler.sde_step_mean`); the caller-side log-prob
and group-normalized advantage live here.

Module-owned, scheduler-delegated (ADR-0005): the anchor grid
(:meth:`~manifold.FlowMatchGRPOScheduler.set_timesteps`, inherited) and the Heun
steps (:meth:`~manifold.FlowMatchHeunDiscreteScheduler.euler_step` /
:meth:`heun_correct`, inherited verbatim) are the scheduler's; only the loop +
the SDE draw + the log-prob/advantage live here. The rollout is **fully
``no_grad``** — the anchor + suffix + reward must NOT retain a graph (the property
that keeps 3D feasible; the policy's single grad eval lives in the
:class:`~manifold.modules.GRPOModule` inner loop, which re-evaluates the UNet at
the stored ``z_k`` under grad).

The reward scores the terminal **latent** — no VAE decode per branch (the cost
saver vs an image-space reward). The anchor + suffix both run under ``no_grad``
(not ``inference_mode``): identical math, but a ``no_grad`` tensor stays out of
autograd cleanly.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import stable_pretraining as spt
import torch
from torch import Tensor

from ..schedulers.scheduling_flow_match_grpo import FlowMatchGRPOScheduler
from .sampler import sample_latent_flow

#: Per-step rollout buffer entry consumed by the GRPO inner loop.
#:
#: ``z_k`` (B, *spatial): the anchor node the SDE step branched off (the UNet is
#: re-evaluated here under grad in the inner loop). ``z_kplus1`` (B, G, *spatial):
#: the G sibling SDE draws. ``old_log_prob`` / ``advantage`` (B, G): the transition
#: log-density at rollout time and the group-normalized advantage. ``t_k`` /
#: ``t_next``: the scalar grid times (the grid is batch-wide). ``std`` is NOT
#: stored — it depends only on ``t`` ⇒ recomputable via ``sde_step_mean``.
RolloutStep = dict[str, Tensor | float]


def gaussian_log_prob(z_next: Tensor, mean: Tensor, std) -> Tensor:
    """``log N(z_next; mean, std²)`` mean-reduced over non-batch dims → ``(B, G)``.

    The transition policy's log-density (the token-analog reduction: the latent
    spatial elements are the "tokens", so the per-element log-density is mean-reduced
    over ``(C, D, H, W)``, keeping the ``(B, G)`` batch/group dims). ``mean`` is the
    ``(B, *spatial)`` scheduler ``sde_step_mean`` output; ``z_next`` is ``(B, G,
    *spatial)``; ``std`` is the scalar-``t`` float from ``sde_step_mean`` (the GRPO
    anchor grid is batch-wide, so ``t`` is a scalar).
    """
    diff = z_next - mean.unsqueeze(1)  # (B, G, *spatial)
    # η > 0 in practice (the exploration knob; std is O(0.1–1)); the 1e-12 floor only
    # bounds the degenerate η = 0 Gaussian (a delta — used by the anchor-parity test /
    # debug), where z_{k+1} = mean exactly regardless, so the floor never changes a
    # real rollout's log-prob.
    var = max(float(std) ** 2, 1e-12)
    log_elem = -0.5 * (diff.pow(2) / var + math.log(2.0 * math.pi * var))  # (B, G, *spatial)
    return log_elem.flatten(start_dim=2).mean(dim=2)  # (B, G)


def group_advantage(rewards: Tensor, adv_clip_max: float = 5.0, eps: float = 1e-8) -> Tensor:
    """Group-normalized advantage ``A = (R − mean R)/(std R + ε)`` over the ``G`` siblings.

    Reduces the policy-gradient variance regardless of reward scale; clipped to
    ``±adv_clip_max`` so a single spoofed-reward outlier cannot dominate a step.
    ``rewards`` is ``(B, G)``; the normalization is over ``G`` (each conditioning's
    own siblings), returning ``(B, G)``.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    adv = (rewards - mean) / (std + eps)
    return adv.clamp(min=-adv_clip_max, max=adv_clip_max)


def _heun_one_step(
    unet, scheduler, z: Tensor, t: float, t_next: float, spacing_t: Tensor, class_labels: Tensor
) -> Tensor:
    """One deterministic two-eval-Heun reverse step (the deployed sampler's step).

    Final-step Euler when ``t_next == 1`` (the ``1/(1 − t_next)`` corrector
    denominator diverges) — the same convention as
    :func:`~manifold.modules.sample_latent_flow` / :func:`partial_denoise_rollout`.
    """
    x0_1 = unet(sample=z, timestep=t, spacing=spacing_t, class_labels=class_labels)
    z_euler, v1 = scheduler.euler_step(x0_1, z, t, t_next)
    if float(t_next) >= 1.0:
        return z_euler
    x0_2 = unet(sample=z_euler, timestep=t_next, spacing=spacing_t, class_labels=class_labels)
    return scheduler.heun_correct(x0_2, z, z_euler, v1, t, t_next)


def _heun_rollout(
    unet, scheduler, z_start: Tensor, nodes: Tensor, spacing_t: Tensor, class_labels: Tensor,
    start_i: int, end_i: int,
) -> list[Tensor]:
    """Deterministic Heun from node ``start_i`` to ``end_i``; returns ``[z_start_i, …, z_end_i]``.

    Used for both the shared anchor (``start_i=0``) and each branch's suffix
    (``start_i=k+1``). ``nodes`` is the scheduler's batch-wide grid
    (:meth:`~manifold.FlowMatchGRPOScheduler.set_timesteps`); the steps are the
    scheduler's inherited ``euler_step`` / ``heun_correct`` (never reimplemented).
    """
    zs = [z_start]
    z = z_start
    for i in range(start_i, end_i):
        z = _heun_one_step(unet, scheduler, z, float(nodes[i]), float(nodes[i + 1]), spacing_t, class_labels)
        zs.append(z)
    return zs


@torch.no_grad()
def singular_branch_rollout(
    unet,
    scheduler: FlowMatchGRPOScheduler,
    reward_model,
    noise: Tensor,
    spacing: Tensor | Sequence[float],
    modality: int | Tensor,
    *,
    G: int,
    eta_step_list: Sequence[int],
    num_steps: int,
    adv_clip_max: float = 5.0,
    reward_transform: Callable[[Tensor], Tensor] | None = None,
) -> list[RolloutStep]:
    """One Granular-GRPO singular-branch rollout (no_grad) → per-step buffer.

    The shared anchor (``z_0 → z_k`` for each perturbed ``k``) runs once on the
    ``(B,)`` batch — identical across the ``G`` siblings (same noise ⇒ same
    trajectory). At each perturbed step the ``G`` siblings branch off ``z_k`` via
    one SDE draw (``sde_step_mean``), roll a deterministic Heun suffix to the
    terminal ``z_K``, and are scored by the frozen reward; the per-group advantage
    is normalized over ``G``. Returns the buffer the
    :class:`~manifold.modules.GRPOModule` inner loop consumes.

    Args:
        unet: the policy UNet (the JiT x0-denoiser; run eval + no_grad).
        scheduler: a :class:`FlowMatchGRPOScheduler` (its grid + inherited Heun +
            ``sde_step_mean`` run; ``set_timesteps`` is called here).
        reward_model: the frozen :class:`~manifold.RewardModel` scoring ``z_K``.
        noise: the pure-noise group start ``(B, C, D, H, W)`` — shared across siblings.
        spacing / modality: the manifold conditioning (voxel spacing + class label).
        G: the group size (siblings per conditioning).
        eta_step_list: the perturbed step indices ``k`` (the noisy-half, e.g.
            ``[0..7]`` of ``num_steps=15``).
        num_steps: the anchor grid resolution (Heun steps over ``t: 0 → 1``).
        adv_clip_max: the advantage-magnitude clip.
        reward_transform: optional monotone bound applied to the raw rewards before the
            group normalization (the v2 tanh cap, ADR-0015); ``None`` ⇒ raw logit (v1).

    Returns:
        One :data:`RolloutStep` per ``k`` in ``eta_step_list`` (sorted ascending).
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    B = noise.shape[0]
    spatial = noise.shape[1:]  # (C, D, H, W)

    spacing_t = torch.as_tensor(spacing, device=device)
    if isinstance(modality, Tensor):
        class_labels = modality.to(device=device, dtype=torch.long)
    else:
        class_labels = torch.full((B,), int(modality), dtype=torch.long, device=device)

    nodes = scheduler.set_timesteps(num_steps, device=device)  # (num_steps+1,)
    eta_steps = sorted(int(k) for k in eta_step_list)
    if eta_steps[-1] >= num_steps:
        raise ValueError(
            f"eta_step_list max ({eta_steps[-1]}) must be < num_steps ({num_steps}) — "
            "a perturbed step k advances node k → k+1 and needs a suffix node."
        )
    max_k = eta_steps[-1]

    unet.eval()
    z0 = noise.to(device=device, dtype=dtype)
    # Suffix conditioning: each conditioning's G siblings share its spacing/label,
    # so the (B,) conditioning is repeat-interleaved G times to the (B·G,) suffix
    # batch (matching z_kplus1.reshape(B·G, ...) — flat index b·G+g is sibling g of b).
    class_labels_bg = class_labels.repeat_interleave(G)  # (B·G,)
    if spacing_t.dim() == 2:  # per-sample (B, 3) spacing
        spacing_bg = spacing_t.repeat_interleave(G, dim=0)
    else:  # broadcast (3,) spacing — fine for any batch
        spacing_bg = spacing_t

    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        # Shared anchor: z at nodes [0..max_k] (anchor_z[k] = latent at node k).
        anchor_z = _heun_rollout(unet, scheduler, z0, nodes, spacing_t, class_labels, 0, max_k)

        buffer: list[RolloutStep] = []
        for k in eta_steps:
            t_k = float(nodes[k])
            t_next = float(nodes[k + 1])
            z_k = anchor_z[k]  # (B, *spatial) — the anchor node

            # SDE transition params at the current (rollout-time) policy.
            x0 = unet(sample=z_k, timestep=t_k, spacing=spacing_t, class_labels=class_labels)
            mean_old, std_old = scheduler.sde_step_mean(x0, z_k, t_k, t_next)
            # G siblings branch off z_k via one SDE draw each (the only per-sibling
            # difference; the anchor + suffix are otherwise deterministic given z_{k+1}).
            xi = torch.randn(B, G, *spatial, device=device, dtype=dtype)
            z_kplus1 = mean_old.unsqueeze(1) + float(std_old) * xi  # (B, G, *spatial)
            old_log_prob = gaussian_log_prob(z_kplus1, mean_old, std_old)  # (B, G)

            # Deterministic Heun suffix from z_{k+1} (node k+1) to the terminal z_K.
            z_g = z_kplus1.reshape(B * G, *spatial)
            suffix = _heun_rollout(
                unet, scheduler, z_g, nodes, spacing_bg, class_labels_bg, k + 1, num_steps
            )
            z_K = suffix[-1]  # (B·G, *spatial)

            # float(): under cuda autocast the PatchGAN emits fp16 rewards; the
            # group-normalization (std / (r−mean)) must run in fp32 to avoid the
            # fp16 square overflowing for large rewards (mirrors validation_step).
            rewards = reward_model(z_K).float().reshape(B, G)  # (B, G)
            if reward_transform is not None:  # the v2 bound (ADR-0015) — caps the raw logit
                rewards = reward_transform(rewards)
            advantage = group_advantage(rewards, adv_clip_max=adv_clip_max)  # (B, G)
            buffer.append({
                "z_k": z_k.detach(),
                "t_k": t_k,
                "t_next": t_next,
                "z_kplus1": z_kplus1.detach(),
                "old_log_prob": old_log_prob.detach(),
                "advantage": advantage.detach(),
                "rewards": rewards.detach(),
            })
    return buffer


def clipped_surrogate_loss(
    new_log_prob: Tensor, old_log_prob: Tensor, advantage: Tensor, clip_range: float
) -> Tensor:
    """The PPO clipped-surrogate loss (mean over the ``(B, G)`` group×step batch).

    ``ratio r = exp(new − old)``; ``L = −mean(min(r·A, clip(r, 1−ε, 1+ε)·A))``.
    Minimizing ``L`` raises the transition log-prob for positive-advantage siblings
    and lowers it for negative ones (descent toward the higher-reward transition);
    the clip freezes the gradient once ``r`` drifts past ``1 ± ε`` (the trust region).
    ``ε = clip_range`` (1e-4 v1) — tight, so each update is microscopic and stable.
    """
    ratio = torch.exp(new_log_prob - old_log_prob)
    eps = float(clip_range)
    clipped_ratio = ratio.clamp(min=1.0 - eps, max=1.0 + eps)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    return -surrogate.mean()


#: A GRPO training/validation batch: manifold conditioning only (GRPO is generative —
#: the latent comes from the rollout, not the batch). ``spacing``: ``[3]`` or
#: ``[B, 3]`` voxel spacing; ``label``: integer class label (scalar or ``(B,)``).
GRPOBatch = dict[str, Any]


class GRPOModule(spt.Module):
    """Granular-GRPO policy post-training of the JiT x0-denoiser (ADR-0011/0012).

    Overrides :meth:`training_step` (NOT ``forward`` — GRPO is multi-term, multi-step,
    so the single-loss seam the :class:`RewardModule` uses cannot hold): a no-grad
    :func:`singular_branch_rollout` fills the buffer; then a **multi-step PPO inner
    loop** over ``eta_step_list`` recomputes ``new_log_prob`` under grad (the policy's
    one live grad eval per branch), builds the :func:`clipped_surrogate_loss`, and runs
    one ``opt.step`` per step so the ratio drifts off 1 and the clip **binds** (a real
    trust region — a single aggregated step would collapse it to REINFORCE).

    Holds the **trainable JiT UNet** (the policy — the only params
    :meth:`configure_optimizers` optimizes) and the **frozen** :class:`~manifold.RewardModel`
    (unregistered via ``object.__setattr__``, like the reward Module holds its denoiser).
    No EMA; resumes / selects / exports the raw arm (ADR-0006/0012).

    Args:
        policy: the trainable JiT x0-denoiser UNet (the policy).
        reward_model: the frozen :class:`~manifold.RewardModel` scoring the terminal latent.
        scheduler: a stateless :class:`FlowMatchGRPOScheduler`.
        G: the group size (siblings per conditioning).
        eta_step_list: the perturbed step indices (the noisy-half, e.g. ``[0..7]``).
        clip_range: ε for the clipped surrogate (the tight PPO trust region).
        lr: the Adam LR over the policy UNet.
        adv_clip_max: the advantage-magnitude clip.
        num_steps: the anchor grid resolution (train rollout + validation Heun steps).
        latent_shape: the rollout latent shape ``(C, D, H, W)`` (GRPO is generative —
            the Module samples the group noise of this shape).
        reference_policy: optional frozen deepcopy of the pretrained JiT UNet — the KL
            anchor (ADR-0015). ``None`` ⇒ no KL (the v1 default).
        kl_coef: β for the KL-to-reference penalty; ``≤ 0`` disables it (v1 default 0.0).
        reward_bound: ``"none"`` (raw logit, v1) or ``"tanh"`` (soft-clip, ADR-0015).
        reward_temp: the tanh temperature (≈ the real-data reward std, ~8 from calibration).
    """

    def __init__(
        self,
        policy,
        reward_model,
        scheduler: FlowMatchGRPOScheduler,
        *,
        G: int = 8,
        eta_step_list: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7),
        clip_range: float = 1e-4,
        lr: float = 1e-6,
        adv_clip_max: float = 5.0,
        num_steps: int = 15,
        latent_shape: Sequence[int] = (4, 64, 64, 32),
        reference_policy: Any = None,
        kl_coef: float = 0.0,
        reward_bound: str = "none",
        reward_temp: float = 8.0,
    ):
        if G < 2:
            # group_advantage normalizes over the G siblings via torch.std (Bessel,
            # needs ≥2 samples); G=1 ⇒ std=NaN ⇒ NaN advantage ⇒ NaN grads destroy
            # the policy in one step. GRPO needs ≥2 siblings by definition.
            raise ValueError(f"G must be >= 2 (need >= 2 siblings per group), got {G}.")
        # NOTE: forward is NOT passed to spt.Module — training_step is overridden
        # directly (the multi-step inner loop cannot fit the single-loss seam).
        super().__init__(hparams={"lr": lr, "G": G, "clip_range": clip_range,
                                  "num_steps": num_steps, "adv_clip_max": adv_clip_max})
        self.unet = policy  # registered → the only optimized / checkpointed arm
        self.scheduler = scheduler  # carries η (the SDE exploration knob)
        # Frozen reward, held UNregistered (object.__setattr__ bypasses nn.Module
        # registration) → absent from parameters()/state_dict()/optimizer/DDP, moved
        # to the device manually in on_fit_start (mirrors RewardModule's denoiser).
        reward_model = reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "reward_model", reward_model)

        # Frozen reference policy (the KL anchor, ADR-0015): a deepcopy of the pretrained
        # JiT UNet, held UNregistered exactly like the reward → off parameters()/
        # state_dict()/optimizer/DDP, moved to the device in on_fit_start. ``None`` ⇒ no
        # KL term (the v1 default; backward-compat). ``kl_coef ≤ 0`` also disables it.
        if reference_policy is not None:
            reference_policy = reference_policy.eval()
            for p in reference_policy.parameters():
                p.requires_grad_(False)
            object.__setattr__(self, "reference_unet", reference_policy)
        else:
            object.__setattr__(self, "reference_unet", None)

        self.G = int(G)
        self.eta_step_list = tuple(int(k) for k in eta_step_list)
        self.clip_range = float(clip_range)
        self.lr = float(lr)
        self.adv_clip_max = float(adv_clip_max)
        self.num_steps = int(num_steps)
        self.latent_shape = tuple(int(s) for s in latent_shape)
        self.kl_coef = float(kl_coef)
        self.reward_bound = str(reward_bound)
        self.reward_temp = float(reward_temp)
        self._val_reward_sum = 0.0
        self._val_reward_count = 0

    def on_validation_epoch_start(self) -> None:
        self._val_reward_sum = 0.0
        self._val_reward_count = 0

    def on_validation_epoch_end(self) -> None:
        agg = torch.tensor(
            [self._val_reward_sum, float(self._val_reward_count)],
            device=self.device, dtype=torch.float32,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(agg, op=torch.distributed.ReduceOp.SUM)
        total, count = float(agg[0]), int(agg[1])
        if count:
            self.log("val/mean_reward", total / count, prog_bar=True, sync_dist=False)

    # -- frozen-reward lifecycle ---------------------------------------------

    def on_fit_start(self) -> None:
        """Move the unregistered frozen reward (and reference policy) to the device.

        The ``object.__setattr__`` bypass keeps these off Lightning's books, so its
        automatic ``.to(device)`` skips them. The real path moves them already; this
        is the safety net for direct ``fit`` calls.
        """
        self.reward_model.to(self.device)
        if self.reference_unet is not None:
            self.reference_unet.to(self.device)

    # -- optimizer ------------------------------------------------------------

    def configure_optimizers(self):
        """Adam over the policy UNet only (the frozen reward is unregistered)."""
        return {"optimizer": torch.optim.Adam(self.unet.parameters(), lr=self.lr)}

    # -- training_step: rollout + multi-step PPO inner loop -------------------

    def _policy_dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def _conditioning_tensors(self, batch: GRPOBatch) -> tuple[Tensor, Tensor, int]:
        """Build ``(spacing_t, class_labels, B)`` for the batch's manifold conditioning."""
        spacing_t = torch.as_tensor(batch["spacing"], device=self.device)
        label = batch["label"]
        if isinstance(label, Tensor):
            class_labels = label.to(device=self.device, dtype=torch.long)
        else:
            class_labels = torch.full((1,), int(label), dtype=torch.long, device=self.device)
        return spacing_t, class_labels, int(class_labels.shape[0])

    def _new_log_prob(
        self, step: RolloutStep, spacing_t: Tensor, class_labels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | float]:
        """Recompute the transition log-prob under grad (the inner-loop grad eval).

        Re-evaluates the UNet at the stored anchor node ``z_k`` (the single live grad
        eval per branch — peak autograd memory is one UNet-forward) and forms the
        Gaussian transition density at the stored ``z_{k+1}``. ``old_log_prob`` was
        computed at rollout time; the ratio ``exp(new − old)`` is what the clip bounds.
        Returns ``(log_prob, mean_new, std_new)`` so the caller can reuse ``mean_new``
        for the KL term (no second policy forward) — ADR-0015.
        """
        z_k = step["z_k"]  # (B, *spatial) — detached anchor node
        x0 = self.unet(sample=z_k, timestep=step["t_k"], spacing=spacing_t, class_labels=class_labels)
        mean_new, std_new = self.scheduler.sde_step_mean(x0, z_k, step["t_k"], step["t_next"])
        log_prob = gaussian_log_prob(step["z_kplus1"], mean_new, std_new)  # (B, G)
        return log_prob, mean_new, std_new

    def _transition_kl(
        self, step: RolloutStep, mean_new: Tensor, std_new, spacing_t: Tensor, class_labels: Tensor
    ) -> Tensor | None:
        """The per-transition KL anchor ``0.5·‖μ_θ − μ_ref‖²/σ²`` (ADR-0015), or ``None``.

        The GRPO transition ``π_θ(z_{k+1}|z_k) = N(μ_θ, σ²_t·Δt·I)`` and the reference
        transition share **equal variance** (``σ_t = η·sqrt((1−t)/t)`` depends only on
        ``t``, not on the policy weights), so the diagonal-Gaussian KL collapses to the
        squared-mean difference over ``σ²`` (the trace / log-det terms cancel). The
        reference mean is a frozen, no-grad forward at the stored ``z_k``; grad flows
        through ``μ_θ`` only. Returns ``None`` (no KL added) when the reference is absent
        or ``kl_coef ≤ 0`` — the v1 backward-compat default. ``mean_new`` / ``std_new``
        are the caller's grad-bearing policy outputs (reused, not recomputed).
        """
        if self.reference_unet is None or self.kl_coef <= 0.0:
            return None
        with torch.no_grad():
            x0_ref = self.reference_unet(
                sample=step["z_k"], timestep=step["t_k"], spacing=spacing_t, class_labels=class_labels
            )
        mean_ref, _ = self.scheduler.sde_step_mean(x0_ref, step["z_k"], step["t_k"], step["t_next"])
        var = max(float(std_new) ** 2, 1e-12)  # the η=0 degenerate floor (mirrors gaussian_log_prob)
        return 0.5 * ((mean_new - mean_ref) ** 2 / var).flatten(start_dim=1).mean(dim=1)  # (B,)

    def _bound_reward(self, r: Tensor) -> Tensor:
        """Bound the raw PatchGAN reward so OOD latents cannot dominate (ADR-0015).

        ``none`` is the identity (the v1 raw logit). ``tanh`` maps ``tanh(r / reward_temp)``
        into the open interval (−1, 1) — a MONOTONIC soft-clip, so distinct sibling rewards
        stay distinct (the group signal survives) while an OOD extreme (the v1 raw 3370)
        saturates near ±1 instead of dominating the advantage. ``reward_temp`` ≈ the
        real-data reward std (~8 from calibration) spreads the in-distribution range.
        """
        if self.reward_bound == "none":
            return r
        if self.reward_bound == "tanh":
            return torch.tanh(r / self.reward_temp)
        raise ValueError(f"unknown reward_bound {self.reward_bound!r} (expected 'none' or 'tanh')")

    def training_step(self, batch: GRPOBatch, batch_idx: int):
        """One GRPO update: no-grad singular-branch rollout → multi-step PPO inner loop.

        The rollout (anchor + G SDE branches + Heun suffix + reward + group advantage)
        runs under ``no_grad`` and fills the buffer; the inner loop then runs one PPO
        update per perturbed step — recompute ``new_log_prob`` under grad, clipped
        surrogate, ``manual_backward``, ``opt.step``, ``zero_grad`` — so the ratio
        drifts off 1 from the second step on and the clip binds (ADR-0012). The trust
        region is the ratio clip + the tiny LR (no grad-clip; Trainer
        ``accumulate_grad_batches`` is assumed 1 — the per-step inner loop is
        load-bearing for the clip, so accumulation > 1 would break it).
        """
        if not isinstance(batch, dict):
            raise ValueError(f"batch is expected to be a dict, not {type(batch)}")
        batch["batch_idx"] = batch_idx
        spacing_t, class_labels, B = self._conditioning_tensors(batch)
        # One shared group noise (identical across the G siblings ⇒ a shared anchor).
        noise = torch.randn(B, *self.latent_shape, device=self.device, dtype=self._policy_dtype())
        # The v2 reward bound (ADR-0015): None for the v1 raw-logit default (no-op),
        # else ``_bound_reward`` (tanh) caps the reward the rollout scores + groups.
        reward_transform = self._bound_reward if self.reward_bound != "none" else None
        buffer = singular_branch_rollout(
            self.unet, self.scheduler, self.reward_model, noise, spacing_t, class_labels,
            G=self.G, eta_step_list=self.eta_step_list, num_steps=self.num_steps,
            adv_clip_max=self.adv_clip_max, reward_transform=reward_transform,
        )

        opt = self.optimizers()
        sched = self.lr_schedulers()
        if not isinstance(sched, (list, tuple)) and sched is not None:
            sched = [sched]
        # Eval mode for every policy eval (rollout + inner loop) — matches the deployed
        # sampler (deterministic; GroupNorm running stats, no dropout).
        self.unet.eval()
        losses = []
        for step in buffer:
            new_lp, mean_new, std_new = self._new_log_prob(step, spacing_t, class_labels)
            loss = clipped_surrogate_loss(
                new_lp, step["old_log_prob"], step["advantage"], self.clip_range
            )
            # The v2 KL anchor (ADR-0015): a separate term outside the PPO ratio clip —
            # ``None`` (kl_coef=0 / no reference) adds nothing, preserving v1 behavior.
            kl = self._transition_kl(step, mean_new, std_new, spacing_t, class_labels)
            if kl is not None:
                loss = loss + self.kl_coef * kl.mean()
            self.manual_backward(loss)
            opt.step()
            if sched is not None:
                for s in sched:
                    s.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.detach())
        mean_loss = torch.stack(losses).mean() if losses else torch.zeros((), device=self.device)
        # ``sync_dist=True`` reduces the per-rank epoch mean to the global mean across
        # ranks (issue #82 / L5). Unlike the metrics callbacks (M6), ``train/loss`` is
        # already logged WITH ``batch_size=B`` so Lightning's epoch aggregate is the
        # sample-weighted mean; sync_dist just adds the cross-rank reduce. Exact even on
        # unbalanced shards because of the weight.
        self.log("train/loss", mean_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        return {"loss": mean_loss}

    # -- generation (the FID callback drives this; ADR-0005) -------------------

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
        """Generate a latent ``[B, C, D, H, W]`` from pure noise via the deployed Heun.

        The FID callback (#58) generates through the Module — never the inference
        Pipeline — delegating to the shared :func:`~manifold.modules.sample_latent_flow`
        primitive. Generation uses the deployed two-eval Heun (NOT the rollout SDE)
        so ``val/fid`` measures the distribution JiT ships. No EMA swap: GRPO
        evaluates the **raw** policy (#59 — the double-EMA callback is not attached).

        Args mirror :meth:`~manifold.modules.LatentFlowModule.sample`: same generator
        + shape produces a bit-identical latent (parity with the JiT path).
        """
        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype
        noise = torch.randn(target_shape, generator=generator, device=device, dtype=dtype)
        return sample_latent_flow(
            self.unet, self.scheduler, noise, spacing, modality,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale, cfg_interval=cfg_interval,
        )

    # -- validation: deployed-Heun generation + reward (val/mean_reward) ------

    def validation_step(self, batch: GRPOBatch, batch_idx: int):
        """Generate from noise via the deployed Heun sampler, score -> ``val/mean_reward``.

        The RL progress signal: generate via the deployed two-eval Heun (NOT the
        rollout SDE) so the reward reflects the distribution JiT ships, then score
        with the frozen PatchGAN. The anti-reward-hacking selection metric
        (``val/fid``, #58) is a SEPARATE generation pass driven by the FID callback
        (:meth:`sample`). ``sample_latent_flow`` takes a scalar modality, so a
        mixed-label val batch is generated under ``label[0]`` (single-modality v1).

        **All-rank under DDP** (ADR-0025): every rank generates + scores its own
        padded ``DistributedSampler`` shard and accumulates only non-padding reward
        sums/counts. Epoch end all-reduces ``(sum, count)`` for the exact global mean;
        padded forwards keep rank symmetry but contribute zero metric weight. The prior
        rank-0-only gate (PR #115) is removed; the rank-asymmetric early-return is gone,
        so no rank blocks.
        """
        batch["batch_idx"] = batch_idx
        spacing_t, class_labels, B = self._conditioning_tensors(batch)
        # Rank/epoch-strided validation noise (codex #116 P2): under DDP the ranks run
        # identical RNG-consuming training steps, so a plain torch.randn produces the
        # SAME noise on every rank - scoring duplicate generations rather than a
        # rank-wise union. Offset the generator seed by rank + batch so each rank's
        # val shard is a distinct draw (the synced val/mean_reward then reflects the
        # full val set, not world x one shard).
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        gen = torch.Generator(device=self.device)
        gen.manual_seed(1234 + 1000 * rank + batch_idx)
        noise = torch.randn(B, *self.latent_shape, generator=gen, device=self.device, dtype=self._policy_dtype())
        z_K = sample_latent_flow(
            self.unet, self.scheduler, noise, spacing_t, int(class_labels[0].item()),
            num_inference_steps=self.num_steps,
        )
        # The same bound the rollout applies (ADR-0015) — val/mean_reward is reported on
        # the bounded scale so it tracks the training signal (raw logit otherwise).
        rewards = self._bound_reward(self.reward_model(z_K).float())
        valid = ~batch.get(
            "_is_padding", torch.zeros(B, dtype=torch.bool, device=rewards.device)
        ).to(rewards.device).bool()
        self._val_reward_sum += float(rewards[valid].sum())
        self._val_reward_count += int(valid.sum())
        return {"mean_reward": rewards[valid].mean() if bool(valid.any()) else rewards.new_zeros(())}
