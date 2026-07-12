"""The G2RPO paired singular-branch rollout + PairedGRPOModule (ADR-0024).

A **fork** of :mod:`manifold.modules.grpo` for the Paired JiT UNet signature.
Granular-GRPO (singular branching, arXiv 2510.01982) post-training of the Paired
JiT UNet against the frozen paired reward, over a **data-to-data Brownian bridge**
``Z_t = (1−t)·X_src + t·X_tgt + √(ηt(1−t))·ε`` (ADR-0024). The bridge makes the
deterministic Paired JiT transport stochastic (GRPO needs a Gaussian transition
density) while keeping **both endpoints as data** (variance vanishes at both).

The ``grpo.py`` objective spine (``gaussian_log_prob``, ``group_advantage``,
``clipped_surrogate_loss``, the multi-step PPO inner loop) **reuses verbatim** — the
bridge ``σ`` is θ-independent, so the equal-variance KL closed form holds. The fork
is the **transport signature**: the paired UNet sees ``concat([z_t, x_src])`` with
the summed contrast embedding ``embed(src)+embed(tgt)`` at every evaluation
(:func:`~manifold.modules.sample_paired_latent_flow`'s ``unet_call`` closure is the
template), and the rollout **starts from the source latent** ``x_src`` (the ``t = 0``
endpoint is a data latent, not Gaussian noise — the bridge is data→data).

The fork threads ``cat([z, x_src]) + embed(src)+embed(tgt)`` through **five** UNet
eval sites (ADR-0024): the anchor, the perturbed-step rollout eval, the suffix, the
inner-loop ``_new_log_prob`` grad re-eval, and the ``_transition_kl`` reference
forward. The reward scores ``reward_model(cat([x_src_bg, z_K]))`` where
``x_src_bg = x_src.repeat_interleave(G, dim=0)`` — ``x_src`` MUST be G-expanded
BEFORE the concat (D9: a wrong G-expansion order pairs each sibling's ``z_K`` with
the wrong source → corrupted advantage, **no error**).

The rollout is fully ``no_grad`` (the anchor + suffix + reward must NOT retain a
graph — the 3D-feasibility invariant); the policy's single grad eval per branch
lives in :class:`PairedGRPOModule`'s inner loop.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import stable_pretraining as spt
import torch
from torch import Tensor

from ..schedulers.scheduling_flow_match_bridge_grpo import FlowMatchBridgeGRPOScheduler
from .grpo import (
    RolloutStep,
    clipped_surrogate_loss,
    gaussian_log_prob,
    group_advantage,
)
from .paired_sampler import _as_label_tensor, sample_paired_latent_flow


def _paired_unet_call(
    unet, z: Tensor, x_src: Tensor, t, spacing_t: Tensor, src_labels: Tensor, tgt_labels: Tensor
) -> Tensor:
    """One paired UNet eval at flow-time ``t`` on ``concat([z, x_src])`` (ADR-0014).

    Mirrors :func:`~manifold.modules.sample_paired_latent_flow`'s ``unet_call``
    closure: the Paired JiT UNet (``in_channels = 2·C_latent``) sees the moving
    latent ``z`` concat with the (shared) source latent, conditioned on the summed
    contrast embedding ``embed(src)+embed(tgt)`` the wrapper injects. This is the
    single building block reused at every one of the five eval sites.
    """
    sample = torch.cat([z, x_src], dim=1)
    return unet(
        sample=sample,
        timestep=t,
        spacing=spacing_t,
        class_labels_src=src_labels,
        class_labels_tgt=tgt_labels,
    )


def _heun_one_step_paired(
    unet, scheduler, z: Tensor, x_src: Tensor, t: float, t_next: float,
    spacing_t: Tensor, src_labels: Tensor, tgt_labels: Tensor,
) -> Tensor:
    """One deterministic two-eval-Heun reverse step on ``concat([z, x_src])``.

    Final-step Euler when ``t_next == 1`` (the ``1/(1 − t_next)`` corrector
    denominator diverges) — the same convention as
    :func:`~manifold.modules.sample_paired_latent_flow`. The source latent is
    threaded through both Heun evaluations (the source does not move with ``t``).
    """
    x0_1 = _paired_unet_call(unet, z, x_src, t, spacing_t, src_labels, tgt_labels)
    z_euler, v1 = scheduler.euler_step(x0_1, z, t, t_next)
    if float(t_next) >= 1.0:
        return z_euler
    x0_2 = _paired_unet_call(unet, z_euler, x_src, t_next, spacing_t, src_labels, tgt_labels)
    return scheduler.heun_correct(x0_2, z, z_euler, v1, t, t_next)


def _heun_rollout_paired(
    unet, scheduler, z_start: Tensor, x_src: Tensor, nodes: Tensor,
    spacing_t: Tensor, src_labels: Tensor, tgt_labels: Tensor,
    start_i: int, end_i: int,
) -> list[Tensor]:
    """Deterministic Heun from node ``start_i`` to ``end_i``; returns ``[z_start_i, …, z_end_i]``.

    Used for both the shared anchor (``start_i=0`` from ``z_0 = x_src``) and each
    branch's suffix (``start_i=k+1``). ``x_src`` / ``src_labels`` / ``tgt_labels`` /
    ``spacing_t`` are already at the batch size of ``z_start`` — ``(B,)`` for the
    anchor, ``(B·G,)`` for the suffix (the caller G-expands them).
    """
    zs = [z_start]
    z = z_start
    for i in range(start_i, end_i):
        z = _heun_one_step_paired(
            unet, scheduler, z, x_src, float(nodes[i]), float(nodes[i + 1]),
            spacing_t, src_labels, tgt_labels,
        )
        zs.append(z)
    return zs


@torch.no_grad()
def singular_branch_rollout_paired(
    unet,
    scheduler: FlowMatchBridgeGRPOScheduler,
    reward_model,
    x_src: Tensor,
    spacing: Tensor | Sequence[float],
    src_label,
    tgt_label,
    *,
    G: int,
    eta_step_list: Sequence[int],
    num_steps: int,
    adv_clip_max: float = 5.0,
    reward_transform: Callable[[Tensor], Tensor] | None = None,
) -> list[RolloutStep]:
    """One G2RPO singular-branch rollout over the src→tgt bridge (no_grad) → per-step buffer.

    Sibling of :func:`~manifold.modules.singular_branch_rollout` for the paired
    data→data bridge. The shared anchor starts from ``z_0 = x_src`` (the ``t = 0``
    data endpoint — NOT sampled noise) and runs the deterministic two-eval Heun to
    ``z_k`` at each perturbed node. At each perturbed step the ``G`` siblings branch
    off ``z_k`` via one bridge SDE draw (:meth:`FlowMatchBridgeGRPOScheduler.sde_step_mean`),
    roll a deterministic Heun suffix to the terminal ``z_K``, and are scored by the
    frozen paired reward over ``cat([x_src_bg, z_K])``; the per-group advantage is
    normalized over ``G``.

    Args:
        unet: the Paired JiT policy UNet (``in_channels = 2·C_latent``; run eval + no_grad).
        scheduler: a :class:`FlowMatchBridgeGRPOScheduler` (its grid + inherited Heun +
            ``sde_step_mean`` run; ``set_timesteps`` is called here).
        reward_model: the frozen paired :class:`~manifold.RewardModel`
            (``in_channels = 2·C_latent``) scoring ``cat([x_src_bg, z_K])``.
        x_src: the source latent ``(B, C_latent, D, H, W)`` — the ``t = 0`` data
            endpoint and the rollout's starting point (the bridge is data→data).
        spacing: raw voxel spacing ``[3]`` or ``[B, 3]``.
        src_label / tgt_label: the contrast labels whose embeddings are summed for
            the translation direction (ADR-0014). Scalar ``int`` (broadcast) or
            ``[B]`` long tensor of per-sample labels.
        G: the group size (siblings per conditioning).
        eta_step_list: the perturbed step indices ``k`` (the first half, e.g.
            ``[0..3]`` of ``num_steps=8`` — away from the §7 var-collapse terminal).
        num_steps: the anchor grid resolution (Heun steps over ``t: 0 → 1``).
        adv_clip_max: the advantage-magnitude clip.
        reward_transform: optional monotone bound applied to the raw rewards before
            group normalization (the tanh cap, ADR-0015); ``None`` ⇒ raw logit.

    Returns:
        One :data:`RolloutStep` per ``k`` in ``eta_step_list`` (sorted ascending).
    """
    device = next(unet.parameters()).device
    dtype = next(unet.parameters()).dtype
    B = x_src.shape[0]
    spatial = x_src.shape[1:]  # (C_latent, D, H, W) — the moving latent's channels

    spacing_t = torch.as_tensor(spacing, device=device)
    src_labels = _as_label_tensor(src_label, B, device)
    tgt_labels = _as_label_tensor(tgt_label, B, device)

    nodes = scheduler.set_timesteps(num_steps, device=device)  # (num_steps+1,)
    eta_steps = sorted(int(k) for k in eta_step_list)
    if eta_steps[-1] >= num_steps:
        raise ValueError(
            f"eta_step_list max ({eta_steps[-1]}) must be < num_steps ({num_steps}) — "
            "a perturbed step k advances node k → k+1 and needs a suffix node."
        )
    # The §7 bridge std² = η·Δt·(1−t_next)/(1−t) vanishes at the terminal node
    # (t_next = nodes[num_steps] = 1.0 ⇒ std = 0). With η>0 a terminal perturbed step
    # (k = num_steps−1) collapses all G siblings to identical z_K ⇒ zero advantage ⇒ a
    # silently wasted inner step (no gradient, no error). Reject it for the real
    # η>0 training regime. (η=0 is the anchor-parity debug case — std=0 everywhere by
    # construction, so the terminal step is allowed there.)
    eta = float(getattr(scheduler, "eta", 0.0))
    if eta > 0.0 and eta_steps[-1] >= num_steps - 1:
        raise ValueError(
            f"eta_step_list {eta_steps} includes the terminal step (num_steps−1="
            f"{num_steps - 1}) with eta={eta} > 0: the §7 bridge std collapses to 0 "
            "there (t_next=1 ⇒ σ²=0), making all G siblings identical (zero advantage, "
            "a silently wasted inner step). Keep eta_step_list in the first half, away "
            "from the var-collapse terminal (the committed recipe uses [0..3] of 8)."
        )
    max_k = eta_steps[-1]

    unet.eval()
    x_src_dev = x_src.to(device=device, dtype=dtype)
    # G-expanded conditioning for the suffix + reward (flat index b·G+g is sibling g
    # of b — matching z_kplus1.reshape(B·G, ...)). The anchor/perturbed evals stay at
    # (B,); the suffix + reward run at (B·G,) on these G-expanded tensors.
    src_labels_bg = src_labels.repeat_interleave(G)  # (B·G,)
    tgt_labels_bg = tgt_labels.repeat_interleave(G)  # (B·G,)
    if spacing_t.dim() == 2:  # per-sample (B, 3) spacing
        spacing_bg = spacing_t.repeat_interleave(G, dim=0)
    else:  # broadcast (3,) spacing — fine for any batch
        spacing_bg = spacing_t
    x_src_bg = x_src_dev.repeat_interleave(G, dim=0)  # (B·G, C_latent, ...)

    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        # Shared anchor: z at nodes [0..max_k], starting from z_0 = x_src (the data
        # endpoint). anchor_z[k] = the moving latent at node k (same path the deployed
        # Paired JiT sampler takes — the anchor-parity guard).
        anchor_z = _heun_rollout_paired(
            unet, scheduler, x_src_dev, x_src_dev, nodes, spacing_t, src_labels, tgt_labels, 0, max_k
        )

        buffer: list[RolloutStep] = []
        for k in eta_steps:
            t_k = float(nodes[k])
            t_next = float(nodes[k + 1])
            z_k = anchor_z[k]  # (B, *spatial) — the anchor node

            # Perturbed-step rollout eval: the policy SDE transition params at z_k.
            x0 = _paired_unet_call(unet, z_k, x_src_dev, t_k, spacing_t, src_labels, tgt_labels)
            mean_old, std_old = scheduler.sde_step_mean(x0, z_k, t_k, t_next)
            # G siblings branch off z_k via one bridge SDE draw each.
            xi = torch.randn(B, G, *spatial, device=device, dtype=dtype)
            z_kplus1 = mean_old.unsqueeze(1) + float(std_old) * xi  # (B, G, *spatial)
            old_log_prob = gaussian_log_prob(z_kplus1, mean_old, std_old)  # (B, G)

            # Deterministic Heun suffix from z_{k+1} (node k+1) to the terminal z_K,
            # threading x_src_bg + the G-expanded labels through every suffix eval.
            z_g = z_kplus1.reshape(B * G, *spatial)
            suffix = _heun_rollout_paired(
                unet, scheduler, z_g, x_src_bg, nodes, spacing_bg, src_labels_bg, tgt_labels_bg,
                k + 1, num_steps,
            )
            z_K = suffix[-1]  # (B·G, *spatial)

            # D9 (silent trap): the paired reward scores cat([x_src_bg, z_K]) — x_src
            # MUST be G-expanded to (B·G, C, ...) BEFORE the concat, so each sibling's
            # terminal z_K pairs with its OWN source. A wrong expansion order pairs
            # siblings with the wrong source ⇒ corrupted advantage, no error.
            reward_input = torch.cat([x_src_bg, z_K], dim=1)  # (B·G, 2·C_latent, ...)
            # float(): under cuda autocast the PatchGAN emits fp16 rewards; the group
            # normalization must run in fp32 (mirrors the noise→data rollout).
            rewards = reward_model(reward_input).float().reshape(B, G)  # (B, G)
            if reward_transform is not None:  # the v2 bound (ADR-0015)
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


#: A G2RPO training/validation batch: the source latent + the contrast direction +
#: the voxel spacing (pure-RL — the target volume is unused at train; mirrors GRPO's
#: conditioning-only batch). ``src_latent``: ``[B, C_latent, …]`` (the ``t = 0`` data
#: endpoint and the concat conditioning); ``src_label`` / ``tgt_label``: the contrast
#: pair (scalar or ``[B]``); ``spacing``: ``[3]`` or ``[B, 3]``.
PairedGRPOBatch = dict[str, Any]


class PairedGRPOModule(spt.Module):
    """Granular-GRPO policy post-training of the Paired JiT UNet (ADR-0024 / G2RPO).

    Overrides :meth:`training_step` (NOT ``forward`` — GRPO is multi-term, multi-step):
    a no-grad :func:`singular_branch_rollout_paired` over the src→tgt bridge fills the
    buffer; then a **multi-step PPO inner loop** over ``eta_step_list`` recomputes
    ``new_log_prob`` under grad (the policy's one live grad eval per branch —
    ``cat([z_k, x_src])`` + summed embed re-evaluated at the stored anchor node),
    builds the :func:`clipped_surrogate_loss`, and runs one ``opt.step`` per step so
    the ratio drifts off 1 and the clip **binds**. The KL anchor ``0.5·‖μ_θ −
    μ_ref‖²/σ²`` (valid because the bridge ``σ`` is θ-independent) and the tanh reward
    bound carry over from ADR-0015.

    Holds the **trainable Paired JiT UNet** (the policy — the only params optimized)
    and the **frozen** paired :class:`~manifold.RewardModel`
    (``in_channels = 2·C_latent``) + the **frozen** reference UNet (the KL anchor),
    both unregistered via ``object.__setattr__``. **No EMA, no CFG** (paired has no
    unconditional path); resume / select / export the raw arm (ADR-0006/0012;
    inverts ADR-0021 for this stage). Validation + deployment use the deterministic
    Heun (:meth:`sample` delegates to :func:`sample_paired_latent_flow`) — the bridge
    is training-only exploration.

    Args:
        policy: the trainable Paired JiT UNet (``in_channels = 2·C_latent``).
        reward_model: the frozen paired :class:`~manifold.RewardModel`
            (``in_channels = 2·C_latent``) scoring ``cat([x_src, z_K])``.
        scheduler: a stateless :class:`FlowMatchBridgeGRPOScheduler`.
        G: the group size (siblings per conditioning).
        eta_step_list: the perturbed step indices (the first half, e.g. ``[0..3]``).
        clip_range: ε for the clipped surrogate (the tight PPO trust region).
        lr: the Adam LR over the policy UNet.
        adv_clip_max: the advantage-magnitude clip.
        num_steps: the anchor grid resolution (train rollout + deployed Heun steps).
        reference_policy: optional frozen deepcopy of the pretrained Paired JiT UNet —
            the KL anchor (ADR-0015). ``None`` ⇒ no KL.
        kl_coef: β for the KL-to-reference penalty (per-element scale — effective
            joint weight = ``kl_coef/D``; set ``kl_coef = β·D`` for a textbook β);
            ``≤ 0`` disables it.
        reward_bound: ``"none"`` (raw logit) or ``"tanh"`` (soft-clip, ADR-0015).
        reward_temp: the tanh temperature (≈ the real-data reward std).
    """

    def __init__(
        self,
        policy,
        reward_model,
        scheduler: FlowMatchBridgeGRPOScheduler,
        *,
        G: int = 8,
        eta_step_list: Sequence[int] = (0, 1, 2, 3),
        clip_range: float = 1e-4,
        lr: float = 1e-6,
        adv_clip_max: float = 5.0,
        num_steps: int = 8,
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
        eta_steps_tuple = tuple(int(k) for k in eta_step_list)
        if len(eta_steps_tuple) == 0:
            # An empty eta_step_list ⇒ an empty rollout buffer ⇒ the inner loop never
            # runs ⇒ no optimizer step ⇒ train/loss=0 with no policy update (a silent
            # no-op that wastes a whole training job). Fail fast at construction.
            raise ValueError(
                "eta_step_list must be non-empty (the perturbed step indices the PPO "
                "inner loop iterates over); got an empty sequence."
            )
        if float(kl_coef) > 0.0 and reference_policy is None:
            # Fail-fast against a silent dead KL anchor: kl_coef>0 expresses the intent
            # to anchor the policy to a frozen reference (ADR-0015), but without a
            # reference_policy _transition_kl short-circuits to None and the term is
            # never added — the policy would reward-hack with no indication the
            # regularizer is off. The committed recipe sets kl_coef=0.1, so #104's
            # _real_inputs MUST supply the reference (a deepcopy of the init policy);
            # this guard makes a forgotten reference crash loudly instead of silently.
            raise ValueError(
                f"kl_coef={kl_coef} > 0 requires a reference_policy (the frozen KL "
                "anchor, ADR-0015); pass reference_policy=deepcopy(policy) or set "
                "kl_coef=0. Without a reference the KL term is silently a no-op."
            )
        # NOTE: forward is NOT passed to spt.Module — training_step is overridden
        # directly (the multi-step inner loop cannot fit the single-loss seam).
        super().__init__(hparams={
            "lr": lr, "G": G, "clip_range": clip_range,
            "num_steps": num_steps, "adv_clip_max": adv_clip_max,
        })
        self.unet = policy  # registered → the only optimized / checkpointed arm
        self.scheduler = scheduler  # carries η (the bridge exploration knob)
        # Frozen paired reward, held UNregistered (object.__setattr__ bypasses
        # nn.Module registration) → absent from parameters()/state_dict()/optimizer/DDP,
        # moved to the device manually in on_fit_start (mirrors GRPOModule's reward).
        reward_model = reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "reward_model", reward_model)

        # Frozen reference policy (the KL anchor, ADR-0015): a deepcopy of the
        # pretrained Paired JiT UNet, held UNregistered exactly like the reward.
        # ``None`` ⇒ no KL term (the v1 backward-compat default). ``kl_coef ≤ 0`` also
        # disables it.
        if reference_policy is not None:
            reference_policy = reference_policy.eval()
            for p in reference_policy.parameters():
                p.requires_grad_(False)
            object.__setattr__(self, "reference_unet", reference_policy)
        else:
            object.__setattr__(self, "reference_unet", None)

        self.G = int(G)
        self.eta_step_list = eta_steps_tuple
        self.clip_range = float(clip_range)
        self.lr = float(lr)
        self.adv_clip_max = float(adv_clip_max)
        self.num_steps = int(num_steps)
        self.kl_coef = float(kl_coef)
        self.reward_bound = str(reward_bound)
        self.reward_temp = float(reward_temp)

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
        """Adam over the policy UNet only (the frozen reward/reference are unregistered)."""
        return {"optimizer": torch.optim.Adam(self.unet.parameters(), lr=self.lr)}

    # -- training_step: bridge rollout + multi-step PPO inner loop -------------

    def _policy_dtype(self) -> torch.dtype:
        return next(self.unet.parameters()).dtype

    def _conditioning_tensors(
        self, batch: PairedGRPOBatch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Build ``(spacing_t, src_labels, tgt_labels, x_src, B)`` for the batch.

        ``x_src`` (``batch["src_latent"]``) is the bridge's ``t = 0`` data endpoint
        and the concat conditioning; it is moved to the policy device/dtype. G2RPO is
        pure-RL — the target volume is unused at train (the bridge pins ``Z_1 → x̂₁_θ``).
        """
        spacing_t = torch.as_tensor(batch["spacing"], device=self.device)
        src_labels = _as_label_tensor(batch["src_label"], int(batch["src_latent"].shape[0]), self.device)
        tgt_labels = _as_label_tensor(batch["tgt_label"], int(batch["src_latent"].shape[0]), self.device)
        x_src = batch["src_latent"].to(device=self.device, dtype=self._policy_dtype())
        return spacing_t, src_labels, tgt_labels, x_src, int(x_src.shape[0])

    def _new_log_prob(
        self, step: RolloutStep, spacing_t: Tensor, src_labels: Tensor, tgt_labels: Tensor, x_src: Tensor
    ) -> tuple[Tensor, Tensor, Tensor | float]:
        """Recompute the transition log-prob under grad (the inner-loop grad eval).

        Re-evaluates the Paired JiT UNet at the stored anchor node ``z_k`` on
        ``cat([z_k, x_src])`` (the single live grad eval per branch — peak autograd
        memory is one UNet-forward) and forms the Gaussian transition density at the
        stored ``z_{k+1}``. Returns ``(log_prob, mean_new, std_new)`` so the caller
        can reuse ``mean_new`` for the KL term (no second policy forward) — ADR-0015.
        """
        z_k = step["z_k"]  # (B, *spatial) — detached anchor node
        x0 = _paired_unet_call(
            self.unet, z_k, x_src, step["t_k"], spacing_t, src_labels, tgt_labels
        )
        mean_new, std_new = self.scheduler.sde_step_mean(x0, z_k, step["t_k"], step["t_next"])
        log_prob = gaussian_log_prob(step["z_kplus1"], mean_new, std_new)  # (B, G)
        return log_prob, mean_new, std_new

    def _transition_kl(
        self, step: RolloutStep, mean_new: Tensor, std_new, spacing_t: Tensor,
        src_labels: Tensor, tgt_labels: Tensor, x_src: Tensor,
    ) -> Tensor | None:
        """The per-transition KL anchor ``0.5·‖μ_θ − μ_ref‖²/σ²`` (ADR-0015), or ``None``.

        The bridge transition ``π_θ(z_{k+1}|z_k) = N(μ_θ, σ²·I)`` and the reference
        transition share **equal variance** (the bridge ``σ`` depends only on
        ``(t, t_next, η)``, not on the policy weights — ADR-0024), so the
        diagonal-Gaussian KL collapses to the squared-mean difference over ``σ²``
        (the trace / log-det terms cancel). The reference mean is a frozen, no-grad
        forward at the stored ``z_k`` on ``cat([z_k, x_src])``; grad flows through
        ``μ_θ`` only. Returns ``None`` (no KL added) when the reference is absent or
        ``kl_coef ≤ 0`` — the backward-compat default.
        """
        if self.reference_unet is None or self.kl_coef <= 0.0:
            return None
        with torch.no_grad():
            x0_ref = _paired_unet_call(
                self.reference_unet, step["z_k"], x_src, step["t_k"], spacing_t, src_labels, tgt_labels
            )
        mean_ref, _ = self.scheduler.sde_step_mean(x0_ref, step["z_k"], step["t_k"], step["t_next"])
        var = max(float(std_new) ** 2, 1e-12)  # the η=0 degenerate floor (mirrors gaussian_log_prob)
        return 0.5 * ((mean_new - mean_ref) ** 2 / var).flatten(start_dim=1).mean(dim=1)  # (B,)

    def _bound_reward(self, r: Tensor) -> Tensor:
        """Bound the raw PatchGAN reward so OOD latents cannot dominate (ADR-0015).

        ``none`` is the identity (the raw logit). ``tanh`` maps ``tanh(r / reward_temp)``
        into (−1, 1) — a MONOTONIC soft-clip, so distinct sibling rewards stay distinct
        while an OOD extreme saturates near ±1 instead of dominating the advantage.
        """
        if self.reward_bound == "none":
            return r
        if self.reward_bound == "tanh":
            return torch.tanh(r / self.reward_temp)
        raise ValueError(f"unknown reward_bound {self.reward_bound!r} (expected 'none' or 'tanh')")

    def training_step(self, batch: PairedGRPOBatch, batch_idx: int):
        """One G2RPO update: no-grad bridge rollout → multi-step PPO inner loop.

        The rollout (anchor from ``x_src`` + G bridge branches + Heun suffix + reward
        + group advantage) runs under ``no_grad`` and fills the buffer; the inner loop
        then runs one PPO update per perturbed step — recompute ``new_log_prob`` under
        grad on ``cat([z_k, x_src])``, clipped surrogate, ``manual_backward``,
        ``opt.step``, ``zero_grad`` — so the ratio drifts off 1 from the second step on
        and the clip binds (ADR-0012). The bridge is training-only; ``sample`` +
        validation deploy the deterministic Heun.
        """
        if not isinstance(batch, dict):
            raise ValueError(f"batch is expected to be a dict, not {type(batch)}")
        batch["batch_idx"] = batch_idx
        spacing_t, src_labels, tgt_labels, x_src, B = self._conditioning_tensors(batch)
        # The v2 reward bound (ADR-0015): None for the raw-logit default, else tanh.
        reward_transform = self._bound_reward if self.reward_bound != "none" else None
        buffer = singular_branch_rollout_paired(
            self.unet, self.scheduler, self.reward_model, x_src, spacing_t, src_labels, tgt_labels,
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
            new_lp, mean_new, std_new = self._new_log_prob(
                step, spacing_t, src_labels, tgt_labels, x_src
            )
            loss = clipped_surrogate_loss(
                new_lp, step["old_log_prob"], step["advantage"], self.clip_range
            )
            # The v2 KL anchor (ADR-0015): outside the PPO ratio clip — ``None``
            # (kl_coef=0 / no reference) adds nothing.
            kl = self._transition_kl(step, mean_new, std_new, spacing_t, src_labels, tgt_labels, x_src)
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
        self.log("train/loss", mean_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        return {"loss": mean_loss}

    # -- generation (the PSNR callback drives this via sample_paired_latent_flow) -

    def sample(
        self,
        src_latent,
        spacing,
        src_label: int,
        tgt_label: int,
        num_inference_steps: int,
    ) -> Tensor:
        """Translate ``src_latent`` → a predicted tgt latent via the deployed Heun.

        G2RPO evaluates the **raw** policy through the shared
        :func:`~manifold.modules.sample_paired_latent_flow` primitive (the deployed
        two-eval Heun, NOT the bridge SDE) so validation / the PSNR callback measure
        the deterministic distribution Paired JiT ships. No EMA swap (G2RPO trains
        no-EMA, ADR-0012). Args mirror :meth:`PairedLatentFlowModule.sample`.
        """
        return sample_paired_latent_flow(
            self.unet, self.scheduler, src_latent, spacing, src_label, tgt_label,
            num_inference_steps=num_inference_steps,
        )

    # -- validation: deployed-Heun generation + reward (val/mean_reward) --------

    def validation_step(self, batch: PairedGRPOBatch, batch_idx: int):
        """Generate from ``x_src`` via the deployed Heun, score → ``val/mean_reward``.

        The RL progress signal: generate via the deployed two-eval Heun (NOT the
        bridge SDE) so the reward reflects the deterministic distribution Paired JiT
        ships, then score with the frozen paired PatchGAN over ``cat([x_src, z_K])``.

        **Rank-0-only** (ADR-0016, M3): generation + reward scoring run on
        ``is_global_zero`` only; the non-root ranks skip and must NOT block on an
        NCCL collective here. ``val/mean_reward`` is therefore rank-0-shard-scoped
        (no ``sync_dist`` — the rank-0 gate removes the cross-rank quantity), the
        same convention as the noise→data GRPOModule. (The PSNR callback, attached
        separately, runs on all ranks + all_gathers — that is the global selection
        metric; this is the rank-0 progress signal.)
        """
        if not self.trainer.is_global_zero:
            return
        batch["batch_idx"] = batch_idx
        spacing_t, src_labels, tgt_labels, x_src, B = self._conditioning_tensors(batch)
        z_K = sample_paired_latent_flow(
            self.unet, self.scheduler, x_src, spacing_t, src_labels, tgt_labels,
            num_inference_steps=self.num_steps,
        )
        # Single sample per source at val (no G-expansion): cat([x_src, z_K]).
        rewards = self._bound_reward(
            self.reward_model(torch.cat([x_src, z_K], dim=1)).float()
        )
        self.log("val/mean_reward", rewards.mean(), on_epoch=True, prog_bar=True, batch_size=B)
        return {"mean_reward": rewards.mean()}


__all__ = [
    "PairedGRPOBatch",
    "PairedGRPOModule",
    "singular_branch_rollout_paired",
]
