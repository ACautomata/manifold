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
from .controlnet_sampler import _controlnet_x0, controlnet_rollout
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
    unet, scheduler, z: Tensor, t: float, t_next: float, spacing_t: Tensor, class_labels: Tensor,
    x0_fn: Callable[..., Tensor] | None = None, fn_labels: Any = None,
) -> Tensor:
    """One deterministic two-eval-Heun reverse step (the deployed sampler's step).

    Final-step Euler when ``t_next == 1`` (the ``1/(1 − t_next)`` corrector
    denominator diverges) — the same convention as
    :func:`~manifold.modules.sample_latent_flow` / :func:`partial_denoise_rollout`.

    ``x0_fn`` (Mode-2) overrides the x0 prediction: called as
    ``x0_fn(z, t, fn_labels)`` and must return the clean-latent prediction. ``None``
    (Mode-1) uses ``unet(sample=z, timestep=t, spacing, class_labels)`` directly.
    """
    if x0_fn is None:
        def x0_fn(z_, t_, _labels):  # noqa: ANN001
            return unet(sample=z_, timestep=t_, spacing=spacing_t, class_labels=class_labels)
    x0_1 = x0_fn(z, t, fn_labels)
    z_euler, v1 = scheduler.euler_step(x0_1, z, t, t_next)
    if float(t_next) >= 1.0:
        return z_euler
    x0_2 = x0_fn(z_euler, t_next, fn_labels)
    return scheduler.heun_correct(x0_2, z, z_euler, v1, t, t_next)


def _heun_rollout(
    unet, scheduler, z_start: Tensor, nodes: Tensor, spacing_t: Tensor, class_labels: Tensor,
    start_i: int, end_i: int,
    x0_fn: Callable[..., Tensor] | None = None, fn_labels: Any = None,
) -> list[Tensor]:
    """Deterministic Heun from node ``start_i`` to ``end_i``; returns ``[z_start_i, …, z_end_i]``.

    Used for both the shared anchor (``start_i=0``) and each branch's suffix
    (``start_i=k+1``). ``nodes`` is the scheduler's batch-wide grid
    (:meth:`~manifold.FlowMatchGRPOScheduler.set_timesteps`); the steps are the
    scheduler's inherited ``euler_step`` / ``heun_correct`` (never reimplemented).
    ``x0_fn`` / ``fn_labels`` (Mode-2) thread a ControlNet-conditioned x0 prediction
    through every step; ``None`` (Mode-1) uses the plain UNet call.
    """
    zs = [z_start]
    z = z_start
    for i in range(start_i, end_i):
        z = _heun_one_step(
            unet, scheduler, z, float(nodes[i]), float(nodes[i + 1]), spacing_t, class_labels,
            x0_fn=x0_fn, fn_labels=fn_labels,
        )
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
    x0_fn: Callable[..., Tensor] | None = None,
    reward_fn: Callable[[Tensor], Tensor] | None = None,
    fn_labels_bg: Any = None,
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
        reward_model: the frozen :class:`~manifold.RewardModel` scoring ``z_K``
            (Mode-1) — ignored when ``reward_fn`` is given (Mode-2).
        noise: the pure-noise group start ``(B, C, D, H, W)`` — shared across siblings.
        spacing / modality: the manifold conditioning (voxel spacing + class label).
        G: the group size (siblings per conditioning).
        eta_step_list: the perturbed step indices ``k`` (the noisy-half, e.g.
            ``[0..7]`` of ``num_steps=15``).
        num_steps: the anchor grid resolution (Heun steps over ``t: 0 → 1``).
        adv_clip_max: the advantage-magnitude clip.
        reward_transform: optional monotone bound applied to the raw rewards before the
            group normalization (the v2 tanh cap, ADR-0015); ``None`` ⇒ raw logit (v1).
        x0_fn: optional Mode-2 policy x0 prediction override, called
            ``x0_fn(z, t_scalar, fn_labels_bg)`` at every eval point (anchor, perturbed,
            suffix). Must be pre-bound to the ControlNet-conditioned frozen-base
            forward and the current conditioning (``z_k``-batch for anchor/perturbed,
            ``(B·G,)`` for the suffix — the same ``fn_labels_bg`` batching). ``None``
            (Mode-1) uses the plain ``unet(sample, timestep, spacing, class_labels)``.
        reward_fn: optional Mode-2 reward override, called ``reward_fn(z_K_bg)`` on the
            ``(B·G,)`` terminal latents (scores the terminal latent unconditionally,
            like Mode-1). ``None`` (Mode-1) calls ``reward_model(z_K)``.
        fn_labels_bg: the ``(B·G,)``-batched conditioning labels forwarded to
            ``x0_fn`` (Mode-2); ignored in Mode-1.

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
        anchor_z = _heun_rollout(
            unet, scheduler, z0, nodes, spacing_t, class_labels, 0, max_k,
            x0_fn=x0_fn, fn_labels=fn_labels_bg,
        )

        buffer: list[RolloutStep] = []
        for k in eta_steps:
            t_k = float(nodes[k])
            t_next = float(nodes[k + 1])
            z_k = anchor_z[k]  # (B, *spatial) — the anchor node

            # SDE transition params at the current (rollout-time) policy.
            if x0_fn is not None:
                x0 = x0_fn(z_k, t_k, fn_labels_bg)
            else:
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
                unet, scheduler, z_g, nodes, spacing_bg, class_labels_bg, k + 1, num_steps,
                x0_fn=x0_fn, fn_labels=fn_labels_bg,
            )
            z_K = suffix[-1]  # (B·G, *spatial)

            # float(): under cuda autocast the PatchGAN emits fp16 rewards; the
            # group-normalization (std / (r−mean)) must run in fp32 to avoid the
            # fp16 square overflowing for large rewards (mirrors validation_step).
            if reward_fn is not None:
                rewards = reward_fn(z_K).float().reshape(B, G)  # (B, G) — Mode-2 (z_K only)
            else:
                rewards = reward_model(z_K).float().reshape(B, G)  # (B, G) — Mode-1 unconditional
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

    Holds the **trainable JiT UNet** (Mode-1 policy) OR the **frozen base UNet +
    trainable ControlNet** (Mode-2 policy), plus the **frozen**
    :class:`~manifold.RewardModel` and (optional) frozen reference policy. The frozen
    arms are **registered + dual-excluded** (ADR-0031): normal ``nn.Module`` children
    so Lightning owns their device placement, kept off the optimizer and the checkpoint
    via the :meth:`state_dict` / :meth:`load_state_dict` overrides, and held in
    :meth:`eval` across ``module.train()`` via the :meth:`train` override. No EMA;
    resumes / selects / exports the raw arm (ADR-0006/0012).

    **Two modes (ADR-0028).** The transition ``x_θ`` source is unified behind
    :meth:`_x0_policy`: Mode-1 reads it from the trainable UNet; Mode-2 reads it from
    the frozen base + trainable ControlNet (the base forward consumes the ControlNet's
    residual injections through the out-of-place residual forward — ADR-0026's
    corrected hazard, so the Mode-2 perturbed-step backward is autograd-safe). The
    optimizer wires the UNet params (Mode-1) or the ControlNet params (Mode-2). The
    KL anchor is ``deepcopy`` of the **policy** — Mode-2 captures base **and**
    ControlNet, so the closed-form diagonal-Gaussian KL stays valid while the
    ControlNet drifts (``σ`` is θ-independent ⇒ trace + log-det cancel). The spine
    (:func:`gaussian_log_prob`, :meth:`_transition_kl`, :func:`group_advantage`,
    :func:`clipped_surrogate_loss`, the multi-step PPO inner loop, the singular-branch
    rollout) reuses verbatim in both modes.

    Args:
        policy: the trainable JiT x0-denoiser UNet (Mode-1 policy). In Mode-2 this is
            the **frozen base** UNet (registered + dual-excluded — see ``controlnet``).
        reward_model: the frozen :class:`~manifold.RewardModel` scoring the terminal
            latent (Mode-1 and Mode-2 both score ``z_K`` unconditionally, ``reward(z_K)``).
        scheduler: a stateless :class:`FlowMatchGRPOScheduler`.
        G: the group size (siblings per conditioning).
        eta_step_list: the perturbed step indices (the noisy-half, e.g. ``[0..7]``).
        clip_range: ε for the clipped surrogate (the tight PPO trust region).
        lr: the Adam LR over the optimized params (UNet in Mode-1, ControlNet in Mode-2).
        adv_clip_max: the advantage-magnitude clip.
        num_steps: the anchor grid resolution (train rollout + validation Heun steps).
        latent_shape: the rollout latent shape ``(C, D, H, W)`` (GRPO is generative —
            the Module samples the group noise of this shape).
        reference_policy: optional frozen deepcopy of the policy — the KL anchor
            (ADR-0015). Mode-2: a ``(base, controlnet)`` pair (see
            :meth:`_transition_kl`). ``None`` ⇒ no KL (the v1 default).
        kl_coef: β for the KL-to-reference penalty; ``≤ 0`` disables it (v1 default 0.0).
        reward_bound: ``"none"`` (raw logit, v1) or ``"tanh"`` (soft-clip, ADR-0015).
        reward_temp: the tanh temperature (≈ the real-data reward std, ~8 from calibration).
        controlnet: optional :class:`~manifold.ControlNet3DConditionModel`. When set
            (with ``freeze_unet=True``) the module runs **Mode-2**: the base UNet is
            frozen + registered (dual-excluded), the ControlNet is the only optimized
            arm, and the batch must carry ``src_latent`` / ``src_label`` / ``tgt_label``.
        freeze_unet: freeze the base UNet (required in Mode-2; a no-op guard in
            Mode-1). When ``controlnet`` is set this must be ``True``.
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
        controlnet: Any = None,
        freeze_unet: bool = False,
    ):
        if G < 2:
            # group_advantage normalizes over the G siblings via torch.std (Bessel,
            # needs ≥2 samples); G=1 ⇒ std=NaN ⇒ NaN advantage ⇒ NaN grads destroy
            # the policy in one step. GRPO needs ≥2 siblings by definition.
            raise ValueError(f"G must be >= 2 (need >= 2 siblings per group), got {G}.")
        if controlnet is not None and not freeze_unet:
            raise ValueError(
                "Mode-2 (controlnet set) requires freeze_unet=True — the base UNet must "
                "be frozen (registered + dual-excluded) so only the ControlNet is optimized."
            )
        # NOTE: forward is NOT passed to spt.Module — training_step is overridden
        # directly (the multi-step inner loop cannot fit the single-loss seam).
        super().__init__(hparams={"lr": lr, "G": G, "clip_range": clip_range,
                                  "num_steps": num_steps, "adv_clip_max": adv_clip_max,
                                  "freeze_unet": freeze_unet})
        self.freeze_unet = bool(freeze_unet)
        if controlnet is not None:
            # Mode-2: the base UNet is FROZEN + registered (a normal nn.Module child
            # so Lightning's automatic .to(device) moves it per-rank), kept off the
            # optimizer and the checkpoint via the dual-exclude in state_dict() /
            # load_state_dict() (ADR-0031). The ControlNet is the only optimized +
            # checkpointed arm. (No manual on_fit_start .to(device) — Lightning owns it.)
            policy = policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
            self.unet = policy
            self.controlnet = controlnet  # registered → optimized / checkpointed
        else:
            # Mode-1: the UNet is the trainable policy (registered, optimized).
            self.unet = policy
            self.controlnet = None
        self.scheduler = scheduler  # carries η (the SDE exploration knob)
        # Frozen reward, registered + dual-excluded (ADR-0031): a normal nn.Module
        # child so Lightning's .to(device) moves it, but kept off the optimizer (the
        # trainable-arm-only configure_optimizers + requires_grad=False) and off the
        # checkpoint (the state_dict() override), and held in eval() across
        # module.train() via the train() override (the mode-management cost of
        # registration — the old object.__setattr__ bypass avoided it by hiding the
        # arm from the module tree, at the cost of Lightning not owning its device).
        reward_model = reward_model.eval()
        for p in reward_model.parameters():
            p.requires_grad_(False)
        self.reward_model = reward_model

        # Frozen reference policy (the KL anchor, ADR-0015): a deepcopy of the policy,
        # registered + dual-excluded exactly like the reward (ADR-0031). Mode-2: a
        # ``(base, controlnet)`` pair — both frozen + registered. ``None`` ⇒ no KL (the
        # v1 default); the None arms stay plain attributes (not submodules), so they
        # contribute no keys to state_dict() and no params to parameters().
        if reference_policy is not None:
            if controlnet is not None:
                ref_base, ref_controlnet = reference_policy
                ref_base = ref_base.eval()
                for p in ref_base.parameters():
                    p.requires_grad_(False)
                ref_controlnet = ref_controlnet.eval()
                for p in ref_controlnet.parameters():
                    p.requires_grad_(False)
                self.reference_unet = ref_base
                self.reference_controlnet = ref_controlnet
            else:
                reference_policy = reference_policy.eval()
                for p in reference_policy.parameters():
                    p.requires_grad_(False)
                self.reference_unet = reference_policy
                self.reference_controlnet = None
        else:
            self.reference_unet = None
            self.reference_controlnet = None

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

        #: The registered submodule prefixes kept off the optimizer + checkpoint
        #: (ADR-0031 dual-exclude). Always the frozen reward; the Mode-2 frozen base
        #: ``unet``; and the frozen KL-anchor arms when present. The Mode-1 ``unet``
        #: (the trainable policy) is intentionally NOT here. Declared once at init —
        #: the arm set is fixed at construction — and shared by state_dict() (strip)
        #: and load_state_dict() (the strict-load allow-list).
        frozen = {"reward_model"}
        if self.controlnet is not None:  # Mode-2: the base UNet is the frozen arm
            frozen.add("unet")
        if self.reference_unet is not None:
            frozen.add("reference_unet")
        if self.reference_controlnet is not None:
            frozen.add("reference_controlnet")
        self._frozen_arm_names: frozenset[str] = frozenset(frozen)

    # -- Mode-2 (ControlNet) helpers ------------------------------------------

    def _policy_device(self) -> torch.device:
        """Device of the optimized arm (Mode-1 UNet / Mode-2 ControlNet)."""
        params_owner = self.controlnet if self.controlnet is not None else self.unet
        return next(params_owner.parameters()).device

    def _controlnet_forward(self, controlnet, z, t, x_src, spacing_t, src_labels, tgt_labels):
        """One ControlNet-conditioned base x0 forward (frozen base + ControlNet residuals)."""
        return _controlnet_x0(self.unet, controlnet, z, t, x_src, spacing_t, src_labels, tgt_labels)

    def _reference_x0(self, z, t, spacing_t, class_labels, x_src=None, src_labels=None, tgt_labels=None):
        """The frozen KL-anchor x0 at ``z`` (Mode-1 UNet / Mode-2 base+ControlNet)."""
        if self.reference_controlnet is not None:
            return _controlnet_x0(
                self.reference_unet, self.reference_controlnet, z, t, x_src, spacing_t, src_labels, tgt_labels
            )
        return self.reference_unet(sample=z, timestep=t, spacing=spacing_t, class_labels=class_labels)

    def _mode2_rollout_fns(self, x_src, spacing_t, src_labels, tgt_labels):
        """Build the Mode-2 ``(x0_fn, reward_fn, fn_labels_bg)`` the rollout consumes.

        ``x0_fn`` is the frozen-base + trainable-ControlNet x0 prediction (the unified
        ``x_θ`` source); the inner-loop grad flows into the ControlNet through the base's
        out-of-place residual forward (ADR-0026 corrected hazard). ``reward_fn`` scores
        the terminal latent ``z_K`` unconditionally (``reward(z_K)``) — the ControlNet's
        conditional fidelity is driven by the policy x0 (which sees ``x_src``), not by
        the reward input. ``x_src`` /
        ``src_labels`` / ``tgt_labels`` are G-expanded ONCE here (flat index ``b·G+g``
        is sibling ``g`` of ``b``) — matching ``z_kplus1.reshape(B·G, ...)`` and the
        suffix ``(B·G,)`` batch — so a single ``fn_labels_bg`` tuple serves the perturbed
        eval and the suffix without a second expansion (D9).
        """
        G = self.G
        x_src_bg = x_src.repeat_interleave(G, dim=0)  # (B·G, C, ...)
        src_labels_bg = src_labels.repeat_interleave(G)
        tgt_labels_bg = tgt_labels.repeat_interleave(G)
        spacing_bg = spacing_t.repeat_interleave(G, dim=0) if spacing_t.dim() == 2 else spacing_t
        fn_labels_bg = (x_src_bg, spacing_bg, src_labels_bg, tgt_labels_bg)

        def x0_fn(z_, t_, labels):
            xs_bg, sp_bg, sl_bg, tl_bg = labels
            # The rollout calls x0_fn on TWO batch sizes: the anchor / perturbed eval at
            # (B,) and the suffix at (B·G,). ``labels`` is pre-G-expanded to (B·G,) in
            # repeat_interleave layout [b0g0, b0g1, …, b1g0, …] (so the suffix + reward
            # concat reuse one expansion, D9); for the (B,) anchor/perturbed batch stride
            # by G (b's sibling-0 slot) to recover the per-b conditioning.
            b = z_.shape[0]
            if b == xs_bg.shape[0]:  # suffix batch (B·G,) — use the full expansion
                xs, sp, sl, tl = xs_bg, sp_bg, sl_bg, tl_bg
            else:  # anchor/perturbed batch (B,) — stride G
                xs, sl, tl = xs_bg[:: self.G], sl_bg[:: self.G], tl_bg[:: self.G]
                sp = sp_bg[:: self.G] if sp_bg.dim() == 2 else sp_bg
            return self._controlnet_forward(self.controlnet, z_, t_, xs, sp, sl, tl)

        def reward_fn(z_K_bg):
            return self.reward_model(z_K_bg)

        return x0_fn, reward_fn, fn_labels_bg

    def _conditioning(self, batch: GRPOBatch):
        """Unified batch conditioning → ``(spacing_t, class_labels, B, cond)``.

        Mode-1: ``class_labels`` is the modality label and ``cond`` is ``None``. Mode-2:
        ``class_labels`` is the per-sample ``tgt_label`` (the base's own modality
        embedding sees the target contrast only) and ``cond`` is
        ``(x_src, src_labels, tgt_labels)`` — the ControlNet's control signal + the
        (src, tgt) direction pair. ``B`` is the batch size (Mode-1 = #labels, Mode-2 =
        ``x_src`` batch).
        """
        spacing_t = torch.as_tensor(batch["spacing"], device=self.device)
        if self.controlnet is not None:
            x_src = batch["src_latent"].to(device=self.device, dtype=self._policy_dtype())
            B = int(x_src.shape[0])
            src_labels = self._mode2_labels(batch["src_label"], B)
            tgt_labels = self._mode2_labels(batch["tgt_label"], B)
            return spacing_t, tgt_labels, B, (x_src, src_labels, tgt_labels)
        label = batch["label"]
        if isinstance(label, Tensor):
            class_labels = label.to(device=self.device, dtype=torch.long)
        else:
            class_labels = torch.full((1,), int(label), dtype=torch.long, device=self.device)
        return spacing_t, class_labels, int(class_labels.shape[0]), None

    def _mode2_labels(self, labels, B: int) -> Tensor:
        """Coerce a (src|tgt) label to a ``(B,)`` long tensor (scalar broadcast or ``(B,)``)."""
        if isinstance(labels, Tensor):
            out = labels.to(device=self.device, dtype=torch.long)
            if out.dim() == 0:
                out = out.expand(B)
            return out
        return torch.full((B,), int(labels), dtype=torch.long, device=self.device)

    def _policy_and_reference_pair(self):
        """``deepcopy`` of the policy → the Mode-aware KL-anchor ``reference_policy`` arg.

        Mode-1: a plain deepcopy of the UNet. Mode-2: the ``(base, controlnet)`` pair
        (``reference_unet`` / ``reference_controlnet``). Callers (grpo_cli) use this to
        snapshot the anchor BEFORE any GRPO update (ADR-0015).
        """
        import copy

        if self.controlnet is not None:
            return (copy.deepcopy(self.unet), copy.deepcopy(self.controlnet))
        return copy.deepcopy(self.unet)

    def _score_reward(self, z_K: Tensor, cond) -> Tensor:
        """Bound the frozen reward on the terminal latent ``z_K`` (Mode-1 / Mode-2 unified).

        The reward scores the generated terminal latent ``z_K`` unconditionally in both
        modes (``reward(z_K)``) — Mode-2 does NOT concat ``x_src``. The ControlNet's
        conditional fidelity is driven by the *policy* x0 (which sees ``x_src``), not by
        the reward input; the reward stays a plain realism/fidelity scorer on ``z_K``.
        """
        raw = self.reward_model(z_K).float()
        return self._bound_reward(raw)

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

    # -- frozen-arm registration + dual-exclude (ADR-0031) --------------------

    def _is_frozen_key(self, key: str) -> bool:
        """Whether a ``state_dict`` key belongs to a registered frozen arm.

        Matched on the key's top-level segment (the prefix before the first ``.``), so
        ``reward_model.conv.weight`` matches the ``reward_model`` arm. Non-frozen keys
        (the trainable ``unet`` in Mode-1, the trainable ``controlnet`` in Mode-2)
        return ``False`` — they stay on the checkpoint and in the optimizer.
        """
        return key.split(".", 1)[0] in self._frozen_arm_names

    def state_dict(self, *args, **kwargs):
        """Strip the registered frozen arms — they are rebuilt fresh each launch.

        The frozen reward / reference / Mode-2 base stay off the checkpoint: the reward
        is reloaded from its own ``.ckpt``, the reference is a launch-time ``deepcopy``
        (ADR-0028), and the Mode-2 base comes from the native export. Registering the
        arms (so Lightning owns their device placement) would otherwise leak their
        weights into the checkpoint; this override restores the off-checkpoint invariant
        at the source, so direct ``mod.state_dict()`` calls see them stripped (ADR-0031).
        """
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if not self._is_frozen_key(k)}

    def load_state_dict(self, state_dict, strict: bool = True, **kwargs):
        """Strict load over the TRAINABLE keys; frozen arms are an explicit allow-list.

        The checkpoint never carries frozen-arm weights (``state_dict()`` strips them);
        the arms are rebuilt fresh each launch (ADR-0031). Strip any stray frozen keys
        a stale pre-refactor checkpoint may carry too, then enforce strict parity on
        the trainable keys ONLY — a missing/unexpected TRAINABLE key (an incomplete or
        mode-mismatched ``.ckpt``) surfaces loudly, NOT silently via a blanket
        ``strict=False`` (which would also hide a missing trainable key and resume on
        random or stale weights, corrupting the experiment). The frozen arms being
        absent from the checkpoint is the one tolerated mismatch.
        """
        incoming = {k: v for k, v in state_dict.items() if not self._is_frozen_key(k)}
        result = super().load_state_dict(incoming, strict=False, **kwargs)
        # super() reports the registered frozen arms as missing (present in the module,
        # absent in the incoming) — that is the allow-listed tolerance. Anything else is
        # a real trainable-key mismatch and (when strict) must raise.
        bad_missing = [k for k in result.missing_keys if not self._is_frozen_key(k)]
        bad_unexpected = [k for k in result.unexpected_keys if not self._is_frozen_key(k)]
        if strict and (bad_missing or bad_unexpected):
            raise RuntimeError(
                f"Error(s) loading state_dict for {type(self).__name__} (frozen arms "
                f"{sorted(self._frozen_arm_names)} allow-listed): missing "
                f"{len(bad_missing)} trainable key(s) {bad_missing[:5]}; unexpected "
                f"{len(bad_unexpected)} key(s) {bad_unexpected[:5]}."
            )
        return result

    def train(self, mode: bool = True):
        """Re-freeze ``eval()`` on the registered frozen arms after ``super().train()``.

        Registration makes Lightning's ``module.train(mode)`` recurse into the frozen
        arms and flip them to training mode — an ``eval()`` set at construction does
        NOT persist. A frozen arm in training mode would let its BatchNorm running
        stats drift during rollout / reward evaluation, corrupting the supposedly-fixed
        function. This override re-applies ``eval()`` to every present frozen arm after
        the recursive call (the mode-management cost registration buys; ADR-0031).
        """
        result = super().train(mode)
        for name in self._frozen_arm_names:
            arm = getattr(self, name, None)
            if arm is not None:
                arm.eval()
        return result

    # -- optimizer ------------------------------------------------------------

    def configure_optimizers(self):
        """Adam over the optimized arm only — the UNet (Mode-1) or the ControlNet (Mode-2).

        The frozen arms (Mode-2 base, reward, reference) are registered but
        ``requires_grad=False`` and never selected here, so the optimizer never touches
        them — the off-optimizer invariant holds at the param-group level (ADR-0031)."""
        params = self.controlnet.parameters() if self.controlnet is not None else self.unet.parameters()
        return {"optimizer": torch.optim.Adam(params, lr=self.lr)}

    # -- training_step: rollout + multi-step PPO inner loop -------------------

    def _policy_dtype(self) -> torch.dtype:
        params_owner = self.controlnet if self.controlnet is not None else self.unet
        return next(params_owner.parameters()).dtype

    def _new_log_prob(
        self, step: RolloutStep, spacing_t: Tensor, class_labels: Tensor, cond=None
    ) -> tuple[Tensor, Tensor, Tensor | float]:
        """Recompute the transition log-prob under grad (the inner-loop grad eval).

        Re-evaluates the policy at the stored anchor node ``z_k`` (the single live grad
        eval per branch — peak autograd memory is one policy-forward) and forms the
        Gaussian transition density at the stored ``z_{k+1}``. The ``x_θ`` source is
        unified: Mode-1 is the trainable UNet; Mode-2 is the frozen base + trainable
        ControlNet (grad reaches the ControlNet through the base's out-of-place residual
        forward). ``old_log_prob`` was computed at rollout time; the ratio
        ``exp(new − old)`` is what the clip bounds. Returns ``(log_prob, mean_new,
        std_new)`` so the caller can reuse ``mean_new`` for the KL term (no second policy
        forward) — ADR-0015.
        """
        z_k = step["z_k"]  # (B, *spatial) — detached anchor node
        if cond is not None:
            x_src, src_labels, tgt_labels = cond
            x0 = self._controlnet_forward(
                self.controlnet, z_k, step["t_k"], x_src, spacing_t, src_labels, tgt_labels
            )
        else:
            x0 = self.unet(sample=z_k, timestep=step["t_k"], spacing=spacing_t, class_labels=class_labels)
        mean_new, std_new = self.scheduler.sde_step_mean(x0, z_k, step["t_k"], step["t_next"])
        log_prob = gaussian_log_prob(step["z_kplus1"], mean_new, std_new)  # (B, G)
        return log_prob, mean_new, std_new

    def _transition_kl(
        self, step: RolloutStep, mean_new: Tensor, std_new, spacing_t: Tensor, class_labels: Tensor,
        cond=None,
    ) -> Tensor | None:
        """The per-transition KL anchor ``0.5·‖μ_θ − μ_ref‖²/σ²`` (ADR-0015), or ``None``.

        The GRPO transition ``π_θ(z_{k+1}|z_k) = N(μ_θ, σ²_t·Δt·I)`` and the reference
        transition share **equal variance** (``σ_t = η·sqrt((1−t)/t)`` depends only on
        ``t``, not on the policy weights), so the diagonal-Gaussian KL collapses to the
        squared-mean difference over ``σ²`` (the trace / log-det terms cancel). The
        reference mean is a frozen, no-grad forward at the stored ``z_k`` (Mode-2: the
        frozen base + frozen ControlNet snapshot); grad flows through ``μ_θ`` only.
        Returns ``None`` (no KL added) when the reference is absent or ``kl_coef ≤ 0`` —
        the v1 backward-compat default. ``mean_new`` / ``std_new`` are the caller's
        grad-bearing policy outputs (reused, not recomputed).
        """
        if self.reference_unet is None or self.kl_coef <= 0.0:
            return None
        with torch.no_grad():
            if cond is not None:
                x_src, src_labels, tgt_labels = cond
                x0_ref = self._reference_x0(
                    step["z_k"], step["t_k"], spacing_t, class_labels,
                    x_src=x_src, src_labels=src_labels, tgt_labels=tgt_labels,
                )
            else:
                x0_ref = self._reference_x0(step["z_k"], step["t_k"], spacing_t, class_labels)
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
        spacing_t, class_labels, B, cond = self._conditioning(batch)
        # One shared group noise (identical across the G siblings ⇒ a shared anchor).
        noise = torch.randn(B, *self.latent_shape, device=self.device, dtype=self._policy_dtype())
        # The v2 reward bound (ADR-0015): None for the v1 raw-logit default (no-op),
        # else ``_bound_reward`` (tanh) caps the reward the rollout scores + groups.
        reward_transform = self._bound_reward if self.reward_bound != "none" else None
        # Mode-2: inject the ControlNet-conditioned x0 prediction + the z_K reward into
        # the shared rollout (Mode-1 leaves all three None — the plain UNet + the same
        # unconditional reward path). The spine reuses verbatim in both modes.
        x0_fn = reward_fn = fn_labels_bg = None
        if cond is not None:
            x_src, src_labels, tgt_labels = cond
            x0_fn, reward_fn, fn_labels_bg = self._mode2_rollout_fns(x_src, spacing_t, src_labels, tgt_labels)
        buffer = singular_branch_rollout(
            self.unet, self.scheduler, self.reward_model, noise, spacing_t, class_labels,
            G=self.G, eta_step_list=self.eta_step_list, num_steps=self.num_steps,
            adv_clip_max=self.adv_clip_max, reward_transform=reward_transform,
            x0_fn=x0_fn, reward_fn=reward_fn, fn_labels_bg=fn_labels_bg,
        )

        opt = self.optimizers()
        sched = self.lr_schedulers()
        if not isinstance(sched, (list, tuple)) and sched is not None:
            sched = [sched]
        # Eval mode for every policy eval (rollout + inner loop) — matches the deployed
        # sampler (deterministic; GroupNorm running stats, no dropout). Mode-2 also evals
        # the ControlNet (the frozen base is already eval).
        self.unet.eval()
        if self.controlnet is not None:
            self.controlnet.eval()
        losses = []
        for step in buffer:
            new_lp, mean_new, std_new = self._new_log_prob(step, spacing_t, class_labels, cond)
            loss = clipped_surrogate_loss(
                new_lp, step["old_log_prob"], step["advantage"], self.clip_range
            )
            # The v2 KL anchor (ADR-0015): a separate term outside the PPO ratio clip —
            # ``None`` (kl_coef=0 / no reference) adds nothing, preserving v1 behavior.
            kl = self._transition_kl(step, mean_new, std_new, spacing_t, class_labels, cond)
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
        spacing_t, class_labels, B, cond = self._conditioning(batch)
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
        if cond is not None:
            # Mode-2: the deployed deterministic ControlNet noise→data rollout from the
            # batch's own source (per-sample direction, NOT the bridge SDE), scored by the
            # frozen reward on the terminal latent z_K (unconditional, like Mode-1).
            x_src, src_labels, tgt_labels = cond
            z_K = controlnet_rollout(
                self.unet, self.controlnet, self.scheduler, noise, x_src, spacing_t,
                src_labels, tgt_labels, num_inference_steps=self.num_steps,
            )
        else:
            # Mode-1: a mixed-label val batch is generated under ``label[0]``
            # (``sample_latent_flow`` takes a scalar modality — single-modality v1).
            z_K = sample_latent_flow(
                self.unet, self.scheduler, noise, spacing_t, int(class_labels[0].item()),
                num_inference_steps=self.num_steps,
            )
        # The same bound the rollout applies (ADR-0015) — val/mean_reward is reported on
        # the bounded scale so it tracks the training signal (raw logit otherwise).
        rewards = self._score_reward(z_K, cond)
        valid = ~batch.get(
            "_is_padding", torch.zeros(B, dtype=torch.bool, device=rewards.device)
        ).to(rewards.device).bool()
        self._val_reward_sum += float(rewards[valid].sum())
        self._val_reward_count += int(valid.sum())
        return {"mean_reward": rewards[valid].mean() if bool(valid.any()) else rewards.new_zeros(())}
