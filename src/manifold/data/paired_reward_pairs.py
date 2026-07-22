"""ControlNet reward pair + probe generation (offline fake-cache builders).

Builds the condition-aware ``[2·C_latent]`` preference pairs for the paired reward
(ADR-0018/0019/0020). The **winner** is ``concat([x_src, real_tgt])`` (the
ground-truth target), the **loser** is ``concat([x_src, generated_tgt])`` (the
frozen ControlNet generator's noise→data rollout, ADR-0027). Both halves share the
``x_src`` channels so the discriminator scores "faithful translation *of* this src"
- an unconditional realism reward would reward copy-src (ADR-0019). The generated
tgt is never VAE-decoded/re-encoded (ADR-0018).

**Offline precompute (ADR-0020).** The fake cache is built **once** before training
(the rollout is deterministic given its noise); the
:class:`~manifold.modules.paired_reward.PairedRewardModule` holds no generator.
Inputs are **already scaled** into the generator's training space
(scale-consistency is the caller's job - ADR-0021; reuse ``vae.scaling_factor``
verbatim, never re-estimate).

The train/val builder uses :func:`~manifold.modules.controlnet_rollout` (the full
``0 -> 1`` noise→data rollout, base scheduler). The generated-end **probe** uses
:func:`~manifold.modules.controlnet_partial_rollout` (the partial ``t_start -> 1``
rollout, ``PartialFlowMatchHeunScheduler``) - both samples generated, ordered by
transport-progress ``t`` (winner = higher ``t``, ADR-0023).

Sibling of :mod:`manifold.data.reward_pairs` (the JiT reward); reuses
:class:`~manifold.data.reward_pairs.RewardPairDataset` verbatim.
"""

from __future__ import annotations

from typing import Sequence

import torch
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from torch import Tensor

from ..modules.controlnet_sampler import (
    controlnet_partial_rollout,
    controlnet_rollout,
)
from ..schedulers.scheduling_flow_match_heun import FlowMatchHeunDiscreteScheduler
from ..schedulers.scheduling_partial_flow_match_heun import PartialFlowMatchHeunScheduler
from .reward_pairs import RewardPairDataset

#: Probe corruption range (ADR-0023): ``t_start ∈ [0, 0.5)`` keeps probe samples
#: genuinely fake (a high ``t`` starts near real ``x_tgt`` -> the probe degenerates
#: to real-vs-fake). Mirrors the JiT probe range.
_PROBE_T_RANGE: tuple[float, float] = (0.0, 0.5)


def _sample_t(
    low: float, high: float, batch_size: int, generator: torch.Generator, device=None
) -> Tensor:
    """``t ~ U[low, high)`` per sample (half-open - never ``high``, avoiding ``t=1``).

    Mirrors :func:`manifold.data.reward_pairs._sample_t`: ``torch.rand`` is
    half-open ``[0, 1)``, so scaling keeps ``t < high`` (the probe never samples
    ``t = 1`` where the Heun endpoint denominator ``1 − t`` would vanish).
    """
    return low + (high - low) * torch.rand(batch_size, generator=generator, device=device)


def _as_per_sample(value, n: int, device: torch.device) -> Tensor | int:
    """Coerce a per-sample or scalar conditioning value to the rollout contract.

    A scalar ``int`` / 0-d tensor / 0-d sequence is returned as a plain ``int``
    (broadcast by the sampler); a ``[n]`` tensor / sequence becomes a ``[n]`` long
    tensor on ``device`` (per-sample labels for a batch mixing contrast directions).
    Normalizing once (before the batch loop) sidesteps the 0-d-tensor slice crash -
    a 0-d tensor cannot be sliced, so it must be reduced to an int up front.
    """
    if isinstance(value, (int, float)):
        return int(value)
    t = torch.as_tensor(value)
    if t.dim() == 0:
        return int(t.item())
    return t.to(device=device, dtype=torch.long)


def _canonical_device(device: torch.device) -> torch.device:
    """Normalize an unspecified CUDA device (``cuda``) to the current device index.

    ``torch.device("cuda") != torch.device("cuda:0")`` by Python equality even though
    they name the same GPU, so a naive compare false-positives on the standard
    ``device="cuda"`` CLI path (the generator lands on ``cuda:0`` after ``.to``).
    The CLI builds ``device = torch.device("cuda")``; canonicalize it to
    ``cuda:<current_device>`` so it compares equal to the generator's ``cuda:0``
    (codex #100 P1).
    """
    if device.type == "cuda" and device.index is None and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _resolve_rollout_device(generator, device) -> torch.device:
    """Resolve the rollout device, failing fast on a generator/device mismatch.

    The paired rollouts derive their execution device from
    ``next(generator.parameters()).device`` and return generated tensors there, so a
    caller-passed ``device`` that differs from the generator's leaves the concat
    ``[src_b (→device), gen_tgt (→gen_device)]`` mixing devices (codex #96 P2).
    Auto-detect when ``device is None``; otherwise require the generator to already
    be on ``device`` (the CLI's ``_real_inputs`` moves it before calling). CUDA device
    indices are canonicalized first (codex #100 P1).
    """
    gen_device = next(generator.parameters()).device
    resolved = torch.device(device) if device is not None else gen_device
    if _canonical_device(resolved) != _canonical_device(gen_device):
        raise ValueError(
            f"Generator is on {gen_device} but the builder was called with "
            f"device={resolved}. Move the generator onto {resolved} before calling "
            f"(the rollout runs on the generator's device), or pass device=None to "
            f"auto-detect from the generator."
        )
    return resolved


def build_paired_reward_pairs(
    x_src: Tensor | Sequence[Tensor],
    x_tgt: Tensor | Sequence[Tensor],
    generator,
    controlnet,
    scheduler: FlowMatchHeunDiscreteScheduler,
    *,
    src_label,
    tgt_label,
    spacing: Sequence[float] | Tensor,
    num_steps: int,
    batch_size: int = 4,
    device: torch.device | str | None = None,
) -> RewardPairDataset:
    """Build real-vs-fake condition-aware pairs: ``cat([x_src, real_tgt])`` vs ``cat([x_src, gen_tgt])``.

    The generated tgt is the **supervised ControlNet's** noise→data generation
    (:func:`~manifold.modules.controlnet_rollout` — frozen base + ControlNet, fresh
    Gaussian ``t = 0`` endpoint, ``x_src`` the control signal). The winner is the
    real target latent, the loser the ControlNet's generation — both concatenated
    with the (shared) source (ADR-0018/0019/0027).

    Args:
        x_src / x_tgt: scaled source / real-target latents ``[N, C_latent, D, H, W]``
            (the caller scales via ``vae.scaling_factor`` - ADR-0021).
        generator: the frozen base UNet.
        controlnet: the frozen :class:`~manifold.ControlNet3DConditionModel`.
        scheduler: the base :class:`FlowMatchHeunDiscreteScheduler` (the loser is a
            full rollout - NOT the Partial subclass).
        src_label / tgt_label: scalar ``int`` (broadcast) or length-``N`` per-sample
            labels (a batch mixing contrast directions).
        spacing: ``[3]`` (broadcast) or ``[N, 3]`` (per-sample).
        num_steps: Heun step budget for the rollout (a one-time precompute cost -
            ADR-0020; larger than the JiT train budget is affordable offline).

    Returns:
        A :class:`RewardPairDataset` of ``[2·C_latent]`` concat pairs.
    """
    src = x_src if isinstance(x_src, Tensor) else torch.stack(list(x_src))
    tgt = x_tgt if isinstance(x_tgt, Tensor) else torch.stack(list(x_tgt))
    if src.shape != tgt.shape:
        raise ValueError(f"x_src {tuple(src.shape)} and x_tgt {tuple(tgt.shape)} must match.")
    device = _resolve_rollout_device(generator, device)
    if not isinstance(spacing, Tensor):
        spacing = torch.as_tensor(spacing)
    n = len(src)
    # Normalize labels once (before the batch loop): a scalar int / 0-d tensor ->
    # int (broadcast); a [N] tensor -> on-device long tensor (per-sample). Slicing a
    # 0-d tensor would crash, so it must be reduced up front (codex P3).
    src_lab_full = _as_per_sample(src_label, n, device)
    tgt_lab_full = _as_per_sample(tgt_label, n, device)
    winners, losers = [], []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        src_b = src[start : start + b].to(device)
        tgt_b = tgt[start : start + b].to(device)
        spacing_b = (
            spacing[start : start + b]
            if isinstance(spacing, Tensor) and spacing.dim() == 2
            else spacing
        )
        src_lab = (
            src_lab_full[start : start + b] if isinstance(src_lab_full, Tensor) else src_lab_full
        )
        tgt_lab = (
            tgt_lab_full[start : start + b] if isinstance(tgt_lab_full, Tensor) else tgt_lab_full
        )
        # ControlNet noise→data fake: fresh Gaussian start (the t = 0 endpoint);
        # x_src is the control signal, not a transport endpoint.
        noise = torch.randn(src_b.shape, device=src_b.device, dtype=src_b.dtype)
        gen_tgt = controlnet_rollout(
            generator, controlnet, scheduler, noise, src_b, spacing_b, src_lab, tgt_lab,
            num_inference_steps=num_steps,
        ).detach()
        # Condition-aware concat: cat([x_src, tgt]) along channels (in_channels = 2·C).
        winners.append(torch.cat([src_b, tgt_b], dim=1).detach().cpu())
        losers.append(torch.cat([src_b, gen_tgt], dim=1).detach().cpu())
    rank_zero_info("build_paired_reward_pairs: %d real-vs-fake pairs (num_steps=%d).", n, num_steps)
    return RewardPairDataset(torch.cat(winners), torch.cat(losers))


def build_paired_reward_probe(
    x_src: Tensor | Sequence[Tensor],
    x_tgt: Tensor | Sequence[Tensor],
    generator,
    controlnet,
    partial_scheduler: PartialFlowMatchHeunScheduler,
    *,
    src_label,
    tgt_label,
    spacing: Sequence[float] | Tensor,
    num_steps: int,
    t_range: tuple[float, float] = _PROBE_T_RANGE,
    batch_size: int = 4,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> RewardPairDataset:
    """Build the generated-end probe: both samples generated, ordered by transport-progress ``t``.

    Both samples start from ``z = add_noise(x_tgt, ε, t_start)`` at
    ``t_start ∈ t_range`` (default ``[0, 0.5)`` - ADR-0023) and roll to ``t = 1`` via
    :func:`~manifold.modules.controlnet_partial_rollout` (frozen base + ControlNet,
    ``x_src`` the control signal); the **winner is the higher-``t``** sample (less
    transport -> closer to real ``x_tgt`` -> higher quality). Concatenated with
    ``x_src`` for condition-aware scoring (``val/gen_pair_acc`` - the within-fake
    ranking the checkpoint monitors; ``val/pair_acc`` saturates, ADR-0023).

    Deterministic given the seed (the ``t`` draws + the ControlNet corruption ``ε``
    use a fresh seeded generator and the rollout is deterministic) -> re-building
    with the same seed yields byte-identical probe pairs.
    """
    src = x_src if isinstance(x_src, Tensor) else torch.stack(list(x_src))
    tgt = x_tgt if isinstance(x_tgt, Tensor) else torch.stack(list(x_tgt))
    if src.shape != tgt.shape:
        raise ValueError(f"x_src {tuple(src.shape)} and x_tgt {tuple(tgt.shape)} must match.")
    device = _resolve_rollout_device(generator, device)
    if not isinstance(spacing, Tensor):
        spacing = torch.as_tensor(spacing)
    gen = torch.Generator(device=device).manual_seed(seed)  # device-aware (CPU gen raises on CUDA)
    n = len(src)
    # Normalize labels once (before the batch loop) - see build_paired_reward_pairs.
    src_lab_full = _as_per_sample(src_label, n, device)
    tgt_lab_full = _as_per_sample(tgt_label, n, device)
    winners, losers = [], []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        src_b = src[start : start + b].to(device)
        tgt_b = tgt[start : start + b].to(device)
        t_a = _sample_t(*t_range, b, gen, device=device)
        t_b = _sample_t(*t_range, b, gen, device=device)
        winner_t = torch.maximum(t_a, t_b)
        loser_t = torch.minimum(t_a, t_b)
        spacing_b = (
            spacing[start : start + b]
            if isinstance(spacing, Tensor) and spacing.dim() == 2
            else spacing
        )
        src_lab = (
            src_lab_full[start : start + b] if isinstance(src_lab_full, Tensor) else src_lab_full
        )
        tgt_lab = (
            tgt_lab_full[start : start + b] if isinstance(tgt_lab_full, Tensor) else tgt_lab_full
        )
        # ControlNet partial noise→data probe: corrupt endpoint is Gaussian ε
        # (the seeded generator keeps the probe reproducible); x_src the control.
        gen_w = controlnet_partial_rollout(
            generator, controlnet, partial_scheduler, src_b, tgt_b, winner_t,
            spacing_b, src_lab, tgt_lab, num_steps=num_steps, generator=gen,
        ).detach()
        gen_l = controlnet_partial_rollout(
            generator, controlnet, partial_scheduler, src_b, tgt_b, loser_t,
            spacing_b, src_lab, tgt_lab, num_steps=num_steps, generator=gen,
        ).detach()
        winners.append(torch.cat([src_b, gen_w], dim=1).detach().cpu())
        losers.append(torch.cat([src_b, gen_l], dim=1).detach().cpu())
    rank_zero_info(
        "build_paired_reward_probe: %d probe pairs (t ∈ %s, num_steps=%d).", n, t_range, num_steps
    )
    return RewardPairDataset(torch.cat(winners), torch.cat(losers))


def _stack_paired_latents(dataset) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Stack a warmed :class:`~manifold.data.PairedLatentDataset` into ``(src, tgt, src_lab, tgt_lab, spacing)``.

    The dataset's ``__getitem__`` emits ``{src_latent, tgt_latent, src_label,
    tgt_label, spacing}`` (both latents **scaled** by ``dataset.scaling_factor`` -
    scale-on-read, ADR-0003); stacking them yields the per-sample tensors the
    reward builders consume. The latents are already scaled into the generator's
    training space (the caller sets ``dataset.scaling_factor`` to the export's
    ``vae.scaling_factor`` verbatim - ADR-0021).
    """
    srcs, tgts, src_labs, tgt_labs, spacings = [], [], [], [], []
    for i in range(len(dataset)):
        item = dataset[i]
        srcs.append(item["src_latent"])
        tgts.append(item["tgt_latent"])
        src_labs.append(torch.as_tensor(item["src_label"], dtype=torch.long))
        tgt_labs.append(torch.as_tensor(item["tgt_label"], dtype=torch.long))
        spacings.append(torch.as_tensor(item["spacing"], dtype=torch.float32))
    # Each item's spacing is a [3] tensor -> stack yields per-sample [N, 3]; the
    # builder slices it per batch (and the rollout also accepts a [3] broadcast).
    return (
        torch.stack(srcs),
        torch.stack(tgts),
        torch.stack(src_labs),
        torch.stack(tgt_labs),
        torch.stack(spacings),
    )


def build_paired_reward_inputs(
    *,
    train_ds,
    val_ds,
    generator,
    controlnet,
    base_scheduler: FlowMatchHeunDiscreteScheduler,
    num_steps: int,
    probe_num_steps: int | None = None,
    n_probe: int = 64,
    batch_size: int = 4,
    seed: int = 0,
    device: torch.device | str | None = None,
):
    """Build the real paired-reward inputs from warmed train/val paired latent datasets.

    The offline fake-cache builder (ADR-0020): stack the warmed train + val paired
    latent datasets (both **scaled** via ``dataset.scaling_factor`` = the export's
    ``vae.scaling_factor``, ADR-0021), then build the condition-aware real-vs-fake
    pairs over each split + the generated-end probe over the val split. The generator
    is used **once** (the rollout is deterministic - fakes are precomputed, not
    re-rolled); the returned :class:`PairedRewardInputs` carries only precomputed
    pairs - the Module holds no generator (ADR-0020).

    The 2-way split is the caller's responsibility (ADR-0022): pass train/val
    :class:`~manifold.data.PairedLatentDataset` over the **paired** split's train /
    val subjects (resolved via ``_train_val_manifests`` / ``val_data_base_dir`` in
    the CLI), NOT JiT reward's ``partition_subjects`` (different subject-id
    derivation -> silent leak).

    Args:
        train_ds / val_ds: warmed :class:`PairedLatentDataset` (scale-on-read; set
            ``scaling_factor`` to the export's before calling).
        generator: the frozen base UNet (the ControlNet generator's frozen arm).
        base_scheduler: the base :class:`FlowMatchHeunDiscreteScheduler` (the loser
            is a full rollout).
        controlnet: the frozen :class:`~manifold.ControlNet3DConditionModel` — the
            fake source is the ControlNet noise→data rollout (ADR-0027).
        num_steps: rollout Heun budget (a one-time precompute cost, ADR-0020).
        probe_num_steps: probe rollout budget (defaults to ``num_steps``).
        n_probe: max val latents for the generated-end probe.
        batch_size: rollout batch size.

    Returns:
        A :class:`PairedRewardInputs` (train pairs + val pairs + the probe).
    """
    from ..training.paired_reward_cli import PairedRewardInputs

    device = _resolve_rollout_device(generator, device)
    partial_scheduler = PartialFlowMatchHeunScheduler(**base_scheduler.config)
    probe_steps = int(probe_num_steps) if probe_num_steps is not None else int(num_steps)

    x_src_tr, x_tgt_tr, src_lab_tr, tgt_lab_tr, spacing_tr = _stack_paired_latents(train_ds)
    train_pairs = build_paired_reward_pairs(
        x_src_tr,
        x_tgt_tr,
        generator,
        controlnet,
        base_scheduler,
        src_label=src_lab_tr,
        tgt_label=tgt_lab_tr,
        spacing=spacing_tr,
        num_steps=num_steps,
        batch_size=batch_size,
        device=device,
    )
    x_src_va, x_tgt_va, src_lab_va, tgt_lab_va, spacing_va = _stack_paired_latents(val_ds)
    val_pairs = build_paired_reward_pairs(
        x_src_va,
        x_tgt_va,
        generator,
        controlnet,
        base_scheduler,
        src_label=src_lab_va,
        tgt_label=tgt_lab_va,
        spacing=spacing_va,
        num_steps=num_steps,
        batch_size=batch_size,
        device=device,
    )
    n_probe = min(n_probe, len(val_ds))
    probe = build_paired_reward_probe(
        x_src_va[:n_probe],
        x_tgt_va[:n_probe],
        generator,
        controlnet,
        partial_scheduler,
        src_label=src_lab_va[:n_probe],
        tgt_label=tgt_lab_va[:n_probe],
        spacing=spacing_va[:n_probe],
        num_steps=probe_steps,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    rank_zero_info(
        "build_paired_reward_inputs: %d train / %d val pairs + %d probe (num_steps=%d).",
        len(train_pairs),
        len(val_pairs),
        len(probe),
        num_steps,
    )
    return PairedRewardInputs(train_pair_ds=train_pairs, val_pair_ds=val_pairs, val_probe=probe)


__all__ = [
    "build_paired_reward_pairs",
    "build_paired_reward_inputs",
    "build_paired_reward_probe",
]
