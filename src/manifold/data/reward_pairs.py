"""Offline preference-pair generation + the ``RewardPairDataset`` (GRPO reward).

Builds ``(winner, loser)`` latent pairs by noising a clean VAE latent to a
per-sample flow-time ``t`` and denoising back to clean with the **frozen** JiT
denoiser: the **winner** is lightly corrupted (``t_w ~ U[0.5, 1)`` — near-clean),
the **loser** heavily corrupted (``t_l ~ U[0, 0.5)`` — near-noise), both denoised
with the same step budget. ``t`` uses the scheduler's ``t → 1 = clean`` convention,
so "more noise" is always *smaller* ``t``. Cached into a
:class:`RewardPairDataset` with a held-out-**subject** validation split.

Pairs are built **once** (the denoiser is frozen, so they are static across
epochs) — reward training never runs a per-step denoiser rollout. The noised
start comes from the scheduler's transport ``z = t·x + (1−t)·e`` verbatim
(:func:`partial_denoise_rollout`'s contract, ADR-0001), so the denoiser sees
exactly its training distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..modules.partial_denoise import partial_denoise_rollout
from ..schedulers.scheduling_partial_flow_match_heun import PartialFlowMatchHeunScheduler

_log = logging.getLogger(__name__)

#: Winner corruption: light, near-clean (the scheduler's t→1 = clean convention).
WINNER_T_RANGE: tuple[float, float] = (0.5, 1.0)
#: Loser corruption: heavy, near-noise.
LOSER_T_RANGE: tuple[float, float] = (0.0, 0.5)

#: File names inside the output directory.
_PAIRS_FILE = "pairs.pt"
_VAL_FILE = "val_pairs.pt"
_PROBE_FILE = "probe_pairs.pt"


def _sample_t(
    low: float, high: float, batch_size: int, generator: torch.Generator, device=None
) -> Tensor:
    """``t ~ U[low, high)`` per sample (half-open — never ``high``, avoiding ``t=1``).

    ``torch.rand`` is half-open ``[0, 1)``, so scaling keeps ``t < high``: the
    winner never samples ``t = 1`` (where the step-start denominator ``1 − t``
    would vanish) and the loser never samples ``t = 0.5`` exactly. *device* is
    passed explicitly so a GPU/MPS ``generator`` allocates its output there (an
    inferred device off a non-CPU generator raises on MPS).
    """
    return low + (high - low) * torch.rand(batch_size, generator=generator, device=device)


class RewardPairDataset(Dataset):
    """A precomputed set of ``(winner, loser)`` latent preference pairs.

    Stores the winner / loser latents as stacked tensors ``[N, C, D, H, W]``;
    ``__getitem__`` emits the ``{"winner", "loser"}`` dict the
    :class:`~manifold.modules.RewardModule` consumes.
    """

    def __init__(self, winners: Tensor, losers: Tensor) -> None:
        if winners.shape != losers.shape:
            raise ValueError(
                f"winner/loser shapes must match, got {tuple(winners.shape)} vs {tuple(losers.shape)}."
            )
        self.winners = winners
        self.losers = losers

    def __len__(self) -> int:
        return int(self.winners.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        return {"winner": self.winners[index], "loser": self.losers[index]}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"winners": self.winners, "losers": self.losers}, path)

    @classmethod
    def load(cls, path: str | Path) -> "RewardPairDataset":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return cls(blob["winners"], blob["losers"])


def load_frozen_denoiser(native_dir: str | Path):
    """Load the frozen JiT denoiser (the GRPO starting policy) from a native export.

    The native dir is the ADR-0006 layout written by
    :meth:`~manifold.LatentFlowPipeline.save_pretrained` /
    :func:`~manifold.training.export_to_native`; the UNet is the trained JiT
    x0-denoiser (one source of truth). The scheduler config is read back as a
    :class:`PartialFlowMatchHeunScheduler` so its ``t_eps`` matches training, and
    the VAE's ``scaling_factor`` is returned so callers can scale raw cached
    latents into the denoiser's training space (the latent cache stores unscaled
    latents — scale-on-read happens at ``__getitem__``).
    """
    from ..pipelines.latent_flow import LatentFlowPipeline

    pipe = LatentFlowPipeline.from_pretrained(str(native_dir))
    # Re-instantiate the scheduler as the partial subclass (same ctor signature /
    # t_eps), so pair generation shares the JiT endpoint clamp exactly.
    scheduler = PartialFlowMatchHeunScheduler(**pipe.scheduler.config)
    scaling_factor = float(pipe.vae.scaling_factor)
    pipe.unet.eval()
    for p in pipe.unet.parameters():
        p.requires_grad_(False)
    return pipe.unet, scheduler, scaling_factor


def generate_reward_pairs(
    clean_latents: Tensor | Sequence[Tensor],
    subject_ids: Sequence[str],
    denoiser,
    scheduler: PartialFlowMatchHeunScheduler,
    *,
    spacing: Sequence[float] | Tensor,
    modality: int | Sequence[int] | Tensor,
    num_steps: int,
    val_fraction: float = 0.2,
    batch_size: int = 4,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[RewardPairDataset, RewardPairDataset]:
    """Build ``(train, val)`` preference pairs by partial-denoise rollout.

    For each clean latent, sample ``t_w ~ U[0.5, 1)`` / ``t_l ~ U[0, 0.5)``, noise
    via the scheduler transport, and denoise both with the frozen denoiser under a
    shared step budget. Per-sample ``t`` lets a whole batch (each sample at its own
    corruption level) denoise in one rollout. Held-out-**subject** split: unique
    subjects are partitioned once (seeded) into train/val, so validation measures
    generalization, not memorization.

    Args:
        clean_latents: ``[N, C, D, H, W]`` clean VAE latents (already scaled to the
            denoiser's training space — apply ``vae.scaling_factor`` to raw cache
            latents before passing them in).
        subject_ids: length-``N`` **subject** id per latent (the split key — derive
            a true per-subject id, not the per-contrast cache sample_id).
        denoiser: the frozen JiT denoiser (e.g. from :func:`load_frozen_denoiser`).
        scheduler: a :class:`PartialFlowMatchHeunScheduler` (transport + steps).
        spacing: voxel spacing — ``[3]`` (broadcast) or ``[N, 3]`` (per-sample, for
            a heterogeneous cache).
        modality: integer class label (broadcast) or a length-``N`` per-sample
            sequence/tensor (a multi-contrast cache).
        num_steps: shared Heun step budget for winner and loser rollouts.
        val_fraction: fraction of unique **subjects** held out for validation.

    Returns:
        ``(train_ds, val_ds)`` — both :class:`RewardPairDataset`.
    """
    clean = clean_latents if isinstance(clean_latents, Tensor) else torch.stack(list(clean_latents))
    if len(clean) != len(subject_ids):
        raise ValueError(f"clean_latents ({len(clean)}) and subject_ids ({len(subject_ids)}) must align.")
    device = torch.device(device) if device is not None else next(denoiser.parameters()).device

    # Normalise per-sample conditioning to tensors before the batch loop — the
    # caller may pass Python sequences (documented), and list-slicing before
    # ``int(modality_b)`` or ``isinstance(modality,Tensor)`` inside the denoising
    # path would crash (Codex #45).
    if not isinstance(spacing, Tensor):
        spacing = torch.as_tensor(spacing)
    if not isinstance(modality, (int, float, Tensor)):
        modality = torch.as_tensor(modality)

    unique = sorted(set(subject_ids))
    g_split = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique), generator=g_split).tolist()
    n_val = max(1, min(len(unique) - 1, int(round(len(unique) * val_fraction)))) if len(unique) > 1 else 0
    val_subjects = {unique[i] for i in perm[:n_val]}

    # Device-aware generator: torch.randn(..., generator=gen, device=device) below
    # requires the generator to live on `device` (a CPU generator raises on CUDA).
    gen = torch.Generator(device=device).manual_seed(seed)
    tr_w, tr_l, va_w, va_l = [], [], [], []
    n = len(clean)
    for start in range(0, n, batch_size):
        batch = clean[start : start + batch_size].to(device)
        b = batch.shape[0]
        t_w = _sample_t(*WINNER_T_RANGE, b, gen, device=device)
        t_l = _sample_t(*LOSER_T_RANGE, b, gen, device=device)
        noise_w = torch.randn(batch.shape, generator=gen, device=device)
        noise_l = torch.randn(batch.shape, generator=gen, device=device)
        z_w = scheduler.add_noise(batch, noise_w, t_w)
        z_l = scheduler.add_noise(batch, noise_l, t_l)
        # Per-sample conditioning: spacing may be [3] (broadcast) or [N,3]; modality
        # may be an int (broadcast) or a length-N sequence/tensor (multi-contrast).
        spacing_b = spacing[start : start + b] if isinstance(spacing, Tensor) and spacing.dim() == 2 else spacing
        modality_b = modality[start : start + b] if not isinstance(modality, (int, float)) else modality
        winners = partial_denoise_rollout(
            denoiser, scheduler, z_w, t_w, spacing_b, modality_b, num_steps=num_steps
        )
        losers = partial_denoise_rollout(
            denoiser, scheduler, z_l, t_l, spacing_b, modality_b, num_steps=num_steps
        )
        for j in range(b):
            sid = subject_ids[start + j]
            pair = (winners[j].detach().cpu(), losers[j].detach().cpu())
            if sid in val_subjects:
                va_w.append(pair[0])
                va_l.append(pair[1])
            else:
                tr_w.append(pair[0])
                tr_l.append(pair[1])

    _log.info(
        "generate_reward_pairs: %d train / %d val pairs (%d val subjects of %d).",
        len(tr_w), len(va_w), n_val, len(unique),
    )
    if not tr_w or not va_w:
        # Fall back to a pair-split if there are too few subjects for a subject split.
        all_w, all_l = torch.stack(tr_w + va_w), torch.stack(tr_l + va_l)
        idx = torch.randperm(len(all_w), generator=torch.Generator().manual_seed(seed)).tolist()
        n_v = max(1, int(round(len(all_w) * val_fraction)))
        vi, ti = set(idx[:n_v]), [i for i in range(len(all_w)) if i not in set(idx[:n_v])]
        return RewardPairDataset(all_w[ti], all_l[ti]), RewardPairDataset(all_w[list(vi)], all_l[list(vi)])
    return RewardPairDataset(torch.stack(tr_w), torch.stack(tr_l)), RewardPairDataset(
        torch.stack(va_w), torch.stack(va_l)
    )


def save_reward_pairs(
    output_dir: str | Path,
    train: RewardPairDataset,
    val: RewardPairDataset,
    probe: RewardPairDataset | None = None,
) -> None:
    """Write the train/val (+ optional probe) pair datasets to ``output_dir``.

    Files: ``pairs.pt``, ``val_pairs.pt``, and (if *probe*) ``probe_pairs.pt``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.save(output_dir / _PAIRS_FILE)
    val.save(output_dir / _VAL_FILE)
    if probe is not None:
        probe.save(output_dir / _PROBE_FILE)


def load_reward_pairs(
    output_dir: str | Path,
) -> tuple[RewardPairDataset, RewardPairDataset, RewardPairDataset | None]:
    """Load ``(train, val, probe)`` written by :func:`save_reward_pairs`.

    The probe is ``None`` if ``probe_pairs.pt`` is absent (the generated-end probe
    is optional — present only when the generation script wrote it).
    """
    output_dir = Path(output_dir)
    probe_path = output_dir / _PROBE_FILE
    probe = RewardPairDataset.load(probe_path) if probe_path.is_file() else None
    return (
        RewardPairDataset.load(output_dir / _PAIRS_FILE),
        RewardPairDataset.load(output_dir / _VAL_FILE),
        probe,
    )


# -- online real-data wiring (issues #48/#51) --------------------------------


def _subject_id(sample_id: str, regex: str | None = None) -> str:
    """Derive a true subject id from a cache ``sample_id`` / file stem.

    BraTS contrast files of one subject differ in the trailing contrast token
    (``BraTS-GLI-0000-000-t1n`` vs ``-t1c``); the full id would split one subject
    across train/val. By default the last ``-<token>`` is stripped (BraTS
    convention). ``regex`` (one capture group) overrides for other naming.
    """
    import re

    if regex:
        m = re.match(regex, sample_id)
        return m.group(1) if (m and m.groups()) else sample_id
    return sample_id.rsplit("-", 1)[0] if "-" in sample_id else sample_id


def load_cached_latents(
    latents_dir: str | Path, subject_regex: str | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load cached ``.pt`` latent items (**unscaled**) + subject ids from a cache dir.

    Shared by the offline generation script and the online real-data path. Items
    are latent-cache dicts ``{latent (unscaled), spacing, label, sample_id}``; a
    bare-tensor file yields ``spacing``/``label`` absent (the consumer supplies
    defaults). The subject id is the dict's ``sample_id`` if present, else the file
    stem, grouped via :func:`_subject_id`. **Scaling is applied by the consumer**
    (:class:`CleanLatentDataset` scale-on-read for train; pre-scaled tensor for
    val/probe gen) — never here, so the cache stays scale-independent (ADR-0003).
    """
    latents_dir = Path(latents_dir)
    files = sorted(p for p in latents_dir.glob("*.pt") if ".tmp.r" not in p.name)
    if not files:
        raise FileNotFoundError(f"No .pt latents under {latents_dir}.")
    items: list[dict[str, Any]] = []
    subject_ids: list[str] = []
    for path in files:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(blob, dict) and "latent" in blob:
            sid_raw = str(blob.get("sample_id", path.stem))
            items.append(
                {
                    "latent": blob["latent"].float(),
                    "spacing": blob.get("spacing"),
                    "label": blob.get("label"),
                    "sample_id": sid_raw,
                }
            )
        else:
            sid_raw = path.stem
            items.append({"latent": blob.float(), "spacing": None, "label": None, "sample_id": sid_raw})
        subject_ids.append(_subject_id(sid_raw, subject_regex))
    return items, subject_ids


def partition_subjects(
    subject_ids: Sequence[str], *, val_fraction: float = 0.2, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Seeded held-out-**subject** split → ``(train_subjects, val_subjects)`` sets.

    A subject's contrasts (multiple cache items sharing one subject id) land wholly
    in train or val — no subject spans both, so validation measures generalization
    and the discriminator never trains on a validation subject (the leakage guard
    is enforced at the train-dataloader construction, issue #51).
    """
    unique = sorted(set(subject_ids))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(unique), generator=g).tolist()
    n_val = max(1, min(len(unique) - 1, int(round(len(unique) * val_fraction)))) if len(unique) > 1 else 0
    val_subjects = {unique[i] for i in perm[:n_val]}
    return set(unique) - val_subjects, val_subjects


def maybe_per_sample(items: Sequence[Any], fallback: Any) -> Any:
    """Per-sample tensor if every item is present, else the broadcast fallback.

    A length-``N`` sequence of per-sample conditioning (spacing / label) becomes a
    stacked ``[N, ...]`` tensor when all entries are present; any ``None`` (a
    bare-tensor cache file lacking that field) collapses to the scalar/sequence
    fallback used across the whole batch.
    """
    if len(items) > 0 and all(it is not None for it in items):
        return torch.stack([torch.as_tensor(it) for it in items])
    return fallback


class CleanLatentDataset(Dataset):
    """In-RAM clean-latent dataset (scale-on-read) for online reward training.

    Reads warmed latent-cache items (**unscaled**) and emits
    ``{latent (scaled), spacing, label, sample_id}`` — the clean-latent batch the
    :class:`~manifold.modules.RewardModule` fit-step online rollout consumes. The
    ``scale_factor`` (the native VAE's) is applied exactly once on read
    (ADR-0003), so the frozen denoiser sees its training space. Restricted to a
    subject subset (train subjects) at construction — the held-out-subject split is
    enforced here, at the data level (no validation subject ever reaches fit).
    """

    _DEFAULT_SPACING = torch.tensor([1.0, 1.0, 1.0])

    def __init__(self, items: list[dict[str, Any]], scaling_factor: float = 1.0) -> None:
        self.items = items
        self.scaling_factor = float(scaling_factor)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        latent = item["latent"].float() * self.scaling_factor  # scale-on-read (once)
        label = item["label"]
        if not torch.is_tensor(label):
            label = torch.tensor(int(label) if label is not None else 0, dtype=torch.long)
        spacing = item["spacing"]
        if spacing is None:
            spacing = self._DEFAULT_SPACING
        out: dict[str, Any] = {
            "latent": latent,
            "spacing": torch.as_tensor(spacing, dtype=torch.float32),
            "label": label,
        }
        if "sample_id" in item:
            out["sample_id"] = item["sample_id"]
        return out


def _generate_ordered_pairs(
    clean_latents: Tensor | Sequence[Tensor],
    denoiser,
    scheduler: PartialFlowMatchHeunScheduler,
    *,
    t_range: tuple[float, float],
    spacing: Sequence[float] | Tensor,
    modality: int | Sequence[int] | Tensor,
    num_steps: int,
    batch_size: int = 4,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> RewardPairDataset:
    """Ordered preference pairs: ``t_a, t_b ~ U[t_range]``; ``winner_t = max``, ``loser_t = min``.

    Shared by :func:`generate_generated_end_probe` (``t_range = [0, 0.5)`` — the
    generated regime) and :func:`generate_full_range_val_pairs`
    (``t_range = [0, 1)`` — mirroring the online train distribution). Both halves
    are noised via the scheduler transport and partial-denoised with the shared
    budget; the winner is the less-corrupted (higher ``t``) sample. ``torch.rand``
    is half-open, so ``high`` is never sampled (``t == 1`` is precluded).
    """
    clean = clean_latents if isinstance(clean_latents, Tensor) else torch.stack(list(clean_latents))
    device = torch.device(device) if device is not None else next(denoiser.parameters()).device

    # Normalise per-sample conditioning (same as in generate_reward_pairs).
    if not isinstance(spacing, Tensor):
        spacing = torch.as_tensor(spacing)
    if not isinstance(modality, (int, float, Tensor)):
        modality = torch.as_tensor(modality)

    gen = torch.Generator(device=device).manual_seed(seed)  # device-aware (CPU gen raises on CUDA)
    winners, losers = [], []
    for start in range(0, len(clean), batch_size):
        batch = clean[start : start + batch_size].to(device)
        b = batch.shape[0]
        t_a = _sample_t(*t_range, b, gen, device=device)
        t_b = _sample_t(*t_range, b, gen, device=device)
        winner_t = torch.maximum(t_a, t_b)
        loser_t = torch.minimum(t_a, t_b)
        noise_w = torch.randn(batch.shape, generator=gen, device=device)
        noise_l = torch.randn(batch.shape, generator=gen, device=device)
        z_w = scheduler.add_noise(batch, noise_w, winner_t)
        z_l = scheduler.add_noise(batch, noise_l, loser_t)
        spacing_b = spacing[start : start + b] if isinstance(spacing, Tensor) and spacing.dim() == 2 else spacing
        modality_b = modality[start : start + b] if not isinstance(modality, (int, float)) else modality
        winners.append(
            partial_denoise_rollout(denoiser, scheduler, z_w, winner_t, spacing_b, modality_b, num_steps=num_steps)
            .detach()
            .cpu()
        )
        losers.append(
            partial_denoise_rollout(denoiser, scheduler, z_l, loser_t, spacing_b, modality_b, num_steps=num_steps)
            .detach()
            .cpu()
        )
    return RewardPairDataset(torch.cat(winners), torch.cat(losers))


def generate_generated_end_probe(
    clean_latents: Tensor | Sequence[Tensor],
    denoiser,
    scheduler: PartialFlowMatchHeunScheduler,
    *,
    spacing: Sequence[float] | Tensor,
    modality: int | Sequence[int] | Tensor,
    num_steps: int,
    batch_size: int = 4,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> RewardPairDataset:
    """Build the generated-end probe: both samples from ``t ∈ [0, 0.5)``, ordered by ``t``.

    Distinct from the reconstruction pairs (winner near-clean, loser near-noise):
    here **both** samples are heavily corrupted (``t ∈ [0, 0.5)``), with the
    *winner* the less-corrupted one (higher ``t``). This directly tests whether the
    reward — calibrated on the full-range online train distribution — ranks quality
    *within the all-generated regime GRPO actually operates in*. This is the
    GRPO-regime metric the checkpoint monitors (``val/gen_pair_acc``).
    """
    probe = _generate_ordered_pairs(
        clean_latents, denoiser, scheduler,
        t_range=(0.0, 0.5), spacing=spacing, modality=modality, num_steps=num_steps,
        batch_size=batch_size, seed=seed, device=device,
    )
    _log.info("generate_generated_end_probe: %d probe pairs (both t ∈ [0, 0.5)).", len(probe))
    return probe


def generate_full_range_val_pairs(
    clean_latents: Tensor | Sequence[Tensor],
    denoiser,
    scheduler: PartialFlowMatchHeunScheduler,
    *,
    spacing: Sequence[float] | Tensor,
    modality: int | Sequence[int] | Tensor,
    num_steps: int,
    batch_size: int = 4,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> RewardPairDataset:
    """Full-range ``[0, 1)`` ordered validation pairs mirroring the online train distribution.

    For each clean latent two ``t``'s are drawn ``~ U[0, 1)``; the larger is the
    winner's start, the smaller the loser's — exactly the online fit-step rollout's
    mechanic (ADR-0010). So ``val/pair_acc`` reflects the new (de-saturated)
    train distribution for diagnosis, instead of the old disjoint
    ``[0.5, 1) / [0, 0.5)`` halves that a single clean-ness threshold could
    trivially separate. Validation is rolled once at startup (the denoiser is
    frozen ⇒ these pairs are static across epochs), not re-rolled every epoch.
    """
    val = _generate_ordered_pairs(
        clean_latents, denoiser, scheduler,
        t_range=(0.0, 1.0), spacing=spacing, modality=modality, num_steps=num_steps,
        batch_size=batch_size, seed=seed, device=device,
    )
    _log.info("generate_full_range_val_pairs: %d full-range val pairs.", len(val))
    return val
