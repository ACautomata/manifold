"""Reward Module: Bradley–Terry preference training for the Reward Model (GRPO).

An :class:`stable_pretraining.Module` (``spt.Module``, manual optimization) that
trains the :class:`~manifold.models.RewardModel` with the Bradley–Terry pairwise
preference loss

.. math:: L = -\\log \\sigma(r_w - r_l)

(σ inside the loss only — the model emits raw rewards). The optimizer covers
**discriminator parameters only**.

**Online rollout-in-the-loop training (ADR-0010, issues #48/#50).** The Module
holds the **frozen JiT x0-denoiser** and runs the preference-pair rollout inside
each fit step: ``forward("fit")`` consumes a **clean-latent** batch
(``{latent, spacing, label}``), samples two flow-times ``t_a, t_b ~ U[0,1)`` per
sample, sets ``winner_t = max`` / ``loser_t = min`` (label by *input* corruption
level — ``t→1 = clean``), noises both halves via the scheduler transport, and
issues **one combined ``[2B]`` winner-first** :func:`partial_denoise_rollout`.
Both ``t``'s share the same ``[0,1)`` distribution (one latent can be a winner in
one pair and a loser in another), destroying the single-threshold shortcut that
saturated the offline ``val/pair_acc`` at 0.997 in epoch 0. The frozen denoiser
produces **no gradients**; the discriminator scores the detached rollout outputs
under Bradley–Terry.

``forward("validate")`` still consumes precomputed ``{winner, loser}`` pairs —
validation is rolled once at startup (the denoiser is frozen ⇒ pairs are static
across epochs), not re-rolled every epoch.

``validation_step`` reports:

- **pairwise accuracy** (``r_w > r_l``) and **ROC-AUC** (a threshold-free
  cross-pair summary) for diagnosis, and
- the **generated-end probe** accuracy (``val/gen_pair_acc`` — a fixed pair set
  where *both* samples are drawn from ``t ∈ [0, 0.5)`` and ordered by ``t``) — the
  GRPO-regime metric the checkpoint monitors.

Precision/F1 are intentionally not reported — a fixed threshold is
scale-arbitrary under Bradley–Terry (only ``r_w − r_l`` is constrained, not
absolute ``r``).
"""

from __future__ import annotations

from typing import Any

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.reward_model import RewardModel
from .partial_denoise import partial_denoise_rollout

#: A reward-training batch. In ``fit`` a clean-latent batch (``{latent, ...}``);
#: in ``validate`` a precomputed ``(winner, loser)`` pair (``{winner, loser}``).
RewardBatch = dict[str, Any]


def bradley_terry_loss(reward_winner: Tensor, reward_loser: Tensor) -> Tensor:
    """``L = -log σ(r_w − r_l)`` (mean over the batch of pairs).

    The canonical pairwise-preference objective. Only the *difference*
    ``r_w − r_l`` enters σ, so the loss is invariant to an additive shift of the
    reward — it calibrates reward *differences* to preferences, not absolute
    values (the rejected ``-(log y_w − log y_l)`` form is scale-unbounded under
    Bradley–Terry).
    """
    return -F.logsigmoid(reward_winner - reward_loser).mean()


def reward_roc_auc(reward_winner: Tensor, reward_loser: Tensor) -> Tensor:
    """ROC-AUC over pooled winners (positive) / losers (negative), rank-based.

    Unlike pairwise accuracy (the matched ``r_w_i > r_l_i`` fraction), this
    threshold-free summary ranks every winner against every loser — winners must
    outrank *all* losers, not just their matched one. Computed via the
    Mann–Whitney ``U`` statistic with average ranks for ties:

        AUC = (Σ ranks⁺ − n⁺(n⁺+1)/2) / (n⁺ · n⁻).

    Returns ``0.5`` (neutral) when one class is absent.
    """
    scores = torch.cat([reward_winner.float(), reward_loser.float()])
    # float32 (not ones_like(reward_*)): under Lightning 16-mixed the rewards are
    # fp16, and an fp16 ``labels`` makes ``n_pos``/``n_neg`` fp16 — then
    # ``n_pos*(n_pos+1)/2`` (~5e5) and ``n_pos*n_neg`` (~1e6) overflow fp16's
    # 65504 max to inf, turning the AUC into (finite - inf) / inf = NaN. Counts
    # are dtype-agnostic; build them in float32 so fp16 rewards behave like fp32.
    labels = torch.cat(
        [torch.ones_like(reward_winner, dtype=torch.float32), torch.zeros_like(reward_loser, dtype=torch.float32)]
    )
    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    if float(n_pos) == 0 or float(n_neg) == 0:
        return torch.tensor(0.5)
    # Average ranks (1-based) with ties sharing the mean rank of their group, via a
    # single sort + a linear sweep over tie groups — O(N log N) (not the O(N²)
    # pairwise comparison matrix), so it scales to realistic validation sets.
    order = scores.argsort()
    sorted_scores = scores[order]
    n = scores.numel()
    ranks = torch.empty(n, dtype=torch.float32)
    positions = torch.arange(1, n + 1, dtype=torch.float32)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (positions[i] + positions[j]) / 2.0  # tie group mean rank
        i = j + 1
    sum_pos_ranks = ranks[labels == 1].sum()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _score_pair(reward_model, winner: Tensor, loser: Tensor) -> tuple[Tensor, Tensor]:
    """Forward both halves of a pair in one batch -> ``(r_w, r_l)`` each ``[B]``.

    Concatenating ``[winner, loser]`` runs a single discriminator forward (and lets
    BatchNorm see both halves together); the per-sample rewards are then split back
    into the winner / loser halves. Shared by :class:`RewardModule` (the JiT reward)
    and :class:`~manifold.modules.paired_reward.PairedRewardModule` (the paired
    reward) - the condition-aware paired module passes already-concatenated
    ``[2·C]`` pairs, so the same scorer applies verbatim (ADR-0019).
    """
    batch_size = winner.shape[0]
    rewards = reward_model(torch.cat([winner, loser], dim=0))  # [2B]
    return rewards[:batch_size], rewards[batch_size:]


class RewardModule(spt.Module):
    """Bradley–Terry preference training over the Reward Model.

    Args:
        reward_model: the :class:`~manifold.models.RewardModel` being trained.
        lr: Adam learning rate over the discriminator parameters.
        denoiser: the frozen JiT x0-denoiser (the GRPO starting policy) held for
            the online rollout. Attached **unregistered** (``object.__setattr__``,
            not plain assignment — see :attr:`denoiser`) so it never enters the
            checkpoint / optimizer / DDP replication. ``None`` only when ``fit`` is
            never called (validation-only).
        scheduler: a :class:`~manifold.PartialFlowMatchHeunScheduler` (frozen;
            its transport + per-sample grid run the rollout). Required with
            ``denoiser``.
        num_steps: per-step **train** Heun step budget for the online rollout (the
            rollout-cost lever — 2 vs 4 Heun evals; ADR-0010).
        winner_t_guard: optional upper cap on the winner's ``t_start``. A
            **pair-difficulty knob, NOT numerical safety**: ``torch.rand``'s
            half-open ``[0,1)`` already precludes ``t == 1`` (the only true NaN
            source), and on the uniform partial grid ``v1·dt`` cancels the
            ``1/(1−t)`` step-start denominator, so the update is bounded by
            ``(x0 − z)`` independent of ``t_start``. Capping the winner lower makes
            pairs harder (winner more corrupted); it is not a load-bearing cap.
        val_probe: optional fixed ``(winner, loser)`` generated-end probe tensors
            (both samples ``t ∈ [0, 0.5)``, ordered by ``t``). Held but **not**
            registered; scored once per validation epoch.
        probe_batch_size: probe scoring chunk (the fixed probe is scored in
            batches this size at epoch end).
    """

    def __init__(
        self,
        reward_model: RewardModel,
        *,
        lr: float = 1.0e-4,
        denoiser: torch.nn.Module | None = None,
        scheduler=None,
        num_steps: int = 4,
        winner_t_guard: float | None = None,
        val_probe: tuple[Tensor, Tensor] | None = None,
        probe_batch_size: int = 8,
    ):
        # NOTE: forward is NOT passed to spt.Module — it would double-bind self.
        super().__init__(hparams={"lr": lr, "num_steps": num_steps})
        self.reward_model = reward_model
        self.lr = float(lr)
        self.num_steps = int(num_steps)
        self.winner_t_guard = float(winner_t_guard) if winner_t_guard is not None else None
        if denoiser is not None:
            if scheduler is None:
                raise ValueError("denoiser requires a scheduler for the online rollout.")
            denoiser = denoiser.eval()
            for p in denoiser.parameters():
                p.requires_grad_(False)
            # object.__setattr__ bypasses nn.Module.__setattr__, which would
            # auto-register the denoiser into _modules → leaking its 3.6 GB into
            # state_dict(), parameters(), the optimizer, and DDP replication. Kept
            # off the books: absent from the checkpoint and the optimizer, and moved
            # to the device manually in on_fit_start (the bypass also hides it from
            # Lightning's automatic .to(device), so it would otherwise stay where
            # from_pretrained left it). The scheduler is NOT an nn.Module, so plain
            # assignment does not register it.
            object.__setattr__(self, "denoiser", denoiser)
            self.scheduler = scheduler
        else:
            self.denoiser = None
            self.scheduler = None
        #: Fixed generated-end probe (winner, loser); set post-construction by
        #: :meth:`set_val_probe` or the ctor. A plain attribute (not a buffer) so
        #: it is device-moved manually and stays out of the checkpoint.
        self.val_probe: tuple[Tensor, Tensor] | None = None
        if val_probe is not None:
            self.set_val_probe(*val_probe)
        #: Probe scoring chunk — the fixed probe is scored in batches this size at
        #: epoch end (not one giant forward), so a small training batch_size does
        #: not OOM validation. Defaults small; run_reward_training threads the
        #: training batch_size through.
        self.probe_batch_size = int(probe_batch_size)
        #: Per-batch validation rewards accumulated for the pooled (cross-batch)
        #: ROC-AUC — a rank statistic, so per-batch averaging would be wrong.
        #: Reset each validation epoch in :meth:`on_validation_epoch_start`.
        self._val_r_w: list[Tensor] = []
        self._val_r_l: list[Tensor] = []

    # -- frozen-denoiser lifecycle -------------------------------------------

    def on_fit_start(self) -> None:
        """Move the unregistered frozen denoiser to the device (Lightning won't).

        The ``object.__setattr__`` bypass keeps the denoiser off Lightning's books,
        so its automatic ``.to(device)`` skips it. The real path moves it already
        (``load_frozen_denoiser`` → ``.to(device)``); this is the safety net for
        direct ``fit`` calls and the resume path (the checkpoint holds no denoiser,
        so it is re-read from ``--native-dir``).
        """
        if self.denoiser is not None:
            self.denoiser.to(self.device)

    def set_val_probe(self, winner: Tensor, loser: Tensor) -> None:
        """Attach the fixed generated-end probe pair set (reused across epochs)."""
        self.val_probe = (winner, loser)

    # -- scoring --------------------------------------------------------------

    def _score_pair(self, winner: Tensor, loser: Tensor) -> tuple[Tensor, Tensor]:
        """Forward both halves of the pair in one batch → ``(r_w, r_l)`` each ``[B]``.

        Concatenating ``[winner, loser]`` runs a single discriminator forward
        (and lets BatchNorm see both halves together); the per-sample rewards are
        then split back into the winner / loser halves. Thin delegate over the
        module-level :func:`_score_pair` (shared with the paired reward module).
        """
        return _score_pair(self.reward_model, winner, loser)

    def _online_rollout(self, batch: RewardBatch) -> tuple[Tensor, Tensor]:
        """Per-step online preference-pair rollout: clean latents → ``(winner, loser)``.

        For each clean latent draw ``t_a, t_b ~ U[0,1)``; the larger is the
        winner's start (less corrupted — ``t→1 = clean``), the smaller the loser's.
        Both halves are noised and partial-denoised in **one combined ``[2B]``
        winner-first** rollout call (per-sample ``spacing``/``modality`` duplicated
        to the combined size); the discriminator scores the detached outputs. The
        label follows INPUT corruption level (cheap, annotation-free); both
        ``t``'s share the ``[0,1)`` distribution, so one latent can be a winner in
        one pair and a loser in another — the de-saturation property (ADR-0010).
        """
        clean = batch["latent"]
        b = clean.shape[0]
        # half-open U[0,1) ⇒ t < 1 always (the step-start denominator 1 − t never
        # vanishes — torch.rand is half-open). Reproducible: run_reward_training
        # seeds via pl.seed_everything; the eval-mode denoiser consumes no RNG, so
        # the only draws are these t's and the noise.
        t_a = torch.rand(b, device=clean.device)
        t_b = torch.rand(b, device=clean.device)
        winner_t = torch.maximum(t_a, t_b)
        loser_t = torch.minimum(t_a, t_b)
        # Difficulty knob only (NOT numerical safety — see the ctor docstring). A
        # cap that would invert the winner/loser ordering (both draws > guard) is
        # resolved to a tie at loser_t, preserving winner ≥ loser.
        if self.winner_t_guard is not None:
            winner_t = torch.maximum(winner_t.clamp(max=self.winner_t_guard), loser_t)
        # Combined [2B], winner-first: the rollout's reward split must match.
        clean_2b = torch.cat([clean, clean], dim=0)
        noise_2b = torch.randn(clean_2b.shape, device=clean.device)
        t_start = torch.cat([winner_t, loser_t], dim=0)
        spacing = batch["spacing"]
        spacing_2b = (
            torch.cat([spacing, spacing], dim=0)
            if (isinstance(spacing, Tensor) and spacing.dim() == 2)
            else spacing
        )
        modality = batch["label"]
        modality_2b = torch.cat([modality, modality], dim=0) if isinstance(modality, Tensor) else modality
        z_2b = self.scheduler.add_noise(clean_2b, noise_2b, t_start)
        out = partial_denoise_rollout(
            self.denoiser, self.scheduler, z_2b, t_start, spacing_2b, modality_2b, num_steps=self.num_steps
        )
        return out[:b], out[b:]

    def forward(self, batch: RewardBatch, stage: str) -> dict[str, Tensor]:
        """Bradley–Terry loss in fit (online rollout over clean latents); metrics in validate.

        ``stage == "fit"``: ``batch`` is a **clean-latent** batch
        (``{latent, spacing, label}``); run the online rollout, score, and return
        ``{"loss": BT loss}`` (spt's ``training_step`` runs ``manual_backward``
        over it, stepping the discriminator optimizer).
        ``stage == "validate"`` (spt's ``validation_step`` calls this under
        ``no_grad``): ``batch`` is a precomputed ``{winner, loser}`` pair; log
        pairwise accuracy and stash the rewards for the pooled ROC-AUC at epoch end.
        A clean-latent batch routed to validate (or a pair batch to fit) raises.
        """
        if stage == "fit":
            if "latent" not in batch:
                raise ValueError(
                    "fit stage expects a clean-latent batch with key 'latent' "
                    "({latent, spacing, label}); got a pair batch — route pairs to validate."
                )
            winner, loser = self._online_rollout(batch)
            r_w, r_l = self._score_pair(winner, loser)
            return {"loss": bradley_terry_loss(r_w, r_l)}
        if stage == "validate":
            if "winner" not in batch or "loser" not in batch:
                raise ValueError(
                    "validate stage expects a {winner, loser} pair batch; got a "
                    "clean-latent batch — route clean latents to fit."
                )
            r_w, r_l = self._score_pair(batch["winner"], batch["loser"])
            # sync_dist: under DDP, all-reduce so ModelCheckpoint sees a global metric
            # (not rank 0's shard). pair_acc is a linear mean → synced averaging is exact.
            self.log("val/pair_acc", (r_w > r_l).float().mean(), on_epoch=True, prog_bar=True, sync_dist=True)
            self._val_r_w.append(r_w.detach().cpu())
            self._val_r_l.append(r_l.detach().cpu())
            return {"r_w": r_w, "r_l": r_l}
        raise ValueError(f"unknown stage {stage!r}; use 'fit' or 'validate'.")

    def on_validation_epoch_start(self) -> None:
        """Reset the per-epoch reward accumulators (pooled ROC-AUC scratch)."""
        self._val_r_w.clear()
        self._val_r_l.clear()

    @staticmethod
    def _gather_global(local: Tensor) -> Tensor:
        """All-gather a 1-D reward tensor across DDP ranks (no-op single-process).

        ROC-AUC is a rank statistic over the WHOLE validation set; each rank only
        sees its DistributedSampler shard, so the rewards must be gathered before
        the global AUC is computed.
        """
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return local
        world = dist.get_world_size()
        gathered: list = [None] * world
        dist.all_gather_object(gathered, local.detach().cpu())
        return torch.cat([torch.as_tensor(x) for x in gathered if x is not None])

    def on_validation_epoch_end(self) -> None:
        """Pooled ROC-AUC over the whole validation set + the fixed generated-end probe.

        ROC-AUC is a rank statistic, so it is computed once over ALL validation
        rewards (gathered across DDP ranks), not per-batch-averaged. The probe
        (both samples ``t ∈ [0, 0.5)``, ordered by ``t``) is precomputed (frozen
        denoiser) and reused across epochs; it is scored in ``probe_batch_size``
        chunks (not one forward) to avoid an epoch-end OOM. The checkpoint monitors
        ``val/gen_pair_acc`` (the GRPO-regime metric).
        """
        if self._val_r_w:
            r_w = self._gather_global(torch.cat(self._val_r_w))
            r_l = self._gather_global(torch.cat(self._val_r_l))
            self.log("val/roc_auc", reward_roc_auc(r_w, r_l))  # already global (gathered across ranks)
        if self.val_probe is not None:
            winner, loser = self.val_probe
            probe_w = winner.to(self.device)
            probe_l = loser.to(self.device)
            with torch.no_grad():
                # Score the probe in chunks (not one giant forward) to bound memory.
                chunk = max(1, self.probe_batch_size)
                accs = []
                for s in range(0, probe_w.shape[0], chunk):
                    pr_w = self.reward_model(probe_w[s : s + chunk])
                    pr_l = self.reward_model(probe_l[s : s + chunk])
                    accs.append((pr_w > pr_l).float())
            self.log("val/gen_pair_acc", torch.cat(accs).mean(), sync_dist=True)

    def configure_optimizers(self):
        """Adam over discriminator parameters only (the reward model).

        The frozen denoiser is unregistered (``object.__setattr__``), so it is
        absent from ``reward_model.parameters()`` and the optimizer never touches
        it — the discriminator-only-gradient invariant holds structurally.
        """
        return {"optimizer": torch.optim.Adam(self.reward_model.parameters(), lr=self.lr)}
