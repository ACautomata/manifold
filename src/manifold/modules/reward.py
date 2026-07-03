"""Reward Module: Bradley–Terry preference training for the Reward Model (GRPO).

An :class:`stable_pretraining.Module` (``spt.Module``, manual optimization) that
trains the :class:`~manifold.models.RewardModel` on **precomputed** ``(winner,
loser)`` preference pairs with the Bradley–Terry pairwise preference loss

.. math:: L = -\\log \\sigma(r_w - r_l)

(σ inside the loss only — the model emits raw rewards). The optimizer covers
**discriminator parameters only**. Pairs are pre-made offline (the denoiser is
frozen, so pairs are static); the Module therefore holds **no** JiT denoiser.

``validation_step`` reports:

- **pairwise accuracy** (``r_w > r_l``, the primary metric — exactly what GRPO
  cares about: does the reward rank correctly),
- **ROC-AUC** (a threshold-free classification summary over all winner/loser
  cross-pairs, not just the matched pairs), and
- the **generated-end probe** accuracy (a fixed pair set where *both* samples are
  drawn from ``t ∈ [0, 0.5]`` and ordered by ``t``, directly testing ranking
  within the all-generated regime GRPO operates in).

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

#: A reward-training batch: a precomputed (winner, loser) latent pair. The data
#: stack (a ``RewardPairDataset``) produces these; this module only consumes them.
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
    labels = torch.cat([torch.ones_like(reward_winner), torch.zeros_like(reward_loser)])
    n_pos = labels.sum()
    n_neg = labels.numel() - n_pos
    if float(n_pos) == 0 or float(n_neg) == 0:
        return torch.tensor(0.5)
    # Average ranks (1-based) with ties sharing the mean rank of their group.
    less = (scores[None, :] < scores[:, None]).sum(dim=1)
    equal = (scores[None, :] == scores[:, None]).sum(dim=1)
    ranks = 1.0 + less.float() + (equal - 1).float() / 2.0
    sum_pos_ranks = ranks[labels == 1].sum()
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


class RewardModule(spt.Module):
    """Bradley–Terry preference training over the Reward Model.

    Args:
        reward_model: the :class:`~manifold.models.RewardModel` being trained.
        lr: Adam learning rate over the discriminator parameters.
        val_probe: optional fixed ``(winner, loser)`` generated-end probe tensors
            (both samples drawn from ``t ∈ [0, 0.5]``, ordered by ``t``). Held but
            **not** registered (excluded from the checkpoint / optimizer); scored
            once per validation epoch to test ranking within the all-generated
            regime GRPO operates in.
    """

    def __init__(
        self,
        reward_model: RewardModel,
        *,
        lr: float = 1.0e-4,
        val_probe: tuple[Tensor, Tensor] | None = None,
    ):
        # NOTE: forward is NOT passed to spt.Module — it would double-bind self.
        super().__init__(hparams={"lr": lr})
        self.reward_model = reward_model
        self.lr = float(lr)
        #: Fixed generated-end probe (winner, loser); set post-construction by
        #: :meth:`set_val_probe` or the ctor. A plain attribute (not a buffer) so
        #: it is device-moved manually and stays out of the checkpoint.
        self.val_probe: tuple[Tensor, Tensor] | None = None
        if val_probe is not None:
            self.set_val_probe(*val_probe)
        #: Per-batch validation rewards accumulated for the pooled (cross-batch)
        #: ROC-AUC — a rank statistic, so per-batch averaging would be wrong.
        #: Reset each validation epoch in :meth:`on_validation_epoch_start`.
        self._val_r_w: list[Tensor] = []
        self._val_r_l: list[Tensor] = []

    def set_val_probe(self, winner: Tensor, loser: Tensor) -> None:
        """Attach the fixed generated-end probe pair set (reused across epochs)."""
        self.val_probe = (winner, loser)

    def _score_pair(self, winner: Tensor, loser: Tensor) -> tuple[Tensor, Tensor]:
        """Forward both halves of the pair in one batch → ``(r_w, r_l)`` each ``[B]``.

        Concatenating ``[winner, loser]`` runs a single discriminator forward
        (and lets BatchNorm see both halves together); the per-sample rewards are
        then split back into the winner / loser halves.
        """
        batch_size = winner.shape[0]
        rewards = self.reward_model(torch.cat([winner, loser], dim=0))  # [2B]
        return rewards[:batch_size], rewards[batch_size:]

    def forward(self, batch: RewardBatch, stage: str) -> dict[str, Tensor]:
        """Bradley–Terry loss in fit; pairwise accuracy + reward accumulation in validate.

        ``stage == "fit"``: return ``{"loss": BT loss}`` (spt's ``training_step``
        runs ``manual_backward`` over it, stepping the discriminator optimizer).
        ``stage == "validate"`` (spt's ``validation_step`` calls this under
        ``no_grad``): log pairwise accuracy (linear → per-batch averaging is
        exact) and stash the rewards for the pooled ROC-AUC computed at epoch end
        (a rank statistic must be pooled across the whole validation set).
        """
        r_w, r_l = self._score_pair(batch["winner"], batch["loser"])
        if stage == "fit":
            return {"loss": bradley_terry_loss(r_w, r_l)}
        self.log("val/pair_acc", (r_w > r_l).float().mean(), on_epoch=True, prog_bar=True)
        self._val_r_w.append(r_w.detach().cpu())
        self._val_r_l.append(r_l.detach().cpu())
        return {"r_w": r_w, "r_l": r_l}

    def on_validation_epoch_start(self) -> None:
        """Reset the per-epoch reward accumulators (pooled ROC-AUC scratch)."""
        self._val_r_w.clear()
        self._val_r_l.clear()

    def on_validation_epoch_end(self) -> None:
        """Pooled ROC-AUC over the whole validation set + the fixed generated-end probe.

        ROC-AUC is a rank statistic, so it is computed once over ALL accumulated
        validation rewards (not per-batch-averaged). The probe (both samples
        ``t ∈ [0, 0.5)``, ordered by ``t``) is precomputed (frozen denoiser) and
        reused across epochs, so its metric isolates discriminator quality.
        """
        if self._val_r_w:
            r_w = torch.cat(self._val_r_w)
            r_l = torch.cat(self._val_r_l)
            self.log("val/roc_auc", reward_roc_auc(r_w, r_l))
        if self.val_probe is not None:
            winner, loser = self.val_probe
            with torch.no_grad():
                pr_w = self.reward_model(winner.to(self.device))
                pr_l = self.reward_model(loser.to(self.device))
            self.log("val/gen_pair_acc", (pr_w > pr_l).float().mean())

    def configure_optimizers(self):
        """Adam over discriminator parameters only (the reward model)."""
        return {"optimizer": torch.optim.Adam(self.reward_model.parameters(), lr=self.lr)}
