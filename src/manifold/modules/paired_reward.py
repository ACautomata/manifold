"""Paired Reward Module: Bradley–Terry preference training for the Paired-JiT reward.

An :class:`stable_pretraining.Module` (``spt.Module``, manual optimization) that
trains the existing :class:`~manifold.models.RewardModel` (reused channel-agnostic,
constructed with ``in_channels = 2·C_latent``) under the Bradley–Terry pairwise
preference loss

.. math:: L = -\\log \\sigma(r_w - r_l)

(σ inside the loss only - the model emits raw rewards). The optimizer covers
**discriminator parameters only**.

**Offline precompute (ADR-0020, inverts ADR-0010).** The Module holds **no**
generator: the paired rollout is deterministic given ``x_src`` (no stochastic
input), so the generated fakes are precomputed **once** into a disk fake-cache
(``roll -> cache -> train``) and ``fit`` consumes precomputed ``{winner, loser}``
pairs - structurally identical to the JiT reward's ``validate`` path. The winner
is the real target latent ``concat([x_src, x_tgt])`` (ADR-0018 real-vs-fake) and
the loser is ``concat([x_src, generated_tgt])`` - the discriminator judges
"faithful translation *of* this src" (ADR-0019 condition-aware; an unconditional
reward would reward copy-src).

``validation_step`` reports:

- **pairwise accuracy** (``r_w > r_l``, real > generated) and **ROC-AUC** (a
  threshold-free cross-pair summary) for diagnosis - both **saturate near 1.0**
  under real-vs-fake (a PatchGAN trivially separates a flawed generation from a
  real latent), so they are diagnosis-only; and
- the **generated-end probe** accuracy (``val/gen_pair_acc`` - a fixed pair set
  where *both* samples are generated, ordered by translation-progress ``t``) -
  the within-fake ranking metric the checkpoint monitors (ADR-0023).

Sibling of :class:`manifold.modules.reward.RewardModule`; only the supervision
(real-vs-fake, not corruption-level) and the offline (no-generator) discipline
differ. ``bradley_terry_loss`` / ``reward_roc_auc`` / ``_score_pair`` are reused
verbatim.
"""

from __future__ import annotations

from typing import Any

import stable_pretraining as spt
import torch
from torch import Tensor

from ..models.reward_model import RewardModel
from .reward import _score_pair, bradley_terry_loss, reward_roc_auc

#: A paired reward-training batch: a precomputed ``(winner, loser)`` pair
#: (both halves are condition-aware ``[B, 2·C_latent, ...]`` concat latents).
PairedRewardBatch = dict[str, Any]


class PairedRewardModule(spt.Module):
    """Bradley–Terry preference training over the Paired-JiT reward model.

    Args:
        reward_model: the :class:`~manifold.models.RewardModel` being trained
            (constructed with ``in_channels = 2·C_latent`` - the caller concats
            ``[x_src, tgt]`` before scoring; ADR-0019).
        lr: Adam learning rate over the discriminator parameters.
        val_probe: optional fixed ``(winner, loser)`` generated-end probe tensors
            (both samples generated, ordered by translation-progress ``t``). Held
            but **not** registered; scored once per validation epoch. Each half is
            a condition-aware ``[N, 2·C_latent, ...]`` concat latent.
        probe_batch_size: probe scoring chunk (the fixed probe is scored in
            batches this size at epoch end).
    """

    def __init__(
        self,
        reward_model: RewardModel,
        *,
        lr: float = 1.0e-4,
        val_probe: tuple[Tensor, Tensor] | None = None,
        probe_batch_size: int = 8,
    ):
        # NOTE: forward is NOT passed to spt.Module - it would double-bind self.
        super().__init__(hparams={"lr": lr})
        self.reward_model = reward_model
        self.lr = float(lr)
        #: Fixed generated-end probe (winner, loser); set post-construction by
        #: :meth:`set_val_probe` or the ctor. A plain attribute (not a buffer) so
        #: it is device-moved manually and stays out of the checkpoint.
        self.val_probe: tuple[Tensor, Tensor] | None = None
        if val_probe is not None:
            self.set_val_probe(*val_probe)
        #: Probe scoring chunk - the fixed probe is scored in batches this size at
        #: epoch end (not one giant forward), so a small training batch_size does
        #: not OOM validation.
        self.probe_batch_size = int(probe_batch_size)
        #: Per-batch validation rewards accumulated for the pooled (cross-batch)
        #: ROC-AUC - a rank statistic, so per-batch averaging would be wrong.
        #: Reset each validation epoch in :meth:`on_validation_epoch_start`.
        self._val_r_w: list[Tensor] = []
        self._val_r_l: list[Tensor] = []

    # -- probe lifecycle ------------------------------------------------------

    def set_val_probe(self, winner: Tensor, loser: Tensor) -> None:
        """Attach the fixed generated-end probe pair set (reused across epochs)."""
        self.val_probe = (winner, loser)

    # -- scoring --------------------------------------------------------------

    def _score_pair(self, winner: Tensor, loser: Tensor) -> tuple[Tensor, Tensor]:
        """Forward both halves of the pair in one batch -> ``(r_w, r_l)`` each ``[B]``.

        Thin delegate over the module-level :func:`~manifold.modules.reward._score_pair`
        (shared with the JiT :class:`RewardModule`); see that function for the
        rationale. The winner/loser are already condition-aware ``[2·C]`` concat
        latents - the scorer applies verbatim (ADR-0019).
        """
        return _score_pair(self.reward_model, winner, loser)

    def forward(self, batch: PairedRewardBatch, stage: str) -> dict[str, Tensor]:
        """Bradley–Terry loss in fit; metrics in validate (both consume precomputed pairs).

        ``stage == "fit"``: ``batch`` is a precomputed ``{winner, loser}`` pair
        (both halves ``[B, 2·C_latent, ...]``); score and return ``{"loss": BT loss}``
        (spt's ``training_step`` runs ``manual_backward`` over it, stepping the
        discriminator optimizer). The Module holds NO generator (ADR-0020) - fit is
        structurally the JiT reward's validate path.
        ``stage == "validate"`` (spt's ``validation_step`` calls this under
        ``no_grad``): ``batch`` is a precomputed ``{winner, loser}`` pair; log
        pairwise accuracy and stash the rewards for the pooled ROC-AUC at epoch end.
        """
        if "winner" not in batch or "loser" not in batch:
            raise ValueError(
                f"stage {stage!r} expects a {{winner, loser}} pair batch of condition-aware "
                "[2·C_latent] concat latents (built offline from the fake cache)."
            )
        r_w, r_l = self._score_pair(batch["winner"], batch["loser"])
        if stage == "fit":
            return {"loss": bradley_terry_loss(r_w, r_l)}
        if stage == "validate":
            valid = ~batch.get(
                "_is_padding", torch.zeros(r_w.shape[0], dtype=torch.bool, device=r_w.device)
            ).to(r_w.device).bool()
            self._val_r_w.append(r_w[valid].detach().cpu())
            self._val_r_l.append(r_l[valid].detach().cpu())
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
        the global AUC is computed. Mirrors :meth:`RewardModule._gather_global`.
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
        (both samples generated, ordered by translation-progress ``t``) is
        precomputed (frozen generator ⇒ static) and reused across epochs; it is
        scored in ``probe_batch_size`` chunks (not one forward) to avoid an
        epoch-end OOM. The checkpoint monitors ``val/gen_pair_acc`` (the
        within-fake-ranking metric; ``val/pair_acc`` / ``val/roc_auc`` saturate).
        """
        if self._val_r_w:
            r_w = self._gather_global(torch.cat(self._val_r_w))
            r_l = self._gather_global(torch.cat(self._val_r_l))
            if r_w.numel():
                self.log("val/pair_acc", (r_w > r_l).float().mean(), prog_bar=True)
                self.log("val/roc_auc", reward_roc_auc(r_w, r_l))  # already global
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

        The Module holds no generator (ADR-0020), so the optimizer covers
        ``reward_model.parameters()`` alone - the discriminator-only-gradient
        invariant holds structurally (no frozen generator to exclude).
        """
        return {"optimizer": torch.optim.Adam(self.reward_model.parameters(), lr=self.lr)}


__all__ = ["PairedRewardBatch", "PairedRewardModule"]
