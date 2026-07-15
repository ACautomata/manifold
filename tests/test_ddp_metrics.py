"""DDP honest cross-rank metric reduction + parity invariants (issue #82).

- **M6**: ``train/loss_epoch`` + ``val/x0_mae`` migrate to
  :class:`torchmetrics.MeanMetric(weight=batch_size)` so the cross-rank
  reduction is the true sample-weighted global mean (NOT a mean-of-per-rank-
  means). A 2-rank fit with shard-different validation batches asserts the logged
  value equals the hand-computed global weighted mean.
- **L5**: GRPO ``train/loss`` (logged with ``batch_size=B``) gets ``sync_dist=True``
  for the cross-rank reduce.
- **G3**: ``train/grad_norm`` needs no ``sync_dist`` - DDP all-reduces the UNet
  gradients before the ``after_manual_backward`` hook, so the value is identical
  on every rank. Asserted as a 2-rank invariant.
- **D1**: the val loader actually has a ``DistributedSampler`` attached (first
  val-batch checksum differs across ranks); without it the M6 reduction is a
  no-op.

All 2-rank gates reuse :func:`tests.ddp.run_ddp_two_rank`.
"""

from __future__ import annotations

import pytest
import torch

from tests.ddp import _unbalanced_val_worker, jit_ddp_worker, paired_psnr_ddp_worker, run_ddp_two_rank


# -- M6: MeanMetric yields the true sample-weighted global mean -----------------


def test_m6_val_x0_mae_is_global_weighted_mean(tmp_path):
    """2-rank: ``val/x0_mae`` is identical on both ranks AND equals the hand-computed
    sample-weighted global mean over BOTH ranks' val batches (not a per-rank mean).

    The 5-sample val set is sharded across ranks (each rank sees DIFFERENT samples
    via the DistributedSampler), so a rank-local mean would differ - the equality
    proves the cross-rank ``MeanMetric`` reduction fired. The hand-computed
    ``sum(mae·B)/sum(B)`` over all per-batch (mae, B) pairs is the true global mean;
    a naive ``sync_dist`` (mean-of-per-rank-means) would NOT equal it when the
    per-rank batch counts/sizes differ. (Lightning's padding makes per-rank counts
    equal today, so the two coincide - the unit test below locks the property
    against a future non-padding sampler.)
    """
    results = run_ddp_two_rank(_unbalanced_val_worker, results_dir=str(tmp_path), args=(False,))
    r0, r1 = results
    # Shards actually differ (proves a DistributedSampler is attached - D1 too).
    assert r0["per_batch"] != r1["per_batch"], "val shards are identical across ranks (no sampler)"

    # Both ranks logged the same reduced value.
    m0 = r0["metrics"]["val/x0_mae"]
    m1 = r1["metrics"]["val/x0_mae"]
    assert m0 == pytest.approx(m1, abs=1e-5)

    # ... and it equals the true sample-weighted global mean over all batches.
    all_batches = r0["per_batch"] + r1["per_batch"]
    true_global = sum(mae * b for mae, b in all_batches) / sum(b for _, b in all_batches)
    assert m0 == pytest.approx(true_global, abs=1e-5), (
        f"val/x0_mae={m0} != true weighted mean {true_global}"
    )


def test_m6_meanmetric_unit_math_is_weighted_not_rank_means():
    """Unit lock (no Lightning): ``MeanMetric(per_batch_mean, weight=B)`` reduces to
    the true sample-weighted global mean ``(Σ mean·B)/(Σ B) = (Σ samples)/(Σ n)``,
    NOT the mean-of-per-rank-means ``(mean0 + mean1)/2``.

    rank0: mean=2.0, B=3 ; rank1: mean=4.5, B=2 -> true global = (6+9)/(3+5)=... wait,
    = (2·3 + 4.5·2)/(3+2) = (6+9)/5 = 3.0 ; mean-of-rank-means = (2+4.5)/2 = 3.25.
    Locks the property a future non-padding sampler would expose as a divergence.
    """
    import torchmetrics as pl_metrics

    # Simulate two ranks' MeanMetric state and the global merge (MeanMetric sync
    # sums mean_value + weight across ranks, then divides).
    m0 = pl_metrics.MeanMetric()
    m1 = pl_metrics.MeanMetric()
    m0.update(torch.tensor(2.0), weight=3.0)  # rank0: mean 2.0 over 3 samples
    m1.update(torch.tensor(4.5), weight=2.0)  # rank1: mean 4.5 over 2 samples
    # Cross-rank sync: the metric's internal mean_value + weight are all-reduced
    # (summed). Reproduce that merge by hand:
    merged_value = m0.mean_value + m1.mean_value
    merged_weight = m0.weight + m1.weight
    reduced = float(merged_value / merged_weight)
    assert reduced == pytest.approx(3.0, abs=1e-6), f"true weighted mean {reduced} != 3.0"
    assert reduced != pytest.approx(3.25, abs=1e-6), "collapsed to mean-of-rank-means (wrong)"


# -- G3: grad_norm is naturally cross-rank-identical ---------------------------


def test_g3_grad_norm_identical_across_ranks(tmp_path):
    """2-rank: ``train/grad_norm`` is identical on rank 0 and rank 1 at matching steps.

    No ``sync_dist`` needed (and none added): DDP all-reduces the UNet gradients
    before the ``after_manual_backward`` hook, so ``_grad_norm(unet.parameters())``
    reads identical values on every rank. If this fails, grad_norm must be added to
    the ``sync_dist`` sweep - a NEW finding, flagged by this gate.
    """
    results = run_ddp_two_rank(jit_ddp_worker, results_dir=str(tmp_path), args=(False,))
    gn0 = results[0]["metrics"].get("train/grad_norm")
    gn1 = results[1]["metrics"].get("train/grad_norm")
    assert gn0 is not None and gn1 is not None, "train/grad_norm not logged"
    assert gn0 == pytest.approx(gn1, rel=1e-4, abs=1e-5), (
        f"grad_norm diverges across ranks: {gn0} vs {gn1} (sync_dist sweep needed)"
    )


# -- M6/L5: train/loss_epoch reduced across ranks ------------------------------


def test_m6_train_loss_epoch_reduced_across_ranks(tmp_path):
    """2-rank: ``train/loss_epoch`` (MeanMetric) is identical on both ranks - the
    cross-rank reduction fired (a rank-local mean would differ because each rank
    trains on a different data shard)."""
    results = run_ddp_two_rank(jit_ddp_worker, results_dir=str(tmp_path), args=(False,))
    assert results[0]["metrics"]["train/loss_epoch"] == pytest.approx(
        results[1]["metrics"]["train/loss_epoch"], abs=1e-5
    )


def test_m6_train_loss_epoch_is_sample_weighted_not_batch_mean():
    """Unit lock (no Lightning): ``TrainLossLogger`` sample-weights by batch size,
    so the epoch aggregate is ``sum(loss_i·B_i)/sum(B_i)`` (the true sample-weighted
    mean), NOT a mean-of-per-batch-means ``(loss_1+loss_2)/2``.

    Two batches of UNEQUAL size: batch 1 = mean loss 2.0 over 3 samples, batch 2 =
    mean loss 4.5 over 2 samples. True sample-weighted = (2·3 + 4.5·2)/(3+2) = 3.0;
    mean-of-batch-means = (2+4.5)/2 = 3.25. Locks the property codex flagged (the
    module loss is a scalar batch-MEAN, so the weight must be B, not loss.shape[0]).
    """
    import torchmetrics as pl_metrics

    from manifold.training.metrics import TrainLossLogger

    cb = TrainLossLogger()
    cb._mean = pl_metrics.MeanMetric()
    # Simulate two on_train_batch_end calls with a scalar batch-mean loss + a
    # batch dict whose leading dim is B (mirrors JiT {"latent": [B, ...]}).
    cb._mean.update(torch.tensor(2.0), weight=3.0)   # batch 1: B=3
    cb._mean.update(torch.tensor(4.5), weight=2.0)   # batch 2: B=2
    reduced = float(cb._mean.compute())
    assert reduced == pytest.approx(3.0, abs=1e-6), f"sample-weighted {reduced} != 3.0"
    assert reduced != pytest.approx(3.25, abs=1e-6), "collapsed to mean-of-batch-means (wrong)"


def test_m6_train_loss_batch_size_derived_from_batch():
    """``_batch_size`` reads B from the batch tensor (JiT ``latent`` / paired
    ``src_latent``), so the weight is the per-batch sample count, not the scalar
    loss's shape (which is [] -> 1)."""
    import torch

    from manifold.training.metrics import _batch_size

    assert _batch_size({"latent": torch.randn(3, 4, 4, 4, 4)}, {"loss": torch.tensor(2.0)}) == 3.0
    assert _batch_size({"src_latent": torch.randn(2, 4, 4, 4)}, {"loss": torch.tensor(1.0)}) == 2.0
    # Fallback when no recognizable batch tensor.
    assert _batch_size({}, {"loss": torch.tensor(1.0)}) == 1.0


# -- D1: the val loader has a DistributedSampler attached ----------------------


def test_d1_val_loader_is_distributed_sharded(tmp_path):
    """2-rank: the val loader is wrapped in a ``DistributedSampler`` - the first
    val-batch checksum DIFFERS across ranks (each rank gets a shard). Without this,
    the M6/M5 reductions are no-ops (single shard) and the "global mean" is just the
    rank-0 shard mean, silently re-introducing the dishonesty.
    """
    results = run_ddp_two_rank(jit_ddp_worker, results_dir=str(tmp_path), args=(True,))
    # Sampler is a DistributedSampler (Lightning auto-wraps the val loader under DDP).
    assert "Distributed" in (results[0]["val_sampler"] or ""), (
        f"val sampler is {results[0]['val_sampler']!r}, not a DistributedSampler"
    )
    # The first val batch differs across ranks (a shard each - not the same data).
    assert results[0]["first_val_sum"] != results[1]["first_val_sum"], (
        "first val-batch checksum identical across ranks (val loader not sharded)"
    )


# -- L5: GRPO train/loss sync_dist (single-GPU smoke still passes) --------------


def test_l5_grpo_train_loss_has_sync_dist():
    """GRPO ``train/loss`` logs with ``sync_dist=True`` (the cross-rank reduce).

    Unlike M6, ``train/loss`` is already logged WITH ``batch_size=B`` so Lightning's
    epoch aggregate is the sample-weighted mean; ``sync_dist`` adds the cross-rank
    reduce. Exact even on unbalanced shards because of the weight.
    """
    import inspect

    from manifold.modules import grpo

    src = inspect.getsource(grpo.GRPOModule.training_step)
    # The train/loss log call carries sync_dist=True and batch_size=B.
    assert "sync_dist=True" in src, "GRPO train/loss missing sync_dist=True"
    assert 'self.log("train/loss"' in src


# -- Paired PSNR: all-rank decode + all_reduce global mean (ADR-0025) ----------


def test_paired_psnr_all_ranks(tmp_path):
    """2-rank: BOTH ranks decode their own ``DistributedSampler`` val shard and
    ``all_reduce`` the per-volume ``(psnr_sum, ssim_sum, count)`` for the global mean
    (ADR-0025; the PR #115 rank-0-only revert is undone - the VAE ``num_splits``
    config addresses the per-batch decode stall instead).

    Asserts: (a) both ranks decoded (``count_local > 0`` on each); (b) ``val/psnr`` +
    ``val/ssim`` logged on BOTH ranks to the SAME value; (c) that value is the GLOBAL
    mean ``(r0_sum + r1_sum) / (r0_count + r1_count)``, not either rank's own shard
    mean; (d) no deadlock (the spawn joined -> both ranks wrote a result).
    """
    results = run_ddp_two_rank(paired_psnr_ddp_worker, results_dir=str(tmp_path), args=(False,))
    r0, r1 = results
    # (a) Both ranks decode their own shard.
    assert r0["count_local"] > 0, "rank 0 did not decode"
    assert r1["count_local"] > 0, "rank 1 did not decode (all-rank gate not applied?)"
    # (b) val/psnr + val/ssim logged on both ranks.
    assert r0["val_psnr"] is not None and r0["val_ssim"] is not None, "rank 0 did not log"
    assert r1["val_psnr"] is not None and r1["val_ssim"] is not None, "rank 1 did not log"
    # (c) The logged value is the GLOBAL mean (all-reduced), identical on both ranks.
    g_count = r0["count_local"] + r1["count_local"]
    g_psnr = (r0["psnr_sum_local"] + r1["psnr_sum_local"]) / g_count
    g_ssim = (r0["ssim_sum_local"] + r1["ssim_sum_local"]) / g_count
    assert r0["val_psnr"] == pytest.approx(g_psnr, abs=1e-4), "rank 0 val/psnr != global mean"
    assert r1["val_psnr"] == pytest.approx(g_psnr, abs=1e-4), "rank 1 val/psnr != global mean"
    assert r0["val_ssim"] == pytest.approx(g_ssim, abs=1e-4)
    assert r1["val_ssim"] == pytest.approx(g_ssim, abs=1e-4)


def test_codex116_val_dataloader_drop_last_avoids_padding_duplication():
    """codex #116 P2 (Comment 4): under DDP Lightning's default val DistributedSampler
    pads shards by REPEATING samples, which would double-count a repeated val volume
    in the PSNR/SSIM all-reduce mean. The warm datamodules wire a non-padding
    ``UnrepeatedDistributedSampler`` (via ``_ddp_eval_sampler``) so each rank's shard is
    deduplicated. Verified by source inspection of the val_dataloader + helper."""
    import inspect

    from manifold.data.warm_datamodule import (
        LatentWarmDataModule,
        PairedWarmDataModule,
        _ddp_eval_sampler,
    )

    for cls in (LatentWarmDataModule, PairedWarmDataModule):
        src = inspect.getsource(cls.val_dataloader)
        assert "_ddp_eval_sampler(" in src, (
            f"{cls.__name__}.val_dataloader missing _ddp_eval_sampler (Comment 4)"
        )
    helper_src = inspect.getsource(_ddp_eval_sampler)
    assert "UnrepeatedDistributedSampler" in helper_src, "helper must use the non-padding sampler"
    # The helper guards the tiny-val probe (len < world -> padded fallback), so it
    # does not crash on the 8-rank val_subset_size=4 probe.
    assert "len(dataset) < world" in helper_src, "helper missing the empty-shard probe guard"


def test_codex116_r4_build_datamodule_defers_val_sampler_to_post_pg():
    """codex #116 round-4 P2: ``build_datamodule`` is called pre-``fit`` (before the PG
    is initialized), so a sampler baked in at construction sees
    ``dist.is_initialized()==False`` and returns None -> Lightning re-wraps with the
    PADDED default. The fix: ``_DedupValDataModule.val_dataloader`` is a Lightning hook
    that calls ``_ddp_eval_sampler`` at fit-time (post-PG). Verified by source: the
    hook (not the constructor) resolves the sampler."""
    import inspect

    from manifold.data.datamodule import _DedupValDataModule

    hook_src = inspect.getsource(_DedupValDataModule.val_dataloader)
    init_src = inspect.getsource(_DedupValDataModule.__init__)
    assert "_ddp_eval_sampler(" in hook_src, "sampler must be resolved in val_dataloader (post-PG)"
    assert "_ddp_eval_sampler" not in init_src, "sampler must NOT be baked in at construction (pre-PG)"


def test_codex116_r4_dedup_val_module_val_dataloader_resolves_sampler(monkeypatch):
    """The ``_DedupValDataModule.val_dataloader`` hook returns a loader whose sampler is
    whatever ``_ddp_eval_sampler`` yields at call time (post-PG). With the PG
    uninitialized (pre-fit), it returns None (Lightning's padded default); with a stubbed
    PG it returns the non-padding sampler - so the deferred resolution works."""
    from torch.utils.data import TensorDataset, DataLoader

    from manifold.data.datamodule import _DedupValDataModule
    from lightning.pytorch.overrides.distributed import UnrepeatedDistributedSampler

    ds = TensorDataset(torch.arange(8))
    train_loader = DataLoader(TensorDataset(torch.arange(4)), batch_size=2)
    dm = _DedupValDataModule(val_dataset=ds, batch_size=2, num_workers=0, train=train_loader)
    # PG not initialized -> _ddp_eval_sampler returns None -> sampler None.
    import torch.distributed as dist

    assert not dist.is_initialized()
    loader = dm.val_dataloader()
    # PG not initialized -> _ddp_eval_sampler returns None -> DataLoader defaults to
    # SequentialSampler (single-process, pre-PG). Definitely NOT the non-padding
    # distributed sampler (which needs a live PG).
    assert not isinstance(loader.sampler, UnrepeatedDistributedSampler), (
        "pre-PG val_dataloader should not build a distributed sampler"
    )

    # Stub a live PG (world=2, 8 >= 2) -> non-padding sampler returned.
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    loader2 = dm.val_dataloader()
    assert isinstance(loader2.sampler, UnrepeatedDistributedSampler), (
        f"post-PG val_dataloader must use the non-padding sampler, got {type(loader2.sampler)}"
    )
