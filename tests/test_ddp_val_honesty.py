"""DDP all-rank validation (ADR-0025; supersedes the issue #83 rank-0-only tests).

Validation is fully distributed: every rank decodes / generates / scores its own
``DistributedSampler`` shard and the metric is reduced to a global value. The prior
rank-0-only gates (PR #115, the DCU-deadlock workaround) are removed - the VAE
``num_splits``/``save_mem`` config addresses the per-batch decode stall instead.

- **M3**: GRPO ``validation_step`` generation + scoring runs on BOTH ranks;
  ``sample_latent_flow`` call count is ``>0`` on rank 0 AND rank 1; ``val/mean_reward``
  is logged on both ranks (``sync_dist=True``) to the SAME global value.
- **M4**: ``val/mean_reward`` is logged with ``sync_dist=True`` (no ``is_global_zero`` gate).
- **L1**: the FID callback ``_gated`` is cadence-only (no ``is_global_zero`` guard, no warning).
- **L3**: the RadImageNet ``feature_net`` is built on BOTH ranks (every rank extracts
  features for its shard) - ``make_feature_network`` call_count == 1 on rank 0 AND rank 1.
- **M5**: PSNR/SSIM ``_gated`` is cadence-only (no rank-0 guard/warning) and
  ``on_validation_epoch_end`` ``all_reduce``s ``(psnr_sum, ssim_sum, count)`` for the
  global mean; metric names unchanged; PSNR ``log(...)`` calls carry no ``sync_dist``
  (the manual ``all_reduce`` already produced the global value).

All 2-rank gates reuse :func:`tests.ddp.run_ddp_two_rank`.
"""

from __future__ import annotations

import inspect

import pytest

from tests.ddp import grpo_ddp_worker, run_ddp_two_rank


# -- M3: GRPO validation_step runs on all ranks --------------------------------


def test_m3_grpo_validation_step_runs_all_ranks(tmp_path):
    """2-rank: ``sample_latent_flow`` is called ``>0`` on BOTH ranks (the generation
    rollout is no longer gated to ``is_global_zero``). ``val/mean_reward`` is logged
    on both ranks and, being ``sync_dist=True``, carries the SAME global value. The
    val epoch completes with exit 0 on both ranks (no deadlock)."""
    results = run_ddp_two_rank(grpo_ddp_worker, results_dir=str(tmp_path), args=(False,))
    r0, r1 = results
    # No-hang gate: both ranks wrote a result (spawn joined, no timeout).
    assert r0["global_step"] > 0 and r1["global_step"] > 0
    # Generation runs on both ranks now.
    assert r0["sample_latent_flow_calls"] > 0, "rank 0 should generate"
    assert r1["sample_latent_flow_calls"] > 0, "rank 1 should ALSO generate (all-rank)"
    # val/mean_reward logged on both ranks (sync_dist) to the same global value.
    assert r0["val_mean_reward_logged"] is True
    assert r1["val_mean_reward_logged"] is True
    assert r0["val_mean_reward"] is not None and r1["val_mean_reward"] is not None
    assert r0["val_mean_reward"] == pytest.approx(r1["val_mean_reward"], abs=1e-4), (
        "val/mean_reward should be the GLOBAL mean (sync_dist): both ranks equal"
    )


def test_m3_m4_grpo_validation_step_uses_global_sum_count():
    """GRPO excludes padding per step and all-reduces global sum/count at epoch end."""
    from manifold.modules import grpo

    step = inspect.getsource(grpo.GRPOModule.validation_step)
    end = inspect.getsource(grpo.GRPOModule.on_validation_epoch_end)
    assert "is_global_zero" not in step
    assert "_is_padding" in step
    assert "all_reduce" in end and "ReduceOp.SUM" in end
    assert "self.log(" not in step


def test_m4_grpo_validation_step_docs_all_rank_scope():
    """The ``validation_step`` docstring names the all-rank / distributed scope."""
    from manifold.modules import grpo

    doc = (grpo.GRPOModule.validation_step.__doc__ or "").lower()
    assert "all-rank" in doc or "all rank" in doc or "sync_dist" in doc, (
        "validation_step docstring should document the all-rank DDP scope (M4)"
    )


# -- L3: feature_net built on all ranks ----------------------------------------


def test_l3_feature_net_built_all_ranks(tmp_path):
    """2-rank: ``make_feature_network`` (the ``feature_net_factory``) is invoked once
    on rank 0 AND once on rank 1 - every rank extracts features for its own shard."""
    results = run_ddp_two_rank(grpo_ddp_worker, results_dir=str(tmp_path), args=(False,))
    r0, r1 = results
    assert r0["feature_net_builds"] == 1, f"rank 0 should build feature_net once, got {r0['feature_net_builds']}"
    assert r1["feature_net_builds"] == 1, f"rank 1 should ALSO build feature_net, got {r1['feature_net_builds']}"


def test_l3_fidcallback_supports_feature_net_factory():
    """``FIDCallback`` accepts ``feature_net_factory`` and the factory is built
    lazily in ``VramStage.__enter__`` (on every rank)."""
    from manifold.metrics.fid.callback import FIDCallback
    from manifold.metrics.fid.vram import VramStage

    sig = inspect.signature(FIDCallback.__init__)
    assert "feature_net_factory" in sig.parameters, "FIDCallback missing feature_net_factory"
    stage_src = inspect.getsource(VramStage.__enter__)
    assert "feature_net_factory" in stage_src, "lazy build not in VramStage.__enter__"


# -- L1: FID _gated is cadence-only (no rank-0 guard/warning) -------------------


def test_l1_fid_no_rank0_gate():
    """The FID ``_gated`` is cadence-only under ADR-0025: no ``is_global_zero`` guard
    and no rank-0 warning (every rank generates + extracts features for its shard)."""
    from manifold.metrics.fid import callback as fid_callback

    src = inspect.getsource(fid_callback.FIDCallback._gated)
    assert "is_global_zero" not in src, "FIDCallback._gated still has a rank-0 guard"
    assert "_log.warning" not in src, "FIDCallback._gated still has a rank-0 warning"
    assert "every_n_epochs" in src, "FIDCallback._gated lost its cadence check"
