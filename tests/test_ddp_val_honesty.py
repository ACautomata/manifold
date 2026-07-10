"""DDP rank-0-only validation honesty (issue #83).

- **M3**: GRPO ``validation_step`` generation + scoring runs on ``is_global_zero``
  only (the rank-asymmetric early-return must not deadlock). ``sample_latent_flow``
  call counts are ``>0`` on rank 0 and exactly ``0`` on rank 1; ``val/mean_reward``
  is logged exactly once (rank 0).
- **M4**: ``val/mean_reward`` scope documented in the ``validation_step``
  docstring (rank-0-shard-scoped; no ``sync_dist``).
- **M5** (retired by the ADR-0016 amendment): ``val/psnr`` / ``val/ssim`` are now
  GLOBAL under DDP (every rank decodes its ``DistributedSampler`` shard +
  ``all_gather`` of the per-volume sums); the rank-0-shard scope + one-shot warning
  are gone. The metric names (``val/psnr`` / ``val/ssim``) are unchanged.
- **L1**: the FID "running only on rank 0" DDP warning is hoisted below the
  ``is_global_zero`` guard (fires rank-0 only). (PSNR no longer warns - distributed.)
- **L3**: the RadImageNet ``feature_net`` is lazy-built inside the rank-0-gated
  FID stage path (``make_feature_network`` call_count == 1 on rank 0, == 0 on rank 1).

Note: GRPO multi-GPU *training* is NOT blocked - codex G2=FALSE confirmed the PPO
inner loop is DDP-correct; ``no_sync()`` would be wrong (the algorithm steps every
inner iteration). M3 only makes the *validation* rank-0-honest.

All 2-rank gates reuse :func:`tests.ddp.run_ddp_two_rank`.
"""

from __future__ import annotations

import inspect

import pytest

from tests.ddp import grpo_ddp_worker, jit_ddp_worker, run_ddp_two_rank


# -- M3: GRPO validation_step is rank-0-only ----------------------------------


def test_m3_grpo_validation_step_runs_rank0_only(tmp_path):
    """2-rank: ``sample_latent_flow`` call count is ``>0`` on rank 0 and exactly
    ``0`` on rank 1 (the generation rollout is gated to ``is_global_zero``).
    ``val/mean_reward`` is logged exactly once (rank 0). The val epoch completes
    with exit 0 (no deadlock - the load-bearing no-hang gate for the rank-
    asymmetric early-return inside ``validation_step``).
    """
    results = run_ddp_two_rank(grpo_ddp_worker, results_dir=str(tmp_path), args=(False,))
    r0, r1 = results
    # The no-hang gate: both ranks wrote a result (spawn joined, no timeout).
    assert r0["global_step"] > 0 and r1["global_step"] > 0
    # Generation runs rank-0-only; rank 1 does no generation.
    assert r0["sample_latent_flow_calls"] > 0, "rank 0 should generate"
    assert r1["sample_latent_flow_calls"] == 0, "rank 1 should NOT generate"
    # val/mean_reward logged exactly once (rank 0).
    assert r0["val_mean_reward_logged"] is True
    assert r1["val_mean_reward_logged"] is False
    assert r0["is_global_zero"] is True
    assert r1["is_global_zero"] is False


def test_m3_m4_grpo_validation_step_has_no_sync_dist():
    """No ``sync_dist=`` argument on any ``self.log(...)`` call inside
    ``validation_step`` (M4: the rank-0 gate removes the cross-rank quantity - no
    ``sync_dist`` needed). The docstring may mention ``sync_dist`` by name (it
    documents the scope), so this checks the actual log-call arguments, not the
    raw string."""
    import re

    from manifold.modules import grpo

    src = inspect.getsource(grpo.GRPOModule.validation_step)
    log_calls = re.findall(r"self\.log\([^)]*\)", src, re.DOTALL)
    assert log_calls, "validation_step logs nothing (unexpected)"
    for call in log_calls:
        assert "sync_dist" not in call, f"log call has sync_dist (M4 forbids): {call}"
    assert "is_global_zero" in src, "validation_step missing the is_global_zero gate (M3)"


def test_m4_grpo_validation_step_docs_rank0_scope():
    """The ``validation_step`` docstring names the rank-0-shard scope (M4 review-only)."""
    from manifold.modules import grpo

    doc = grpo.GRPOModule.validation_step.__doc__ or ""
    assert "rank-0-shard" in doc.lower() or "rank 0" in doc.lower(), (
        "validation_step docstring should document the rank-0-shard scope (M4)"
    )


# -- L3: feature_net lazy-built on rank 0 only ---------------------------------


def test_l3_feature_net_built_rank0_only(tmp_path):
    """2-rank: ``make_feature_network`` (the ``feature_net_factory``) is invoked
    ``==1`` time on rank 0 and ``==0`` on rank 1 (lazy build inside the rank-0-
    gated FID stage path). Rank 1 does no ``torch.hub``/disk load."""
    results = run_ddp_two_rank(grpo_ddp_worker, results_dir=str(tmp_path), args=(False,))
    r0, r1 = results
    assert r0["feature_net_builds"] == 1, f"rank 0 should build feature_net once, got {r0['feature_net_builds']}"
    assert r1["feature_net_builds"] == 0, f"rank 1 should NOT build feature_net, got {r1['feature_net_builds']}"


def test_l3_fidcallback_supports_feature_net_factory():
    """``FIDCallback`` accepts ``feature_net_factory`` and builds it lazily in
    ``_stage_eval_on_device`` (the rank-0-gated path)."""
    from manifold.metrics.fid_callback import FIDCallback

    sig = inspect.signature(FIDCallback.__init__)
    assert "feature_net_factory" in sig.parameters, "FIDCallback missing feature_net_factory"
    # The build lives in _stage_eval_on_device (rank-0-gated).
    stage_src = inspect.getsource(FIDCallback._stage_eval_on_device)
    assert "feature_net_factory" in stage_src, "lazy build not in _stage_eval_on_device"


# -- L1: DDP warnings hoisted below is_global_zero (fire rank-0 only) ----------


def test_l1_fid_warning_below_guard():
    """The FID ``_log.warning`` is hoisted BELOW the ``is_global_zero: return``
    guard (fires rank-0 only). FID-only: the PSNR callback no longer has a guard or
    warning (it is distributed under DDP - ADR-0016 amendment). Verified by source
    order in the FID callback file."""
    from manifold.metrics import fid_callback

    src = inspect.getsource(fid_callback.FIDCallback._gated)
    guard_idx = src.find("if not trainer.is_global_zero")
    warn_idx = src.find("_log.warning")
    assert guard_idx >= 0 and warn_idx >= 0, "FIDCallback._gated missing guard or warning"
    assert guard_idx < warn_idx, (
        "FIDCallback: warning is BEFORE the is_global_zero guard (should be below - L1)"
    )


# -- M5: PSNR is distributed (no rank-0 gate) ---------------------------------


def test_m5_psnr_gate_is_distributed_no_rank0():
    """The PSNR callback's ``_gated`` is cadence-only under DDP - no ``is_global_zero``
    guard and no rank-0 scope warning (every rank decodes its ``DistributedSampler``
    shard + ``all_gather``'s the per-volume sums, per the ADR-0016 amendment). The
    metric names (``val/psnr`` / ``val/ssim``) are unchanged (the consumer keys on them)."""
    from manifold.metrics import psnr_ssim_callback

    src = inspect.getsource(psnr_ssim_callback.PairedPSNRSSIMCallback._gated)
    assert "is_global_zero" not in src, "PSNR _gated still has a rank-0 guard (should be distributed)"
    assert "_log.warning" not in src, "PSNR _gated still has a rank-0 warning (should be distributed)"
    # Metric names unchanged (the log calls stay val/psnr / val/ssim).
    log_src = inspect.getsource(psnr_ssim_callback.PairedPSNRSSIMCallback.on_validation_epoch_end)
    assert 'log("val/psnr"' in log_src
    assert 'log("val/ssim"' in log_src


def test_m5_metric_names_unchanged():
    """``val/psnr`` / ``val/ssim`` / ``val/mean_reward`` metric names are unchanged
    (the ``monitor_psnr`` / GRPO monitor consumers key on them). No ``sync_dist=``
    argument on any ``log(...)`` call in the PSNR callback: the value is already
    cross-rank-reduced by the epoch-end ``all_gather``, so ``sync_dist`` would
    double-sync (and a plain-float ``sync_dist`` would give a mean-of-per-rank-means)."""
    import re

    from manifold.metrics import psnr_ssim_callback

    src = inspect.getsource(psnr_ssim_callback.PairedPSNRSSIMCallback)
    assert 'log("val/psnr"' in src
    assert 'log("val/ssim"' in src
    log_calls = re.findall(r"\.log\([^)]*\)", src, re.DOTALL)
    for call in log_calls:
        assert "sync_dist" not in call, f"PSNR log call has sync_dist (M5 forbids): {call}"
