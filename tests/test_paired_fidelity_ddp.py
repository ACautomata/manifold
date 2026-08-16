"""DDP keeps the paired-fidelity monitor safe: no deadlock, consistent (issue #240).

The monitor's DDP design (ADR-0037 / ADR-0025) is **redundant evaluation**: every
rank runs the same fixed subset under the same seeded noise on DDP-synchronized
weights, so the per-rank result is identical and Lightning's torchmetrics sync is
the only cross-rank interaction — no FID-style sharding / error-rendezvous
machinery (collective-count invariance, ADR-0030, holds trivially). This gate
verifies that empirically on the existing 2-rank ControlNet DDP harness
(``run_ddp_two_rank`` + the ``controlnet_*_ddp_worker`` pattern of issue #146).
"""

from __future__ import annotations

import math

from tests.ddp import controlnet_paired_fidelity_ddp_worker, run_ddp_two_rank


def test_ddp_paired_fidelity_monitor_no_deadlock_and_consistent(tmp_path):
    """2-rank DDP with the monitor default-on: the fit completes (no hang), both
    ranks log the SAME finite ``val/psnr`` / ``val/ssim``, and the monitor leaves
    the rank structure + validation contract untouched (``is_global_zero`` split
    intact, checkpoint still monitors ``val/x0_mae``, the monitored ckpt is
    written, ``val/x0_mae`` still finite on both ranks).
    """
    results = run_ddp_two_rank(
        controlnet_paired_fidelity_ddp_worker, results_dir=str(tmp_path), args=(False,)
    )
    r0, r1 = results
    # No-hang: both ranks completed the fit — reaching the per-rank JSONs means the
    # synchronous spawn returned on both sides: neither rank stalled in the
    # monitor's rollout, its decode, or the metric sync collective.
    assert r0["global_step"] > 0 and r1["global_step"] > 0
    for r in (r0, r1):
        # The monitor logged its two metrics on EVERY rank (no missing values)...
        assert r["val_psnr"] is not None, f"rank {r['rank']}: val/psnr missing"
        assert r["val_ssim"] is not None, f"rank {r['rank']}: val/ssim missing"
        # ...finite (the rollout + decode + metric ran)...
        assert math.isfinite(r["val_psnr"]), f"rank {r['rank']}: val/psnr not finite"
        assert math.isfinite(r["val_ssim"]), f"rank {r['rank']}: val/ssim not finite"
        # ...and the existing validation contract is unchanged (observe-only).
        assert r["ckpt_monitor"] == "val/x0_mae", (
            f"rank {r['rank']}: monitor changed to {r['ckpt_monitor']}"
        )
        assert r["ckpt_mode"] == "min"
        assert math.isfinite(r["val_x0_mae"]), f"rank {r['rank']}: val/x0_mae not finite"
    # Consistency: identical fixed input + DDP-synchronized weights + identical
    # seeded noise => the SAME value on both ranks (redundant evaluation, not a
    # sharded one — the mean-of-identical metric reduction cannot move the value).
    assert r0["val_psnr"] == r1["val_psnr"], (
        f"val/psnr diverged across ranks: {r0['val_psnr']} vs {r1['val_psnr']}"
    )
    assert r0["val_ssim"] == r1["val_ssim"], (
        f"val/ssim diverged across ranks: {r0['val_ssim']} vs {r1['val_ssim']}"
    )
    # ...and observed at the SOURCE, not just in the synced logged values: each
    # rank's OWN (pre-sync) rollout score is the same. The metric sync averages
    # across ranks, so a silent per-rank divergence would be masked in the logged
    # values above — this locks the redundant-evaluation design claim directly
    # (identical fixed input + DDP-synchronized weights => identical result).
    for key in ("val_psnr_local", "val_ssim_local"):
        assert r0[key] is not None and r1[key] is not None, (
            f"{key} missing on a rank: {r0[key]} vs {r1[key]}"
        )
        assert r0[key] == r1[key], f"{key} diverged across ranks: {r0[key]} vs {r1[key]}"
    # The rank structure is unchanged: the is_global_zero split is intact and rank 0
    # still writes the monitored checkpoint (not just ``last.ckpt``).
    assert r0["is_global_zero"] and not r1["is_global_zero"], (r0, r1)
    monitored = [n for n in r0["written_ckpts"] if n.startswith("controlnet-") and n != "last.ckpt"]
    assert monitored, f"no monitored (val/x0_mae) checkpoint written: {r0['written_ckpts']}"
