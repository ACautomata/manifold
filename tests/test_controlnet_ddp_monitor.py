"""DDP keeps the globally-reduced val monitor (issue #146).

Under DDP, ``controlnet_cli._build_checkpoint`` used to drop the ``val/x0_mae``
monitor and fall back to ``save_top_k=1`` + ``save_last``. But ``val/x0_mae`` is
**globally reduced** across ranks: :class:`~manifold.training.metrics.LatentX0MAE`
accumulates into a sample-weighted ``torchmetrics.MeanMetric`` and logs the Metric
object, so Lightning fires the cross-rank reduction to the true global mean. A
rank-local fallback threw away best-checkpoint selection for no correctness gain.

Gate (2 CPU ranks): the checkpoint still monitors ``val/x0_mae`` (mode ``min``)
under DDP and writes the monitored checkpoint file (not just ``last``).
"""

from __future__ import annotations

from tests.ddp import controlnet_monitor_ddp_worker, run_ddp_two_rank


def test_ddp_keeps_globally_reduced_val_monitor(tmp_path):
    """2-rank DDP: the ControlNet checkpoint monitors the globally-reduced val/x0_mae.

    Asserts (a) ``ckpt.monitor == "val/x0_mae"`` (not dropped to save_top_k=1),
    (b) a monitored checkpoint file (``controlnet-epoch=...-val/x0_mae=...ckpt``)
    is written — not just ``last.ckpt``, and (c) the metric is finite on both ranks
    (the global reduction ran).
    """
    results = run_ddp_two_rank(controlnet_monitor_ddp_worker, results_dir=str(tmp_path), args=(False,))
    r0, r1 = results
    # No-hang: both ranks completed.
    assert r0["global_step"] > 0 and r1["global_step"] > 0
    # (a) The monitor stays ON under DDP (mode min for the x0-MAE).
    for r in (r0, r1):
        assert r["ckpt_monitor"] == "val/x0_mae", (
            f"rank {r['rank']}: monitor dropped to a save_top_k=1 fallback ({r['ckpt_monitor']})"
        )
        assert r["ckpt_mode"] == "min"
    # (b) A monitored checkpoint file is written (not just last.ckpt). The filename
    # embeds the monitored metric's VALUE (``controlnet-{epoch}-{x0_mae:.3f}.ckpt``),
    # so assert on a non-``last`` controlnet ckpt (the monitored save) being present.
    monitored = [n for n in r0["written_ckpts"] if n.startswith("controlnet-") and n != "last.ckpt"]
    assert monitored, f"no monitored (val/x0_mae) checkpoint written: {r0['written_ckpts']}"
    # (c) val/x0_mae is finite on both ranks (the global reduction ran).
    import math

    for r in (r0, r1):
        assert math.isfinite(r["val_x0_mae"]), f"rank {r['rank']} val/x0_mae not finite"
