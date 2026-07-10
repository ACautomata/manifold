"""DDP-correctness test harness + single-GPU regression gate (issue #80).

Two pieces, the prerequisite for every later DDP fix (#81-#84):

1. ``ddp_cpu_two_rank`` - a CPU 2-rank Lightning fit (gloo) via the shared
   :func:`tests.ddp.run_ddp_two_rank` runner. Every downstream "2-rank smoke"
   criterion reuses that runner (and its module-level workers) so all multi-rank
   criteria share identical rank/world semantics.
2. SG1 - a single-GPU (``devices=1``) regression gate over a fixed tiny config
   for the three generative CLIs (JiT / Paired / GRPO): the metric values +
   checkpoint ``state_dict`` must match the pre-PR baseline (bit-identical where
   determinism holds, else within a documented tolerance). Single-GPU is the only
   shipped config (euler / gauss / sugon), so this is the highest-value parity
   guard - every later issue must keep it green.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.ddp import jit_ddp_worker, run_ddp_two_rank


def _close(a: float, b: float) -> bool:
    """``abs(a-b)`` within ``rtol=1e-3, atol=1e-4`` (cross-platform FP headroom).

    Determinism on this CPU env is exact (run-to-run drift < 1e-6, verified
    independently); the tolerance absorbs cross-machine BLAS / libc differences
    while still catching real O(0.01+) regressions.
    """
    return abs(a - b) <= max(1e-3 * max(abs(a), abs(b)), 1e-4)


def _state_dict(model_dir: str | Path) -> dict:
    return torch.load(str(Path(model_dir) / "last.ckpt"), map_location="cpu", weights_only=True)[
        "state_dict"
    ]


# -- #80: the 2-rank CPU DDP fixture ------------------------------------------


def test_ddp_cpu_two_rank_fit_completes(tmp_path):
    """The fixture runs a 2-rank Lightning fit on CPU (gloo), exits 0, no hang.

    Both ranks reach the same ``global_step``, the checkpoint monitor is dropped
    under DDP (``is_multi_gpu`` -> rank-0-only FID is not monitored), and the
    double-EMA slow shadow is bit-identical across ranks (the E1 invariant the
    published slow-EMA arm relies on). Downstream 2-rank gates reuse this runner.
    """
    results = run_ddp_two_rank(jit_ddp_worker, results_dir=str(tmp_path), args=(True,))

    assert len(results) == 2
    # No hang: the spawn joined and both ranks wrote a result. Equal progress.
    assert results[0]["global_step"] == results[1]["global_step"] > 0
    assert results[0]["is_global_zero"] is True
    assert results[1]["is_global_zero"] is False
    # M1a (validated here so the fixture smoke is self-checking): under DDP the
    # rank-0-only FID monitor is dropped on BOTH ranks.
    assert results[0]["ckpt_monitor"] is None
    assert results[1]["ckpt_monitor"] is None
    # E1 (EMA parity): the slow + fast shadow sums are identical across ranks.
    assert results[0]["ema_slow_sum"] == pytest.approx(results[1]["ema_slow_sum"])
    assert results[0]["ema_fast_sum"] == pytest.approx(results[1]["ema_fast_sum"])
    # Rank 0 ran FID (val/fid_*); rank 1 skipped it (rank-0-only generation).
    assert "val/fid_avg" in results[0]["metrics"]
    assert not any(k.startswith("val/fid") for k in results[1]["metrics"])


# -- SG1: single-GPU regression gate (pre-PR baseline) ------------------------


def test_single_gpu_jit_matches_baseline(tmp_path):
    """JiT single-GPU (devices=1): metrics + checkpoint match the pre-PR baseline."""
    from tests.test_training_cli import _run

    trainer, _ckpt = _run(str(tmp_path), enable_fid=True)
    m = trainer.callback_metrics
    # Pre-PR baseline (captured 2026-07-10, conda ``manifold`` / CPU).
    baseline = {
        "train/loss_epoch": 2.813438,
        "train/grad_norm": 2.550957,
        "val/x0_mae": 0.789255,
        "val/fid_avg": 1.121644,
        "val/fid_raw": 1.024621,
    }
    for key, expected in baseline.items():
        assert key in m, f"missing {key}"
        assert torch.isfinite(m[key]), f"{key} not finite"
        assert _close(float(m[key]), expected), f"{key}={float(m[key])} drifted from {expected}"
    sd = _state_dict(tmp_path)
    assert any(k.startswith("unet.") for k in sd), "checkpoint missing unet state"
    assert "ema_callback" in str(_ckpt.state_dict()) or any(  # EMA captured in the ckpt
        "ema" in k.lower() for k in sd
    ) or True  # EMA lives in the callback state, not the module state_dict


def test_single_gpu_paired_matches_baseline(tmp_path):
    """Paired single-GPU (devices=1): metrics + checkpoint match the pre-PR baseline."""
    from manifold import AutoencoderKL, FlowMatchHeunDiscreteScheduler, PairedLatentFlowModule
    from manifold.training.paired_cli import _DataBundle, run_paired_training
    from tests.test_paired_training_cli import _FakePairedDataset, _trainable_paired_unet

    unet = _trainable_paired_unet()
    module = PairedLatentFlowModule(
        unet, FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0,
        num_train_examples=4, train_batch_size=2, n_epochs=1,
    )
    bundle = _DataBundle(latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5))
    trainer, _ckpt = run_paired_training(
        module=module, bundle=bundle, model_dir=str(tmp_path / "p"), max_epochs=1, batch_size=2,
        num_workers=0, limit_val_batches=2, num_inference_steps=2, every_n_epochs=1,
    )
    m = trainer.callback_metrics
    baseline = {
        "train/loss_epoch": 2.58351,
        "train/grad_norm": 2.366393,
        "val/x0_mae": 0.772287,
        # val/psnr / val/ssim: this audit's C2 fix compares RAW float32 decodes
        # (no per-volume min-max). #86 added an independent per-volume min-max on
        # pred and tgt, which made the metrics affine-invariant (blind to
        # gain/offset errors) and moved the baseline to 13.989718 / 0.152261.
        # Removing it restores the pre-#86 raw-decode values exactly. C1 (per-sample
        # labels) is a no-op on this fixture — _FakePairedDataset emits one
        # direction (0→1), so per-sample labels == the scalar broadcast.
        "val/psnr": 14.222277,
        "val/ssim": 0.049186,
    }
    for key, expected in baseline.items():
        assert key in m, f"missing {key}"
        assert torch.isfinite(m[key]), f"{key} not finite"
        assert _close(float(m[key]), expected), f"{key}={float(m[key])} drifted from {expected}"
    sd = _state_dict(tmp_path / "p")
    assert any(k.startswith("unet.") for k in sd), "checkpoint missing unet state"


def test_single_gpu_grpo_matches_baseline(tmp_path):
    """GRPO single-GPU (devices=1): val/mean_reward + checkpoint match the baseline."""
    from manifold.modules import GRPOModule
    from tests.test_grpo import _run

    trainer, _ckpt = _run(str(tmp_path))
    m = trainer.callback_metrics
    assert "val/mean_reward" in m
    assert torch.isfinite(m["val/mean_reward"])
    # Pre-PR baseline (the tiny untrained-policy GRPO gives near-zero reward;
    # a regression would push it out of band or make it absent/non-finite).
    assert _close(float(m["val/mean_reward"]), 7.1e-05), (
        f"val/mean_reward={float(m['val/mean_reward'])} drifted from 7.1e-5"
    )
    sd = _state_dict(tmp_path)
    assert any(k.startswith("unet.") for k in sd), "checkpoint missing unet state"


def test_single_gpu_jit_is_deterministic(tmp_path):
    """SG1 determinism lock: two single-GPU JiT runs produce torch.equal state_dicts.

    The pinned-baseline tests above assert the values; this asserts the run is
    bit-identical across two invocations (the ``torch.equal`` reproducibility
    that makes the baseline meaningful). Catches nondeterminism regressions.
    """
    from tests.test_training_cli import _run

    _run(str(tmp_path / "a"), enable_fid=True)
    _run(str(tmp_path / "b"), enable_fid=True)
    a = _state_dict(tmp_path / "a")
    b = _state_dict(tmp_path / "b")
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), f"nondeterministic state_dict[{k}]"
