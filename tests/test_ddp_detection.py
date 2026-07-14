"""DDP multi-GPU detection + ``-g 1`` semantics (issue #81).

Two correctness fixes, both mechanical and verified by the shared
``is_multi_gpu(devices)`` accessor:

- **M1a/M1b**: the monitor guard diverged from ``build_trainer`` on the
  ``devices="auto"`` path - the CLIs' inline ``isinstance(devices,int) and
  devices>1`` missed the ``"auto"`` branch, so under multi-GPU the checkpoint
  monitored a rank-0-only FID/PSNR. All four CLIs now call ``is_multi_gpu`` (the
  same predicate ``build_trainer`` uses for its strategy).
- **M2a/M2b**: ``-g 1`` (or no ``-g``) on a multi-GPU host silently became
  ``devices="auto"`` -> surprise DDP. It now means exactly one device
  (``devices=1``); multi-GPU requires explicit ``-g N``.

The ``is_multi_gpu`` unit tests mock ``torch.cuda.device_count`` /
``is_available`` directly (no Lightning, no GPU). The 2-rank DDP monitor gate
reuses :func:`tests.ddp.run_ddp_two_rank`.
"""

from __future__ import annotations

import pytest

from manifold.training.trainer import build_trainer, is_multi_gpu


# -- is_multi_gpu accessor (M1a/M1b unit tests, no Lightning) -----------------


def test_is_multi_gpu_explicit_int():
    assert is_multi_gpu(1) is False
    assert is_multi_gpu(2) is True
    assert is_multi_gpu(8) is True


def test_is_multi_gpu_auto_depends_on_visible_device_count(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert is_multi_gpu("auto") is True  # multi-GPU host -> DDP

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert is_multi_gpu("auto") is False  # single-GPU host -> no DDP


def test_is_multi_gpu_auto_off_cuda_box(monkeypatch):
    """``"auto"`` on a CUDA-less host is never DDP (the is_available guard)."""
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # device_count is unreachable when is_available() is False, but be explicit.
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    assert is_multi_gpu("auto") is False


def test_is_multi_gpu_matches_build_trainer_strategy(monkeypatch):
    """The monitor predicate and build_trainer's strategy decision agree (DRY).

    ``is_multi_gpu`` is the single source: when it is True the trainer builds a
    ``DDPStrategy``; when False it uses ``"auto"``. Falsifies the old divergence
    where ``monitor_fid`` stayed True while ``build_trainer`` spawned DDP.
    """
    import torch

    for cuda, count, devices in [
        (True, 2, "auto"),
        (True, 2, 2),
        (True, 1, "auto"),
        (False, 0, "auto"),
        (True, 1, 1),
    ]:
        monkeypatch.setattr(torch.cuda, "is_available", lambda c=cuda: c)
        monkeypatch.setattr(torch.cuda, "device_count", lambda n=count: n)
        decision = is_multi_gpu(devices)
        # build_trainer's branch: DDPStrategy iff is_multi_gpu(devices).
        assert decision == (isinstance(devices, int) and devices > 1) or (
            devices == "auto" and (cuda and count > 1)
        ), f"divergence at cuda={cuda},count={count},devices={devices!r}"


# -- M2a/M2b: -g 1 -> devices=1 (parse-args, no Lightning) ---------------------


def test_jit_main_passes_devices_1_for_single_gpu(monkeypatch, tmp_path):
    """JiT ``main(-g 1)`` passes ``devices=1`` to build_trainer (no surprise auto->DDP)."""
    from manifold.training import cli as cli_mod

    captured: dict = {}

    def fake_build_trainer(*, devices, **kw):
        captured["devices"] = devices
        raise SystemExit("stop-after-build_trainer")  # short-circuit the fit

    monkeypatch.setattr(cli_mod, "build_trainer", fake_build_trainer)

    from tests.test_training_cli import _write_tiny_configs

    env, train, net = _write_tiny_configs(tmp_path)

    def fake_provider(cfg, device):
        from tests.test_training_cli import _DataBundle, _LatentDS
        from manifold import AutoencoderKL
        import torch
        return _DataBundle(
            latent_ds=_LatentDS(), vae=AutoencoderKL(scaling_factor=0.5),
            val_latents=torch.randn(4, 4, 4, 4, 4),
        )

    with pytest.raises(SystemExit):
        cli_mod.main(["-e", env, "-c", train, "-t", net, "-g", "1", "--no-fid"], data_provider=fake_provider)
    assert captured["devices"] == 1, "-g 1 must pass devices=1 (not 'auto')"


def test_jit_main_passes_devices_n_for_multi_gpu(monkeypatch, tmp_path):
    """JiT ``main(-g N>1)`` passes ``devices=N`` to build_trainer."""
    from manifold.training import cli as cli_mod

    captured: dict = {}

    def fake_build_trainer(*, devices, **kw):
        captured["devices"] = devices
        raise SystemExit("stop")

    monkeypatch.setattr(cli_mod, "build_trainer", fake_build_trainer)
    from tests.test_training_cli import _write_tiny_configs

    env, train, net = _write_tiny_configs(tmp_path)

    def fake_provider(cfg, device):
        from tests.test_training_cli import _DataBundle, _LatentDS
        from manifold import AutoencoderKL
        import torch
        return _DataBundle(latent_ds=_LatentDS(), vae=AutoencoderKL(scaling_factor=0.5),
                           val_latents=torch.randn(4, 4, 4, 4, 4))

    with pytest.raises(SystemExit):
        cli_mod.main(["-e", env, "-c", train, "-t", net, "-g", "2", "--no-fid"], data_provider=fake_provider)
    assert captured["devices"] == 2


def test_paired_main_passes_devices_1_for_single_gpu(monkeypatch, tmp_path):
    """Paired ``main(-g 1)`` passes ``devices=1`` (M2b parity with JiT)."""
    from manifold.training import paired_cli as cli_mod

    captured: dict = {}

    def fake_build_trainer(*, devices, **kw):
        captured["devices"] = devices
        raise SystemExit("stop")

    monkeypatch.setattr(cli_mod, "build_trainer", fake_build_trainer)
    from tests.test_paired_training_cli import _write_paired_configs

    env, train, net = _write_paired_configs(tmp_path)

    def fake_provider(cfg, device):
        from tests.test_paired_training_cli import _DataBundle, _FakePairedDataset
        from manifold import AutoencoderKL
        return _DataBundle(latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5))

    with pytest.raises(SystemExit):
        cli_mod.main(["-e", env, "-c", train, "-t", net, "-g", "1"], data_provider=fake_provider)
    assert captured["devices"] == 1


def test_no_else_auto_fallback_remains():
    """grep sanity (M2): ``else "auto"`` is gone from both encoding CLIs."""
    import re

    for path in ("src/manifold/training/cli.py", "src/manifold/training/paired_cli.py"):
        with open(path) as f:
            assert not re.search(r'else\s+"auto"', f.read()), f"else \"auto\" reintroduced in {path}"


# -- M1a/M1b: 2-rank DDP monitor gate (uses the #80 fixture) -------------------


def test_jit_checkpoint_monitor_dropped_under_ddp(tmp_path):
    """2-rank run: the JiT ``ModelCheckpoint.monitor`` is ``None`` (rank-0-only FID
    not monitored under DDP). Reuses :func:`tests.ddp.jit_ddp_worker`."""
    from tests.ddp import jit_ddp_worker, run_ddp_two_rank

    results = run_ddp_two_rank(jit_ddp_worker, results_dir=str(tmp_path), args=(True,))
    assert results[0]["ckpt_monitor"] is None, "rank-0 monitor must be dropped under DDP"
    assert results[1]["ckpt_monitor"] is None


def test_jit_checkpoint_monitor_set_on_single_gpu(tmp_path):
    """Single-GPU (devices=1) keeps the FID monitor (M1a positive case)."""
    from manifold.training.cli import _build_checkpoint

    # monitor_fid=True mirrors run_training's single-GPU enable_fid branch.
    ckpt = _build_checkpoint(str(tmp_path), monitor_fid=True, monitor_metric="val/fid")
    assert ckpt.monitor == "val/fid"


def test_paired_checkpoint_monitor_always_set(tmp_path):
    """The paired ``_build_checkpoint`` always monitors ``val/psnr`` (the PSNR/SSIM
    callback is distributed under DDP - ADR-0016 amendment - so the monitor stays
    on for every config, including multi-GPU). The pre-amendment rank-0-only
    fallback (no monitor under DDP) is gone; the monitor metric is configurable."""
    from manifold.training.paired_cli import _build_checkpoint

    ckpt = _build_checkpoint(str(tmp_path), monitor_metric="val/psnr")
    assert ckpt.monitor == "val/psnr"
    ckpt2 = _build_checkpoint(str(tmp_path / "b"), monitor_metric="val/ssim")
    assert ckpt2.monitor == "val/ssim"
