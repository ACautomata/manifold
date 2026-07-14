"""DDP sharded VAE warm via DataModule.setup() (issue #84, ADR-0017).

The VAE-encode warm moves from ``main()`` (pre-PG) to a Lightning
``DataModule.setup()`` (post-PG) so the per-rank ``i % world == rank`` sharding
machinery in :meth:`~manifold.data.LatentDataset.warm_cache` activates - one
writer per cache file instead of every rank re-encoding the full set (the ~2.7h
cold-start cost). ``warm_latent_pipeline`` / ``warm_cache`` derive rank/world from
``dist`` (F3); the FIDCallback's ``real_latents`` and ``_inference_recipe``'s
``latent_shape`` go lazy (F5) because they no longer exist at ``run_training``
construction time.

Gates (2-rank, reuse :func:`tests.ddp.run_ddp_two_rank`):
- **F1 cold-cache**: the SUM of per-rank encode-call counts across ranks == N
  (= len(vol_ds)); each rank's count is in {floor(N/world), ceil(N/world)}.
- **F3**: ``warm_latent_pipeline`` uses dist-derived rank/world (sharded branch
  taken post-PG, NOT taken pre-PG).
- **F4**: ``dist.is_initialized()`` is True at warm-execution time (the warm moved
  post-PG).
- **F5**: ``FIDCallback.real_latents`` is None at construction and populated before
  the first validation epoch; ``_inference_recipe``'s ``latent_shape`` no longer
  eagerly requires ``val_latents``.
- **SG1**: single-GPU parity preserved (the warm refactor is a no-op on the warmed
  test path - 300 tests green).

GRPO multi-GPU training is NOT blocked (codex G2=FALSE); GRPO loads pre-computed
``.pt`` latents and is unaffected by this warm refactor.
"""

from __future__ import annotations

import inspect

import pytest

from tests.ddp import cold_cache_ddp_worker, run_ddp_two_rank


# -- F1/F4: cold-cache 2-rank sharded warm ------------------------------------


def test_f1_cold_cache_encodes_each_volume_once(tmp_path):
    """2-rank cold-cache: the SUM of per-rank encode-call counts == N (each volume
    encoded ONCE total, not world-N redundant encodes); each rank's count is in
    ``{floor(N/world), ceil(N/world)}`` (the strided ``range(rank, n, world)`` branch
    gives ceil to the first ``N%world`` ranks only - do NOT assert == ceil on every).

    F4: ``dist.is_initialized()`` is True at warm-execution time (the warm moved
    post-PG). The fit completes with exit 0 (no hang - the NCCL barrier inside
    ``setup()`` is symmetric across ranks).
    """
    n = 6
    results = run_ddp_two_rank(cold_cache_ddp_worker, results_dir=str(tmp_path), args=(n,))
    r0, r1 = results
    # No-hang: both ranks completed.
    assert r0["global_step"] > 0 and r1["global_step"] > 0
    # F1: each volume encoded once total (sum == N).
    total = r0["encode_count"] + r1["encode_count"]
    assert total == n, f"encode sum {total} != N={n} (redundant encoding)"
    # Each rank's count is in {floor(N/world), ceil(N/world)}.
    import math

    lo, hi = n // 2, math.ceil(n / 2)
    for r in (r0, r1):
        assert lo <= r["encode_count"] <= hi, (
            f"rank {r['rank']} encoded {r['encode_count']} (not in [{lo},{hi}])"
        )
    # F4: the warm ran with a live process group (post-PG) - the load-bearing
    # proof. (``dist.is_initialized()`` is also False in production ``main()``
    # both before ``fit`` and after it returns - Lightning owns the PG lifecycle;
    # verified separately on the single-device path. The 2-rank worker keeps the
    # PG alive until ``ddp_fini``, so ``dist_at_exit`` is True here by construction.)
    assert r0["dist_at_warm"] is True, "dist not initialized at warm (F4: warm pre-PG)"
    assert r1["dist_at_warm"] is True


def test_f1_warm_cache_0_encode_on_hits(tmp_path):
    """2-rank WARM-cache (disk cache pre-populated): 0 encode calls (all hits).
    Pairs with the cold-cache gate to prove the sharded branch is encode-only-once."""
    n = 6
    # First run: cold (populates the cache).
    run_ddp_two_rank(cold_cache_ddp_worker, results_dir=str(tmp_path), args=(n,))
    # Second run: warm (same cache_dir -> all hits).
    results = run_ddp_two_rank(cold_cache_ddp_worker, results_dir=str(tmp_path), args=(n,))
    for r in results:
        assert r["encode_count"] == 0, f"rank {r['rank']} encoded {r['encode_count']} on a warm cache"


# -- F3: rank/world derived from dist ------------------------------------------


def test_f3_warm_latent_pipeline_derives_rank_world_from_dist():
    """``warm_latent_pipeline`` and both ``warm_cache`` methods derive rank/world from
    ``dist`` when the PG is initialized (fallback 0/1) - the post-PG
    ``DataModule.setup()`` path relies on this. Verified by source: a
    ``dist.is_initialized()`` check precedes the sharded branch."""
    from manifold.data import latent_pipeline
    from manifold.data.latent_dataset import LatentDataset
    from manifold.data.paired_latent_dataset import PairedLatentDataset

    src = inspect.getsource(latent_pipeline.warm_latent_pipeline)
    assert "dist.is_initialized()" in src, "warm_latent_pipeline missing dist check (F3)"
    src = inspect.getsource(LatentDataset.warm_cache)
    assert "dist.is_initialized()" in src, "LatentDataset.warm_cache missing dist check (F3)"
    src = inspect.getsource(PairedLatentDataset.warm_cache)
    assert "dist.is_initialized()" in src, "PairedLatentDataset.warm_cache missing dist check (F3)"


def test_f3_no_rank_world_kwargs_at_cli_call_sites():
    """grep sanity (F3): the CLI warm call sites no longer pass ``rank=`` / ``world=``
    (the warm derives them from dist)."""
    import re

    for path in ("src/manifold/training/cli.py", "src/manifold/training/paired_cli.py"):
        with open(path) as f:
            body = f.read()
        # The warm_fn closures call warm_latent_pipeline / warm_cache WITHOUT
        # rank=/world= kwargs (F3 removed them from the call sites).
        assert "rank=rank" not in body, f"{path} still passes rank=rank (F3)"
        assert "world=world" not in body, f"{path} still passes world=world (F3)"


# -- F5: lazy real_latents + latent_shape --------------------------------------


def test_f5_fidcallback_real_latents_lazy():
    """``FIDCallback`` accepts ``real_latents=None`` at construction and pulls it
    lazily from ``real_latents_source`` at the first ``_real_features`` call (F5 -
    the val reference does not exist until ``DataModule.setup()`` runs)."""
    from manifold.metrics.fid_callback import FIDCallback

    sig = inspect.signature(FIDCallback.__init__)
    assert "real_latents_source" in sig.parameters, "FIDCallback missing real_latents_source"
    # The lazy pull lives in _real_features.
    src = inspect.getsource(FIDCallback._real_features)
    assert "real_latents is None" in src, "_real_features does not lazy-pull real_latents (F5)"


def test_f5_inference_recipe_latent_shape_is_lazy():
    """``_inference_recipe`` accepts an explicit ``latent_shape`` (does NOT eagerly
    require ``val_latents`` at construction - F5). The cold path passes it derived
    from the VAE stride + target_dim; the warmed path derives from val_latents."""
    import torch

    from manifold.training.cli import _inference_recipe

    # Explicit latent_shape (cold path, val_latents=None).
    inf = _inference_recipe(_tiny_module_for_recipe(), cfg=None, val_latents=None, latent_shape=(1, 4, 7, 9, 11))
    assert inf["latent_shape"] == (1, 4, 7, 9, 11)
    # Warmed path: latent_shape derived from val_latents.
    inf2 = _inference_recipe(_tiny_module_for_recipe(), cfg=None, val_latents=torch.randn(6, 4, 7, 9, 11))
    assert inf2["latent_shape"] == (1, 4, 7, 9, 11)


def _tiny_module_for_recipe():
    from manifold import FlowMatchHeunDiscreteScheduler, LatentFlowModule, UNet3DConditionModel
    import torch

    torch.manual_seed(0)
    return LatentFlowModule(
        UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True),
        FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0,
        num_train_examples=6, train_batch_size=2, n_epochs=1,
    )


# -- F4: build_trainer's find_unused_parameters=True preserved -----------------


def test_f4_find_unused_parameters_preserved():
    """``build_trainer`` keeps ``find_unused_parameters=True`` (the paired summed-label
    / ``_PinnedClassEmbedding`` shim depends on it - removing it deadlocks the
    paired CLI on its first multi-GPU run)."""
    from manifold.training.trainer import build_trainer

    src = inspect.getsource(build_trainer)
    assert "find_unused_parameters=True" in src, (
        "build_trainer dropped find_unused_parameters=True (paired DDP deadlock risk)"
    )


# -- P1 (codex #85): warm uses the per-rank local CUDA device -----------------


def test_p1_resolve_warm_device_returns_local_rank_under_ddp(monkeypatch):
    """``resolve_warm_device`` returns ``cuda:{local_rank}`` under DDP (post-PG),
    NOT the launch-time ``cuda:0``. P1: the warm_fn captured the launch-time device
    (the default cuda:0 before LOCAL_RANK is known), so under DDP every rank would
    warm on GPU 0. Locks the LOCAL_RANK-derived device resolution."""
    import torch

    from manifold.data.latent_pipeline import resolve_warm_device

    # No PG initialized -> the fallback device is returned unchanged (single-process).
    assert resolve_warm_device(torch.device("cpu")) == torch.device("cpu")
    assert resolve_warm_device(torch.device("cuda")) == torch.device("cuda")

    # PG initialized -> cuda:{LOCAL_RANK}. Mock dist.is_initialized + the env var.
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert resolve_warm_device(torch.device("cuda")) == torch.device("cuda:3")
    # A non-CUDA fallback is returned unchanged even under DDP (warm stays CPU).
    assert resolve_warm_device(torch.device("cpu")) == torch.device("cpu")


def test_p1_warm_fn_uses_local_rank_device_not_launch_device():
    """The JiT + paired ``_warm_data`` warm_fn rebuild the encode_fn on the
    per-rank device (P1): the VAE is built on CPU pre-PG and re-staged inside
    warm_fn via ``resolve_warm_device`` + ``make_encode_fn``. Source-level guard so
    the launch-time ``device=device`` capture cannot sneak back into warm_fn."""
    from manifold.data.latent_pipeline import make_encode_fn, resolve_warm_device
    from manifold.training import cli as jit_cli
    from manifold.training import paired_cli

    for mod, name in [(jit_cli, "JiT"), (paired_cli, "Paired")]:
        src = inspect.getsource(mod._warm_data)
        assert "resolve_warm_device" in src, f"{name} _warm_data missing resolve_warm_device (P1)"
        assert "make_encode_fn" in src, f"{name} _warm_data missing make_encode_fn rebuild (P1)"
        # The VAE is built on CPU pre-PG (no GPU-0 placement before LOCAL_RANK).
        assert 'torch.device("cpu")' in src, f"{name} _warm_data must build the VAE on CPU pre-PG (P1)"
    assert callable(make_encode_fn)
    assert callable(resolve_warm_device)



# -- P2 (codex #85 re-review): fail-safe lazy factory + graceful skip ----------


def test_p2_feature_net_factory_failure_skips_fid_and_logs_sentinel(tmp_path):
    """A fail-safe feature_net factory (bad cache / no network -> None) makes
    FIDCallback SKIP FID and log the monitored metrics as +inf (so ModelCheckpoint
    mode='min' does not crash on a never-logged metric) instead of aborting the run.

    P2 (codex #85 re-review): the previous cache-check pre-disable either crashed
    at eval on a bad cache or wrongly disabled FID when the online fallback could
    work. The factory is now try/except -> None; the callback handles None.
    """
    import torch

    from manifold import AutoencoderKL, FlowMatchHeunDiscreteScheduler, LatentFlowModule, UNet3DConditionModel
    from manifold.metrics import FIDCallback

    def _failing_factory():
        # Mimic the real fail-safe factory in main(): try make_feature_network,
        # except -> None (FIDCallback handles None, never a raise).
        return None

    torch.manual_seed(0)
    module = LatentFlowModule(
        UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True),
        FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0,
        num_train_examples=6, train_batch_size=2, n_epochs=1,
    )
    fid = FIDCallback(
        module=module, vae=AutoencoderKL(scaling_factor=0.5),
        real_latents=torch.randn(2, 4, 4, 4, 4),
        feature_net_factory=_failing_factory,  # returns None (raises -> caught)
        latent_shape=(1, 4, 4, 4, 4), spacing=[1.0, 1.0, 1.0], modality=1,
        num_inference_steps=2, num_synth=2, cov_ridge=1e-2, seed=0,
    )

    # A fake trainer with is_global_zero + a sane world-size + current_epoch.
    class _Tr:
        is_global_zero = True
        current_epoch = 0
    fid._stage_eval_on_device()
    assert getattr(fid, "_fid_disabled", False) is True, "factory failure must set _fid_disabled"
    assert fid.feature_net is None

    # on_validation_epoch_end logs sentinels (inf) for the monitored metrics, no raise.
    logged = {}
    module.log = lambda key, value, **k: logged.__setitem__(key, value)  # type: ignore[assignment]
    fid.on_validation_epoch_end(_Tr(), module)
    assert logged.get("val/fid") == float("inf")


def test_p2_raising_factory_caught_and_skips_fid(tmp_path):
    """A RAISING factory (a bad/corrupt cache the fail-safe wrapper in main missed,
    or a direct caller's non-fail-safe factory) is caught IN FIDCallback (codex #85
    re-review): the call site itself is wrapped, so it never aborts training mid-fit.

    This is the case codex flagged - the previous fix made the FACTORY fail-safe but
    left the call site unwrapped, relying on every caller passing a fail-safe factory.
    FIDCallback is now self-contained.
    """
    import torch

    from manifold import AutoencoderKL, FlowMatchHeunDiscreteScheduler, LatentFlowModule, UNet3DConditionModel
    from manifold.metrics import FIDCallback

    def _raising_factory():
        raise RuntimeError("simulated corrupt checkpoint / version mismatch")

    torch.manual_seed(0)
    module = LatentFlowModule(
        UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True),
        FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0,
        num_train_examples=6, train_batch_size=2, n_epochs=1,
    )
    fid = FIDCallback(
        module=module, vae=AutoencoderKL(scaling_factor=0.5),
        real_latents=torch.randn(2, 4, 4, 4, 4),
        feature_net_factory=_raising_factory,
        latent_shape=(1, 4, 4, 4, 4), spacing=[1.0, 1.0, 1.0], modality=1,
        num_inference_steps=2, num_synth=2, cov_ridge=1e-2, seed=0,
    )

    class _Tr:
        is_global_zero = True
        current_epoch = 0
    # The raising factory is caught -> no exception escapes _stage_eval_on_device.
    fid._stage_eval_on_device()
    assert getattr(fid, "_fid_disabled", False) is True
    assert fid.feature_net is None

    logged = {}
    module.log = lambda key, value, **k: logged.__setitem__(key, value)  # type: ignore[assignment]
    fid.on_validation_epoch_end(_Tr(), module)  # no raise; logs inf sentinels
    assert logged.get("val/fid") == float("inf")

    # codex #85 re-review P2: the skip-path early return must NOT leave the VAE on
    # the training GPU. on_validation_epoch_end's finally -> _restore_eval_to_cpu
    # restores it to CPU when _eval_staged is True (set before the early return).
    assert fid._eval_staged is False, "_eval_staged not reset by the restore"
    assert next(fid.vae.parameters()).device.type == "cpu", "VAE left on GPU after the FID skip"
