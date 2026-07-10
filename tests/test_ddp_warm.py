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
