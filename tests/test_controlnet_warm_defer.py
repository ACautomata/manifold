"""ControlNet paired-cache warm deferred into ``DataModule.setup()`` (issue #145).

Both ``controlnet_cli._real_inputs`` and ``paired_reward_cli`` used to warm the paired
latent cache in the CLI before ``Trainer.fit`` — i.e. before Lightning spawns DDP
workers, so every rank re-warmed (or raced on) the same cache on the parent's view.
The ControlNet path now defers the warm into ``DataModule.setup()`` (ADR-0017), so each
DDP rank warms its own shard after process spawn — matching ``paired_reward_cli``.

Gates:
- **single-GPU**: the cold path (``warm_fn`` set) warms exactly once inside
  ``DataModule.setup()`` and completes a fit (parity with the pre-warmed path).
- **DDP (2 CPU ranks)**: each rank encodes only its ``i % world == rank`` shard; the
  summed encode count equals the unique-volume count (no cross-rank double-encode),
  and the warm runs with a live process group (post-PG).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from tests.ddp import controlnet_cold_cache_ddp_worker, run_ddp_two_rank


class _ToyPairedVolDS(Dataset):
    """A tiny paired volume source (single-GPU cold-path gate)."""

    def __init__(self, n_pairs=4, n_volumes=6, seed=0):
        torch.manual_seed(seed)
        self._n = n_pairs
        self._ids = [f"v{i}" for i in range(n_volumes)]
        self._imgs = {sid: torch.randn(1, 4, 4, 4) for sid in self._ids}

    def __len__(self):
        return self._n

    def unique_sample_ids(self):
        return list(self._ids)

    def pair_meta(self, i):
        return {
            "src_id": self._ids[i % len(self._ids)],
            "tgt_id": self._ids[(i + 1) % len(self._ids)],
            "src_label": torch.tensor(0, dtype=torch.long),
            "tgt_label": torch.tensor(1, dtype=torch.long),
        }

    def _load_volume(self, sid):
        return {
            "image": self._imgs[sid],
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "label": 0,
        }


def test_single_gpu_warm_runs_once_inside_setup(tmp_path):
    """Single-GPU cold path: ``warm_fn`` runs exactly once, inside ``setup()`` (not at
    ``run_controlnet_training`` call time), and the fit completes."""
    from manifold import (
        ControlNet3DConditionModel,
        FlowMatchHeunDiscreteScheduler,
        UNet3DConditionModel,
    )
    from manifold.data.paired_latent_dataset import PairedLatentDataset
    from manifold.modules.controlnet_latent_flow import ControlNetLatentFlowModule
    from manifold.training.controlnet_cli import ControlNetInputs, run_controlnet_training

    warm_calls = [0]
    warmed_marker = {"armed": False}

    def counting_encode(images):
        return images.float().repeat(1, 4, 1, 1, 1)

    train_ds = PairedLatentDataset(
        _ToyPairedVolDS(n_pairs=4), encode_fn=None, cache_dir=str(tmp_path / "c"), cache_tag="t"
    )
    val_ds = PairedLatentDataset(
        _ToyPairedVolDS(n_pairs=2, seed=1), encode_fn=None, cache_dir=str(tmp_path / "c"), cache_tag="v"
    )

    def warm_fn():
        warm_calls[0] += 1
        warmed_marker["armed"] = True
        for ds in (train_ds, val_ds):
            ds.encode_fn = counting_encode
            ds.warm_cache(torch.device("cpu"), show_progress=False)
            ds.free_encoder()
        return train_ds, val_ds

    torch.manual_seed(0)
    base = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in base.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    controlnet.load_base_encoder_weights(base)
    module = ControlNetLatentFlowModule(
        base, controlnet, FlowMatchHeunDiscreteScheduler(),
        lr=1e-3, lr_warmup_steps=0, num_train_examples=4, train_batch_size=2, n_epochs=1,
    )
    inputs = ControlNetInputs(
        unet=base, controlnet=controlnet, scheduler=FlowMatchHeunDiscreteScheduler(),
        train_ds=train_ds, val_ds=val_ds, warm_fn=warm_fn,
    )
    # The warm has NOT run yet (it is deferred to setup()).
    assert warm_calls[0] == 0 and train_ds._ram is None

    trainer, _ = run_controlnet_training(
        module=module, inputs=inputs, model_dir=str(tmp_path / "m"),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2,
    )
    # setup() ran the warm exactly once (single-GPU parity: one warm, before training).
    assert warm_calls[0] == 1
    assert train_ds._ram is not None and val_ds._ram is not None
    assert int(trainer.global_step) > 0


def test_ddp_warm_is_per_rank_sharded(tmp_path):
    """2-rank DDP: the paired warm runs per-rank inside setup() (post-PG); each rank
    encodes its shard — the summed encode count equals 2×N (train N + val N unique
    volumes, encoded once each), NOT world× that (no cross-rank double-encode)."""
    n = 6
    results = run_ddp_two_rank(controlnet_cold_cache_ddp_worker, results_dir=str(tmp_path), args=(n,))
    r0, r1 = results
    # No-hang: both ranks completed a fit.
    assert r0["global_step"] > 0 and r1["global_step"] > 0
    # The warm ran with a live process group (post-PG) on BOTH ranks.
    assert r0["dist_at_warm"] is True and r1["dist_at_warm"] is True
    # Each rank encoded ~half of each split's unique set: train N + val N per rank's
    # shard sums to 2N total across ranks (each unique volume encoded once total).
    total = r0["encode_count"] + r1["encode_count"]
    assert total == 2 * n, f"encode sum {total} != 2N={2*n} (cross-rank double-encode)"
    # Per-rank count is in {floor, ceil} of each split's half (train N + val N → ~N each).
    import math

    lo, hi = 2 * (n // 2), 2 * math.ceil(n / 2)
    for r in (r0, r1):
        assert lo <= r["encode_count"] <= hi, (
            f"rank {r['rank']} encoded {r['encode_count']} (not in [{lo},{hi}])"
        )
