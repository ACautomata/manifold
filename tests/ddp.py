"""CPU 2-rank DDP test harness (issue #80).

A reusable ``run_ddp_two_rank`` runner + module-level gloo workers so every
downstream DDP-correctness criterion shares identical rank/world semantics.
Spawns ``world_size`` processes via :func:`torch.multiprocessing.spawn`; each
worker inits a gloo process group, runs a tiny Lightning fit, and writes a
per-rank result JSON the calling test asserts on.

Why spawn + manual ``init_process_group("gloo")`` (not ``torchrun``):
``torchrun`` launches a fresh interpreter whose rendezvous is flaky on macOS
CI, and it re-runs the whole script. ``mp.spawn`` inherits the parent's
``sys.path`` (so ``manifold`` and the tiny-config builders below import cleanly
in the child) and lets the worker control Lightning construction between PG
init/teardown. Lightning's ``DDPStrategy(process_group_backend="gloo")``
reuses the already-initialized PG (no re-spawn). Verified end-to-end on this
CPU env: a 2-rank fit reaches equal ``global_step`` on both ranks and a
``sync_dist``-logged metric is identical across ranks.

Picklability: ``mp.spawn`` pickles the worker by reference, so every worker
AND the tiny-config builders here are module-level functions. Tests must pass
a module-level worker (one defined here) - not a closure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import Dataset

# Lightning is imported lazily inside workers (keeps ``import tests.ddp`` light
# for tests that only want the runner), but the tiny builders below need it.

_DEFAULT_PORT = 29600


def _free_port() -> str:
    """A port keyed off the pid so sequential DDP tests in one session do not collide."""
    return str(_DEFAULT_PORT + (os.getpid() % 400))


def ddp_init(rank: int, world: int, port: str | None = None) -> None:
    """Init a gloo process group for *rank* in a spawned worker.

    Idempotent: a worker that is re-entered (it is not, but be safe) will not
    double-init. Sets the env vars Lightning's DDPStrategy reads so it reuses
    this PG instead of launching its own.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port or _free_port()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(rank)
    if not dist.is_initialized():
        dist.init_process_group("gloo", rank=rank, world_size=world)


def ddp_fini() -> None:
    """Barrier + destroy the process group (symmetric with :func:`ddp_init`)."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def run_ddp_two_rank(
    worker: Callable[..., None],
    *,
    world_size: int = 2,
    results_dir: str | Path,
    args: tuple = (),
) -> list[dict[str, Any]]:
    """Spawn ``world_size`` CPU gloo workers and return their per-rank results.

    Each worker is ``worker(rank, world, results_dir, *args)``; it MUST be a
    module-level function (picklable). The worker is responsible for calling
    :func:`ddp_init` / :func:`ddp_fini` around its Lightning fit, and for writing
    ``results_dir/r{rank}.json`` (a JSON-serializable dict) before it returns.
    Returns the list of result dicts ordered by rank.

    The whole spawn runs synchronously (``join=True``); a hang surfaces as a
    pytest timeout (the no-deadlock gate downstream relies on this).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    os.environ["MANIFOLD_DDP_PORT"] = port
    mp.spawn(worker, args=(world_size, str(results_dir), port, *args), nprocs=world_size, join=True)
    return [json.loads((results_dir / f"r{rank}.json").read_text()) for rank in range(world_size)]


# -- tiny-config builders (module-level so spawned workers can import them) ----


class _FakeFeatureNet(nn.Module):
    """A tiny stand-in for the RadImageNet ResNet50 FID backbone.

    Maps a 2.5D plane ``[K, C, h, w]`` to ``[K, 8]`` so the FID math has a
    non-degenerate feature space without a 100 MB ``torch.hub`` download.
    Mirrors the fake in ``tests/test_training_cli.py``.
    """

    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        b = plane.shape[0]
        flat = plane.reshape(b, -1)[:, :8]
        if flat.shape[1] < 8:
            flat = torch.nn.functional.pad(flat, (0, 8 - flat.shape[1]))
        return flat


class _LatentDS(Dataset):
    """Tiny in-RAM latent dataset (mirrors ``tests/test_training_cli.py``)."""

    def __init__(self, n: int = 6, *, seed: int = 0):
        torch.manual_seed(seed)
        self.items = [
            {
                "latent": torch.randn(4, 4, 4, 4),
                "spacing": torch.tensor([1.0, 1.0, 1.0]),
                "label": torch.tensor(i % 3, dtype=torch.long),
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _tiny_jit_module():
    """A tiny trainable JiT module sized for a fast CPU DDP fit."""
    from manifold import FlowMatchHeunDiscreteScheduler, LatentFlowModule, UNet3DConditionModel

    torch.manual_seed(0)
    return LatentFlowModule(
        UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True),
        FlowMatchHeunDiscreteScheduler(),
        lr=1e-2,
        lr_warmup_steps=0,
        num_train_examples=6,
        train_batch_size=2,
        n_epochs=1,
    )


def _jit_callbacks(module, *, enable_fid: bool, devices, seed: int = 0):
    """Build the JiT callback stack (train metrics + optional FID + ckpt).

    ``devices`` is the value passed to ``build_trainer``. Under ADR-0025 the FID
    monitor STAYS ON under DDP (val/fid is global), mirroring ``run_training``.
    """
    from manifold import AutoencoderKL
    from manifold.metrics import FIDCallback
    from manifold.training.cli import _build_checkpoint, _inference_recipe
    from manifold.training.metrics import LatentX0MAE, TrainLossLogger

    callbacks: list = [TrainLossLogger(), LatentX0MAE()]
    val_latents = torch.randn(5, 4, 4, 4, 4)
    inf = _inference_recipe(module, cfg=None, val_latents=val_latents)
    if enable_fid:
        fid = FIDCallback(
            module=module,
            vae=AutoencoderKL(scaling_factor=0.5),
            real_latents=val_latents,
            feature_net=_FakeFeatureNet(),
            latent_shape=inf["latent_shape"],
            spacing=inf["spacing"],
            modality=inf["modality"],
            num_inference_steps=inf["num_inference_steps"],
            guidance_scale=inf["guidance_scale"],
            cfg_interval=inf["cfg_interval"],
            num_synth=2,
            every_n_epochs=1,
            center_slices_ratio=0.5,
            cov_ridge=1e-2,
            seed=seed,
        )
        callbacks.append(fid)
    ckpt = _build_checkpoint(
        model_dir="/tmp/_unused_ckpt_dir",
        monitor_fid=enable_fid,
        every_n_epochs=1,
    )
    callbacks.append(ckpt)
    return callbacks, ckpt


# -- generic JiT 2-rank worker (serves #80 fixture + #82 + #81 monitor gate) ----


def jit_ddp_worker(rank: int, world: int, results_dir: str, port: str, enable_fid: bool) -> None:
    """Run a tiny 2-rank JiT fit; dump per-rank metrics + grad_norm.

    Captures everything the downstream 2-rank gates need in one fit:
    ``train/loss_epoch`` / ``val/x0_mae`` (M6 #82), ``train/grad_norm`` (G3 #82),
    the val-loader first-batch checksum (D1 #82), and the checkpoint
    ``monitor`` (M1a #81).
    """
    import lightning.pytorch as pl
    from lightning.pytorch.strategies import DDPStrategy

    from manifold.data.datamodule import build_datamodule
    from manifold.training.trainer import build_trainer

    ddp_init(rank, world, port)
    try:
        torch.manual_seed(0)
        module = _tiny_jit_module()
        callbacks, ckpt = _jit_callbacks(module, enable_fid=enable_fid, devices=world)
        datamodule = build_datamodule(
            _LatentDS(n=6), batch_size=2, num_workers=0, allow_train_as_val=True
        )
        trainer = build_trainer(
            max_epochs=1,
            callbacks=callbacks,
            model_dir=results_dir,
            devices=world,
            accelerator="cpu",
            limit_val_batches=2,
        )

        # D1: capture the first val-batch latent sum on each rank + the val
        # loader's sampler type. Under DDP Lightning wraps the val loader in a
        # DistributedSampler -> the sums DIFFER across ranks (a shard each). A
        # non-wrapped loader would give identical sums (the regression D1 guards).
        first_val_sum = [None]
        sampler_type = [None]

        class _D1Capture(pl.Callback):
            def on_validation_batch_end(self, tr, pl_module, outputs, batch, batch_idx, *a, **k):
                if first_val_sum[0] is None:
                    first_val_sum[0] = float(batch["latent"].sum().item())

            def on_validation_epoch_start(self, tr, pl_module):
                if sampler_type[0] is None:
                    try:
                        vls = tr.val_dataloaders
                        loader = vls[0] if isinstance(vls, (list, tuple)) else getattr(vls, "loaders", [vls])[0]
                        sampler_type[0] = type(loader.sampler).__name__
                    except Exception:  # noqa: BLE001 - diagnostic only
                        sampler_type[0] = "unknown"

        trainer.callbacks.append(_D1Capture())
        trainer.fit(module, datamodule=datamodule)

        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        result = {
            "rank": rank,
            "world": world,
            "global_step": int(trainer.global_step),
            "is_global_zero": bool(trainer.is_global_zero),
            "metrics": metrics,
            "ckpt_monitor": ckpt.monitor,
            "first_val_sum": first_val_sum[0],
            "val_sampler": sampler_type[0],
            "dist_initialized_at_fit": True,  # the warm/metrics ran with a live PG
        }
        Path(results_dir, f"r{rank}.json").write_text(json.dumps(result))
    finally:
        ddp_fini()


def _unbalanced_val_worker(rank: int, world: int, results_dir: str, port: str, _unused: bool) -> None:
    """2-rank fit with an INTENTIONALLY unbalanced val shard (M6 #82).

    A 5-sample val set under DistributedSampler(batch_size=2, world=2) gives rank 0
    indices [0,2,4] (3 samples -> 2 batches) and rank 1 indices [1,3] (2 samples
    -> 1 batch). The gate asserts ``val/x0_mae`` is IDENTICAL on both ranks and
    equals the true sample-weighted global mean ``(sum0·B0 + sum1·B1) / (B0+B1)``
    - which naive ``sync_dist`` (a mean-of-per-rank-means) would get WRONG. This
    locks the property against a future non-padding sampler. Also captures the
    per-rank val-batch count to confirm the shard is actually unbalanced.
    """
    import lightning.pytorch as pl
    from lightning.pytorch.strategies import DDPStrategy

    from manifold.data.datamodule import build_datamodule
    from manifold.training.trainer import build_trainer
    from manifold.training.metrics import LatentX0MAE, TrainLossLogger

    ddp_init(rank, world, port)
    try:
        torch.manual_seed(0)
        module = _tiny_jit_module()
        # 5-sample val set -> equal padded rank forwards (3 each), with the tagged
        # padding mask preserving real counts (rank0=3, rank1=2). Disable FID.
        callbacks: list = [TrainLossLogger(), LatentX0MAE()]
        from manifold.training.cli import _build_checkpoint
        ckpt = _build_checkpoint(model_dir="/tmp/_unused_ckpt_dir", monitor_fid=False, every_n_epochs=1)
        callbacks.append(ckpt)
        datamodule = build_datamodule(_LatentDS(n=6), batch_size=2, num_workers=0,
                                     val_dataset=_LatentDS(n=5, seed=99))
        trainer = build_trainer(max_epochs=1, callbacks=callbacks, model_dir=results_dir,
                                devices=world, accelerator="cpu", limit_val_batches=1.0,
                                extra_kwargs={"num_sanity_val_steps": 0})

        # Capture the raw per-sample MAEs + batch sizes so the test can compute
        # the true global weighted mean and compare to the logged val/x0_mae.
        per_batch = []

        class _Capture(pl.Callback):
            def on_validation_batch_end(self, tr, pl_module, outputs, batch, batch_idx, *a, **k):
                if isinstance(outputs, dict) and "pred" in outputs and "target" in outputs:
                    per_sample = (outputs["pred"] - outputs["target"]).abs().flatten(1).mean(1)
                    valid = ~batch["_is_padding"].bool()
                    if bool(valid.any()):
                        per_batch.append((float(per_sample[valid].mean()), int(valid.sum())))

        trainer.callbacks.append(_Capture())
        trainer.fit(module, datamodule=datamodule)
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        Path(results_dir, f"r{rank}.json").write_text(json.dumps({
            "rank": rank, "metrics": metrics,
            "val_batch_count": len(per_batch),
            "val_samples": sum(b for _, b in per_batch),
            "per_batch": per_batch,
            "is_global_zero": bool(trainer.is_global_zero),
        }))
    finally:
        ddp_fini()


# -- GRPO 2-rank worker (M3 rank-0 val gate + L3 lazy feature_net) ------------


def _grpo_inputs_factory(feature_net_factory):
    """Build the toy GRPO inputs (fake policy + frozen reward + toy conditioning).

    Passes ``feature_net_factory`` so L3's lazy build runs only on rank 0.
    """
    from manifold import FlowMatchGRPOScheduler, RewardModel, UNet3DConditionModel
    from manifold.training.grpo_cli import GRPOInputs

    torch.manual_seed(0)
    policy = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    rm = RewardModel(spatial_dims=3, in_channels=4, channels=8, num_layers_d=1)
    return GRPOInputs(
        policy=policy, reward_model=rm, scheduler=FlowMatchGRPOScheduler(eta=0.5),
        train_ds=_ToyCondDS(), val_ds=_ToyCondDS(), latent_shape=(4, 8, 8, 8),
        vae=_ToyVAE(), real_latents=torch.randn(2, 4, 8, 8, 8),
        feature_net_factory=feature_net_factory,
    )


class _ToyVAE(nn.Module):
    """A no-op decode stand-in so the GRPO FID path runs on CPU without a real VAE.

    The FID callback decodes real + synthetic latents; the fake feature_net is the
    signal under test, so the VAE just needs to produce finite images of the right
    channel count. ``decode`` returns ``[B, 1, D, H, W]`` float32. Carries one
    dummy parameter so ``next(vae.parameters()).device`` resolves (FIDCallback reads
    the device for staging).
    """

    scaling_factor = 1.0

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))

    def decode(self, latents):
        x = latents[:, :1].float()
        return x.repeat(1, 1, *(s for s in latents.shape[2:])) if latents.dim() == 5 else x

    def eval(self):
        return self

    def state_dict(self, *a, **k):
        return {"_dummy": self._dummy.detach()}

    def load_state_dict(self, state_dict, strict=True):
        # FIDCallback clones the pre-stage state_dict then restores it; accept
        # whatever it stashed (the dummy is invariant across the round-trip).
        return

    def modules(self):
        return iter([])


class _ToyCondDS(Dataset):
    """Tiny conditioning dataset (train/val): emits {spacing, label}."""

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"spacing": torch.tensor([1.0, 1.0, 1.0]), "label": torch.tensor(1, dtype=torch.long)}


def grpo_ddp_worker(rank: int, world: int, results_dir: str, port: str, _unused: bool) -> None:
    """Run a 2-rank GRPO fit; capture the M3 rank-0-only val gate + L3 lazy build.

    M3 (ADR-0016): ``validation_step`` must run generation + scoring on
    ``is_global_zero`` ONLY -> ``sample_latent_flow`` + ``reward_model`` call
    counts are ``>0`` on rank 0 and exactly ``0`` on rank 1, and
    ``val/mean_reward`` is logged exactly once (rank 0). The rank-asymmetric
    early-return must not deadlock (the no-hang gate). L3: ``feature_net_factory``
    is invoked only on rank 0 (the FID stage path is rank-0-gated).
    """
    import lightning.pytorch as pl

    from manifold.modules import grpo as grpo_mod
    from manifold.modules.grpo import GRPOModule
    from manifold.training.grpo_cli import run_grpo_training

    ddp_init(rank, world, port)
    try:
        # L3: a counting factory so the test asserts call_count == 1 on rank 0,
        # == 0 on rank 1 (lazy build only on the rank-0 stage path).
        build_count = [0]

        def factory():
            build_count[0] += 1
            return _FakeFeatureNet()

        inputs = _grpo_inputs_factory(feature_net_factory=factory)
        torch.manual_seed(0)
        module = GRPOModule(
            inputs.policy, inputs.reward_model, inputs.scheduler,
            G=2, eta_step_list=(0,), num_steps=3, latent_shape=inputs.latent_shape, lr=1e-3,
        )

        # M3: spy on sample_latent_flow + reward_model to count per-rank calls.
        sample_calls = [0]
        reward_calls = [0]
        orig_sample = grpo_mod.sample_latent_flow
        orig_reward_forward = type(inputs.reward_model).forward

        def sample_spy(*a, **k):
            sample_calls[0] += 1
            return orig_sample(*a, **k)

        def reward_spy(self, *a, **k):
            reward_calls[0] += 1
            return orig_reward_forward(self, *a, **k)

        grpo_mod.sample_latent_flow = sample_spy
        type(inputs.reward_model).forward = reward_spy
        try:
            trainer, _ckpt = run_grpo_training(
                module=module, inputs=inputs, model_dir=results_dir, max_epochs=1,
                devices=world, accelerator="cpu", batch_size=2, seed=0,
                limit_val_batches=1.0,
            )
        finally:
            grpo_mod.sample_latent_flow = orig_sample
            type(inputs.reward_model).forward = orig_reward_forward

        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        Path(results_dir, f"r{rank}.json").write_text(json.dumps({
            "rank": rank,
            "is_global_zero": bool(trainer.is_global_zero),
            "sample_latent_flow_calls": sample_calls[0],
            "reward_model_calls": reward_calls[0],
            "feature_net_builds": build_count[0],
            "val_mean_reward_logged": "val/mean_reward" in metrics,
            "val_mean_reward": metrics.get("val/mean_reward"),
            "global_step": int(trainer.global_step),
        }))
    finally:
        ddp_fini()


# -- cold-cache 2-rank worker (F1/F2/F4 sharded warm) ------------------------


class _CountingVolDS(Dataset):
    """A fake image-emitting volume dataset that records encode-call counts per rank.

    Emits ``{"image", "spacing", "label", "sample_id"}`` (the LatentDataset contract)
    so :class:`LatentDataset` can encode through ``encode_fn``. The encode_fn itself
    (a no-op identity) counts how many times EACH rank invokes it - the F1 gate
    asserts the SUM of per-rank counts == N (each volume encoded once total, not
    world-N redundant encodes) and each rank's count is in ``{floor(N/world), ceil}``.
    """

    def __init__(self, n: int = 6, *, seed: int = 0):
        torch.manual_seed(seed)
        self._n = n
        self._imgs = [torch.randn(1, 4, 4, 4) for _ in range(n)]

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return {
            "image": self._imgs[i],
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "label": torch.tensor(i % 3, dtype=torch.long),
            "sample_id": f"vol_{i:03d}",
        }

    def sample_ids(self):
        return [f"vol_{i:03d}" for i in range(self._n)]


def cold_cache_ddp_worker(rank: int, world: int, results_dir: str, port: str, n_volumes: int) -> None:
    """Run a 2-rank cold-cache JiT fit via the deferred ``DataModule.setup()`` warm.

    F1/F4 (ADR-0017): the warm runs in ``setup()`` (post-PG); the
    ``i % world == rank`` sharded branch activates so each rank encodes only its
    shard. Captures per-rank encode-call count + ``dist.is_initialized()`` at warm
    time (True) vs at worker exit (False - proves the warm moved post-PG).
    """
    import lightning.pytorch as pl

    from manifold import AutoencoderKL
    from manifold.data.latent_dataset import LatentDataset
    from manifold.data.warm_datamodule import LatentWarmDataModule
    from manifold.training.cli import _build_checkpoint
    from manifold.training.metrics import LatentX0MAE, TrainLossLogger
    from manifold.training.trainer import build_trainer

    ddp_init(rank, world, port)
    encode_count = [0]
    dist_at_warm = [None]

    def counting_encode(images):
        encode_count[0] += 1
        # image is [1, 1, D, H, W]; emit a [1, C_latent=4, D, H, W] latent (the
        # LatentDataset contracts 4 latent channels). Repeat the single image
        # channel to 4 (a no-op encode that still produces a valid latent shape).
        return images.float().repeat(1, 4, 1, 1, 1)

    vol_ds = _CountingVolDS(n=n_volumes)
    vae = AutoencoderKL(scaling_factor=0.5)

    def warm_fn():
        dist_at_warm[0] = __import__("torch").distributed.is_initialized()
        from manifold.data.latent_pipeline import warm_latent_pipeline

        return warm_latent_pipeline(
            vol_ds, counting_encode, vae,
            cache_dir=str(__import__("pathlib").Path(results_dir) / "cache"),
            cache_tag="cold_test", device=__import__("torch").device("cpu"),
            scale_factor_sample_size=min(n_volumes, 4),
        )

    torch.manual_seed(0)
    module = _tiny_jit_module()
    callbacks: list = [TrainLossLogger(), LatentX0MAE()]
    ckpt = _build_checkpoint(model_dir=results_dir, monitor_fid=False, every_n_epochs=1)
    callbacks.append(ckpt)
    datamodule = LatentWarmDataModule(
        latent_ds=None, vae=vae, batch_size=2, num_workers=0,
        val_latents=None, warm_fn=warm_fn, val_subset_size=min(n_volumes, 4),
    )
    trainer = build_trainer(max_epochs=1, callbacks=callbacks, model_dir=results_dir,
                            devices=world, accelerator="cpu", limit_val_batches=1.0)
    trainer.fit(module, datamodule=datamodule)
    dist_at_exit = __import__("torch").distributed.is_initialized()
    Path(results_dir, f"r{rank}.json").write_text(json.dumps({
        "rank": rank,
        "encode_count": encode_count[0],
        "dist_at_warm": dist_at_warm[0],
        "dist_at_exit": bool(dist_at_exit),
        "n_volumes": n_volumes,
        "global_step": int(trainer.global_step),
    }))


# -- ControlNet cold-cache 2-rank worker (issue #145: warm deferred to setup()) --


class _CountingPairedVolDS(Dataset):
    """A fake paired-image volume source that records per-rank encode counts.

    Mirrors the ``PairedNiftiVolumeDataset`` surface that ``PairedLatentDataset``
    drives (``unique_sample_ids`` / ``pair_meta`` / ``_load_volume``), so the deferred
    paired warm encodes each unique volume once total under DDP (the sharded branch).
    Emits the 5-key paired contract after the warm.
    """

    def __init__(self, n_pairs: int = 4, n_volumes: int = 6, *, seed: int = 0):
        torch.manual_seed(seed)
        self._n_pairs = n_pairs
        self._ids = [f"vol_{i:03d}" for i in range(n_volumes)]
        self._imgs = {sid: torch.randn(1, 4, 4, 4) for sid in self._ids}

    def __len__(self):
        return self._n_pairs

    def unique_sample_ids(self):
        return list(self._ids)

    def pair_meta(self, index):
        src = self._ids[index % len(self._ids)]
        tgt = self._ids[(index + 1) % len(self._ids)]
        return {
            "src_id": src,
            "tgt_id": tgt,
            "src_label": torch.tensor(0, dtype=torch.long),
            "tgt_label": torch.tensor(1, dtype=torch.long),
        }

    def _load_volume(self, sample_id):
        return {
            "image": self._imgs[sample_id],
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "label": 0,
        }


def controlnet_cold_cache_ddp_worker(
    rank: int, world: int, results_dir: str, port: str, n_volumes: int
) -> None:
    """Run a 2-rank ControlNet fit via the deferred ``DataModule.setup()`` paired warm.

    Issue #145: the paired latent-cache warm runs inside ``setup()`` (post-PG), so each
    rank encodes only its ``i % world == rank`` shard (no cross-rank double-encode).
    Captures the per-rank encode count + ``dist.is_initialized()`` at warm time.
    """
    from manifold import (
        ControlNet3DConditionModel,
        FlowMatchHeunDiscreteScheduler,
        UNet3DConditionModel,
    )
    from manifold.data.paired_latent_dataset import PairedLatentDataset
    from manifold.modules.controlnet_latent_flow import ControlNetLatentFlowModule
    from manifold.training.controlnet_cli import ControlNetInputs

    ddp_init(rank, world, port)
    encode_count = [0]
    dist_at_warm = [None]

    def counting_encode(images):
        encode_count[0] += 1
        return images.float().repeat(1, 4, 1, 1, 1)  # [1,1,D,H,W] -> [1,4,D,H,W] latent

    vol_train = _CountingPairedVolDS(n_pairs=4, n_volumes=n_volumes)
    vol_val = _CountingPairedVolDS(n_pairs=2, n_volumes=n_volumes, seed=1)
    cache_dir = str(Path(results_dir) / "cache")
    # Distinct tags so train/val caches stay disjoint (the gate counts train encodes).
    train_ds = PairedLatentDataset(vol_train, encode_fn=None, cache_dir=cache_dir, cache_tag="cn_train")
    val_ds = PairedLatentDataset(vol_val, encode_fn=None, cache_dir=cache_dir, cache_tag="cn_val")

    def warm_fn():
        dist_at_warm[0] = torch.distributed.is_initialized()
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
    from manifold.training.controlnet_cli import run_controlnet_training

    trainer, _ckpt = run_controlnet_training(
        module=module, inputs=inputs, model_dir=results_dir,
        max_epochs=1, devices=world, accelerator="cpu", batch_size=2,
        limit_val_batches=1.0,
    )
    Path(results_dir, f"r{rank}.json").write_text(json.dumps({
        "rank": rank,
        "encode_count": encode_count[0],
        "dist_at_warm": dist_at_warm[0],
        "global_step": int(trainer.global_step),
    }))
    ddp_fini()


# -- ControlNet DDP monitor worker (issue #146: keep the global val monitor) -----


class _ToyPairedCondDS(Dataset):
    """Tiny paired latent dataset for the ControlNet DDP fit (pre-warmed contract).

    Emits the ControlNet batch contract ``{src_latent, tgt_latent, spacing, src_label,
    tgt_label}`` (both latents scaled) so the module's validation forward returns
    ``pred`` / ``target`` and ``LatentX0MAE`` logs ``val/x0_mae``.
    """

    def __init__(self, n: int = 4):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {
            "src_latent": torch.randn(4, 8, 8, 4),
            "tgt_latent": torch.randn(4, 8, 8, 4),
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "src_label": torch.tensor(1, dtype=torch.long),
            "tgt_label": torch.tensor(2, dtype=torch.long),
        }


def controlnet_monitor_ddp_worker(rank: int, world: int, results_dir: str, port: str, _unused: bool) -> None:
    """Run a 2-rank ControlNet fit; capture the checkpoint monitor + written ckpts.

    Issue #146: ``val/x0_mae`` is globally reduced (LatentX0MAE logs a sample-weighted
    MeanMetric), so the monitored checkpoint must stay ON under DDP — no
    ``save_top_k=1`` fallback. Captures ``ckpt.monitor`` and the written ckpt filenames
    so the test asserts the monitored checkpoint (not just ``last``) is produced.
    """
    from manifold import (
        ControlNet3DConditionModel,
        FlowMatchHeunDiscreteScheduler,
        UNet3DConditionModel,
    )
    from manifold.modules.controlnet_latent_flow import ControlNetLatentFlowModule
    from manifold.training.controlnet_cli import ControlNetInputs, run_controlnet_training

    ddp_init(rank, world, port)
    try:
        torch.manual_seed(0)
        base = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
        for p in base.unet.out.parameters():
            if p.abs().sum().item() == 0.0:
                nn.init.normal_(p, std=0.01)
        controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
        controlnet.load_base_encoder_weights(base)
        module = ControlNetLatentFlowModule(
            base, controlnet, FlowMatchHeunDiscreteScheduler(),
            lr=1e-3, lr_warmup_steps=0, num_train_examples=4, train_batch_size=2, n_epochs=2,
        )
        inputs = ControlNetInputs(
            unet=base, controlnet=controlnet, scheduler=FlowMatchHeunDiscreteScheduler(),
            train_ds=_ToyPairedCondDS(), val_ds=_ToyPairedCondDS(),
        )
        trainer, ckpt = run_controlnet_training(
            module=module, inputs=inputs, model_dir=results_dir,
            max_epochs=2, devices=world, accelerator="cpu", batch_size=2,
            limit_val_batches=1.0,
        )
        written = sorted(p.name for p in Path(results_dir).glob("*.ckpt"))
        Path(results_dir, f"r{rank}.json").write_text(json.dumps({
            "rank": rank,
            "ckpt_monitor": ckpt.monitor,
            "ckpt_mode": ckpt.mode,
            "global_step": int(trainer.global_step),
            "written_ckpts": written,
            "val_x0_mae": float(trainer.callback_metrics.get("val/x0_mae", float("nan"))),
        }))
    finally:
        ddp_fini()




__all__ = [
    "ddp_init",
    "ddp_fini",
    "run_ddp_two_rank",
    "jit_ddp_worker",
    "_unbalanced_val_worker",
    "grpo_ddp_worker",
    "cold_cache_ddp_worker",
    "controlnet_cold_cache_ddp_worker",
    "controlnet_monitor_ddp_worker",
]
