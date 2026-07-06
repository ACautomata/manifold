"""Paired JiT training-stack tests (Slice 4, issue #69).

Covers the launch-readiness acceptance criteria:

- **CLI core smoke**: :func:`run_paired_training` (the testable core of
  ``manifold-train-paired``) fits one epoch over a fake paired latent cache +
  tiny VAE and writes a checkpoint, with the callbacks wired (double-EMA,
  PSNR/SSIM, train-metrics) and ``val/psnr`` / ``val/ssim`` reported on the
  slow-EMA arm (criterion 1 + 2).
- **grad-norm hook**: ``after_manual_backward`` stashes the AMP-corrected grad
  norm (criterion 4 — ``train/grad_norm`` logging).
- **EMA-arm swap**: the PSNR/SSIM callback swaps the slow-EMA shadow in around
  the rollout (criterion 2 — the reported metric reflects the published model).

Persistence (seam #6) is covered in ``tests/test_paired_persistence.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    PairedLatentFlowModule,
    UNet3DConditionModel,
)
from manifold.metrics import PairedPSNRSSIMCallback
from manifold.pipelines.paired_latent_flow import PairedLatentFlowPipeline
from manifold.training.ema import DoubleEMACallback
from manifold.training.paired_cli import _DataBundle, run_paired_training

C_LATENT = 4


class _FakePairedDataset(torch.utils.data.Dataset):
    """In-RAM paired latent dataset (the 5-key contract) for the CPU smoke."""

    def __init__(self, n: int = 4):
        torch.manual_seed(0)
        self.items = [
            {
                "src_latent": torch.randn(C_LATENT, 4, 4, 4),
                "tgt_latent": torch.randn(C_LATENT, 4, 4, 4),
                "src_label": torch.tensor(0, dtype=torch.long),
                "tgt_label": torch.tensor(1, dtype=torch.long),
                "spacing": torch.tensor([1.0, 1.0, 1.0]),
            }
            for _ in range(n)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def _trainable_paired_unet() -> UNet3DConditionModel:
    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    for p in unet.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return unet


# -- CLI core smoke ----------------------------------------------------------


def test_run_paired_training_smoke(tmp_path):
    """run_paired_training fits one epoch on a fake bundle + writes a checkpoint,
    wiring double-EMA + PSNR/SSIM + train-metrics and reporting val/psnr+val/ssim."""
    unet = _trainable_paired_unet()
    module = PairedLatentFlowModule(
        unet,
        FlowMatchHeunDiscreteScheduler(),
        lr=1e-2,
        lr_warmup_steps=0,
        num_train_examples=4,
        train_batch_size=2,
        n_epochs=1,
    )
    bundle = _DataBundle(latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5))

    trainer, ckpt = run_paired_training(
        module=module,
        bundle=bundle,
        model_dir=str(tmp_path / "paired_run"),
        max_epochs=1,
        batch_size=2,
        num_workers=0,
        limit_val_batches=2,
        num_inference_steps=2,
        every_n_epochs=1,
    )

    metrics = trainer.callback_metrics
    assert "val/psnr" in metrics, "PSNR/SSIM callback must log val/psnr"
    assert "val/ssim" in metrics
    assert torch.isfinite(metrics["val/psnr"])
    assert torch.isfinite(metrics["val/ssim"])
    # train/grad_norm stashed by after_manual_backward (criterion 4).
    assert module._last_grad_norm is not None and module._last_grad_norm > 0
    # A checkpoint was written (criterion 1 — the run produces an artifact).
    assert ckpt.last_model_path or any((tmp_path / "paired_run").glob("*.ckpt"))


def test_main_data_provider_seam_runs_end_to_end(tmp_path):
    """The full main() path (argparse → compose → build → fit) runs with the
    data_provider injection seam, exercising the console entry without BraTS data."""
    from manifold.training import paired_cli

    model_dir = tmp_path / "model"
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "data_base_dir: /tmp/_unused_\n"
        f"model_dir: {model_dir}\n"
        "model_filename: paired_jit.pt\n"
        "trained_autoencoder_path: /tmp/_unused_vae_\n"
        "val_subset_size: 4\n"
        "random_seed: 0\n"
        "num_gpus: 1\n"
    )
    # Network config: the paired UNet doubles in_channels (2·C_latent = 8).
    net_yaml = tmp_path / "net.yaml"
    net_yaml.write_text(
        "spatial_dims: 3\nlatent_channels: 4\n"
        "diffusion_unet:\n"
        "  spatial_dims: 3\n  in_channels: 8\n  out_channels: 4\n"
        "  num_channels: [8, 8]\n  num_res_blocks: 1\n  norm_num_groups: 8\n"
        "  num_head_channels: [4, 4]\n  attention_levels: [false, false]\n"
        "  use_flash_attention: false\n  include_spacing_input: true\n"
        "  num_class_embeds: 4\n  num_train_timesteps: 1000\n"
        "scheduler:\n  num_train_timesteps: 1000\n  t_eps: 0.05\n"
    )
    train_yaml = tmp_path / "paired.yaml"
    train_yaml.write_text(
        "diffusion_unet_train: {batch_size: 2, lr: 1.0e-2, n_epochs: 1, lr_warmup_steps: 0, cache_rate: 0}\n"
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05}\n"
        "diffusion_unet_inference: {dim: [4, 4, 4], spacing: [1.0, 1.0, 1.0], modality: 1, num_inference_steps: 2}\n"
        "paired_eval: {num_inference_steps: 2, every_n_epochs: 1}\n"
    )

    bundle = _DataBundle(latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5))

    def provider(cfg, device):
        return bundle

    argv = ["-e", str(env_yaml), "-c", str(train_yaml), "-t", str(net_yaml), "--max-epochs", "1"]
    rc = paired_cli.main(argv, data_provider=provider)
    assert rc == 0
    assert model_dir.exists()


# -- grad-norm hook (criterion 4) -------------------------------------------


def test_grad_norm_hook_stashes_amp_corrected_value():
    """after_manual_backward stashes the grad norm; off-GPU the AMP scale is 1.0."""
    unet = _trainable_paired_unet()
    module = PairedLatentFlowModule(unet, FlowMatchHeunDiscreteScheduler())
    batch = {
        "src_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "tgt_latent": torch.randn(1, C_LATENT, 4, 4, 4),
        "src_label": torch.tensor([0]),
        "tgt_label": torch.tensor([1]),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }
    module(batch, "fit")["loss"].backward()
    assert module._last_grad_norm is None  # not set until the hook runs
    module.after_manual_backward()

    expected = torch.sqrt(
        sum((p.grad.detach().float() ** 2).sum() for p in module.unet.parameters())
    )
    assert module._last_grad_norm == pytest.approx(float(expected))
    assert module._amp_scale() == 1.0  # no Trainer / off-GPU


# -- EMA-arm swap (criterion 2) ---------------------------------------------


def test_psnr_callback_swaps_slow_ema_around_rollout(tmp_path, monkeypatch):
    """The callback swaps the slow-EMA shadow in around the val rollout so the
    reported metric reflects the published model (mirrors FIDCallback's slow arm)."""
    import lightning.pytorch as pl

    unet = _trainable_paired_unet()
    module = PairedLatentFlowModule(unet, FlowMatchHeunDiscreteScheduler(), lr=1e-2)
    pipeline = PairedLatentFlowPipeline(unet, AutoencoderKL(scaling_factor=0.5), module.scheduler)
    ema = DoubleEMACallback(module)
    cb = PairedPSNRSSIMCallback(
        pipeline=pipeline, num_inference_steps=2, every_n_epochs=1, ema_callback=ema
    )

    calls = {"swap": 0, "restore": 0}
    real_swap, real_restore = ema.swap_in, ema.restore

    def spy_swap(m):
        calls["swap"] += 1
        return real_swap(m)

    def spy_restore(m):
        calls["restore"] += 1
        return real_restore(m)

    ema.swap_in = spy_swap  # type: ignore[assignment]
    ema.restore = spy_restore  # type: ignore[assignment]

    trainer = pl.Trainer(
        accelerator="cpu", devices=1, max_epochs=1, logger=False,
        enable_progress_bar=False, enable_checkpointing=False, enable_model_summary=False,
        callbacks=[ema, cb], num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=_datamodule())
    # One val epoch → ≥1 swap_in + restore (per val batch).
    assert calls["swap"] >= 1, "EMA swap_in must wrap the rollout (slow-arm reporting)"
    assert calls["restore"] >= 1
    assert calls["swap"] == calls["restore"], "every swap_in must be paired with restore"


def _datamodule():
    from stable_pretraining.data import DataModule
    from torch.utils.data import DataLoader

    ds = _FakePairedDataset(n=4)
    train = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    val = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    return DataModule(train=train, val=val)
