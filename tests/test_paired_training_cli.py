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
    # allow_train_as_val=True: the smoke reuses the train fixture as val to
    # exercise the PSNR/SSIM plumbing (it tests wiring, not held-out
    # generalization). Production leaves this False -> validation disabled.
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

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


def test_run_paired_training_threads_ema_decays(tmp_path):
    """``ema_decays`` flows run_paired_training -> DoubleEMACallback.

    Guards the ema_decays wiring: without it the callback hardcodes JiT's
    ``(0.9999, 0.9996)`` and ``formulation.ema_decays`` is a dead knob. Asserts the
    decays land on the callback (the ``decays`` property at ema.py).
    """
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
    # allow_train_as_val=True: the smoke reuses the train fixture as val to
    # exercise the PSNR/SSIM plumbing (it tests wiring, not held-out
    # generalization). Production leaves this False -> validation disabled.
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

    trainer, _ = run_paired_training(
        module=module,
        bundle=bundle,
        model_dir=str(tmp_path / "paired_run"),
        max_epochs=1,
        batch_size=2,
        num_workers=0,
        limit_val_batches=2,
        num_inference_steps=2,
        every_n_epochs=1,
        ema_decays=(0.999, 0.9996),
    )
    ema_cb = next(cb for cb in trainer.callbacks if isinstance(cb, DoubleEMACallback))
    assert ema_cb.decays == (0.999, 0.9996), "ema_decays must flow to the DoubleEMACallback"


def test_run_paired_training_threads_check_val_every_n_epoch(tmp_path):
    """``check_val_every_n_epoch`` flows run_paired_training -> Trainer kwargs.

    Guards the last-epoch-only-val wiring (autoresearch): when set, the Trainer is
    built with ``check_val_every_n_epoch=<N>`` AND ``num_sanity_val_steps=0`` so
    validation runs only at the final epoch. ``None`` (default) leaves Lightning's
    per-epoch cadence untouched.
    """
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
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

    trainer, _ = run_paired_training(
        module=module,
        bundle=bundle,
        model_dir=str(tmp_path / "paired_run"),
        max_epochs=1,
        batch_size=2,
        num_workers=0,
        limit_val_batches=2,
        num_inference_steps=2,
        every_n_epochs=1,
        check_val_every_n_epoch=20,
    )
    assert trainer.check_val_every_n_epoch == 20, "check_val_every_n_epoch must reach the Trainer"
    assert trainer.num_sanity_val_steps == 0, "num_sanity_val_steps=0 must be applied alongside"


def test_check_val_every_n_epoch_forces_psnr_cadence_to_1(tmp_path):
    """Last-epoch-only val forces the PSNR/SSIM callback cadence to 1 (codex #91).

    With ``check_val_every_n_epoch`` set, Lightning validates only every N epochs; the
    callback's own ``every_n_epochs`` gate (0-based ``epoch % n``) could otherwise SKIP
    that single pass when ``every_n_epochs>1`` (e.g. the final epoch 19 % 5 != 0),
    leaving the run with no val/psnr. The callback cadence is forced to 1 so the decode
    always runs whenever Lightning validates.
    """
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
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

    trainer, _ = run_paired_training(
        module=module,
        bundle=bundle,
        model_dir=str(tmp_path / "paired_run"),
        max_epochs=1,
        batch_size=2,
        num_workers=0,
        limit_val_batches=2,
        num_inference_steps=2,
        every_n_epochs=5,
        check_val_every_n_epoch=20,
    )
    psnr_cb = next(cb for cb in trainer.callbacks if isinstance(cb, PairedPSNRSSIMCallback))
    assert psnr_cb.every_n_epochs == 1, (
        "under last-epoch-only val the PSNR cadence must be forced to 1 so the gate "
        "never skips Lightning's single final-epoch validation pass"
    )


def test_main_reads_ema_decays_from_config(tmp_path, monkeypatch):
    """``formulation.ema_decays`` flows config -> main -> DoubleEMACallback.

    Mirrors ``test_main_reads_loss_weight_from_config``: a regression in the config
    read would silently revert the callback to the hardcoded ``(0.9999, 0.9996)``
    while the run_paired_training-level test (which passes the kwarg directly) stays
    green. Spies on the DoubleEMACallback constructor.
    """
    from manifold.training import paired_cli

    captured: dict = {}
    real_cls = paired_cli.DoubleEMACallback

    def spy(*args, **kwargs):
        captured["decays"] = tuple(kwargs.get("decays", (0.9999, 0.9996)))
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(paired_cli, "DoubleEMACallback", spy)

    model_dir = tmp_path / "model"
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "data_base_dir: /tmp/_unused_\n"
        f"model_dir: {model_dir}\n"
        "model_filename: paired_jit.pt\n"
        "trained_autoencoder_path: /tmp/_unused_vae_\n"
        "val_subset_size: 4\nrandom_seed: 0\nnum_gpus: 1\n"
    )
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
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05, ema_decays: [0.999, 0.9996]}\n"
        "diffusion_unet_inference: {dim: [4, 4, 4], spacing: [1.0, 1.0, 1.0], modality: 1, num_inference_steps: 2}\n"
        "paired_eval: {num_inference_steps: 2, every_n_epochs: 1}\n"
    )
    # allow_train_as_val=True: the smoke reuses the train fixture as val to
    # exercise the PSNR/SSIM plumbing (it tests wiring, not held-out
    # generalization). Production leaves this False -> validation disabled.
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

    argv = ["-e", str(env_yaml), "-c", str(train_yaml), "-t", str(net_yaml), "--max-epochs", "1"]
    rc = paired_cli.main(argv, data_provider=lambda cfg, device: bundle)
    assert rc == 0
    assert captured.get("decays") == (0.999, 0.9996), "ema_decays must flow config -> DoubleEMACallback"


def test_main_falls_back_to_recipe_default_ema_when_unset(tmp_path, monkeypatch):
    """Regression (codex #113): a custom paired YAML that omits
    ``formulation.ema_decays`` must fall back to the paired recipe default
    ``(0.999, 0.99)`` — NOT the legacy ``(0.9999, 0.9996)``. The legacy fallback
    left custom-YAML (and direct ``run_paired_training``) runs validating on the
    stale 0.9999 arm this change retires. Mirrors
    ``test_main_reads_ema_decays_from_config`` but with ``ema_decays`` UNSET, so
    it fails if the ``opt(cfg.formulation, "ema_decays", ...)`` default reverts.
    """
    from manifold.training import paired_cli

    captured: dict = {}
    real_cls = paired_cli.DoubleEMACallback

    def spy(*args, **kwargs):
        captured["decays"] = tuple(kwargs.get("decays"))
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(paired_cli, "DoubleEMACallback", spy)

    model_dir = tmp_path / "model"
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "data_base_dir: /tmp/_unused_\n"
        f"model_dir: {model_dir}\n"
        "model_filename: paired_jit.pt\n"
        "trained_autoencoder_path: /tmp/_unused_vae_\n"
        "val_subset_size: 4\nrandom_seed: 0\nnum_gpus: 1\n"
    )
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
        # NOTE: ema_decays is intentionally OMITTED — the test guards the fallback.
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05}\n"
        "diffusion_unet_inference: {dim: [4, 4, 4], spacing: [1.0, 1.0, 1.0], modality: 1, num_inference_steps: 2}\n"
        "paired_eval: {num_inference_steps: 2, every_n_epochs: 1}\n"
    )
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

    argv = ["-e", str(env_yaml), "-c", str(train_yaml), "-t", str(net_yaml), "--max-epochs", "1"]
    rc = paired_cli.main(argv, data_provider=lambda cfg, device: bundle)
    assert rc == 0
    assert captured.get("decays") == (0.999, 0.99), (
        "fallback must be the paired recipe default (0.999, 0.99), "
        "not the legacy (0.9999, 0.9996)"
    )


def test_run_paired_training_wires_held_out_val_dataset(tmp_path, monkeypatch):
    """When the bundle carries a ``val_latent_ds``, it is wired as the held-out val
    (NOT the train-as-val fallback). Spies on the ``PairedWarmDataModule`` kwargs
    (F2/F4: the warm moved into ``setup()``, so the val_dataset is plumbed via the
    DataModule, not ``build_datamodule``)."""
    from manifold.data import warm_datamodule
    from manifold.training import paired_cli

    captured: dict = {}
    real_cls = warm_datamodule.PairedWarmDataModule

    def spy(**kw):
        captured["train"] = kw.get("latent_ds")
        captured["val"] = kw.get("val_latent_ds")
        return real_cls(**kw)

    monkeypatch.setattr(warm_datamodule, "PairedWarmDataModule", spy)

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
    train_ds = _FakePairedDataset(n=4)
    val_ds = _FakePairedDataset(n=4)  # distinct held-out set
    bundle = _DataBundle(
        latent_ds=train_ds, vae=AutoencoderKL(scaling_factor=0.5), val_latent_ds=val_ds
    )

    run_paired_training(
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
    assert captured["train"] is train_ds
    assert captured["val"] is val_ds, "held-out val_latent_ds must be wired to the DataModule"


def test_run_paired_training_skips_validation_when_no_held_out_val(tmp_path):
    """No val_latent_ds and not opted in -> validation DISABLED (never reuse train).

    The paired flow's safety guard: with no held-out val split (val_fraction=0 /
    no val_data_base_dir -> empty val_manifest) and allow_train_as_val=False, no
    PairedPSNRSSIMCallback is attached, the checkpoint monitor is dropped, and no
    ``val/*`` metric is logged - the train set is never reused as val.
    """
    unet = _trainable_paired_unet()
    module = PairedLatentFlowModule(
        unet, FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0,
        num_train_examples=4, train_batch_size=2, n_epochs=1,
    )
    # No val_latent_ds, allow_train_as_val defaults False -> has_val False -> disabled.
    bundle = _DataBundle(latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5))

    trainer, ckpt = run_paired_training(
        module=module, bundle=bundle, model_dir=str(tmp_path / "paired_run"),
        max_epochs=1, batch_size=2, num_workers=0, limit_val_batches=2,
        num_inference_steps=2, every_n_epochs=1,
    )
    assert not any(type(c).__name__ == "PairedPSNRSSIMCallback" for c in trainer.callbacks)
    assert ckpt.monitor is None  # no val/psnr to monitor
    assert not any(k.startswith("val/") for k in trainer.callback_metrics)
    assert any((tmp_path / "paired_run").glob("*.ckpt"))  # training still produced a ckpt
    # Regression (codex #90 P2): the no-val path must NOT set
    # ``check_val_every_n_epoch=None`` - Lightning's contract then requires an integer
    # ``val_check_interval``, which the float default violates. ``limit_val_batches=0``
    # alone makes val a no-op.
    assert trainer.check_val_every_n_epoch is not None


# -- console-entry config helper (mirrors test_training_cli._write_tiny_configs) --


def _write_paired_configs(tmp_path):
    """Write the tiny env/train/network YAMLs the paired ``main`` consumes.

    Factored out so the DDP-detection tests (and any future test driving the
    full ``main`` path) share one pinned config. Returns ``(env, train, net)``.
    """
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
    # Network config: the paired UNet doubles in_channels (2*C_latent = 8).
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
    return str(env_yaml), str(train_yaml), str(net_yaml)


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

    # allow_train_as_val=True: the smoke reuses the train fixture as val to
    # exercise the PSNR/SSIM plumbing (it tests wiring, not held-out
    # generalization). Production leaves this False -> validation disabled.
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

    def provider(cfg, device):
        return bundle

    argv = ["-e", str(env_yaml), "-c", str(train_yaml), "-t", str(net_yaml), "--max-epochs", "1"]
    rc = paired_cli.main(argv, data_provider=provider)
    assert rc == 0
    assert model_dir.exists()


def test_main_reads_loss_weight_from_config(tmp_path, monkeypatch):
    """``formulation.loss_weight`` flows config → main → PairedLatentFlowModule.

    Guards the headline regime fix: a regression in the wiring would silently
    revert the loss to the legacy ``1mt_sq`` while every module-level test stays
    green (they construct the module directly). Spies on the module constructor.
    """
    from manifold.training import paired_cli

    captured: dict = {}
    real_cls = paired_cli.PairedLatentFlowModule

    def spy(*args, **kwargs):
        captured["loss_weight"] = kwargs.get("loss_weight")
        return real_cls(*args, **kwargs)

    monkeypatch.setattr(paired_cli, "PairedLatentFlowModule", spy)

    model_dir = tmp_path / "model"
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "data_base_dir: /tmp/_unused_\n"
        f"model_dir: {model_dir}\n"
        "model_filename: paired_jit.pt\n"
        "trained_autoencoder_path: /tmp/_unused_vae_\n"
        "val_subset_size: 4\nrandom_seed: 0\nnum_gpus: 1\n"
    )
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
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05, loss_weight: uniform}\n"
        "diffusion_unet_inference: {dim: [4, 4, 4], spacing: [1.0, 1.0, 1.0], modality: 1, num_inference_steps: 2}\n"
        "paired_eval: {num_inference_steps: 2, every_n_epochs: 1}\n"
    )
    # allow_train_as_val=True: the smoke reuses the train fixture as val to
    # exercise the PSNR/SSIM plumbing (it tests wiring, not held-out
    # generalization). Production leaves this False -> validation disabled.
    bundle = _DataBundle(
        latent_ds=_FakePairedDataset(n=4), vae=AutoencoderKL(scaling_factor=0.5),
        allow_train_as_val=True,
    )

    argv = ["-e", str(env_yaml), "-c", str(train_yaml), "-t", str(net_yaml), "--max-epochs", "1"]
    rc = paired_cli.main(argv, data_provider=lambda cfg, device: bundle)
    assert rc == 0
    assert captured.get("loss_weight") == "uniform", "loss_weight must flow config -> module"


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


# -- train/val manifest resolution (native split vs val_fraction) ------------


def test_train_val_manifests_native_split_uses_val_data_base_dir(tmp_path, monkeypatch):
    """val_data_base_dir set to a real directory → full manifest is the train set,
    val is built from that dir; split_brats_pair_manifest is NOT called (native
    split, no leakage). Guards the "注意对应训练和验证数据集的加载" wiring:
    BraTS-2024's official train↔val (1621/188) must be honored as the held-out split.
    """
    from omegaconf import OmegaConf

    import manifold.data.paired_brats as pb
    from manifold.training import paired_cli

    val_dir = tmp_path / "BraTS-GLI-val"
    val_dir.mkdir()  # a real directory (contents are irrelevant — build is stubbed)
    train_manifest = [{"src": f"/train/s{i}-t1n.nii.gz"} for i in range(12)]
    val_manifest = [{"src": f"/val/s{i}-t1n.nii.gz"} for i in range(4)]
    split_calls = {"n": 0}

    def fake_build(brats_dir, labels=None):
        return val_manifest if "val" in str(brats_dir) else train_manifest

    def fake_split(manifest, val_fraction):  # pragma: no cover - must not run
        split_calls["n"] += 1
        return manifest, []

    monkeypatch.setattr(pb, "build_brats_pair_manifest", fake_build)
    monkeypatch.setattr(pb, "split_brats_pair_manifest", fake_split)

    cfg = OmegaConf.create({"val_data_base_dir": str(val_dir), "val_fraction": 0.1})
    out_train, out_val = paired_cli._train_val_manifests(cfg, train_manifest)

    assert out_train is train_manifest          # full manifest is the train set
    assert out_val is val_manifest              # val built from val_data_base_dir
    assert split_calls["n"] == 0                # fraction path not taken


def test_train_val_manifests_native_split_raises_on_empty_val(tmp_path, monkeypatch):
    """val_data_base_dir is a real directory but yields no pairable subjects → clear
    FileNotFoundError (not a silent val=train fallback or an opaque downstream crash)."""
    from omegaconf import OmegaConf

    import manifold.data.paired_brats as pb
    from manifold.training import paired_cli

    empty_val_dir = tmp_path / "empty-val"
    empty_val_dir.mkdir()
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: [])
    cfg = OmegaConf.create({"val_data_base_dir": str(empty_val_dir)})
    with pytest.raises(FileNotFoundError, match="val_data_base_dir"):
        paired_cli._train_val_manifests(cfg, [{"src": "a.nii.gz"}])


def test_train_val_manifests_non_directory_val_dir_falls_back(tmp_path, monkeypatch):
    """P1 regression (codex #78): val_data_base_dir pointing at a manifest JSON (or
    any non-directory) must NOT crash — it falls back to the val_fraction split
    (the pre-native-split behavior). The BraTS2023 profile sets this field to a
    ``brats_all_val.json``; before the directory guard, paired runs with that
    profile hit FileNotFoundError. build_brats_pair_manifest is not called for val.
    """
    from omegaconf import OmegaConf

    import manifold.data.paired_brats as pb
    from manifold.training import paired_cli

    manifest_json = tmp_path / "brats_all_val.json"
    manifest_json.write_text("{}")  # a file, not a directory
    captured: dict = {}

    def spy_split(manifest, val_fraction):
        captured["frac"] = val_fraction
        return manifest, []

    val_build_calls: list = []
    monkeypatch.setattr(
        pb, "build_brats_pair_manifest", lambda *a, **k: val_build_calls.append(a) or []
    )
    monkeypatch.setattr(pb, "split_brats_pair_manifest", spy_split)

    cfg = OmegaConf.create({"val_data_base_dir": str(manifest_json), "val_fraction": 0.1})
    paired_cli._train_val_manifests(cfg, [{"src": "a.nii.gz"}])

    assert captured.get("frac") == 0.1          # fraction path taken (fallback)
    assert val_build_calls == []                 # native build NOT attempted → no crash


def test_train_val_manifests_falls_back_to_fraction_split(monkeypatch):
    """No val_data_base_dir → delegates to split_brats_pair_manifest(manifest,
    val_fraction) (the PR #77 path; unchanged behavior)."""
    from omegaconf import OmegaConf

    import manifold.data.paired_brats as pb
    from manifold.training import paired_cli

    captured: dict = {}

    def spy_split(manifest, val_fraction):
        captured["manifest"] = manifest
        captured["frac"] = val_fraction
        return [{"src": "train"}], [{"src": "val"}]

    build_calls: list = []
    monkeypatch.setattr(
        pb, "build_brats_pair_manifest", lambda *a, **k: build_calls.append(a) or []
    )
    monkeypatch.setattr(pb, "split_brats_pair_manifest", spy_split)

    full = [{"src": "a.nii.gz"}]
    cfg = OmegaConf.create({"val_fraction": 0.1})
    train, val = paired_cli._train_val_manifests(cfg, full)

    assert captured["manifest"] is full and captured["frac"] == 0.1
    assert train == [{"src": "train"}] and val == [{"src": "val"}]
    assert build_calls == []                    # native-split build not invoked


@pytest.mark.parametrize(
    # val_data_base_dir absent / null / ??? must ALL read as unset → fraction path.
    # (opt() wraps OmegaConf.select with default=None; linchpin of "no regression
    # when val_data_base_dir is not a usable BraTS directory".)
    "val_dir_yaml",
    ["__absent__", "val_data_base_dir: null", "val_data_base_dir: ???"],
)
def test_train_val_manifests_unset_val_dir_falls_back_to_fraction(monkeypatch, val_dir_yaml):
    from omegaconf import OmegaConf

    import manifold.data.paired_brats as pb
    from manifold.training import paired_cli

    captured: dict = {}

    def spy_split(manifest, val_fraction):
        captured["frac"] = val_fraction
        return manifest, []

    build_calls: list = []
    monkeypatch.setattr(
        pb, "build_brats_pair_manifest", lambda *a, **k: build_calls.append(a) or []
    )
    monkeypatch.setattr(pb, "split_brats_pair_manifest", spy_split)

    body = "val_fraction: 0.2\n"
    if val_dir_yaml != "__absent__":
        body += val_dir_yaml + "\n"
    cfg = OmegaConf.create(body)

    paired_cli._train_val_manifests(cfg, [{"src": "a.nii.gz"}])
    assert captured.get("frac") == 0.2          # fraction path taken
    assert build_calls == []                     # no native-split build attempted
