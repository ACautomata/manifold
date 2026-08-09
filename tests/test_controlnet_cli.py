"""ControlNet supervised CLI smoke (issue #141 / ADR-0027 stage 1).

``controlnet_cli.main`` runs end-to-end via the ``data_provider`` injection seam
(argparse -> compose -> build -> fit -> checkpoint) on a tiny frozen base +
trainable ControlNet + toy paired batches, mirroring the GRPO/reward CLI smokes.
The ``--native-dir`` / ``--latents-dir`` real-path validation is pinned too.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from manifold import (
    ControlNet3DConditionModel,
    FlowMatchHeunDiscreteScheduler,
    UNet3DConditionModel,
)
from manifold.training.callbacks import CheckpointSpec, TrainLossSpec
from manifold.training.controlnet_cli import (
    ControlNetInputs,
    main as controlnet_main,
    run_controlnet_training,
)
from manifold.training.metrics import LatentX0MAE


def _frozen_base() -> UNet3DConditionModel:
    """Tiny base UNet with the zero-init output conv re-initialized (see test_controlnet_module_training)."""
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in unet.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return unet


def _controlnet(base: UNet3DConditionModel) -> ControlNet3DConditionModel:
    torch.manual_seed(1)
    cn = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    cn.load_base_encoder_weights(base)
    return cn


class _ToyPairedDS(Dataset):
    """A tiny paired latent dataset (train/val): emits the ControlNet batch contract."""

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


def _provider(cfg, device) -> ControlNetInputs:
    base = _frozen_base()
    return ControlNetInputs(
        unet=base,
        controlnet=_controlnet(base),
        scheduler=FlowMatchHeunDiscreteScheduler(),
        train_ds=_ToyPairedDS(),
        val_ds=_ToyPairedDS(),
    )


def _write_tiny_configs(tmp_path: Path) -> tuple[str, str, str]:
    net = tmp_path / "network.yaml"
    net.write_text("spatial_dims: 3\nlatent_channels: 4\n")
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
    )
    train = tmp_path / "train.yaml"
    train.write_text(
        "diffusion_unet_train: {batch_size: 2, lr: 1.0e-3, n_epochs: 1, "
        "lr_warmup_steps: 0, lr_ref_batch_size: 8, lr_scale_rule: sqrt}\n"
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05, l1_weight: 0.0}\n"
        "checkpoint: {save_top_k: 1}\n"
    )
    return str(env), str(train), str(net)


def test_run_controlnet_training_delegates_to_training_spine(tmp_path, monkeypatch):
    """The ControlNet shell preserves its defaults while delegating orchestration."""
    captured = {}
    trainer_sentinel = object()
    checkpoint_sentinel = object()
    spine = MagicMock()
    spine.run.side_effect = lambda **kwargs: (
        captured.update(kwargs) or (trainer_sentinel, checkpoint_sentinel)
    )
    monkeypatch.setattr(
        "manifold.training.controlnet_cli.TrainingSpine", MagicMock(return_value=spine)
    )

    vae = object()
    inputs = ControlNetInputs(
        unet=object(),
        controlnet=object(),
        scheduler=object(),
        train_ds=_ToyPairedDS(),
        val_ds=_ToyPairedDS(),
        vae=vae,
    )
    module = object()
    result = run_controlnet_training(
        module=module,
        inputs=inputs,
        model_dir=str(tmp_path),
        max_epochs=2,
        devices=1,
        accelerator="cpu",
        batch_size=2,
        save_top_k=2,
        seed=9,
        ckpt_path="resume.ckpt",
        limit_val_batches=0.25,
    )

    assert result == (trainer_sentinel, checkpoint_sentinel)
    assert spine.registry.register.call_args_list == [
        call("train_loss", TrainLossSpec),
        call("checkpoint", CheckpointSpec),
    ]
    assert captured["default_names"] == ["train_loss", "checkpoint"]
    assert captured["callback_names_override"] is None
    assert len(captured["extra_callbacks"]) == 1
    assert isinstance(captured["extra_callbacks"][0], LatentX0MAE)
    assert captured["module"] is module
    assert captured["datamodule"] is captured["ctx"].datamodule
    assert captured["ctx"].module is module
    assert captured["ctx"].vae is vae
    assert captured["ctx"].model_dir == str(tmp_path)
    assert captured["ctx"].seed == 9
    assert captured["callback_cfg"] == {
        "checkpoint": {
            "monitor_metric": "val/x0_mae",
            "mode": "min",
            "save_top_k": 2,
            "filename": "controlnet-{epoch:03d}-{val/x0_mae:.3f}",
        }
    }
    assert captured["max_epochs"] == 2
    assert captured["devices"] == 1
    assert captured["accelerator"] == "cpu"
    assert captured["limit_val_batches"] == 0.25
    assert captured["ckpt_path"] == "resume.ckpt"


def test_run_controlnet_training_merges_callback_config(tmp_path, monkeypatch):
    """Callback overrides win before the checkpoint filename is derived."""
    captured = {}
    spine = MagicMock()
    spine.run.side_effect = lambda **kwargs: (
        captured.update(kwargs) or (object(), object())
    )
    monkeypatch.setattr(
        "manifold.training.controlnet_cli.TrainingSpine", MagicMock(return_value=spine)
    )
    inputs = ControlNetInputs(
        unet=object(),
        controlnet=object(),
        scheduler=object(),
        train_ds=_ToyPairedDS(),
        val_ds=_ToyPairedDS(),
    )
    supplied_cfg = {
        "checkpoint": {
            "monitor_metric": "train/loss_epoch",
            "save_last": False,
        }
    }

    run_controlnet_training(
        module=object(),
        inputs=inputs,
        model_dir=str(tmp_path),
        max_epochs=1,
        devices=1,
        accelerator="cpu",
        callback_names=["train_loss", "checkpoint"],
        callback_cfg=supplied_cfg,
    )

    assert captured["callback_names_override"] == ["train_loss", "checkpoint"]
    assert captured["callback_cfg"] == {
        "checkpoint": {
            "monitor_metric": "train/loss_epoch",
            "mode": "min",
            "save_top_k": 3,
            "save_last": False,
            "filename": "controlnet-{epoch:03d}-{train/loss_epoch:.3f}",
        }
    }
    assert supplied_cfg == {
        "checkpoint": {
            "monitor_metric": "train/loss_epoch",
            "save_last": False,
        }
    }


def test_main_forwards_callback_selection_and_checkpoint_config(tmp_path, monkeypatch):
    """CLI callback names replace YAML names while YAML knobs still reach the spine."""
    env, train, net = _write_tiny_configs(tmp_path)
    Path(train).write_text(
        "diffusion_unet_train: {batch_size: 2, lr: 1.0e-3, n_epochs: 1, "
        "lr_warmup_steps: 0, lr_ref_batch_size: 8, lr_scale_rule: sqrt}\n"
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05, l1_weight: 0.0}\n"
        "callbacks: [train_loss, checkpoint]\n"
        "checkpoint: {save_top_k: 1, save_last: false, every_n_epochs: 2}\n"
    )
    captured = {}

    def _capture_run(**kwargs):
        captured.update(kwargs)
        return object(), object()

    monkeypatch.setattr(
        "manifold.training.controlnet_cli.run_controlnet_training", _capture_run
    )

    rc = controlnet_main(
        [
            "-e",
            env,
            "-c",
            train,
            "-t",
            net,
            "-g",
            "1",
            "--callbacks",
            "checkpoint",
        ],
        data_provider=_provider,
    )

    assert rc == 0
    assert captured["callback_names"] == ["checkpoint"]
    assert captured["callback_cfg"] == {
        "checkpoint": {
            "save_top_k": 1,
            "save_last": False,
            "every_n_epochs": 2,
        }
    }


def test_main_runs_end_to_end_with_fake_data(tmp_path):
    """main(): argparse -> compose -> build -> fit -> ckpt (the fake-data injection seam)."""
    env, train, net = _write_tiny_configs(tmp_path)
    rc = controlnet_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=_provider,
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_main_native_dirs_default_none_and_validated(tmp_path):
    """--native-dir/--latents-dir default None and are required without a data_provider."""
    env, train, net = _write_tiny_configs(tmp_path)
    with pytest.raises(ValueError, match="native-dir"):
        controlnet_main(["-e", env, "-c", train, "-t", net, "-g", "1"])
    # With a data_provider the missing args are NOT required (smoke seam intact).
    rc = controlnet_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=_provider,
    )
    assert rc == 0


def test_main_rejects_non_controlnet_recipe(tmp_path):
    """A config without `diffusion_unet_train` (e.g. a GRPO recipe) fails fast."""
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
    )
    train = tmp_path / "train.yaml"
    train.write_text("grpo_train: {batch_size: 2}\n")  # wrong recipe
    net = tmp_path / "network.yaml"
    net.write_text("spatial_dims: 3\nlatent_channels: 4\n")
    with pytest.raises(ValueError, match="diffusion_unet_train"):
        controlnet_main(
            ["-e", str(env), "-c", str(train), "-t", str(net), "-g", "1"],
            data_provider=_provider,
        )


def test_main_real_path_pins_each_rank_to_local_gpu(tmp_path):
    """2-rank DDP: controlnet's real path hands ``_real_inputs`` a per-rank
    cuda:{LOCAL_RANK} device (PR #156 regression lock — no cuda:0 collapse).

    Drives ``main`` on the real branch (``data_provider=None``) under spawn with
    ``_real_inputs`` stubbed to surface the device it receives; the lock is that the
    two ranks DISAGREE on that device (a bare cuda:0 resolution would collapse both
    onto GPU 0 — the PR #156 race / DDP init timeout). CUDA is mocked in the spawned
    children so ``pin()`` resolves cuda devices on CPU gloo; ``LOCAL_RANK`` is set per
    rank by the harness (the torchrun contract). (ADR-0031 §A2 required this 2-rank
    smoke; it had never landed.)
    """
    from tests.ddp import controlnet_device_pin_ddp_worker, run_ddp_two_rank

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "network.yaml").write_text("spatial_dims: 3\nlatent_channels: 4\n")
    (cfg_dir / "env.yaml").write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
    )
    (cfg_dir / "train.yaml").write_text(
        "diffusion_unet_train: {batch_size: 2, lr: 1.0e-3, n_epochs: 1, "
        "lr_warmup_steps: 0, lr_ref_batch_size: 8, lr_scale_rule: sqrt}\n"
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05, l1_weight: 0.0}\n"
        "checkpoint: {save_top_k: 1}\n"
    )
    results = run_ddp_two_rank(
        controlnet_device_pin_ddp_worker, results_dir=str(tmp_path), args=(str(cfg_dir),)
    )
    r0, r1 = results
    # The stub fired on both ranks (main's real path reached _real_inputs).
    assert not r0["bypassed"] and not r1["bypassed"], (r0, r1)
    # ddp_init set LOCAL_RANK per rank (the torchrun contract).
    assert r0["local_rank_env"] == "0" and r1["local_rank_env"] == "1"
    # _real_inputs received each rank's OWN cuda:{LOCAL_RANK} device.
    assert r0["device"] == "cuda:0", r0
    assert r1["device"] == "cuda:1", r1
    # The PR #156 regression lock: the two ranks DIVERGE on the device _real_inputs
    # stages onto (a bare cuda:0 resolution would collapse both onto cuda:0).
    assert r0["device"] != r1["device"]
