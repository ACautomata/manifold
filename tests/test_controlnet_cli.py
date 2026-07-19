"""ControlNet supervised CLI smoke (issue #141 / ADR-0027 stage 1).

``controlnet_cli.main`` runs end-to-end via the ``data_provider`` injection seam
(argparse -> compose -> build -> fit -> checkpoint) on a tiny frozen base +
trainable ControlNet + toy paired batches, mirroring the GRPO/reward CLI smokes.
The Mode-2 guard and the ``--native-dir`` / ``--latents-dir`` real-path validation
are pinned too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from manifold import (
    ControlNet3DConditionModel,
    FlowMatchHeunDiscreteScheduler,
    UNet3DConditionModel,
)
from manifold.training.controlnet_cli import ControlNetInputs, main as controlnet_main


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
