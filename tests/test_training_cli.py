"""manifold-train + ModelCheckpoint + Export integration (Slice E, issue #28).

The end-to-end Training-pipeline + Export round-trip (the highest seam, per the
issue's testing plan): ``run_training`` on a tiny CPU config (tiny UNet + a
fake-latent cache + a fake feature network) writes a Lightning ``.ckpt`` and
logs ``train/loss_epoch`` / ``train/grad_norm`` / ``val/x0_mae`` / ``val/fid_avg``;
Export bakes the slowest EMA shadow into a native dir ``Pipeline.from_pretrained``
loads, whose decode matches ``Module.sample()`` + the held-VAE decode. Plus resume
via ``ckpt_path`` and the ``main`` console entry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import Dataset

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowModule,
    LatentFlowPipeline,
    UNet3DConditionModel,
)
from manifold.training import export_to_native, run_training
from manifold.training.cli import _DataBundle, _inference_recipe, main as train_main


class _LatentDS(Dataset):
    def __init__(self, n: int = 6):
        torch.manual_seed(0)
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


class _FakeFeatureNet(nn.Module):
    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        b = plane.shape[0]
        flat = plane.reshape(b, -1)[:, :8]
        if flat.shape[1] < 8:
            flat = torch.nn.functional.pad(flat, (0, 8 - flat.shape[1]))
        return flat


def _module():
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    return LatentFlowModule(
        unet, FlowMatchHeunDiscreteScheduler(), lr=1e-2, lr_warmup_steps=0,
        num_train_examples=6, train_batch_size=2, n_epochs=1,
    )


def _bundle():
    vae = AutoencoderKL(scaling_factor=0.5)
    return _DataBundle(
        latent_ds=_LatentDS(), vae=vae, val_latents=torch.randn(5, 4, 4, 4, 4)
    )


def _run(tmp_path, *, module=None, bundle=None, enable_fid=True, ckpt_path=None, max_epochs=1, inference_recipe=None):
    return run_training(
        module=module or _module(),
        bundle=bundle or _bundle(),
        feature_net=_FakeFeatureNet(),
        model_dir=str(tmp_path),
        max_epochs=max_epochs,
        devices=1,
        accelerator="cpu",
        enable_fid=enable_fid,
        num_synth=2,
        limit_val_batches=2,
        cov_ridge=1e-2,
        ckpt_path=ckpt_path,
        inference_recipe=inference_recipe,
    )


def test_inference_recipe_latent_shape_derived_from_val_latents():
    """_inference_recipe derives latent_shape as one real-sample's latent shape.

    The synthetic latent must match the real validation latents' spatial shape
    (the FID compares features in the same image space), so the recipe template
    is the leading-1, single-sample shape of the real latents — not a hardcoded
    constant. A unique non-round shape proves the value is derived, not stubbed.
    """
    val_latents = torch.randn(6, 4, 7, 9, 11)  # [N, C, D, H, W]
    inf = _inference_recipe(_module(), cfg=None, val_latents=val_latents)
    assert inf["latent_shape"] == (1, 4, 7, 9, 11)


def test_inference_recipe_uses_configured_sampling_knobs():
    """FID generation uses the composed JiT inference recipe, not tiny defaults."""
    val_latents = torch.randn(6, 4, 7, 9, 11)
    cfg = OmegaConf.create(
        {
            "diffusion_unet_inference": {
                "spacing": [1.7, 1.7, 2],
                "modality": 1,
                "num_inference_steps": 15,
                "cfg_guidance_scale": 1.5,
            },
            "formulation": {"cfg_interval": [0.1, 1.0]},
        }
    )

    inf = _inference_recipe(_module(), cfg=cfg, val_latents=val_latents)

    assert inf == {
        "latent_shape": (1, 4, 7, 9, 11),
        "spacing": [1.7, 1.7, 2],
        "modality": 1,
        "num_inference_steps": 15,
        "guidance_scale": 1.5,
        "cfg_interval": [0.1, 1.0],
    }


def test_run_training_threads_inference_recipe_to_fid_callback(tmp_path):
    """run_training passes configured sampling knobs into FIDCallback."""
    recipe = {
        "latent_shape": (1, 4, 4, 4, 4),
        "spacing": [1.7, 1.7, 2],
        "modality": 1,
        "num_inference_steps": 15,
        "guidance_scale": 1.5,
        "cfg_interval": [0.1, 1.0],
    }

    trainer, _ = _run(tmp_path, inference_recipe=recipe)
    fid = next(c for c in trainer.callbacks if type(c).__name__ == "FIDCallback")

    assert fid.latent_shape == recipe["latent_shape"]
    assert fid.spacing == recipe["spacing"]
    assert fid.modality == recipe["modality"]
    assert fid.num_inference_steps == recipe["num_inference_steps"]
    assert fid.guidance_scale == recipe["guidance_scale"]
    assert fid.cfg_interval == recipe["cfg_interval"]


def test_run_training_writes_ckpt_and_logs_metrics(tmp_path):
    trainer, ckpt = _run(tmp_path, enable_fid=True)
    metrics = trainer.callback_metrics
    for key in ("train/loss_epoch", "train/grad_norm", "val/x0_mae", "val/fid_avg"):
        assert key in metrics, f"missing {key}"
        assert torch.isfinite(metrics[key])
    # A Lightning .ckpt was written (best-by-FID + last), full state.
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    # The monitored checkpoint resolved a best path distinct from last.ckpt.
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()
    assert Path(ckpt.best_model_path).name != "last.ckpt"


def test_export_bakes_slowest_ema_and_round_trips(tmp_path):
    """Export -> native dir -> from_pretrained -> decode == Module.sample + VAE."""
    module = _module()
    bundle = _bundle()  # ONE bundle: same held VAE for training, export, and decode
    _run(tmp_path, module=module, bundle=bundle, enable_fid=True)
    # The trainer's EMA callback holds the slow shadow module.sample must use.
    ema_cb = next(
        c for c in module.trainer.callbacks if type(c).__name__ == "DoubleEMACallback"
    )
    ckpt_path = str(Path(str(tmp_path)) / "last.ckpt")

    # Fresh UNet (built from a network config in the real CLI); export bakes the
    # slowest EMA shadow into it, the SAME held VAE carries scaling_factor.
    fresh_unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    export_to_native(
        ckpt_path, str(tmp_path / "native"),
        unet=fresh_unet, vae=bundle.vae, scheduler=FlowMatchHeunDiscreteScheduler(),
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "native"))

    # module.sample on the EMA-swapped weights, decoded through the held VAE,
    # must equal the exported pipeline's generate+decode (same seed).
    ema_cb.swap_in(module)
    try:
        g = torch.Generator().manual_seed(3)
        decode_mod = bundle.vae.decode(
            module.sample((1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, 2, generator=g)
        )
    finally:
        ema_cb.restore(module)
    g = torch.Generator().manual_seed(3)
    decode_pipe = loaded(
        (1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, num_inference_steps=2, generator=g
    )
    assert torch.allclose(decode_mod.float(), decode_pipe.float(), atol=1e-5)


def test_export_no_ema_bakes_raw_weights(tmp_path):
    module = _module()
    _run(tmp_path, module=module, enable_fid=True)
    ckpt_path = str(Path(str(tmp_path)) / "last.ckpt")
    fresh_unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    source = export_to_native(
        ckpt_path, str(tmp_path / "native_raw"),
        unet=fresh_unet, vae=AutoencoderKL(scaling_factor=0.5),
        scheduler=FlowMatchHeunDiscreteScheduler(), prefer_ema=False,
    )
    assert source == "unet_state_dict"
    LatentFlowPipeline.from_pretrained(str(tmp_path / "native_raw"))  # loads cleanly


def test_resume_from_checkpoint(tmp_path):
    """A second fit resumes via ckpt_path (full state: optim + LR + EMA)."""
    module = _module()
    _run(tmp_path / "run1", module=module, enable_fid=False)
    ckpt_path = str(Path(str(tmp_path / "run1")) / "last.ckpt")

    module2 = _module()
    trainer, _ = _run(tmp_path / "run2", module=module2, enable_fid=False, ckpt_path=ckpt_path)
    # Resumed fit ran (global_step advanced past the resumed epoch's batches).
    assert trainer.global_step > 0


# -- console entry -----------------------------------------------------------


_TINY_NETWORK_YAML = """\
spatial_dims: 3
image_channels: 1
latent_channels: 4
autoencoder:
  spatial_dims: ${spatial_dims}
  in_channels: ${image_channels}
  out_channels: ${image_channels}
  latent_channels: ${latent_channels}
  num_channels: [8, 8]
  num_res_blocks: [1, 1]
  norm_num_groups: 8
  norm_float16: false
  num_splits: 1
  save_mem: false
  scaling_factor: 1.0
diffusion_unet:
  spatial_dims: ${spatial_dims}
  in_channels: ${latent_channels}
  out_channels: ${latent_channels}
  num_channels: [8, 8]
  num_res_blocks: 1
  norm_num_groups: 8
  num_head_channels: [4, 4]
  attention_levels: [false, false]
  use_flash_attention: false
  include_spacing_input: true
  num_class_embeds: 4
  num_train_timesteps: 1000
scheduler:
  num_train_timesteps: 1000
  t_eps: 0.05
"""


def _write_tiny_configs(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(_TINY_NETWORK_YAML)
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
        + "model_filename: diff_unet_3d_rflow-brats2023.pt\n"
        "trained_autoencoder_path: /tmp/_unused_vae_\n"
        "val_subset_size: 4\n"
    )
    train = tmp_path / "train.yaml"
    train.write_text(
        "diffusion_unet_train: {batch_size: 2, lr: 1.0e-2, n_epochs: 1, lr_warmup_steps: 0}\n"
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05, ema: true, ema_decays: [0.9999, 0.9996]}\n"
        "fid_eval: {num_synth: 2, every_n_epochs: 1, center_slices_ratio: 0.5, cov_ridge: 1.0e-2}\n"
    )
    return str(env), str(train), str(net)


def test_main_runs_end_to_end_with_fake_data(tmp_path):
    """main(): argparse -> compose -> build -> fit -> ckpt (fake-data seam)."""
    env, train, net = _write_tiny_configs(tmp_path)

    def fake_provider(cfg, device):
        return _DataBundle(
            latent_ds=_LatentDS(), vae=AutoencoderKL(scaling_factor=0.5),
            val_latents=torch.randn(4, 4, 4, 4, 4),
        )

    rc = train_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1", "--no-fid"],
        data_provider=fake_provider,
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_export_cli_matches_function(tmp_path):
    """scripts/export_checkpoint.py CLI mirrors export_to_native (converter-style)."""
    module = _module()
    _run(tmp_path / "run", module=module, enable_fid=True)
    ckpt_path = str(tmp_path / "run" / "last.ckpt")

    env, train, net = _write_tiny_configs(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import export_checkpoint as cli  # type: ignore[import-not-found]

    rc = cli.main(
        ["--ckpt", ckpt_path, "--output", str(tmp_path / "cli_native"), "--network-config", net]
    )
    assert rc == 0
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "cli_native"))
    vol = loaded(
        (1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert vol.shape == (1, 1, 8, 8, 8)
    assert torch.isfinite(vol).all()
