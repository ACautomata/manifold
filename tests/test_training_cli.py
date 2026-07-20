"""manifold-train + ModelCheckpoint + Export integration (Slice E, issue #28).

The end-to-end Training-pipeline + Export round-trip (the highest seam, per the
issue's testing plan): ``run_training`` on a tiny CPU config (tiny UNet + a
fake-latent cache + a fake feature network) writes a Lightning ``.ckpt`` and
logs ``train/loss_epoch`` / ``train/grad_norm`` / ``val/x0_mae`` / ``val/fid``
(single raw arm); Export always bakes the RAW UNet weights into a native dir
``Pipeline.from_pretrained`` loads, whose decode matches ``Module.sample()`` +
the held-VAE decode. Plus resume via ``ckpt_path`` and the ``main`` console entry.
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
    # allow_train_as_val=True: the smoke reuses the train fixture as val to
    # exercise the FID + x0-MAE plumbing (it tests wiring, not held-out
    # generalization). Production leaves this False -> validation disabled.
    return _DataBundle(
        latent_ds=_LatentDS(), vae=vae, val_latents=torch.randn(5, 4, 4, 4, 4),
        allow_train_as_val=True,
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
    for key in ("train/loss_epoch", "train/grad_norm", "val/x0_mae", "val/fid"):
        assert key in metrics, f"missing {key}"
        assert torch.isfinite(metrics[key])
    # A Lightning .ckpt was written (best-by-FID + last), full state.
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    # The monitored checkpoint resolved a best path distinct from last.ckpt.
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()
    assert Path(ckpt.best_model_path).name != "last.ckpt"


def test_run_training_skips_validation_when_no_held_out_val(tmp_path):
    """Production (allow_train_as_val=False, no val plumbing): validation is DISABLED.

    The regular ``manifold-train`` flow has no held-out val source, so it must NOT
    silently reuse train as val. With ``allow_train_as_val=False`` (the default) and
    ``enable_fid=True``, no FIDCallback is attached, no ``val/x0_mae`` is logged, the
    checkpoint monitor is dropped (no val/* metric to monitor), and no ``val/*`` key
    appears - validation is skipped, never run on train data.
    """
    vae = AutoencoderKL(scaling_factor=0.5)
    bundle = _DataBundle(  # allow_train_as_val defaults False -> production: no val
        latent_ds=_LatentDS(), vae=vae, val_latents=torch.randn(5, 4, 4, 4, 4)
    )
    trainer, ckpt = run_training(
        module=_module(), bundle=bundle, feature_net=_FakeFeatureNet(),
        model_dir=str(tmp_path), max_epochs=1, devices=1, accelerator="cpu",
        enable_fid=True, num_synth=2, limit_val_batches=2, cov_ridge=1e-2,
    )
    assert not any(type(c).__name__ == "FIDCallback" for c in trainer.callbacks)
    assert not any(type(c).__name__ == "LatentX0MAE" for c in trainer.callbacks)
    assert ckpt.monitor is None  # no val/* metric to monitor
    assert not any(k.startswith("val/") for k in trainer.callback_metrics)
    assert any(Path(str(tmp_path)).glob("*.ckpt"))  # training still produced a ckpt
    # Regression (codex #90 P1): the no-val path must NOT set
    # ``check_val_every_n_epoch=None`` - Lightning's contract then requires an integer
    # ``val_check_interval``, which the float default violates (a MisconfigurationException
    # on versions that enforce it). ``limit_val_batches=0`` alone makes val a no-op.
    assert trainer.check_val_every_n_epoch is not None


def test_run_training_enables_val_when_has_val(tmp_path):
    """bundle.has_val=True (a held-out val split is configured) enables validation:
    FIDCallback + LatentX0MAE attach and val/fid is logged. Mirrors the production
    path where _warm_data warms val_data_base_dir into val_latent_ds (cold path:
    warm_fn returns the (train_ds, vae, val_latent_ds) triple)."""
    class _ValLatentDS(Dataset):
        scaling_factor = 0.5

        def __len__(self):
            return 6

        def _lat(self, i):
            torch.manual_seed(i)
            return torch.randn(4, 4, 4, 4)

        def raw_latent(self, i):
            return self._lat(i)

        def __getitem__(self, i):
            return {
                "latent": self._lat(i) * self.scaling_factor,
                "spacing": torch.tensor([1.0, 1.0, 1.0]),
                "label": torch.tensor(i % 3, dtype=torch.long),
            }

    vae = AutoencoderKL(scaling_factor=0.5)
    train_ds, val_ds = _LatentDS(), _ValLatentDS()

    def warm_fn():
        return train_ds, vae, val_ds

    bundle = _DataBundle(vae=vae, warm_fn=warm_fn, has_val=True)
    trainer, ckpt = run_training(
        module=_module(), bundle=bundle, feature_net=_FakeFeatureNet(),
        model_dir=str(tmp_path), max_epochs=1, devices=1, accelerator="cpu",
        enable_fid=True, num_synth=2, limit_val_batches=2, cov_ridge=1e-2,
        val_subset_size=4,
        # Cold path (val_latents=None): the latent_shape must be supplied explicitly
        # (main() derives it via _derive_latent_shape); match _ValLatentDS's (4,4,4,4).
        inference_recipe={
            "latent_shape": (1, 4, 4, 4, 4), "spacing": [1.0, 1.0, 1.0], "modality": 1,
            "num_inference_steps": 2, "guidance_scale": 1.0, "cfg_interval": None,
        },
    )
    assert any(type(c).__name__ == "FIDCallback" for c in trainer.callbacks)
    assert any(type(c).__name__ == "LatentX0MAE" for c in trainer.callbacks)
    assert ckpt.monitor == "val/fid"  # held-out val -> monitor active
    assert "val/fid" in trainer.callback_metrics


# -- held-out val plumbing (_warm_data reads val_data_base_dir) ----------------


def test_warm_data_marks_has_val_for_val_dir(tmp_path, monkeypatch):
    """_warm_data sets bundle.has_val=True when val_data_base_dir is a directory
    and its warm_fn returns the (train_latent_ds, vae, val_latent_ds) triple with
    the train-estimated scale_factor propagated to the val cache (scale-consistency:
    one factor over both splits, never re-estimated on val)."""
    from manifold.data import latent_dataset as lds_mod
    from manifold.data import latent_pipeline, volume_dataset
    from manifold.training.cli import _warm_data

    (tmp_path / "val").mkdir()  # existing directory -> has_val=True

    class _FakeVAE:
        def to(self, device):
            return self

    class _FakeVolDs:
        def __len__(self):
            return 8

    class _FakeLatentDs:
        def __init__(self):
            self.scaling_factor = 1.0

        def warm_cache(self, device, show_progress=True):
            pass

        def free_encoder(self):
            pass

    class _FakePipeline:
        latent_ds = _FakeVolDs()
        autoencoder = _FakeVAE()
        scale_factor = 2.5

    fake_val_ds = _FakeLatentDs()
    monkeypatch.setattr(latent_pipeline, "build_volume_dataset",
                        lambda cfg, **kw: (_FakeVolDs(), lambda fn, meta: 0))
    monkeypatch.setattr(latent_pipeline, "build_encode_pipeline",
                        lambda cfg, **kw: (_FakeVAE(), "encode_fn"))
    monkeypatch.setattr(volume_dataset, "NiftiVolumeDataset",
                        lambda *a, **kw: _FakeVolDs())
    monkeypatch.setattr(lds_mod, "LatentDataset", lambda *a, **kw: fake_val_ds)
    monkeypatch.setattr(latent_pipeline, "warm_latent_pipeline",
                        lambda *a, **kw: _FakePipeline())

    cfg = OmegaConf.create({
        "data_base_dir": str(tmp_path / "train"),
        "val_data_base_dir": str(tmp_path / "val"),
        "model_dir": str(tmp_path / "model"),
        "val_subset_size": 4,
        "diffusion_unet_inference": {"dim": [4, 4, 4], "modality": 1},
        "autoencoder": {"num_channels": [8, 8]},
    })
    bundle, num_examples = _warm_data(cfg, torch.device("cpu"))
    assert bundle.has_val is True
    assert num_examples == 8

    train_ds, vae, val_ds = bundle.warm_fn()
    assert val_ds is fake_val_ds
    assert fake_val_ds.scaling_factor == 2.5  # train scale propagated to val


def test_warm_data_no_val_when_val_data_base_dir_unset(tmp_path, monkeypatch):
    """_warm_data leaves has_val=False when val_data_base_dir is unset (validation
    stays disabled — the pre-held-out-val behavior)."""
    from manifold.data import latent_pipeline
    from manifold.training.cli import _warm_data

    class _FakeVolDs:
        def __len__(self):
            return 8

    monkeypatch.setattr(latent_pipeline, "build_volume_dataset",
                        lambda cfg, **kw: (_FakeVolDs(), lambda fn, meta: 0))
    monkeypatch.setattr(latent_pipeline, "build_encode_pipeline",
                        lambda cfg, **kw: ("vae", "encode_fn"))

    cfg = OmegaConf.create({
        "data_base_dir": str(tmp_path / "train"),
        "model_dir": str(tmp_path / "model"),
        "diffusion_unet_inference": {"dim": [4, 4, 4], "modality": 1},
        "autoencoder": {"num_channels": [8, 8]},
    })
    bundle, _ = _warm_data(cfg, torch.device("cpu"))
    assert bundle.has_val is False


def test_build_checkpoint_monitor_matches_logged_arm(tmp_path):
    """The checkpoint monitor tracks the single arm that is logged: ``val/fid``
    (the raw optimizer arm). A mismatch would make Lightning error on a
    never-logged monitored metric."""
    from manifold.training.cli import _build_checkpoint

    assert _build_checkpoint(str(tmp_path / "a"), monitor_fid=True).monitor == "val/fid"


def test_export_bakes_raw_and_round_trips(tmp_path):
    """Export -> native dir -> from_pretrained -> decode == Module.sample + VAE
    (on the raw optimizer weights; export always bakes raw now)."""
    module = _module()
    bundle = _bundle()  # ONE bundle: same held VAE for training, export, and decode
    _run(tmp_path, module=module, bundle=bundle, enable_fid=True)
    ckpt_path = str(Path(str(tmp_path)) / "last.ckpt")

    # Fresh UNet (built from a network config in the real CLI); export bakes the
    # RAW optimizer weights into it, the SAME held VAE carries scaling_factor.
    fresh_unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    export_to_native(
        ckpt_path, str(tmp_path / "native"),
        unet=fresh_unet, vae=bundle.vae, scheduler=FlowMatchHeunDiscreteScheduler(),
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "native"))

    # module.sample on the raw weights, decoded through the held VAE, must equal
    # the exported pipeline's generate+decode (same seed).
    g = torch.Generator().manual_seed(3)
    decode_mod = bundle.vae.decode(
        module.sample((1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, 2, generator=g)
    )
    g = torch.Generator().manual_seed(3)
    decode_pipe = loaded(
        (1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, num_inference_steps=2, generator=g
    )
    assert torch.allclose(decode_mod.float(), decode_pipe.float(), atol=1e-5)


def test_export_default_bakes_raw_weights(tmp_path):
    """The export bakes the RAW UNet weights (aligned with the val/fid
    checkpoint monitor)."""
    module = _module()
    _run(tmp_path, module=module, enable_fid=True)
    ckpt_path = str(Path(str(tmp_path)) / "last.ckpt")
    fresh_unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    source = export_to_native(
        ckpt_path, str(tmp_path / "native_raw"),
        unet=fresh_unet, vae=AutoencoderKL(scaling_factor=0.5),
        scheduler=FlowMatchHeunDiscreteScheduler(),
    )
    assert source == "unet_state_dict"
    LatentFlowPipeline.from_pretrained(str(tmp_path / "native_raw"))  # loads cleanly


def test_resume_from_checkpoint(tmp_path):
    """A second fit resumes via ckpt_path (full state: optim + LR)."""
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
        "formulation: {p_mean: -0.8, p_std: 0.8, t_eps: 0.05}\n"
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


# -- ControlNet export baking (issue #144) -------------------------------------


def _tiny_base_and_controlnet():
    """A seeded tiny base UNet + ControlNet with NON-trivial ControlNet weights.

    The ControlNet clones the base encoder, then its zero-conv out projections are
    perturbed so the round-trip assertion is load-bearing (a zero-init ControlNet
    would round-trip trivially). The base ``out`` conv is re-initialized too (the
    MONAI zero-init would mask the residual effect, mirroring the Mode-2 helpers).
    """
    from manifold import ControlNet3DConditionModel

    torch.manual_seed(0)
    base = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in base.unet.out.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    controlnet.load_base_encoder_weights(base)
    with torch.no_grad():
        for p in controlnet.parameters():
            p.add_(0.001 * torch.randn_like(p))  # non-trivial ControlNet weights
    return base, controlnet


def _write_supervised_controlnet_ckpt(path, controlnet) -> None:
    """Write a supervised-ControlNet Lightning ckpt (state_dict = controlnet.* only).

    Mirrors the ControlNetLatentFlowModule checkpoint layout: the trainable ControlNet
    is the ONLY registered arm (the frozen base is held unregistered, off the
    checkpoint), so the state_dict keys are rooted at ``controlnet.``.
    """
    state = {f"controlnet.{k}": v for k, v in controlnet.state_dict().items()}
    torch.save({"state_dict": state}, str(path))


def test_export_controlnet_bakes_and_round_trips(tmp_path):
    """A supervised ControlNet ckpt exports to a native dir that from_pretrained loads
    back into a base UNet + ControlNet + VAE with bit-identical weights (#144)."""
    from manifold import ControlNetLatentFlowPipeline

    base, controlnet = _tiny_base_and_controlnet()
    ckpt_path = tmp_path / "controlnet.ckpt"
    _write_supervised_controlnet_ckpt(ckpt_path, controlnet)

    # Fresh base + ControlNet (built from the network config in the real CLI). The
    # export bakes the ckpt's controlnet.* into the fresh ControlNet; the base is
    # passed through verbatim (the supervised ckpt carries no base keys).
    fresh_base, fresh_controlnet = _tiny_base_and_controlnet()
    # Scramble the fresh ControlNet so the bake is what restores the weights.
    with torch.no_grad():
        for p in fresh_controlnet.parameters():
            p.add_(torch.randn_like(p))
    vae = AutoencoderKL(scaling_factor=0.5)

    source = export_to_native(
        str(ckpt_path), str(tmp_path / "native"),
        unet=fresh_base, controlnet=fresh_controlnet, vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=ControlNetLatentFlowPipeline,
    )
    assert source == "controlnet_state_dict"

    loaded = ControlNetLatentFlowPipeline.from_pretrained(str(tmp_path / "native"))
    # Base UNet + ControlNet round-trip bit-identical to the originals.
    for k, v in base.state_dict().items():
        assert torch.equal(loaded.unet.state_dict()[k], v), f"base mismatch at {k}"
    for k, v in controlnet.state_dict().items():
        assert torch.equal(loaded.controlnet.state_dict()[k], v), f"controlnet mismatch at {k}"
    assert float(loaded.vae.scaling_factor) == 0.5


def test_export_controlnet_raises_without_controlnet_keys(tmp_path):
    """A ckpt with no controlnet.* keys (e.g. a plain JiT ckpt) -> clear ValueError."""
    from manifold import ControlNetLatentFlowPipeline

    base, _ = _tiny_base_and_controlnet()
    ckpt_path = tmp_path / "jit.ckpt"
    # A plain JiT-style ckpt: only unet.unet.* keys, no controlnet.* keys.
    torch.save(
        {"state_dict": {f"unet.unet.{k}": v for k, v in base.unet.state_dict().items()}},
        str(ckpt_path),
    )
    fresh_base, fresh_controlnet = _tiny_base_and_controlnet()
    import pytest

    with pytest.raises(ValueError, match="controlnet"):
        export_to_native(
            str(ckpt_path), str(tmp_path / "native"),
            unet=fresh_base, controlnet=fresh_controlnet,
            vae=AutoencoderKL(scaling_factor=0.5),
            scheduler=FlowMatchHeunDiscreteScheduler(),
            pipeline_cls=ControlNetLatentFlowPipeline,
        )


def test_export_cli_controlnet_pipeline(tmp_path):
    """--pipeline controlnet: load the frozen base export + bake the ControlNet (codex #152 P1).

    The export CLI's controlnet mode loads the frozen base UNet + VAE scale from
    --base-native-dir (the JiT export the supervised ControlNet trained against),
    builds the ControlNet from the network config, bakes the ckpt's controlnet.*
    weights, and writes a ControlNetLatentFlowPipeline dir that from_pretrained
    round-trips bit-identical.
    """
    from manifold import ControlNetLatentFlowPipeline
    from manifold.config import load_config
    from manifold.config.builder import build_controlnet, build_pipeline

    env, train, net = _write_tiny_configs(tmp_path)
    cfg = load_config(net, None, net)

    # A JiT base export (the supervised stage's frozen base + VAE scale source).
    base_pipe = build_pipeline(cfg)
    base_dir = tmp_path / "base_native"
    base_pipe.save_pretrained(str(base_dir))

    # A supervised-ControlNet ckpt (controlnet.* keys only) with non-trivial weights.
    controlnet = build_controlnet(cfg)
    controlnet.load_base_encoder_weights(base_pipe.unet)
    with torch.no_grad():
        for p in controlnet.parameters():
            p.add_(0.001 * torch.randn_like(p))
    ckpt_path = tmp_path / "controlnet.ckpt"
    _write_supervised_controlnet_ckpt(ckpt_path, controlnet)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import export_checkpoint as cli  # type: ignore[import-not-found]

    rc = cli.main(
        ["--ckpt", str(ckpt_path), "--output", str(tmp_path / "cn_native"),
         "--network-config", net, "--pipeline", "controlnet",
         "--base-native-dir", str(base_dir)]
    )
    assert rc == 0

    loaded = ControlNetLatentFlowPipeline.from_pretrained(str(tmp_path / "cn_native"))
    # Base UNet + ControlNet round-trip bit-identical to the originals.
    for k, v in base_pipe.unet.state_dict().items():
        assert torch.equal(loaded.unet.state_dict()[k], v), f"base mismatch at {k}"
    for k, v in controlnet.state_dict().items():
        assert torch.equal(loaded.controlnet.state_dict()[k], v), f"controlnet mismatch at {k}"


def test_export_cli_controlnet_requires_base_native_dir(tmp_path):
    """--pipeline controlnet without --base-native-dir -> clear ValueError (codex #152 P1)."""
    import pytest

    env, train, net = _write_tiny_configs(tmp_path)
    ckpt_path = tmp_path / "c.ckpt"
    torch.save({"state_dict": {}}, str(ckpt_path))

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import export_checkpoint as cli  # type: ignore[import-not-found]

    with pytest.raises(ValueError, match="base-native-dir"):
        cli.main(
            ["--ckpt", str(ckpt_path), "--output", str(tmp_path / "cn_native"),
             "--network-config", net, "--pipeline", "controlnet"]
        )
