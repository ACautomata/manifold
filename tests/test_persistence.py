"""Checkpoint persistence + hope converter (Seam 1 migration half, issue #7).

Native per-component round-trip (``save_pretrained`` → ``from_pretrained``), the
one-shot hope converter (``unet_state_dict → unet`` with the slowest EMA shadow
baked as inference weights, ``scale_factor → vae.scaling_factor``), and the rule
that ``from_pretrained`` refuses hope's flat format.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowPipeline,
    UNet3DConditionModel,
)
from manifold.pipelines import convert_hope_checkpoint

#: The raw MONAI MAISI UNet class hope builds its ``unet_state_dict`` from.
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

#: The raw MONAI MAISI autoencoder class hope's ``autoencoder_v1.pt`` holds.
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
)

# Tiny latent shape consistent with the default 2-level VAE (latent divisor 2).
LATENT_SHAPE = (1, 4, 4, 4, 4)


def _tiny_unet() -> UNet3DConditionModel:
    torch.manual_seed(0)
    return UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)


def _build_pipeline():
    return LatentFlowPipeline(
        _tiny_unet(), AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    )


def _raw_maisi_unet():
    """A raw MONAI MAISI UNet matching the manifold wrapper's default config.

    hope's ``unet_state_dict`` (and EMA shadows) are a raw MAISI UNet's state dict
    — no wrapper prefix — so the fixture builds them this way to faithfully mimic
    the legacy format being migrated.
    """
    return DiffusionModelUNetMaisi(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        num_channels=(8, 8),
        num_res_blocks=(1, 1),
        attention_levels=(False, False),
        norm_num_groups=8,
        num_head_channels=4,
        num_class_embeds=4,
        include_spacing_input=True,
    )


def _raw_maisi_vae():
    """A raw MONAI MAISI autoencoder matching the manifold VAE wrapper's default.

    hope's ``autoencoder_v1.pt`` is a raw MAISI autoencoder state dict (bare keys,
    no wrapper prefix, optionally wrapped in ``{"unet_state_dict": ...}``). Built
    with the wrapper's tiny-CPU construction kwargs so the keys line up for a
    strict load.
    """
    return AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=(8, 8),
        num_res_blocks=(1, 1),
        attention_levels=(False, False),
        latent_channels=4,
        norm_num_groups=8,
        num_splits=1,
        save_mem=False,
    )


def _hope_fixture(tmp_path, *, with_ema: bool, scale_factor: float = 0.815):
    """Synthesize a hope flat checkpoint from random raw MAISI UNets."""
    torch.manual_seed(1)
    raw = _raw_maisi_unet()
    torch.manual_seed(2)
    ema = _raw_maisi_unet()
    payload = OrderedDict(
        unet_state_dict=raw.state_dict(),
        scale_factor=torch.tensor(scale_factor),
        num_train_timesteps=1000,
    )
    if with_ema:
        payload["ema"] = {"shadows": [ema.state_dict()], "decays": [0.9999]}
    path = str(tmp_path / "hope.pt")
    torch.save(payload, path)
    return path, raw, ema


# -- native round-trip -------------------------------------------------------


def test_save_load_round_trip(tmp_path):
    pipe = _build_pipeline()
    pipe.save_pretrained(str(tmp_path / "pipe"))
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "pipe"))

    seed = torch.Generator().manual_seed(7)
    a = pipe([1, 4, 4, 4, 4], [1.0, 1.0, 1.0], 2, num_inference_steps=3, generator=seed)
    seed = torch.Generator().manual_seed(7)
    b = loaded([1, 4, 4, 4, 4], [1.0, 1.0, 1.0], 2, num_inference_steps=3, generator=seed)
    assert torch.allclose(a, b)


def test_scaling_factor_round_trips(tmp_path):
    pipe = _build_pipeline()
    pipe.save_pretrained(str(tmp_path / "pipe"))
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "pipe"))
    assert loaded.vae.scaling_factor.item() == pytest.approx(0.5)


def test_components_keep_their_class_after_load(tmp_path):
    pipe = _build_pipeline()
    pipe.save_pretrained(str(tmp_path / "pipe"))
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "pipe"))
    assert type(loaded.unet) is UNet3DConditionModel
    assert type(loaded.vae) is AutoencoderKL
    assert type(loaded.scheduler) is FlowMatchHeunDiscreteScheduler


# -- converter ---------------------------------------------------------------


def test_converter_bakes_ema_as_unet_weights(tmp_path):
    path, raw, ema = _hope_fixture(tmp_path, with_ema=True)
    convert_hope_checkpoint(
        path,
        str(tmp_path / "conv"),
        _tiny_unet(),
        AutoencoderKL(scaling_factor=1.0),
        FlowMatchHeunDiscreteScheduler(),
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))
    inner = loaded.unet.unet.state_dict()  # raw MAISI keys (no wrapper prefix)
    ema_sd = ema.state_dict()
    assert all(torch.equal(ema_sd[k], inner[k]) for k in ema_sd)  # EMA shadow baked in
    # ... and NOT the raw unet_state_dict (distinct from the EMA copy):
    raw_sd = raw.state_dict()
    assert not all(torch.equal(raw_sd[k], inner[k]) for k in raw_sd)


def test_converter_maps_scale_factor(tmp_path):
    path, _, _ = _hope_fixture(tmp_path, with_ema=True, scale_factor=0.815)
    convert_hope_checkpoint(
        path,
        str(tmp_path / "conv"),
        _tiny_unet(),
        AutoencoderKL(scaling_factor=1.0),
        FlowMatchHeunDiscreteScheduler(),
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))
    assert loaded.vae.scaling_factor.item() == pytest.approx(0.815)


def test_converter_without_ema_uses_unet_state_dict(tmp_path):
    path, raw, _ = _hope_fixture(tmp_path, with_ema=False)
    convert_hope_checkpoint(
        path,
        str(tmp_path / "conv"),
        _tiny_unet(),
        AutoencoderKL(scaling_factor=1.0),
        FlowMatchHeunDiscreteScheduler(),
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))
    inner = loaded.unet.unet.state_dict()
    raw_sd = raw.state_dict()
    assert all(torch.equal(raw_sd[k], inner[k]) for k in raw_sd)


def test_end_to_end_convert_load_generate(tmp_path):
    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    convert_hope_checkpoint(
        path,
        str(tmp_path / "conv"),
        _tiny_unet(),
        AutoencoderKL(scaling_factor=1.0),
        FlowMatchHeunDiscreteScheduler(),
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))
    vol = loaded(
        [1, 4, 4, 4, 4],
        [1.0, 1.0, 1.0],
        2,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(11),
    )
    assert vol.shape == (1, 1, 8, 8, 8)
    assert torch.isfinite(vol).all()


def test_end_to_end_convert_load_generate_steps_15(tmp_path):
    """CPU smoke (issue #17): from_pretrained + __call__ at steps=15 -> finite."""
    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    convert_hope_checkpoint(
        path,
        str(tmp_path / "conv"),
        _tiny_unet(),
        AutoencoderKL(scaling_factor=1.0),
        FlowMatchHeunDiscreteScheduler(),
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))
    vol = loaded(
        [1, 4, 4, 4, 4],
        [1.0, 1.0, 1.0],
        2,
        num_inference_steps=15,
        generator=torch.Generator().manual_seed(11),
    )
    assert vol.shape == (1, 1, 8, 8, 8)
    assert torch.isfinite(vol).all()


# -- converter: VAE-weights fix (issue #17) ---------------------------------


def _vae_fixture(tmp_path, *, wrapped: bool) -> str:
    """Synthesize a hope VAE checkpoint from a distinct raw MAISI autoencoder."""
    torch.manual_seed(3)
    raw_vae = _raw_maisi_vae()
    sd = raw_vae.state_dict()
    payload = {"unet_state_dict": sd} if wrapped else sd  # MAISI wrapper form / bare
    path = str(tmp_path / "autoencoder_v1.pt")
    torch.save(payload, path)
    return path, raw_vae


def test_converter_bakes_vae_weights_decode_non_random(tmp_path):
    """A VAE checkpoint is baked into vae.autoencoder so the converted VAE decodes.

    Without the fix the converted VAE keeps its random-init weights (decodes
    noise); with ``vae_checkpoint`` the loaded VAE's decode equals the source
    autoencoder's decode (non-random). Exercises both the ``{unet_state_dict}``
    wrapper form and the bare form.
    """
    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    vae_ckpt, raw_vae = _vae_fixture(tmp_path, wrapped=True)
    convert_hope_checkpoint(
        path,
        str(tmp_path / "conv"),
        _tiny_unet(),
        AutoencoderKL(scaling_factor=1.0),
        FlowMatchHeunDiscreteScheduler(),
        vae_checkpoint=vae_ckpt,
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))

    # The source autoencoder's weights are baked (not the wrapper's random init).
    raw_sd = raw_vae.state_dict()
    loaded_sd = loaded.vae.autoencoder.state_dict()
    assert all(torch.equal(raw_sd[k], loaded_sd[k]) for k in raw_sd)

    # The decode is non-random: loaded VAE decode (scaled input) == source decode.
    z_raw = torch.randn(*LATENT_SHAPE)
    scale = loaded.vae.scaling_factor.item()
    decoded_loaded = loaded.vae.decode(z_raw * scale)
    decoded_source = raw_vae.decode_stage_2_outputs(z_raw)
    assert decoded_loaded.shape == decoded_source.shape
    assert torch.allclose(decoded_loaded.float(), decoded_source.float(), atol=1e-5)


def test_converter_accepts_bare_vae_state_dict(tmp_path):
    """The VAE checkpoint may also be a bare autoencoder state dict (no wrapper)."""
    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    vae_ckpt, raw_vae = _vae_fixture(tmp_path, wrapped=False)
    convert_hope_checkpoint(
        path,
        str(tmp_path / "conv"),
        _tiny_unet(),
        AutoencoderKL(scaling_factor=1.0),
        FlowMatchHeunDiscreteScheduler(),
        vae_checkpoint=vae_ckpt,
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))
    raw_sd = raw_vae.state_dict()
    loaded_sd = loaded.vae.autoencoder.state_dict()
    assert all(torch.equal(raw_sd[k], loaded_sd[k]) for k in raw_sd)


def test_converter_without_vae_checkpoint_keeps_construction_vae(tmp_path):
    """Back-compat: no vae_checkpoint -> the VAE keeps its (random-init) weights."""
    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    vae = AutoencoderKL(scaling_factor=1.0)
    before = {k: v.clone() for k, v in vae.autoencoder.state_dict().items()}
    convert_hope_checkpoint(
        path, str(tmp_path / "conv"), _tiny_unet(), vae, FlowMatchHeunDiscreteScheduler()
    )
    loaded = LatentFlowPipeline.from_pretrained(str(tmp_path / "conv"))
    after = loaded.vae.autoencoder.state_dict()
    # VAE weights unchanged (only scaling_factor was mapped from the hope ckpt).
    assert all(torch.equal(before[k], after[k]) for k in before)


# -- pipeline autocast parity (issue #17) -----------------------------------


def test_pipeline_runs_rollout_under_autocast(pipeline, monkeypatch):
    """The rollout AND the VAE decode run under torch.autocast, cuda-only (issues #17/#18).

    The rollout matches hope's ``sample_x0``; the decode matches hope's autocast-
    wrapped ReconModel decode (the migrated VAE carries norm_float16). Both are
    disabled off-cuda so CPU results are unchanged.
    """
    import manifold.pipelines.latent_flow as lf

    calls: list[dict] = []
    real_autocast = torch.autocast

    def spy_autocast(device_type, enabled=None, **kwargs):
        calls.append({"device_type": device_type, "enabled": enabled})
        return real_autocast(device_type, enabled=enabled or False, **kwargs)

    monkeypatch.setattr(torch, "autocast", spy_autocast)
    lf.LatentFlowPipeline  # touch the module so the import is used (no-op guard)

    pipeline(
        [1, 4, 4, 4, 4],
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(0),
    )
    assert calls, "torch.autocast was never entered"
    # Two autocast regions: the Heun rollout AND the VAE decode. The migrated VAE
    # carries norm_float16, so the decode must run under autocast too (issue #18);
    # both are enabled only on cuda.
    assert len(calls) == 2
    assert [c["device_type"] for c in calls] == ["cpu", "cpu"]
    assert all(c["enabled"] is False for c in calls)  # both disabled off-cuda


def test_from_pretrained_rejects_hope_flat(tmp_path):
    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    with pytest.raises(ValueError, match="hope flat checkpoint"):
        LatentFlowPipeline.from_pretrained(path)


def test_cli_convert_matches_function(tmp_path):
    """The scripts/ CLI builds from the OmegaConf network config + bakes the VAE."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import convert_hope_checkpoint as cli  # type: ignore[import-not-found]

    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    vae_ckpt, raw_vae = _vae_fixture(tmp_path, wrapped=True)
    # A tiny CPU-runnable network config (mirrors the conftest fixtures).
    (tmp_path / "network.yaml").write_text(
        "spatial_dims: 3\nimage_channels: 1\nlatent_channels: 4\n"
        "autoencoder:\n"
        "  spatial_dims: ${spatial_dims}\n  in_channels: ${image_channels}\n"
        "  out_channels: ${image_channels}\n  latent_channels: ${latent_channels}\n"
        "  num_channels: [8, 8]\n  num_res_blocks: [1, 1]\n  norm_num_groups: 8\n"
        "  norm_float16: false\n  num_splits: 1\n  save_mem: false\n  scaling_factor: 1.0\n"
        "diffusion_unet:\n"
        "  spatial_dims: ${spatial_dims}\n  in_channels: ${latent_channels}\n"
        "  out_channels: ${latent_channels}\n  num_channels: [8, 8]\n"
        "  num_res_blocks: 1\n  norm_num_groups: 8\n  num_head_channels: [4, 4]\n"
        "  attention_levels: [false, false]\n  use_flash_attention: false\n"
        "  include_spacing_input: true\n  num_class_embeds: 4\n  num_train_timesteps: 1000\n"
        "scheduler:\n  num_train_timesteps: 1000\n  t_eps: 0.05\n"
    )
    out = str(tmp_path / "cli_conv")
    rc = cli.main(
        [
            "--hope",
            path,
            "--output",
            out,
            "--network-config",
            str(tmp_path / "network.yaml"),
            "--vae-checkpoint",
            vae_ckpt,
        ]
    )
    assert rc == 0
    loaded = LatentFlowPipeline.from_pretrained(out)
    assert loaded.vae.scaling_factor.item() == pytest.approx(0.815)  # mapped from hope ckpt
    # VAE weights baked from the checkpoint (not the wrapper's random init).
    raw_sd = raw_vae.state_dict()
    loaded_sd = loaded.vae.autoencoder.state_dict()
    assert all(torch.equal(raw_sd[k], loaded_sd[k]) for k in raw_sd)
