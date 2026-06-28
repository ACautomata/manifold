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


def test_from_pretrained_rejects_hope_flat(tmp_path):
    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    with pytest.raises(ValueError, match="hope flat checkpoint"):
        LatentFlowPipeline.from_pretrained(path)


def test_cli_convert_matches_function(tmp_path):
    """The scripts/ CLI produces the same output as the converter function."""
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import convert_hope_checkpoint as cli  # type: ignore[import-not-found]

    path, _, _ = _hope_fixture(tmp_path, with_ema=True)
    cfg = {"num_class_embeds": 4, "include_spacing_input": True}
    vcfg = {"scaling_factor": 1.0}
    (tmp_path / "unet.json").write_text(json.dumps(cfg))
    (tmp_path / "vae.json").write_text(json.dumps(vcfg))
    out = str(tmp_path / "cli_conv")
    rc = cli.main(
        [
            "--hope",
            path,
            "--output",
            out,
            "--unet-config",
            str(tmp_path / "unet.json"),
            "--vae-config",
            str(tmp_path / "vae.json"),
        ]
    )
    assert rc == 0
    loaded = LatentFlowPipeline.from_pretrained(out)
    assert loaded.vae.scaling_factor.item() == pytest.approx(0.815)
