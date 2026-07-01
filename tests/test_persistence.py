"""Native per-component checkpoint persistence (issue #7).

``save_pretrained`` → ``from_pretrained`` round-trip for the per-component
directory layout, plus the autocast regions the inference rollout and VAE decode
run under. The hope→native converter has been retired (ADR-0007): training emits
Lightning ``.ckpt``, exported to native via ``export_to_native``.
"""

from __future__ import annotations

import pytest
import torch

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowPipeline,
    UNet3DConditionModel,
)


def _tiny_unet() -> UNet3DConditionModel:
    torch.manual_seed(0)
    return UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)


def _build_pipeline():
    return LatentFlowPipeline(
        _tiny_unet(), AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    )


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


# -- pipeline autocast parity (issue #17) -----------------------------------


def test_pipeline_runs_rollout_under_autocast(pipeline, monkeypatch):
    """The rollout AND the VAE decode run under torch.autocast, cuda-only (issues #17/#18).

    The migrated VAE carries ``norm_float16``, so the decode must run under
    autocast too; both regions are disabled off-cuda so CPU results are unchanged.
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
