"""``manifold-eval`` console-entry tests (issue #229, ADR-0036 / ADR-0033).

The console-entry seam: a tiny **before native export** on disk plus a tiny
**fake after training checkpoint** (a ``{"state_dict": ...}`` dict carrying the
same ``unet.unet.*`` / ``controlnet.*`` prefixes a real GRPO Lightning ``.ckpt``
registers), driven through ``main(argv)`` on CPU. This pins the end-to-end
wiring — export the after-ckpt via the ADR-0006 bridge, build both pipelines
via ``from_pretrained``, run the #228 driver, write the artifacts — without any
GPU or real checkpoint (the repo's tiny-component injection pattern; prior art:
``test_before_after_eval.py`` / ``test_export_checkpoint.py``).

The ControlNet data path (BraTS manifest + paired latent cache) is injected
through the ``pairs_provider`` seam, mirroring ``grpo_cli``'s ``data_provider``.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest
import torch
import torch.nn as nn

from manifold import (
    AutoencoderKL,
    ControlNet3DConditionModel,
    ControlNetLatentFlowPipeline,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowPipeline,
    UNet3DConditionModel,
)
from manifold.eval.cli import main

# Tiny shapes mirroring tests/test_before_after_eval.py: the JiT grid needs no
# SSIM (4³ latent → 8³ image); the paired pass scores 3D SSIM (win_size 11), so
# its decoded image must be ≥ 11 voxels per dim (8³ latent → 16³ image).
JIT_LATENT_SHAPE = (1, 4, 4, 4, 4)
PAIRED_LATENT_SHAPE = (1, 4, 8, 8, 8)

#: The constant weight offset baked into the fake after-ckpt — any non-zero
#: value makes "the after export carries the ckpt weights, not the before ones"
#: assertable elementwise.
_PERTURB = 0.25


def _is_valid_png(path):
    with open(path, "rb") as fh:
        return fh.read(8) == b"\x89PNG\r\n\x1a\n" and os.path.getsize(path) > 100


def _jit_native_dir(tmp_path):
    """A tiny JiT native export (the *before* artifact)."""
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    vae = AutoencoderKL(scaling_factor=0.5)
    pipe = LatentFlowPipeline(unet, vae, FlowMatchHeunDiscreteScheduler())
    before_dir = tmp_path / "before_jit"
    pipe.save_pretrained(str(before_dir))
    return pipe, before_dir


def _controlnet_native_dir(tmp_path):
    """A tiny supervised-ControlNet native export (the *before* artifact)."""
    torch.manual_seed(0)
    base = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in base.unet.out.parameters():  # MONAI zero-inits the final projection
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    torch.manual_seed(1)
    controlnet = ControlNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    controlnet.load_base_encoder_weights(base)
    vae = AutoencoderKL(scaling_factor=0.5)
    pipe = ControlNetLatentFlowPipeline(
        base, controlnet, vae, FlowMatchHeunDiscreteScheduler()
    )
    before_dir = tmp_path / "before_controlnet"
    pipe.save_pretrained(str(before_dir))
    return pipe, before_dir


def _fake_ckpt(path, component, prefix):
    """A fake GRPO-style Lightning ckpt registering *component*'s weights.

    Mirrors what the real checkpoints carry: a GRPO JiT ckpt registers the
    trainable policy under ``unet.unet.*`` (the wrapper's MAISI backbone) and a
    GRPO ControlNet ckpt registers the trainable ControlNet under
    ``controlnet.*`` (the frozen base is dual-excluded off the ckpt). The +0.25
    offset makes the baked weights distinguishable from the before export's.
    """
    state = {f"{prefix}{k}": v.detach() + _PERTURB for k, v in component.state_dict().items()}
    torch.save({"state_dict": state}, str(path))
    return state


# -- JiT (unconditional) end-to-end -------------------------------------------


def test_jit_end_to_end(tmp_path):
    """JiT: export the after-ckpt, run the driver, write PNG + metrics JSON."""
    pipe, before_dir = _jit_native_dir(tmp_path)
    ckpt_state = _fake_ckpt(tmp_path / "after.ckpt", pipe.unet.unet, "unet.unet.")
    out = tmp_path / "run"

    rc = main(
        [
            "--before-dir", str(before_dir),
            "--after-ckpt", str(tmp_path / "after.ckpt"),
            "--output", str(out),
            "--num-samples", str(JIT_LATENT_SHAPE[0]),
            "--latent-shape", ",".join(str(s) for s in JIT_LATENT_SHAPE[1:]),
            "--num-inference-steps", "3",
            "--modality", "2",
            "--device", "cpu",
        ]
    )
    assert rc == 0

    # The after export is a loadable JiT native dir carrying the CKPT weights.
    after_dir = out / "after_native"
    after = LatentFlowPipeline.from_pretrained(str(after_dir))
    for key, baked in after.unet.unet.state_dict().items():
        assert torch.allclose(baked, ckpt_state[f"unet.unet.{key}"])

    # The before export on disk is NOT polluted by the in-place export bake.
    before = LatentFlowPipeline.from_pretrained(str(before_dir))
    for key, w in before.unet.unet.state_dict().items():
        assert torch.allclose(w, pipe.unet.unet.state_dict()[key])

    # The driver's artifacts: one valid slice-grid PNG + a well-formed JSON.
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["policy"] == "jit"
    assert metrics["num_samples"] == JIT_LATENT_SHAPE[0]
    assert len(metrics["grids"]) == JIT_LATENT_SHAPE[0]
    for grid in metrics["grids"]:
        assert _is_valid_png(out / grid)


def test_jit_bake_changes_after_pipeline(tmp_path):
    """The after side differs from the before side (the ckpt weights took)."""
    pipe, before_dir = _jit_native_dir(tmp_path)
    _fake_ckpt(tmp_path / "after.ckpt", pipe.unet.unet, "unet.unet.")
    out = tmp_path / "run"
    main(
        [
            "--before-dir", str(before_dir),
            "--after-ckpt", str(tmp_path / "after.ckpt"),
            "--output", str(out),
            "--num-samples", "1",
            "--latent-shape", "4,4,4,4",
            "--num-inference-steps", "3",
            "--device", "cpu",
        ]
    )
    after = LatentFlowPipeline.from_pretrained(str(out / "after_native"))
    before = LatentFlowPipeline.from_pretrained(str(before_dir))
    differing = [
        k
        for k in before.unet.unet.state_dict()
        if not torch.allclose(before.unet.unet.state_dict()[k], after.unet.unet.state_dict()[k])
    ]
    assert differing  # the +0.25 offset must reach the after UNet


# -- ControlNet (paired) end-to-end -------------------------------------------


def _tiny_pairs():
    """One injected validation pair: a src control signal + a real target."""
    src = torch.randn(*PAIRED_LATENT_SHAPE, generator=torch.Generator().manual_seed(10))
    tgt = torch.randn(*PAIRED_LATENT_SHAPE, generator=torch.Generator().manual_seed(20))
    return [
        {
            "src_latent": src[0],
            "tgt_latent": tgt[0],
            "spacing": [1.0, 1.0, 1.0],
            "src_label": 1,
            "tgt_label": 2,
        }
    ]


def test_controlnet_end_to_end(tmp_path):
    """ControlNet: export via the bridge (base passed through), run, score."""
    pipe, before_dir = _controlnet_native_dir(tmp_path)
    ckpt_state = _fake_ckpt(tmp_path / "after.ckpt", pipe.controlnet, "controlnet.")
    out = tmp_path / "run"

    rc = main(
        [
            "--before-dir", str(before_dir),
            "--after-ckpt", str(tmp_path / "after.ckpt"),
            "--output", str(out),
            "--num-inference-steps", "3",
            "--device", "cpu",
        ],
        pairs_provider=lambda args: _tiny_pairs(),
    )
    assert rc == 0

    # The after export is a ControlNet native dir (controlnet component declared
    # + subdir present) carrying the CKPT's controlnet weights.
    after_dir = out / "after_native"
    index = json.loads((after_dir / "model_index.json").read_text())
    assert "controlnet" in index["components"]
    after = ControlNetLatentFlowPipeline.from_pretrained(str(after_dir))
    for key, baked in after.controlnet.state_dict().items():
        assert torch.allclose(baked, ckpt_state[f"controlnet.{key}"])

    # The before export's base + ControlNet are NOT polluted by the bake.
    before = ControlNetLatentFlowPipeline.from_pretrained(str(before_dir))
    for key, w in before.controlnet.state_dict().items():
        assert torch.allclose(w, pipe.controlnet.state_dict()[key])

    # The driver's artifacts: grid PNGs + PSNR/SSIM against the real target.
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["policy"] == "controlnet"
    assert metrics["num_samples"] == 1
    for side in ("before", "after"):
        assert isinstance(metrics[side]["psnr"], float)
        assert isinstance(metrics[side]["ssim"], float)
    for grid in metrics["grids"]:
        assert _is_valid_png(out / grid)


def test_controlnet_after_base_is_before_base(tmp_path):
    """The after export passes the frozen base through verbatim (no re-bake).

    A GRPO ControlNet ckpt carries ONLY controlnet.* keys; the base UNet comes
    from the before export (the supervised stage's frozen JiT base), so the two
    pipelines' base weights must be identical.
    """
    pipe, before_dir = _controlnet_native_dir(tmp_path)
    _fake_ckpt(tmp_path / "after.ckpt", pipe.controlnet, "controlnet.")
    out = tmp_path / "run"
    main(
        [
            "--before-dir", str(before_dir),
            "--after-ckpt", str(tmp_path / "after.ckpt"),
            "--output", str(out),
            "--num-inference-steps", "3",
            "--device", "cpu",
        ],
        pairs_provider=lambda args: _tiny_pairs(),
    )
    after = ControlNetLatentFlowPipeline.from_pretrained(str(out / "after_native"))
    before = ControlNetLatentFlowPipeline.from_pretrained(str(before_dir))
    for key, w in before.unet.unet.state_dict().items():
        assert torch.allclose(w, after.unet.unet.state_dict()[key])


# -- fail-fast paths -----------------------------------------------------------


def test_controlnet_real_path_requires_data_args(tmp_path):
    """The real ControlNet path validates --data-base-dir/--latents-dir up front
    (not argparse-required, so the pairs_provider seam stays usable)."""
    pipe, before_dir = _controlnet_native_dir(tmp_path)
    _fake_ckpt(tmp_path / "after.ckpt", pipe.controlnet, "controlnet.")
    with pytest.raises(ValueError, match="data-base-dir"):
        main(
            [
                "--before-dir", str(before_dir),
                "--after-ckpt", str(tmp_path / "after.ckpt"),
                "--output", str(tmp_path / "run"),
                "--device", "cpu",
            ]
        )


def test_before_dir_must_be_a_native_export(tmp_path):
    """A dir without model_index.json fails fast with a clear pointer."""
    empty = tmp_path / "not_an_export"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="model_index.json"):
        main(
            [
                "--before-dir", str(empty),
                "--after-ckpt", str(tmp_path / "whatever.ckpt"),
                "--output", str(tmp_path / "run"),
            ]
        )


# -- the real ControlNet data path (manifest -> split -> warm cache) ------------


def test_controlnet_real_pairs_path(tmp_path, monkeypatch):
    """``_load_paired_val_pairs``' happy path: subject split -> warm cache ->
    scale-on-read, without the pairs_provider seam.

    ``build_brats_pair_manifest`` is patched to a hand-built manifest (the NIfTI
    scan is irrelevant here — the warmed disk cache means no volume is ever
    read), and the cache files are written in the shared on-disk format
    (``_save_cache``), keyed by the same path-derived ``sample_id`` /
    geometry-folded ``cache_tag`` the real warm produces.
    """
    from manifold.data.latent_dataset import _save_cache
    from manifold.data.paired_latent_dataset import paired_cache_tag

    pipe, before_dir = _controlnet_native_dir(tmp_path)
    _fake_ckpt(tmp_path / "after.ckpt", pipe.controlnet, "controlnet.")

    # Two subjects' t1n->t1c pairs; the deterministic by-subject split holds out
    # the LAST ceil(fraction * n) subjects -> subjB is the val subject.
    fake_paths = {
        "subjA": [f"/fake/subjA-{c}.nii.gz" for c in ("t1n", "t1c")],
        "subjB": [f"/fake/subjB-{c}.nii.gz" for c in ("t1n", "t1c")],
    }
    manifest = [
        {
            "src": paths[0],
            "tgt": paths[1],
            # Labels within the tiny fixture's num_class_embeds=4 (the real
            # DEFAULT_BRATS_LABELS 34/35 need the full-size class embedding).
            "src_label": 1,
            "tgt_label": 2,
        }
        for paths in fake_paths.values()
    ]
    monkeypatch.setattr(
        "manifold.data.paired_brats.build_brats_pair_manifest", lambda root: manifest
    )

    # A warmed disk cache: one unscaled latent per unique volume, keyed by the
    # path-derived sample_id under the geometry-folded tag (target 16³, divisor 2).
    cache_dir = tmp_path / "paired_cache"
    cache_dir.mkdir()  # warm_cache's mkdir equivalent — _save_cache assumes it
    target_dim = (16, 16, 16)
    tag = paired_cache_tag("paired_train", target_dim, 2)
    torch.manual_seed(0)
    for paths in fake_paths.values():
        for path in paths:
            digest = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
            sample_id = f"{os.path.basename(path)}__{digest}"
            _save_cache(
                str(cache_dir),
                sample_id,
                tag,
                {
                    "latent": torch.randn(4, 8, 8, 8),
                    "sample_id": sample_id,
                    "spacing": [1.0, 1.0, 1.0],
                },
            )

    out = tmp_path / "run"
    rc = main(
        [
            "--before-dir", str(before_dir),
            "--after-ckpt", str(tmp_path / "after.ckpt"),
            "--output", str(out),
            "--device", "cpu",
            "--num-inference-steps", "3",
            "--data-base-dir", str(tmp_path),
            "--val-fraction", "0.5",
            "--latents-dir", str(cache_dir),
            "--target-dim", "16,16,16",
        ]
    )
    assert rc == 0
    # Only the held-out subject's pair is evaluated (subjB; no train leakage).
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["policy"] == "controlnet"
    assert metrics["num_samples"] == 1
    for side in ("before", "after"):
        assert isinstance(metrics[side]["psnr"], float)
    assert _is_valid_png(out / metrics["grids"][0])
