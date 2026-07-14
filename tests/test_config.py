"""OmegaConf experiment-config layer tests (issue #15, ADR-0004).

Covers the mechanics the rest of the codebase depends on — ``???`` required
paths raise on read, ``null`` optionals read as ``None`` via :func:`opt`,
:func:`require_paths` fails fast, ``--<key>`` flags + Hydra dotlist overrides
merge with the right precedence, ``_base_`` deep-merges, and the three-file
merge replaces top-level blocks wholesale — and the construction seam: a
composed config builds a live :class:`LatentFlowPipeline` (UNet + VAE +
Scheduler) and ``__call__`` produces a finite decoded volume.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowPipeline,
    UNet3DConditionModel,
)
from manifold.config import (
    autoencoder_divisor,
    build_pipeline,
    build_vae,
    load_config,
    merge_overrides,
    opt,
    require_paths,
)
from manifold.config.builder import build_unet

# A tiny CPU-runnable network config (mirrors the conftest fixtures: 2-level VAE
# divisor 2 → latent [1,4,4,4,4] decodes to image [1,1,8,8,8]). The shipped real
# network config (the shipped GPU architecture) is too large to forward on CPU, so the
# build+call seam is exercised through this tiny block.
_TINY_NETWORK = """\
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
  scaling_factor: 0.5
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

_BASE_ENV = """\
data_base_dir: ???
model_dir: ???
model_filename: foo.pt
trained_autoencoder_path: ???
json_data_list: null
dataset_type: fixed
"""

_TRAIN = """\
diffusion_unet_train:
  batch_size: 1
formulation:
  mode: x0-denoiser
  cfg_interval: [0.1, 1.0]
diffusion_unet_inference:
  num_inference_steps: 15
  modality: 1
  cfg_guidance_scale: 1.5
"""

_REPOS = Path(__file__).resolve().parent.parent


def _write(path: Path, text: str) -> str:
    path.write_text(text)
    return str(path)


# -- path semantics ---------------------------------------------------------


def test_required_missing_path_raises_on_direct_read() -> None:
    cfg = OmegaConf.create("data_base_dir: ???\nmodel_dir: /x\n")
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.data_base_dir


def test_optional_null_reads_as_none() -> None:
    cfg = OmegaConf.create("json_data_list: null\nexisting_ckpt_filepath: null\n")
    assert getattr(cfg, "json_data_list", "sentinel") is None
    assert opt(cfg, "existing_ckpt_filepath") is None
    assert opt(cfg, "never_set", "d") == "d"  # absent key -> default


def test_require_paths_raises_when_required_unset() -> None:
    cfg = OmegaConf.create(
        "data_base_dir: ???\nmodel_dir: ???\nmodel_filename: foo.pt\n"
        "trained_autoencoder_path: ???\n"
    )
    with pytest.raises(ValueError, match="data_base_dir"):
        require_paths(cfg)


def test_require_paths_passes_when_all_set() -> None:
    cfg = OmegaConf.create(
        "data_base_dir: /d\nmodel_dir: /m\nmodel_filename: foo.pt\n"
        "trained_autoencoder_path: /a.pt\n"
    )
    require_paths(cfg)  # no raise


# -- overrides --------------------------------------------------------------


def test_dotlist_overrides_base_and_flags() -> None:
    base = OmegaConf.create(
        "model_dir: /base\ndata_base_dir: /data\ndiffusion_unet_train:\n  lr: 0.1\n"
    )
    # precedence: base < flags < dotlist (dotlist wins)
    merged = merge_overrides(base, {"model_dir": "/from_flag"}, ["model_dir=/from_dotlist"])
    assert merged.model_dir == "/from_dotlist"
    # dotlist creates nested keys
    merged2 = merge_overrides(base, {}, ["diffusion_unet_train.lr=1.0e-4"])
    assert merged2.diffusion_unet_train.lr == 1.0e-4


# -- _base_ composition + top-level wholesale replace ------------------------


def test_base_composition_deep_merges(tmp_path: Path) -> None:
    base_p = _write(
        tmp_path / "base.yaml",
        "spatial_dims: 3\nlatent_channels: 4\nautoencoder:\n  a: 1\n  b: 2\n",
    )
    variant_p = _write(
        tmp_path / "variant.yaml",
        f"_base_: [{Path(base_p).name}]\nautoencoder:\n  b: 99\n  c: 3\n",
    )
    cfg = load_config(variant_p, None, variant_p)  # same file for env + network
    assert cfg.spatial_dims == 3
    assert cfg.autoencoder.a == 1  # inherited from base
    assert cfg.autoencoder.b == 99  # variant overrides
    assert cfg.autoencoder.c == 3  # variant-only
    assert "_base_" not in cfg  # stripped


def test_bundle_merge_replaces_top_level_blocks_wholesale(tmp_path: Path) -> None:
    """The three files merge by top-level REPLACEMENT, not deep-merge.

    A later file's ``diffusion_unet`` block replaces the earlier one outright;
    ``OmegaConf.merge`` (which recurses) would silently keep stale sub-keys.
    """
    from manifold.config.loader import _merge_top_level

    env = OmegaConf.create("autoencoder:\n  channels: 32\n  levels: 4\n  dropout: 0.1\n")
    net = OmegaConf.create("autoencoder:\n  channels: 64\n")  # PARTIAL block
    merged = _merge_top_level(env, OmegaConf.create(), net)
    assert merged.autoencoder.channels == 64  # later wins
    # Earlier sub-keys are GONE (wholesale replacement), not deep-merged in.
    assert OmegaConf.select(merged, "autoencoder.levels", default="ABSENT") == "ABSENT"
    assert OmegaConf.select(merged, "autoencoder.dropout", default="ABSENT") == "ABSENT"


# -- construction seam ------------------------------------------------------


def test_compose_builds_live_pipeline(tmp_path: Path) -> None:
    """Composed tiny configs build a live pipeline of the correct classes."""
    env = _write(tmp_path / "env.yaml", _BASE_ENV)
    train = _write(tmp_path / "train.yaml", _TRAIN)
    net = _write(tmp_path / "net.yaml", _TINY_NETWORK)
    cfg = load_config(env, train, net)
    # ${image_channels}/${latent_channels} interpolated against the root.
    assert cfg.autoencoder.in_channels == 1
    assert cfg.diffusion_unet.in_channels == 4

    pipe = build_pipeline(cfg)
    assert type(pipe) is LatentFlowPipeline
    assert type(pipe.unet) is UNet3DConditionModel
    assert type(pipe.vae) is AutoencoderKL
    assert type(pipe.scheduler) is FlowMatchHeunDiscreteScheduler

    vol = pipe(
        [1, 4, 4, 4, 4],
        spacing=[1.0, 1.0, 1.0],
        modality=2,
        num_inference_steps=3,
        generator=torch.Generator().manual_seed(0),
    )
    assert vol.shape == (1, 1, 8, 8, 8)
    assert torch.isfinite(vol).all()


def test_compose_then_dotlist_override_retargets(tmp_path: Path) -> None:
    """A dotlist override changes a constructed value (dotlist wins)."""
    env = _write(tmp_path / "env.yaml", _BASE_ENV)
    train = _write(tmp_path / "train.yaml", _TRAIN)
    net = _write(tmp_path / "net.yaml", _TINY_NETWORK)
    cfg = load_config(env, train, net)
    assert cfg.diffusion_unet.num_class_embeds == 4
    cfg = merge_overrides(cfg, {}, ["diffusion_unet.num_class_embeds=8"])
    assert cfg.diffusion_unet.num_class_embeds == 8


def test_unet_wrapper_accepts_widened_architectural_knobs(tmp_path: Path) -> None:
    """The widened UNet construction surface builds + forwards on CPU.

    Scalar ``num_res_blocks`` (broadcast), per-level ``num_head_channels``, and
    the architectural knobs ``resblock_updown`` / ``include_fc`` a JiT checkpoint
    needs for ``strict=True`` load are accepted and round-trip through config.
    ``use_flash_attention`` (a GPU runtime knob) is accepted by the wrapper — it
    reaches MAISI's CUDA check rather than a wrapper ``TypeError``.
    """
    env = _write(tmp_path / "env.yaml", _BASE_ENV)
    train = _write(tmp_path / "train.yaml", _TRAIN)
    net = _write(
        tmp_path / "net.yaml",
        _TINY_NETWORK.replace(
            "  num_res_blocks: 1\n  norm_num_groups: 8\n  num_head_channels: [4, 4]\n",
            "  num_res_blocks: 1\n  norm_num_groups: 8\n  num_head_channels: [0, 4]\n"
            "  resblock_updown: true\n  include_fc: true\n",
        ),
    )
    cfg = load_config(env, train, net)
    unet = build_unet(cfg)
    assert type(unet) is UNet3DConditionModel
    assert unet.config["num_res_blocks"] == 1  # scalar accepted
    assert list(unet.config["num_head_channels"]) == [0, 4]  # per-level accepted
    assert unet.config["resblock_updown"] is True
    assert unet.config["include_fc"] is True
    out = unet(
        torch.randn(1, 4, 4, 4, 4),
        0.5,
        spacing=torch.tensor([1.0, 1.0, 1.0]),
        class_labels=torch.tensor([2]),
    )
    assert out.shape == (1, 4, 4, 4, 4)

    # use_flash_attention=True is accepted by the wrapper (reaches MAISI's CUDA
    # gate, not a wrapper TypeError) — a GPU checkpoint's runtime knob flows through.
    with pytest.raises(ValueError, match="Flash attention"):
        UNet3DConditionModel(
            num_channels=(8, 8),
            num_res_blocks=1,
            norm_num_groups=8,
            num_head_channels=[4, 4],
            use_flash_attention=True,
            num_class_embeds=4,
        )


# -- migrated (shipped) real configs ----------------------------------------


def test_real_network_config_builds_vae_and_divisor() -> None:
    """The shipped network config (the BraTS GPU architecture) builds the VAE on CPU
    and reports the BraTS latent divisor (2**(3-1) == 4)."""
    cfg = load_config(
        str(_REPOS / "configs/env/environment_brats2023.yaml"),
        str(_REPOS / "configs/train/config_rflow_jit.yaml"),
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    # The shipped BraTS GPU architecture values.
    assert list(cfg.autoencoder.num_channels) == [64, 128, 256]
    assert list(cfg.diffusion_unet.num_channels) == [64, 128, 256, 512]
    assert cfg.autoencoder.norm_float16 is True
    assert cfg.diffusion_unet.resblock_updown is True
    assert cfg.diffusion_unet.include_fc is True
    assert cfg.diffusion_unet.num_res_blocks == 2  # scalar form
    assert autoencoder_divisor(cfg) == 4

    vae = build_vae(cfg)  # CPU-constructible (no forward — GPU knobs)
    assert type(vae) is AutoencoderKL
    assert vae.config["norm_float16"] is True
    assert vae.config["dim_split"] == 1


def test_gauss_env_passes_require_paths_base_env_raises() -> None:
    """The gauss BraTS profile is runnable (no ??? required paths); the base
    template is not (required paths are ??? — fail fast)."""
    base = load_config(
        str(_REPOS / "configs/env/environment.yaml"),
        None,
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    with pytest.raises(ValueError, match="data_base_dir"):
        require_paths(base)

    gauss = load_config(
        str(_REPOS / "configs/env/environment_brats2023.yaml"),
        str(_REPOS / "configs/train/config_rflow_jit.yaml"),
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    require_paths(gauss)  # concrete paths set — no raise


def test_jit_recipe_carries_x0_formulation() -> None:
    """The migrated JiT train recipe selects the x0-denoiser Formulation."""
    cfg = load_config(
        str(_REPOS / "configs/env/environment_brats2023.yaml"),
        str(_REPOS / "configs/train/config_rflow_jit.yaml"),
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    assert cfg.formulation.mode == "x0-denoiser"
    assert list(cfg.formulation.cfg_interval) == [0.1, 1.0]
    # The numerical-validation defaults (issue #18): 15 Heun steps, cfg 1.5.
    assert cfg.diffusion_unet_inference.num_inference_steps == 15
    assert cfg.diffusion_unet_inference.cfg_guidance_scale == 1.5


def test_paired_recipe_warmup_is_ratio_not_absolute() -> None:
    """Regression (autoresearch best-experiment finding): the paired-JiT recipe's
    default ``lr_warmup_steps: 1000`` exceeded the ~500-step horizon of a 20-ep
    8-DDU run, so ``cosine_with_warmup`` never left its linear ramp and the run
    trained on a monotonic 0->peak ramp - ending at peak LR (worst convergence
    point). Setting it to 50 gave +1 dB. The recipe now defaults to
    ``lr_warmup_ratio: 0.1`` (0.1 x 500 = 50 there), which tracks the horizon at
    any run length. This fails if the ratio is removed (the absolute 1000 returns
    and again exceeds a short horizon).
    """
    from manifold.modules.latent_flow import resolve_warmup_steps

    cfg = load_config(
        str(_REPOS / "configs/env/environment_brats2023.yaml"),
        str(_REPOS / "configs/train/config_paired_jit.yaml"),
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    train = cfg.diffusion_unet_train
    assert float(train.lr_warmup_ratio) == pytest.approx(0.1)
    # The 20-ep/8-DDU horizon (~500 steps): resolved warmup must be 50, NOT the
    # old absolute 1000 that exceeds the horizon -> a pure linear ramp.
    resolved = resolve_warmup_steps(
        int(train.lr_warmup_steps), train.lr_warmup_ratio, total_steps=500
    )
    assert resolved == 50
    assert resolved < 500


def test_env_configs_are_tracked_in_repo() -> None:
    """Regression: a bare ``env/`` gitignore pattern shadowed ``configs/env/``, so
    the migrated env profiles were present in the working tree but UNTRACKED — a
    fresh clone had no ``configs/env/environment.yaml``. This fails loudly if the
    YAMLs disappear from the tracked tree again (the pattern is now root-anchored).
    """
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "configs/env/"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPOS,
    ).stdout
    tracked = {Path(p).name for p in out.split() if p}
    assert "environment.yaml" in tracked, (
        f"configs/env/environment.yaml is NOT git-tracked (gitignore shadow?). "
        f"Tracked: {sorted(tracked)}"
    )
    assert "environment_brats2023.yaml" in tracked
