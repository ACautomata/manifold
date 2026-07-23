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
    ControlNet3DConditionModel,
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
from manifold.config.builder import build_controlnet, build_unet

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
  p_mean: -0.8
  p_std: 0.8
  t_eps: 0.05
  l1_weight: 0.0
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


# -- ControlNet recipes + builder (issue #141 / ADR-0027) ---------------------


def test_controlnet_supervised_recipe_loads() -> None:
    """The ControlNet supervised recipe loads + carries the x0 Formulation + the L1 knob."""
    cfg = load_config(
        str(_REPOS / "configs/env/environment_brats2023.yaml"),
        str(_REPOS / "configs/train/config_controlnet_supervised.yaml"),
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    assert cfg.formulation.mode == "x0-denoiser"
    assert cfg.formulation.l1_weight == 0.0  # the L1+direction lever, default off
    assert cfg.diffusion_unet_train.lr == 1.0e-4
    assert cfg.controlnet.num_inference_steps == 15  # validation rollout resolution


def test_unified_grpo_recipe_loads_both_policy_paths() -> None:
    """The unified GRPO recipe (ADR-0034 / issue #180) loads + carries the GRPO train
    block for BOTH policy paths in one config — no mode-specific preset.

    The shared GRPO knobs (kl_coef / reward_bound / reward_temp / G / adv_clip_max)
    carry over unchanged. The ControlNet path is driven by the native artifact under
    --native-dir (ADR-0034, no preset), and its warm-start LR lives in the optional
    ``controlnet`` block (1.0e-7 vs the UNet's ``grpo_train.lr`` 1.0e-6). The
    ControlNet path's paired geometry (``diffusion_unet_inference.dim``) rides here too
    (harmless on the UNet path — ``_unet_real_inputs`` does not read it).
    """
    cfg = load_config(
        str(_REPOS / "configs/env/environment_brats2023.yaml"),
        str(_REPOS / "configs/train/config_grpo.yaml"),
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    # The shared GRPO knobs — carry over unchanged.
    assert cfg.grpo_train.G == 8
    assert cfg.grpo_train.adv_clip_max == 5.0
    assert cfg.grpo_train.kl_coef == 0.1
    assert cfg.grpo_train.reward_bound == "tanh"
    assert cfg.grpo_train.reward_temp == 8.0
    assert list(cfg.grpo_train.eta_step_list) == [0, 1, 2, 3, 4, 5, 6, 7]
    # The UNet-policy LR (the from-scratch default).
    assert cfg.grpo_train.lr == 1.0e-6
    assert cfg.checkpoint.save_top_k == 1
    # The ControlNet-path warm-start LR override (the optional ``controlnet`` block).
    assert cfg.controlnet.lr == 1.0e-7
    # The ControlNet path's paired geometry (_controlnet_real_inputs reads ``dim``).
    assert tuple(int(d) for d in cfg.diffusion_unet_inference.dim) == (256, 256, 128)


def test_build_controlnet_maps_diffusion_unet_block(tmp_path: Path) -> None:
    """build_controlnet maps the diffusion_unet block to ControlNet kwargs + forwards on CPU.

    Two keys differ from the base (ADR-0026): ``out_channels`` is base-only (the
    ControlNet has no output conv) and ``controlnet_cond_channels`` defaults to
    ``in_channels`` (the ``x_src`` latent width). The ControlNet forward emits the
    base's residual-injection args.
    """
    env = _write(tmp_path / "env.yaml", _BASE_ENV)
    train = _write(tmp_path / "train.yaml", _TRAIN)
    net = _write(tmp_path / "net.yaml", _TINY_NETWORK)
    cfg = load_config(env, train, net)
    cn = build_controlnet(cfg)
    assert type(cn) is ControlNet3DConditionModel
    assert cn.config["controlnet_cond_channels"] == cfg.diffusion_unet.in_channels == 4
    # out_channels is NOT a ControlNet config key (it was popped — base-only).
    assert "out_channels" not in cn.config

    z = torch.randn(1, 4, 4, 4, 4)
    down_res, mid_res = cn(
        sample=z,
        controlnet_cond=torch.randn_like(z),
        timestep=torch.tensor(0.5),
        spacing=torch.tensor([1.0, 1.0, 1.0]),
        class_labels_src=torch.tensor([1]),
        class_labels_tgt=torch.tensor([2]),
    )
    assert mid_res is not None
    assert len(down_res) > 0


def test_build_unet_pops_controlnet_only_paired_direction_offset() -> None:
    """Regression (codex #142): the shipped ``config_network.yaml`` carries
    ``paired_direction_offset`` (the ControlNet's direction-MLP knob), but the base
    UNet wrapper no longer accepts it. ``build_unet`` must pop it so the shared
    block builds the base without a TypeError; ``build_controlnet`` forwards it."""
    from manifold.config import merge_overrides

    cfg = load_config(
        str(_REPOS / "configs/env/environment_brats2023.yaml"),
        str(_REPOS / "configs/train/config_rflow_jit.yaml"),
        str(_REPOS / "configs/network/config_network.yaml"),
    )
    # The shipped block has the ControlNet-only knob; GPU-only flash attention is
    # disabled so the wrapper constructs on CPU.
    assert cfg.diffusion_unet.paired_direction_offset == 0
    cfg = merge_overrides(cfg, {}, ["diffusion_unet.use_flash_attention=false"])

    unet = build_unet(cfg)  # would TypeError on paired_direction_offset before the fix
    assert type(unet) is UNet3DConditionModel
    assert "paired_direction_offset" not in unet.config
    # build_controlnet consumes the same block (forwards the knob) without popping it.
    assert build_controlnet(cfg).config["paired_direction_offset"] == 0


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


def test_no_mode_specific_grpo_preset_remains() -> None:
    """Regression (issue #180 / ADR-0034): the two GRPO mode-specific presets are merged
    into one ``config_grpo.yaml``. A mode-split preset (``config_controlnet_grpo.yaml``)
    must not reappear — the ControlNet path is driven by the native artifact under
    ``--native-dir`` + the optional ``controlnet`` block, not a separate preset. Fails
    loudly if the mode-specific YAML is tracked again.
    """
    import subprocess

    out = subprocess.run(
        ["git", "ls-files", "configs/train/"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPOS,
    ).stdout
    tracked = {Path(p) for p in out.split() if p}
    assert Path("configs/train/config_controlnet_grpo.yaml") not in tracked, (
        f"config_controlnet_grpo.yaml is still tracked — the GRPO mode-split preset was "
        f"not merged (issue #180). Tracked train configs: {sorted(p.name for p in tracked)}"
    )
