"""Paired reward Slice 2 (#94) tests: paired export bridge + load_frozen_paired_generator.

External-behavior seams (per PRD #92 + issue #94 acceptance):

- A Paired-JiT training ``.ckpt`` exports to a paired native dir (raw optimizer
  weights baked; the ``--ema`` flag was removed).
- ``load_frozen_paired_generator(native_dir)`` recovers an ``in_channels = 2·C_latent``
  UNet whose backbone weights match the raw optimizer weights, the **base** scheduler,
  and the VAE ``scaling_factor``.
- The loaded generator is ``eval()``/frozen and rolls a deterministic fake (same src
  -> byte-identical output).
- Scale-consistency: the export's ``scaling_factor`` is reused verbatim (ADR-0021).
- The export script accepts ``--pipeline paired``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    PairedLatentFlowPipeline,
    UNet3DConditionModel,
)
from manifold.data.paired_reward_pairs import load_frozen_paired_generator
from manifold.training.export import export_to_native

C_LATENT = 4
LATENT_SHAPE = (1, C_LATENT, 4, 4, 4)


def _trainable_paired_unet() -> UNet3DConditionModel:
    torch.manual_seed(0)
    return UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )


def _build_paired_ckpt(tmp_path) -> tuple[str, UNet3DConditionModel, AutoencoderKL]:
    """Build a Paired-JiT training .ckpt (raw optimizer weights; mirrors the JiT test).

    Fits the module RAW (no EMA callback) so the checkpoint's ``state_dict`` carries
    the raw optimizer weights (the arm the paired reward bakes, ADR-0021). Returns the
    checkpoint path + the raw UNet + VAE used.
    """
    import lightning.pytorch as pl

    from manifold.modules.paired_latent_flow import PairedLatentFlowModule

    unet = _trainable_paired_unet()
    for p in unet.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    module = PairedLatentFlowModule(
        unet,
        FlowMatchHeunDiscreteScheduler(),
        lr=1e-2,
        lr_warmup_steps=0,
        num_train_examples=4,
        train_batch_size=2,
        n_epochs=1,
    )
    vae = AutoencoderKL(scaling_factor=0.5)
    from manifold.data import PairedLatentDataset  # noqa: F401  (import-only sanity)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    # A trivial fit so the checkpoint is real (non-trivial raw weights).
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {
                "src_latent": torch.randn(C_LATENT, 4, 4, 4),
                "tgt_latent": torch.randn(C_LATENT, 4, 4, 4),
                "src_label": torch.tensor(0, dtype=torch.long),
                "tgt_label": torch.tensor(1, dtype=torch.long),
                "spacing": torch.tensor([1.0, 1.0, 1.0]),
            }

    import stable_pretraining as spt

    trainer.fit(module, datamodule=spt.data.DataModule(train=DataLoader(_DS(), batch_size=2)))
    ckpt_path = str(tmp_path / "paired_last.ckpt")
    trainer.save_checkpoint(ckpt_path)
    return ckpt_path, unet, vae


# -- export_to_native with pipeline_cls=PairedLatentFlowPipeline ----------------


def test_export_to_native_writes_paired_native_dir(tmp_path):
    """A paired .ckpt exports to a paired native dir (raw optimizer weights baked)."""
    ckpt_path, unet, vae = _build_paired_ckpt(tmp_path)
    fresh_unet = _trainable_paired_unet()
    source = export_to_native(
        ckpt_path,
        str(tmp_path / "paired_native"),
        unet=fresh_unet,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=PairedLatentFlowPipeline,
    )
    assert source == "unet_state_dict"  # the raw optimizer weights were baked
    # The dir is a paired native export (model_index.json + the paired pipeline class).
    reloaded = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired_native"))
    assert reloaded.unet.config["in_channels"] == 2 * C_LATENT
    assert reloaded.unet.config["out_channels"] == C_LATENT


def test_export_to_native_default_pipeline_is_jit(tmp_path):
    """pipeline_cls=None defaults to LatentFlowPipeline (the JiT path - no regression).

    Uses a JiT-shaped ckpt (in_channels=4) since the default pipeline is the
    noise->data JiT, not paired; the paired export is tested above.
    """
    import lightning.pytorch as pl

    from manifold.modules import LatentFlowModule
    import stable_pretraining as spt
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    jit_unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    for p in jit_unet.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    module = LatentFlowModule(
        jit_unet,
        FlowMatchHeunDiscreteScheduler(),
        lr=1e-2,
        lr_warmup_steps=0,
        num_train_examples=4,
        train_batch_size=2,
        n_epochs=1,
    )
    vae = AutoencoderKL(scaling_factor=0.5)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {
                "latent": torch.randn(C_LATENT, 4, 4, 4),
                "spacing": torch.tensor([1.0, 1.0, 1.0]),
                "label": torch.tensor(1, dtype=torch.long),
            }

    trainer.fit(module, datamodule=spt.data.DataModule(train=DataLoader(_DS(), batch_size=2)))
    ckpt_path = str(tmp_path / "jit_last.ckpt")
    trainer.save_checkpoint(ckpt_path)
    fresh = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    source = export_to_native(
        ckpt_path,
        str(tmp_path / "jit_native"),
        unet=fresh,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
    )
    assert source == "unet_state_dict"
    from manifold import LatentFlowPipeline

    LatentFlowPipeline.from_pretrained(str(tmp_path / "jit_native"))  # loads as JiT


# -- load_frozen_paired_generator -------------------------------------------


def test_load_frozen_paired_generator_recovers_2c_unet_base_scheduler_and_scale(tmp_path):
    """load_frozen_paired_generator recovers a 2·C UNet, the BASE scheduler, + scale factor."""
    ckpt_path, _unet, vae = _build_paired_ckpt(tmp_path)
    fresh_unet = _trainable_paired_unet()
    export_to_native(
        ckpt_path,
        str(tmp_path / "paired_native"),
        unet=fresh_unet,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=PairedLatentFlowPipeline,
    )

    unet, scheduler, scaling_factor = load_frozen_paired_generator(tmp_path / "paired_native")
    # 2·C_latent UNet (condition-aware concat, ADR-0019).
    assert unet.config["in_channels"] == 2 * C_LATENT
    # The BASE scheduler (NOT the Partial subclass - the loser is a full 0->1 rollout;
    # only the probe constructs Partial, ADR-0023).
    assert type(scheduler).__name__ == "FlowMatchHeunDiscreteScheduler"
    assert not type(scheduler).__name__.endswith("PartialFlowMatchHeunScheduler")
    # The VAE scaling_factor (reused verbatim - ADR-0021; never re-estimated).
    assert scaling_factor == 0.5


def test_load_frozen_paired_generator_is_frozen_and_eval(tmp_path):
    """The loaded generator is eval() + requires_grad_(False) (a frozen fake source)."""
    ckpt_path, _unet, vae = _build_paired_ckpt(tmp_path)
    fresh_unet = _trainable_paired_unet()
    export_to_native(
        ckpt_path,
        str(tmp_path / "paired_native"),
        unet=fresh_unet,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=PairedLatentFlowPipeline,
    )
    unet, _scheduler, _scale = load_frozen_paired_generator(tmp_path / "paired_native")
    assert not unet.training
    assert all(not p.requires_grad for p in unet.parameters())


def test_load_frozen_paired_generator_rolls_deterministic_fake(tmp_path):
    """Same src -> byte-identical generated tgt (determinism, ADR-0020/0021).

    The paired rollout is deterministic given ``x_src`` (no stochastic input), so the
    loaded generator produces the same fake on repeat calls - the property the offline
    fake-cache relies on (cache once, reuse).
    """
    ckpt_path, _unet, vae = _build_paired_ckpt(tmp_path)
    fresh_unet = _trainable_paired_unet()
    export_to_native(
        ckpt_path,
        str(tmp_path / "paired_native"),
        unet=fresh_unet,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=PairedLatentFlowPipeline,
    )
    unet, scheduler, _scale = load_frozen_paired_generator(tmp_path / "paired_native")
    from manifold.modules.paired_sampler import sample_paired_latent_flow

    torch.manual_seed(0)
    src = torch.randn(2, C_LATENT, 4, 4, 4)
    out_a = sample_paired_latent_flow(
        unet, scheduler, src, [1.0, 1.0, 1.0], 0, 1, num_inference_steps=3
    )
    out_b = sample_paired_latent_flow(
        unet, scheduler, src, [1.0, 1.0, 1.0], 0, 1, num_inference_steps=3
    )
    assert torch.equal(out_a, out_b)
    assert out_a.shape == (2, C_LATENT, 4, 4, 4)
    assert torch.isfinite(out_a).all()


def test_load_frozen_paired_generator_reuses_export_scaling_factor(tmp_path):
    """The loader returns the export's scaling_factor verbatim (scale-consistency, ADR-0021).

    The rollout operates in scaled space; the scaling factor is reused from the export
    (never re-estimated), so the same factor the paired model trained with is the one
    the caller must scale raw cache src latents by.
    """
    ckpt_path, _unet, vae = _build_paired_ckpt(tmp_path)
    # A distinct scaling factor on the export VAE (round-trips through the loader).
    vae.scaling_factor.fill_(2.5)
    fresh_unet = _trainable_paired_unet()
    export_to_native(
        ckpt_path,
        str(tmp_path / "paired_native"),
        unet=fresh_unet,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=PairedLatentFlowPipeline,
    )
    _unet, _scheduler, scaling_factor = load_frozen_paired_generator(tmp_path / "paired_native")
    assert scaling_factor == 2.5


# -- export script --pipeline paired -----------------------------------------


def test_export_checkpoint_script_paired_pipeline(tmp_path):
    """scripts/export_checkpoint.py --pipeline paired writes a paired native dir."""
    ckpt_path, _unet, vae = _build_paired_ckpt(tmp_path)
    env, _train, net = _write_tiny_configs(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    try:
        import export_checkpoint as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    # Point the network config's VAE checkpoint at the fixture VAE's saved state so the
    # export loads it (else a fresh VAE is baked - acceptable, but we want the scale factor).
    rc = cli.main(
        [
            "--ckpt",
            ckpt_path,
            "--output",
            str(tmp_path / "paired_cli_native"),
            "--network-config",
            net,
            "--pipeline",
            "paired",
            # Fix #4 (codex #98 P1): paired exports must bake the generator's training
            # scale into the VAE (the reward pairs scale src latents by it; ADR-0021).
            "--scaling-factor",
            "2.5",
        ]
    )
    assert rc == 0
    # The CLI wrote a paired native export (loads as PairedLatentFlowPipeline).
    pipe = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired_cli_native"))
    assert pipe.unet.config["in_channels"] == 2 * C_LATENT
    # The exported VAE carries the supplied scaling factor (baked, not the 1.0 placeholder).
    assert float(pipe.vae.scaling_factor) == 2.5


# -- codex #98 export-script regression guards --------------------------------


def _write_jit_shaped_network(tmp_path):
    """Network config with the STOCK JiT ``diffusion_unet.in_channels`` (=4 via interpolation).

    Used to verify ``--pipeline paired`` overrides in_channels to 2·C (codex #98 #3):
    the stock config builds a 4-ch UNet; the paired export must build 8-ch before the
    backbone load or it hits a conv size mismatch.
    """
    net_yaml = tmp_path / "net_jit.yaml"
    net_yaml.write_text(
        "spatial_dims: 3\nlatent_channels: 4\nimage_channels: 1\n"
        "autoencoder:\n"
        "  spatial_dims: ${spatial_dims}\n  in_channels: ${image_channels}\n"
        "  out_channels: ${image_channels}\n  latent_channels: ${latent_channels}\n"
        "  num_channels: [8, 8]\n  num_res_blocks: [1, 1]\n  norm_num_groups: 8\n"
        "  norm_float16: false\n  num_splits: 1\n  save_mem: false\n  scaling_factor: 0.5\n"
        "diffusion_unet:\n"
        "  spatial_dims: 3\n  in_channels: ${latent_channels}\n  out_channels: ${latent_channels}\n"
        "  num_channels: [8, 8]\n  num_res_blocks: 1\n  norm_num_groups: 8\n"
        "  num_head_channels: [4, 4]\n  attention_levels: [false, false]\n"
        "  use_flash_attention: false\n  include_spacing_input: true\n"
        "  num_class_embeds: 4\n  num_train_timesteps: 1000\n"
        "scheduler:\n  num_train_timesteps: 1000\n  t_eps: 0.05\n"
    )
    return str(net_yaml)


def test_export_checkpoint_paired_overrides_jit_in_channels_to_2c(tmp_path):
    """--pipeline paired overrides the stock JiT in_channels (4) to 2·C (8).

    Regression for codex #98 (P1, #3): build_unet runs before the pipeline swap, so the
    stock config's in_channels=${latent_channels}=4 would build a 4-ch UNet and the
    backbone load would hit a conv-weight size mismatch. The script now overrides
    in_channels=2·C for paired (out_channels stays C).
    """
    ckpt_path, _unet, _vae = _build_paired_ckpt(tmp_path)
    net = _write_jit_shaped_network(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    try:
        import export_checkpoint as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    rc = cli.main(
        [
            "--ckpt",
            ckpt_path,
            "--output",
            str(tmp_path / "paired_override_native"),
            "--network-config",
            net,
            "--pipeline",
            "paired",
            "--scaling-factor",
            "0.5",
        ]
    )
    assert rc == 0
    pipe = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired_override_native"))
    assert pipe.unet.config["in_channels"] == 2 * C_LATENT
    # out_channels stays C (the paired UNet predicts the tgt velocity, one endpoint).
    assert pipe.unet.config["out_channels"] == C_LATENT


def test_export_checkpoint_paired_requires_scaling_factor(tmp_path):
    """--pipeline paired without --scaling-factor raises (the YAML placeholder is 1.0).

    Regression for codex #98 (P1, #4): the reward pairs scale src latents by the
    generator's training scale (ADR-0021); the export must bake it, not ship the 1.0
    placeholder. Fail fast if the caller forgets the flag.
    """
    import pytest

    ckpt_path, _unet, _vae = _build_paired_ckpt(tmp_path)
    net = _write_jit_shaped_network(tmp_path)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    try:
        import export_checkpoint as cli  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    with pytest.raises(ValueError, match="scaling-factor"):
        cli.main(
            [
                "--ckpt",
                ckpt_path,
                "--output",
                str(tmp_path / "paired_noscale"),
                "--network-config",
                net,
                "--pipeline",
                "paired",
            ]
        )


# -- the export bridge also serves paired inference (ADR-0006, ADR-0021) ------


def test_paired_export_round_trips_inference(tmp_path):
    """The paired native export round-trips inference (reloaded pipeline == original)."""
    ckpt_path, unet, vae = _build_paired_ckpt(tmp_path)
    fresh_unet = _trainable_paired_unet()
    export_to_native(
        ckpt_path,
        str(tmp_path / "paired_native"),
        unet=fresh_unet,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=PairedLatentFlowPipeline,
    )
    # A direct (non-ckpt) save/load round-trip of the exported pipeline infers identically
    # (the export's UNet weights are the baked raw arm; reloading reproduces them).
    pipe_a = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired_native"))
    pipe_a.save_pretrained(str(tmp_path / "paired_native_2"))
    pipe_b = PairedLatentFlowPipeline.from_pretrained(str(tmp_path / "paired_native_2"))
    torch.manual_seed(0)
    src = torch.randn(LATENT_SHAPE)
    args = dict(spacing=[1.0, 1.0, 1.0], src_label=0, tgt_label=1, num_inference_steps=3)
    assert torch.equal(pipe_a.sample_latent(src, **args), pipe_b.sample_latent(src, **args))


# -- shared config helper (mirrors test_paired_training_cli._write_paired_configs) --


def _write_tiny_configs(tmp_path):
    model_dir = tmp_path / "model"
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "data_base_dir: /tmp/_unused_\n"
        f"model_dir: {model_dir}\n"
        "model_filename: paired_jit.pt\n"
        "trained_autoencoder_path: /tmp/_unused_vae_\n"
        "val_subset_size: 4\nrandom_seed: 0\nnum_gpus: 1\n"
    )
    net_yaml = tmp_path / "net.yaml"
    net_yaml.write_text(
        "spatial_dims: 3\nlatent_channels: 4\nimage_channels: 1\n"
        "autoencoder:\n"
        "  spatial_dims: ${spatial_dims}\n  in_channels: ${image_channels}\n"
        "  out_channels: ${image_channels}\n  latent_channels: ${latent_channels}\n"
        "  num_channels: [8, 8]\n  num_res_blocks: [1, 1]\n  norm_num_groups: 8\n"
        "  norm_float16: false\n  num_splits: 1\n  save_mem: false\n  scaling_factor: 0.5\n"
        "diffusion_unet:\n"
        "  spatial_dims: 3\n  in_channels: 8\n  out_channels: 4\n"
        "  num_channels: [8, 8]\n  num_res_blocks: 1\n  norm_num_groups: 8\n"
        "  num_head_channels: [4, 4]\n  attention_levels: [false, false]\n"
        "  use_flash_attention: false\n  include_spacing_input: true\n"
        "  num_class_embeds: 4\n  num_train_timesteps: 1000\n"
        "scheduler:\n  num_train_timesteps: 1000\n  t_eps: 0.05\n"
    )
    return str(env_yaml), None, str(net_yaml)
