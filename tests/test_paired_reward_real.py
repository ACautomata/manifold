"""Paired reward Slice 3 (#95) tests: the real fake-cache builder + 2-way split.

External-behavior seams (per PRD #92 + issue #95 acceptance):

- ``build_paired_reward_inputs`` builds train/val/probe condition-aware pair
  datasets over a fake warmed paired latent cache (the offline fake-cache builder,
  ADR-0020), with the 2-way split enforced by the caller (ADR-0022).
- The real ``_real_inputs`` CLI path wires the generator (#94) + the paired cache +
  the 2-way split -> ``run_paired_reward_training`` (smoke; the real BraTS+VAE path
  is cluster-only, so the seam is a fake dataset).
- Scale-consistency: the export's ``scaling_factor`` is threaded through
  (dataset.scaling_factor = export factor); an unscaled src would be caught
  downstream (the builder asserts nothing - it trusts the caller, but the wiring
  sets the factor verbatim, ADR-0021).
- DDP monitor fallback + resume carry through (Slice 1 wires; exercised via the
  build_paired_reward_inputs seam).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

from manifold import FlowMatchHeunDiscreteScheduler, PartialFlowMatchHeunScheduler, RewardModel
from manifold.data.paired_reward_pairs import build_paired_reward_inputs
from manifold.modules import PairedRewardModule

C_LATENT = 4
_LAT = (C_LATENT, 8, 8, 8)


class _IdentityPairedGen(nn.Module):
    """Identity paired generator: ``x0 = z`` (the z half) - copy-src fake.

    Batch-size-agnostic (no fixed target) + deterministic (ADR-0020).
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(0))

    def forward(
        self, sample, timestep, spacing, class_labels_src=None, class_labels_tgt=None, **kw
    ):
        return sample[:, :C_LATENT]


class _FakePairedLatentDS(Dataset):
    """A warmed ``PairedLatentDataset`` stand-in: emits the 5-key paired contract.

    Mirrors :meth:`PairedLatentDataset.__getitem__` exactly - scaled src/tgt
    latents, contrast labels, spacing. The latents are pre-scaled (the caller sets
    ``scaling_factor`` = the export's, ADR-0021); here they are unit-scaled for the
    smoke. Carries ``scaling_factor`` so the builder can read it.
    """

    def __init__(self, n: int = 8, scaling_factor: float = 1.0):
        torch.manual_seed(0)
        self.src = torch.randn(n, *_LAT)
        self.tgt = torch.randn(n, *_LAT)
        self.src_label = torch.tensor([0] * n, dtype=torch.long)
        self.tgt_label = torch.tensor([1] * n, dtype=torch.long)
        self.spacing = torch.tensor([[1.0, 1.0, 1.0]] * n)
        self.scaling_factor = float(scaling_factor)
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        return {
            "src_latent": self.src[i],
            "tgt_latent": self.tgt[i],
            "src_label": self.src_label[i],
            "tgt_label": self.tgt_label[i],
            "spacing": self.spacing[i],
        }


def test_build_paired_reward_inputs_emits_train_val_probe_pairs():
    """The fake-cache builder produces train/val/probe {winner, loser} pair datasets."""
    gen = _IdentityPairedGen()
    sched = FlowMatchHeunDiscreteScheduler()
    train_ds = _FakePairedLatentDS(n=8)
    val_ds = _FakePairedLatentDS(n=4)
    inputs = build_paired_reward_inputs(
        train_ds=train_ds,
        val_ds=val_ds,
        generator=gen,
        base_scheduler=sched,
        num_steps=2,
        probe_num_steps=2,
        n_probe=4,
        batch_size=4,
        seed=0,
        device="cpu",
    )
    # Train: 8 pairs (one per src). Val: 4. Probe: 4 (capped by n_probe + len(val)).
    assert len(inputs.train_pair_ds) == 8
    assert len(inputs.val_pair_ds) == 4
    assert len(inputs.val_probe) == 4
    # All are condition-aware [2·C] concat latents.
    for ds in (inputs.train_pair_ds, inputs.val_pair_ds, inputs.val_probe):
        assert ds.winners.shape[1] == 2 * C_LATENT
        assert ds.losers.shape == ds.winners.shape
        assert torch.isfinite(ds.winners).all()


def test_build_paired_reward_inputs_is_deterministic():
    """Re-building with the same seed + generator yields byte-identical pairs (ADR-0020).

    The paired rollout is deterministic given x_src (no stochastic input); only the
    probe's t-draws are seeded, so the same seed reproduces the probe. Train/val
    pairs are fully deterministic (no RNG in the loser rollout).
    """
    gen = _IdentityPairedGen()
    sched = FlowMatchHeunDiscreteScheduler()
    train_ds = _FakePairedLatentDS(n=6)
    val_ds = _FakePairedLatentDS(n=4)
    kw = dict(
        train_ds=train_ds,
        val_ds=val_ds,
        generator=gen,
        base_scheduler=sched,
        num_steps=2,
        probe_num_steps=2,
        n_probe=4,
        batch_size=4,
        seed=0,
        device="cpu",
    )
    a = build_paired_reward_inputs(**kw)
    b = build_paired_reward_inputs(**kw)
    assert torch.equal(a.train_pair_ds.winners, b.train_pair_ds.winners)
    assert torch.equal(a.train_pair_ds.losers, b.train_pair_ds.losers)
    assert torch.equal(a.val_probe.winners, b.val_probe.winners)


def test_build_paired_reward_inputs_threads_scaling_factor():
    """The caller's dataset.scaling_factor is honored (scale-consistency, ADR-0021).

    The builder consumes whatever the dataset emits (the caller pre-scales). Here
    scaling_factor=2.0 -> the stacked latents are 2× the raw; the builder's output
    reflects that (the concat pairs carry the scaled values).
    """
    gen = _IdentityPairedGen()
    sched = FlowMatchHeunDiscreteScheduler()
    # Two datasets with the SAME raw latents but different scaling factors.
    torch.manual_seed(0)
    raw_src = torch.randn(4, *_LAT)
    raw_tgt = torch.randn(4, *_LAT)

    def _ds(factor):
        ds = _FakePairedLatentDS(n=4, scaling_factor=factor)
        ds.src = raw_src * factor
        ds.tgt = raw_tgt * factor
        return ds

    a = build_paired_reward_inputs(
        train_ds=_ds(1.0),
        val_ds=_ds(1.0),
        generator=gen,
        base_scheduler=sched,
        num_steps=2,
        probe_num_steps=2,
        n_probe=4,
        batch_size=4,
        seed=0,
        device="cpu",
    )
    b = build_paired_reward_inputs(
        train_ds=_ds(2.0),
        val_ds=_ds(2.0),
        generator=gen,
        base_scheduler=sched,
        num_steps=2,
        probe_num_steps=2,
        n_probe=4,
        batch_size=4,
        seed=0,
        device="cpu",
    )
    # The 2×-scaled train winners (cat([x_src, x_tgt])) are 2× the 1× ones.
    assert torch.allclose(b.train_pair_ds.winners, a.train_pair_ds.winners * 2.0)


def test_build_paired_reward_inputs_runs_end_to_end_training(tmp_path):
    """The builder's output drives run_paired_reward_training end-to-end (the real seam)."""
    from manifold.training import run_paired_reward_training

    gen = _IdentityPairedGen()
    sched = FlowMatchHeunDiscreteScheduler()
    inputs = build_paired_reward_inputs(
        train_ds=_FakePairedLatentDS(n=8),
        val_ds=_FakePairedLatentDS(n=4),
        generator=gen,
        base_scheduler=sched,
        num_steps=2,
        probe_num_steps=2,
        n_probe=4,
        batch_size=4,
        seed=0,
        device="cpu",
    )
    torch.manual_seed(0)
    module = PairedRewardModule(
        RewardModel(spatial_dims=3, in_channels=2 * C_LATENT, channels=8, num_layers_d=1), lr=1e-2
    )
    trainer, ckpt = run_paired_reward_training(
        module=module,
        inputs=inputs,
        model_dir=str(tmp_path),
        max_epochs=1,
        devices=1,
        accelerator="cpu",
        batch_size=2,
        limit_val_batches=1.0,
    )
    metrics = trainer.callback_metrics
    for key in ("val/pair_acc", "val/roc_auc", "val/gen_pair_acc"):
        assert key in metrics, f"missing {key}"
    assert Path(ckpt.best_model_path).is_file()


def test_build_paired_reward_inputs_uses_partial_scheduler_for_probe():
    """The probe path constructs the PartialFlowMatchHeunScheduler (ADR-0023).

    The train/val builders use the BASE scheduler (full 0->1 rollout); only the
    probe constructs the Partial subclass (for set_timesteps_partial). Spy on the
    PartialFlowMatchHeunScheduler constructor to confirm.
    """
    calls = {"partial": 0}
    real_init = PartialFlowMatchHeunScheduler.__init__

    def spy_init(self, *a, **kw):
        calls["partial"] += 1
        return real_init(self, *a, **kw)

    gen = _IdentityPairedGen()
    sched = FlowMatchHeunDiscreteScheduler()
    with patch.object(PartialFlowMatchHeunScheduler, "__init__", spy_init):
        build_paired_reward_inputs(
            train_ds=_FakePairedLatentDS(n=4),
            val_ds=_FakePairedLatentDS(n=4),
            generator=gen,
            base_scheduler=sched,
            num_steps=2,
            probe_num_steps=2,
            n_probe=4,
            batch_size=4,
            seed=0,
            device="cpu",
        )
    assert calls["partial"] == 1, "exactly one PartialFlowMatchHeunScheduler (the probe)"


# -- the real _real_inputs CLI path (smoke with a faked paired cache) ---------


def test_real_inputs_loads_generator_and_resolves_paired_split(tmp_path, monkeypatch):
    """_real_inputs loads the #94 generator + resolves the paired 2-way split.

    The real BraTS+VAE path is cluster-only; this smoke fakes build_brats_pair_manifest
    + _train_val_manifests + the warmed cache so _real_inputs runs end-to-end
    without NIfTIs. Asserts the generator is loaded (raw arm via #94), the
    paired split resolves, and the returned inputs carry precomputed pairs (no
    generator held by the Module downstream).
    """
    import lightning.pytorch as pl
    import omegaconf
    import stable_pretraining as spt
    from torch.utils.data import DataLoader

    from manifold import AutoencoderKL, PairedLatentFlowPipeline, UNet3DConditionModel
    from manifold.data import paired_brats as pb
    from manifold.modules.paired_latent_flow import PairedLatentFlowModule
    from manifold.training import paired_cli as pcli
    from manifold.training import paired_reward_cli
    from manifold.training.export import export_to_native

    # Build a paired native export (raw arm) via #94's export bridge.
    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    for p in unet.parameters():
        if p.abs().sum().item() == 0.0:
            torch.nn.init.normal_(p, std=0.01)
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

    class _D(torch.utils.data.Dataset):
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

    trainer.fit(module, datamodule=spt.data.DataModule(train=DataLoader(_D(), batch_size=2)))
    ckpt_path = str(tmp_path / "paired.ckpt")
    trainer.save_checkpoint(ckpt_path)
    fresh = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    export_to_native(
        ckpt_path,
        str(tmp_path / "native"),
        unet=fresh,
        vae=vae,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        pipeline_cls=PairedLatentFlowPipeline,
    )

    # Fake the BraTS manifest + the paired split (no NIfTIs on CPU): a train + val
    # manifest so build_paired_reward_inputs gets non-empty splits. Patch the SOURCE
    # modules _real_inputs imports from (lazy imports - patch at the source).
    train_manifest = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(4)
    ]
    val_manifest = [
        {"src": f"/v/s{i}-t1n.nii.gz", "tgt": f"/v/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(2)
    ]
    monkeypatch.setattr(
        pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest + val_manifest
    )
    monkeypatch.setattr(
        pcli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, val_manifest)
    )

    # Fake the warmed cache: replace PairedLatentDataset with a fake that serves
    # pre-built scaled latents (scale-consistency: factor = the export's 0.5).
    # _real_inputs builds two instances (train then val); alternate between them.
    # Patch at the SOURCE module (PairedLatentDataset is a lazy import in _real_inputs).
    from manifold.data import paired_latent_dataset as pld_mod

    fake_train = _FakePairedLatentDS(n=4, scaling_factor=0.5)
    fake_val = _FakePairedLatentDS(n=4, scaling_factor=0.5)
    built_instances = []
    _built = []

    class _FakePLD:
        # scaling_factor starts at a sentinel so the test can assert _real_inputs
        # set it to the export's (0.5) - the ADR-0021 scale-threading guard.
        scaling_factor = None

        def __init__(self, vol_ds, encode_fn=None, cache_dir=None, cache_tag=None):
            self._ds = _built.pop() if _built else fake_train
            self._n = 4
            self.scaling_factor = None  # sentinel: _real_inputs must overwrite this
            built_instances.append(self)

        def warm_cache(self, *a, **k):
            return None

        def __len__(self):
            return self._n

        def __getitem__(self, i):
            return self._ds[i]

    _built.extend([fake_val, fake_train])  # first build -> train (pop), second -> val
    monkeypatch.setattr(pld_mod, "PairedLatentDataset", _FakePLD)

    cfg = omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "latent_cache_dir": str(tmp_path / "cache"),
            "diffusion_unet_inference": {"dim": [8, 8, 8]},
            "autoencoder": {"num_channels": [8, 8]},
            "paired_reward": {
                "num_steps": 2,
                "precompute_num_steps": 2,
                "n_probe": 4,
                "gen_batch_size": 4,
                "cache_tag": "paired_train",
            },
            "random_seed": 0,
        }
    )
    inputs = paired_reward_cli._real_inputs(
        cfg, str(tmp_path / "native"), str(tmp_path / "cache"), torch.device("cpu")
    )
    assert len(inputs.train_pair_ds) == 4
    assert len(inputs.val_pair_ds) == 4
    assert inputs.val_probe is not None and len(inputs.val_probe) > 0
    assert inputs.train_pair_ds.winners.shape[1] == 2 * C_LATENT
    # ADR-0021 scale-threading (codex/verify scale-consistency): _real_inputs set
    # each dataset's scaling_factor to the export's (0.5 = AutoencoderKL above), not
    # the PairedLatentDataset default - guards the rollout operates in scaled space.
    assert len(built_instances) == 2
    assert all(ds.scaling_factor == 0.5 for ds in built_instances), (
        "_real_inputs must set ds.scaling_factor = the export's scaling_factor (ADR-0021)"
    )


def test_real_inputs_raises_on_no_val_split(tmp_path, monkeypatch):
    """No held-out val split -> clear ValueError (train never reused as val, ADR-0022)."""
    import omegaconf

    from manifold import AutoencoderKL, PairedLatentFlowPipeline, UNet3DConditionModel
    from manifold.data import paired_brats as pb
    from manifold.training import paired_cli as pcli
    from manifold.training import paired_reward_cli

    # A minimal paired native export so the generator loads.
    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    PairedLatentFlowPipeline(
        unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    ).save_pretrained(str(tmp_path / "native"))

    train_manifest = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(4)
    ]
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest)
    monkeypatch.setattr(
        pcli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, [])
    )  # empty val

    cfg = omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "diffusion_unet_inference": {"dim": [8, 8, 8]},
            "autoencoder": {"num_channels": [8, 8]},
            "paired_reward": {"num_steps": 2},
        }
    )
    with __import__("pytest").raises(ValueError, match="val split"):
        paired_reward_cli._real_inputs(
            cfg, str(tmp_path / "native"), str(tmp_path / "cache"), torch.device("cpu")
        )


def test_paired_reward_recipe_resolves_inference_dim():
    """The committed paired-reward recipe defines diffusion_unet_inference.dim.

    _real_inputs reads cfg.diffusion_unet_inference.dim directly (load-bearing for
    cache consistency). Regression guard (codex/verify cli-wiring): the recipe must
    define it so the documented launch does not raise ConfigAttributeError.
    """
    from manifold.config import load_config

    net = "configs/network/config_network.yaml"
    env = "configs/env/environment.yaml"
    train = "configs/train/config_paired_reward.yaml"
    cfg = load_config(env, train, net)
    assert tuple(int(d) for d in cfg.diffusion_unet_inference.dim) == (256, 256, 128)


# -- codex #96/#99 regression guards -------------------------------------------


def test_build_paired_reward_pairs_rejects_device_mismatch():
    """A generator on a different device than the requested ``device`` -> ValueError.

    Regression for codex #96 (P2): the builder moves ``src_b``/``tgt_b`` to the
    requested device but ``gen_tgt`` returns on the generator's device, so a mismatch
    would mix devices in the concat. The CLI path pre-moves the generator (in
    ``_real_inputs``); direct callers must too - fail fast rather than a CUDA error.
    """
    import pytest

    from manifold.data.paired_reward_pairs import build_paired_reward_pairs

    gen = _IdentityPairedGen()  # params on CPU
    sched = FlowMatchHeunDiscreteScheduler()
    x_src = torch.randn(2, *_LAT)
    x_tgt = torch.randn(2, *_LAT)
    # Request a device the generator is NOT on (torch.device("cuda") is constructible
    # without a GPU present; the CPU generator != cuda -> mismatch -> raise).
    with pytest.raises(ValueError, match="device"):
        build_paired_reward_pairs(
            x_src,
            x_tgt,
            gen,
            sched,
            src_label=0,
            tgt_label=1,
            spacing=[1.0, 1.0, 1.0],
            num_steps=2,
            batch_size=2,
            device="cuda",
        )


def test_build_paired_reward_probe_rejects_device_mismatch():
    """Same device guard on the probe builder (codex #96 P2, sibling of the pairs builder)."""
    import pytest

    from manifold.data.paired_reward_pairs import build_paired_reward_probe

    gen = _IdentityPairedGen()  # CPU
    sched = PartialFlowMatchHeunScheduler()
    x_src = torch.randn(2, *_LAT)
    x_tgt = torch.randn(2, *_LAT)
    with pytest.raises(ValueError, match="device"):
        build_paired_reward_probe(
            x_src,
            x_tgt,
            gen,
            sched,
            src_label=0,
            tgt_label=1,
            spacing=[1.0, 1.0, 1.0],
            num_steps=2,
            batch_size=2,
            device="cuda",
        )


def test_real_inputs_mirrors_paired_reward_val_fraction_to_root(tmp_path, monkeypatch):
    """_real_inputs mirrors paired_reward.val_fraction to root for _train_val_manifests.

    Regression for codex #99 (P1): _train_val_manifests reads ROOT cfg.val_fraction,
    but the recipe defines the held-out fraction under paired_reward.val_fraction.
    Without the mirror, a non-native-split env (val_data_base_dir unset) resolves
    val_fraction to 0 -> empty val -> the guard raises. Here the mirror makes
    _train_val_manifests receive the nested 0.3.
    """
    import omegaconf

    from manifold import AutoencoderKL, PairedLatentFlowPipeline, UNet3DConditionModel
    from manifold.data import paired_brats as pb
    from manifold.training import paired_cli as pcli
    from manifold.training import paired_reward_cli

    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    PairedLatentFlowPipeline(
        unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    ).save_pretrained(str(tmp_path / "native"))

    train_manifest = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(4)
    ]
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest)

    seen = {}

    def fake_split(cfg, manifest):
        seen["val_fraction"] = float(omegaconf.OmegaConf.select(cfg, "val_fraction", default=-1.0))
        # Return empty val to short-circuit before the cache build (we only assert the mirror).
        return manifest, []

    monkeypatch.setattr(pcli, "_train_val_manifests", fake_split)

    cfg = omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "diffusion_unet_inference": {"dim": [8, 8, 8]},
            "autoencoder": {"num_channels": [8, 8]},
            "paired_reward": {"num_steps": 2, "val_fraction": 0.3},
        }
    )
    import pytest

    with pytest.raises(ValueError, match="val split"):
        paired_reward_cli._real_inputs(
            cfg, str(tmp_path / "native"), str(tmp_path / "cache"), torch.device("cpu")
        )
    assert seen["val_fraction"] == 0.3, (
        "paired_reward.val_fraction must be mirrored to root cfg.val_fraction"
    )


def test_real_inputs_rejects_stale_cache_shape(tmp_path, monkeypatch):
    """_real_inputs rejects a paired_train cache whose shape mismatches target_dim.

    Regression for codex #99 (P2): the cache key is sample_id+tag (NOT target_dim),
    so a target_dim change silently reuses stale wrong-shape latents. _real_inputs
    validates a hit's spatial shape against target_dim // divisor and raises.
    """
    import omegaconf
    import pytest

    from manifold import AutoencoderKL, PairedLatentFlowPipeline, UNet3DConditionModel
    from manifold.data import paired_brats as pb
    from manifold.data import paired_latent_dataset as pld_mod
    from manifold.training import paired_cli as pcli
    from manifold.training import paired_reward_cli

    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    PairedLatentFlowPipeline(
        unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    ).save_pretrained(str(tmp_path / "native"))

    train_manifest = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(4)
    ]
    val_manifest = [
        {"src": f"/v/s{i}-t1n.nii.gz", "tgt": f"/v/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(2)
    ]
    monkeypatch.setattr(
        pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest + val_manifest
    )
    monkeypatch.setattr(
        pcli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, val_manifest)
    )

    # A fake PairedLatentDataset that exposes the real cache interface (source +
    # raw_latent) but serves a WRONG-shape latent (mismatches target_dim // divisor).
    fake_train = _FakePairedLatentDS(n=4, scaling_factor=0.5)
    fake_val = _FakePairedLatentDS(n=4, scaling_factor=0.5)
    _built = [fake_val, fake_train]

    class _FakePLDWithCache:
        class source:
            @staticmethod
            def unique_sample_ids():
                return ["s0", "s1", "s2", "s3"]

        def __init__(self, vol_ds, encode_fn=None, cache_dir=None, cache_tag=None):
            self._ds = _built.pop()
            self.scaling_factor = None

        def warm_cache(self, *a, **k):
            return None

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return self._ds[i]

        def raw_latent(self, sid):
            # WRONG shape: the fake latents are (C,8,8,8); target_dim [8,8,8] // divisor 4
            # = (2,2,2). Returning the (C,8,8,8) latent -> mismatch -> raise.
            return self._ds.src[0]

    monkeypatch.setattr(pld_mod, "PairedLatentDataset", _FakePLDWithCache)

    cfg = omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "latent_cache_dir": str(tmp_path / "cache"),
            "diffusion_unet_inference": {"dim": [8, 8, 8]},
            "autoencoder": {"num_channels": [8, 8]},  # 2 levels -> divisor 4 -> expected (2,2,2)
            "paired_reward": {
                "num_steps": 2,
                "precompute_num_steps": 2,
                "n_probe": 4,
                "gen_batch_size": 4,
            },
            "random_seed": 0,
        }
    )
    with pytest.raises(ValueError, match="Cached paired latent"):
        paired_reward_cli._real_inputs(
            cfg, str(tmp_path / "native"), str(tmp_path / "cache"), torch.device("cpu")
        )


# -- codex #100 round-3 regression guards --------------------------------------


def test_real_inputs_preserves_explicit_root_val_fraction(tmp_path, monkeypatch):
    """An explicit ROOT val_fraction override is NOT clobbered by paired_reward.val_fraction.

    Regression for codex #100 (P2 round-3 #E): the mirror fired unconditionally, so a
    CLI/profile root ``val_fraction=0.5`` would be overwritten by paired_reward.val_fraction
    (0.1). Now the mirror fires only when the root key is absent.
    """
    import omegaconf
    import pytest

    from manifold import AutoencoderKL, PairedLatentFlowPipeline, UNet3DConditionModel
    from manifold.data import paired_brats as pb
    from manifold.training import paired_cli as pcli
    from manifold.training import paired_reward_cli

    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    PairedLatentFlowPipeline(
        unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    ).save_pretrained(str(tmp_path / "native"))

    train_manifest = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(4)
    ]
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest)

    seen = {}

    def fake_split(cfg, manifest):
        seen["val_fraction"] = float(omegaconf.OmegaConf.select(cfg, "val_fraction", default=-1.0))
        return manifest, []

    monkeypatch.setattr(pcli, "_train_val_manifests", fake_split)

    # Root val_fraction=0.5 is set EXPLICITLY; paired_reward.val_fraction=0.1 must NOT
    # overwrite it. _train_val_manifests reads the root -> sees 0.5.
    cfg = omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "val_fraction": 0.5,
            "diffusion_unet_inference": {"dim": [8, 8, 8]},
            "autoencoder": {"num_channels": [8, 8]},
            "paired_reward": {"num_steps": 2, "val_fraction": 0.1},
        }
    )
    with pytest.raises(ValueError, match="val split"):
        paired_reward_cli._real_inputs(
            cfg, str(tmp_path / "native"), str(tmp_path / "cache"), torch.device("cpu")
        )
    assert seen["val_fraction"] == 0.5, "explicit root val_fraction must not be clobbered"


def test_real_inputs_rejects_mixed_cache_shape(tmp_path, monkeypatch):
    """A mixed cache (first entry OK, a later entry wrong) is caught (#100 P2 round-3 #F).

    Validating only the first entry would let a partial/mixed cache slip through; the
    check now walks every entry in both splits.
    """
    import omegaconf
    import pytest

    from manifold import AutoencoderKL, PairedLatentFlowPipeline, UNet3DConditionModel
    from manifold.data import paired_brats as pb
    from manifold.data import paired_latent_dataset as pld_mod
    from manifold.training import paired_cli as pcli
    from manifold.training import paired_reward_cli

    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    PairedLatentFlowPipeline(
        unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    ).save_pretrained(str(tmp_path / "native"))

    train_manifest = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(4)
    ]
    val_manifest = [
        {"src": f"/v/s{i}-t1n.nii.gz", "tgt": f"/v/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(2)
    ]
    monkeypatch.setattr(
        pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest + val_manifest
    )
    monkeypatch.setattr(
        pcli, "_train_val_manifests", lambda cfg, manifest: (train_manifest, val_manifest)
    )

    # A mixed cache: the FIRST train entry matches expected (2,2,2); a LATER train
    # entry is wrong (8,8,8). A first-only check would miss it; the every-entry check
    # catches it.
    good = torch.randn(C_LATENT, 4, 4, 4)  # matches target_dim [8,8,8] / divisor 2
    bad = torch.randn(C_LATENT, 8, 8, 8)
    sids = ["s0", "s1", "s2", "s3", "v0", "v1"]
    # Map: first sid good, second bad -> the first-only check would pass; every-entry catches.
    latents_by_sid = {"s0": good, "s1": bad, "s2": good, "s3": good, "v0": good, "v1": good}

    class _FakePLDMixed:
        class source:
            @staticmethod
            def unique_sample_ids():
                return sids

        def __init__(self, vol_ds, encode_fn=None, cache_dir=None, cache_tag=None):
            self.scaling_factor = None

        def warm_cache(self, *a, **k):
            return None

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return {
                "src_latent": good,
                "tgt_latent": good,
                "src_label": 0,
                "tgt_label": 1,
                "spacing": torch.tensor([1.0, 1.0, 1.0]),
            }

        def raw_latent(self, sid):
            return latents_by_sid[sid]

    monkeypatch.setattr(pld_mod, "PairedLatentDataset", _FakePLDMixed)

    cfg = omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "latent_cache_dir": str(tmp_path / "cache"),
            "diffusion_unet_inference": {"dim": [8, 8, 8]},
            "autoencoder": {"num_channels": [8, 8]},  # divisor 2 -> expected (4,4,4)
            "paired_reward": {
                "num_steps": 2,
                "precompute_num_steps": 2,
                "n_probe": 4,
                "gen_batch_size": 4,
            },
            "random_seed": 0,
        }
    )
    with pytest.raises(ValueError, match="Cached paired latent.*s1"):
        paired_reward_cli._real_inputs(
            cfg, str(tmp_path / "native"), str(tmp_path / "cache"), torch.device("cpu")
        )


# -- codex #100 round-2 regression guards --------------------------------------


def test_canonical_device_treats_unspecified_cuda_as_current():
    """``torch.device("cuda")`` canonicalizes to ``cuda:<current>`` (#100 P1 #A).

    The device guard would otherwise false-positive on the standard ``device="cuda"``
    CLI path: the generator lands on ``cuda:0`` after ``.to``, but
    ``torch.device("cuda") != torch.device("cuda:0")`` by equality. Canonicalize before
    comparing (mocked CUDA - no GPU required).
    """
    from unittest.mock import patch

    from manifold.data.paired_reward_pairs import _canonical_device

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
    ):
        assert _canonical_device(torch.device("cuda")) == torch.device("cuda", 0)
        assert _canonical_device(torch.device("cuda", 0)) == torch.device("cuda", 0)
        # A specific other index is preserved (cuda:1 != cuda:0).
        assert _canonical_device(torch.device("cuda", 1)) == torch.device("cuda", 1)
    # cpu passes through unchanged regardless of CUDA availability.
    assert _canonical_device(torch.device("cpu")) == torch.device("cpu")


def test_real_inputs_mirrors_val_fraction_for_non_directory_val_dir(tmp_path, monkeypatch):
    """_real_inputs mirrors paired_reward.val_fraction when val_data_base_dir is a
    NON-DIRECTORY (e.g. the BraTS2023 profile's brats_all_val.json) (#100 P1 #C).

    _train_val_manifests rejects a non-directory val_data_base_dir and falls back to the
    root val_fraction subject split, so the mirror must fire here too - not only when
    val_data_base_dir is unset.
    """
    import omegaconf
    import pytest

    from manifold import AutoencoderKL, PairedLatentFlowPipeline, UNet3DConditionModel
    from manifold.data import paired_brats as pb
    from manifold.training import paired_cli as pcli
    from manifold.training import paired_reward_cli

    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    PairedLatentFlowPipeline(
        unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler()
    ).save_pretrained(str(tmp_path / "native"))

    train_manifest = [
        {"src": f"/t/s{i}-t1n.nii.gz", "tgt": f"/t/s{i}-t1c.nii.gz", "src_label": 0, "tgt_label": 1}
        for i in range(4)
    ]
    monkeypatch.setattr(pb, "build_brats_pair_manifest", lambda *a, **k: train_manifest)

    seen = {}

    def fake_split(cfg, manifest):
        seen["val_fraction"] = float(omegaconf.OmegaConf.select(cfg, "val_fraction", default=-1.0))
        return manifest, []  # empty val -> short-circuit (we only assert the mirror)

    monkeypatch.setattr(pcli, "_train_val_manifests", fake_split)

    # A NON-DIRECTORY val_data_base_dir (a JSON manifest path, as the BraTS2023 profile
    # sets). _train_val_manifests falls back to val_fraction -> the mirror must fire.
    cfg = omegaconf.OmegaConf.create(
        {
            "data_base_dir": "/tmp/_unused_",
            "val_data_base_dir": "/tmp/_does_not_exist_as_dir_/brats_all_val.json",
            "diffusion_unet_inference": {"dim": [8, 8, 8]},
            "autoencoder": {"num_channels": [8, 8]},
            "paired_reward": {"num_steps": 2, "val_fraction": 0.25},
        }
    )
    with pytest.raises(ValueError, match="val split"):
        paired_reward_cli._real_inputs(
            cfg, str(tmp_path / "native"), str(tmp_path / "cache"), torch.device("cpu")
        )
    assert seen["val_fraction"] == 0.25, (
        "paired_reward.val_fraction must mirror to root even for a non-directory val_data_base_dir"
    )
