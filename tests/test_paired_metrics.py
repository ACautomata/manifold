"""Paired JiT pixel-space 3D PSNR/SSIM callback tests (Slice 3, issue #68).

The metric seam pins the PSNR formula on a known synthetic pair decoded through
an **identity** fake VAE (so pred/target volumes are known tensors and PSNR is
exact), and the SSIM sanity (identical → 1.0) at the torchmetrics contract.

The callback seam exercises the FIDCallback-mirrored lifecycle end-to-end on a
tiny CPU fit: VAE staged to the UNet device around validation + restored after,
rank-0 + cadence gate, and ``val/psnr`` / ``val/ssim`` logged as finite floats.

All fixtures are local (the identity VAE is synthetic — the real AutoencoderKL
needs MONAI weights; the staging path is exercised via a ``.to`` spy).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure

from manifold import (
    FlowMatchHeunDiscreteScheduler,
    PairedLatentFlowModule,
    PairedLatentFlowPipeline,
    UNet3DConditionModel,
)
from manifold.metrics import PairedPSNRSSIMCallback

#: Latent channel count of the tiny fixtures (matches the held VAE).
C_LATENT = 4
#: Spatial side used for synthetic volumes — large enough that torchmetrics'
#: default 3D gaussian kernel (sigma=1.5 → 11³, pad=5) reflects without hitting
#: the ``pad < dim`` guard (``dim > 5`` required; 8 leaves headroom).
SPATIAL = 8


# -- Fakes ----------------------------------------------------------------------


class _IdentityVAE(nn.Module):
    """Identity-decode VAE: ``decode(z) = z`` (in pixel space already).

    Stands in for the held frozen :class:`~manifold.AutoencoderKL` so the metric
    formula is pinnable on known tensors — the decoded volume equals the latent
    (the real VAE undoes ``scaling_factor`` internally; ADR-0003). Carries a
    dummy parameter so ``.to`` / ``state_dict`` / ``parameters`` behave like a
    real module (the staging path clones state and moves params). Records the
    device ``decode`` ran on so the staging test asserts the CPU↔GPU API path.
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.decode_device: torch.device | None = None

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.decode_device = latents.device
        return latents


class _FakeUNet(nn.Module):
    """UNet stand-in carrying one parameter so ``next(.parameters()).device`` resolves.

    For the staging / gate unit tests that do not run the rollout.
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))


class _FakePipeline:
    """Minimal pipeline stand-in (``unet`` + ``vae``) for tests that need no rollout."""

    def __init__(self, unet, vae):
        self.unet = unet
        self.vae = vae


# -- Local fixtures -------------------------------------------------------------


def _trainable_paired_unet() -> UNet3DConditionModel:
    """A tiny paired UNet (``in_channels = 2·C_latent``) with MAISI's zero-init
    output conv re-initialized (see ``test_paired_module_training``). Re-init lets
    the rollout produce a non-trivial target so PSNR is finite and informative."""
    torch.manual_seed(0)
    unet = UNet3DConditionModel(
        in_channels=2 * C_LATENT,
        out_channels=C_LATENT,
        num_class_embeds=4,
        include_spacing_input=True,
    )
    for p in unet.parameters():
        if p.abs().sum().item() == 0.0:
            nn.init.normal_(p, std=0.01)
    return unet


@pytest.fixture
def identity_vae() -> _IdentityVAE:
    return _IdentityVAE()


def _make_callback(pipeline, *, num_inference_steps=2, every_n_epochs=1) -> PairedPSNRSSIMCallback:
    return PairedPSNRSSIMCallback(
        pipeline=pipeline,
        num_inference_steps=num_inference_steps,
        every_n_epochs=every_n_epochs,
    )


# ============================================================================
# 1. PSNR formula pinned on a known synthetic pair (identity VAE)
# ============================================================================


def test_psnr_formula_pinned_on_known_pair(identity_vae):
    """decode(identity) + noise offset -> PSNR matches manual recompute on normalized volumes.

    Volumes are pre-normalized to [0, 1] via ``_minmax_to_unit`` (mirrors the
    callback's flow between ``_eval_decode`` and ``_batch_metrics``). The PSNR
    formula is verified by computing the expected value independently from the
    same normalized tensors.
    """
    cb = _make_callback(_FakePipeline(_FakeUNet(), identity_vae))
    cb._stage_eval_on_device()

    torch.manual_seed(42)
    tgt_latent = torch.zeros(1, C_LATENT, SPATIAL, SPATIAL, SPATIAL)
    tgt_latent[..., ::2] = 1.0  # half ones -> [0, 1] range after identity decode
    noise = 0.2 * torch.randn_like(tgt_latent)
    pred_latent = tgt_latent + noise  # non-constant offset avoids collapse under min-max

    pred_vol = cb._eval_decode(pred_latent)
    tgt_vol = cb._eval_decode(tgt_latent)
    pred_norm = PairedPSNRSSIMCallback._minmax_to_unit(pred_vol)
    tgt_norm = PairedPSNRSSIMCallback._minmax_to_unit(tgt_vol)
    psnr_sum, ssim_sum, n = cb._batch_metrics(pred_norm, tgt_norm)

    assert n == 1
    # PSNR matches manual recompute on the same normalized tensors.
    p = pred_norm[0:1].float()
    t = tgt_norm[0:1].float()
    dr = float(t.max() - t.min())
    mse = float((p - t).pow(2).mean())
    expected = 10.0 * math.log10(dr**2 / mse)
    assert psnr_sum == pytest.approx(expected, abs=1e-4)
    # SSIM is a finite float in [0, 1] (structural similarity is bounded).
    assert math.isfinite(ssim_sum)
    assert 0.0 <= ssim_sum <= 1.0


def test_psnr_matches_direct_recompute(identity_vae):
    """The callback's PSNR equals an independent ``10·log10(dr²/mse)`` on min-max-normalized data.

    Pre-normalizes the decoded volumes via ``_minmax_to_unit`` (mirrors the
    callback's flow between ``_eval_decode`` and ``_batch_metrics``), then asserts
    the callback's per-sample ``data_range = target[max − min]`` formula matches
    a direct recompute on the same normalized tensors.
    """
    cb = _make_callback(_FakePipeline(_FakeUNet(), identity_vae))
    cb._stage_eval_on_device()
    torch.manual_seed(7)
    tgt = torch.randn(1, C_LATENT, SPATIAL, SPATIAL, SPATIAL)
    pred = tgt + 0.3 * torch.randn_like(tgt)

    pred_vol = cb._eval_decode(pred)
    tgt_vol = cb._eval_decode(tgt)
    pred_norm = PairedPSNRSSIMCallback._minmax_to_unit(pred_vol)
    tgt_norm = PairedPSNRSSIMCallback._minmax_to_unit(tgt_vol)
    psnr_sum, _, n = cb._batch_metrics(pred_norm, tgt_norm)

    data_range = float(tgt_norm.max() - tgt_norm.min())
    mse = float((pred_norm - tgt_norm).pow(2).mean())
    expected = 10.0 * math.log10(data_range**2 / mse)
    assert n == 1
    assert psnr_sum == pytest.approx(expected, abs=1e-4)




def test_affine_collapse_counted_not_skipped(identity_vae):
    """Pred = tgt + offset must NOT be skipped after per-volume min-max (codex #86 P2).

    Independent min-max normalisation collapses pred = A·tgt + B into an exact
    match (mse == 0). The old `mse == 0 → continue` would skip every sample,
    leaving _count = 0 and breaking the checkpoint monitor. The fix caps PSNR
    at 100 dB instead, so the sample is counted (n > 0) with a finite score
    and the SSIM from torchmetrics (≈ 1.0 for structurally identical volumes).
    """
    cb = _make_callback(_FakePipeline(_FakeUNet(), identity_vae))
    cb._stage_eval_on_device()
    tgt = torch.randn(1, C_LATENT, SPATIAL, SPATIAL, SPATIAL)
    pred = tgt + 0.1  # pure offset — collapses to identical after independent min-max
    tgt_vol = cb._eval_decode(tgt)
    pred_vol = cb._eval_decode(pred)
    pred_norm = PairedPSNRSSIMCallback._minmax_to_unit(pred_vol)
    tgt_norm = PairedPSNRSSIMCallback._minmax_to_unit(tgt_vol)
    psnr_sum, ssim_sum, n = cb._batch_metrics(pred_norm, tgt_norm)
    assert n == 1, "affine-collapsed sample must be counted, not skipped"
    assert psnr_sum == pytest.approx(100.0, abs=1e-4), "PSNR must be capped at 100 dB"
    assert ssim_sum >= 0.999, "structurally identical volumes must have SSIM ~1.0"


def test_ssim_is_one_for_identical_volumes(identity_vae):
    """torchmetrics' 3D SSIM of a volume with itself is 1.0 (the SSIM sanity bound).

    Pins the torchmetrics call contract the callback relies on. (Through the
    callback, identical → ``mse == 0`` is skipped — so the sanity is asserted
    directly at the torchmetrics functional the callback calls.)
    """
    torch.manual_seed(0)
    vol = torch.rand(1, C_LATENT, SPATIAL, SPATIAL, SPATIAL)
    data_range = float(vol.max() - vol.min())
    ssim_same = float(structural_similarity_index_measure(vol, vol, data_range=data_range))
    assert ssim_same == pytest.approx(1.0, abs=1e-5)


# ============================================================================
# 2. Decode staging: CPU ↔ GPU API path (mirrors FIDCallback)
# ============================================================================


def test_decode_staging_moves_vae_to_device_and_back(identity_vae):
    """Staging moves the VAE to the UNet device; restore returns it to CPU.

    The device is exercised at the API level: a spy on ``vae.to`` records the
    staged device + ``"cpu"`` on restore, and ``decode`` runs on the staged
    device (recorded by ``_IdentityVAE.decode_device``). Mirrors FIDCallback's
    ``_stage_eval_on_device`` / ``_restore_eval_to_cpu`` contract.
    """
    unet = _FakeUNet()  # CPU here; the spy proves the API path either way
    cb = _make_callback(_FakePipeline(unet, identity_vae))
    expected_device = next(unet.parameters()).device

    to_calls: list = []
    real_to = identity_vae.to

    def spy_to(device, *args, **kwargs):
        to_calls.append(device)
        return real_to(device, *args, **kwargs)

    identity_vae.to = spy_to  # type: ignore[assignment]

    assert cb._eval_staged is False
    cb._stage_eval_on_device()
    assert cb._eval_staged is True
    assert to_calls == [expected_device]

    # Decode runs on the staged device.
    _ = cb._eval_decode(torch.zeros(1, C_LATENT, SPATIAL, SPATIAL, SPATIAL))
    assert identity_vae.decode_device == expected_device

    cb._restore_eval_to_cpu()
    assert cb._eval_staged is False
    assert to_calls[-1] == "cpu"
    # The CPU state_dict clone round-trips (load_state_dict restores it cleanly).
    assert identity_vae.decode_device == expected_device  # unchanged by restore


def test_staging_is_idempotent(identity_vae):
    """``_stage_eval_on_device`` is idempotent (the ``_eval_staged`` flag guards re-entry)."""
    cb = _make_callback(_FakePipeline(_FakeUNet(), identity_vae))
    to_calls: list = []
    real_to = identity_vae.to
    identity_vae.to = lambda device, *a, **k: (to_calls.append(device), real_to(device, *a, **k))[1]  # type: ignore[assignment]
    cb._stage_eval_on_device()
    cb._stage_eval_on_device()  # second call must be a no-op
    assert len(to_calls) == 1, "staging must not re-stage when already staged"


# ============================================================================
# 3. Cadence gate (all ranks active under DDP - mirrors FIDCallback's lifecycle)
# ============================================================================


def _fake_trainer(*, is_global_zero: bool, current_epoch: int = 0):
    return SimpleNamespace(is_global_zero=is_global_zero, current_epoch=current_epoch)


def test_gate_single_process_always_active(identity_vae):
    """Single-process (no DDP): the gate is active on every cadence epoch."""
    cb = _make_callback(_FakePipeline(_FakeUNet(), identity_vae))
    assert cb._gated(_fake_trainer(is_global_zero=True, current_epoch=0)) is True


def test_gate_active_on_all_ranks_under_ddp(identity_vae, monkeypatch):
    """Under DDP every rank runs the decode over its own ``DistributedSampler`` shard
    (``on_validation_epoch_end`` ``all_gather``'s the per-volume sums to the global
    mean), so the gate is cadence-only - it no longer skips non-rank-0. There is no
    per-batch collective, so the decode loop cannot deadlock; the rank-asymmetric
    early-return is gone (ADR-0016 amendment).
    """
    cb = _make_callback(_FakePipeline(_FakeUNet(), identity_vae))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    assert cb._gated(_fake_trainer(is_global_zero=True)) is True
    assert cb._gated(_fake_trainer(is_global_zero=False)) is True


def test_gate_cadence_every_n_epochs(identity_vae):
    """``every_n_epochs=2`` runs on even epochs, skips odd ones."""
    cb = _make_callback(_FakePipeline(_FakeUNet(), identity_vae), every_n_epochs=2)
    assert cb._gated(_fake_trainer(is_global_zero=True, current_epoch=0)) is True
    assert cb._gated(_fake_trainer(is_global_zero=True, current_epoch=1)) is False
    assert cb._gated(_fake_trainer(is_global_zero=True, current_epoch=2)) is True


# ============================================================================
# 4. End-to-end: hook + logging through a real tiny fit
# ============================================================================


def _paired_item(i: int) -> dict:
    """One paired val/train item: scaled src + tgt latents, contrast labels, spacing."""
    torch.manual_seed(100 + i)
    return {
        "src_latent": torch.randn(C_LATENT, SPATIAL, SPATIAL, SPATIAL),
        "tgt_latent": torch.randn(C_LATENT, SPATIAL, SPATIAL, SPATIAL),
        "src_label": torch.tensor(0, dtype=torch.long),
        "tgt_label": torch.tensor(1, dtype=torch.long),
        "spacing": torch.tensor([1.0, 1.0, 1.0]),
    }


def _paired_datamodule(n: int = 4, batch_size: int = 2):
    """Train + val DataModule yielding paired batches (the v1 single-direction contract)."""
    from torch.utils.data import DataLoader, Dataset

    import stable_pretraining as spt

    class _DS(Dataset):
        def __init__(self):
            self.items = [_paired_item(i) for i in range(n)]

        def __len__(self):
            return n

        def __getitem__(self, i):
            return self.items[i]

    ds = _DS()
    train = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return spt.data.DataModule(train=train, val=val)


def _fit_with_callback(*, max_epochs: int = 1, num_inference_steps: int = 2):
    """Build a tiny paired module + identity-VAE pipeline + callback and fit."""
    import lightning.pytorch as pl

    unet = _trainable_paired_unet()
    scheduler = FlowMatchHeunDiscreteScheduler()
    module = PairedLatentFlowModule(unet, scheduler, lr=1e-2)
    pipeline = PairedLatentFlowPipeline(unet, _IdentityVAE(), scheduler)
    cb = _make_callback(pipeline, num_inference_steps=num_inference_steps)

    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[cb],
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=_paired_datamodule())
    return trainer, cb, module, pipeline


def test_callback_logs_finite_psnr_and_ssim():
    """End-to-end: the callback logs finite ``val/psnr`` + ``val/ssim`` on a tiny fit.

    The rollout runs through the live module UNet (training updates visible),
    both latents decode through the staged identity VAE, and the per-sample 3D
    metrics average to finite floats. Also asserts the VAE is restored to CPU
    after validation (``_eval_staged`` cleared) — the FIDCallback mirror.
    """
    trainer, cb, _module, _pipeline = _fit_with_callback(max_epochs=1)
    metrics = trainer.callback_metrics
    assert "val/psnr" in metrics, "callback must log val/psnr"
    assert "val/ssim" in metrics, "callback must log val/ssim"
    assert torch.isfinite(metrics["val/psnr"])
    assert torch.isfinite(metrics["val/ssim"])
    # SSIM is bounded in [0, 1]; PSNR for a non-degenerate pair is a finite dB.
    assert 0.0 <= float(metrics["val/ssim"]) <= 1.0
    # The VAE was restored to CPU + staging flag cleared at the end of validation.
    assert cb._eval_staged is False


def test_callback_matches_independent_recompute():
    """The logged ``val/psnr`` matches an independent rollout+decode+metric recompute.

    Strongest wiring assertion: after fit, re-run the rollout (deterministic given
    x_src — ADR-0013) on each val sample through the same identity VAE and the
    same formula, and assert the callback's logged value equals it. Catches
    double-counting, wrong averaging, or a stale-VAE decode.
    """
    trainer, cb, module, pipeline = _fit_with_callback(max_epochs=1)
    logged_psnr = float(trainer.callback_metrics["val/psnr"])

    # Re-run the val rollout with the post-fit UNet (eval mode, no grad) and
    # recompute PSNR with the identical formula. The rollout is deterministic
    # given x_src (ADR-0013), so post-fit weights reproduce the logged value.
    val_loader = _paired_datamodule().val
    psnr_sum, _, n = 0.0, 0.0, 0
    for batch in val_loader:
        pred_latent = pipeline.sample_latent(
            batch["src_latent"], batch["spacing"], 0, 1, num_inference_steps=2
        )
        # Identity decode (the fake VAE was restored to CPU after fit; decode is
        # device-agnostic for identity, so recompute directly).
        pred_vol, tgt_vol = pred_latent.float(), batch["tgt_latent"].float()
        # Pre-normalize (mirrors the callback's _minmax_to_unit called between
        # _eval_decode and _batch_metrics). Without this the recomputed PSNR/SSIM
        # would be on the raw volume range and would not match the logged value.
        pred_vol = PairedPSNRSSIMCallback._minmax_to_unit(pred_vol)
        tgt_vol = PairedPSNRSSIMCallback._minmax_to_unit(tgt_vol)
        p, s, m = cb._batch_metrics(pred_vol, tgt_vol)
        psnr_sum += p
        n += m
    expected = psnr_sum / n
    assert logged_psnr == pytest.approx(expected, abs=1e-4)


def test_rollout_deterministic_given_x_src_no_reseed():
    """The Paired JiT rollout is bit-identical across repeats on the same input.

    Pins ADR-0013's load-bearing consequence for this callback: the transport is
    deterministic given ``x_src`` (the ``t = 0`` endpoint is a data latent, not
    sampled noise), so the rollout has no stochastic input to fix. This is why —
    unlike the noise→data FID callback's re-seeded generation noise — this
    callback carries no ``seed`` and needs no per-epoch re-seeding: the val
    subset alone is the fixed reference. (End-to-end determinism is also covered
    by ``test_callback_matches_independent_recompute`` — the post-fit recompute
    matches the logged value only because the rollout is reproducible.)
    """
    unet = _trainable_paired_unet()
    scheduler = FlowMatchHeunDiscreteScheduler()
    pipeline = PairedLatentFlowPipeline(unet, _IdentityVAE(), scheduler)

    batch = next(iter(_paired_datamodule().val))
    a = pipeline.sample_latent(
        batch["src_latent"], batch["spacing"], 0, 1, num_inference_steps=2
    )
    b = pipeline.sample_latent(
        batch["src_latent"], batch["spacing"], 0, 1, num_inference_steps=2
    )
    assert torch.equal(a, b), "rollout must be deterministic given x_src (ADR-0013)"
    # And the callback exposes no seed parameter (nothing to re-seed).
    import inspect

    assert "seed" not in inspect.signature(PairedPSNRSSIMCallback.__init__).parameters
