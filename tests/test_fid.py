"""Unbiased 2.5D FID tests (Slice D, issue #27).

The focused numerical seam pins the ``Tr(Σ)/n`` mean-term correction vs the
plug-in (biased) estimator on a fixed small-``N`` synthetic feature set, and the
callback seam runs the per-epoch FID end-to-end on a tiny set with a fake
feature network (the RadImageNet backbone needs ``torch.hub`` / network — out of
CI scope; the factory is the injected seam).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from manifold.metrics import (
    FIDCallback,
    frechet_distance_unbiased,
    get_features_2p5d,
)
from manifold.metrics.fid import _cov_unbiased, _tr


def _plug_in_fid(gen: torch.Tensor, real: torch.Tensor, ridge: float = 1e-6) -> float:
    """Biased plug-in FID (no mean-term subtraction) on the SAME unbiased cov.

    Reproduces the legacy ``FIDMetric`` shape so the test pins EXACTLY the
    ``Tr(Σ)/n`` subtraction the unbiased estimator applies.
    """
    from manifold.metrics.fid import _principal_sqrtm

    mu_g, mu_r = gen.mean(0), real.mean(0)
    sg, sr = _cov_unbiased(gen), _cov_unbiased(real)
    eye = torch.eye(sg.shape[0]) * ridge
    cov_term = _tr(sg) + _tr(sr) - 2.0 * _tr(_principal_sqrtm((sg + eye) @ (sr + eye)))
    return float((mu_g - mu_r).pow(2).sum() + cov_term)


def test_unbiased_fid_below_biased_by_trace_over_n():
    """unbiased == plug-in − (Tr(Σ_g)/n1 + Tr(Σ_r)/n2); and unbiased < plug-in."""
    torch.manual_seed(0)
    n1, n2, d = 12, 10, 8
    gen = torch.randn(n1, d) * 0.7 + 0.3
    real = torch.randn(n2, d) * 0.5

    unbiased = float(frechet_distance_unbiased(gen, real, ridge=1e-6))
    biased = _plug_in_fid(gen, real, ridge=1e-6)
    sg, sr = _cov_unbiased(gen), _cov_unbiased(real)
    correction = float(_tr(sg) / n1 + _tr(sr) / n2)

    assert unbiased < biased
    # The unbiased estimator subtracts exactly the Tr(Σ)/n mean-term bias.
    assert unbiased == pytest.approx(biased - correction, abs=1e-4)


def test_unbiased_fid_is_nonneg_and_finite():
    torch.manual_seed(1)
    gen = torch.randn(9, 5)
    real = torch.randn(7, 5) + 0.1
    fid = frechet_distance_unbiased(gen, real)
    assert torch.isfinite(fid)
    assert float(fid) >= 0.0


class _FakeFeatureNet(nn.Module):
    """Deterministic 2D-plane → feature: flatten + a fixed linear (no RNG)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 6, bias=False)  # 8x8 plane -> 64 dims -> 6 feats
        with torch.no_grad():
            self.proj.weight.copy_(torch.linspace(0.01, 0.06, self.proj.weight.numel()).reshape_as(self.proj.weight))

    def forward(self, plane: torch.Tensor) -> torch.Tensor:
        b = plane.shape[0]
        flat = plane.reshape(b, -1)[:, :64]
        if flat.shape[1] < 64:
            flat = torch.nn.functional.pad(flat, (0, 64 - flat.shape[1]))
        return self.proj(flat)


def test_get_features_2p5d_returns_three_planes():
    net = _FakeFeatureNet()
    volumes = torch.randn(2, 1, 8, 8, 8)
    planes = get_features_2p5d(volumes, net, center_slices_ratio=0.1)  # k=1 slice/axis
    assert len(planes) == 3
    for p in planes:
        assert p.dim() == 2 and p.shape[1] == 6
        assert p.shape[0] == 2  # one center slice/volume × 2 volumes


def _make_fid_callback(module, vae, ema, real_latents, *, num_synth=3, seed=0, ridge=1e-2):
    return FIDCallback(
        module=module,
        vae=vae,
        ema_callback=ema,
        real_latents=real_latents,
        feature_net=_FakeFeatureNet(),
        latent_shape=(1, 4, 4, 4, 4),
        spacing=[1.0, 1.0, 1.0],
        modality=1,
        num_inference_steps=2,
        num_synth=num_synth,
        center_slices_ratio=0.5,
        cov_ridge=ridge,
        seed=seed,
    )


def _datamodule(n=6, batch_size=2):
    from torch.utils.data import DataLoader, Dataset

    import stable_pretraining as spt

    class _DS(Dataset):
        def __init__(self):
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
            return n

        def __getitem__(self, i):
            return self.items[i]

    ds = _DS()
    train = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return spt.data.DataModule(train=train, val=val)


def _fit_with_fid(module, fid_cb, *, max_epochs=1):
    import lightning.pytorch as pl

    from manifold.training import DoubleEMACallback

    ema = fid_cb.ema_callback if isinstance(fid_cb.ema_callback, DoubleEMACallback) else DoubleEMACallback(module)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        callbacks=[ema, fid_cb],
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule=_datamodule())
    return trainer


def test_fid_callback_logs_finite_avg(latent_module):
    """The callback logs a finite val/fid_avg (+ per-plane) on a tiny set."""
    from manifold import AutoencoderKL
    from manifold.training import DoubleEMACallback

    vae = AutoencoderKL(scaling_factor=0.5)
    ema = DoubleEMACallback(latent_module)
    real_latents = torch.randn(6, 4, 4, 4, 4)
    cb = _make_fid_callback(latent_module, vae, ema, real_latents, ridge=1e-2)

    trainer = _fit_with_fid(latent_module, cb)
    metrics = trainer.callback_metrics
    assert "val/fid_avg" in metrics
    assert torch.isfinite(metrics["val/fid_avg"])
    assert any(k.startswith("val/fid_") and k != "val/fid_avg" for k in metrics)


def test_fid_callback_same_seed_reproduces(latent_module):
    """Same model + same seed -> identical synth features each epoch (fixed samples).

    The re-seeded generation noise makes the synthetic arm a deterministic
    function of the model: two passes over the same (frozen) module produce
    bit-identical per-plane features. (The training RNG is intentionally out of
    scope here — only the model + the callback's seed must match.)
    """
    from manifold import AutoencoderKL
    from manifold.training import DoubleEMACallback

    vae = AutoencoderKL(scaling_factor=0.5)
    ema = DoubleEMACallback(latent_module)
    real_latents = torch.randn(6, 4, 4, 4, 4)
    cb = _make_fid_callback(latent_module, vae, ema, real_latents, seed=42, ridge=1e-2)

    first = cb._synth_features()
    second = cb._synth_features()
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    # The real arm is cached identically across calls too.
    assert all(torch.equal(a, b) for a, b in zip(cb._real_features(), cb._real_features()))
