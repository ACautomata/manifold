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


def test_fid_callback_logs_raw_alongside_slow(latent_module):
    """log_raw_fid (default) logs val/fid_raw (+ per-plane) alongside val/fid_avg."""
    from manifold import AutoencoderKL
    from manifold.training import DoubleEMACallback

    vae = AutoencoderKL(scaling_factor=0.5)
    ema = DoubleEMACallback(latent_module)
    real_latents = torch.randn(6, 4, 4, 4, 4)
    cb = _make_fid_callback(latent_module, vae, ema, real_latents, ridge=1e-2)

    trainer = _fit_with_fid(latent_module, cb)
    metrics = trainer.callback_metrics
    assert "val/fid_raw" in metrics, "raw arm must log val/fid_raw"
    assert torch.isfinite(metrics["val/fid_raw"])
    assert any(k.startswith("val/fid_raw_") for k in metrics), "raw per-plane must log"
    # The slow-EMA arm is still logged alongside.
    assert "val/fid_avg" in metrics


def test_synth_features_raw_is_blind_to_ema_shadow(latent_module):
    """raw=True bypasses ema.swap_in: perturbing the slow shadow moves the
    slow-arm features but leaves the raw-arm features bit-identical. This is the
    load-bearing property that lets val/fid_raw track raw learning decoupled from
    the 0.9999 EMA's convergence lag."""
    from manifold import AutoencoderKL
    from manifold.training import DoubleEMACallback

    vae = AutoencoderKL(scaling_factor=0.5)
    ema = DoubleEMACallback(latent_module)
    real_latents = torch.randn(6, 4, 4, 4, 4)
    cb = _make_fid_callback(latent_module, vae, ema, real_latents, seed=0, ridge=1e-2)
    cb._stage_eval_on_device()

    raw_before = cb._synth_features(raw=True)
    slow_before = cb._synth_features(raw=False)

    # Replace the slow EMA shadow with fresh random weights — the raw arm must
    # not see this. (Random replacement, not a 2x mul: the UNet's group norms are
    # scale-invariant, so a uniform scaling would leave the sample unchanged.)
    g = torch.Generator().manual_seed(123)
    slow_shadow = ema._shadows.shadows[ema._shadows.slow_index]
    for n in slow_shadow:
        slow_shadow[n].copy_(
            torch.randn(slow_shadow[n].shape, generator=g, dtype=slow_shadow[n].dtype)
        )

    raw_after = cb._synth_features(raw=True)
    slow_after = cb._synth_features(raw=False)

    for a, b in zip(raw_before, raw_after):
        assert torch.equal(a, b), "raw arm must be blind to the EMA shadow"
    assert not all(torch.equal(a, b) for a, b in zip(slow_before, slow_after)), \
        "slow arm must reflect the replaced shadow"


# --- Offline RadImageNet loader (issue #32) -------------------------------------
# The factory's ``torch.hub.load`` path is offline-broken on gauss (wrong repo
# name + ``source='local'`` branch fragility), so the FID backbone is built from
# the cached ``RadImageNet-ResNet50_notop.pth`` state_dict directly. These tests
# pin the loader's contract with a *synthetic* state_dict — no torch.hub, no
# network, no real RadImageNet weights (CI has neither the repo nor the ckpt).


def test_radimagenet_checkpoint_path_honours_torch_home(monkeypatch, tmp_path):
    """Checkpoint path resolves under ``$TORCH_HOME/checkpoints/``."""
    from manifold.metrics.fid import _radimagenet_checkpoint_path

    monkeypatch.setenv("TORCH_HOME", str(tmp_path))
    assert _radimagenet_checkpoint_path() == str(
        tmp_path / "checkpoints" / "RadImageNet-ResNet50_notop.pth"
    )


def test_load_radimagenet_resnet50_keeps_conv_bias_and_strips_head(tmp_path):
    """The offline loader builds a bias-True resnet50 and strict-loads the full
    ``_notop`` state_dict without dropping the 53 conv biases a naive
    ``strict=False`` would silently discard (those biases shift features by ~3%
    in cosine — too much for a metric backbone). ``fc``/``avgpool`` become
    Identity so the model returns the post-layer4 spatial map (penultimate
    features), matching ``torch.hub.load(..., 'radimagenet_resnet50')``.
    """
    from torchvision.models import resnet50

    from manifold.metrics.fid import _load_radimagenet_resnet50, _match_radimagenet_arch

    # Synthesise a state_dict in the exact RadImageNet `_notop` format: bias on
    # every conv, no fc head, BN eps=1.001e-5.
    ref = resnet50(weights=None)
    _match_radimagenet_arch(ref)
    ref_state = {k: v.clone() for k, v in ref.state_dict().items() if not k.startswith("fc.")}
    assert "conv1.bias" in ref_state  # biases present
    assert not any(k.startswith("fc.") for k in ref_state)  # `_notop`: no head
    ckpt = tmp_path / "RadImageNet-ResNet50_notop.pth"
    torch.save(ref_state, ckpt)

    model = _load_radimagenet_resnet50(str(ckpt))
    model.eval()
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 2048 * 7 * 7)  # spatial penultimate features (flattened)
    # The bias params survived the strict load (not silently dropped).
    assert model.conv1.bias is not None
    assert torch.equal(model.conv1.bias, ref_state["conv1.bias"])
    assert isinstance(model.fc, nn.Identity)
    assert isinstance(model.avgpool, nn.Identity)


def test_match_radimagenet_arch_moves_stride_and_eps():
    """All three RadImageNet adaptations are pinned (a silent refactor of any one
    corrupts features despite matching keys/shapes):

    * conv bias=True (the 53 trained bias keys load at all);
    * bottleneck stride on conv1 (1×1), NOT torchvision's default conv2 (3×3);
    * BN eps = 1.001e-5.
    """
    from torchvision.models import resnet50
    from torchvision.models.resnet import Bottleneck

    from manifold.metrics.fid import _match_radimagenet_arch

    model = resnet50(weights=None)
    # torchvision default: stride on conv2, no conv bias, eps=1e-5.
    b = model.layer2[0]
    assert b.conv1.stride == (1, 1) and b.conv2.stride == (2, 2)
    assert model.conv1.bias is None
    assert any(isinstance(m, nn.BatchNorm2d) and m.eps == 1e-5 for m in model.modules())

    _match_radimagenet_arch(model)

    # Stride moved to conv1 on every downsampling bottleneck.
    for block in [*model.layer2, *model.layer3, *model.layer4]:
        first = block
        if isinstance(first, Bottleneck) and first.downsample is not None:
            assert first.conv1.stride == (2, 2)
            assert first.conv2.stride == (1, 1)
    # Every conv now carries bias.
    assert model.conv1.bias is not None
    assert all(
        m.bias is not None for m in model.modules() if isinstance(m, nn.Conv2d)
    )
    # BN eps matched.
    assert all(
        m.eps == 1.001e-5 for m in model.modules() if isinstance(m, nn.BatchNorm2d)
    )


def test_load_radimagenet_resnet50_rejects_mismatched_state(tmp_path):
    """A state_dict missing keys the bias-True model expects must raise (strict),
    not silently load a partial backbone."""
    from manifold.metrics.fid import _load_radimagenet_resnet50

    # An empty dict is missing every param → strict load must fail loudly.
    torch.save({}, tmp_path / "RadImageNet-ResNet50_notop.pth")
    with pytest.raises(Exception):
        _load_radimagenet_resnet50(str(tmp_path / "RadImageNet-ResNet50_notop.pth"))


# --- Global pooling of the spatial feature map --------------------------------
# _RadImageNetFeatures wraps the bare resnet (avgpool/fc=Identity, returns a
# 2048*h*w spatial map) and global-average-pools it to 2048-dim — matching hope:
# the hub RadImageNet ResNet50 has no avgpool/fc (its forward returns the layer4
# (B,2048,7,7) map) and hope pools it to (B,2048) via spatial_average. NOT a
# divergence.


def test_radimagenet_features_global_pool_to_2048():
    """The feature wrapper global-average-pools the spatial map to 2048-dim."""
    from manifold.metrics.fid import _RadImageNetFeatures

    # A bare bias-True resnet50 with avgpool/fc=Identity returns [B,2048,h,w].
    from torchvision.models import resnet50

    from manifold.metrics.fid import _match_radimagenet_arch

    model = resnet50(weights=None)
    _match_radimagenet_arch(model)
    model.avgpool = torch.nn.Identity()
    model.fc = torch.nn.Identity()
    model.eval()
    wrapped = _RadImageNetFeatures(model)
    out = wrapped(torch.rand(1, 1, 224, 224))
    assert out.shape == (1, 2048)  # pooled, not 2048*7*7 flattened


# --- RadImageNet preprocessing contract (matches hope radimagenet_intensity_) ----
# hope's RadImageNet preprocessing (metrics.fid.radimagenet_intensity_normalisation,
# applied per plane-batch): replicate 1->3 -> RGB->BGR -> per-plane-batch min-max to
# [0,1] -> caffe-mode ImageNet-mean subtract (no std division — the hub model has no
# internal normalisation, so this lands directly on conv1). This test pins that exact
# contract so a regression to mean-only (the original D1 bug — no min-max, which made
# FID scale-sensitive on the unclamped BraTS MR decode), to the torchvision mean+std
# recipe, or to a mean/flip channel-order mismatch, is caught at the numeric seam.
# The backbone is a passthrough so the preprocessed tensor is directly observable.


class _PassthroughBackbone(nn.Module):
    """Returns its input unchanged so the wrapper's preprocessing is observable."""

    def __init__(self):
        super().__init__()
        # Present so _RadImageNetFeatures.__init__ can read .avgpool; ignored here.
        self.avgpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_radimagenet_features_preprocessing_minmax_then_mean_bgr():
    """Preprocessing is replicate -> BGR flip -> per-plane-batch min-max -> mean-subtract.

    Matches hope's ``radimagenet_intensity_normalisation``: a single global
    min/max over the whole plane-batch maps it to ``[0,1]``, then caffe-mode
    ImageNet-mean subtract (no std division).

    * The mean buffer is the BGR constant ``(0.406, 0.456, 0.485)`` (pairs with the
      post-flip channel order — channel 0 is B, so it gets B's mean).
    * No ``_std`` buffer is registered (caffe-mode drops the ``/std``).
    * On replicated grayscale the channel flip is a value-level no-op, so the
      constant ORDER is the only thing the mean-subtract pairs with.
    * The min-max makes the output scale-invariant — feeding ``c*x`` yields the
      same features (the load-bearing fix for the unclamped BraTS MR decode).
    """
    from manifold.metrics.fid import _RadImageNetFeatures

    wrapped = _RadImageNetFeatures(_PassthroughBackbone())

    # Sanity: flip is a value-level no-op on replicated grayscale (all 3 channels
    # identical), so the mean-constant order is what's actually under test.
    x = torch.rand(2, 1, 4, 4)
    replicated = x.repeat(1, 3, 1, 1)
    assert torch.equal(replicated, replicated.flip(1))

    out = wrapped(x)  # backbone is a passthrough -> out == preprocessed x, flattened

    mean_bgr = torch.tensor((0.406, 0.456, 0.485)).view(1, 3, 1, 1)
    bgr = replicated.flip(1)
    minv, maxv = torch.min(bgr), torch.max(bgr)
    normed = (bgr - minv) / (maxv - minv + 1e-10)  # hope's per-plane-batch min-max
    expected = (normed - mean_bgr).flatten(1)
    assert out.shape == expected.shape == (2, 48)
    assert torch.allclose(out, expected)
    # The mean is the BGR constant, in channel order (float32 repr → approx).
    assert wrapped._mean.flatten().tolist() == pytest.approx(
        [0.406, 0.456, 0.485], abs=1e-6
    )
    # No std division: the std buffer is gone.
    assert not hasattr(wrapped, "_std")
    # Scale invariance: feeding 3x the input yields identical features (min-max
    # cancels the scale) — the property the old mean-only preprocessing lacked.
    assert torch.allclose(wrapped(x * 3.0), out, atol=1e-6)
