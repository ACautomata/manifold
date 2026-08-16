"""PairedFidelityCallback unit-at-hook tests (issue #238, ADR-0037).

The seam under test is the callback's validation hook, driven directly with a stub
module (stubbed ``sample()``), a toy VAE, and a tiny paired dataset — asserting
**external behavior only** (the logged ``val/psnr`` / ``val/ssim``, the gating, the
determinism, the observe-only contract), never the implementation (not the rollout
loop, not which helper is invoked). Prior art: the ``LatentX0MAE`` unit-at-hook test
(``test_training.py``) and the FID same-seed-reproduces test (``test_fid.py``).

The toy VAE decodes ``[N, 4, 8, 8, 8]`` latents to ``[N, 1, 16, 16, 16]`` images so the
real MONAI-backed metric runs (its SSIM ``win_size=11`` needs ≥ 11 voxels per side).
The metric itself is validated in ``test_paired_fidelity.py``; here it validates the
**wiring** (decode → min-max → score) end-to-end.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from manifold.metrics import PairedFidelityCallback, PairedFidelityMetrics, PairedFidelityScores

#: Latent shape per sample (C=4, 8³ spatial) → decodes to 16³ image (≥ 11 for SSIM).
LATENT_SHAPE = (4, 8, 8, 8)


class _ToyVAE(nn.Module):
    """A VAE-like module: a movable parameter + a deterministic decode.

    Decode maps a ``[N, 4, 8, 8, 8]`` latent to a ``[N, 1, 16, 16, 16]`` image
    (channel-mean then 2× trilinear upsample) — shape-preserving in the spatial rank
    so a wrong-shaped generated latent surfaces the metric's shape-mismatch error.
    """

    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1))
        self.decode_calls = 0

    def decode(self, latents):
        self.decode_calls += 1
        img = latents.float().mean(dim=1, keepdim=True)  # [N, 1, 8, 8, 8]
        return F.interpolate(img, size=(16, 16, 16), mode="trilinear", align_corners=False)


class _StubModule:
    """The ControlNet-module seam: stubbed ``sample()`` + device markers + log capture.

    A plain object (not ``nn.Module``) so the callback's ``setattr`` of the two
    MeanMetrics lands as plain attributes; ``unet`` is a real tiny module so the
    ``next(module.unet.parameters()).device`` device read works.
    """

    def __init__(self, sample_fn):
        self.unet = nn.Linear(1, 1)
        self.device = torch.device("cpu")
        self._sample_fn = sample_fn
        self.logged: dict = {}
        self.sample_calls: list = []

    def sample(self, noise, src, spacing, src_label, tgt_label, num_inference_steps):
        # Record what observe-only generation saw: the src subset + whether grads ran.
        self.sample_calls.append(
            {"src": src, "inference_mode": torch.is_inference_mode_enabled()}
        )
        return self._sample_fn(noise, src, spacing, src_label, tgt_label)

    def log(self, name, value):
        self.logged[name] = value


class _TinyPairedDataset:
    """Sized + indexable source of paired sample dicts (the fixed-subset source).

    Emits the ``PairedLatentDataset`` contract: ``src_latent`` / ``tgt_latent``
    ``[C, D, H, W]``, ``src_label`` / ``tgt_label`` 0-d long, ``spacing`` ``[3]``.
    """

    def __init__(self, n: int = 6, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.items = [
            {
                "src_latent": torch.randn(*LATENT_SHAPE, generator=g),
                "tgt_latent": torch.randn(*LATENT_SHAPE, generator=g),
                "src_label": torch.tensor(i % 4, dtype=torch.long),
                "tgt_label": torch.tensor((i + 1) % 4, dtype=torch.long),
                "spacing": torch.tensor([1.0, 1.0, 1.0]),
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _trainer(epoch: int):
    """A fake trainer carrying only what the gate reads: ``current_epoch``."""
    return SimpleNamespace(current_epoch=epoch)


def _logged_scalars(module):
    """Resolve the module-attached MeanMetrics the callback logged to plain floats."""
    return {name: float(m.compute()) for name, m in module.logged.items()}


def test_logs_psnr_and_ssim_on_fixed_subset():
    """The hook logs finite val/psnr + val/ssim on the fixed paired subset, computed
    via the real paired-fidelity metric (decode → min-max → score wiring)."""
    module = _StubModule(sample_fn=lambda noise, *a: noise)  # generated := the noise
    vae = _ToyVAE()
    cb = PairedFidelityCallback(
        module=module, vae=vae, paired_data=_TinyPairedDataset(), num_inference_steps=3
    )
    cb.on_validation_epoch_end(_trainer(epoch=0), module)

    assert vae.decode_calls > 0, "the decode ran through the staged VAE"
    assert next(vae.parameters()).device.type == "cpu", "VAE restored to CPU after"
    scalars = _logged_scalars(module)
    assert set(scalars) == {"val/psnr", "val/ssim"}
    assert math.isfinite(scalars["val/psnr"])
    assert -1.0 <= scalars["val/ssim"] <= 1.0


def test_every_n_epochs_gates_off_epochs():
    """The every_n_epochs gate skips off-epochs: no generation, no decode, no log."""
    module = _StubModule(sample_fn=lambda noise, *a: noise)
    vae = _ToyVAE()
    cb = PairedFidelityCallback(
        module=module, vae=vae, paired_data=_TinyPairedDataset(),
        every_n_epochs=2, num_inference_steps=3,
    )
    cb.on_validation_epoch_end(_trainer(epoch=1), module)  # off-epoch (1 % 2 != 0)
    assert module.sample_calls == [], "sample() ran on an off-epoch"
    assert vae.decode_calls == 0, "decode ran on an off-epoch"
    assert module.logged == {}, "logged on an off-epoch"

    cb.on_validation_epoch_end(_trainer(epoch=2), module)  # on-epoch (2 % 2 == 0)
    assert len(module.sample_calls) == 1
    assert set(module.logged) == {"val/psnr", "val/ssim"}


def test_same_seed_and_subset_reproduces_value():
    """Same seed + fixed subset ⇒ identical metric across epochs (determinism).

    The stub returns the *noise*, so the value only reproduces if the generation
    noise is genuinely re-seeded each epoch (a persisted/advanced generator would
    drift) — and if the metric is reset between epochs (not accumulated).
    """
    module = _StubModule(sample_fn=lambda noise, *a: noise)
    cb = PairedFidelityCallback(
        module=module, vae=_ToyVAE(), paired_data=_TinyPairedDataset(),
        seed=42, num_inference_steps=3,
    )
    cb.on_validation_epoch_end(_trainer(epoch=0), module)
    first = _logged_scalars(module)
    cb.on_validation_epoch_end(_trainer(epoch=1), module)  # every_n_epochs=1: runs
    second = _logged_scalars(module)
    assert first["val/psnr"] == second["val/psnr"]
    assert first["val/ssim"] == second["val/ssim"]


def test_fixed_subset_is_stable_across_epochs():
    """The same fixed subset is scored every epoch (the model changes, not the data):
    the src batch the rollout sees is identical across epochs."""
    module = _StubModule(sample_fn=lambda noise, *a: noise)
    cb = PairedFidelityCallback(
        module=module, vae=_ToyVAE(), paired_data=_TinyPairedDataset(n=6),
        subset_size=4, num_inference_steps=3,
    )
    cb.on_validation_epoch_end(_trainer(epoch=0), module)
    cb.on_validation_epoch_end(_trainer(epoch=1), module)
    assert len(module.sample_calls) == 2
    src_first = module.sample_calls[0]["src"]
    src_second = module.sample_calls[1]["src"]
    assert src_first.shape == (4, *LATENT_SHAPE)  # subset_size=4, batched
    assert torch.equal(src_first, src_second)


def test_metric_resets_between_epochs():
    """The logged value is the current epoch's, not an accumulation over past epochs.

    A changing score (the model improving) must surface as the latest value, not a
    running mean — the constant-value determinism test above cannot see a missing
    reset, so an incrementing scorer exercises it directly.
    """

    class _IncrementingFidelity:
        def __init__(self):
            self.n = 0

        def __call__(self, generated, real):
            self.n += 1
            return PairedFidelityScores(psnr=float(self.n), ssim=0.5)

    module = _StubModule(sample_fn=lambda noise, *a: noise)
    cb = PairedFidelityCallback(
        module=module, vae=_ToyVAE(), paired_data=_TinyPairedDataset(),
        num_inference_steps=3, fidelity=_IncrementingFidelity(),
    )
    cb.on_validation_epoch_end(_trainer(epoch=0), module)
    cb.on_validation_epoch_end(_trainer(epoch=1), module)
    # Two epochs ran; the second logged value is 2.0, NOT the accumulated mean 1.5.
    assert _logged_scalars(module)["val/psnr"] == 2.0


def test_generated_matching_real_gives_unit_volumes_and_maximal_psnr():
    """When generated ≈ real, the wiring yields [0,1] volumes and PSNR → +inf, SSIM → 1.

    A recording scorer (the injected fidelity seam) captures the decoded + normalized
    volumes so the [0,1] contract is asserted on the wiring, then delegates to the real
    metric so the +inf / 1.0 ceiling is the genuine MONAI result.
    """

    class _SpyFidelity:
        def __init__(self):
            self.seen = []

        def __call__(self, generated, real):
            self.seen.append((generated, real))
            return PairedFidelityMetrics()(generated, real)

    spy = _SpyFidelity()
    module = _StubModule(sample_fn=lambda noise, *a: None)  # replaced below
    cb = PairedFidelityCallback(
        module=module, vae=_ToyVAE(), paired_data=_TinyPairedDataset(),
        num_inference_steps=3, fidelity=spy,
    )
    # Stub sample() to return the fixed subset's real target latent (generated == real).
    real_tgt = cb._fixed_subset()["tgt_latent"]
    module._sample_fn = lambda *a: real_tgt

    cb.on_validation_epoch_end(_trainer(epoch=0), module)

    assert len(spy.seen) == 1
    generated_vol, real_vol = spy.seen[0]
    for vol in (generated_vol, real_vol):
        assert vol.min().item() >= -1e-5 and vol.max().item() <= 1.0 + 1e-5, "not [0,1]"
    scalars = _logged_scalars(module)
    assert math.isinf(scalars["val/psnr"]), "identical volumes must give +inf PSNR"
    assert scalars["val/ssim"] == pytest.approx(1.0, abs=1e-6)


def test_shape_mismatch_surfaces_paired_metric_error():
    """A generated/real shape mismatch surfaces the paired-fidelity metric's error."""
    module = _StubModule(sample_fn=lambda *a: torch.randn(2, *LATENT_SHAPE[:1], 6, 6, 6))
    cb = PairedFidelityCallback(
        module=module, vae=_ToyVAE(), paired_data=_TinyPairedDataset(), num_inference_steps=3
    )
    with pytest.raises(ValueError, match="matching shapes"):
        cb.on_validation_epoch_end(_trainer(epoch=0), module)


def test_observe_only_contract():
    """Observe-only: generation runs under inference_mode (no grads), and the callback
    logs only the two monitor metrics — it never touches the optimizer / EMA / loss."""
    module = _StubModule(sample_fn=lambda noise, *a: noise)
    cb = PairedFidelityCallback(
        module=module, vae=_ToyVAE(), paired_data=_TinyPairedDataset(), num_inference_steps=3
    )
    cb.on_validation_epoch_end(_trainer(epoch=0), module)

    assert module.sample_calls[0]["inference_mode"] is True, "generation formed grads"
    assert set(module.logged) <= {"val/psnr", "val/ssim"}, "logged a non-monitor key"
    # The two monitor metrics are declared as *validatable* (a future opt-in), without
    # the callback claiming checkpoint selection — that stays val/x0_mae (a wiring fact).
    assert cb.logged_metrics == frozenset({"val/psnr", "val/ssim"})
