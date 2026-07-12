"""Paired-JiT reward model tests (issues #93/#94/#95).

External-behavior seams (per the PRD #92 testing plan):

- **``partial_paired_rollout`` parity/correctness** (the Q7 gate, ADR-0023): the
  ``(B,)``-timestep-through-paired-summed-label combination is novel - gated by a
  scalar-`t` vs ``(B,)``-`t` parity test. Plus ``z`` at ``t_start = add_noise(x_tgt,
  x_src, t_start)`` and winner = higher-`t` (closer to real tgt).
- **``PairedRewardModule``**: BT loss descends over fits on precomputed
  condition-aware ``[2C]`` pairs; ``r(real_tgt) > r(generated_tgt)`` after a few
  fits; the three metrics log; the optimizer covers discriminator params only; the
  Module holds NO generator.
- **CLI smoke** (the primary seam): the full ``manifold-train-paired-reward`` path
  runs on a fake generator + toy pairs, writes a checkpoint, and the fake-cache
  rebuild is byte-identical (determinism).
- **Condition-aware concat** reaches the discriminator (the source channels are
  seen).
- **Export-bridge round-trip** + **``load_frozen_paired_generator``** (#94).
- **Real fake-cache builder + 2-way split** (#95).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from manifold import (
    FlowMatchHeunDiscreteScheduler,
    PartialFlowMatchHeunScheduler,
    RewardModel,
)
from manifold.modules import PairedRewardModule
from manifold.modules.paired_sampler import partial_paired_rollout, sample_paired_latent_flow
from manifold.training.paired_reward_cli import PairedRewardInputs

#: Latent channel count (C_latent); the paired reward scores ``2·C`` concat input.
C_LATENT = 4
#: A tiny latent shape that survives the PatchGAN strided convs on CPU
#: (num_layers_d=1: initial_conv 8->4, final_conv 4->2).
_LAT = (C_LATENT, 8, 8, 8)


# -- fake paired generators (the smoke's frozen generator stand-ins) ---------


class _IdentityPairedGen(nn.Module):
    """A paired generator that predicts the z half unchanged (x0_pred = z) - no movement.

    The paired UNet takes ``concat([z, x_src])`` (``2·C`` channels) and returns the
    predicted target (``C`` channels); an identity generator returns the ``z`` half
    unchanged. Carries a dummy parameter so it mimics a real module (the rollout
    reads the device off ``next(unet.parameters())``).
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(0))

    def forward(
        self, sample, timestep, spacing, class_labels_src=None, class_labels_tgt=None, **kw
    ):
        return sample[:, :C_LATENT]


class _SoftPairedGen(nn.Module):
    """A NON-identity paired generator: ``x0 = 0.5·x_tgt + 0.5·z`` (pulls toward tgt).

    Pulls toward the real target at every eval (non-zero Heun velocities), so the
    rollout provably moves the latent toward ``x_tgt`` - used to assert the
    winner = higher-``t`` ordering and the descent of ``r(real) > r(generated)``.
    Returns the predicted target (``C`` channels) from the ``concat([z, x_src])``
    input.
    """

    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(0))
        self.target = target

    def forward(
        self, sample, timestep, spacing, class_labels_src=None, class_labels_tgt=None, **kw
    ):
        return 0.5 * self.target + 0.5 * sample[:, :C_LATENT]


# -- partial_paired_rollout: the parity gate (ADR-0023) ----------------------


def test_partial_paired_rollout_parity_scalar_t_vs_per_sample_t(paired_unet):
    """At ``t_start = 0`` the ``(B,)``-t partial rollout == the scalar-t full rollout.

    The novel-combination gate (ADR-0023): ``sample_paired_latent_flow`` passes only
    scalar ``t``; ``partial_paired_rollout`` passes ``(B,)`` ``t`` through the paired
    summed-label pathway. At ``t_start = 0`` the start (``add_noise(x_tgt, x_src, 0)``
    = ``x_src``) and the per-sample grid (each row = ``linspace(0, 1, n+1)``) both
    degenerate to the full rollout's, so the two must agree bit-for-bit - proving the
    ``(B,)``-timestep + summed-label combination is sound (the fallback is a
    ``num_steps`` axis only if this fails).
    """
    sched = PartialFlowMatchHeunScheduler()
    torch.manual_seed(0)
    x_src = torch.randn(2, *_LAT)
    x_tgt = torch.randn_like(x_src)
    steps = 3
    # Scalar-t full rollout (the published inference contract).
    z_full = sample_paired_latent_flow(
        paired_unet, sched, x_src, [1.0, 1.0, 1.0], 0, 1, num_inference_steps=steps
    )
    # (B,)-t partial rollout at t_start = 0 (start = x_src, grid = full).
    z_partial = partial_paired_rollout(
        paired_unet, sched, x_src, x_tgt, torch.zeros(2), [1.0, 1.0, 1.0], 0, 1, num_steps=steps
    )
    assert torch.equal(z_partial, z_full)


def test_partial_paired_rollout_starts_from_add_noise_of_tgt_src():
    """``z`` at ``t_start`` = ``add_noise(x_tgt, x_src, t_start)`` for an identity generator.

    An identity generator (``x0 = z``) leaves the start latent unchanged (the
    no-movement invariant): the Heun velocities are zero for every ``t``, so the
    output equals ``z_start = add_noise(x_tgt, x_src, t_start)`` = ``t_start·x_tgt +
    (1−t_start)·x_src``. Guards both the startpoint and the transport.
    """
    sched = PartialFlowMatchHeunScheduler()
    gen = _IdentityPairedGen()
    torch.manual_seed(1)
    x_src = torch.randn(3, *_LAT)
    x_tgt = torch.randn_like(x_src)
    t_start = torch.tensor([0.1, 0.3, 0.45])
    out = partial_paired_rollout(
        gen, sched, x_src, x_tgt, t_start, [1.0, 1.0, 1.0], 0, 1, num_steps=4
    )
    z_start = sched.add_noise(x_tgt, x_src, t_start)
    assert torch.equal(out, z_start)


def test_partial_paired_rollout_winner_higher_t_closer_to_real_tgt():
    """Winner = higher ``t``: a higher-``t_start`` output is closer to the real ``x_tgt``.

    Under a soft generator (``x0 = 0.5·x_tgt + 0.5·z``, pulling toward the real
    target), a start near ``x_tgt`` (high ``t_start``) needs less translation -> the
    output ends closer to ``x_tgt`` than a low-``t_start`` start. This is the
    graded-within-fake ranking the probe measures (ADR-0023) and the opposite of the
    full rollout's start-from-``x_src``.
    """
    sched = PartialFlowMatchHeunScheduler()
    torch.manual_seed(0)
    x_src = torch.randn(1, *_LAT)
    x_tgt = torch.randn_like(x_src)
    gen = _SoftPairedGen(x_tgt)

    def dist_to_tgt(t0: float) -> float:
        out = partial_paired_rollout(
            gen, sched, x_src, x_tgt, torch.tensor([t0]), [1.0, 1.0, 1.0], 0, 1, num_steps=4
        )
        return float((out - x_tgt).norm())

    # Higher t_start -> start nearer x_tgt -> output closer to the real target.
    assert dist_to_tgt(0.45) < dist_to_tgt(0.05)


def test_partial_paired_rollout_per_sample_t_does_not_mix_samples():
    """Each sample rolls independently - sample j's output depends only on sample j."""
    sched = PartialFlowMatchHeunScheduler()
    gen = _IdentityPairedGen()
    torch.manual_seed(2)
    x_src = torch.randn(3, *_LAT)
    x_tgt = torch.randn_like(x_src)
    t_start = torch.tensor([0.1, 0.2, 0.3])
    out = partial_paired_rollout(
        gen, sched, x_src, x_tgt, t_start, [1.0, 1.0, 1.0], 0, 1, num_steps=2
    )
    # Identity generator -> out == z_start per sample (no cross-sample mixing).
    z_start = sched.add_noise(x_tgt, x_src, t_start)
    assert out.shape == x_src.shape
    assert torch.equal(out, z_start)


def test_partial_paired_rollout_threads_per_sample_labels_and_spacing():
    """Per-sample ``[B]`` labels + ``[B,3]`` spacing reach the UNet verbatim."""
    sched = PartialFlowMatchHeunScheduler()

    class _Recording(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(0))
            self.seen_src = []
            self.seen_tgt = []
            self.seen_spacing = []

        def forward(
            self, sample, timestep, spacing, class_labels_src=None, class_labels_tgt=None, **kw
        ):
            self.seen_src.append(class_labels_src.detach().cpu().clone())
            self.seen_tgt.append(class_labels_tgt.detach().cpu().clone())
            self.seen_spacing.append(spacing.detach().cpu().clone())
            return sample[:, :C_LATENT]

    rec = _Recording()
    x_src = torch.randn(2, *_LAT)
    x_tgt = torch.randn_like(x_src)
    t_start = torch.tensor([0.2, 0.4])
    src_labels = torch.tensor([0, 2])
    tgt_labels = torch.tensor([1, 3])
    spacing = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    partial_paired_rollout(
        rec, sched, x_src, x_tgt, t_start, spacing, src_labels, tgt_labels, num_steps=2
    )
    # Every UNet eval got the per-sample src/tgt labels and the per-sample spacing.
    assert all(torch.equal(s, torch.tensor([0, 2])) for s in rec.seen_src)
    assert all(torch.equal(t, torch.tensor([1, 3])) for t in rec.seen_tgt)
    assert any(torch.allclose(s, spacing) for s in rec.seen_spacing)


def test_partial_paired_rollout_mismatched_batch_raises():
    """A mismatched ``t_start`` or per-sample spacing raises a clear error."""
    sched = PartialFlowMatchHeunScheduler()
    gen = _IdentityPairedGen()
    x_src = torch.randn(4, *_LAT)
    x_tgt = torch.randn_like(x_src)
    # x_src batch (4) != t_start (3).
    import pytest

    with pytest.raises(ValueError, match="t_start"):
        partial_paired_rollout(
            gen, sched, x_src, x_tgt, torch.zeros(3), [1.0, 1.0, 1.0], 0, 1, num_steps=1
        )
    # per-sample [B,3] spacing rows (3) != batch (4).
    with pytest.raises(ValueError, match="spacing"):
        partial_paired_rollout(
            gen, sched, x_src, x_tgt, torch.zeros(4), torch.zeros(3, 3), 0, 1, num_steps=1
        )


# -- PairedRewardModule ------------------------------------------------------
#
# The Module trains a condition-aware RewardModel (in_channels = 2·C_latent) on
# precomputed ``[2C]`` concat pairs: winner = concat([x_src, real_tgt]), loser =
# concat([x_src, generated_tgt]). Both halves share the src channels - the
# discriminator must rank the real-tgt half above the generated one.

#: RewardModel config for the paired (condition-aware) reward: 2·C_latent channels.
_PAIRED_RM_KW = dict(spatial_dims=3, in_channels=2 * C_LATENT, channels=8, num_layers_d=1)


def _paired_reward_model() -> RewardModel:
    torch.manual_seed(0)
    return RewardModel(**_PAIRED_RM_KW)


class _ToyPairedPairDS(Dataset):
    """Handmade learnable condition-aware pairs: winner = real tgt, loser = corrupted.

    Each item is a ``{winner, loser}`` dict of ``[2·C_latent, ...]`` concat latents
    (``concat([x_src, tgt])``); the src half is shared, the tgt half differs - the
    discriminator learns to rank the real-tgt half above the corrupted one.
    """

    def __init__(self, n: int = 8):
        torch.manual_seed(0)
        self.src = torch.randn(n, *_LAT)
        real_tgt = torch.randn(n, *_LAT)
        self.real_tgt = real_tgt
        gen_tgt = torch.stack([0.3 * real_tgt[i] + 0.7 * torch.randn(*_LAT) for i in range(n)])
        self.gen_tgt = gen_tgt
        self.items = [
            {
                "winner": torch.cat([self.src[i], real_tgt[i]], dim=0),
                "loser": torch.cat([self.src[i], gen_tgt[i]], dim=0),
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _module() -> PairedRewardModule:
    return PairedRewardModule(_paired_reward_model(), lr=1e-2)


def test_paired_module_fit_returns_finite_bt_loss():
    """forward("fit") on precomputed pairs returns a finite BT loss."""
    mod = _module()
    ds = _ToyPairedPairDS(n=2)
    batch = {
        "winner": torch.stack([ds.items[0]["winner"], ds.items[1]["winner"]]),
        "loser": torch.stack([ds.items[0]["loser"], ds.items[1]["loser"]]),
    }
    out = mod.forward(batch, "fit")
    assert "loss" in out
    assert torch.isfinite(out["loss"])


def test_paired_module_backward_updates_discriminator_only():
    """backward populates grads on every discriminator param; no generator is held.

    The Module holds NO generator (ADR-0020) - so ``parameters()`` is the
    discriminator only, and backward touches exactly those (the offline-precompute
    invariant: the optimizer cannot reach a nonexistent generator).
    """
    m = _paired_reward_model()
    mod = PairedRewardModule(m, lr=1e-2)
    ds = _ToyPairedPairDS(n=2)
    batch = {
        "winner": torch.stack([ds.items[0]["winner"], ds.items[1]["winner"]]),
        "loser": torch.stack([ds.items[0]["loser"], ds.items[1]["loser"]]),
    }
    mod.forward(batch, "fit")["loss"].backward()
    params = list(m.parameters())
    assert params, "reward model has parameters"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    # No generator held: the Module's parameters ARE the discriminator's only.
    assert {id(p) for p in mod.parameters()} == {id(p) for p in m.parameters()}
    # The optimizer covers discriminator params only.
    opt_ids = {id(p) for p in mod.configure_optimizers()["optimizer"].param_groups[0]["params"]}
    assert opt_ids == {id(p) for p in m.parameters()}


def test_paired_module_optimizer_step_raises_real_above_generated():
    """A few Adam steps on the BT loss raise r(real_tgt) above r(generated_tgt).

    The load-bearing BT property on the condition-aware pair: after descent the
    discriminator ranks the real-target half above the generated-target half
    (winner > loser), the real-vs-fake signal (ADR-0018).
    """
    torch.manual_seed(0)
    m = _paired_reward_model()
    mod = PairedRewardModule(m, lr=1e-2)
    ds = _ToyPairedPairDS(n=8)
    winner = torch.stack([it["winner"] for it in ds.items])
    loser = torch.stack([it["loser"] for it in ds.items])

    def margin() -> float:
        with torch.no_grad():
            r = m(torch.cat([winner, loser]))
            return float((r[: len(winner)] - r[len(winner) :]).mean())

    before = margin()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        mod.forward({"winner": winner, "loser": loser}, "fit")["loss"].backward()
        opt.step()
    after = margin()
    assert after > before


def test_paired_module_forward_stage_mismatch_raises():
    """A batch missing winner/loser, or an unknown stage, raises clearly."""
    mod = _module()
    import pytest

    with pytest.raises(ValueError, match="winner"):
        mod.forward({"foo": torch.randn(1, 2 * C_LATENT, 8, 8, 8)}, "fit")
    with pytest.raises(ValueError, match="stage"):
        mod.forward(
            {
                "winner": torch.randn(1, 2 * C_LATENT, 8, 8, 8),
                "loser": torch.randn(1, 2 * C_LATENT, 8, 8, 8),
            },
            "test",
        )


# -- offline fake-cache + probe builders (determinism + condition-aware) -----


def _toy_src_tgt(n: int = 8):
    """Toy scaled src/tgt latents + a fake paired generator (pulls toward tgt)."""
    torch.manual_seed(0)
    x_src = torch.randn(n, *_LAT)
    x_tgt = torch.randn(n, *_LAT)
    gen = _SoftPairedGen(x_tgt)  # x0 = 0.5·x_tgt + 0.5·z -> gen_tgt is a real-ish fake
    return x_src, x_tgt, gen


def test_build_paired_reward_pairs_emits_condition_aware_concat():
    """winner = cat([x_src, real_tgt]); loser = cat([x_src, gen_tgt]); src half shared."""
    from manifold.data.paired_reward_pairs import build_paired_reward_pairs

    x_src, x_tgt, gen = _toy_src_tgt(n=4)
    ds = build_paired_reward_pairs(
        x_src,
        x_tgt,
        gen,
        FlowMatchHeunDiscreteScheduler(),
        src_label=0,
        tgt_label=1,
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=4,
        device="cpu",
    )
    assert ds.winners.shape == (4, 2 * C_LATENT, *_LAT[1:])
    assert ds.losers.shape == ds.winners.shape
    # The src half (first C channels) is shared between winner and loser.
    assert torch.equal(ds.winners[:, :C_LATENT], ds.losers[:, :C_LATENT])
    # The winner's tgt half IS the real target; the loser's is the generated fake
    # (pulled toward x_tgt but not equal under the soft generator).
    assert torch.equal(ds.winners[:, C_LATENT:], x_tgt)
    assert not torch.equal(ds.losers[:, C_LATENT:], x_tgt)
    assert torch.isfinite(ds.winners).all() and torch.isfinite(ds.losers).all()


def test_build_paired_reward_pairs_is_deterministic():
    """Re-building the fake cache yields byte-identical fakes (determinism, ADR-0020).

    The paired rollout is deterministic given ``x_src`` (no stochastic input), so
    re-running the builder produces byte-identical pairs - the property that lets
    the fakes be cached once (offline precompute) instead of re-rolled each epoch.
    """
    from manifold.data.paired_reward_pairs import build_paired_reward_pairs

    x_src, x_tgt, gen = _toy_src_tgt(n=4)
    kw = dict(
        generator=gen,
        scheduler=FlowMatchHeunDiscreteScheduler(),
        src_label=0,
        tgt_label=1,
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=4,
        device="cpu",
    )
    a = build_paired_reward_pairs(x_src, x_tgt, **kw)
    b = build_paired_reward_pairs(x_src, x_tgt, **kw)
    assert torch.equal(a.winners, b.winners)
    assert torch.equal(a.losers, b.losers)


def test_build_paired_reward_probe_is_deterministic_and_concat():
    """Re-building the probe with the same seed yields byte-identical pairs; cat([x_src, gen])."""
    from manifold.data.paired_reward_pairs import build_paired_reward_probe

    x_src, x_tgt, gen = _toy_src_tgt(n=4)
    kw = dict(
        generator=gen,
        partial_scheduler=PartialFlowMatchHeunScheduler(),
        src_label=0,
        tgt_label=1,
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=4,
        seed=0,
        device="cpu",
    )
    a = build_paired_reward_probe(x_src, x_tgt, **kw)
    b = build_paired_reward_probe(x_src, x_tgt, **kw)
    assert torch.equal(a.winners, b.winners)
    assert torch.equal(a.losers, b.losers)
    # Both halves are condition-aware [2·C] concat latents (src half shared).
    assert a.winners.shape == (4, 2 * C_LATENT, *_LAT[1:])
    assert torch.equal(a.winners[:, :C_LATENT], a.losers[:, :C_LATENT])


def test_condition_aware_concat_reaches_discriminator():
    """The discriminator sees the [2·C] concat - the source channels are in view (ADR-0019).

    A recording RewardModel captures its input; forwarding a pair shows the
    discriminator received ``[2B, 2·C_latent, ...]`` (the concat reached it), with
    the src half shared across the winner/loser - so the scorer judges the tgt
    *given* the src (it cannot score the tgt alone).
    """

    class _Recording(RewardModel):
        def __init__(self):
            super().__init__(**_PAIRED_RM_KW)
            self.seen = None

        def forward(self, latent):
            self.seen = latent.detach().clone()
            return super().forward(latent)

    m = _Recording()
    mod = PairedRewardModule(m, lr=1e-2)
    winner = torch.randn(2, 2 * C_LATENT, 8, 8, 8)
    loser = torch.randn(2, 2 * C_LATENT, 8, 8, 8)
    mod.forward({"winner": winner, "loser": loser}, "fit")
    # The discriminator saw the full [2B, 2·C, ...] concat (src channels in view).
    assert m.seen.shape == (4, 2 * C_LATENT, 8, 8, 8)


def test_build_paired_reward_pairs_accepts_scalar_zero_d_tensor_labels():
    """A 0-d tensor label (``torch.tensor(0)``) is normalized to an int (codex P3).

    Slicing a 0-d tensor would crash; the builder reduces scalar / 0-d labels to an
    int once before the batch loop, so ``torch.tensor(0)`` broadcasts like ``0``.
    """
    from manifold.data.paired_reward_pairs import build_paired_reward_pairs

    x_src, x_tgt, gen = _toy_src_tgt(n=4)
    ds = build_paired_reward_pairs(
        x_src,
        x_tgt,
        gen,
        FlowMatchHeunDiscreteScheduler(),
        src_label=torch.tensor(0),
        tgt_label=torch.tensor(1),
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=4,
        device="cpu",
    )
    assert ds.winners.shape == (4, 2 * C_LATENT, *_LAT[1:])
    assert torch.isfinite(ds.winners).all()


def test_build_paired_reward_pairs_threads_per_sample_labels():
    """A length-N per-sample label tensor is sliced per batch and reaches the rollout."""
    from manifold.data.paired_reward_pairs import build_paired_reward_pairs

    class _Recording(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(0))
            self.seen = []

        def forward(
            self, sample, timestep, spacing, class_labels_src=None, class_labels_tgt=None, **kw
        ):
            self.seen.append(
                (class_labels_src.detach().cpu().clone(), class_labels_tgt.detach().cpu().clone())
            )
            return sample[:, :C_LATENT]

    torch.manual_seed(0)
    x_src = torch.randn(4, *_LAT)
    x_tgt = torch.randn(4, *_LAT)
    rec = _Recording()
    build_paired_reward_pairs(
        x_src,
        x_tgt,
        rec,
        FlowMatchHeunDiscreteScheduler(),
        src_label=torch.tensor([0, 1, 2, 3]),
        tgt_label=torch.tensor([3, 2, 1, 0]),
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=2,
        device="cpu",
    )
    seen_src = torch.stack([s[0] for s in rec.seen])
    seen_tgt = torch.stack([s[1] for s in rec.seen])
    # Each batch's labels are a slice of the per-sample tensors (order-independent).
    assert torch.equal(seen_src.unique().sort().values, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(seen_tgt.unique().sort().values, torch.tensor([0, 1, 2, 3]))


# -- CLI smoke (the end-to-end seam) -----------------------------------------


def _smoke_inputs() -> PairedRewardInputs:
    """The injection-seam bundle: fake generator + toy pairs/probe via the builders.

    Uses an identity paired generator (``x0 = z``) -> the generated tgt is the src
    itself (copy-src). The condition-aware reward must rank the real tgt above
    copy-src - the documented dominant paired failure (ADR-0019) - so this is the
    most meaningful wiring smoke. Identity is batch-size-agnostic (no fixed target)
    and deterministic (ADR-0020).
    """
    from manifold.data.paired_reward_pairs import (
        build_paired_reward_pairs,
        build_paired_reward_probe,
    )

    torch.manual_seed(0)
    x_src = torch.randn(8, *_LAT)
    x_tgt = torch.randn(8, *_LAT)
    gen = _IdentityPairedGen()  # x0 = z -> gen_tgt = copy-src (the fake)
    train = build_paired_reward_pairs(
        x_src,
        x_tgt,
        gen,
        FlowMatchHeunDiscreteScheduler(),
        src_label=0,
        tgt_label=1,
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=4,
        device="cpu",
    )
    val = build_paired_reward_pairs(
        x_src[:4],
        x_tgt[:4],
        gen,
        FlowMatchHeunDiscreteScheduler(),
        src_label=0,
        tgt_label=1,
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=4,
        device="cpu",
    )
    probe = build_paired_reward_probe(
        x_src[:4],
        x_tgt[:4],
        gen,
        PartialFlowMatchHeunScheduler(),
        src_label=0,
        tgt_label=1,
        spacing=[1.0, 1.0, 1.0],
        num_steps=2,
        batch_size=4,
        seed=0,
        device="cpu",
    )
    return PairedRewardInputs(train_pair_ds=train, val_pair_ds=val, val_probe=probe)


def _run(tmp_path, **kw):
    from manifold.training import run_paired_reward_training

    return run_paired_reward_training(
        module=_module(),
        inputs=_smoke_inputs(),
        model_dir=str(tmp_path),
        max_epochs=2,
        devices=1,
        accelerator="cpu",
        batch_size=2,
        num_workers=0,
        limit_val_batches=1.0,
        **kw,
    )


def test_run_paired_reward_training_writes_ckpt_and_logs_metrics(tmp_path):
    """run_paired_reward_training fits, logs all three metrics, writes a checkpoint."""
    from pathlib import Path

    trainer, ckpt = _run(tmp_path)
    metrics = trainer.callback_metrics
    for key in ("val/pair_acc", "val/roc_auc", "val/gen_pair_acc"):
        assert key in metrics, f"missing {key}"
        assert torch.isfinite(metrics[key])
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()


def test_run_paired_reward_training_resumes_from_ckpt(tmp_path):
    """A checkpoint writes and resumes (Lightning ckpt_path)."""
    trainer, ckpt = _run(tmp_path)
    assert ckpt.last_model_path and Path(ckpt.last_model_path).is_file()
    # Resume from the written last.ckpt (a second fit picks up the checkpoint).
    _run(tmp_path, ckpt_path=ckpt.last_model_path)


_PAIRED_NETWORK_YAML = """\
spatial_dims: 3
latent_channels: 4
reward_model:
  spatial_dims: ${spatial_dims}
  in_channels: 8   # 2·C_latent (condition-aware concat, ADR-0019)
  channels: 8
  num_layers_d: 1
  norm: BATCH
"""


def _write_paired_tiny_configs(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(_PAIRED_NETWORK_YAML)
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model") + "val_subset_size: 4\n"
    )
    train = tmp_path / "train.yaml"
    train.write_text(
        "paired_reward_train: {batch_size: 2, lr: 1.0e-2, n_epochs: 2}\n"
        "paired_reward: {num_steps: 2, val_fraction: 0.34}\n"
    )
    return str(env), str(train), str(net)


def test_paired_reward_main_runs_end_to_end_with_fake_data(tmp_path):
    """main(): argparse -> compose -> build -> fit -> ckpt (the fake-data seam)."""
    from pathlib import Path

    from manifold.training.paired_reward_cli import main as paired_reward_main

    env, train, net = _write_paired_tiny_configs(tmp_path)

    def fake_provider(cfg, device):
        return _smoke_inputs()

    rc = paired_reward_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "2"],
        data_provider=fake_provider,
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_paired_reward_main_native_dir_latents_dir_validated(tmp_path):
    """--native-dir/--latents-dir default None and are validated only without a data_provider."""
    import pytest

    from manifold.training.paired_reward_cli import main as paired_reward_main

    env, train, net = _write_paired_tiny_configs(tmp_path)
    # No data_provider AND no --native-dir/--latents-dir -> clear error (not a crash).
    with pytest.raises(ValueError, match="native-dir"):
        paired_reward_main(["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"])
    # With a data_provider, the missing args are NOT required (smoke seam intact).
    rc = paired_reward_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=lambda cfg, device: _smoke_inputs(),
    )
    assert rc == 0


def test_paired_reward_main_forces_2c_in_channels_regardless_of_config(tmp_path):
    """main() forces RewardModel in_channels = 2·C_latent even when the network
    config's reward_model.in_channels is the JiT default (latent_channels).

    Regression for codex #96 (P2) / #99 (P1): the ``opt(..., 2*C)`` fallback was
    dead because the network config carried ``${latent_channels}`` (=4), so the
    reward model would be built with 4 channels and crash on the first 8-channel
    concat batch. The paired reward's in_channels is structural (2·C), not config-driven.
    """
    from unittest.mock import patch

    from manifold.training.paired_reward_cli import main as paired_reward_main

    # A network config with the WRONG (JiT-style) reward_model.in_channels = C (4),
    # not 2·C. main() must ignore it and force 2·C_latent.
    net = tmp_path / "network.yaml"
    net.write_text(
        "spatial_dims: 3\nlatent_channels: 4\n"
        "reward_model:\n  spatial_dims: ${spatial_dims}\n  in_channels: 4\n"
        "  channels: 8\n  num_layers_d: 1\n  norm: BATCH\n"
    )
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\nmodel_dir: %s\nval_subset_size: 4\n" % (tmp_path / "model")
    )
    train = tmp_path / "train.yaml"
    train.write_text("paired_reward_train: {batch_size: 2, lr: 1.0e-2, n_epochs: 1}\n")

    seen = {}
    real_init = RewardModel.__init__

    def spy(self, *a, **kw):
        seen["in_channels"] = kw.get("in_channels")
        return real_init(self, *a, **kw)

    with patch.object(RewardModel, "__init__", spy):
        rc = paired_reward_main(
            ["-e", str(env), "-c", str(train), "-t", str(net), "-g", "1", "--max-epochs", "1"],
            data_provider=lambda cfg, device: _smoke_inputs(),
        )
    assert rc == 0
    assert seen["in_channels"] == 2 * 4, (
        "paired reward model must be 2·C_latent regardless of reward_model.in_channels"
    )


def test_paired_reward_build_checkpoint_monitors_gen_pair_acc_and_ddp_fallback():
    """_build_checkpoint monitors val/gen_pair_acc (single-GPU) and drops it under DDP."""
    from pathlib import Path

    from manifold.training.reward_cli import _build_checkpoint

    single = _build_checkpoint(str(Path("/tmp/_prw_ckpt_a").resolve()), multi_gpu=False)
    multi = _build_checkpoint(str(Path("/tmp/_prw_ckpt_b").resolve()), multi_gpu=True)
    assert single.monitor == "val/gen_pair_acc"  # the within-fake-ranking probe metric
    assert multi.monitor is None  # DDP: no rank-0-shard selection
    assert multi.save_last and multi.save_top_k == 1
    import shutil

    shutil.rmtree("/tmp/_prw_ckpt_a", ignore_errors=True)
    shutil.rmtree("/tmp/_prw_ckpt_b", ignore_errors=True)


def test_run_paired_reward_training_fails_fast_without_probe(tmp_path):
    """Monitoring val/gen_pair_acc without a probe raises clearly (codex P1).

    The probe is mandatory (ADR-0023); the checkpoint monitors ``val/gen_pair_acc``,
    which the Module only logs when a probe is attached. Without one Lightning would
    raise an opaque MisconfigurationException at fit - fail fast instead.
    """
    import pytest

    from manifold.training import run_paired_reward_training
    from manifold.training.paired_reward_cli import PairedRewardInputs

    # A toy pair dataset (val), but NO probe.
    ds = _ToyPairedPairDS(n=4)
    inputs = PairedRewardInputs(train_pair_ds=ds, val_pair_ds=ds, val_probe=None)
    with pytest.raises(ValueError, match="probe"):
        run_paired_reward_training(
            module=_module(),
            inputs=inputs,
            model_dir=str(tmp_path),
            max_epochs=1,
            devices=1,
            accelerator="cpu",
            batch_size=2,
            limit_val_batches=1.0,
        )
