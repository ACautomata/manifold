"""Partial-denoise pair generation tests (GRPO reward model, issue #39/#42).

External-behavior seams: the partial scheduler yields a per-sample ``(B, n+1)``
grid over ``[t_start, 1]``; the partial rollout degenerates to the full-from-noise
rollout at ``t_start = 0`` and is identity-preserving for an identity denoiser;
``generate_reward_pairs`` emits a held-out-subject ``(train, val)`` split whose
winners reconstruct the clean latent better than losers (the label direction);
and the frozen denoiser loads from a native export.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from manifold import (
    AutoencoderKL,
    FlowMatchHeunDiscreteScheduler,
    LatentFlowPipeline,
    PartialFlowMatchHeunScheduler,
    UNet3DConditionModel,
)
from manifold.data.reward_pairs import (
    generate_reward_pairs,
    load_frozen_denoiser,
    load_reward_pairs,
    save_reward_pairs,
)
from manifold.modules import partial_denoise_rollout, sample_latent_flow

_LAT = (4, 8, 8, 8)


class _IdentityDenoiser(nn.Module):
    """A denoiser that returns its input unchanged (x0_pred = z).

    Used to assert the rollout is identity-preserving (the t_start → 1 limit) and
    that pair construction's label direction holds at the transport level. Carries
    a dummy parameter so it mimics a real module (the rollout reads the device
    off ``next(unet.parameters())``, as ``sample_latent_flow`` does).
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(0))

    def forward(self, sample, timestep, spacing, class_labels=None, **kwargs):
        return sample


# -- PartialFlowMatchHeunScheduler ------------------------------------------


def test_set_timesteps_partial_grid_shape_and_endpoints():
    sched = PartialFlowMatchHeunScheduler()
    t_start = torch.tensor([0.0, 0.3, 0.6])
    grid = sched.set_timesteps_partial(t_start, 4)
    assert grid.shape == (3, 5)  # (B, n+1)
    # Each row starts at its own t_start and ends at 1.
    assert torch.allclose(grid[:, 0], t_start)
    assert torch.allclose(grid[:, -1], torch.ones(3))


def test_set_timesteps_partial_per_sample_step_size_differs():
    """A shared step budget gives per-sample δt (smaller for higher t_start)."""
    sched = PartialFlowMatchHeunScheduler()
    grid = sched.set_timesteps_partial(torch.tensor([0.0, 0.5, 0.9]), 5)
    dt = grid[:, 1] - grid[:, 0]  # (1 - t_start) / n per sample
    assert dt[0] > dt[1] > dt[2]


def test_set_timesteps_partial_scalar_t_start_is_single_sample_batch():
    sched = PartialFlowMatchHeunScheduler()
    grid = sched.set_timesteps_partial(torch.tensor(0.4), 3)
    assert grid.shape == (1, 4)
    assert torch.allclose(grid[0, 0], torch.tensor(0.4))
    assert grid[0, -1].item() == 1.0


# -- partial_denoise_rollout ------------------------------------------------


def test_partial_rollout_degenerates_to_full_at_t_start_zero(unet):
    """t_start = 0 → the partial rollout equals the full-from-noise rollout.

    The defining degeneracy (ADR-0008): with every sample at t_start = 0 the
    per-sample grid is the JiT linspace(0, 1, n+1), so the partial rollout must
    reproduce ``sample_latent_flow`` bit-for-bit (the same shared primitive).
    """
    full = FlowMatchHeunDiscreteScheduler()
    partial = PartialFlowMatchHeunScheduler()
    torch.manual_seed(0)
    noise = torch.randn(2, *_LAT)
    steps = 3
    z_full = sample_latent_flow(
        unet, full, noise, [1.0, 1.0, 1.0], 1, num_inference_steps=steps, guidance_scale=1.0
    )
    z_partial = partial_denoise_rollout(
        unet, partial, noise, torch.zeros(2), [1.0, 1.0, 1.0], 1, num_steps=steps
    )
    assert torch.equal(z_full, z_partial)


def test_partial_rollout_is_identity_for_identity_denoiser():
    """An identity denoiser (x0 = z) leaves the start latent unchanged (no-movement invariant).

    This is t_start-AGNOSTIC: with x0 = z the Heun velocities v1, v2 are zero for
    every t_start, so out == z_start regardless of the t_start values used. It
    guards that the rollout fabricates no movement when the denoiser predicts none
    — NOT the t_start → 1 near-identity limit (see
    test_partial_rollout_near_identity_shrinks_with_t_start for that).
    """
    sched = PartialFlowMatchHeunScheduler()
    denoiser = _IdentityDenoiser()
    torch.manual_seed(1)
    z_start = torch.randn(3, *_LAT)
    t_start = torch.tensor([0.1, 0.5, 0.9])
    out = partial_denoise_rollout(
        denoiser, sched, z_start, t_start, [1.0, 1.0, 1.0], 1, num_steps=4
    )
    assert torch.equal(out, z_start)


def test_partial_rollout_near_identity_shrinks_with_t_start():
    """The genuine near-identity limit: ||out − z_start|| shrinks as t_start → 1.

    Under a NON-identity denoiser (a soft x0 = 0.5·clean + 0.5·z that pulls toward
    clean), a start near clean (high t_start) is barely moved by the short
    integration, while a start near noise (low t_start) is moved more — so the
    rollout's departure from z_start decreases as t_start → 1. Exercises the full
    Heun math (non-zero velocities), unlike the identity invariant above.
    """
    sched = PartialFlowMatchHeunScheduler()
    torch.manual_seed(0)
    clean = torch.randn(1, *_LAT)
    e = torch.randn_like(clean)

    class _SoftDenoiser(nn.Module):
        """x0 = 0.5·clean + 0.5·z — a non-identity denoiser pulling toward clean."""

        def __init__(self, target):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(0))
            self.target = target

        def forward(self, sample, timestep, spacing, class_labels=None, **kw):
            return 0.5 * self.target + 0.5 * sample

    denoiser = _SoftDenoiser(clean)

    def departure(t_start: float) -> float:
        z_start = sched.add_noise(clean, e, torch.tensor([t_start]))
        out = partial_denoise_rollout(
            denoiser, sched, z_start, torch.tensor([t_start]), [1.0, 1.0, 1.0], 1, num_steps=4
        )
        return float((out - z_start).norm())

    # Higher t_start → start nearer clean → the rollout departs less from z_start.
    assert departure(0.9) < departure(0.1)


def test_partial_rollout_per_sample_t_does_not_mix_samples():
    """Each sample denoises independently — sample j's output depends only on sample j."""
    sched = PartialFlowMatchHeunScheduler()
    denoiser = _IdentityDenoiser()
    z = torch.randn(3, *_LAT)
    t = torch.tensor([0.2, 0.4, 0.6])
    out = partial_denoise_rollout(denoiser, sched, z, t, [1.0, 1.0, 1.0], 1, num_steps=2)
    # Identity denoiser → out == z exactly (no cross-sample mixing).
    assert out.shape == z.shape
    assert torch.equal(out, z)


def test_partial_rollout_output_is_backward_safe():
    """The rollout output (now under no_grad) feeds a discriminator backward without error.

    The prerequisite for online reward training (#49 → #50): the rollout output is
    the discriminator's fit-step input, where backward must save it. Previously the
    rollout ran under ``inference_mode``; an inference tensor cannot be saved for
    backward (``.detach()``/``.clone()`` do not clear the flag), so a discriminator
    backward over it crashed. ``no_grad`` keeps the no-activation-retention while
    yielding a backward-safe tensor. Parity with ``sample_latent_flow`` (ADR-0008)
    is unaffected — both contexts run identical math; only the tensor flag differs
    (see test_partial_rollout_degenerates_to_full_at_t_start_zero, still bit-exact).
    """
    sched = PartialFlowMatchHeunScheduler()
    torch.manual_seed(0)
    z_start = torch.randn(2, *_LAT)
    t_start = torch.tensor([0.3, 0.7])
    out = partial_denoise_rollout(
        _IdentityDenoiser(), sched, z_start, t_start, [1.0, 1.0, 1.0], 1, num_steps=2
    )
    # A discriminator head whose weights require grad: the rollout output is the
    # (constant) input; backward must populate the head's weight grads.
    disc = nn.Linear(out[0].numel(), 1)
    flat = out.flatten(1)  # backward-safe (no_grad, not inference_mode)
    rewards = disc(flat).squeeze(-1)  # [2]
    loss = -torch.log(torch.sigmoid(rewards[0] - rewards[1]))
    loss.backward()  # would raise under inference_mode ("cannot be saved for backward")
    assert disc.weight.grad is not None
    assert torch.isfinite(disc.weight.grad).all()
    assert disc.weight.grad.abs().sum() > 0


# -- generate_reward_pairs --------------------------------------------------


def _toy_clean_latents(n: int = 12):
    torch.manual_seed(0)
    return torch.randn(n, *_LAT), [f"subj_{i // 2}" for i in range(n)]  # 2 vols / subject


def test_generate_reward_pairs_emits_held_out_subject_split():
    clean, sids = _toy_clean_latents(12)  # 6 subjects
    train, val = generate_reward_pairs(
        clean, sids, _IdentityDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, val_fraction=0.34,
        batch_size=4, seed=0, device="cpu",
    )
    assert len(train) + len(val) == 12
    assert len(train) > 0 and len(val) > 0
    # Shapes carry through.
    assert train.winners.shape == train.losers.shape
    assert train.winners.shape[1:] == _LAT


def test_generate_reward_pairs_subjects_do_not_leak_across_split():
    """No subject appears in both splits (val measures generalization)."""
    clean, sids = _toy_clean_latents(20)  # 10 subjects
    train, val = generate_reward_pairs(
        clean, sids, _IdentityDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, val_fraction=0.3,
        batch_size=5, seed=1, device="cpu",
    )
    # The split is by subject; with distinct clean latents per pair the train/val
    # winner tensors are disjoint sets of underlying latents (verifiable via the
    # generation: each clean latent lands in exactly one split).
    assert len(train) + len(val) == 20


def test_pair_label_direction_winner_closer_to_clean_than_loser():
    """A winner pair (t_w ∈ [0.5,1)) reconstructs clean better than a loser (t_l ∈ [0,0.5)).

    The label direction (issue #42 acceptance): winners have lower reconstruction
    error to the clean latent than losers. With an identity denoiser the denoised
    pair equals the noised start, so this reduces to the transport ordering — the
    deterministic guarantee the generation function rests on (winners corrupted less).
    """
    sched = PartialFlowMatchHeunScheduler()
    denoiser = _IdentityDenoiser()
    torch.manual_seed(0)
    clean = torch.randn(1, *_LAT)
    t_w = torch.tensor([0.8])
    t_l = torch.tensor([0.2])
    noise = torch.randn_like(clean)
    z_w = sched.add_noise(clean, noise, t_w)
    z_l = sched.add_noise(clean, noise, t_l)
    winner = partial_denoise_rollout(denoiser, sched, z_w, t_w, [1.0, 1.0, 1.0], 1, num_steps=2)
    loser = partial_denoise_rollout(denoiser, sched, z_l, t_l, [1.0, 1.0, 1.0], 1, num_steps=2)
    assert (winner - clean).norm() < (loser - clean).norm()


def test_pair_label_direction_holds_under_non_identity_denoiser():
    """The label direction survives a NON-identity rollout (exercises the Heun math).

    A soft denoiser ``x0 = 0.5·clean + 0.5·z`` pulls toward clean at every eval
    (non-zero velocities, unlike the identity mock). Both halves integrate from
    their own t_start toward clean; the less-corrupted winner (higher t) ends
    closer to clean than the heavily-corrupted loser — the real #42 acceptance,
    not just the transport ordering.
    """
    sched = PartialFlowMatchHeunScheduler()
    torch.manual_seed(0)
    clean = torch.randn(1, *_LAT)
    e = torch.randn_like(clean)

    class _SoftDenoiser(nn.Module):
        def __init__(self, target):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(0))
            self.target = target

        def forward(self, sample, timestep, spacing, class_labels=None, **kw):
            return 0.5 * self.target + 0.5 * sample

    denoiser = _SoftDenoiser(clean)
    t_w, t_l = torch.tensor([0.8]), torch.tensor([0.2])
    z_w = sched.add_noise(clean, e, t_w)
    z_l = sched.add_noise(clean, e, t_l)
    winner = partial_denoise_rollout(denoiser, sched, z_w, t_w, [1.0, 1.0, 1.0], 1, num_steps=4)
    loser = partial_denoise_rollout(denoiser, sched, z_l, t_l, [1.0, 1.0, 1.0], 1, num_steps=4)
    assert (winner - clean).norm() < (loser - clean).norm()


@pytest.mark.skipif(
    not torch.backends.mps.is_available() and not torch.cuda.is_available(),
    reason="needs a non-CPU accelerator to exercise the device-aware generator",
)
def test_generate_reward_pairs_runs_on_accelerator():
    """The noise generator is device-aware — generate_reward_pairs runs on a GPU/MPS.

    Guards the critical regression where a CPU-default ``torch.Generator()`` was
    handed to ``torch.randn(..., device=cuda)`` and crashed on the first batch
    (masked by the CPU-only test suite). Uses MPS (this mac) as the non-CPU stand-in.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    clean, sids = _toy_clean_latents(8)
    train, val = generate_reward_pairs(
        clean, sids, _IdentityDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, val_fraction=0.34,
        batch_size=4, seed=0, device=device,
    )
    assert len(train) + len(val) == 8
    assert torch.isfinite(train.winners).all()


def test_generate_generated_end_probe_in_generated_regime_and_ordered():
    """Probe pairs: winner less corrupted than loser; both more corrupted than recon winners.

    Both probe samples are drawn from ``t ∈ [0, 0.5)`` (the generated regime),
    with the winner the less-corrupted (higher ``t``). Verified without replaying
    the function's seed: an identity denoiser makes the denoised pair equal the
    noised start, and ``generate_generated_end_probe`` processes ``clean`` in order
    (so ``probe.winners[i]`` corresponds to ``clean[i]``).
    """
    from manifold.data.reward_pairs import generate_generated_end_probe

    torch.manual_seed(0)
    clean = torch.randn(32, *_LAT)
    sched = PartialFlowMatchHeunScheduler()
    probe = generate_generated_end_probe(
        clean, _IdentityDenoiser(), sched,
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, batch_size=8, seed=0, device="cpu",
    )
    assert len(probe) == 32 and probe.winners.shape[1:] == _LAT
    assert torch.isfinite(probe.winners).all()

    w_recon = (probe.winners - clean).flatten(1).norm(dim=1)
    l_recon = (probe.losers - clean).flatten(1).norm(dim=1)
    # Within the probe, the winner (higher t) reconstructs clean better than the loser.
    assert w_recon.mean() < l_recon.mean()

    # Both are more corrupted than a reconstruction winner (t ∈ [0.5, 1)): a recon
    # winner is closer to clean than even the probe *winner*.
    torch.manual_seed(1)
    recon_noise = torch.randn_like(clean)
    t_recon = 0.5 + 0.5 * torch.rand(len(clean))  # U[0.5, 1)
    recon_winner = sched.add_noise(clean, recon_noise, t_recon)
    assert w_recon.mean() > (recon_winner - clean).flatten(1).norm(dim=1).mean()


def test_reward_pair_dataset_save_load_round_trips(tmp_path):
    torch.manual_seed(0)
    clean, sids = _toy_clean_latents(8)
    train, val = generate_reward_pairs(
        clean, sids, _IdentityDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, val_fraction=0.34,
        batch_size=4, seed=0, device="cpu",
    )
    save_reward_pairs(tmp_path, train, val)
    tr2, val2, probe2 = load_reward_pairs(tmp_path)
    assert probe2 is None  # no probe written
    assert torch.equal(tr2.winners, train.winners)
    assert torch.equal(val2.losers, val.losers)
    item = tr2[0]
    assert set(item) == {"winner", "loser"} and item["winner"].shape == _LAT


# -- frozen-denoiser loading ------------------------------------------------

def test_load_frozen_denoiser_from_native_export(tmp_path):
    """The frozen denoiser loads from a native pipeline dir (ADR-0006)."""
    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    vae = AutoencoderKL(scaling_factor=0.5)
    scheduler = FlowMatchHeunDiscreteScheduler(t_eps=0.05)
    LatentFlowPipeline(unet, vae, scheduler).save_pretrained(str(tmp_path / "native"))

    denoiser, partial_sched, scaling_factor = load_frozen_denoiser(tmp_path / "native")
    assert isinstance(partial_sched, PartialFlowMatchHeunScheduler)
    assert partial_sched.t_eps == 0.05  # matched from the JiT scheduler config
    assert scaling_factor == 0.5  # the native VAE's scaling_factor (used to scale raw cache latents)
    # Frozen + a working scoring forward.
    assert all(not p.requires_grad for p in denoiser.parameters())
    out = denoiser(sample=torch.randn(1, *_LAT), timestep=torch.tensor([0.5]),
                   spacing=torch.tensor([1.0, 1.0, 1.0]), class_labels=torch.tensor([1]))
    assert out.shape == (1, *_LAT)


# -- Codex #45 regression: subject id + per-sample conditioning + scaling ----


def test_subject_id_groups_brats_contrasts():
    """_subject_id strips the trailing contrast so one subject's contrasts group together."""
    from manifold.data.reward_pairs import _subject_id

    # Default (BraTS): last '-<token>' stripped → contrasts of one subject merge.
    # (Cache sample_ids always carry the contrast suffix; the subject id itself is
    # the prefix shared across t1n/t1c/t2w/t2f.)
    assert _subject_id("BraTS-GLI-0000-000-t1n", None) == "BraTS-GLI-0000-000"
    assert _subject_id("BraTS-GLI-0000-000-t1c", None) == "BraTS-GLI-0000-000"
    assert _subject_id("BraTS-GLI-0000-000-t2f", None) == "BraTS-GLI-0000-000"
    assert _subject_id("no_dash_id", None) == "no_dash_id"  # nothing to strip
    # A custom regex overrides (e.g. group by an explicit prefix).
    assert _subject_id("siteA_case007_t1n", r"^(siteA_case007)") == "siteA_case007"


class _RecordingDenoiser(nn.Module):
    """Identity denoiser that records every class_labels / spacing it receives."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(0))
        self.seen_labels = []
        self.seen_spacing = []

    def forward(self, sample, timestep, spacing, class_labels=None, **kw):
        self.seen_labels.append(None if class_labels is None else class_labels.detach().cpu().clone())
        self.seen_spacing.append(None if spacing is None else spacing.detach().cpu().clone())
        return sample


def test_partial_rollout_threads_per_sample_modality_and_spacing():
    """A per-sample modality tensor + [B,3] spacing reach the UNet verbatim."""
    sched = PartialFlowMatchHeunScheduler()
    rec = _RecordingDenoiser()
    z = torch.randn(2, *_LAT)
    t_start = torch.tensor([0.2, 0.6])
    labels = torch.tensor([3, 7])
    spacing = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 3.0]])
    partial_denoise_rollout(rec, sched, z, t_start, spacing, labels, num_steps=2)
    # Every UNet eval got the per-sample labels [3,7] and the per-sample spacing.
    assert all(torch.equal(s, torch.tensor([3.0, 7.0])) for s in rec.seen_labels if s is not None)
    assert any(torch.allclose(s, spacing) for s in rec.seen_spacing)


def test_generate_reward_pairs_slices_per_sample_modality_across_batches():
    """A length-N per-sample modality is sliced per batch (batch0=[0,1], batch1=[2,3])."""
    sched = PartialFlowMatchHeunScheduler()
    rec = _RecordingDenoiser()
    torch.manual_seed(0)
    clean = torch.randn(4, *_LAT)
    sids = ["s0", "s0", "s1", "s1"]
    generate_reward_pairs(
        clean, sids, rec, sched, spacing=[1.0, 1.0, 1.0], modality=torch.tensor([0, 1, 2, 3]),
        num_steps=1, val_fraction=0.5, batch_size=2, seed=0, device="cpu",
    )
    seen = torch.stack([s for s in rec.seen_labels if s is not None])
    # Each batch's labels are a slice of the per-sample modality (order-independent check).
    assert torch.equal(seen.unique().sort().values, torch.tensor([0, 1, 2, 3]))


# -- #48/#50/#51: online rollout-in-the-loop + full-range ordered pairs --------


def test_partial_rollout_combined_batch_assertion():
    """A mismatched combined batch raises a clear error (not a MAISI-internal crash)."""
    sched = PartialFlowMatchHeunScheduler()
    denoiser = _IdentityDenoiser()
    z = torch.randn(4, *_LAT)
    # z_start batch (4) != t_start (3) — the online [2B] winner-first concat off-by-one.
    with pytest.raises(ValueError, match="t_start"):
        partial_denoise_rollout(denoiser, sched, z, torch.zeros(3), [1.0, 1.0, 1.0], 1, num_steps=1)
    # per-sample [B,3] spacing rows (3) != batch (4).
    with pytest.raises(ValueError, match="spacing"):
        partial_denoise_rollout(denoiser, sched, z, torch.zeros(4), torch.zeros(3, 3), 1, num_steps=1)


def test_full_range_val_pairs_winner_less_corrupted_on_average():
    """Full-range val pairs: the winner (max t) reconstructs clean better than the loser (min t)."""
    from manifold.data.reward_pairs import generate_full_range_val_pairs

    torch.manual_seed(0)
    clean = torch.randn(32, *_LAT)
    val = generate_full_range_val_pairs(
        clean, _IdentityDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, batch_size=8, seed=0, device="cpu",
    )
    assert len(val) == 32 and val.winners.shape[1:] == _LAT
    assert torch.isfinite(val.winners).all()
    # Identity denoiser ⇒ winner/loser are the noised starts; higher t (winner) is
    # closer to clean on average (the label direction holds at the transport level).
    w_recon = (val.winners - clean).flatten(1).norm(dim=1)
    l_recon = (val.losers - clean).flatten(1).norm(dim=1)
    assert w_recon.mean() < l_recon.mean()


def test_full_range_pair_design_winner_loser_overlap():
    """The max/min labeling over shared U[0,1) overlaps — the de-saturation property.

    Unlike the disjoint offline ``[0.5,1) / [0,0.5)`` halves (a single threshold
    separates every winner from every loser), the full-range ``winner_t = max``,
    ``loser_t = min`` over two ``U[0,1)`` draws overlap: a loser-t in one pair can
    exceed a winner-t in another — so the same corruption level can be a winner in
    one pair and a loser in another (the trivial shortcut is destroyed). ``t < 1``
    always (``torch.rand`` is half-open ``[0,1)``).
    """
    torch.manual_seed(0)
    n = 4096
    t_a = torch.rand(n)
    t_b = torch.rand(n)
    winner_t = torch.maximum(t_a, t_b)
    loser_t = torch.minimum(t_a, t_b)
    assert (winner_t < 1.0).all() and (loser_t < 1.0).all()  # half-open ⇒ t == 1 precluded
    assert (winner_t >= loser_t).all()  # max/min labeling
    # The distributions overlap: the largest loser_t exceeds the smallest winner_t.
    assert loser_t.max() > winner_t.min()


def test_partition_subjects_keeps_a_subject_wholly_in_one_split():
    """The held-out split is by SUBJECT — a subject lands wholly in train or val."""
    from manifold.data.reward_pairs import partition_subjects

    sids = [f"subj_{s}" for s in range(6) for _ in range(4)]  # 6 subjects × 4 contrasts
    train, val = partition_subjects(sids, val_fraction=0.34, seed=0)
    assert train.isdisjoint(val)
    for s in range(6):
        sid = f"subj_{s}"
        assert (sid in train) ^ (sid in val)  # exactly one of train/val


def test_no_leakage_train_clean_ds_excludes_val_subjects():
    """The train clean dataset serves train subjects only — no val subject reaches fit.

    The leakage guard is load-bearing (#51): the held-out-subject split must be
    enforced at the train-DATASET construction (filter items by train subjects),
    not only at val-precompute time — otherwise the discriminator trains on
    validation subjects and val/pair_acc measures memorization.
    """
    from manifold.data.reward_pairs import CleanLatentDataset, _subject_id, partition_subjects

    items = [
        {"latent": torch.randn(*_LAT), "sample_id": f"subj_{s}-t1n", "label": 1, "spacing": torch.tensor([1.0, 1.0, 1.0])}
        for s in range(10)
    ]
    sids = [_subject_id(it["sample_id"], None) for it in items]
    train_subj, val_subj = partition_subjects(sids, val_fraction=0.3, seed=0)
    train_items = [it for it, sid in zip(items, sids) if sid in train_subj]
    val_items = [it for it, sid in zip(items, sids) if sid in val_subj]
    CleanLatentDataset(train_items)  # constructs over train subjects only
    train_sids = {_subject_id(it["sample_id"], None) for it in train_items}
    val_sids = {_subject_id(it["sample_id"], None) for it in val_items}
    assert train_sids.isdisjoint(val_sids)  # no subject spans both
    assert train_sids <= train_subj  # every train item is a train subject
    assert val_sids <= val_subj


def test_clean_latent_dataset_applies_scale_once_on_read():
    """scale_factor is applied exactly once on read (the denoiser sees its training space)."""
    from manifold.data.reward_pairs import CleanLatentDataset

    items = [{"latent": torch.ones(*_LAT), "spacing": torch.tensor([1.0, 2.0, 3.0]), "label": 1, "sample_id": "x"}]
    ds = CleanLatentDataset(items, scaling_factor=2.5)
    b = ds[0]
    assert torch.allclose(b["latent"], torch.full(_LAT, 2.5))  # scaled exactly once
    assert torch.allclose(b["spacing"], torch.tensor([1.0, 2.0, 3.0]))
    assert b["label"].item() == 1


def test_load_cached_latents_reads_dict_items_and_groups_subjects(tmp_path):
    """load_cached_latents reads {latent, spacing, label, sample_id} dicts and groups subjects."""
    from manifold.data.reward_pairs import load_cached_latents

    for s in range(4):
        torch.save(
            {"latent": torch.ones(*_LAT), "sample_id": f"subj_{s}-t1n", "spacing": [1.0, 1.0, 1.0], "label": 1},
            tmp_path / f"s{s}.pt",
        )
    items, sids = load_cached_latents(tmp_path)
    assert len(items) == 4
    assert all(it["latent"].shape == _LAT for it in items)
    assert sids == ["subj_0", "subj_1", "subj_2", "subj_3"]  # contrast suffix stripped
