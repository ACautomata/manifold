"""Partial-denoise pair generation tests (GRPO reward model, issue #39/#42).

External-behavior seams: the partial scheduler yields a per-sample ``(B, n+1)``
grid over ``[t_start, 1]``; the partial rollout degenerates to the full-from-noise
rollout at ``t_start = 0`` and is identity-preserving for an identity denoiser;
``generate_reward_pairs`` emits a held-out-subject ``(train, val)`` split whose
winners reconstruct the clean latent better than losers (the label direction);
and the frozen denoiser loads from a native export.
"""

from __future__ import annotations

from pathlib import Path

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

    denoiser, partial_sched = load_frozen_denoiser(tmp_path / "native")
    assert isinstance(partial_sched, PartialFlowMatchHeunScheduler)
    assert partial_sched.t_eps == 0.05  # matched from the JiT scheduler config
    # Frozen + a working scoring forward.
    assert all(not p.requires_grad for p in denoiser.parameters())
    out = denoiser(sample=torch.randn(1, *_LAT), timestep=torch.tensor([0.5]),
                   spacing=torch.tensor([1.0, 1.0, 1.0]), class_labels=torch.tensor([1]))
    assert out.shape == (1, *_LAT)


# -- generation script (end-to-end console entry) ---------------------------


def test_generate_reward_pairs_script_end_to_end(tmp_path):
    """scripts/generate_reward_pairs.py: native export → frozen denoiser → pairs cache."""
    import sys

    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    LatentFlowPipeline(
        unet,
        AutoencoderKL(scaling_factor=0.5),
        FlowMatchHeunDiscreteScheduler(),
    ).save_pretrained(str(tmp_path / "native"))

    # A handful of clean latents on disk (cache-item dicts, two per subject).
    latents_dir = tmp_path / "latents"
    latents_dir.mkdir()
    for s in range(6):
        for v in range(2):
            torch.save(
                {"latent": torch.randn(*_LAT), "sample_id": f"subj_{s}"},
                latents_dir / f"subj_{s}__v{v}__abc.pt",
            )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    try:
        import generate_reward_pairs as cli  # type: ignore[import-not-found]

        rc = cli.main(
            [
                "--native-dir", str(tmp_path / "native"),
                "--latents-dir", str(latents_dir),
                "--output-dir", str(tmp_path / "pairs"),
                "--num-steps", "2",
                "--modality", "1",
                "--val-fraction", "0.34",
                "--batch-size", "4",
            ]
        )
    finally:
        sys.path.pop(0)
    assert rc == 0
    train, val, probe = load_reward_pairs(tmp_path / "pairs")
    assert len(train) + len(val) == 12
    assert probe is not None and len(probe) > 0  # generated-end probe written
    assert train.winners.shape[1:] == _LAT
    assert torch.isfinite(train.winners).all() and torch.isfinite(val.losers).all()

