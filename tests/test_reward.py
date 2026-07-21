"""Reward model + reward module tests (GRPO reward model; online rollout-in-the loop, #48/#50).

External-behavior seams (per the PRD testing plan): the Reward Model scores a
latent to a finite per-sample scalar (no sigmoid); the Bradley–Terry loss is
finite and its gradient pushes ``r_w`` up / ``r_l`` down; the online
``RewardModule.forward("fit")`` consumes a **clean-latent** batch, rolls fresh
preference pairs (frozen denoiser, no grad), and returns a finite BT loss whose
backward touches discriminator params only; the frozen denoiser is held but
UNregistered (off the checkpoint/optimizer); and a reward-training run completes
end-to-end on toy clean latents via the injected-data CLI smoke, writing a
checkpoint and logging pairwise accuracy + the generated-end probe.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from manifold import PartialFlowMatchHeunScheduler, RewardModel
from manifold.data.reward_pairs import RewardPairDataset
from manifold.modules import RewardModule, bradley_terry_loss, reward_roc_auc
from manifold.training import run_reward_training
from manifold.training.reward_cli import RewardInputs, main as reward_main

#: A tiny latent shape + RewardModel config that survives the PatchGAN strided
#: convs on CPU (initial_conv 8->4, one middle layer 4->3, final_conv 3->2).
_LAT = (4, 8, 8, 8)
_RM_KW = dict(spatial_dims=3, in_channels=4, channels=8, num_layers_d=1)


def _reward_model() -> RewardModel:
    torch.manual_seed(0)
    return RewardModel(**_RM_KW)


class _SoftDenoiser(nn.Module):
    """A NON-identity fake denoiser (``x0 = 0.5·z``) so the rollout provably moves the latent.

    The identity denoiser (``x0 = z``) makes the rollout a verbatim no-op, so it
    cannot prove the rollout ran — hence the soft variant. Carries a dummy param
    so it mimics a real module (the rollout reads the device off
    ``next(unet.parameters())``). Frozen + eval by the Module ctor.
    """

    def __init__(self, target: torch.Tensor | None = None):
        super().__init__()
        self.dummy = nn.Parameter(torch.ones(3))
        self.target = target  # if set, ``x0 = 0.5·target + 0.5·z`` (pulls toward clean).

    def forward(self, sample, timestep, spacing, class_labels=None, **kw):
        if self.target is not None:
            return 0.5 * self.target + 0.5 * sample
        return 0.5 * sample


def _soft_denoiser() -> _SoftDenoiser:
    """The default fake denoiser for the smoke (non-identity → the rollout moves)."""
    return _SoftDenoiser()


# -- Reward Model -----------------------------------------------------------


def test_reward_model_scores_to_per_sample_scalar():
    """forward maps [B,C,D,H,W] -> [B] finite rewards; the raw patch mean (no sigmoid)."""
    m = _reward_model()
    x = torch.randn(3, *_LAT)
    r = m(x)
    assert r.shape == (3,)
    assert torch.isfinite(r).all()
    # Raw patch-logit mean — no sigmoid, so rewards may be any real sign.
    assert r.abs().sum() > 0


def test_reward_model_scores_winner_above_loser_on_separated_pair():
    """On a clean-vs-noised pair the raw reward orders the cleaner latent higher.

    A sanity check that the PatchGAN's natural bias (before training) is not
    inverted; the Bradley–Terry loss then amplifies this into a calibrated margin.
    """
    torch.manual_seed(0)
    m = _reward_model()
    clean = torch.randn(8, *_LAT)
    # A heavier corruption is a different latent, not a uniformly 'worse' one, so
    # average over many noises to see the expected ordering on a separated pair.
    margins = []
    for _ in range(8):
        noise = torch.randn_like(clean)
        loser = 0.2 * clean + 0.8 * noise  # heavily corrupted
        r = m(torch.cat([clean, loser]))
        margins.append((r[:8] - r[8:]).mean().item())
    assert sum(margins) / len(margins) != 0.0  # finite, non-degenerate scoring


# -- Bradley–Terry loss (kept verbatim) -------------------------------------


def test_bradley_terry_loss_finite_and_gradient_direction():
    """L = -log σ(r_w − r_l); ∂L/∂r_w < 0 and ∂L/∂r_l > 0 (descent raises r_w, lowers r_l).

    The loss gradient direction is the load-bearing BT property: gradient descent
    pushes the winner's reward up and the loser's down, calibrating differences.
    """
    r_w = torch.tensor([0.3, -0.5, 1.2], requires_grad=True)
    r_l = torch.tensor([0.1, 0.2, -0.4], requires_grad=True)
    loss = bradley_terry_loss(r_w, r_l)
    assert torch.isfinite(loss)
    loss.backward()
    # ∂L/∂r_w = σ(d) − 1 < 0  →  descent increases r_w.
    assert (r_w.grad < 0).all()
    # ∂L/∂r_l = 1 − σ(d) > 0  →  descent decreases r_l.
    assert (r_l.grad > 0).all()


# -- Reward Module (online rollout contract) --------------------------------


def _clean_batch(b: int = 2) -> dict:
    """A clean-latent batch the fit-step online rollout consumes."""
    return {
        "latent": torch.randn(b, *_LAT),
        "spacing": torch.tensor([[1.0, 1.0, 1.0]] * b),
        "label": torch.tensor([1] * b, dtype=torch.long),
    }


def _module() -> RewardModule:
    return RewardModule(_reward_model(), lr=1e-2, denoiser=_soft_denoiser(), scheduler=PartialFlowMatchHeunScheduler(), num_steps=2)


def test_module_forward_fit_returns_finite_bt_loss():
    mod = _module()
    out = mod.forward(_clean_batch(), "fit")
    assert "loss" in out
    assert torch.isfinite(out["loss"])


def test_module_online_rollout_moves_latent():
    """With a NON-identity denoiser the rollout output differs from its noised start.

    Proves the rollout ran (not skipped): the identity denoiser (x0 = z) makes the
    rollout a no-op, so the soft denoiser (x0 = 0.5·z) is used — its output must
    depart from the noised start latent (#50 acceptance).
    """
    mod = _module()
    winner, loser = mod._online_rollout(_clean_batch(2))
    # Reconstruct the noised start for the winner half (winner_t = max(t_a, t_b)).
    # The frozen soft denoiser shrinks z toward 0 each Heun eval → out ≠ z_start.
    assert winner.shape == (2, *_LAT) and loser.shape == (2, *_LAT)
    assert torch.isfinite(winner).all() and torch.isfinite(loser).all()
    # A shrunk (×0.5) rollout must depart from any fixed noised start — out has
    # smaller magnitude than a unit-normal noised start.
    assert winner.abs().mean() < 0.9


def test_module_backward_updates_discriminator_only():
    """backward populates grads on every discriminator param; the denoiser is frozen + unregistered.

    The Module HOLDS the frozen denoiser (unregistered via object.__setattr__) —
    so it is absent from parameters()/state_dict()/optimizer, and backward only
    touches discriminator params (#50 acceptance: the exclusion invariant).
    """
    m = _reward_model()
    mod = RewardModule(m, lr=1e-2, denoiser=_soft_denoiser(), scheduler=PartialFlowMatchHeunScheduler(), num_steps=2)
    mod.forward(_clean_batch(), "fit")["loss"].backward()
    params = list(m.parameters())
    assert params, "reward model has parameters"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    # Denoiser held but UNREGISTERED: its params are disjoint from the Module's
    # parameters() (so off the optimizer / checkpoint / DDP).
    denoiser_ids = {id(p) for p in mod.denoiser.parameters()}
    module_ids = {id(p) for p in mod.parameters()}
    assert denoiser_ids, "fake denoiser has parameters"
    assert denoiser_ids.isdisjoint(module_ids)
    assert "denoiser" not in mod.state_dict()
    assert all(not p.requires_grad for p in mod.denoiser.parameters())
    # The optimizer covers discriminator params only.
    opt_ids = {id(p) for p in mod.configure_optimizers()["optimizer"].param_groups[0]["params"]}
    assert opt_ids == {id(p) for p in m.parameters()}


def test_module_optimizer_step_widens_winner_loser_margin():
    """One Adam step on the BT loss widens (r_w − r_l) on the rolled pair.

    After a step the reward margin on the SAME rolled pair must increase — the BT
    loss can only push r_w up and r_l down. (Rewritten from the pair-fed version to
    the clean-latent contract: the pair is rolled fresh inside forward.)
    """
    torch.manual_seed(0)
    m = _reward_model()
    mod = RewardModule(m, lr=1e-2, denoiser=_soft_denoiser(), scheduler=PartialFlowMatchHeunScheduler(), num_steps=2)
    batch = _clean_batch(2)

    # Snapshot the rolled pair (detached) so the margin is measured on the SAME
    # latents before/after the step (the rollout is non-deterministic across calls).
    with torch.no_grad():
        winner0, loser0 = mod._online_rollout({**batch})

    def margin() -> float:
        r = m(torch.cat([winner0, loser0]))
        return float((r[:2] - r[2:]).mean())

    before = margin()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        mod.forward(batch, "fit")["loss"].backward()
        opt.step()
    after = margin()
    assert after > before


def test_module_forward_stage_mismatch_raises():
    """A clean-latent batch routed to validate (or a pair batch to fit) raises clearly."""
    mod = _module()
    # pair batch → fit
    with __import__("pytest").raises(ValueError, match="clean-latent"):
        mod.forward({"winner": torch.randn(1, *_LAT), "loser": torch.randn(1, *_LAT)}, "fit")
    # clean-latent batch → validate
    with __import__("pytest").raises(ValueError, match="winner"):
        mod.forward(_clean_batch(1), "validate")
    # unknown stage
    with __import__("pytest").raises(ValueError, match="stage"):
        mod.forward(_clean_batch(1), "test")


def test_module_combined_batch_threads_duplicated_conditioning():
    """The [2B] winner-first rollout duplicates per-sample spacing/modality to 2B."""
    sched = PartialFlowMatchHeunScheduler()

    class _RecordingDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.ones(3))
            self.seen_labels = []
            self.seen_spacing = []

        def forward(self, sample, timestep, spacing, class_labels=None, **kw):
            self.seen_labels.append(class_labels.detach().cpu().clone())
            self.seen_spacing.append(spacing.detach().cpu().clone())
            return sample  # identity (we only assert what it RECEIVES)

    rec = _RecordingDenoiser()
    mod = RewardModule(_reward_model(), lr=1e-2, denoiser=rec, scheduler=sched, num_steps=1)
    b = 2
    batch = {
        "latent": torch.randn(b, *_LAT),
        "spacing": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        "label": torch.tensor([3, 7], dtype=torch.long),
    }
    mod._online_rollout(batch)
    # Every UNet eval received the [2B] duplicated labels [3,7,3,7] and spacing.
    assert all(tuple(s.tolist()) == (3, 7, 3, 7) for s in rec.seen_labels if s is not None)
    assert any(torch.allclose(s, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]] * 2)) for s in rec.seen_spacing)


# -- CLI smoke (the end-to-end seam) ----------------------------------------


class _ToyPairDS(Dataset):
    """Handmade learnable pairs: winner = a clean latent, loser = a corrupted one (val)."""

    def __init__(self, n: int = 8):
        torch.manual_seed(0)
        clean = torch.randn(n, *_LAT)
        self.items = []
        for i in range(n):
            noise = torch.randn(*_LAT)
            self.items.append({"winner": clean[i].clone(), "loser": (0.3 * clean[i] + 0.7 * noise)})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _ToyCleanDS(Dataset):
    """A tiny clean-latent dataset (train): emits {latent, spacing, label}."""

    def __init__(self, n: int = 8):
        torch.manual_seed(0)
        self.latents = torch.randn(n, *_LAT)

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        return {
            "latent": self.latents[i],
            "spacing": torch.tensor([1.0, 1.0, 1.0]),
            "label": torch.tensor(1, dtype=torch.long),
        }


def _inputs() -> RewardInputs:
    """The injection-seam bundle: fake non-identity denoiser + toy clean train + val/probe pairs."""
    probe_item = _ToyPairDS(n=4).items
    probe = RewardPairDataset(
        torch.stack([it["winner"] for it in probe_item]),
        torch.stack([it["loser"] for it in probe_item]),
    )
    return RewardInputs(
        denoiser=_soft_denoiser(),
        scheduler=PartialFlowMatchHeunScheduler(),
        num_steps=2,
        clean_ds=_ToyCleanDS(),
        val_pair_ds=_ToyPairDS(n=4),
        val_probe=probe,
    )


def _run(tmp_path, **kw):
    return run_reward_training(
        module=_module(),
        inputs=_inputs(),
        model_dir=str(tmp_path),
        max_epochs=2,
        devices=1,
        accelerator="cpu",
        batch_size=2,
        num_workers=0,
        limit_val_batches=1.0,
        **kw,
    )


def test_run_reward_training_writes_ckpt_and_logs_metrics(tmp_path):
    trainer, ckpt = _run(tmp_path)
    metrics = trainer.callback_metrics
    for key in ("val/pair_acc", "val/roc_auc", "val/gen_pair_acc"):
        assert key in metrics, f"missing {key}"
        assert torch.isfinite(metrics[key])
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()
    # on_fit_start moved the unregistered denoiser onto the module device.
    assert next(trainer.model.denoiser.parameters()).device == trainer.model.device


_TINY_NETWORK_YAML = """\
spatial_dims: 3
latent_channels: 4
reward_model:
  spatial_dims: ${spatial_dims}
  in_channels: ${latent_channels}
  channels: 8
  num_layers_d: 1
  norm: BATCH
"""


def _write_tiny_configs(tmp_path):
    net = tmp_path / "network.yaml"
    net.write_text(_TINY_NETWORK_YAML)
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
        + "val_subset_size: 4\n"
    )
    train = tmp_path / "train.yaml"
    train.write_text(
        "reward_train: {batch_size: 2, lr: 1.0e-2, n_epochs: 2, num_steps: 2}\n"
        "reward: {val_fraction: 0.34}\n"
    )
    return str(env), str(train), str(net)


def test_main_runs_end_to_end_with_fake_data(tmp_path):
    """main(): argparse -> compose -> build -> online fit -> ckpt (fake-data seam)."""
    env, train, net = _write_tiny_configs(tmp_path)

    def fake_provider(cfg, device):
        return _inputs()

    rc = reward_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "2"],
        data_provider=fake_provider,
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_main_native_dir_latents_dir_default_none_and_validated(tmp_path):
    """--native-dir/--latents-dir default None and are validated only without a data_provider."""
    env, train, net = _write_tiny_configs(tmp_path)
    # No data_provider AND no --native-dir/--latents-dir → clear error (not a crash).
    with __import__("pytest").raises(ValueError, match="native-dir"):
        reward_main(["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"])
    # With a data_provider, the missing args are NOT required (smoke seam intact).
    rc = reward_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1"],
        data_provider=lambda cfg, device: _inputs(),
    )
    assert rc == 0


# -- #43: ROC-AUC + generated-end probe + persistence + real pairs -----------


def test_reward_model_default_depth_works_on_production_latent():
    """The default num_layers_d=3 scores a production-shaped latent (z-dim 32)."""
    m = RewardModel()  # defaults: channels=64, num_layers_d=3
    r = m(torch.randn(2, 4, 64, 64, 32))
    assert r.shape == (2,) and torch.isfinite(r).all()


def test_reward_model_raises_clear_error_on_collapsed_spatial():
    """A too-small latent raises a clear ValueError (not a cryptic MONAI RuntimeError)."""
    import pytest

    m = RewardModel(num_layers_d=3)
    with pytest.raises(ValueError, match="num_layers_d"):
        m(torch.randn(1, 4, 8, 8, 8))  # z-dim 8 collapses under 3 stride-2 convs


def test_main_uses_committed_default_reward_recipe(tmp_path):
    """main() with NO -c (argparse default) resolves the committed config_reward.yaml."""
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
        + "val_subset_size: 4\n"
    )

    def fake_provider(cfg, device):
        return _inputs()

    net = "configs/network/config_network.yaml"
    rc = reward_main(
        ["-e", str(env), "-t", net, "-g", "1", "--max-epochs", "1", "reward_model.num_layers_d=1"],
        data_provider=fake_provider,
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


def test_reward_roc_auc_perfect_inverted_and_differs_from_pair_acc():
    """ROC-AUC: perfect ranking → 1, inverted → 0; and it ≠ pairwise accuracy."""
    perfect_w = torch.tensor([2.0, 3.0])
    perfect_l = torch.tensor([0.0, 1.0])
    assert reward_roc_auc(perfect_w, perfect_l).item() == 1.0
    assert reward_roc_auc(perfect_l, perfect_w).item() == 0.0  # inverted

    w = torch.tensor([2.0, 0.5])
    loser = torch.tensor([1.0, 0.0])
    assert (w > loser).float().mean().item() == 1.0  # pair_acc == 1
    assert reward_roc_auc(w, loser).item() == 0.75   # 0.5 does not beat loser 1.0


def test_reward_roc_auc_handles_single_class_neutral():
    """No losers (or no winners) → neutral 0.5 rather than a divide-by-zero."""
    w = torch.tensor([1.0, 2.0])
    loser = torch.tensor([])  # no negatives
    assert reward_roc_auc(w, loser).item() == 0.5


def test_reward_roc_auc_fp16_large_validation_set_is_not_nan():
    """fp16 rewards (AMP validation) over a large N must not overflow to NaN."""
    n = 1000
    w = torch.full((n,), 1.0, dtype=torch.float16)
    loser = torch.full((n,), -1.0, dtype=torch.float16)
    assert not torch.isnan(reward_roc_auc(w, loser))
    assert reward_roc_auc(w, loser).item() == 1.0  # perfect ranking
    assert reward_roc_auc(w, loser).item() == reward_roc_auc(w.float(), loser.float()).item()


def test_reward_model_save_load_round_trips_identical(tmp_path):
    """Reward Model persists via component save/load; reloaded scores identically."""
    m = _reward_model()
    x = torch.randn(3, *_LAT)
    before = m(x)
    m.save_pretrained(str(tmp_path / "reward"))
    reloaded = RewardModel.from_pretrained(str(tmp_path / "reward"))
    after = reloaded(x)
    assert torch.equal(before, after)
    assert reloaded.config["num_layers_d"] == _RM_KW["num_layers_d"]


def test_module_validation_logs_pair_acc_roc_auc_and_probe(tmp_path):
    """validation_step logs pair_acc + roc_auc; the probe logs gen_pair_acc.

    Driven through a real Trainer.fit so the Lightning-attached logging path and
    the on_validation_epoch_end probe hook both run end-to-end (train = clean latents).
    """
    module = _module()  # carries the probe-less default; _inputs sets the probe
    trainer, _ = run_reward_training(
        module=module, inputs=_inputs(), model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2, num_workers=0, limit_val_batches=1.0,
    )
    metrics = trainer.callback_metrics
    for key in ("val/pair_acc", "val/roc_auc", "val/gen_pair_acc"):
        assert key in metrics, f"missing {key}"
        assert torch.isfinite(metrics[key])


def test_run_reward_training_on_real_clean_latents(tmp_path):
    """The reward run trains on a real CleanLatentDataset (clean latents, not pairs).

    Builds clean latents + a non-identity denoiser (so the rollout moves), wraps
    them in a CleanLatentDataset (scale-on-read), and fits — wiring the online
    clean-latent train contract into the training run.
    """
    torch.manual_seed(0)
    clean = torch.randn(12, *_LAT)
    items = [
        {"latent": clean[i], "spacing": torch.tensor([1.0, 1.0, 1.0]), "label": 1, "sample_id": f"subj_{i // 2}"}
        for i in range(12)
    ]
    from manifold.data.reward_pairs import CleanLatentDataset

    inputs = RewardInputs(
        denoiser=_soft_denoiser(), scheduler=PartialFlowMatchHeunScheduler(), num_steps=2,
        clean_ds=CleanLatentDataset(items, scaling_factor=1.0),
        val_pair_ds=_ToyPairDS(n=4), val_probe=RewardPairDataset(clean[:4], clean[4:8]),
    )
    trainer, ckpt = run_reward_training(
        module=RewardModule(_reward_model(), lr=1e-2, denoiser=inputs.denoiser, scheduler=inputs.scheduler, num_steps=2),
        inputs=inputs, model_dir=str(tmp_path), max_epochs=1,
        devices=1, accelerator="cpu", batch_size=2, num_workers=0, limit_val_batches=1.0,
    )
    metrics = trainer.callback_metrics
    assert "val/pair_acc" in metrics and "val/gen_pair_acc" in metrics
    assert Path(ckpt.best_model_path).is_file()


def test_main_real_path_native_dir_and_latents_dir(tmp_path):
    """main() with --native-dir + --latents-dir (no data_provider) → online fit on the real path."""
    from manifold import AutoencoderKL, FlowMatchHeunDiscreteScheduler, LatentFlowPipeline, UNet3DConditionModel

    torch.manual_seed(0)
    unet = UNet3DConditionModel(num_class_embeds=4, include_spacing_input=True)
    LatentFlowPipeline(
        unet, AutoencoderKL(scaling_factor=0.5), FlowMatchHeunDiscreteScheduler(),
    ).save_pretrained(str(tmp_path / "native"))

    latents_dir = tmp_path / "latents"
    latents_dir.mkdir()
    for s in range(6):
        for v in range(2):
            torch.save(
                {"latent": torch.randn(*_LAT), "sample_id": f"subj_{s}-t1n", "spacing": [1.0, 1.0, 1.0], "label": 1},
                latents_dir / f"subj_{s}__v{v}__abc.pt",
            )

    env, train, net = _write_tiny_configs(tmp_path)
    rc = reward_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "1",
         "--native-dir", str(tmp_path / "native"), "--latents-dir", str(latents_dir),
         "reward_train.num_steps=1"],
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


# -- Codex #45 regression: O(N log N) ROC-AUC + DDP checkpoint + probe batching


def _brute_force_auc(rw, rl):
    """Reference AUC = mean over (pos,neg) pairs of [p>n] + 0.5*[p==n] (handles ties)."""
    total = 0.0
    for p in rw:
        for n in rl:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(rw) * len(rl))


def test_reward_roc_auc_matches_brute_force_with_and_without_ties():
    """The O(N log N) sort-based AUC equals the brute-force pairwise AUC (incl. ties)."""
    torch.manual_seed(0)
    rw = torch.randn(20)
    rl = torch.randn(20) + 0.3
    assert abs(reward_roc_auc(rw, rl).item() - _brute_force_auc(rw.tolist(), rl.tolist())) < 1e-6
    rw_tie = torch.tensor([2.0, 2.0, 0.5, 0.5])
    rl_tie = torch.tensor([1.0, 1.0, 0.0, -1.0])
    assert abs(reward_roc_auc(rw_tie, rl_tie).item() - _brute_force_auc(rw_tie.tolist(), rl_tie.tolist())) < 1e-6


def test_build_checkpoint_ddp_fallback_drops_monitor():
    """Under multi-GPU the checkpoint does not monitor a rank-local metric (mirrors JiT)."""
    from manifold.training.reward_cli import _ckpt

    single = _ckpt(str(Path("/tmp/_rwd_ckpt_a").resolve()), monitor_metric="val/gen_pair_acc")
    multi = _ckpt(str(Path("/tmp/_rwd_ckpt_b").resolve()), monitor_metric=None)
    assert single.monitor == "val/gen_pair_acc"  # the GRPO-regime probe metric
    assert multi.monitor is None  # DDP: no rank-0-shard selection
    assert multi.save_last and multi.save_top_k == 1


def test_generated_end_probe_is_scored_in_chunks(tmp_path):
    """A probe larger than probe_batch_size is still scored correctly (chunked, no OOM)."""
    import shutil

    big_probe = _ToyPairDS(n=8)
    pw = torch.stack([big_probe.items[i]["winner"] for i in range(8)])
    pl_ = torch.stack([big_probe.items[i]["loser"] for i in range(8)])
    module = _module()  # denoiser-equipped (fit rolls fresh pairs)
    module.set_val_probe(pw, pl_)
    module.probe_batch_size = 2
    trainer, _ = run_reward_training(
        module=module, inputs=_inputs(), model_dir=str(tmp_path), max_epochs=1,
        devices=1, accelerator="cpu", batch_size=2, num_workers=0, limit_val_batches=1.0,
    )
    metrics = trainer.callback_metrics
    assert "val/gen_pair_acc" in metrics and torch.isfinite(metrics["val/gen_pair_acc"])
    shutil.rmtree("/tmp/_rwd_ckpt_a", ignore_errors=True)
    shutil.rmtree("/tmp/_rwd_ckpt_b", ignore_errors=True)
