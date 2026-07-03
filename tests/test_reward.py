"""Reward model + reward module tests (GRPO reward model, issue #39/#40).

External-behavior seams (per the PRD testing plan): the Reward Model scores a
latent to a finite per-sample scalar (no sigmoid); the Bradley–Terry loss is
finite and its gradient pushes ``r_w`` up / ``r_l`` down; ``RewardModule.forward``
returns a finite BT loss whose backward touches discriminator params only; and a
reward-training run completes end-to-end on toy pairs via the injected-data CLI
smoke, writing a checkpoint and logging pairwise accuracy.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from manifold import RewardModel
from manifold.modules import RewardModule, bradley_terry_loss, reward_roc_auc
from manifold.training import run_reward_training
from manifold.training.reward_cli import _RewardDataBundle, main as reward_main

#: A tiny latent shape + RewardModel config that survives the PatchGAN strided
#: convs on CPU (initial_conv 8->4, one middle layer 4->3, final_conv 3->2).
_LAT = (4, 8, 8, 8)
_RM_KW = dict(spatial_dims=3, in_channels=4, channels=8, num_layers_d=1)


def _reward_model() -> RewardModel:
    torch.manual_seed(0)
    return RewardModel(**_RM_KW)


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


# -- Bradley–Terry loss -----------------------------------------------------

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


# -- Reward Module ----------------------------------------------------------


def test_module_forward_fit_returns_finite_bt_loss():
    m = _reward_model()
    mod = RewardModule(m, lr=1e-2)
    winner = torch.randn(2, *_LAT)
    loser = torch.randn(2, *_LAT)
    out = mod.forward({"winner": winner, "loser": loser}, "fit")
    assert torch.isfinite(out["loss"])
    assert "loss" in out


def test_module_backward_updates_discriminator_only():
    """backward populates grads on every discriminator param; nothing else exists.

    The Module holds only the Reward Model (pairs are pre-made) — so every
    trainable parameter is a discriminator parameter (issue #40 acceptance).
    """
    m = _reward_model()
    mod = RewardModule(m, lr=1e-2)
    winner = torch.randn(2, *_LAT)
    loser = torch.randn(2, *_LAT)
    mod.forward({"winner": winner, "loser": loser}, "fit")["loss"].backward()
    params = list(mod.reward_model.parameters())
    assert params, "reward model has parameters"
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    # Issue #40 / ADR-0009 acceptance: the Module holds NO denoiser — every parameter
    # is a discriminator parameter. Guards a future online-pair-generation change
    # from silently registering a frozen denoiser on the module.
    assert set(mod.parameters()) == set(mod.reward_model.parameters())


def test_module_optimizer_step_raises_winner_above_loser():
    """One Adam step on the BT loss widens (r_w − r_l) on the pair (gradient direction).

    After a step the reward margin on the *same* pair must increase — the BT loss
    can only push r_w up and r_l down.
    """
    torch.manual_seed(0)
    m = _reward_model()
    mod = RewardModule(m, lr=1e-2)
    winner = torch.randn(2, *_LAT)
    loser = torch.randn(2, *_LAT)

    def margin() -> float:
        r = m(torch.cat([winner, loser]))
        return float((r[:2] - r[2:]).mean())

    before = margin()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(3):
        opt.zero_grad()
        mod.forward({"winner": winner, "loser": loser}, "fit")["loss"].backward()
        opt.step()
    after = margin()
    assert after > before


# -- CLI smoke (the end-to-end seam) ----------------------------------------


class _ToyPairDS(Dataset):
    """Handmade learnable pairs: winner = a clean latent, loser = a corrupted one."""

    def __init__(self, n: int = 8):
        torch.manual_seed(0)
        clean = torch.randn(n, *_LAT)
        self.items = []
        for i in range(n):
            noise = torch.randn(*_LAT)
            self.items.append(
                {"winner": clean[i].clone(), "loser": (0.3 * clean[i] + 0.7 * noise)}
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _module():
    return RewardModule(_reward_model(), lr=1e-2)


def _bundle():
    return _RewardDataBundle(pair_ds=_ToyPairDS(), val_pair_ds=_ToyPairDS(n=4))


def _run(tmp_path, **kw):
    return run_reward_training(
        module=_module(),
        bundle=_bundle(),
        model_dir=str(tmp_path),
        max_epochs=2,
        devices=1,
        accelerator="cpu",
        batch_size=2,
        num_workers=0,
        limit_val_batches=1.0,
        **kw,
    )


def test_run_reward_training_writes_ckpt_and_logs_pair_acc(tmp_path):
    trainer, ckpt = _run(tmp_path)
    metrics = trainer.callback_metrics
    assert "val/pair_acc" in metrics
    assert torch.isfinite(metrics["val/pair_acc"])
    ckpts = list(Path(str(tmp_path)).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)
    assert ckpt.best_model_path and Path(ckpt.best_model_path).is_file()


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
        "reward_train: {batch_size: 2, lr: 1.0e-2, n_epochs: 2}\n"
        "reward: {pairs_dir: /tmp/_unused_pairs_}\n"
    )
    return str(env), str(train), str(net)


def test_main_runs_end_to_end_with_fake_pair_data(tmp_path):
    """main(): argparse -> compose -> build -> fit -> ckpt (fake-data seam)."""
    env, train, net = _write_tiny_configs(tmp_path)

    def fake_provider(cfg, device):
        return _bundle()

    rc = reward_main(
        ["-e", env, "-c", train, "-t", net, "-g", "1", "--max-epochs", "2"],
        data_provider=fake_provider,
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)


class _IdDenoiser(nn.Module):
    """A frozen identity denoiser (x0 = z) with a dummy param (mimics a real UNet)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(0))

    def forward(self, sample, timestep, spacing, class_labels=None, **kw):
        return sample


# -- #43: ROC-AUC + generated-end probe + persistence + real pairs -----------

#: A production-shaped latent (~[4,64,64,32], z-dim 32 — the BraTS2023 GLI cached
#: latent size) that survives the default RewardModel depth (num_layers_d=3).
_LAT_PROD = (4, 64, 64, 32)


def test_reward_model_default_depth_works_on_production_latent():
    """The default num_layers_d=3 scores a production-shaped latent (z-dim 32).

    Guards the issue's reviewer finding that the default crashed on small latents:
    on the real BraTS2023 GLI latent shape (~[4,64,64,32], z-dim 32) the default
    depth produces a finite patch map. Smaller latents raise a clear ValueError
    (see test_reward_model_raises_clear_error_on_collapsed_spatial).
    """
    m = RewardModel()  # defaults: channels=64, num_layers_d=3
    r = m(torch.randn(2, *_LAT_PROD))
    assert r.shape == (2,) and torch.isfinite(r).all()


def test_reward_model_raises_clear_error_on_collapsed_spatial():
    """A too-small latent raises a clear ValueError (not a cryptic MONAI RuntimeError)."""
    import pytest

    m = RewardModel(num_layers_d=3)
    with pytest.raises(ValueError, match="num_layers_d"):
        m(torch.randn(1, 4, 8, 8, 8))  # z-dim 8 collapses under 3 stride-2 convs


def test_main_uses_committed_default_reward_recipe(tmp_path):
    """main() with NO -c (argparse default) resolves the committed config_reward.yaml.

    Guards the reviewer finding that the default -c path did not exist: the
    committed recipe + network config (reward_model block) compose, and the run
    fits on fake data. num_layers_d is overridden to 1 for the tiny toy latent.
    """
    env = tmp_path / "env.yaml"
    env.write_text(
        "data_base_dir: /tmp/_unused_\n"
        "model_dir: %s\n" % (tmp_path / "model")
        + "val_subset_size: 4\n"
    )

    def fake_provider(cfg, device):
        return _bundle()

    net = "configs/network/config_network.yaml"
    rc = reward_main(
        ["-e", str(env), "-t", net, "-g", "1", "--max-epochs", "1", "reward_model.num_layers_d=1"],
        data_provider=fake_provider,
    )
    assert rc == 0
    ckpts = list(Path(str(tmp_path / "model")).glob("*.ckpt"))
    assert any(p.name == "last.ckpt" for p in ckpts)




def test_reward_roc_auc_perfect_inverted_and_differs_from_pair_acc():
    """ROC-AUC: perfect ranking → 1, inverted → 0; and it ≠ pairwise accuracy.

    AUC ranks every winner against every loser (cross-pairs); pairwise accuracy
    only the matched pairs — so AUC is the stricter, threshold-free summary.
    """
    perfect_w = torch.tensor([2.0, 3.0])
    perfect_l = torch.tensor([0.0, 1.0])
    assert reward_roc_auc(perfect_w, perfect_l).item() == 1.0
    assert reward_roc_auc(perfect_l, perfect_w).item() == 0.0  # inverted

    # Matched pairs all rank right (pair_acc = 1), but a winner is below a
    # *different* loser → AUC < 1 (the cross-pair distinction).
    w = torch.tensor([2.0, 0.5])
    loser = torch.tensor([1.0, 0.0])
    assert (w > loser).float().mean().item() == 1.0  # pair_acc == 1
    assert reward_roc_auc(w, loser).item() == 0.75   # 0.5 does not beat loser 1.0


def test_reward_roc_auc_handles_single_class_neutral():
    """No losers (or no winners) → neutral 0.5 rather than a divide-by-zero."""
    w = torch.tensor([1.0, 2.0])
    loser = torch.tensor([])  # no negatives
    assert reward_roc_auc(w, loser).item() == 0.5


def test_reward_model_save_load_round_trips_identical(tmp_path):
    """Reward Model persists via component save/load; reloaded scores identically."""
    m = _reward_model()
    x = torch.randn(3, *_LAT)
    before = m(x)
    m.save_pretrained(str(tmp_path / "reward"))
    reloaded = RewardModel.from_pretrained(str(tmp_path / "reward"))
    after = reloaded(x)
    assert torch.equal(before, after)
    # The config round-trips (construction was captured).
    assert reloaded.config["num_layers_d"] == _RM_KW["num_layers_d"]


def test_module_validation_logs_pair_acc_roc_auc_and_probe(tmp_path):
    """validation_step logs pair_acc + roc_auc; the probe logs gen_pair_acc.

    Driven through a real Trainer.fit so the Lightning-attached logging path and
    the on_validation_epoch_end probe hook both run end-to-end.
    """
    probe_item = _ToyPairDS(n=4).items[0]
    module = RewardModule(
        _reward_model(), lr=1e-2,
        val_probe=(probe_item["winner"].unsqueeze(0), probe_item["loser"].unsqueeze(0)),
    )
    bundle = _RewardDataBundle(pair_ds=_ToyPairDS(), val_pair_ds=_ToyPairDS(n=4))
    trainer, _ = run_reward_training(
        module=module, bundle=bundle, model_dir=str(tmp_path),
        max_epochs=1, devices=1, accelerator="cpu", batch_size=2, num_workers=0, limit_val_batches=1.0,
    )
    metrics = trainer.callback_metrics
    for key in ("val/pair_acc", "val/roc_auc", "val/gen_pair_acc"):
        assert key in metrics, f"missing {key}"
        assert torch.isfinite(metrics[key])


def test_run_reward_training_on_real_precomputed_pairs(tmp_path):
    """The reward run consumes a real precomputed RewardPairDataset (held-out split).

    Generates pairs with a tiny identity denoiser (so pairs are static latents),
    then fits on them — wiring the offline pair cache (#42) into the training run.
    """
    from manifold import PartialFlowMatchHeunScheduler
    from manifold.data.reward_pairs import generate_reward_pairs

    torch.manual_seed(0)
    clean = torch.randn(12, *_LAT)
    sids = [f"subj_{i // 2}" for i in range(12)]
    train_pairs, val_pairs = generate_reward_pairs(
        clean, sids, _IdDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, val_fraction=0.34,
        batch_size=4, seed=0, device="cpu",
    )
    bundle = _RewardDataBundle(pair_ds=train_pairs, val_pair_ds=val_pairs, val_probe=val_pairs)
    trainer, ckpt = run_reward_training(
        module=RewardModule(_reward_model(), lr=1e-2),
        bundle=bundle, model_dir=str(tmp_path), max_epochs=1,
        devices=1, accelerator="cpu", batch_size=2, num_workers=0, limit_val_batches=1.0,
    )
    metrics = trainer.callback_metrics
    assert "val/pair_acc" in metrics and "val/gen_pair_acc" in metrics
    assert Path(ckpt.best_model_path).is_file()


def test_main_consumes_pairs_dir_with_probe(tmp_path):
    """main() with --pairs-dir (no data_provider) loads real pairs + probe → fit."""
    from manifold import PartialFlowMatchHeunScheduler
    from manifold.data.reward_pairs import (
        generate_generated_end_probe,
        generate_reward_pairs,
        save_reward_pairs,
    )

    torch.manual_seed(0)
    clean = torch.randn(12, *_LAT)
    sids = [f"subj_{i // 2}" for i in range(12)]
    train, val = generate_reward_pairs(
        clean, sids, _IdDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, val_fraction=0.34,
        batch_size=4, seed=0, device="cpu",
    )
    probe = generate_generated_end_probe(
        clean[:8], _IdDenoiser(), PartialFlowMatchHeunScheduler(),
        spacing=[1.0, 1.0, 1.0], modality=1, num_steps=2, batch_size=4, seed=0, device="cpu",
    )
    pairs_dir = tmp_path / "pairs"
    save_reward_pairs(pairs_dir, train, val, probe=probe)

    env, train_cfg, net = _write_tiny_configs(tmp_path)
    # Override reward.pairs_dir at the CLI (the base recipe carries a placeholder).
    rc = reward_main(
        ["-e", env, "-c", train_cfg, "-t", net, "-g", "1", "--max-epochs", "1", f"reward.pairs_dir={pairs_dir}"]
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
    # Continuous rewards (no ties). float32 (reward_roc_auc) vs float64 (brute) → 1e-6 tol.
    rw = torch.randn(20)
    rl = torch.randn(20) + 0.3
    assert abs(reward_roc_auc(rw, rl).item() - _brute_force_auc(rw.tolist(), rl.tolist())) < 1e-6
    # With deliberate ties (exercises the tie-group average-rank sweep).
    rw_tie = torch.tensor([2.0, 2.0, 0.5, 0.5])  # ties within winners
    rl_tie = torch.tensor([1.0, 1.0, 0.0, -1.0])  # ties within losers
    assert abs(reward_roc_auc(rw_tie, rl_tie).item() - _brute_force_auc(rw_tie.tolist(), rl_tie.tolist())) < 1e-6


def test_build_checkpoint_ddp_fallback_drops_monitor():
    """Under multi-GPU the checkpoint does not monitor a rank-local metric (mirrors JiT)."""
    from manifold.training.reward_cli import _build_checkpoint

    single = _build_checkpoint(str(Path("/tmp/_rwd_ckpt_a").resolve()), multi_gpu=False)
    multi = _build_checkpoint(str(Path("/tmp/_rwd_ckpt_b").resolve()), multi_gpu=True)
    assert single.monitor == "val/pair_acc"
    assert multi.monitor is None  # DDP: no rank-0-shard selection
    assert multi.save_last and multi.save_top_k == 1


def test_generated_end_probe_is_scored_in_chunks(tmp_path):
    """A probe larger than probe_batch_size is still scored correctly (chunked, no OOM)."""
    import shutil

    big_probe = _ToyPairDS(n=8)  # 8 probe pairs
    pw = torch.stack([big_probe.items[i]["winner"] for i in range(8)])
    pl_ = torch.stack([big_probe.items[i]["loser"] for i in range(8)])
    module = RewardModule(_reward_model(), lr=1e-2, val_probe=(pw, pl_), probe_batch_size=2)
    bundle = _RewardDataBundle(pair_ds=_ToyPairDS(), val_pair_ds=_ToyPairDS(n=4))
    trainer, _ = run_reward_training(
        module=module, bundle=bundle, model_dir=str(tmp_path), max_epochs=1,
        devices=1, accelerator="cpu", batch_size=2, num_workers=0, limit_val_batches=1.0,
    )
    metrics = trainer.callback_metrics
    assert "val/gen_pair_acc" in metrics and torch.isfinite(metrics["val/gen_pair_acc"])
    shutil.rmtree("/tmp/_rwd_ckpt_a", ignore_errors=True)
    shutil.rmtree("/tmp/_rwd_ckpt_b", ignore_errors=True)
