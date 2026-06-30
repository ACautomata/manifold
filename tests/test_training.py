"""Trainer / EMA / metrics tests (Slice C, issue #26).

A tiny CPU ``Trainer.fit`` (1–2 epochs, tiny UNet + an in-RAM latent cache)
exercises the whole stack: ``train/loss_epoch`` / ``train/grad_norm`` /
``val/x0_mae`` log finite, the double-EMA shadows populate, ``swap_in`` changes
the decoded output vs the raw weights, and the EMA ``state_dict`` round-trips.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from manifold import (
    AutoencoderKL,
)
from manifold.training import (
    DoubleEMACallback,
    LatentX0MAE,
    TrainLossLogger,
    build_trainer,
)


class _FakeLatentDataset(Dataset):
    """In-RAM ``{"latent","spacing","label"}`` cache — the warmed-cache stand-in."""

    def __init__(self, n: int = 6):
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
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _datamodule(n=6, batch_size=2):
    ds = _FakeLatentDataset(n)
    train = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    import stable_pretraining as spt

    return spt.data.DataModule(train=train, val=val)


def test_double_ema_state_dict_roundtrips(latent_module):
    cb = DoubleEMACallback(latent_module)
    # Populated shadows start == raw, then drift after an update.
    raw_named = dict(latent_module.unet.named_parameters())
    # Nudge a parameter so the optimizer-step surrogate makes raw != shadow.
    with torch.no_grad():
        next(iter(latent_module.unet.parameters())).add_(0.5)
    cb.update(latent_module)
    slow = cb.state_dict()["shadows"][cb.slow_index]
    assert any(not torch.equal(slow[n], raw_named[n]) for n in raw_named)

    # Round-trip: a fresh callback loads the captured shadows exactly.
    state = cb.state_dict()
    cb2 = DoubleEMACallback(latent_module)
    cb2.load_state_dict(state)
    restored = cb2.state_dict()["shadows"][cb2.slow_index]
    for n in raw_named:
        assert torch.equal(restored[n], slow[n])


def test_swap_in_changes_decode_vs_raw(latent_module):
    """swap_in copies the slow EMA shadow in; decode then differs from raw."""
    vae = AutoencoderKL(scaling_factor=0.5)
    cb = DoubleEMACallback(latent_module)

    # Make the slow shadow materially different from the raw weights, then
    # swap it in: generation/decode must reflect the swapped weights.
    with torch.no_grad():
        for n, p in latent_module.unet.named_parameters():
            cb.state_dict()  # ensure shadows exist (constructed in __init__)
    # Force a drift: step the EMA toward perturbed params repeatedly.
    for _ in range(20):
        with torch.no_grad():
            for p in latent_module.unet.parameters():
                p.add_(0.1)
        cb.update(latent_module)

    gen = torch.Generator().manual_seed(0)
    decode_raw = vae.decode(
        latent_module.sample(
            (1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, num_inference_steps=2, generator=gen
        )
    ).clone()
    # Same seed → same latent input, so a decode difference is purely the weights.
    cb.swap_in(latent_module)
    try:
        gen2 = torch.Generator().manual_seed(0)
        decode_ema = vae.decode(
            latent_module.sample(
                (1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, num_inference_steps=2, generator=gen2
            )
        )
        assert not torch.allclose(decode_ema, decode_raw)
    finally:
        cb.restore(latent_module)
    # After restore, decode matches the raw decode again.
    gen3 = torch.Generator().manual_seed(0)
    decode_restored = vae.decode(
        latent_module.sample(
            (1, 4, 4, 4, 4), [1.0, 1.0, 1.0], 1, num_inference_steps=2, generator=gen3
        )
    )
    assert torch.allclose(decode_restored, decode_raw)


def test_trainer_fit_logs_metrics_and_populates_ema(latent_module, tmp_path):
    """A tiny CPU fit logs train/loss_epoch / train/grad_norm / val/x0_mae finite."""
    ema = DoubleEMACallback(latent_module)
    metrics = [TrainLossLogger(), LatentX0MAE()]
    trainer = build_trainer(
        max_epochs=2,
        callbacks=[ema, *metrics],
        model_dir=str(tmp_path),
        devices=1,
        accelerator="cpu",
        limit_val_batches=2,
        extra_kwargs=dict(
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,  # assertions read trainer.callback_metrics directly
        ),
    )
    trainer.fit(latent_module, datamodule=_datamodule())

    m = trainer.callback_metrics
    assert torch.is_tensor(m["train/loss_epoch"]) and torch.isfinite(m["train/loss_epoch"])
    assert "val/x0_mae" in m and torch.isfinite(m["val/x0_mae"])
    # train/grad_norm is logged on step; the last value remains in callback_metrics.
    assert "train/grad_norm" in m and torch.isfinite(m["train/grad_norm"])

    # The EMA shadows populated (the last-step value differs from the init clone).
    raw_first = next(iter(latent_module.unet.parameters())).detach()
    slow = ema.state_dict()["shadows"][ema.slow_index]
    first_name = next(iter(latent_module.unet.named_parameters()))[0]
    assert not torch.equal(slow[first_name], raw_first)
