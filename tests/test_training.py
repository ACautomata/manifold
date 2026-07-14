"""Trainer / metrics tests (Slice C, issue #26).

A tiny CPU ``Trainer.fit`` (1–2 epochs, tiny UNet + an in-RAM latent cache)
exercises the whole stack: ``train/loss_epoch`` / ``train/grad_norm`` /
``val/x0_mae`` log finite.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from manifold.training import (
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


def test_trainer_fit_logs_metrics(latent_module, tmp_path):
    """A tiny CPU fit logs train/loss_epoch / train/grad_norm / val/x0_mae finite."""
    metrics = [TrainLossLogger(), LatentX0MAE()]
    trainer = build_trainer(
        max_epochs=2,
        callbacks=metrics,
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
