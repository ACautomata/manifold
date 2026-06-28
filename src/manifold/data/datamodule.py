"""Build ``spt.data.DataModule`` (train/val) from a :class:`MedicalDataset`.

``spt.data.DataModule`` accepts plain torch ``DataLoader``s directly, so this is
a thin factory. DDP data distribution is delegated to Lightning's
``DistributedSampler`` (auto-installed by the Trainer when ``strategy="ddp"``);
per-rank latent *encoding* sharding is handled inside
:meth:`~manifold.data.LatentDataset.warm_cache`.
"""

from __future__ import annotations

import stable_pretraining as spt
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .base import MedicalDataset


def build_datamodule(
    train_dataset: MedicalDataset,
    batch_size: int,
    val_dataset: MedicalDataset | None = None,
    *,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> LightningDataModule:
    """Wrap train/val datasets in a ``stable_pretraining.data.DataModule``.

    ``num_workers`` defaults to 0: latent datasets serve tiny in-RAM tensors with
    no I/O to hide, and this avoids duplicating the cache into worker processes.
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    if val_dataset is None:
        # spt requires a val loader; reuse train when no separate split is given.
        val_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    else:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    return spt.data.DataModule(train=train_loader, val=val_loader)
