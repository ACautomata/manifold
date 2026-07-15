"""Build ``spt.data.DataModule`` (train/val) from a :class:`MedicalDataset`.

``spt.data.DataModule`` accepts plain torch ``DataLoader``s directly, so this is
a thin factory. DDP data distribution is delegated to Lightning's
``DistributedSampler`` (auto-installed by the Trainer when ``strategy="ddp"``);
per-rank latent *encoding* sharding is handled inside
:meth:`~manifold.data.LatentDataset.warm_cache`.
"""

from __future__ import annotations

import logging

import stable_pretraining as spt
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .base import MedicalDataset
from .warm_datamodule import _ddp_eval_sampler

_log = logging.getLogger(__name__)


def build_datamodule(
    train_dataset: MedicalDataset,
    batch_size: int,
    val_dataset: MedicalDataset | None = None,
    *,
    num_workers: int = 0,
    shuffle_train: bool = True,
    allow_train_as_val: bool = False,
) -> LightningDataModule:
    """Wrap train/val datasets in a ``stable_pretraining.data.DataModule``.

    ``num_workers`` defaults to 0: latent datasets serve tiny in-RAM tensors with
    no I/O to hide, and this avoids duplicating the cache into worker processes.

    A held-out ``val_dataset`` is REQUIRED for validation - reusing the train set
    silently reports train metrics as validation (train/val leakage). When
    ``val_dataset is None`` the factory raises unless ``allow_train_as_val=True``
    (a smoke-only opt-in that reuses train with a loud warning; never set this on
    a run whose ``val/*`` metrics inform decisions).
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
    )
    if val_dataset is None:
        if not allow_train_as_val:
            raise ValueError(
                "No held-out validation dataset provided. Pass a disjoint val_dataset "
                "(a subject-level split - reusing train as val leaks train metrics into "
                "validation). Set allow_train_as_val=True ONLY for smoke tests that do not "
                "rely on val/* metrics."
            )
        _log.warning(
            "build_datamodule: val_dataset is None; reusing the TRAIN set as val "
            "(allow_train_as_val=True). val/* metrics are NOT held-out - do not use "
            "them for selection."
        )
        val_source = train_dataset
    else:
        val_source = val_dataset
    val_loader = DataLoader(
        val_source, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        sampler=_ddp_eval_sampler(val_source),
    )

    return spt.data.DataModule(train=train_loader, val=val_loader)
