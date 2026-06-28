"""General dataset interface — the seam between data and trainers.

Trainers (the deferred trainer stack) consume **only** the dict keys defined
here; they never import NIfTI, BraTS, or any dataset-specific logic. Dataset
implementations (:class:`~manifold.data.NiftiVolumeDataset`,
:class:`~manifold.data.LatentDataset`, …) and label providers
(:mod:`manifold.data.labels`) are pluggable and live alongside.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

import torch
from torch import Tensor


class SampleDict(TypedDict, total=False):
    """Dict-batch contract emitted by every dataset (stable-pretraining is dict-first).

    Exactly one of ``image`` / ``latent`` is present:

    - ``image``:     ``(C, D0, D1, D2)`` float, normalized to ~[0, 1], for VAE encode.
    - ``latent``:    ``(C, d0, d1, d2)`` float, VAE latent. **Scaled** by the
                      data stack (scale-on-read, ADR-0003) — the diffusion Module
                      and Pipeline never reference ``scaling_factor``.
    - ``spacing``:   ``(3,)`` float, physical voxel size the UNet conditions on.
    - ``label``:     ``()`` long int, class label (modality / contrast) for the class embedding.
    - ``sample_id``: ``str``, stable id (cache keys).
    - ``meta``:      ``dict``, free-form metadata (filename, manifest fields, …).
    """

    image: Tensor
    latent: Tensor
    spacing: Tensor
    label: Tensor
    sample_id: str
    meta: dict[str, Any]


@runtime_checkable
class LabelProvider(Protocol):
    """Map a sample's filename/metadata to an integer class label, or ``None`` to skip.

    The decoupling seam: BraTS contrast detection, CT/MR modality codes, or a
    fixed label are each *one* provider. Returning ``None`` means "this file is
    not a valid training input" (e.g. a BraTS segmentation mask) and the dataset
    drops it.
    """

    def __call__(self, filename: str, meta: dict[str, Any]) -> int | None: ...


class MedicalDataset(torch.utils.data.Dataset):
    """Base class for datasets emitting :class:`SampleDict`.

    Subclasses implement ``__getitem__`` (and ``__len__``) and may override
    :meth:`label_counts` when the label distribution is meaningful.
    """

    def __len__(self) -> int:
        raise NotImplementedError

    def label_counts(self) -> dict[int, int]:
        """Return ``{label: count}`` over the dataset, or raise if unsupported."""
        raise NotImplementedError(
            f"{type(self).__name__} does not expose label_counts; override it "
            "if the label distribution is needed."
        )
