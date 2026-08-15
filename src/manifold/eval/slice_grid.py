"""2.5D three-plane slice-grid renderer for the before/after comparison (issue #228).

Renders a handful of ``[0, 1]`` volumes as one PNG: **rows** are the three
orthogonal planes (xy / yz / zx — the same axis convention as
:func:`~manifold.metrics.get_features_2p5d`, axis D→xy, H→yz, W→zx), **columns** are
the volumes (``before | after`` for the unconditional JiT policy, ``before | after |
real`` for the paired ControlNet one). Each cell is the center slice of that volume
on that plane, so a 3D structure can be inspected without a 3D viewer.

matplotlib is imported lazily inside :meth:`SliceGrid.render` (the ``Agg`` backend,
headless) — but unlike the training plot callback, a missing matplotlib **raises**:
the slice grid is this driver's deliverable, so it must fail loudly rather than
degrade to a silent skip.
"""

from __future__ import annotations

import os
from typing import Sequence

from torch import Tensor

#: Plane row labels, in render order (axis 0=D→xy, 1=H→yz, 2=W→zx on a [D, H, W] vol).
_PLANES = ("xy", "yz", "zx")


class SliceGrid:
    """Render a 2.5D three-plane grid from ``[0, 1]`` volumes to one PNG.

    Args:
        dpi: render resolution.
    """

    def __init__(self, *, dpi: int = 150) -> None:
        self._dpi = int(dpi)

    @staticmethod
    def _center_slices(vol: Tensor) -> list[Tensor]:
        """The center slice of one ``[C, D, H, W]`` volume on each of the 3 planes.

        Channel 0 is taken (these pipelines decode single-channel medical volumes);
        each plane's center index is ``length // 2`` along its axis.
        """
        v = vol[0].detach().cpu()  # [D, H, W]
        d, h, w = v.shape
        return [
            v[d // 2, :, :],  # xy: perpendicular to D -> [H, W]
            v[:, h // 2, :],  # yz: perpendicular to H -> [D, W]
            v[:, :, w // 2],  # zx: perpendicular to W -> [D, H]
        ]

    def render(self, volumes: Sequence[Tensor], labels: Sequence[str], out_path: str) -> str:
        """Render ``volumes`` (each ``[C, D, H, W]``, ``[0, 1]``) as a grid PNG.

        Rows = the three planes, cols = one per volume, titled by ``labels``. The PNG
        is written atomically (``tmp`` + :func:`os.replace`), so an interrupted render
        cannot leave a truncated file. Returns ``out_path``.
        """
        import matplotlib

        matplotlib.use("Agg")  # headless — remote servers have no display.
        import matplotlib.pyplot as plt

        if len(volumes) != len(labels):
            raise ValueError(
                f"volumes ({len(volumes)}) and labels ({len(labels)}) must align."
            )
        ncols = len(volumes)
        fig, axes = plt.subplots(
            len(_PLANES), ncols, figsize=(3.2 * ncols, 3.0 * len(_PLANES)), squeeze=False
        )
        try:
            for col, (vol, label) in enumerate(zip(volumes, labels)):
                for row, plane in enumerate(self._center_slices(vol)):
                    ax = axes[row][col]
                    ax.imshow(
                        plane.numpy(), cmap="gray", vmin=0.0, vmax=1.0, aspect="auto"
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row == 0:
                        ax.set_title(label)
                    if col == 0:
                        ax.set_ylabel(_PLANES[row])
            fig.tight_layout()
            tmp_path = out_path + ".tmp"
            # format="png" explicitly: the .tmp suffix is not a recognized extension.
            fig.savefig(tmp_path, format="png", dpi=self._dpi, bbox_inches="tight")
            os.replace(tmp_path, out_path)  # atomic rename on the same filesystem.
        finally:
            plt.close(fig)
        return out_path
