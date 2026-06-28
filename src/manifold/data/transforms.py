"""Volume preprocessing transforms (numpy/tensor ops, no I/O).

Ported verbatim from hope's ``data/transforms.py`` (itself ported from the
NV-Generate-CTMR reference + the BraTS / build-cache scripts). MR takes a
per-volume percentile window with no clip; CT takes a fixed HU window with a
hard clip. BraTS labels (34-37) are ``>= 8`` so they take the MR branch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def normalize_to_01(data: np.ndarray, modality: int) -> np.ndarray:
    """Rescale a volume to roughly [0, 1] using the modality-appropriate window.

    - **CT** (``modality < 8``): ``[-1000, 1000] -> [0, 1]`` with hard clip — HU
      is physically calibrated, so a fixed window is correct.
    - **MR** (``modality >= 8``): per-volume ``[p0, p99.5] -> [0, 1]`` with **no**
      clip (``clip=False``) — MR intensities have no fixed scale; leaving the
      high-intensity tail unclipped reproduces the dynamic range the VAE expects.

    For BraTS the class labels (34-37) are ``>= 8`` so they take the MR branch,
    matching the original trainer's behaviour exactly.
    """
    if modality >= 8:
        if data.size == 0:
            return np.zeros_like(data, dtype=np.float32)
        lo = float(np.percentile(data, 0.0))
        hi = float(np.percentile(data, 99.5))
        if hi <= lo:
            return np.zeros_like(data, dtype=np.float32)
        return ((data - lo) / (hi - lo)).astype(np.float32)
    return (np.clip(data, -1000, 1000) + 1000) / 2000.0


def pad_to_divisible(arr: np.ndarray, divisor: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Zero-pad spatial dims so each is a multiple of *divisor*.

    Returns the padded array and the original spatial shape (for unpadding).
    """
    orig_shape = arr.shape
    pad_widths = []
    for d in orig_shape:
        rem = d % divisor
        pad_widths.append((0, (divisor - rem) % divisor))
    if all(hi == 0 for _, hi in pad_widths):
        return arr, orig_shape
    padded = np.pad(arr, pad_widths, mode="constant", constant_values=0)
    return padded, orig_shape


def floor_to_divisible(arr: np.ndarray, divisor: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """Crop spatial dims down to the largest multiple of *divisor*.

    Returns the cropped array and the new spatial shape.
    """
    new_shape = tuple((d // divisor) * divisor for d in arr.shape)
    slices = tuple(slice(0, s) for s in new_shape)
    return np.ascontiguousarray(arr[slices]), new_shape


def resize_to(volume: np.ndarray, target_dim: tuple[int, int, int]) -> np.ndarray:
    """Trilinear-resize a 3D volume to *target_dim* (matches ``F.interpolate``)."""
    t = torch.from_numpy(volume)[None, None]  # [1, 1, D0, D1, D2]
    t = F.interpolate(t, size=list(target_dim), mode="trilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).numpy()
