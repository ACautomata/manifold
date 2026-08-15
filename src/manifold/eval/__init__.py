"""Offline before/after-GRPO comparison eval (ADR-0036).

The injectable driver core (issue #228): given a before pipeline and an after
pipeline plus a fixed seed + conditioning, it generates both sides under identical
initial noise (the fairness invariant), decodes via the frozen VAE, min-max
normalizes to ``[0, 1]``, renders 2.5D three-plane slice grids, and — for the paired
ControlNet policy — scores 3D PSNR/SSIM against the real target. The console entry
that wires this to real checkpoints (issue #229) and the Artifact page builder
(issue #230) are layered on top.
"""

from .before_after import BeforeAfterEval, BeforeAfterResult
from .slice_grid import SliceGrid

__all__ = [
    "BeforeAfterEval",
    "BeforeAfterResult",
    "SliceGrid",
]
