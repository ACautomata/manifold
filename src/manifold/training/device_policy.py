"""The per-rank CUDA device decision (ADR-0035).

``DevicePolicy`` is the single owner of "which GPU this rank uses". It answers
only *which* device — how a model is *staged* onto it (``.to(device).eval()``
+ ``requires_grad_(False)``) belongs to the FrozenArm staging chain (ADR-0031
A1), and the two candidates stay decoupled.

Construction is side-effect free: ``__init__`` snapshots the launcher's
``LOCAL_RANK`` env var (missing -> 0) and touches neither CUDA nor the process
group. The snapshot is taken at construction; post-PG callers do NOT re-resolve
via ``dist.get_rank()``.

Three methods:

- :meth:`pin` — the pre-PG call for training shells: performs the one-time
  ``set_device`` side effect and returns ``cuda:{local_rank}``. The out-of-range
  guard (``local_rank >= device_count``) skips ``set_device``; when CUDA is
  unavailable it returns ``cpu`` untouched.
- :meth:`device` — side-effect free read of the same device, for read-only /
  debug paths.
- :meth:`warm_device` — the post-PG VAE-warm device resolution (the former
  ``resolve_warm_device`` free function in ``latent_pipeline``, behavior
  byte-identical): ``dist`` is imported lazily inside the method body so
  importing this module never touches the process group; under an initialized
  PG with a CUDA fallback it returns ``cuda:{local_rank}``, otherwise the
  fallback unchanged.
"""

from __future__ import annotations

import os

import torch

__all__ = ["DevicePolicy"]


class DevicePolicy:
    """Resolves the per-rank CUDA device from the launcher environment."""

    def __init__(self) -> None:
        # The launcher env var is snapshot at construction (missing -> 0); it is
        # NOT re-read post-PG, so the policy is stable across its lifetime.
        local_rank_env = os.environ.get("LOCAL_RANK")
        self._local_rank = int(local_rank_env) if local_rank_env is not None else 0

    def pin(self) -> torch.device:
        """Pin this rank to its own GPU: one ``set_device`` side effect, then
        return ``cuda:{local_rank}``. Out-of-range ``LOCAL_RANK`` (>= actual GPU
        count) skips ``set_device`` (twin guard semantics); no CUDA -> ``cpu``."""
        if torch.cuda.is_available():
            if self._local_rank < torch.cuda.device_count():
                torch.cuda.set_device(self._local_rank)
            return torch.device(f"cuda:{self._local_rank}")
        return torch.device("cpu")

    def device(self) -> torch.device:
        """The policy device, side-effect free (read-only / debug paths)."""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{self._local_rank}")
        return torch.device("cpu")

    def warm_device(self, fallback: torch.device) -> torch.device:
        """The device for the post-PG VAE warm: the per-rank local CUDA device
        under DDP, else the ``fallback`` unchanged (former ``resolve_warm_device``).

        The launch-time ``fallback`` is captured in the shell's ``main()`` BEFORE
        Lightning initializes the process group, so under DDP it is the default
        ``cuda:0`` (or whatever GPU index the rank's ``CUDA_VISIBLE_DEVICES``
        exposes as 0). After the PG is up (inside ``DataModule.setup()``),
        ``LOCAL_RANK`` names the rank's GPU: return ``cuda:{local_rank}``.
        Off-CUDA / single-process -> the ``fallback`` unchanged.

        ``dist`` is imported lazily inside the method body: importing this module
        never touches the process group. The device comes from the construction-time
        ``LOCAL_RANK`` snapshot (ADR-0035 contract: no ``dist.get_rank()``
        re-resolution post-PG); unset ``LOCAL_RANK`` -> the snapshot's 0 ->
        ``cuda:0`` (a torchrun launch always sets it).
        """
        import torch.distributed as dist

        if fallback.type == "cuda" and dist.is_initialized():
            # LOCAL_RANK (set by the launcher) names the rank's GPU.
            return torch.device(f"cuda:{self._local_rank}")
        return fallback
