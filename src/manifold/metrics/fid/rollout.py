"""FixedSampleRollout — rank-strided seeded generation for the synthetic FID arm.

Generates ``num_synth`` unconditional volumes through ``Module.sample``. Each
rank generates its strided slice (``seed + i`` for ``i % world == rank``) so the
global synthetic set is the union across ranks, not ``world × rank-0``. Generation
samples the live (raw) optimizer weights — no EMA swap (EMA training was removed;
ADR-0006).
"""

from __future__ import annotations

from typing import Generator

import torch


def _rank_world() -> tuple[int, int]:
    """Return (rank, world_size) from torch.distributed, or (0, 1) if not initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank()), int(torch.distributed.get_world_size())
    return 0, 1


class FixedSampleRollout:
    """Rank-strided seeded generation over the synthetic set.

    Stateless callable object — all config is in ``__init__``.

    Args:
        module: the ``LatentFlowModule`` (its ``.sample()`` generates).
        latent_shape / spacing / modality / num_inference_steps / guidance_scale
        / cfg_interval: the generation recipe.
        num_synth: synthetic volumes generated per run (re-seeded every epoch).
        seed: the re-seeded generation noise seed (fixed across epochs).
    """

    def __init__(
        self,
        *,
        module,
        latent_shape: tuple,
        spacing: list[float],
        modality: int,
        num_inference_steps: int,
        guidance_scale: float = 1.0,
        cfg_interval: list[float] | None = None,
        num_synth: int = 16,
        seed: int = 0,
    ) -> None:
        self._module = module
        self._latent_shape = tuple(latent_shape)
        self._spacing = spacing
        self._modality = int(modality)
        self._num_inference_steps = int(num_inference_steps)
        self._guidance_scale = float(guidance_scale)
        self._cfg_interval = cfg_interval
        self._num_synth = int(num_synth)
        self._seed = int(seed)

    @torch.no_grad()
    def __call__(self, device: torch.device) -> Generator[torch.Tensor, None, None]:
        """Generate this rank's strided slice of ``num_synth`` as a lazy
        generator — each latent is yielded and released before the next,
        preventing OOM from retaining all latents simultaneously (codex #171 P2).

        Yields:
            ``[1, C, D, H, W]`` latent tensors (one per assigned index).
        """
        rank, world = _rank_world()
        for i in range(rank, self._num_synth, world):
            gen = torch.Generator(device=device).manual_seed(self._seed + i)
            yield self._module.sample(
                self._latent_shape,
                self._spacing,
                self._modality,
                self._num_inference_steps,
                guidance_scale=self._guidance_scale,
                cfg_interval=self._cfg_interval,
                generator=gen,
            )
