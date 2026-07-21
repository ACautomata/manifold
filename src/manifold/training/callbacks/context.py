"""The runtime-objects bag a :class:`CallbackSpec` builds from.

:class:`CallbackContext` is the typed payload :func:`run_training` populates as
its runtime objects (module, vae, datamodule, inference recipe, model dir, seed)
materialize, then hands to ``CallbackSpec.build`` so deferred callback
construction (ADR-0029) can inject them. Only the objects ``resolve`` / ``build``
need live here; a spec reads only what it uses (e.g. ``TrainLossSpec`` reads
nothing — its logger carries no config).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CallbackContext:
    """Runtime objects a ``CallbackSpec.build(ctx)`` may inject.

    Fields are ``Any``-typed to keep the callbacks package free of import cycles
    back into the module / data / metrics layers (the real types are noted
    below; a spec's ``build`` casts as needed).

    Attributes:
        module: the ``LatentFlowModule`` being trained.
        vae: the held-out VAE used to decode latents (FID / generative specs).
        datamodule: the ``LatentWarmDataModule`` (lazy real-latents source).
        inference_recipe: the FID sampling-knob dict, or ``None`` when FID is off.
        model_dir: the run output directory (checkpoints, logs).
        seed: the run seed (deterministic sampling).
        feature_net: the FID feature network (RadImageNet, or a test fake) injected
            directly by the CLI/test seam; ``None`` -> the spec uses
            ``feature_net_factory``'s lazy build.
        feature_net_factory: the lazy fail-safe feature-net builder (rank-0-gated).
        real_latents: GRPO's FID reference (ADR-0032); ``None`` for JiT, which
            pulls lazily from ``datamodule`` instead.
    """

    module: Any
    vae: Any
    datamodule: Any
    inference_recipe: dict | None
    model_dir: str
    seed: int
    feature_net: Any = None
    feature_net_factory: Any = None
    real_latents: Any = None
