"""OmegaConf experiment-config layer (ADR-0004) — composes + builds components.

The run-driver above the JSON *component* config: composes env -> train ->
network YAML (top-level replace, ``_base_`` inheritance, ``???`` fail-fast,
CLI/dotlist overrides) and builds the manifold components at launch. It never
persists; component config stays JSON via ``ConfigMixin`` round-trips.
"""

from .builder import (
    autoencoder_divisor,
    build_pipeline,
    build_scheduler,
    build_unet,
    build_vae,
)
from .loader import (
    REQUIRED_PATH_KEYS,
    load_config,
    merge_overrides,
    opt,
    require_paths,
)

__all__ = [
    "REQUIRED_PATH_KEYS",
    "autoencoder_divisor",
    "build_pipeline",
    "build_scheduler",
    "build_unet",
    "build_vae",
    "load_config",
    "merge_overrides",
    "opt",
    "require_paths",
]
