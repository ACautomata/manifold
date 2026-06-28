"""OmegaConf experiment-config layer: 3-file merge -> ``DictConfig`` + overrides.

The run-driver above the JSON *component* config (ADR-0004). Composes
env (paths) -> train (recipe) -> network (construction) YAML into one
``DictConfig``: each later file overwrites earlier **top-level** keys WHOLE
(not deep-merged), ``_base_`` composition deep-merges inside a single file,
required paths are ``???`` (OmegaConf MISSING — fail-fast on read), and CLI
``--<key>`` flags + Hydra-style dotlist overrides layer on top (dotlist wins).

The composed config **builds** components at launch (see
:mod:`manifold.config.builder`); it never persists. Component config stays the
JSON ``config.json`` each component writes via ``register_to_config`` /
``ConfigMixin`` and ``from_pretrained`` / ``save_pretrained`` round-trips
(ADR-0004) — independent of how the component was launched.

Path semantics
--------------
- **Required** paths are ``???`` (OmegaConf MISSING) in the YAML. Reading one
  before it is set raises ``MissingMandatoryValue`` — the loud failure we want.
  Call :func:`require_paths` to surface a helpful message up front.
- **Optional** paths are ``null``. :func:`opt` returns ``None`` for a ``null``,
  ``???``, or absent key alike (unlike ``getattr``, which RAISES on ``???``).

CLI overrides
-------------
- Legacy ``--<key> <value>`` flags (a flat ``{key: value}`` dict).
- Hydra-style dotlist trailing tokens: ``model_dir=/x diffusion_unet_train.lr=1e-4``.

Precedence: base config < ``--<key>`` flags < dotlist (dotlist wins).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

#: Filesystem paths that MUST be set for a training / convert run. ``???`` in the
#: env YAML; :func:`require_paths` asserts none remain MISSING before components
#: are built. (A pipeline-build demo reads no paths, so it tolerates ``???`` env.)
REQUIRED_PATH_KEYS: tuple[str, ...] = (
    "data_base_dir",
    "model_dir",
    "model_filename",
    "trained_autoencoder_path",
)

#: Top-level key listing base config(s) this file composes from (relative to the
#: file's directory, or absolute). A variant only spells what it changes.
_BASE_KEY = "_base_"


def _load_one(path: str) -> DictConfig:
    """Load one YAML/JSON file as a ``DictConfig``, resolving ``_base_`` composition.

    If the file declares ``_base_: [paths...]`` (or a single string), those base
    files are loaded first (recursively) and deep-merged, then this file's keys
    override — so a variant only spells what it changes. ``_base_`` itself is
    stripped from the result so it never leaks into the built config.
    """
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__} in {path}")
    base_ref = cfg.pop(_BASE_KEY, None)
    if base_ref is None:
        return cfg
    bases = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    here = Path(path).resolve().parent
    merged = OmegaConf.create()
    for b in bases:
        bp = b if Path(b).is_absolute() else str(here / b)
        merged = cast(DictConfig, OmegaConf.merge(merged, _load_one(bp)))
    return cast(DictConfig, OmegaConf.merge(merged, cfg))


def load_config(env_path: str, train_path: str | None, network_path: str) -> DictConfig:
    """Merge env -> train -> network YAML/JSON into one ``DictConfig``.

    Each later file overwrites earlier top-level keys WHOLE (not deep-merged) —
    so a later file's ``diffusion_unet`` block replaces the earlier one outright,
    never silently keeping stale sub-keys the trainer/sampler would then consume.
    Deep-merge is still used for ``_base_`` composition inside :func:`_load_one`
    (the intentional inheritance/extends layer, not the file merge).

    ``train_path`` may be ``None`` (a pipeline-build demo has no recipe); the
    network file is always required (it carries the construction kwargs).
    """
    configs = [_load_one(env_path), _load_one(network_path)]
    if train_path is not None:
        configs.insert(1, _load_one(train_path))
    return _merge_top_level(*configs)


def _merge_top_level(*configs: DictConfig) -> DictConfig:
    """Left-to-right top-level key replacement (later wins, wholesale).

    Distinct from :func:`OmegaConf.merge`, which recurses into nested mappings
    and would keep sub-keys the later file omits. Here a later file's value for a
    top-level key REPLACES the earlier one outright.
    """
    merged: dict[str, Any] = {}
    for cfg in configs:
        container = OmegaConf.to_container(cfg, resolve=False)
        if isinstance(container, dict):
            merged.update({str(k): v for k, v in container.items()})
    return cast(DictConfig, OmegaConf.create(merged))


def merge_overrides(
    base: DictConfig, flag_overrides: dict[str, Any], dotlist: list[str]
) -> DictConfig:
    """Apply ``--<key>`` flag overrides then dotlist overrides onto *base*.

    Precedence: base < flag_overrides < dotlist (dotlist wins). Each flag
    override is a flat ``key=value``; dotlist tokens are Hydra-style dotted
    paths (``diffusion_unet_train.lr=1e-4``).
    """
    merged = base
    if flag_overrides:
        merged = cast(DictConfig, OmegaConf.merge(merged, OmegaConf.create(flag_overrides)))
    if dotlist:
        merged = cast(DictConfig, OmegaConf.merge(merged, OmegaConf.from_dotlist(dotlist)))
    return merged


def require_paths(cfg: DictConfig, keys: tuple[str, ...] = REQUIRED_PATH_KEYS) -> None:
    """Raise ``ValueError`` if any required path is still ``???`` (MISSING) or null.

    Call before building components so the failure points at the unset path with
    CLI-override guidance, not at a deep torch/MONAI call later.
    """
    OmegaConf.resolve(cfg)
    missing = [k for k in keys if OmegaConf.is_missing(cfg, k) or OmegaConf.select(cfg, k) is None]
    if missing:
        raise ValueError(
            "Missing required config path(s): "
            + ", ".join(f"{k}" for k in missing)
            + ". Set them in the env config or override on the CLI, e.g. "
            + " ".join(f"{k}=<path>" for k in missing)
            + "."
        )


def opt(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Read an OPTIONAL config key safely.

    Returns *default* for a MISSING (``???``), ``null``, or absent key — unlike
    ``getattr(cfg, key, default)``, which RAISES ``MissingMandatoryValue`` on a
    ``???`` key.
    """
    return OmegaConf.select(cfg, key, default=default)
