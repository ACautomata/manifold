"""Deletion guards for the condition-aware paired-reward pipeline (ADR-0034, issue #178).

The whole pipeline is deleted — ``PairedRewardModule``, ``paired_reward_cli``,
``paired_reward_pairs``, the ``config_paired_reward.yaml`` recipe, the
``manifold-train-paired-reward`` console entry point, and its tests. These guards
assert that deletion is permanent: the modules no longer import, the entry point is
no longer registered, and the ``data`` / ``training`` / ``modules`` packages no longer
re-export the paired-reward symbols (the two relocated survivors —
``load_frozen_controlnet_generator`` and ``_train_val_manifests`` — stay exported from
their new homes and are NOT touched here).

The entry-point guard reads ``pyproject.toml`` (the single source of truth that the
console scripts are generated from) rather than ``importlib.metadata``, so it reflects
the working tree and does not depend on the package being re-installed.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - the dev/CI env runs Python >= 3.11
    import tomli as tomllib  # type: ignore[no-redef]

#: Every module that made up the paired-reward pipeline and is now deleted.
_DELETED_MODULES = (
    "manifold.modules.paired_reward",
    "manifold.data.paired_reward_pairs",
    "manifold.training.paired_reward_cli",
)

#: The deleted console entry point (was manifold.training.paired_reward_cli:main).
_DELETED_SCRIPT = "manifold-train-paired-reward"

#: Paired-reward symbols that the package ``__init__`` modules used to re-export.
_DELETED_EXPORTS = {
    "manifold.modules": ("PairedRewardModule", "PairedRewardBatch"),
    "manifold.training": ("run_paired_reward_training",),
    "manifold.data": (
        "build_paired_reward_pairs",
        "build_paired_reward_probe",
    ),
}


def _console_scripts() -> dict[str, str]:
    """The ``[project.scripts]`` entry points declared in ``pyproject.toml``."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        data = tomllib.load(fh)
    return dict(data["project"]["scripts"])


def test_paired_reward_console_script_is_removed():
    """The manifold-train-paired-reward console entry point is no longer registered."""
    assert _DELETED_SCRIPT not in _console_scripts(), (
        f"{_DELETED_SCRIPT!r} is still declared in pyproject.toml [project.scripts] "
        f"- the paired-reward CLI was deleted (ADR-0034)."
    )


@pytest.mark.parametrize("module_name", _DELETED_MODULES)
def test_paired_reward_module_import_raises(module_name):
    """Importing each deleted paired-reward module raises ImportError."""
    with pytest.raises(ImportError):
        importlib.import_module(module_name)


@pytest.mark.parametrize(
    "package_name, names", list(_DELETED_EXPORTS.items())
)
def test_package_init_no_longer_exports_paired_reward(package_name, names):
    """The data/training/modules packages no longer re-export paired-reward symbols."""
    package = importlib.import_module(package_name)
    leaked = [name for name in names if hasattr(package, name)]
    assert not leaked, (
        f"{package_name} still exports paired-reward symbols {leaked!r} "
        f"(ADR-0034 deleted the pipeline)."
    )


def test_relocated_survivors_still_exported():
    """The two relocated survivors stay exported from their new homes (ADR-0034).

    ``load_frozen_controlnet_generator`` moved to ``training.controlnet_inputs``
    (re-exported from ``manifold.data``); ``_train_val_manifests`` moved to
    ``data.paired_manifests``. Deleting the paired-reward pipeline must NOT take them.
    """
    from manifold.data import paired_manifests
    from manifold.training import controlnet_inputs

    assert hasattr(paired_manifests, "_train_val_manifests")
    assert hasattr(controlnet_inputs, "load_frozen_controlnet_generator")
    # load_frozen_controlnet_generator stays reachable via the data-package surface.
    import manifold.data

    assert hasattr(manifold.data, "load_frozen_controlnet_generator")
