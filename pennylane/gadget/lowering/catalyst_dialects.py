# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the loader for Catalyst's xDSL dialect definitions.

Catalyst's QEC dialects are defined in Python modules under
``frontend/catalyst/python_interface/dialects/`` that depend only on xDSL. When Catalyst is
installed they are imported from it. Otherwise they are loaded from a Catalyst source
checkout located by the ``CATALYST_SRC`` environment variable (or ``~/catalyst``), so that
emitted IR can be verified against Catalyst's own definitions during development.
"""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType

_CANDIDATES = (
    os.environ.get("CATALYST_SRC"),
    str(Path.home() / "catalyst"),
)

_DIALECT_SUBPATH = Path("frontend/catalyst/python_interface/dialects")


class CatalystNotFound(RuntimeError):
    """Raised when neither an installed Catalyst nor a source checkout can be located."""


def _checkout() -> Path | None:
    for candidate in _CANDIDATES:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / _DIALECT_SUBPATH / "qecl.py").is_file():
            return root
    return None


@lru_cache(maxsize=None)
def load_dialect_module(name: str) -> ModuleType:
    """Load one of Catalyst's dialect definition modules.

    The result is cached. This is required for correctness: loading a module twice creates
    two distinct copies of its types, and values of one copy fail type checks against the
    other even though they print identically.

    Args:
        name (str): module name, for example ``"qecl"`` or ``"qecp"``

    Returns:
        ModuleType: the dialect module

    Raises:
        CatalystNotFound: if Catalyst is not importable and no source checkout is found
    """
    try:
        module = importlib.import_module(f"catalyst.python_interface.dialects.{name}")
        return module
    except ImportError:
        pass

    root = _checkout()
    if root is None:
        raise CatalystNotFound(
            f"cannot load Catalyst's {name} dialect: catalyst is not importable and no "
            "source checkout was found. Either install catalyst, or set CATALYST_SRC to a "
            "checkout containing frontend/catalyst/python_interface/dialects/"
        )
    path = root / _DIALECT_SUBPATH / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_catalyst_{name}", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise CatalystNotFound(f"cannot load a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_of(module: ModuleType) -> str:
    """The file a loaded dialect module was loaded from.

    Args:
        module (ModuleType): the dialect module

    Returns:
        str: the path, or ``"<unknown>"``
    """
    return getattr(module, "__file__", "<unknown>")


__all__ = ["CatalystNotFound", "load_dialect_module", "source_of"]
