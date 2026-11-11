# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains functions that act on operators and tapes.

.. currentmodule:: pennylane

.. autosummary::
    :toctree: api

    ~ops.functions.bind_new_parameters
    ~dot
    ~eigvals
    ~equal
    ~assert_equal
    ~evolve
    ~generator
    ~is_commuting
    ~is_hermitian
    ~is_unitary
    ~map_wires
    ~matrix
    ~simplify
    ~iterative_qpe
    ~commutator
    ~comm
    ~single_qubit_zyz_angles
    ~subcircuit

"""

from collections.abc import Callable
from importlib import import_module as _import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .is_unitary import is_unitary
from .iterative_qpe import iterative_qpe
from .subcircuit import subcircuit

if TYPE_CHECKING:
    from .bind_new_parameters import bind_new_parameters
    from .equal import equal, assert_equal
    from .commutator import comm, commutator
    from .dot import dot
    from .eigvals import eigvals
    from .evolve import evolve
    from .generator import generator
    from .is_commuting import is_commuting
    from .is_hermitian import is_hermitian
    from .map_wires import map_wires
    from .matrix import matrix
    from .simplify import simplify
    from .single_qubit_zyz_angles import single_qubit_zyz_angles

# Built-in operators import subcircuit while ops and templates are still loading.
# Defer utilities whose imports and dispatch registrations need those classes.
_LAZY_IMPORTS = {
    "bind_new_parameters": "bind_new_parameters",
    "equal": "equal",
    "assert_equal": "equal",
    "comm": "commutator",
    "commutator": "commutator",
    "dot": "dot",
    "eigvals": "eigvals",
    "evolve": "evolve",
    "generator": "generator",
    "is_commuting": "is_commuting",
    "is_hermitian": "is_hermitian",
    "map_wires": "map_wires",
    "matrix": "matrix",
    "simplify": "simplify",
    "single_qubit_zyz_angles": "single_qubit_zyz_angles",
}

__all__ = [*_LAZY_IMPORTS, "is_unitary", "iterative_qpe", "subcircuit"]


def __getattr__(name: str) -> Callable:
    """Load an operator utility when it is first imported."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = _LAZY_IMPORTS[name]
    module = _import_module(f".{module_name}", __name__)
    # Dependencies can also set package attributes to modules, e.g. is_hermitian imports matrix.
    loaded = {owner: globals().get(owner) for owner in _LAZY_IMPORTS.values()}
    loaded[module_name] = module
    for exported, owner in _LAZY_IMPORTS.items():
        if isinstance(loaded[owner], ModuleType):
            globals()[exported] = getattr(loaded[owner], exported)
    return globals()[name]
