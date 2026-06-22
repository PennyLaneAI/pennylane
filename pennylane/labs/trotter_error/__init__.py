# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains tools for estimating the Trotter
error of product formulas used in Hamiltonian simulation.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.


Abstract Base Classes
~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.trotter_error

.. autosummary::
    :toctree: api

    ~Fragment
    ~TrotterState

Fragments
~~~~~~~~~

.. currentmodule:: pennylane.labs.trotter_error

.. autosummary::
    :toctree: api

    ~NumpyFragment
    ~NumpyState
    ~sparse_fragments
    ~vibrational_fragments
    ~vibronic_fragments

Product Formulas
~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.trotter_error

.. autosummary::
    :toctree: api

    ~ProductFormula
    ~ImportanceConfig
    ~bch_expansion
    ~effective_hamiltonian
    ~perturbation_error

Realspace Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.trotter_error

.. autosummary::
    :toctree: api

    ~HOState
    ~VibronicHO
    ~RealspaceCoeffs
    ~RealspaceMatrix
    ~RealspaceOperator
    ~RealspaceSum

"""

from .abstract import TrotterState, Fragment
from .fragments import (
    NumpyFragment,
    NumpyState,
    sparse_fragments,
    vibrational_fragments,
    vibronic_fragments,
)
from .product_formulas import (
    ImportanceConfig,
    ProductFormula,
    bch_expansion,
    effective_hamiltonian,
    perturbation_error,
)
from .realspace import (
    HOState,
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
    VibronicHO,
)

__all__ = [
    "TrotterState",
    "Fragment",
    "NumpyFragment",
    "NumpyState",
    "sparse_fragments",
    "vibrational_fragments",
    "vibronic_fragments",
    "bch_expansion",
    "ImportanceConfig",
    "effective_hamiltonian",
    "perturbation_error",
    "ProductFormula",
    "HOState",
    "VibronicHO",
    "RealspaceCoeffs",
    "RealspaceMatrix",
    "RealspaceOperator",
    "RealspaceSum",
]
