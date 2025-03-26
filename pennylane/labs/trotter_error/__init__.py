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
"""Module containing functionality for computing Trotter error.

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

.. currentmodule:: pennylane.labs.trotter_error

Trotter Base Classes
~~~~~~~~~~~~~~~~~~~~
test

.. autosummary::
    :toctree: api

    ~AbstractState
    ~Fragment

Fragment Classes
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~GenericFragment
    ~RealspaceMatrix
    ~RealspaceSum

Realspace Hamiltonian Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~RealspaceCoeffs
    ~RealspaceOperator

Fragment Functions:
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~generic_fragments
    ~vibrational_fragments
    ~vibronic_fragments

State Classes:
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~HOState
    ~VibronicHO

Error Estimation Functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~perturbation_error
    ~trotter_error
"""

from .abstract import AbstractState, Fragment
from .fragments import GenericFragment, generic_fragments, vibrational_fragments, vibronic_fragments
from .product_formulas import perturbation_error, trotter_error
from .realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
    HOState,
    VibronicHO,
)
