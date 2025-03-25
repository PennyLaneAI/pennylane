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

Trotter Base Classes:
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~AbstractState
    ~Fragment

Realspace Hamiltonian Classes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~RealspaceCoeffs
    ~RealspaceMatrix
    ~RealspaceOperator
    ~RealspaceSum

State Classes:
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~HOState
    ~VibronicHO
"""

from .abstract import AbstractState, Fragment
from .realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
    HOState,
    VibronicHO,
)
