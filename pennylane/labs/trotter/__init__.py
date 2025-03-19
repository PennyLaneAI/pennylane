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
"""Module containing functions for computing Trotter error

.. warning::

    This module is experimental. Frequent changes will occur,
    with no guarantees of stability or backwards compatibility.

.. currentmodule:: pennylane.labs.trotter

Trotter Abstract Classes and Functions:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~Fragment
    ~commutator
    ~nested_commutator

Realspace Hamiltonian Classes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api

    ~RealspaceCoeffs
    ~RealspaceOperator
    ~RealspaceSum
    ~VibronicMatrix
"""

from .abstract_fragment import Fragment, commutator, nested_commutator
from .realspace import RealspaceCoeffs, RealspaceOperator, RealspaceSum, VibronicMatrix
