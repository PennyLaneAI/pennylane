# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Experimental Pauli Based Computation (PBC) functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.pbc

.. autosummary::
    :toctree: api

    ~compare_circuits


Custom operators
~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.pbc

.. autosummary::
    :toctree: api

    ~ControlledPauli
    ~controlled
    ~MeasurePauliWord
    ~measure
    ~ppr


"""

from .controlled import ControlledPauli, controlled
from .compare_circuits import compare_circuits
from .pauli_measure import MeasurePauliWord, measure
from .ops import ppr
