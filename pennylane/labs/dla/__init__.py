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
Experimental Lie theory features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.dla

.. autosummary::
    :toctree: api

    ~lie_closure_dense
    ~structure_constants_dense
    ~cartan_decomposition


Utility functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.dla

.. autosummary::
    :toctree: api

    ~adjvec_to_op
    ~change_basis_ad_rep
    ~op_to_adjvec
    ~pauli_coefficients
    ~pauli_decompose
    ~trace_inner_product

Involutions
~~~~~~~~~~~

.. currentmodule:: pennylane.labs.dla

.. autosummary::
    :toctree: api

    ~even_odd_involution
    ~concurrence_involution


"""

from .lie_closure_dense import lie_closure_dense
from .structure_constants_dense import structure_constants_dense
from .cartan import cartan_decomposition, even_odd_involution, concurrence_involution
from .dense_util import (
    change_basis_ad_rep,
    pauli_coefficients,
    pauli_decompose,
    orthonormalize,
    check_orthonormal,
    trace_inner_product,
    adjvec_to_op,
    op_to_adjvec,
)
