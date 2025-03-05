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
Experimental dynamical Lie algebra (DLA) functionality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.dla

.. autosummary::
    :toctree: api

    ~recursive_cartan_decomp
    ~cartan_subalgebra
    ~variational_kak_adj


Utility functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.dla

.. autosummary::
    :toctree: api

    ~adjvec_to_op
    ~op_to_adjvec
    ~orthonormalize
    ~pauli_coefficients
    ~batched_pauli_decompose
    ~check_orthonormal
    ~check_cartan_decomp
    ~change_basis_ad_rep
    ~validate_kak
    ~run_opt


"""

from .recursive_cartan_decomp import (
    recursive_cartan_decomp,
)
from .dense_util import (
    adjvec_to_op,
    change_basis_ad_rep,
    check_all_commuting,
    check_orthonormal,
    pauli_coefficients,
    batched_pauli_decompose,
    op_to_adjvec,
    orthonormalize,
)

from .cartan_subalgebra import cartan_subalgebra
from .variational_kak import validate_kak, variational_kak_adj, run_opt
