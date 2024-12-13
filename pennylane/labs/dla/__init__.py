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
    ~cartan_decomp
    ~recursive_cartan_decomp
    ~cartan_subalgebra


Utility functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: pennylane.labs.dla

.. autosummary::
    :toctree: api

    ~adjvec_to_op
    ~op_to_adjvec
    ~trace_inner_product
    ~orthonormalize
    ~pauli_coefficients
    ~batched_pauli_decompose
    ~check_orthonormal
    ~check_commutation
    ~check_all_commuting
    ~check_cartan_decomp
    ~change_basis_ad_rep

Involutions
~~~~~~~~~~~

A map :math:`\theta: \mathfrak{g} \rightarrow \mathfrak{g}` from the Lie algebra :math:`\mathfrak{g}` to itself is called an involution
when it fulfills :math:`\theta(\theta(g)) = g \ \forall g \in \mathfrak{g}` and is compatible with commutators,
:math:`[\theta(g), \theta(g')]=\theta([g, g']).` Involutions are used to construct a :func:`~cartan_decomp`. There are seven canonical
Cartan involutions of real simple Lie algebras (``AI, AII, AIII, BDI, CI, CII, DIII``),
see `Wikipedia <https://en.wikipedia.org/wiki/Symmetric_space#Classification_result>`__.
In addition, there is a canonical Cartan involution for real semisimple algebras that consist of
two isomorphic simple components (``ClassB``), see `here <https://en.wikipedia.org/wiki/Symmetric_space#Classification_scheme>`__.

.. currentmodule:: pennylane.labs.dla

.. autosummary::
    :toctree: api

    ~even_odd_involution
    ~concurrence_involution
    ~khaneja_glaser_involution
    ~AI
    ~AII
    ~AIII
    ~BDI
    ~CI
    ~CII
    ~DIII
    ~ClassB


"""

from .lie_closure_dense import lie_closure_dense
from .structure_constants_dense import structure_constants_dense
from .cartan import (
    cartan_decomp,
    recursive_cartan_decomp,
)
from .dense_util import (
    adjvec_to_op,
    change_basis_ad_rep,
    check_all_commuting,
    check_cartan_decomp,
    check_commutation,
    check_orthonormal,
    pauli_coefficients,
    batched_pauli_decompose,
    trace_inner_product,
    op_to_adjvec,
    orthonormalize,
)

from .involutions import (
    khaneja_glaser_involution,
    even_odd_involution,
    concurrence_involution,
    AI,
    AII,
    AIII,
    BDI,
    CI,
    CII,
    DIII,
    ClassB,
)

from .cartan_subalgebra import cartan_subalgebra
