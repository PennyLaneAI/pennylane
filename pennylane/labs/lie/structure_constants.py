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
"""A function to compute the structure constants that make up the adjoint representation of a Lie algebra"""

import numpy as np

from pennylane.typing import TensorLike


def structure_constants_dense(g: TensorLike) -> TensorLike:
    r"""
    Compute the structure constants that make up the adjoint representation of a Lie algebra.

    This function computes the structure constants of a Lie algebra provided by their dense matrix representation,
    obtained from, e.g., :func:`~lie_closure_dense`.
    This is sometimes more efficient than using the sparse Pauli representations of :class:`~PauliWord` and
    `~PauliSentence` that are employed in :func:`~structure_constants`, e.g., when ther are few but dense sums of Paulis.

    .. seealso:: For details on the mathematical definitions, see :func:`~structure_constants` and the section "Lie algebra basics" in our `g-sim demo <https://pennylane.ai/qml/demos/tutorial_liesim/#lie-algebra-basics>`__.

    Args:
        g (np.array): The (dynamical) Lie algebra provided as dense matrices, as generated from :func:`pennylane.labs.lie.lie_closure_dense`.

    Returns:
        TensorLike: The adjoint representation of shape ``(d, d, d)``, corresponding to indices ``(gamma, alpha, beta)``.

    **Example**

    Let us generate the DLA of the transverse field Ising model using :func:`~lie_closure_dense`.

    >>> n = 4
    >>> gens = [X(i) @ X(i+1) + Y(i) @ Y(i+1) + Z(i) @ Z(i+1) for i in range(n-1)]
    >>> g = lie_closure_dense(gens)
    >>> g.shape
    (12, 16, 16)

    The DLA is represented by a collection of twelve :math:`2^4 \times 2^4` matrices.
    Hence, the dimension of the DLA is :math:`d = 12`. Hence, the structure constants have shape ``(12, 12, 12)``.

    >>> adj = structure_constants_dense(g)
    >>> adj.shape
    (12, 12, 12)

    """
    dimg, chi, _ = g.shape

    # compute all commutators
    m0m1 = np.einsum("aij,bjk->abik", g, g)
    m0m1 = np.reshape(m0m1, (-1, chi, chi))

    m1m0 = np.einsum("aij,bki->abkj", g, g)
    m1m0 = np.reshape(m1m0, (-1, chi, chi))
    all_coms = m0m1 - m1m0

    # project commutators on basis of g
    # vectorized computation to obtain coefficients in decomposition:
    # v = ∑ (tr(v @ e_j) / ||e_j||^2) * e_j
    adj = np.tensordot(g, all_coms, axes=[[1, 2], [-1, -2]])
    adj = -np.reshape(adj, (dimg, dimg, dimg)).imag

    return adj
