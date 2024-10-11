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


def structure_constants_dense(g: TensorLike, is_orthogonal: bool = False) -> TensorLike:
    r"""
    Compute the structure constants that make up the adjoint representation of a Lie algebra.

    This function computes the structure constants of a Lie algebra provided by their dense matrix representation,
    obtained from, e.g., :func:`~lie_closure_dense`.
    This is sometimes more efficient than using the sparse Pauli representations of :class:`~PauliWord` and
    `~PauliSentence` that are employed in :func:`~structure_constants`, e.g., when there are few generators
    that are sums of many Paulis.

    .. seealso:: For details on the mathematical definitions, see :func:`~structure_constants` and the section "Lie algebra basics" in our `g-sim demo <https://pennylane.ai/qml/demos/tutorial_liesim/#lie-algebra-basics>`__.

    Args:
        g (np.array): The (dynamical) Lie algebra provided as dense matrices, as generated from :func:`pennylane.labs.lie.lie_closure_dense`.
            ``g`` should have shape ``(d, 2**n, 2**n)`` where ``d`` is the dimension of the algebra and ``n`` is the number of qubits.
        is_orthogonal (bool): Whether or not the matrices in ``g`` are orthogonal with respect to the Hilbert-Schmidt inner product on
            (skew-)Hermitian matrices. If the inputs are orthogonal, it is recommended to set ``is_orthogonal`` to ``True`` to reduce
            computational cost. Defaults to ``False``.

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
    Hence, the dimension of the DLA is :math:`d = 12` and the structure constants have shape ``(12, 12, 12)``.

    >>> adj = structure_constants_dense(g)
    >>> adj.shape
    (12, 12, 12)

    """
    dimg, chi, _ = g.shape
    assert g.shape[2] == g.shape[1]

    # compute all commutators by computing all products first according to "aij,bjk->abik"
    prod = np.moveaxis(np.tensordot(g, g, axes=[[2], [1]]), 1, 2)
    all_coms = np.reshape(prod - prod.transpose((1, 0, 2, 3)), (-1, chi, chi))

    # project commutators on basis of g
    # vectorized computation to obtain coefficients in decomposition:
    # v = ∑ (tr(v @ e_j) / ||e_j||^2) * e_j
    if is_orthogonal:
        # Compute norms of orthogonal entries
        norms = np.einsum("aij,aji->a", g, g).real
        gram_inv = np.diag(1 / norms)
    else:
        # Compute the full inverse Gram matrix of the entries.
        gram_inv = np.linalg.pinv(np.tensordot(g, g, axes=[[1, 2], [2, 1]])).real
    adj = gram_inv @ np.tensordot(g, all_coms, axes=[[1, 2], [2, 1]]).imag
    adj = -np.reshape(adj, (dimg, dimg, dimg))
    return adj
