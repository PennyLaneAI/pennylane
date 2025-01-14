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
"""Functionality to compute the Cartan subalgebra"""
# pylint: disable=too-many-arguments, too-many-positional-arguments
import copy

import numpy as np
from scipy.linalg import null_space

import pennylane as qml
from pennylane.pauli.dla.center import _intersect_bases

from .dense_util import adjvec_to_op, change_basis_ad_rep, op_to_adjvec


def _gram_schmidt(X):
    """Orthogonalize basis of column vectors in X"""
    Q, _ = np.linalg.qr(X.T, mode="reduced")
    return Q.T


def _is_independent(v, A, tol=1e-14):
    """Check whether ``v`` is independent of the columns of A."""
    v /= np.linalg.norm(v)
    v = v - A @ qml.math.linalg.solve(A.conj().T @ A, A.conj().T) @ v
    return np.linalg.norm(v) > tol


def _orthogonal_complement_basis(a, m, tol):
    """find mtilde = m - a"""
    # Step 1: Find the span of a
    a = np.array(a)
    m = np.array(m)

    # Compute the orthonormal basis of a using QR decomposition

    Q = _gram_schmidt(a)

    # Step 2: Project each vector in m onto the orthogonal complement of span(a)
    projections = m - np.dot(np.dot(m, Q.T), Q)
    assert np.allclose(
        np.tensordot(a, projections, axes=[[1], [1]]), 0.0
    ), f"{np.tensordot(a, projections, axes=[[1], [1]])}"

    # Step 3: Find a basis for the non-zero projections
    # We'll use SVD to find the basis
    U, S, _ = np.linalg.svd(projections.T)

    # Choose columns of U corresponding to non-zero singular values
    rank = np.sum(S > tol)
    basis = U[:, :rank]
    assert np.allclose(
        np.tensordot(a, basis, axes=[[1], [0]]), 0.0
    ), f"{np.tensordot(a, basis, axes=[[1], [0]])}"

    return basis.T  # Transpose to get row vectors


def cartan_subalgebra(
    g, k, m, ad, start_idx=0, tol=1e-10, verbose=0, return_adjvec=False, is_orthogonal=True
):
    r"""
    Compute a Cartan subalgebra (CSA) :math:`\mathfrak{a} \subseteq \mathfrak{m}`.

    A non-unique CSA is a maximal Abelian subalgebra in the horizontal subspace :math:`\mathfrak{m}` of a Cartan decomposition.
    Note that this is sometimes called a horizontal CSA, and is different from the definition of a CSA on `Wikipedia <https://en.wikipedia.org/wiki/Cartan_subalgebra>`__.

    .. seealso:: :func:`~cartan_decomp`, :func:`~structure_constants`, `The KAK decomposition  theory(demo) <https://pennylane.ai/qml/demos/tutorial_kak_decomposition>`__, `The KAK decomposition in practice (demo) <https://pennylane.ai/qml/demos/tutorial_fixed_depth_hamiltonian_simulation_via_cartan_decomposition>`__.

    Args:
        g (List[Union[PauliSentence, np.ndarray]]): Lie algebra :math:`\mathfrak{g}`, which is assumed to be ordered as :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`
        k (List[Union[PauliSentence, np.ndarray]]): Vertical space :math:`\mathfrak{k}` from Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`
        m (List[Union[PauliSentence, np.ndarray]]): Horizontal space :math:`\mathfrak{m}` from Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`
        ad (Array): The :math:`|\mathfrak{g}| \times |\mathfrak{g}| \times |\mathfrak{g}|` dimensional adjoint representation of :math:`\mathfrak{g}` (see :func:`~structure_constants`)
        start_idx (bool): Indicates from which element in ``m`` the CSA computation starts.
        tol (float): Numerical tolerance for linear independence check
        verbose (bool): Whether or not to output progress during computation
        return_adjvec (bool): The output format. If ``False``, returns operators in their original
            input format (matrices or :class:`~PauliSentence`). If ``True``, returns the spaces as adjoint representation vectors.
        is_orthogonal (bool): Whether the basis elements are all orthogonal, both within
            and between ``g``, ``k`` and ``m``.

    Returns:
        Tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray): A tuple of adjoint vector representations
        ``(newg, k, mtilde, a, new_adj)``, corresponding to
        :math:`\mathfrak{g}`, :math:`\mathfrak{k}`, :math:`\tilde{\mathfrak{m}}`, :math:`\mathfrak{a}` and the new adjoint representation.
        The dimensions are ``(|g|, |g|)``, ``(|k|, |g|)``, ``(|mtilde|, |g|)``, ``(|a|, |g|)`` and ``(|g|, |g|, |g|)``, respectively.

    **Example**

    A quick example computing a Cartan subalgebra of :math:`\mathfrak{su}(4)` using the Cartan involution :func:`~even_odd_involution`.

    >>> import pennylane as qml
    >>> from pennylane.labs.dla import cartan_decomp, cartan_subalgebra, even_odd_involution
    >>> g = list(qml.pauli.pauli_group(2)) # u(4)
    >>> g = g[1:] # remove identity -> su(4)
    >>> g = [op.pauli_rep for op in g] # optional; turn to PauliSentence for convenience
    >>> k, m = cartan_decomp(g, even_odd_involution)
    >>> g = k + m # re-order g to separate k and m
    >>> adj = qml.structure_constants(g)
    >>> newg, k, mtilde, a, new_adj = cartan_subalgebra(g, k, m, adj)
    >>> newg == k + mtilde + a
    True
    >>> a
    [-1.0 * Z(0) @ Z(1), 1.0 * Y(0) @ Y(1), -1.0 * X(0) @ X(1)]

    We can confirm that these all commute with each other, as the CSA is Abelian (= all operators commute).

    >>> from pennylane.labs.dla import check_all_commuting
    >>> check_all_commuting(a)
    True

    We can opt-in to return what we call adjoint vectors of dimension :math:`|\mathfrak{g}|`, where each component corresponds to an entry in (the ordered) ``g``.
    The adjoint vectors for the Cartan subalgebra are in ``np_a``.

    >>> np_newg, np_k, np_mtilde, np_a, new_adj = cartan_subalgebra(g, k, m, adj, return_adjvec=True)

    We can reconstruct an operator by computing :math:`\hat{O}_v = \sum_i v_i g_i` for an adjoint vector :math:`v` and :math:`g_i \in \mathfrak{g}`.

    >>> v = np_a[0]
    >>> op = sum(v_i * g_i for v_i, g_i in zip(v, g))
    >>> op.simplify()
    >>> op
    -1.0 * Z(0) @ Z(1)

    For convenience, we provide a helper function :func:`~adjvec_to_op` for the collections of adjoint vectors in the returns.

    >>> from pennylane.labs.dla import adjvec_to_op
    >>> a = adjvec_to_op(np_a, g)
    >>> a
    [-1.0 * Z(0) @ Z(1), 1.0 * Y(0) @ Y(1), -1.0 * X(0) @ X(1)]

    .. details::
        :title: Usage Details

        Let us walk through an example of computing the Cartan subalgebra. The basis for computing
        the Cartan subalgebra is having the Lie algebra :math:`\mathfrak{g}`, a Cartan decomposition
        :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` and its adjoint representation.

        We start by computing these ingredients using :func:`~cartan_decomp` and :func:`~structure_constants`.
        As an example, we take the Lie algebra of the Heisenberg model with generators :math:`\{X_i X_{i+1}, Y_i Y_{i+1}, Z_i Z_{i+1}\}`.

        >>> from pennylane.labs.dla import lie_closure_dense, cartan_decomp
        >>> from pennylane import X, Y, Z
        >>> n = 3
        >>> gens = [X(i) @ X(i+1) for i in range(n-1)]
        >>> gens += [Y(i) @ Y(i+1) for i in range(n-1)]
        >>> gens += [Z(i) @ Z(i+1) for i in range(n-1)]
        >>> g = lie_closure_dense(gens)

        Taking the Heisenberg Lie algebra, we can perform the Cartan decomposition. We take the :func:`~even_odd_involution` as a valid Cartan involution.
        The resulting vertical and horizontal subspaces :math:`\mathfrak{k}` and :math:`\mathfrak{m}` need to fulfill the commutation relations
        :math:`[\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k}`, :math:`[\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m}` and :math:`[\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}`,
        which we can check using the helper function :func:`~check_cartan_decomp`.

        >>> from pennylane.labs.dla import even_odd_involution, check_cartan_decomp
        >>> k, m = cartan_decomp(g, even_odd_involution)
        >>> check_cartan_decomp(k, m) # check commutation relations of Cartan decomposition
        True

        Our life is easier when we use a canonical ordering of the operators. This is why we re-define ``g`` with the new ordering in terms of operators in ``k`` first, and then
        all remaining operators from ``m``.

        >>> import numpy as np
        >>> from pennylane.labs.dla import structure_constants_dense
        >>> g = np.vstack([k, m]) # re-order g to separate k and m operators
        >>> adj = structure_constants_dense(g) # compute adjoint representation of g

        Finally, we can compute a Cartan subalgebra :math:`\mathfrak{a}`, a maximal Abelian subalgebra of :math:`\mathfrak{m}`.

        >>> newg, k, mtilde, a, new_adj = cartan_subalgebra(g, k, m, adj, start_idx=3)

        The new DLA ``newg`` is just the concatenation of ``k``, ``mtilde``, ``a``. Each component is returned in the original input format.
        Here we obtain collections of :math:`8\times 8` matrices (``numpy`` arrays), as this is what we started from.

        >>> newg.shape, k.shape, mtilde.shape, a.shape, new_adj.shape
        ((15, 8, 8), (6, 8, 8), (6, 8, 8), (3, 8, 8), (15, 15, 15))

        We can also let the function return what we call adjoint representation vectors.

        >>> kwargs = {"start_idx": 3, "return_adjvec": True}
        >>> np_newg, np_k, np_mtilde, np_a, new_adj = cartan_subalgebra(g, k, m, adj, **kwargs)
        >>> np_newg.shape, np_k.shape, np_mtilde.shape, np_a.shape, new_adj.shape
        ((15, 15), (6, 15), (6, 15), (3, 15), (15, 15, 15))

        These are dense vector representations of dimension :math:`|\mathfrak{g}|`, in which each entry corresponds to the respective operator in :math:`\mathfrak{g}`.
        Given an adjoint representation vector :math:`v`, we can reconstruct the respective operator by simply computing :math:`\hat{O}_v = \sum_i v_i g_i`, where
        :math:`g_i \in \mathfrak{g}` (hence the need for a canonical ordering).

        We provide a convenience function :func:`~adjvec_to_op` that works with both ``g`` represented as dense matrices or PL operators.
        Because we used dense matrices in this example, we transform the operators back to PennyLane operators using :func:`~pauli_decompose`.

        >>> from pennylane.labs.dla import adjvec_to_op
        >>> a = adjvec_to_op(np_a, g)
        >>> h_op = [qml.pauli_decompose(op).pauli_rep for op in a]
        >>> h_op
        [-1.0 * Y(1) @ Y(2), -1.0 * Z(1) @ Z(2), 1.0 * X(1) @ X(2)]

        In that case we chose a Cartan subalgebra from which we can readily see that it is commuting, but we also provide a small helper function to check that.

        >>> from pennylane.labs.dla import check_all_commuting
        >>> assert check_all_commuting(h_op)

        Last but not least, the adjoint representation ``new_adj`` is updated to represent the new basis and its ordering of ``g``.
    """

    g_copy = copy.deepcopy(g)
    np_m = op_to_adjvec(m, g, is_orthogonal=is_orthogonal)
    np_a = op_to_adjvec([m[start_idx]], g, is_orthogonal=is_orthogonal)

    iteration = 1
    while True:
        if verbose:
            print(f"iteration {iteration}: Found {len(np_a)} independent Abelian operators.")
        kernel_intersection = np_m
        for h_i in np_a:

            # obtain adjoint rep of candidate h_i
            adjoint_of_h_i = np.tensordot(ad, h_i, axes=[[1], [0]])
            # compute kernel of adjoint
            new_kernel = null_space(adjoint_of_h_i, rcond=tol)

            # intersect kernel to stay in m
            kernel_intersection = _intersect_bases(kernel_intersection.T, new_kernel, rcond=tol).T

        kernel_intersection = _gram_schmidt(kernel_intersection)  # orthogonalize
        for vec in kernel_intersection:
            if _is_independent(vec, np.array(np_a).T, tol):
                np_a = np.vstack([np_a, vec])
                break
        else:
            # No new vector was added from all the kernels
            break

        iteration += 1

    np_a = _gram_schmidt(np_a)  # orthogonalize Abelian subalgebra
    np_k = op_to_adjvec(
        k, g, is_orthogonal=is_orthogonal
    )  # adjoint vectors of k space for re-ordering
    np_oldg = np.vstack([np_k, np_m])
    np_k = _gram_schmidt(np_k)

    np_mtilde = _orthogonal_complement_basis(np_a, np_m, tol=tol)  # the "rest" of m without a
    np_newg = np.vstack([np_k, np_mtilde, np_a])

    # Instead of recomputing the adjoint representation, take the basis transformation
    # oldg -> newg and transform the adjoint representation accordingly
    basis_change = np.tensordot(np_newg, np.linalg.pinv(np_oldg), axes=[[1], [0]])
    new_adj = change_basis_ad_rep(ad, basis_change)

    if return_adjvec:
        return np_newg, np_k, np_mtilde, np_a, new_adj

    newg, k, mtilde, a = [
        adjvec_to_op(adjvec, g_copy, is_orthogonal=is_orthogonal)
        for adjvec in [np_newg, np_k, np_mtilde, np_a]
    ]

    return newg, k, mtilde, a, new_adj
