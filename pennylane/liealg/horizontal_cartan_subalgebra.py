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
"""Functionality to compute the Cartan subalgebra"""
# pylint: disable=too-many-arguments, too-many-positional-arguments, possibly-used-before-assignment
import copy
from collections.abc import Iterable
from itertools import combinations, combinations_with_replacement

from scipy.linalg import null_space, sqrtm

from pennylane import math
from pennylane.liealg.center import _intersect_bases
from pennylane.operation import Operator
from pennylane.ops import Identity
from pennylane.ops.functions import commutator, equal, matrix, simplify
from pennylane.pauli import PauliSentence, trace_inner_product
from pennylane.typing import TensorLike

from .structure_constants import structure_constants


def _gram_schmidt(X):
    """Orthogonalize basis of column vectors in X"""
    Q, _ = math.linalg.qr(X.T, mode="reduced")
    return Q.T


def _is_independent(v, A, tol=1e-14):
    """Check whether ``v`` is independent of the columns of A."""
    v /= math.linalg.norm(v)
    v = v - A @ math.linalg.solve(A.conj().T @ A, A.conj().T) @ v
    return math.linalg.norm(v) > tol


def _orthogonal_complement_basis(a, m, tol):
    """find mtilde = m - a"""
    # Step 1: Find the span of a
    a = math.array(a)
    m = math.array(m)

    # Compute the orthonormal basis of a using QR decomposition

    Q = _gram_schmidt(a)

    # Step 2: Project each vector in m onto the orthogonal complement of span(a)
    projections = m - math.dot(math.dot(m, Q.T), Q)
    assert math.allclose(
        math.tensordot(a, projections, axes=[[1], [1]]), 0.0
    ), f"{math.tensordot(a, projections, axes=[[1], [1]])}"

    # Step 3: Find a basis for the non-zero projections
    # We'll use SVD to find the basis
    U, S, _ = math.linalg.svd(projections.T)

    # Choose columns of U corresponding to non-zero singular values
    rank = math.sum(S > tol)
    basis = U[:, :rank]
    assert math.allclose(
        math.tensordot(a, basis, axes=[[1], [0]]), 0.0
    ), f"{math.tensordot(a, basis, axes=[[1], [0]])}"

    return basis.T  # Transpose to get row vectors


def horizontal_cartan_subalgebra(
    k, m, adj=None, start_idx=0, tol=1e-10, verbose=0, return_adjvec=False, is_orthogonal=True
):
    r"""
    Compute a Cartan subalgebra (CSA) :math:`\mathfrak{a} \subseteq \mathfrak{m}`.

    A non-unique CSA is a maximal Abelian subalgebra in the horizontal subspace :math:`\mathfrak{m}` of a Cartan decomposition.
    Note that this is sometimes called a horizontal CSA, and is different from the `other definitions of a CSA <https://en.wikipedia.org/wiki/Cartan_subalgebra>`__.

    The final decomposition yields

    .. math:: \mathfrak{g} = \mathfrak{k} \oplus (\tilde{\mathfrak{m}} \oplus \mathfrak{a}),

    where :math:`\mathfrak{a})` is the CSA and :math:`\tilde{\mathfrak{m}}` is the remainder of the horizontal subspace :math:`\mathfrak{m}`.

    .. seealso:: :func:`~cartan_decomp`, :func:`~structure_constants`, `The KAK decomposition in theory (demo) <demos/tutorial_kak_decomposition>`__, `The KAK decomposition in practice (demo) <demos/tutorial_fixed_depth_hamiltonian_simulation_via_cartan_decomposition>`__.

    Args:
        k (List[Union[PauliSentence, TensorLike]]): Vertical space :math:`\mathfrak{k}` from Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`.
        m (List[Union[PauliSentence, TensorLike]]): Horizontal space :math:`\mathfrak{m}` from Cartan decomposition :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}`.
        adj (Array): The :math:`|\mathfrak{g}| \times |\mathfrak{g}| \times |\mathfrak{g}|` dimensional adjoint representation of :math:`\mathfrak{g}`.
            When ``None`` is provided, :func:`~structure_constants` is used internally by default to compute the adjoint representation.
        start_idx (bool): Indicates from which element in ``m`` the CSA computation starts.
        tol (float): Numerical tolerance for linear independence check.
        verbose (bool): Whether or not to output progress during computation.
        return_adjvec (bool): Determine the output format. If ``False``, returns operators in their original
            input format (matrices or :class:`~.PauliSentence`). If ``True``, returns the spaces as adjoint representation vectors (see :func:`~op_to_adjvec` and :func:`~adjvec_to_op`).
        is_orthogonal (bool): Whether the basis elements are all orthogonal, both within
            and between ``g``, ``k`` and ``m``.

    Returns:
        Tuple(TensorLike, TensorLike, TensorLike, TensorLike, TensorLike): A tuple of adjoint vector representations
        ``(newg, k, mtilde, a, new_adj)``, corresponding to
        :math:`\mathfrak{g}`, :math:`\mathfrak{k}`, :math:`\tilde{\mathfrak{m}}`, :math:`\mathfrak{a}` and the new adjoint representation.
        The dimensions are ``(|g|, |g|)``, ``(|k|, |g|)``, ``(|mtilde|, |g|)``, ``(|a|, |g|)`` and ``(|g|, |g|, |g|)``, respectively.

    **Example**

    A quick example computing a Cartan subalgebra of :math:`\mathfrak{su}(4)` using the Cartan involution :func:`~even_odd_involution`.

    >>> g = list(qml.pauli.pauli_group(2)) # u(4)
    >>> g = g[1:] # remove identity -> su(4)
    >>> g = [op.pauli_rep for op in g] # optional; turn to PauliSentence for convenience
    >>> k, m = qml.liealg.cartan_decomp(g, qml.liealg.even_odd_involution)
    >>> g = k + m # re-order g to separate k and m
    >>> newg, k, mtilde, a, new_adj = qml.liealg.horizontal_cartan_subalgebra(k, m)
    >>> newg == k + mtilde + a
    True
    >>> a # doctest: +SKIP
    [-1.0 * Z(0) @ Z(1), -1.0 * Y(0) @ Y(1), 1.0 * X(0) @ X(1)]

    We can confirm that these all commute with each other, as the CSA is Abelian (i.e., all operators commute).

    >>> qml.liealg.check_abelian(a)
    True

    We can opt-in to return what we call adjoint vectors of dimension :math:`|\mathfrak{g}|`, where each component corresponds to an entry in (the ordered) ``g``.
    The adjoint vectors for the Cartan subalgebra are in ``np_a``.

    .. code-block:: python

        from pennylane.liealg import horizontal_cartan_subalgebra
        np_newg, np_k, np_mtilde, np_a, new_adj = horizontal_cartan_subalgebra(k, m, return_adjvec=True)

    We can reconstruct an operator by computing :math:`\hat{O}_v = \sum_i v_i g_i` for an adjoint vector :math:`v` and :math:`g_i \in \mathfrak{g}`.

    >>> v = np_a[0]
    >>> op = sum(v_i * g_i for v_i, g_i in zip(v, g))
    >>> op.simplify()
    >>> op
    -1.0 * Z(0) @ Z(1)

    For convenience, we provide a helper function :func:`~adjvec_to_op` for conversion of the returned collections of adjoint vectors.

    >>> a = qml.liealg.adjvec_to_op(np_a, g)
    >>> a # doctest: +SKIP
    [-1.0 * Z(0) @ Z(1), -1.0 * Y(0) @ Y(1), 1.0 * X(0) @ X(1)]

    .. details::
        :title: Usage Details

        Let us walk through an example of computing the Cartan subalgebra. The basis for computing
        the Cartan subalgebra is having the Lie algebra :math:`\mathfrak{g}`, a Cartan decomposition
        :math:`\mathfrak{g} = \mathfrak{k} \oplus \mathfrak{m}` and its adjoint representation.

        We start by computing these ingredients using :func:`~cartan_decomp` and :func:`~structure_constants`.
        As an example, we take the Lie algebra of the Heisenberg model with generators :math:`\{X_i X_{i+1}, Y_i Y_{i+1}, Z_i Z_{i+1}\}`.

        >>> from pennylane.liealg import cartan_decomp
        >>> from pennylane import X, Y, Z
        >>> n = 3
        >>> gens = [X(i) @ X(i+1) for i in range(n-1)]
        >>> gens += [Y(i) @ Y(i+1) for i in range(n-1)]
        >>> gens += [Z(i) @ Z(i+1) for i in range(n-1)]
        >>> g = qml.lie_closure(gens, matrix=True)

        Taking the Heisenberg Lie algebra, we can perform the Cartan decomposition. We take the :func:`~even_odd_involution` as a valid Cartan involution.
        The resulting vertical and horizontal subspaces :math:`\mathfrak{k}` and :math:`\mathfrak{m}` need to fulfill the commutation relations
        :math:`[\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k}`, :math:`[\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m}` and :math:`[\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}`,
        which we can check using the helper function :func:`~check_cartan_decomp`.

        >>> from pennylane.liealg import even_odd_involution, check_cartan_decomp
        >>> k, m = cartan_decomp(g, even_odd_involution)
        >>> check_cartan_decomp(k, m) # check commutation relations of Cartan decomposition
        True

        Our life is easier when we use a canonical ordering of the operators. This is why we re-define ``g`` with the new ordering in terms of operators in ``k`` first, and then
        all remaining operators from ``m``.

        >>> g = np.vstack([k, m]) # re-order g to separate k and m operators
        >>> adj = qml.structure_constants(g, matrix=True) # compute adjoint representation of g

        Finally, we can compute a Cartan subalgebra :math:`\mathfrak{a}`, a maximal Abelian subalgebra of :math:`\mathfrak{m}`.

        >>> newg, k, mtilde, a, new_adj = horizontal_cartan_subalgebra(k, m, adj, start_idx=3)

        The new DLA ``newg`` is just the concatenation of ``k``, ``mtilde``, ``a``. Each component is returned in the original input format.
        Here we obtain collections of :math:`8\times 8` matrices (``numpy`` arrays), as this is what we started from.

        >>> newg.shape, k.shape, mtilde.shape, a.shape, new_adj.shape
        ((15, 8, 8), (6, 8, 8), (6, 8, 8), (3, 8, 8), (15, 15, 15))

        We can also let the function return what we call adjoint representation vectors.

        >>> kwargs = {"start_idx": 3, "return_adjvec": True}
        >>> np_newg, np_k, np_mtilde, np_a, new_adj = horizontal_cartan_subalgebra(k, m, adj, **kwargs)
        >>> np_newg.shape, np_k.shape, np_mtilde.shape, np_a.shape, new_adj.shape
        ((15, 15), (6, 15), (6, 15), (3, 15), (15, 15, 15))

        These are dense vector representations of dimension :math:`|\mathfrak{g}|`, in which each entry corresponds to the respective operator in :math:`\mathfrak{g}`.
        Given an adjoint representation vector :math:`v`, we can reconstruct the respective operator by simply computing :math:`\hat{O}_v = \sum_i v_i g_i`, where
        :math:`g_i \in \mathfrak{g}` (hence the need for a canonical ordering).

        We provide a convenience function :func:`~adjvec_to_op` that works with both ``g`` represented as dense matrices or PL operators.
        Because we used dense matrices in this example, we transform the operators back to PennyLane operators using :func:`~pauli_decompose`.

        >>> from pennylane.liealg import adjvec_to_op
        >>> a = adjvec_to_op(np_a, g)
        >>> h_op = [qml.pauli_decompose(op).pauli_rep for op in a]
        >>> h_op # doctest: +SKIP
        [-1.0 * Y(1) @ Y(2), -1.0 * Z(1) @ Z(2), 1.0 * X(1) @ X(2)]

        In that case we chose a Cartan subalgebra from which we can readily see that it is commuting, but we also provide a small helper function to check that.

        >>> from pennylane.liealg import check_abelian
        >>> check_abelian(h_op)
        True

        Last but not least, the adjoint representation ``new_adj`` is updated to represent the new basis and its ordering of ``g``.
    """
    if isinstance(k, (list, tuple)) and isinstance(m, (list, tuple)):
        g = k + m
    else:
        g = math.vstack([k, m])

    if adj is None:
        adj = structure_constants(g, matrix=isinstance(g[0], TensorLike))

    g_copy = copy.deepcopy(g)
    np_m = op_to_adjvec(m, g, is_orthogonal=is_orthogonal)
    np_a = op_to_adjvec([m[start_idx]], g, is_orthogonal=is_orthogonal)

    iteration = 1
    while True:
        if verbose:
            print(f"iteration {iteration}: Found {len(np_a)} independent Abelian operators.")
        # todo: avoid re-computing this overlap in every while-loop iteration.
        kernel_intersection = np_m
        for h_i in np_a:

            # obtain adjoint rep of candidate h_i
            adjoint_of_h_i = math.tensordot(adj, h_i, axes=[[1], [0]])
            # compute kernel of adjoint
            new_kernel = null_space(adjoint_of_h_i, rcond=tol)

            # intersect kernel to stay in m
            kernel_intersection = _intersect_bases(kernel_intersection.T, new_kernel, rcond=tol).T

        kernel_intersection = _gram_schmidt(kernel_intersection)  # orthogonalize
        for vec in kernel_intersection:
            if _is_independent(vec, math.array(np_a).T, tol):
                np_a = math.vstack([np_a, vec])
                break
        else:
            # No new vector was added from all the kernels
            break

        iteration += 1

    np_a = _gram_schmidt(np_a)  # orthogonalize Abelian subalgebra
    np_k = op_to_adjvec(
        k, g, is_orthogonal=is_orthogonal
    )  # adjoint vectors of k space for re-ordering
    np_oldg = math.vstack([np_k, np_m])
    np_k = _gram_schmidt(np_k)

    np_mtilde = _orthogonal_complement_basis(np_a, np_m, tol=tol)  # the "rest" of m without a
    np_newg = math.vstack([np_k, np_mtilde, np_a])

    # Instead of recomputing the adjoint representation, take the basis transformation
    # oldg -> newg and transform the adjoint representation accordingly
    basis_change = math.tensordot(np_newg, math.linalg.pinv(np_oldg), axes=[[1], [0]])
    new_adj = change_basis_ad_rep(adj, basis_change)

    if return_adjvec:
        return np_newg, np_k, np_mtilde, np_a, new_adj

    newg, k, mtilde, a = (
        adjvec_to_op(adjvec, g_copy, is_orthogonal=is_orthogonal)
        for adjvec in [np_newg, np_k, np_mtilde, np_a]
    )

    return newg, k, mtilde, a, new_adj


def adjvec_to_op(adj_vecs, basis, is_orthogonal=True):
    r"""Transform adjoint vector representations back into operator format.

    This function simply reconstructs :math:`\hat{O} = \sum_j c_j \hat{b}_j` given the adjoint vector
    representation :math:`c_j` and basis :math:`\hat{b}_j`.

    .. seealso:: :func:`~op_to_adjvec`

    Args:
        adj_vecs (TensorLike): collection of vectors with shape ``(batch, len(basis))``
        basis (List[Union[PauliSentence, Operator, TensorLike]]): collection of basis operators
        is_orthogonal (bool): Whether the ``basis`` consists of orthogonal elements.

    Returns:
        list: collection of operators corresponding to the input vectors read in the input basis.
        The operators are in the format specified by the elements in ``basis``.

    **Example**

    >>> from pennylane.liealg import adjvec_to_op
    >>> c = np.array([[0.5, 0.3, 0.7]])
    >>> basis = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> adjvec_to_op(c, basis)
    [0.5 * X(0) + 0.3 * Y(0) + 0.7 * Z(0)]

    """

    assert math.shape(adj_vecs)[1] == len(basis)

    if all(isinstance(op, PauliSentence) for op in basis):
        if not is_orthogonal:
            gram = _gram_ps(basis)
            adj_vecs = math.tensordot(adj_vecs, math.linalg.pinv(sqrtm(gram)), axes=[[1], [0]])
        res = []
        for vec in adj_vecs:
            op_j = sum(c * op for c, op in zip(vec, basis))
            op_j.simplify()
            res.append(op_j)
        return res

    if all(isinstance(op, Operator) for op in basis):
        if not is_orthogonal:
            basis_ps = [op.pauli_rep for op in basis]
            gram = _gram_ps(basis_ps)
            adj_vecs = math.tensordot(adj_vecs, math.linalg.pinv(sqrtm(gram)), axes=[[1], [0]])
        res = []
        for vec in adj_vecs:
            op_j = sum(c * op for c, op in zip(vec, basis))
            op_j = simplify(op_j)
            res.append(op_j)
        return res

    if all(isinstance(op, TensorLike) for op in basis):
        if not is_orthogonal:
            gram = trace_inner_product(basis, basis).real
            adj_vecs = math.tensordot(adj_vecs, math.linalg.pinv(sqrtm(gram)), axes=[[1], [0]])
        return math.tensordot(adj_vecs, basis, axes=1)

    raise NotImplementedError(
        "At least one operator in the specified basis is of unsupported type, "
        "or not all operators are of the same type."
    )


def _gram_ps(basis: Iterable[PauliSentence]):
    gram = math.zeros((len(basis), len(basis)))
    for (i, b_i), (j, b_j) in combinations_with_replacement(enumerate(basis), r=2):
        gram[i, j] = gram[j, i] = (b_i @ b_j).trace()
    return gram


def _op_to_adjvec_ps(ops: PauliSentence, basis: PauliSentence, is_orthogonal: bool = True):
    """Pauli sentence branch of ``op_to_adjvec``."""

    res = []
    if is_orthogonal:
        norms_squared = [(basis_i @ basis_i).trace() for basis_i in basis]
    else:
        # Fake the norm correction if we anyways will apply the inverse Gram matrix later
        norms_squared = math.ones(len(basis))
        gram = _gram_ps(basis)
        inv_gram = math.linalg.pinv(sqrtm(gram))

    for op in ops:
        rep = math.zeros((len(basis),))
        for i, basis_i in enumerate(basis):
            # v = ∑ (v · e_j / ||e_j||^2) * e_j
            rep[i] = (basis_i @ op).trace() / norms_squared[i]

        res.append(rep)
    res = math.array(res)
    if not is_orthogonal:
        res = math.einsum("ij,kj->ki", inv_gram, res)

    return res


def op_to_adjvec(
    ops: Iterable[PauliSentence | Operator | TensorLike],
    basis: PauliSentence | Operator | TensorLike,
    is_orthogonal: bool = True,
):
    r"""Decompose a batch of operators into a given operator basis.

    The adjoint vector representation is provided by the coefficients :math:`c_j` in a given operator
    basis of the operator :math:`\hat{b}_j` such that the input operator can be written as
    :math:`\hat{O} = \sum_j c_j \hat{b}_j`.

    .. seealso:: :func:`~adjvec_to_op`

    Args:
        ops (Iterable[Union[PauliSentence, Operator, TensorLike]]): List of operators to decompose.
        basis (Iterable[Union[PauliSentence, Operator, TensorLike]]): Operator basis.
        is_orthogonal (bool): Whether the basis is orthogonal with respect to the trace inner
            product. Defaults to ``True``, which allows to skip some computations.

    Returns:
        TensorLike: The batch of coefficient vectors of the operators' ``ops`` expressed in
        ``basis``. The shape is ``(len(ops), len(basis)``.

    The format of the resulting operators is determined by the ``type`` in ``basis``.
    If ``is_orthogonal=True`` (the default), only normalization is taken into account
    in the projection. For ``is_orthogonal=False``, orthogonalization also is considered.

    **Example**

    The basis can be numerical or operators.

    >>> from pennylane.liealg import op_to_adjvec
    >>> op = qml.X(0) + 0.5 * qml.Y(0)
    >>> basis = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> op_to_adjvec([op], basis)
    array([[1. , 0.5, 0. ]])
    >>> op_to_adjvec([op], [op.matrix() for op in basis])
    array([[1. , 0.5, 0. ]])

    Note how the function always expects an ``Iterable`` of operators as input.

    The ``ops`` can also be numerical, but then ``basis`` has to be numerical as well.

    >>> op = op.matrix()
    >>> op_to_adjvec([op], [op.matrix() for op in basis])
    array([[1. , 0.5, 0. ]])
    """

    if all(isinstance(op, Operator) for op in basis):
        ops = [op.pauli_rep for op in ops]
        basis = [op.pauli_rep for op in basis]

    # PauliSentence branch
    if all(isinstance(op, PauliSentence) for op in basis):
        return _op_to_adjvec_ps(ops, basis, is_orthogonal)

    # dense branch
    if all(
        isinstance(op, TensorLike) and not isinstance(op, (int, float, complex)) for op in basis
    ):
        if not all(isinstance(op, TensorLike) for op in ops):
            _n = int(math.round(math.log2(basis[0].shape[-1])))
            ops = math.array([matrix(op, wire_order=range(_n)) for op in ops])

        basis = math.array(basis)
        res = trace_inner_product(math.array(ops), basis).real
        if is_orthogonal:
            norm = math.real(math.einsum("bij,bji->b", basis, basis)) / math.shape(basis[0])[0]
            return res / norm
        gram = math.real(trace_inner_product(basis, basis))
        sqrtm_gram = sqrtm(gram)
        # Imaginary component is an artefact
        assert math.allclose(math.imag(sqrtm_gram), 0.0, atol=1e-16)
        return math.einsum("ij,kj->ki", math.linalg.pinv(sqrtm_gram.real), res)

    raise NotImplementedError(
        "At least one operator in the specified basis is of unsupported type, "
        "or not all operators are of the same type."
    )


def change_basis_ad_rep(adj: TensorLike, basis_change: TensorLike):
    r"""Apply a ``basis_change`` between bases of operators to the adjoint representation ``adj``.

    Assume the adjoint repesentation is given in terms of a basis :math:`\{b_j\}`,
    :math:`\text{ad}^\mu_{\alpha \beta} \propto \text{tr}\left(b_\mu \cdot [b_\alpha, b_\beta] \right)`.
    We can represent the adjoint representation in terms of a new basis :math:`c_i = \sum_j T_{ij} b_j`
    with the basis transformation matrix :math:`T` using ``change_basis_ad_rep``.

    Args:
        adj (TensorLike): Adjoint representation in old basis.
        basis_change (TensorLike): Basis change matrix from old to new basis.

    Returns:
        TensorLike: Adjoint representation in new basis.

    .. seealso: :func:`~liealg.structure_constants`

    **Example**

    We choose a basis of a Lie algebra, compute its adjoint representation.

    >>> from pennylane.liealg import change_basis_ad_rep
    >>> basis = [qml.X(0), qml.Y(0), qml.Z(0)]
    >>> adj = qml.structure_constants(basis)

    Now we change the basis and re-compute the adjoint representation in that new basis.

    >>> basis_change = np.array([[1., 1., 0.], [0., 1., 1.], [0., 1., 1.]])
    >>> new_ops = [qml.sum(*[basis_change[i,j] * basis[j] for j in range(3)]) for i in range(3)]
    >>> new_adj = qml.structure_constants(new_ops)

    We confirm that instead of re-computing the adjoint representation (typically expensive), we can
    transform the old adjoint representation with the change of basis matrix.

    >>> new_adj_re = change_basis_ad_rep(adj, basis_change)
    >>> np.allclose(new_adj, new_adj_re)
    True
    """
    # Perform the einsum contraction "mnp, hm, in, jp -> hij" via three einsum steps
    new_adj = math.einsum("mnp,im->inp", adj, math.linalg.pinv(basis_change.T))
    new_adj = math.einsum("mnp,in->mip", new_adj, basis_change)
    return math.einsum("mnp,ip->mni", new_adj, basis_change)


def check_abelian(ops: list[PauliSentence | TensorLike | Operator]):
    r"""Helper function to check if all operators in ``ops`` commute, i.e., form an Abelian set of operators.

    .. warning:: This function is expensive to compute

    Args:
        ops (List[Union[PauliSentence, TensorLike, Operator]]): List of operators to check for mutual commutation

    Returns:
        bool: Whether or not all operators commute with each other

    **Example**

    >>> from pennylane.liealg import check_abelian
    >>> from pennylane import X
    >>> ops = [X(i) for i in range(10)]
    >>> check_abelian(ops)
    True

    Operators on different wires (trivially) commute with each other.
    """
    return_True = None
    if all(isinstance(op, PauliSentence) for op in ops):
        for oi, oj in combinations(ops, 2):
            com = oj.commutator(oi)
            com.simplify()
            if len(com) != 0:
                return False

        return_True = True

    if all(isinstance(op, Operator) for op in ops):
        for oi, oj in combinations(ops, 2):
            com = simplify(commutator(oj, oi))
            if not equal(com, 0 * Identity()):
                return False

        return_True = True

    if all(isinstance(op, TensorLike) for op in ops):
        for oi, oj in combinations(ops, 2):
            com = oj @ oi - oi @ oj
            if not math.allclose(com, math.zeros_like(com)):
                return False

        return_True = True

    if not return_True:
        raise NotImplementedError(
            "At least one operator in the specified basis is of unsupported type, "
            "or not all operators are of the same type."
        )
    return return_True
