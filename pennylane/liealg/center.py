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
"""A function to compute the center of a Lie algebra"""
from itertools import combinations

import numpy as np
from scipy.linalg import norm, null_space

from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord

from .structure_constants import structure_constants


def _intersect_bases(basis_0, basis_1, rcond=None):
    r"""Compute the intersection of two vector spaces that are given by a basis each.
    This is done by constructing a matrix [basis_0 | -basis_1] and computing its null space
    in form of vectors (u, v)^T, which is equivalent to solving the equation
    ``basis_0 @ u = basis_1 @ v``.
    Given a basis for this null space, the vectors ``basis_0 @ u`` (or equivalently
    ``basis_1 @ v``) form a basis for the intersection of the vector spaces.

    Also see https://math.stackexchange.com/questions/25371/how-to-find-a-basis-for-the-intersection-of-two-vector-spaces-in-mathbbrn
    """
    # Compute (orthonormal) basis for the null space of the augmented matrix [basis_0, -basis_1]
    augmented_basis = null_space(np.hstack([basis_0, -basis_1]), rcond=rcond)
    # Compute basis_0 @ u for each vector u from the basis (u, v)^T in the augmented basis
    intersection_basis = basis_0 @ augmented_basis[: basis_0.shape[1]]
    # Normalize the output for cleaner results, because the augmented kernel was normalized
    intersection_basis = intersection_basis / norm(intersection_basis, axis=0)
    return intersection_basis


def _center_pauli_words(g, pauli):
    """Compute the center of an algebra given in a PauliWord basis."""
    # Guarantees all operators to be given as a PauliWord
    # If `pauli=True` we know that they are PauliWord or PauliSentence instances
    g_pws = [o if isinstance(o, PauliWord) else next(iter(o.pauli_rep.keys())) for o in g]
    d = len(g_pws)
    commutators = np.zeros((d, d), dtype=int)
    for (j, op1), (k, op2) in combinations(enumerate(g_pws), r=2):
        if not op1.commutes_with(op2):
            commutators[j, k] = 1  # dummy value to indicate operators dont commute
            commutators[k, j] = 1

    ids = np.where(np.all(commutators == 0, axis=0))[0]
    res = [g[idx] for idx in ids]

    if not pauli:
        res = [op.operation() if isinstance(op, (PauliWord, PauliSentence)) else op for op in res]
    return res


def center(
    g: list[Operator | PauliWord | PauliSentence], pauli: bool = False
) -> list[Operator | PauliSentence]:
    r"""
    Compute the center of a Lie algebra.

    Given a Lie algebra :math:`\mathfrak{g} = \{h_1,.., h_d\}`, the center :math:`\mathfrak{\xi}(\mathfrak{g})`
    is given by all elements in :math:`\mathfrak{g}` that commute with `all` other elements in :math:`\mathfrak{g}`,

    .. math:: \mathfrak{\xi}(\mathfrak{g}) := \{h \in \mathfrak{g} | [h, h_i]=0 \ \forall h_i \in \mathfrak{g} \}

    Args:
        g (List[Union[Operator, PauliSentence, PauliWord]]): List of operators that spans
            the algebra for which to find the center.
        pauli (bool): Indicates whether it is assumed that :class:`~.PauliSentence` or
            :class:`~.PauliWord` instances are input and returned. This can help with performance
            to avoid unnecessary conversions to :class:`~pennylane.operation.Operator`
            and vice versa. Default is ``False``.

    Returns:
        List[Union[Operator, PauliSentence]]: The center of the Lie algebra ``g``.

    .. seealso:: :func:`~lie_closure`, :func:`~structure_constants`, :class:`~pennylane.pauli.PauliVSpace`, `Demo: Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__

    **Example**

    We can compute the center of a DLA ``g``. First we compute the DLA via :func:`~lie_closure`.

    >>> generators = [qml.X(0), qml.X(0) @ qml.X(1), qml.Y(1)]
    >>> g = qml.lie_closure(generators)
    >>> g
    [X(0), X(0) @ X(1), Y(1), X(0) @ Z(1)]

    The ``center`` is then the collection of operators that commute with `all` other operators in the DLA.
    In this case, just ``X(0)``.

    >>> qml.center(g)
    [X(0)]

    .. details::
        :title: Derivation
        :href: derivation

        The center :math:`\mathfrak{\xi}(\mathfrak{g})` of an algebra :math:`\mathfrak{g}`
        can be computed in the following steps. First, compute the
        :func:`~.pennylane.structure_constants`, or adjoint representation, of the algebra
        with respect to some basis :math:`\mathbb{B}` of :math:`\mathfrak{g}`.
        The center of :math:`\mathfrak{g}` is then given by

        .. math::

            \mathfrak{\xi}(\mathfrak{g}) = \operatorname{span}\left\{\bigcap_{x\in\mathbb{B}}
            \operatorname{ker}(\operatorname{ad}_x)\right\},

        i.e., the intersection of the kernels, or null spaces, of all basis elements in the
        adjoint representation.

        The kernel can be computed with ``scipy.linalg.null_space``, and vector space
        intersections are computed recursively from pairwise intersections. The intersection
        between two vectors spaces :math:`V_1` and :math:`V_2` given by (orthonormal) bases
        :math:`\mathbb{B}_i` can be computed from the kernel of the matrix that has all
        basis vectors from :math:`\mathbb{B}_1` and :math:`-\mathbb{B}_2` as columns, i.e.,
        :math:`\operatorname{ker}(\left[\ \mathbb{B}_1 \ | -\mathbb{B}_2\ \right])`. For an
        (orthonormal) basis of this kernel, consisting of two stacked column vectors
        :math:`u^{(i)}_1` and :math:`u^{(i)}_2` for each basis, a basis of the
        intersection space :math:`V_1 \cap V_2` is given by :math:`\{\mathbb{B}_1 u_1^{(i)}\}_i`
        (or equivalently by :math:`\{\mathbb{B}_2 u_2^{(i)}\}_i`).
        Also see `this post <https://math.stackexchange.com/questions/25371/how-to-find-a-basis-for-the-intersection-of-two-vector-spaces-in-mathbbrn>`_
        for details.

        If the input consists of :class:`~.pennylane.PauliWord` instances only, we can
        instead compute pairwise commutators and know that the center consists solely of
        basis elements that commute with all other basis elements. This can be seen in the
        following way.

        Assume that the center elements identified based on the basis have been removed
        already and we are left with a basis :math:`\mathbb{B}=\{p_i\}_i` of Pauli
        words such that :math:`\forall i\ \exists j:\ [p_i, p_j] \neq 0`. Assume that there is
        another center element :math:`x\neq 0`, which was missed before because it is a linear
        combination of Pauli words:

        .. math::

            \forall j: \ [x, p_j] = [\sum_i x_i p_i, p_j] = 0.

        As products of Paulis are unique when fixing one of the factors (:math:`p_j` is fixed
        above), we then know that

        .. math::

            &\forall j: \ 0 = \sum_i x_i [p_i, p_j] = 2 \sum_i x_i \chi_{i,j} p_ip_j\\
            \Rightarrow &\forall i,j \text{ s.t. } \chi_{i,j}\neq 0: x_i = 0,

        where :math:`\chi_{i,j}` denotes an indicator that is :math:`0` if the commutator
        :math:`[p_i, p_j]` vanishes and :math:`1` otherwise.
        However, we know that for each :math:`i` there is at least one :math:`j` such that
        :math:`\chi_{i,j}\neq 0`. This means that :math:`x_i = 0` is guaranteed for all
        :math:`i` by at least one :math:`j`. Therefore :math:`x=0`, which is a contradiction
        to our initial assumption that :math:`x\neq 0`.
    """
    if len(g) < 2:
        # A length-zero list has zero center, a length-one list has full center
        return g
    if all(isinstance(x, PauliWord) or len(x.pauli_rep) == 1 for x in g):
        return _center_pauli_words(g, pauli)

    adjoint_repr = structure_constants(g, pauli)
    # Start kernels intersection with kernel of first DLA element
    kernel_intersection = null_space(adjoint_repr[0])
    for ad_x in adjoint_repr[1:]:
        # Compute the next kernel and intersect it with previous intersection
        next_kernel = null_space(ad_x)
        kernel_intersection = _intersect_bases(kernel_intersection, next_kernel)

        # If the intersection is zero-dimensional, exit early
        if kernel_intersection.shape[1] == 0:
            return []

    # Construct operators from numerical output and convert to desired format
    res = [sum(c * x for c, x in zip(c_coeffs, g)) for c_coeffs in kernel_intersection.T]

    have_paulis = all(isinstance(x, (PauliWord, PauliSentence)) for x in res)
    if pauli or have_paulis:
        _ = [el.simplify() for el in res]
        if not pauli:
            res = [el.operation() for el in res]
    else:
        res = [el.simplify() for el in res]

    return res
