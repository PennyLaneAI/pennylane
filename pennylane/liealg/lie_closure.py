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
"""A function to compute the Lie closure of a set of operators"""
import warnings
from collections.abc import Iterable
from copy import copy

# pylint: disable=too-many-arguments
from itertools import product

import numpy as np

import pennylane.ops.functions as op_func
from pennylane import math
from pennylane.operation import Operator
from pennylane.pauli import (
    PauliSentence,
    PauliVSpace,
    PauliWord,
    pauli_sentence,
    trace_inner_product,
)
from pennylane.typing import TensorLike
from pennylane.wires import Wires


def lie_closure(
    generators: Iterable[PauliWord | PauliSentence | Operator | TensorLike],
    *,  # force non-positional kwargs of the following
    max_iterations: int = 10000,
    verbose: bool = False,
    pauli: bool = False,
    matrix: bool = False,
    tol: float = None,
) -> Iterable[PauliWord | PauliSentence | Operator | np.ndarray]:
    r"""Compute the (dynamical) Lie algebra from a set of generators.

    The Lie closure, pronounced "Lee" closure, is a way to compute the so-called dynamical Lie algebra (DLA) of a set of generators :math:`\mathcal{G} = \{G_1, .. , G_N\}`.
    For such generators, one computes all nested commutators :math:`[G_i, [G_j, .., [G_k, G_\ell]]]` until no new operators are generated from commutation.
    All these operators together form the DLA, see e.g. section IIB of `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__.

    Args:
        generators (Iterable[Union[PauliWord, PauliSentence, Operator, TensorLike]]): generating set for which to compute the
            Lie closure.
        max_iterations (int): maximum depth of nested commutators to consider. Default is ``10000``.
        verbose (bool): whether to print out progress updates during Lie closure
            calculation. Default is ``False``.
        pauli (bool): Indicates whether it is assumed that :class:`~.PauliSentence` or :class:`~.PauliWord` instances are input and returned.
            This can help with performance to avoid unnecessary conversions to :class:`~pennylane.operation.Operator`
            and vice versa. Default is ``False``.
        matrix (bool): Whether or not matrix representations should be used and returned in the Lie closure computation. This can help
            speed up the computation when using sums of Paulis with many terms. Default is ``False``.
        tol (float): Numerical tolerance for the linear independence check used in :class:`~.PauliVSpace`.

    Returns:
        Union[list[:class:`~.PauliSentence`], list[:class:`~.Operator`], np.ndarray]: A basis of either :class:`~.PauliSentence`,
        :class:`~.Operator`, or ``np.ndarray`` instances that is closed under commutators (Lie closure).

    .. seealso:: :func:`~structure_constants`, :func:`~center`, :class:`~pennylane.pauli.PauliVSpace`, `Demo: Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__

    **Example**

    >>> from pennylane import X, Y, Z
    >>> ops = [X(0) @ X(1), Z(0), Z(1)]
    >>> dla = qml.lie_closure(ops)

    Let us walk through what happens in this simple example of computing the Lie closure of these generators (the transverse field Ising model on two qubits).
    A first round of commutators between all elements yields:

    >>> qml.commutator(X(0) @ X(1), Z(0))
    -2j * (Y(0) @ X(1))
    >>> qml.commutator(X(0) @ X(1), Z(1))
    -2j * (X(0) @ Y(1))

    A next round of commutators between all elements further yields the new operator ``Y(0) @ Y(1)``.

    >>> qml.commutator(X(0) @ Y(1), Z(0))
    -2j * (Y(0) @ Y(1))

    After that, no new operators emerge from taking nested commutators and we have the resulting DLA.
    This can be done in short via ``lie_closure`` as follows.

    >>> ops = [X(0) @ X(1), Z(0), Z(1)]
    >>> dla = qml.lie_closure(ops)
    >>> dla
    [X(0) @ X(1), Z(0), Z(1), -1.0 * (Y(0) @ X(1)), -1.0 * (X(0) @ Y(1)), Y(0) @ Y(1)]

    Note that we normalize by removing the factors of :math:`2i`, though minus signs are left intact.

    .. details::
        :title: Usage Details

        Note that by default, ``lie_closure`` returns PennyLane operators. Internally we use the more
        efficient representation in terms of :class:`~pennylane.pauli.PauliSentence` by making use of the ``op.pauli_rep``
        attribute of operators composed of Pauli operators. If desired, this format can be returned by using
        the keyword ``pauli=True``. In that case, the input is also assumed to be a :class:`~pennylane.pauli.PauliSentence` instance.

        >>> ops = [
        ...     PauliSentence({PauliWord({0: "X", 1: "X"}): 1.}),
        ...     PauliSentence({PauliWord({0: "Z"}): 1.}),
        ...     PauliSentence({PauliWord({1: "Z"}): 1.}),
        ... ]
        >>> dla = qml.lie_closure(ops, pauli=True)
        >>> dla
        [1.0 * X(0) @ X(1), 1.0 * Z(0), 1.0 * Z(1), -1.0 * Y(0) @ X(1), -1.0 * X(0) @ Y(1), 1.0 * Y(0) @ Y(1)]
        >>> type(dla[0])
        <class 'pennylane.pauli.pauli_arithmetic.PauliSentence'>

        In the case of sums of Pauli operators with many terms, it is often faster to use the matrix representation of the operators rather than
        the semi-analytic :class:`~pennylane.pauli.PauliSentence` or :class:`~Operator` representation.
        We can force this by using the ``matrix`` keyword. The resulting ``dla`` is a ``np.ndarray`` of dimension ``(dim_g, 2**n, 2**n)``, where ``dim_g`` is the
        dimension of the DLA and ``n`` the number of qubits.

        >>> dla = qml.lie_closure(ops, matrix=True)
        >>> dla.shape
        (6, 4, 4)

        You can retrieve a semi-analytic representation again by using :func:`~pauli_decompose`.

        >>> dla_ops = [qml.pauli_decompose(op) for op in dla]
        >>> dla_ops
        [1.0 * (X(0) @ X(1)),
         1.0 * (Z(0) @ I(1)),
         1.0 * (I(0) @ Z(1)),
         1.0 * (Y(0) @ X(1)),
         1.0 * (X(0) @ Y(1)),
         1.0 * (Y(0) @ Y(1))]

    """
    if matrix:
        return _lie_closure_matrix(generators, max_iterations, verbose, tol)

    if not all(isinstance(op, (PauliSentence, PauliWord)) for op in generators):
        if pauli:
            raise TypeError(
                "All generators need to be of type PauliSentence or PauliWord when using pauli=True in lie_closure."
            )

        generators = [
            rep if (rep := op.pauli_rep) is not None else pauli_sentence(op) for op in generators
        ]

    vspace = PauliVSpace(generators, tol=tol)

    epoch = 0
    old_length = 0  # dummy value
    new_length = initial_length = len(vspace)

    while (new_length > old_length) and (epoch < max_iterations):
        if verbose:
            print(f"epoch {epoch+1} of lie_closure, DLA size is {new_length}")

        # compute all commutators. We compute the commutators between all newly added operators
        # and all original generators. This limits the number of commutators added in each
        # iteration, but it gives us a correspondence between the while loop iteration and the
        # nesting level of the commutators.
        for ps1, ps2 in product(vspace.basis[old_length:], vspace.basis[:initial_length]):
            com = ps1.commutator(ps2)
            com.simplify(tol=vspace.tol)

            if len(com) == 0:  # skip because operators commute
                continue

            # result is always purely imaginary
            # remove common factor 2 with Pauli commutators
            for pw, val in com.items():
                com[pw] = val.imag / 2

            vspace.add(com, tol=tol)

        # Updated number of linearly independent PauliSentences from previous and current step
        old_length = new_length
        new_length = len(vspace)
        epoch += 1

        if epoch == max_iterations:
            warnings.warn(f"reached the maximum number of iterations {max_iterations}", UserWarning)

    if verbose > 0:
        print(f"After {epoch} epochs, reached a DLA size of {new_length}")

    res = vspace.basis
    if not pauli:
        res = [op.operation() for op in res]

    return res


def _hermitian_basis(matrices: Iterable[np.ndarray], tol: float = None, subbasis_length: int = 0):
    """Find a linearly independent basis of a list of (skew-) Hermitian matrices

    .. note:: The first ``subbasis_length`` elements of ``matrices`` are assumed to already be orthogonal and Hermitian and will not be changed.

    Args:
        matrices (Union[numpy.ndarray, Iterable[numpy.ndarray]]): A list of Hermitian matrices.
        tol (float): Tolerance for linear dependence check. Defaults to ``1e-10``.
        subbasis_length (int): The first `subbasis_length` elements in `matrices` are left untouched.

    Returns:
        np.ndarray: Stacked array of linearly independent basis matrices.

    Raises:
        ValueError: If not all input matrices are (skew-) Hermitian.
    """
    if tol is None:
        tol = 1e-10

    basis = list(matrices[:subbasis_length])
    for A in matrices[subbasis_length:]:
        if not math.is_abstract(A):
            if not math.allclose(math.transpose(math.conj(A)), A):
                A = 1j * A
                if not math.allclose(math.transpose(math.conj(A)), A):
                    raise ValueError(f"At least one basis matrix is not (skew-)Hermitian:\n{A}")

        B = copy(A)
        if len(basis) > 0:
            lhs = trace_inner_product(basis, A)
            B -= math.tensordot(lhs, math.stack(basis), axes=[[0], [0]])
        if (
            norm := math.real(math.sqrt(trace_inner_product(B, B)))
        ) > tol:  # Tolerance for numerical stability
            B /= math.cast_like(norm, B)
            basis.append(B)
    return math.array(basis)


def _lie_closure_matrix(
    generators: Iterable[PauliWord | PauliSentence | Operator | np.ndarray],
    max_iterations: int = 10000,
    verbose: bool = False,
    tol: float = None,
):
    r"""Compute the dynamical Lie algebra :math:`\mathfrak{g}` from a set of generators using their matrix representation.

    This function computes the Lie closure of a set of generators using their matrix representation.
    This is sometimes more efficient than using the sparse Pauli representations of :class:`~PauliWord` and
    :class:`~PauliSentence` employed in :func:`~lie_closure`, e.g., when few generators are sums of many Paulis.

    .. seealso::

        For details on the mathematical definitions, see :func:`~lie_closure` and our
        `Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__.

    Args:
        generators (Iterable[Union[PauliWord, PauliSentence, Operator, np.ndarray]]): generating set for which to compute the
            Lie closure.
        max_iterations (int): maximum depth of nested commutators to consider. Default is ``10000``.
        verbose (bool): whether to print out progress updates during Lie closure
            calculation. Default is ``False``.
        tol (float): Numerical tolerance for the linear independence check between algebra elements

    Returns:
        numpy.ndarray: The ``(dim(g), 2**n, 2**n)`` array containing the linearly independent basis of the DLA :math:`\mathfrak{g}` as matrices.

    **Example**

    Compute the Lie closure of the isotropic Heisenberg model with generators :math:`\{X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}\}_{i=0}^{n-1}`.

    >>> n = 5
    >>> gens = [X(i) @ X(i+1) + Y(i) @ Y(i+1) + Z(i) @ Z(i+1) for i in range(n-1)]
    >>> g = _lie_closure_matrix(gens)

    The result is a ``numpy`` array. We can turn the matrices back into PennyLane operators by employing :func:`~batched_pauli_decompose`.

    >>> g_ops = [qml.pauli_decompose(op) for op in g]

    **Internal representation**

    The input operators are converted to Hermitian matrices internally. This means
    that we compute the operators :math:`G_\alpha` in the algebra :math:`\{iG_\alpha\}_\alpha`,
    which itself consists of skew-Hermitian objects (commutators produce skew-Hermitian objects,
    so Hermitian operators alone can not form an algebra with the standard commutator).
    """

    if not isinstance(generators[0], TensorLike):
        # operator input
        all_wires = Wires.all_wires([_.wires for _ in generators])

        n = len(all_wires)
        assert all_wires.toset() == set(range(n))

        generators = np.array(
            [op_func.matrix(op, wire_order=range(n)) for op in generators], dtype=complex
        )
        chi = 2**n
        assert np.shape(generators) == (len(generators), chi, chi)

    elif isinstance(generators[0], TensorLike) and isinstance(generators, (list, tuple)):
        # list of matrices
        interface = math.get_interface(generators[0])
        generators = math.stack(generators, like=interface)

    chi = math.shape(generators[0])[0]
    assert math.shape(generators) == (len(generators), chi, chi)

    epoch = 0
    old_length = 0
    vspace = _hermitian_basis(generators, tol, old_length)
    new_length = initial_length = len(vspace)

    while (new_length > old_length) and (epoch < max_iterations):
        if verbose:
            print(f"epoch {epoch+1} of lie_closure, DLA size is {new_length}")

        # compute all commutators. We compute the commutators between all newly added operators
        # and all original generators. This limits the amount of vectorization we are doing but
        # gives us a correspondence between the while loop iteration and the nesting level of
        # the commutators.
        # [m0, m1] = m0 m1 - m1 m0
        # Implement einsum "aij,bjk->abik" by tensordot and moveaxis
        m0m1 = math.moveaxis(
            math.tensordot(vspace[old_length:], vspace[:initial_length], axes=[[2], [1]]), 1, 2
        )
        m0m1 = math.reshape(m0m1, (-1, chi, chi))

        # Implement einsum "aij,bki->abkj" by tensordot and moveaxis
        m1m0 = math.moveaxis(
            math.tensordot(vspace[old_length:], vspace[:initial_length], axes=[[1], [2]]), 1, 3
        )
        m1m0 = math.reshape(m1m0, (-1, chi, chi))
        all_coms = m0m1 - m1m0

        # sub-select linearly independent subset
        vspace = math.concatenate([vspace, all_coms])
        vspace = _hermitian_basis(vspace, tol, old_length)

        # Updated number of linearly independent PauliSentences from previous and current step
        old_length = new_length
        new_length = len(vspace)
        epoch += 1

        if epoch == max_iterations:
            warnings.warn(f"reached the maximum number of iterations {max_iterations}", UserWarning)

    if verbose:
        print(f"After {epoch} epochs, reached a DLA size of {new_length}")

    return vspace
