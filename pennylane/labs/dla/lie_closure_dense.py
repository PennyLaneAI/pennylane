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
"""A function to compute the Lie closure of a set of operators"""
# pylint: disable=too-many-arguments

import warnings
from collections.abc import Iterable
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord

from .dense_util import trace_inner_product


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
        if not np.allclose(A.conj().T, A):
            A = 1j * A
            if not np.allclose(A.conj().T, A):
                raise ValueError(f"At least one basis matrix is not (skew-)Hermitian:\n{A}")

        B = A.copy()
        if len(basis) > 0:
            B -= np.tensordot(trace_inner_product(np.array(basis), A), basis, axes=[[0], [0]])
        if (norm := np.sqrt(trace_inner_product(B, B))) > tol:  # Tolerance for numerical stability
            B /= norm
            basis.append(B)
    return np.array(basis)


def lie_closure_dense(
    generators: Iterable[Union[PauliWord, PauliSentence, Operator, np.ndarray]],
    n: int = None,
    max_iterations: int = 10000,
    verbose: bool = False,
    tol: float = None,
):
    r"""Compute the dynamical Lie algebra :math:`\mathfrak{g}` from a set of generators using their dense matrix representation.

    This function computes the Lie closure of a set of generators using their dense matrix representation.
    This is sometimes more efficient than using the sparse Pauli representations of :class:`~PauliWord` and
    :class:`~PauliSentence` employed in :func:`~lie_closure`, e.g., when few generators are sums of many Paulis.

    .. seealso::

        For details on the mathematical definitions, see :func:`~lie_closure` and our
        `Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__.

    Args:
        generators (Iterable[Union[PauliWord, PauliSentence, Operator, np.ndarray]]): generating set for which to compute the
            Lie closure.
        n (int): The number of qubits involved. If ``None`` is provided, it is automatically deduced from the generators.
            Ignored when the inputs are ``np.ndarray`` instances.
        max_iterations (int): maximum depth of nested commutators to consider. Default is ``10000``.
        verbose (bool): whether to print out progress updates during Lie closure
            calculation. Default is ``False``.
        tol (float): Numerical tolerance for the linear independence check between algebra elements

    Returns:
        numpy.ndarray: The ``(dim(g), 2**n, 2**n)`` array containing the linearly independent basis of the DLA :math:`\mathfrak{g}` as dense matrices.

    **Example**

    Compute the Lie closure of the isotropic Heisenberg model with generators :math:`\{X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}\}_{i=0}^{n-1}`.

    >>> n = 5
    >>> gens = [X(i) @ X(i+1) + Y(i) @ Y(i+1) + Z(i) @ Z(i+1) for i in range(n-1)]
    >>> g = lie_closure_mat(gens, n)

    The result is a ``numpy`` array. We can turn the matrices back into PennyLane operators by employing :func:`~batched_pauli_decompose`.

    >>> g_ops = [qml.pauli_decompose(op) for op in g]

    **Internal representation**

    The input operators are converted to Hermitian matrices internally. This means
    that we compute the operators :math:`G_\alpha` in the algebra :math:`\{iG_\alpha\}_\alpha`,
    which itself consists of skew-Hermitian objects (commutators produce skew-Hermitian objects,
    so Hermitian operators alone can not form an algebra with the standard commutator).
    """

    dense_in = isinstance(generators, np.ndarray) or all(
        isinstance(op, np.ndarray) for op in generators
    )

    if not dense_in:
        if n is None:
            all_wires = qml.wires.Wires.all_wires([_.wires for _ in generators])
            n = len(all_wires)
            assert all_wires.toset() == set(range(n))

        gens = np.array([qml.matrix(op, wire_order=range(n)) for op in generators], dtype=complex)
        chi = 2**n
        assert gens.shape == (len(generators), chi, chi)

    else:
        gens = np.array(generators)
        chi = generators[0].shape[0]
        assert gens.shape == (len(generators), chi, chi)

    epoch = 0
    old_length = 0
    vspace = _hermitian_basis(gens, tol, old_length)
    new_length = initial_length = len(vspace)

    while (new_length > old_length) and (epoch < max_iterations):
        if verbose:
            print(f"epoch {epoch+1} of lie_closure_dense, DLA size is {new_length}")

        # compute all commutators. We compute the commutators between all newly added operators
        # and all original generators. This limits the amount of vectorization we are doing but
        # gives us a correspondence between the while loop iteration and the nesting level of
        # the commutators.
        # [m0, m1] = m0 m1 - m1 m0
        # Implement einsum "aij,bjk->abik" by tensordot and moveaxis
        m0m1 = np.moveaxis(
            np.tensordot(vspace[old_length:], vspace[:initial_length], axes=[[2], [1]]), 1, 2
        )
        m0m1 = np.reshape(m0m1, (-1, chi, chi))

        # Implement einsum "aij,bki->abkj" by tensordot and moveaxis
        m1m0 = np.moveaxis(
            np.tensordot(vspace[old_length:], vspace[:initial_length], axes=[[1], [2]]), 1, 3
        )
        m1m0 = np.reshape(m1m0, (-1, chi, chi))
        all_coms = m0m1 - m1m0

        # sub-select linearly independent subset
        vspace = np.concatenate([vspace, all_coms])
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
