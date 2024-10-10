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

from pennylane.pauli import PauliWord, PauliSentence
from pennylane.operation import Operator


def _hermitian_basis(matrices, tol=None):
    """Find a linear independent basis of a list of Hermitian matrices

    Args:
        matrices (Iterable[numpy.ndarray]): A list of Hermitian matrices.
        tol (float): Tolerance for linear dependence check. Defaults to ``1e-10``.
    """
    if tol is None:
        tol = 1e-10

    basis = []
    for A in matrices:
        if not np.allclose(A.conj().T, A):
            A = 1j * A
            if not np.allclose(A.conj().T, A):
                warnings.warn("Some basis matrices are not Hermitian", UserWarning)

        B = A.copy()
        for C in basis:
            B -= np.trace(np.dot(C.conj().T, A)) * C
        if np.linalg.norm(B) > tol:  # Tolerance for numerical stability
            B /= np.linalg.norm(B)
            basis.append(B)
    return np.array(basis)


def lie_closure_dense(
    generators: Iterable[Union[PauliWord, PauliSentence, Operator]],
    n=None,
    max_iterations: int = 10000,
    verbose: bool = False,
    tol: float = None,
):
    r"""Compute the dynamical Lie algebra :math:`frak{g}` from a set of generators using their dense matrix representation.

    This function computes the Lie closure of a set of generators using their dense matrix representation.
    This is sometimes more efficient than using the sparse Pauli representations of :class:`~PauliWord` and
    `~PauliSentence` that are employed in :func:`~lie_closure`, e.g., when ther are few but dense sums of Paulis.

    .. seealso:: For details on the mathematical definitions, see :func:`~lie_closure` and our `Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__.

    Args:
        generators (Iterable[Union[PauliWord, PauliSentence, Operator]]): generating set for which to compute the
            Lie closure.
        n (int): The number of qubits involved. If ``None`` is provided, it is automatically deduced from the generators.
        max_iterations (int): maximum depth of nested commutators to consider. Default is ``10000``.
        verbose (bool): whether to print out progress updates during Lie closure
            calculation. Default is ``False``.
        tol (float): Numerical tolerance for the linear independence check used in :class:`~.PauliVSpace`.

    Returns:
        numpy.ndarray: The ``(dim(g), 2**n, 2**n)`` array containing the linear independent basis of the DLA g as dense matrices.

    **Example**

    Compute the Lie closure of the isotropic Heisenberg model with generators :math:`\{X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}}_{i=0}^{n-1}`.

    >>> n = 5
    >>> gens = [X(i) @ X(i+1) + Y(i) @ Y(i+1) + Z(i) @ Z(i+1) for i in range(n-1)]
    >>> g = lie_closure_mat(gens, n)

    The result is a ``numpy`` array. We can turn the matrices back into PennyLane operators by employing :func:`~pauli_decompose`.

    >>> g_ops = [qml.pauli_decompose(op) for op in g]

    """

    if n is None:
        n = len(qml.wires.Wires.all_wires([_.wires for _ in generators]))

    gens = np.array([qml.matrix(op, wire_order=range(n)) for op in generators], dtype=complex)
    _, chi, _ = gens.shape
    vspace = _hermitian_basis(gens, tol)

    epoch = 0
    old_length = 0  # dummy value
    new_length = len(vspace)

    while (new_length > old_length) and (epoch < max_iterations):
        if verbose:
            print(f"epoch {epoch+1} of lie_closure_dense, DLA size is {new_length}")

        # compute all commutators
        m0m1 = np.einsum("aij,bjk->abik", vspace, gens)
        m0m1 = np.reshape(m0m1, (-1, chi, chi))

        m1m0 = np.einsum("aij,bki->abkj", vspace, gens)
        m1m0 = np.reshape(m1m0, (-1, chi, chi))
        all_coms = m0m1 - m1m0

        # sub-select linear independent subset
        vspace = np.concatenate([vspace, all_coms])
        vspace = _hermitian_basis(vspace, tol)

        # Updated number of linearly independent PauliSentences from previous and current step
        old_length = new_length
        new_length = len(vspace)
        epoch += 1

        if epoch == max_iterations:
            warnings.warn(f"reached the maximum number of iterations {max_iterations}", UserWarning)

    if verbose > 0:
        print(f"After {epoch} epochs, reached a DLA size of {new_length}")

    return vspace
