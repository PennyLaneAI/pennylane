# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This submodule contains the discrete-variable quantum observables,
excepting the Pauli gates and Hadamard gate in ``non_parametric_ops.py``.
"""

from scipy.sparse import coo_matrix

import numpy as np

import pennylane as qml
from pennylane.operation import AllWires, AnyWires, Observable
from pennylane.wires import Wires
from .matrix_ops import QubitUnitary


class Hermitian(Observable):
    r"""Hermitian(A, wires)
    An arbitrary Hermitian observable.

    For a Hermitian matrix :math:`A`, the expectation command returns the value

    .. math::
        \braket{A} = \braketT{\psi}{\cdots \otimes I\otimes A\otimes I\cdots}{\psi}

    where :math:`A` acts on the requested wires.

    If acting on :math:`N` wires, then the matrix :math:`A` must be of size
    :math:`2^N\times 2^N`.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        A (array): square hermitian matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_wires = AnyWires
    num_params = 1
    par_domain = "A"
    grad_method = "F"
    _eigs = {}

    @classmethod
    def _matrix(cls, *params):
        A = np.asarray(params[0])

        if A.shape[0] != A.shape[1]:
            raise ValueError("Observable must be a square matrix.")

        if not np.allclose(A, A.conj().T):
            raise ValueError("Observable must be Hermitian.")

        return A

    @property
    def eigendecomposition(self):
        """Return the eigendecomposition of the matrix specified by the Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the eigenvectors of the Hermitian observable
        """
        Hmat = self.matrix
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in Hermitian._eigs:
            w, U = np.linalg.eigh(Hmat)
            Hermitian._eigs[Hkey] = {"eigvec": U, "eigval": w}

        return Hermitian._eigs[Hkey]

    @property
    def eigvals(self):
        """Return the eigenvalues of the specified Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the Hermitian observable
        """
        return self.eigendecomposition["eigval"]

    def diagonalizing_gates(self):
        """Return the gate set that diagonalizes a circuit according to the
        specified Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            list: list containing the gates diagonalizing the Hermitian observable
        """
        return [QubitUnitary(self.eigendecomposition["eigvec"].conj().T, wires=list(self.wires))]


class SparseHamiltonian(Observable):
    r"""SparseHamiltonian(H)
    A Hamiltonian represented directly as a sparse matrix in coordinate list (COO) format.

    .. warning::

        ``SparseHamiltonian`` observables can only be used to return expectation values.
        Variances and samples are not supported.

    .. note::

        Note that the ``SparseHamiltonian`` observable should not be used with a subset of wires.

    **Details:**

    * Number of wires: All
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        H (coo_matrix): a sparse matrix in SciPy coordinate list (COO) format with
            dimension :math:`(2^n, 2^n)`, where :math:`n` is the number of wires
    """
    num_wires = AllWires
    num_params = 1
    par_domain = None
    grad_method = None

    @classmethod
    def _matrix(cls, *params):
        A = params[0]
        if not isinstance(A, coo_matrix):
            raise TypeError("Observable must be a scipy sparse coo_matrix.")
        return A

    def diagonalizing_gates(self):
        return []


class Projector(Observable):
    r"""Projector(basis_state, wires)
    Observable corresponding to the computational basis state projector :math:`P=\ket{i}\bra{i}`.

    The expectation of this observable returns the value

    .. math::
        |\langle \psi | i \rangle |^2

    corresponding to the probability of measuring the quantum state in the :math:`i` -th eigenstate of the specified :math:`n` qubits.

    For example, the projector :math:`\ket{11}\bra{11}` , or in integer notation :math:`\ket{3}\bra{3}`, is created by ``basis_state=np.array([1, 1])``.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        basis_state (tensor-like): binary input of shape ``(n, )``
        wires (Iterable): wires that the projector acts on
    """
    num_wires = AnyWires
    num_params = 1
    par_domain = "A"

    def __init__(self, basis_state, wires, do_queue=True):
        wires = Wires(wires)
        shape = qml.math.shape(basis_state)

        if len(shape) != 1:
            raise ValueError(f"Basis state must be one-dimensional; got shape {shape}.")

        n_basis_state = shape[0]
        if n_basis_state != len(wires):
            raise ValueError(
                f"Basis state must be of length {len(wires)}; got length {n_basis_state}."
            )

        basis_state = list(qml.math.toarray(basis_state))

        if not set(basis_state).issubset({0, 1}):
            raise ValueError(f"Basis state must only consist of 0s and 1s; got {basis_state}")

        super().__init__(basis_state, wires=wires, do_queue=do_queue)

    @classmethod
    def _eigvals(cls, *params):
        """Eigenvalues of the specific projector operator.

        Returns:
            array: eigenvalues of the projector observable in the computational basis
        """
        w = np.zeros(2 ** len(params[0]))
        idx = int("".join(str(i) for i in params[0]), 2)
        w[idx] = 1
        return w

    def diagonalizing_gates(self):
        """Return the gate set that diagonalizes a circuit according to the
        specified Projector observable.

        Returns:
            list: list containing the gates diagonalizing the projector observable
        """
        return []
