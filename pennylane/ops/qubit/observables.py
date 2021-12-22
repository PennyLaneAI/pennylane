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
    grad_method = "F"
    _eigs = {}

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "𝓗")

    @staticmethod
    def compute_matrix(A):  # pylint: disable=arguments-differ
        """Canonical matrix representation of the Hermitian operator.

        Args:
            A (tensor_like): hermitian matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> qml.Hermitian.compute_matrix(A)
        [[ 6.+0.j  1.-2.j]
         [ 1.+2.j -1.+0.j]]
        """
        A = qml.math.asarray(A)

        if A.shape[0] != A.shape[1]:
            raise ValueError("Observable must be a square matrix.")

        if not qml.math.allclose(A, A.conj().T):
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
        Hmat = self.matrix()
        Hmat = qml.math.to_numpy(Hmat)
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in Hermitian._eigs:
            w, U = np.linalg.eigh(Hmat)
            Hermitian._eigs[Hkey] = {"eigvec": U, "eigval": w}

        return Hermitian._eigs[Hkey]

    def eigvals(self):
        """Return the eigenvalues of the specified Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        Returns:
            array: array containing the eigenvalues of the Hermitian observable
        """
        return self.eigendecomposition["eigval"]

    @staticmethod
    def compute_diagonalizing_gates(eigenvectors, wires):  # pylint: disable=arguments-differ
        """Diagonalizing gates of this operator.

        Args:
            eigenvectors (array): eigenvectors of this operator, as extracted from op.eigendecomposition["eigvec"]
            wires (Iterable): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> A = np.array([[-6, 2 + 1j], [2 - 1j, 0]])
        >>> _, evecs = np.linalg.eigh(A)
        >>> qml.Hermitian.compute_diagonalizing_gates(evecs, wires=[0])
        [QubitUnitary(tensor([[-0.94915323-0.j,  0.2815786 +0.1407893j ],
                              [ 0.31481445-0.j,  0.84894846+0.42447423j]], requires_grad=True), wires=[0])]

        """
        return [QubitUnitary(eigenvectors.conj().T, wires=wires)]

    def diagonalizing_gates(self):
        """Return the gate set that diagonalizes a circuit according to the
        specified Hermitian observable.

        Returns:
            list: list containing the gates diagonalizing the Hermitian observable
        """
        # note: compute_diagonalizing_gates has a custom signature, which is why we overwrite this method
        return self.compute_diagonalizing_gates(self.eigendecomposition["eigvec"], self.wires)


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
    grad_method = None

    def __init__(self, H, wires=None, do_queue=True, id=None):
        if not isinstance(H, coo_matrix):
            raise TypeError("Observable must be a scipy sparse coo_matrix.")
        super().__init__(H, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        return super().label(decimals=decimals, base_label=base_label or "𝓗")

    @staticmethod
    def compute_matrix(H):  # pylint: disable=arguments-differ
        """Canonical matrix representation of the SparseHamiltonian operator.

        This method returns a dense matrix. For a sparse matrix representation, see
        :meth:`~.SparseHamiltonian.compute_sparse_matrix`.

        Args:
            H (scipy.sparse.coo_matrix): sparse matrix used to create this operator

        Returns:
            array: dense matrix

        **Example**

        >>> from scipy.sparse import coo_matrix
        >>> H = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> H = coo_matrix(H)
        >>> res = qml.SparseHamiltonian.compute_matrix(H)
        >>> res
        [[ 6.+0.j  1.-2.j]
         [ 1.+2.j -1.+0.j]]
        >>> type(res)
        <class 'numpy.ndarray'>
        """
        return H.toarray()

    @staticmethod
    def compute_sparse_matrix(H):  # pylint: disable=arguments-differ
        """Canonical matrix representation of the SparseHamiltonian operator, using a sparse matrix type.

        This method returns a sparse matrix. For a dense matrix representation, see
        :meth:`~.SparseHamiltonian.compute_matrix`.

        Args:
            H (scipy.sparse.coo_matrix): sparse matrix used to create this operator

        Returns:
            scipy.sparse.coo_matrix: sparse matrix

        **Example**

        >>> from scipy.sparse import coo_matrix
        >>> H = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> H = coo_matrix(H)
        >>> res = qml.SparseHamiltonian.compute_sparse_matrix(H)
        >>> res
        (0, 0)	(6+0j)
        (0, 1)	(1-2j)
        (1, 0)	(1+2j)
        (1, 1)	(-1+0j)
        >>> type(res)
        <class 'scipy.sparse.coo_matrix'>
        """
        return H


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

    @property
    def num_params(self):
        return 1

    def label(self, decimals=None, base_label=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label

        Returns:
            str: label to use in drawings

        **Example:**

        >>> qml.Projector([0, 1,0], wires=(0,1,2)).label()
        '|010⟩⟨010|'

        """

        if base_label is not None:
            return base_label
        basis_string = "".join(str(int(i)) for i in self.parameters[0])
        return f"|{basis_string}⟩⟨{basis_string}|"

    @staticmethod
    def compute_matrix(basis_state):  # pylint: disable=arguments-differ
        """Canonical matrix representation of the Projector operator.

        Args:
            basis_state (Iterable): basis state to project on

        Returns:
            array: canonical matrix

        **Example**

        >>> qml.Projector.compute_matrix([0, 1])
        [[0. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 0.]
         [0. 0. 0. 0.]]
        """
        m = np.zeros((2 ** len(basis_state), 2 ** len(basis_state)))
        idx = int("".join(str(i) for i in basis_state), 2)
        m[idx, idx] = 1
        return m

    @staticmethod
    def compute_eigvals(basis_state):  # pylint: disable=,arguments-differ
        """Eigenvalues of the Projector operator.

        Args:
            basis_state (Iterable): basis state to project on

        Returns:
            array: eigenvalues

        **Example**

        >>> qml.Projector.compute_eigvals([0, 1])
        [0. 1. 0. 0.]
        """
        w = np.zeros(2 ** len(basis_state))
        idx = int("".join(str(i) for i in basis_state), 2)
        w[idx] = 1
        return w

    @staticmethod
    def compute_diagonalizing_gates(
        basis_state, wires
    ):  # pylint: disable=arguments-differ,unused-argument
        """Diagonalizing gates of this operator.

        Args:
            basis_state (Iterable): basis state that the operator projects on
            wires (Iterable): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> qml.Projector.compute_diagonalizing_gates([0, 1, 0, 0], wires=[0, 1])
        []
        """
        return []
