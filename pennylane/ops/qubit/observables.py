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

from copy import copy

import numpy as np
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane.operation import AllWires, AnyWires, Observable
from pennylane.wires import Wires

from .matrix_ops import QubitUnitary


class Hermitian(Observable):
    r"""
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
        A (array or Sequence): square hermitian matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = "F"
    _eigs = {}

    def __init__(self, A, wires, do_queue=True, id=None):
        A = qml.math.asarray(A)
        if not qml.math.is_abstract(A):
            Hermitian._validate_input(A)

        super().__init__(A, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def _validate_input(A):
        """Validate the input matrix."""
        if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Observable must be a square matrix.")

        if not qml.math.allclose(A, qml.math.T(qml.math.conj(A))):
            raise ValueError("Observable must be Hermitian.")

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "𝓗", cache=cache)

    @staticmethod
    def compute_matrix(A):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Hermitian.matrix`

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
        Hermitian._validate_input(A)
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
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Hermitian.diagonalizing_gates`.

        Args:
            eigenvectors (array): eigenvectors of the operator, as extracted from op.eigendecomposition["eigvec"]
            wires (Iterable[Any], Wires): wires that the operator acts on
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
    r"""
    A Hamiltonian represented directly as a sparse matrix in Compressed Sparse Row (CSR) format.

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
        H (csr_matrix): a sparse matrix in SciPy Compressed Sparse Row (CSR) format with
            dimension :math:`(2^n, 2^n)`, where :math:`n` is the number of wires
        wires (Sequence[int] or int): the wire(s) the operation acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)

    **Example**

    Sparse Hamiltonians can be constructed directly with a SciPy-compatible sparse matrix.

    Alternatively, you can construct your Hamiltonian as usual using :class:`~.Hamiltonian`, and then use
    the utility function :func:`~.utils.sparse_hamiltonian` to construct the sparse matrix that serves as the input
    to ``SparseHamiltonian``:

    >>> wires = 20
    >>> coeffs = [1 for _ in range(wires)]
    >>> observables = [qml.PauliZ(i) for i in range(wires)]
    >>> H = qml.Hamiltonian(coeffs, observables)
    >>> Hmat = qml.utils.sparse_hamiltonian(H)
    >>> H_sparse = qml.SparseHamiltonian(Hmat, wires=wires)
    """
    num_wires = AllWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None

    def __init__(self, H, wires=None, do_queue=True, id=None):
        if not isinstance(H, csr_matrix):
            raise TypeError("Observable must be a scipy sparse csr_matrix.")
        super().__init__(H, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return super().label(decimals=decimals, base_label=base_label or "𝓗", cache=cache)

    @staticmethod
    def compute_matrix(H):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SparseHamiltonian.matrix`


        This method returns a dense matrix. For a sparse matrix representation, see
        :meth:`~.SparseHamiltonian.compute_sparse_matrix`.

        Args:
            H (scipy.sparse.csr_matrix): sparse matrix used to create the operator

        Returns:
            array: dense matrix

        **Example**

        >>> from scipy.sparse import csr_matrix
        >>> H = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> H = csr_matrix(H)
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
        r"""Representation of the operator as a sparse canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SparseHamiltonian.sparse_matrix`

        This method returns a sparse matrix. For a dense matrix representation, see
        :meth:`~.SparseHamiltonian.compute_matrix`.

        Args:
            H (scipy.sparse.csr_matrix): sparse matrix used to create the operator

        Returns:
            scipy.sparse.csr_matrix: sparse matrix

        **Example**

        >>> from scipy.sparse import csr_matrix
        >>> H = np.array([[6+0j, 1-2j],[1+2j, -1]])
        >>> H = csr_matrix(H)
        >>> res = qml.SparseHamiltonian.compute_sparse_matrix(H)
        >>> res
        (0, 0)	(6+0j)
        (0, 1)	(1-2j)
        (1, 0)	(1+2j)
        (1, 1)	(-1+0j)
        >>> type(res)
        <class 'scipy.sparse.csr_matrix'>
        """
        return H


class Projector(Observable):
    r"""
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
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """
    num_wires = AnyWires
    num_params = 1
    """int: Number of trainable parameters that the operator depends on."""

    def __init__(self, basis_state, wires, do_queue=True, id=None):
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

        super().__init__(basis_state, wires=wires, do_queue=do_queue, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        r"""A customizable string representation of the operator.

        Args:
            decimals=None (int): If ``None``, no parameters are included. Else,
                specifies how to round the parameters.
            base_label=None (str): overwrite the non-parameter component of the label
            cache=None (dict): dictionary that caries information between label calls
                in the same drawing

        Returns:
            str: label to use in drawings

        **Example:**

        >>> qml.Projector([0, 1, 0], wires=(0, 1, 2)).label()
        '|010⟩⟨010|'

        """

        if base_label is not None:
            return base_label
        basis_string = "".join(str(int(i)) for i in self.parameters[0])
        return f"|{basis_string}⟩⟨{basis_string}|"

    @staticmethod
    def compute_matrix(basis_state):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Projector.matrix`

        Args:
            basis_state (Iterable): basis state to project on

        Returns:
            ndarray: matrix

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
    def compute_eigvals(basis_state):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Projector.eigvals`

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
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Projector.diagonalizing_gates`.

        Args:
            basis_state (Iterable): basis state that the operator projects on
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> qml.Projector.compute_diagonalizing_gates([0, 1, 0, 0], wires=[0, 1])
        []
        """
        return []

    def pow(self, z):
        return [copy(self)] if (isinstance(z, int) and z > 0) else super().pow(z)
