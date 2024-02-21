# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule contains the qutrit quantum observables.
"""
import numpy as np

import pennylane as qml  # pylint: disable=unused-import
from pennylane.operation import Observable
from pennylane.ops.qubit import Hermitian
from pennylane.ops.qutrit import QutritUnitary


# Array containing all 8 Gell-Mann matrices. This order is chosen for the Gell-Mann
# matrices as they loosely follow the X-Y-Z structure of the Pauli matrices. The
# matrices analogous to Pauli X, Y, and Z are at indices (0, 3, 5), (1, 4, 6), and
# (2, 7) respectively.
gm_mats = np.zeros((8, 3, 3), dtype=np.complex128)
gm_mats[0] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
gm_mats[1] = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
gm_mats[2] = np.diag([1, -1, 0])
gm_mats[3] = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
gm_mats[4] = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
gm_mats[5] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
gm_mats[6] = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
gm_mats[7] = np.diag([1, 1, -2]) / np.sqrt(3)


class THermitian(Hermitian):
    r"""An arbitrary Hermitian observable for qutrits.

    For a Hermitian matrix :math:`A`, the expectation command returns the value

    .. math::
        \braket{A} = \braketT{\psi}{\cdots \otimes I\otimes A\otimes I\cdots}{\psi}

    where :math:`A` acts on the requested wires.

    If acting on :math:`N` wires, then the matrix :math:`A` must be of size
    :math:`3^N\times 3^N`.

    **Details:**

    * Number of wires: Any
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        A (array): square Hermitian matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str or None): String representing the operation (optional)

    .. note::
        :class:`Hermitian` cannot be used with qutrit devices due to its use of
        :class:`QubitUnitary` in :meth:`~.Hermitian.compute_diagonalizing_gates`.

    """

    # Qutrit case
    _num_basis_states = 3
    _eigs = {}

    # This method is overridden to update the docstring.
    @staticmethod
    def compute_matrix(A):
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.THermitian.matrix`

        Args:
            A (tensor_like): Hermitian matrix

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> A = np.array([[6+0j, 1-2j, 0],[1+2j, -1, 0], [0, 0, 1]])
        >>> qml.THermitian.compute_matrix(A)
        [[ 6.+0.j  1.-2.j  0.+0.j]
         [ 1.+2.j -1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  1.+0.j]]
        """
        return Hermitian.compute_matrix(A)

    @property
    def eigendecomposition(self):
        """Return the eigendecomposition of the matrix specified by the Hermitian observable.

        This method uses pre-stored eigenvalues for standard observables where
        possible and stores the corresponding eigenvectors from the eigendecomposition.

        It transforms the input operator according to the wires specified.

        Returns:
            dict[str, array]: dictionary containing the eigenvalues and the eigenvectors of the
                Hermitian observable
        """
        Hmat = self.matrix()
        Hmat = qml.math.to_numpy(Hmat)
        Hkey = tuple(Hmat.flatten().tolist())
        if Hkey not in THermitian._eigs:
            w, U = np.linalg.eigh(Hmat)
            THermitian._eigs[Hkey] = {"eigvec": U, "eigval": w}

        return THermitian._eigs[Hkey]

    @staticmethod
    def compute_diagonalizing_gates(eigenvectors, wires):  # pylint: disable=arguments-differ
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.THermitian.diagonalizing_gates`.

        Args:
            eigenvectors (array): eigenvectors of the operator, as extracted from op.eigendecomposition["eigvec"]
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> A = np.array([[-6, 2 + 1j, 0], [2 - 1j, 0, 0], [0, 0, 1]])
        >>> _, evecs = np.linalg.eigh(A)
        >>> qml.THermitian.compute_diagonalizing_gates(evecs, wires=[0])
        [QutritUnitary(tensor([[-0.94915323-0.j    0.1407893 +0.2815786j  -0.        -0.j  ]
                               [ 0.31481445-0.j    0.42447423+0.84894846j  0.        -0.j  ]
                               [ 0.        -0.j    0.        -0.j          1.        -0.j  ]], requires_grad=True), wires=[0])]

        """
        return [QutritUnitary(eigenvectors.conj().T, wires=wires)]


class GellMann(Observable):
    r"""
    The Gell-Mann observables for qutrits

    The Gell-Mann matrices are a set of 8 linearly independent :math:`3 \times 3` traceless, Hermitian matrices which
    naturally generalize the Pauli matrices from :math:`SU(2)` to :math:`SU(3)`.

    .. math::
        \displaystyle \begin{align} \lambda_{1} &= \left(\begin{array}{ccc} 0 & 1 & 0 \\ 1 & 0 & 0\\ 0 & 0 & 0\end{array}\right) \;\;\;\;\;\;\;\;\;\;
        \lambda_{2} = \left(\begin{array}{ccc} 0 & -i & 0 \\ i & 0 & 0\\ 0 & 0 & 0\end{array}\right)\;\;\;\;\;\;\;\;\;\;
        \lambda_{3} = \left(\begin{array}{ccc} 1 & 0 & 0 \\ 0 & -1 & 0\\ 0 & 0 & 0\end{array}\right) \\
        \lambda_{4} &= \left(\begin{array}{ccc} 0 & 0 & 1 \\ 0 & 0 & 0\\ 1 & 0 & 0\end{array}\right)\;\;\;\;\;\;\;\;\;\;
        \lambda_{5} = \left(\begin{array}{ccc} 0 & 0 & -i \\ 0 & 0 & 0\\ i & 0 & 0\end{array}\right) \\
        \lambda_{6} &= \left(\begin{array}{ccc} 0 & 0 & 0 \\ 0 & 0 & 1\\ 0 & 1 & 0\end{array}\right)\;\;\;\;\;\;\;\;\;\;
        \lambda_{7} = \left(\begin{array}{ccc} 0 & 0 & 0 \\ 0 & 0 & -i\\ 0 & i & 0\end{array}\right)\;\;\;\;\;\;\;\;\;\;
        \lambda_{8} = \frac{1}{\sqrt{3}}\left(\begin{array}{ccc} 1 & 0 & 0 \\ 0 & 1 & 0\\ 0 & 0 & -2\end{array}\right)\\ \end{align}

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        wires (Sequence[int] or int): the wire(s) the observable acts on
        index (int): The index of the Gell-Mann matrix to be used. Must be between 1
            and 8 inclusive
        id (str or None): String representing the operation (optional)

    **Example:**

    >>> dev = qml.device("default.qutrit", wires=2)
    >>> @qml.qnode(dev)
    ... def test_qnode():
    ...     qml.TShift(wires=0)
    ...     qml.TClock(wires=0)
    ...     qml.TShift(wires=1)
    ...     qml.TAdd(wires=[0, 1])
    ...     return qml.expval(qml.GellMann(wires=0, index=1))
    >>> print(test_qnode())
    0.0
    >>> print(qml.draw(test_qnode)())
    0: ──TShift──TClock─╭●────┤  <GellMann(1)>
    1: ──TShift─────────╰TAdd─┤

    """

    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters the operator depends on"""

    def __init__(self, wires, index=1, id=None):
        if not isinstance(index, int) or index < 1 or index > 8:
            raise ValueError(
                "The index of a Gell-Mann observable must be an integer between 1 and 8 inclusive."
            )

        self.hyperparameters["index"] = index

        super().__init__(wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or f"GellMann({self.hyperparameters['index']})"

    def __repr__(self):
        return f"GellMann{self.hyperparameters['index']}(wires=[{self.wires[0]}])"

    _eigvecs = {
        1: np.array(
            [[1 / np.sqrt(2), -1 / np.sqrt(2), 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 1]],
            dtype=np.complex128,
        ),
        2: np.array(
            [
                [-1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                [-1j / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 0, 1],
            ],
            dtype=np.complex128,
        ),
        4: np.array(
            [[1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 0, 1], [1 / np.sqrt(2), -1 / np.sqrt(2), 0]],
            dtype=np.complex128,
        ),
        5: np.array(
            [
                [1j / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 0, 1],
                [-1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            ],
            dtype=np.complex128,
        ),
        6: np.array(
            [[0, 0, 1], [1 / np.sqrt(2), -1 / np.sqrt(2), 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0]],
            dtype=np.complex128,
        ),
        7: np.array(
            [
                [0, 0, 1],
                [-1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                [-1j / np.sqrt(2), 1j / np.sqrt(2), 0],
            ],
            dtype=np.complex128,
        ),
    }

    @staticmethod
    def compute_matrix(index):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.GellMann.matrix`

        Args:
            index (int): The index of the Gell-Mann matrix to be used. Must be between 1
            and 8 inclusive

        Returns:
            tensor_like: canonical matrix

        **Example**

        >>> qml.GellMann.compute_matrix(8)
        array([[ 0.57735027+0.j,  0.        +0.j,  0.        +0.j],
               [ 0.        +0.j,  0.57735027+0.j,  0.        +0.j],
               [ 0.        +0.j,  0.        +0.j, -1.15470054+0.j]])
        """
        return gm_mats[index - 1]

    @staticmethod
    def compute_eigvals(index):  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U^{\dagger}`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.GellMann.eigvals`

        Args:
            index (int): The index of the Gell-Mann matrix to be used. Must be between 1
            and 8 inclusive

        Returns:
            array: eigenvalues

        **Example**

        >>> qml.GellMann.compute_eigvals(1)
        [1. -1.  0.]
        """
        if index != 8:
            return np.array([1, -1, 0])

        return np.array([1, 1, -2]) / np.sqrt(3)

    @staticmethod
    def compute_diagonalizing_gates(
        wires, index
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.GellMann.diagonalizing_gates`.

        Args:
            index (int): The index of the Gell-Mann matrix to be used. Must be between 1 and 8 inclusive
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> qml.GellMann.compute_diagonalizing_gates(wires=0, index=4)
        [QutritUnitary(array([[ 0.70710678-0.j,  0.        -0.j,  0.70710678-0.j],
               [ 0.70710678-0.j,  0.        -0.j, -0.70710678-0.j],
               [ 0.        -0.j,  1.        -0.j,  0.        -0.j]]), wires=[0])]
        """
        if index in (3, 8):
            return []  # already diagonal

        v = GellMann._eigvecs[index]
        return [QutritUnitary(v.conj().T, wires=wires)]
