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
This submodule contains the discrete-variable quantum operations that do
not depend on any parameters.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import cmath
import warnings
import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.operation import AnyWires, Observable, Operation
from pennylane.utils import pauli_eigs
from pennylane.wires import Wires

INV_SQRT2 = 1 / qml.math.sqrt(2)


class Hadamard(Observable, Operation):
    r"""Hadamard(wires)
    The Hadamard operator

    .. math:: H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1\\ 1 & -1\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "H"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.Hadamard.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Hadamard.compute_matrix())
        [[ 0.70710678  0.70710678]
         [ 0.70710678 -0.70710678]]
        """
        return np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.Hadamard.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.Hadamard.compute_eigvals())
        [ 1 -1]
        """
        return pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.Hadamard.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.Hadamard.compute_diagonalizing_gates(wires=[0]))
        [RY(-0.7853981633974483, wires=[0])]
        """
        return [qml.RY(-np.pi / 4, wires=wires)]

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.Hadamard.decomposition`.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> print(qml.Hadamard.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0]),
        RX(1.5707963267948966, wires=[0]),
        PhaseShift(1.5707963267948966, wires=[0])]

        """
        decomp_ops = [
            qml.PhaseShift(np.pi / 2, wires=wires),
            qml.RX(np.pi / 2, wires=wires),
            qml.PhaseShift(np.pi / 2, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return Hadamard(wires=self.wires)

    def single_qubit_rot_angles(self):
        # H = RZ(\pi) RY(\pi/2) RZ(0)
        return [np.pi, np.pi / 2, 0.0]


class PauliX(Observable, Operation):
    r"""PauliX(wires)
    The Pauli X operator

    .. math:: \sigma_x = \begin{bmatrix} 0 & 1 \\ 1 & 0\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "X"

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "X"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PauliX.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.PauliX.compute_matrix())
        [[0 1]
         [1 0]]
        """
        return np.array([[0, 1], [1, 0]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PauliX.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.PauliX.compute_eigvals())
        [ 1 -1]
        """
        return pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.PauliX.diagonalizing_gates`.

        Args:
           wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
           list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.PauliX.compute_diagonalizing_gates(wires=[0]))
        [Hadamard(wires=[0])]
        """
        return [Hadamard(wires=wires)]

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.PauliX.decomposition`.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.PauliX.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0]),
        RX(3.141592653589793, wires=[0]),
        PhaseShift(1.5707963267948966, wires=[0])]

        """
        decomp_ops = [
            qml.PhaseShift(np.pi / 2, wires=wires),
            qml.RX(np.pi, wires=wires),
            qml.PhaseShift(np.pi / 2, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return PauliX(wires=self.wires)

    def _controlled(self, wire):
        CNOT(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self):
        # X = RZ(-\pi/2) RY(\pi) RZ(\pi/2)
        return [np.pi / 2, np.pi, -np.pi / 2]


class PauliY(Observable, Operation):
    r"""PauliY(wires)
    The Pauli Y operator

    .. math:: \sigma_y = \begin{bmatrix} 0 & -i \\ i & 0\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    """int: Number of wires that the operator acts on."""

    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Y"

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "Y"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PauliY.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.PauliY.compute_matrix())
        [[ 0.+0.j -0.-1.j]
         [ 0.+1.j  0.+0.j]]
        """
        return np.array([[0, -1j], [1j, 0]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PauliY.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.PauliY.compute_eigvals())
        [ 1 -1]
        """
        return pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires):
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.PauliY.diagonalizing_gates`.

        Args:
            wires (Iterable[Any], Wires): wires that the operator acts on
        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.PauliY.compute_diagonalizing_gates(wires=[0]))
        [PauliZ(wires=[0]), S(wires=[0]), Hadamard(wires=[0])]
        """
        return [
            PauliZ(wires=wires),
            S(wires=wires),
            Hadamard(wires=wires),
        ]

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.PauliY.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.PauliY.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0]),
        RY(3.141592653589793, wires=[0]),
        PhaseShift(1.5707963267948966, wires=[0])]

        """
        decomp_ops = [
            qml.PhaseShift(np.pi / 2, wires=wires),
            qml.RY(np.pi, wires=wires),
            qml.PhaseShift(np.pi / 2, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return PauliY(wires=self.wires)

    def _controlled(self, wire):
        CY(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self):
        # Y = RZ(0) RY(\pi) RZ(0)
        return [0.0, np.pi, 0.0]


class PauliZ(Observable, Operation):
    r"""PauliZ(wires)
    The Pauli Z operator

    .. math:: \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "Z"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.PauliZ.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.PauliZ.compute_matrix())
        [[ 1  0]
         [ 0 -1]]
        """
        return np.array([[1, 0], [0, -1]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.PauliZ.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.PauliZ.compute_eigvals())
        [ 1 -1]
        """
        return pauli_eigs(1)

    @staticmethod
    def compute_diagonalizing_gates(wires):  # pylint: disable=unused-argument
        r"""Sequence of gates that diagonalize the operator in the computational basis (static method).

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        .. seealso:: :meth:`~.PauliZ.diagonalizing_gates`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operator acts on

        Returns:
            list[.Operator]: list of diagonalizing gates

        **Example**

        >>> print(qml.PauliZ.compute_diagonalizing_gates(wires=[0]))
        []
        """
        return []

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.

        .. seealso:: :meth:`~.PauliZ.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.PauliZ.compute_decomposition(0))
        [PhaseShift(3.141592653589793, wires=[0])]

        """
        return [qml.PhaseShift(np.pi, wires=wires)]

    def adjoint(self):
        return PauliZ(wires=self.wires)

    def _controlled(self, wire):
        CZ(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self):
        # Z = RZ(\pi) RY(0) RZ(0)
        return [np.pi, 0.0, 0.0]


class S(Operation):
    r"""S(wires)
    The single-qubit phase gate

    .. math:: S = \begin{bmatrix}
                1 & 0 \\
                0 & i
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.S.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.S.compute_matrix())
        [[1.+0.j 0.+0.j]
         [0.+0.j 0.+1.j]]
        """
        return np.array([[1, 0], [0, 1j]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.S.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.S.compute_eigvals())
        [1.+0.j 0.+1.j]
        """
        return np.array([1, 1j])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.S.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.S.compute_decomposition(0))
        [PhaseShift(1.5707963267948966, wires=[0])]

        """
        return [qml.PhaseShift(np.pi / 2, wires=wires)]

    def adjoint(self):
        op = S(wires=self.wires)
        op.inverse = not self.inverse
        return op

    def single_qubit_rot_angles(self):
        # S = RZ(\pi/2) RY(0) RZ(0)
        return [np.pi / 2, 0.0, 0.0]


class T(Operation):
    r"""T(wires)
    The single-qubit T gate

    .. math:: T = \begin{bmatrix}
                1 & 0 \\
                0 & e^{\frac{i\pi}{4}}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "Z"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.T.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.T.compute_matrix())
        [[1.+0.j         0.        +0.j        ]
         [0.+0.j         0.70710678+0.70710678j]]
        """
        return np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.T.eigvals`

        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.T.compute_eigvals())
        [1.+0.j 0.70710678+0.70710678j]
        """
        return np.array([1, cmath.exp(1j * np.pi / 4)])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.T.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.T.compute_decomposition(0))
        [PhaseShift(0.7853981633974483, wires=[0])]

        """
        return [qml.PhaseShift(np.pi / 4, wires=wires)]

    def adjoint(self):
        op = T(wires=self.wires)
        op.inverse = not self.inverse
        return op

    def single_qubit_rot_angles(self):
        # T = RZ(\pi/4) RY(0) RZ(0)
        return [np.pi / 4, 0.0, 0.0]


class SX(Operation):
    r"""SX(wires)
    The single-qubit Square-Root X operator.

    .. math:: SX = \sqrt{X} = \frac{1}{2} \begin{bmatrix}
            1+i &   1-i \\
            1-i &   1+i \\
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_wires = 1
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    basis = "X"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SX.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.SX.compute_matrix())
        [[0.5+0.5j 0.5-0.5j]
         [0.5-0.5j 0.5+0.5j]]
        """
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.SX.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.SX.compute_eigvals())
        [1.+0.j 0.+1.j]
        """
        return np.array([1, 1j])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SX.decomposition`.

        Args:
            wires (Any, Wires): Single wire that the operator acts on.

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SX.compute_decomposition(0))
        [RZ(1.5707963267948966, wires=[0]),
        RY(1.5707963267948966, wires=[0]),
        RZ(-3.141592653589793, wires=[0]),
        PhaseShift(1.5707963267948966, wires=[0])]

        """
        decomp_ops = [
            qml.RZ(np.pi / 2, wires=wires),
            qml.RY(np.pi / 2, wires=wires),
            qml.RZ(-np.pi, wires=wires),
            qml.PhaseShift(np.pi / 2, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        op = SX(wires=self.wires)
        op.inverse = not self.inverse
        return op

    def single_qubit_rot_angles(self):
        # SX = RZ(-\pi/2) RY(\pi/2) RZ(\pi/2)
        return [np.pi / 2, np.pi / 2, -np.pi / 2]


class SWAP(Operation):
    r"""SWAP(wires)
    The swap operator

    .. math:: SWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.SWAP.compute_matrix())
        [[1 0 0 0]
         [0 0 1 0]
         [0 1 0 0]
         [0 0 0 1]]
        """
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SWAP.compute_decomposition((0,1)))
        [CNOT(wires=[0, 1]), CNOT(wires=[1, 0]), CNOT(wires=[0, 1])]

        """
        decomp_ops = [
            qml.CNOT(wires=[wires[0], wires[1]]),
            qml.CNOT(wires=[wires[1], wires[0]]),
            qml.CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        return SWAP(wires=self.wires)

    def _controlled(self, wire):
        CSWAP(wires=wire + self.wires)


class ISWAP(Operation):
    r"""ISWAP(wires)
    The i-swap operator

    .. math:: ISWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & i & 0\\
            0 & i & 0 & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ISWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.ISWAP.compute_matrix())
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+1.j 0.+0.j]
         [0.+0.j 0.+1.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
        """
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ISWAP.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.ISWAP.compute_eigvals())
        [1j, -1j, 1, 1]
        """
        return np.array([1j, -1j, 1, 1])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.ISWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.ISWAP.compute_decomposition((0,1)))
        [S(wires=[0]),
        S(wires=[1]),
        Hadamard(wires=[0]),
        CNOT(wires=[0, 1]),
        CNOT(wires=[1, 0]),
        Hadamard(wires=[1])]

        """
        decomp_ops = [
            S(wires=wires[0]),
            S(wires=wires[1]),
            Hadamard(wires=wires[0]),
            CNOT(wires=[wires[0], wires[1]]),
            CNOT(wires=[wires[1], wires[0]]),
            Hadamard(wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        op = ISWAP(wires=self.wires)
        op.inverse = not self.inverse
        return op


class SISWAP(Operation):
    r"""SISWAP(wires)
    The square root of i-swap operator. Can also be accessed as ``qml.SQISW``

    .. math:: SISWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1/ \sqrt{2} & i/\sqrt{2} & 0\\
            0 & i/ \sqrt{2} & 1/ \sqrt{2} & 0\\
            0 & 0 & 0 & 1
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.SISWAP.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.SISWAP.compute_matrix())
        [[1.+0.j          0.+0.j          0.+0.j  0.+0.j]
         [0.+0.j  0.70710678+0.j  0.+0.70710678j  0.+0.j]
         [0.+0.j  0.+0.70710678j  0.70710678+0.j  0.+0.j]
         [0.+0.j          0.+0.j          0.+0.j  1.+0.j]]
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, INV_SQRT2, INV_SQRT2 * 1j, 0],
                [0, INV_SQRT2 * 1j, INV_SQRT2, 0],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.SISWAP.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.SISWAP.compute_eigvals())
        [0.70710678+0.70710678j 0.70710678-0.70710678j 1.+0.j 1.+0.j]
        """
        return np.array([INV_SQRT2 * (1 + 1j), INV_SQRT2 * (1 - 1j), 1, 1])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.SISWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.SISWAP.compute_decomposition((0,1)))
        [SX(wires=[0]),
        RZ(1.5707963267948966, wires=[0]),
        CNOT(wires=[0, 1]),
        SX(wires=[0]),
        RZ(5.497787143782138, wires=[0]),
        SX(wires=[0]),
        RZ(1.5707963267948966, wires=[0]),
        SX(wires=[1]),
        RZ(5.497787143782138, wires=[1]),
        CNOT(wires=[0, 1]),
        SX(wires=[0]),
        SX(wires=[1])]

        """
        decomp_ops = [
            SX(wires=wires[0]),
            qml.RZ(np.pi / 2, wires=wires[0]),
            CNOT(wires=[wires[0], wires[1]]),
            SX(wires=wires[0]),
            qml.RZ(7 * np.pi / 4, wires=wires[0]),
            SX(wires=wires[0]),
            qml.RZ(np.pi / 2, wires=wires[0]),
            SX(wires=wires[1]),
            qml.RZ(7 * np.pi / 4, wires=wires[1]),
            CNOT(wires=[wires[0], wires[1]]),
            SX(wires=wires[0]),
            SX(wires=wires[1]),
        ]
        return decomp_ops

    def adjoint(self):
        op = SISWAP(wires=self.wires)
        op.inverse = not self.inverse
        return op


SQISW = SISWAP


class Barrier(Operation):
    r"""Barrier(wires)
    The Barrier operator, used to separate the compilation process into blocks or as a visual tool.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        only_visual (bool): True if we do not want it to have an impact on the compilation process. Default is False.
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    num_wires = AnyWires
    par_domain = None

    def __init__(self, only_visual=False, wires=Wires([]), do_queue=True, id=None):
        self.only_visual = only_visual
        self.hyperparameters["only_visual"] = only_visual
        super().__init__(wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(wires, only_visual=False):  # pylint: disable=unused-argument
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Barrier.decomposition`.

        ``Barrier`` decomposes into an empty list for all arguments.

        Args:
            wires (Iterable, Wires): wires that the operator acts on
            only_visual (Bool): True if we do not want it to have an impact on the compilation process. Default is False.

        Returns:
            list: decomposition of the operator

        **Example:**

        >>> print(qml.Barrier.compute_decomposition(0))
        []

        """
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return "||"

    def _controlled(self, _):
        return Barrier(wires=self.wires)

    def adjoint(self, do_queue=False):
        return Barrier(wires=self.wires)


class WireCut(Operation):
    r"""WireCut(wires)
    The wire cut operation, used to manually mark locations for wire cuts.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    **Details:**

    * Number of wires: AnyWires
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    num_wires = AnyWires
    grad_method = None

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        Since this operator is a placeholder inside a circuit, it decomposes into an empty list.

        Args:
            wires (Any, Wires): Wire that the operator acts on.

        Returns:
            list[Operator]: decomposition of the operator

        **Example:**

        >>> print(qml.WireCut.compute_decomposition(0))
        []

        """
        return []

    def label(self, decimals=None, base_label=None, cache=None):
        return "//"

    def adjoint(self):
        return WireCut(wires=self.wires)
