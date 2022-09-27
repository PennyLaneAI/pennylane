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
# pylint:disable=abstract-method,arguments-differ,protected-access,invalid-overridden-method, no-member
import cmath
import warnings
from copy import copy

import numpy as np

from scipy import sparse
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

    _queue_category = "_ops"

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
    def compute_sparse_matrix(*params, **hyperparams):
        return sparse.csr_matrix([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

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
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

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

    def pow(self, z):
        return super().pow(z % 2)


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

    _queue_category = "_ops"

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
    def compute_sparse_matrix(*params, **hyperparams):
        return sparse.csr_matrix([[0, 1], [1, 0]])

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
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

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

    def pow(self, z):
        z_mod2 = z % 2
        if abs(z_mod2 - 0.5) < 1e-6:
            return [SX(wires=self.wires)]
        return super().pow(z_mod2)

    def _controlled(self, wire):
        return CNOT(wires=Wires(wire) + self.wires)

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

    _queue_category = "_ops"

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
    def compute_sparse_matrix(*params, **hyperparams):
        return sparse.csr_matrix([[0, -1j], [1j, 0]])

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
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

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

    def pow(self, z):
        return super().pow(z % 2)

    def _controlled(self, wire):
        return CY(wires=Wires(wire) + self.wires)

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

    _queue_category = "_ops"

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
    def compute_sparse_matrix(*params, **hyperparams):
        return sparse.csr_matrix([[1, 0], [0, -1]])

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
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

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

    def pow(self, z):
        z_mod2 = z % 2
        if z_mod2 == 0:
            return []
        if z_mod2 == 1:
            return [copy(self)]

        if abs(z_mod2 - 0.5) < 1e-6:
            return [S(wires=self.wires)]
        if abs(z_mod2 - 0.25) < 1e-6:
            return [T(wires=self.wires)]

        return [qml.PhaseShift(np.pi * z_mod2, wires=self.wires)]

    def _controlled(self, wire):
        return CZ(wires=wire + self.wires)

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

    def pow(self, z):
        z_mod4 = z % 4
        pow_map = {
            0: lambda op: [],
            0.5: lambda op: [T(wires=op.wires)],
            1: lambda op: [copy(op)],
            2: lambda op: [PauliZ(wires=op.wires)],
        }
        return pow_map.get(z_mod4, lambda op: [qml.PhaseShift(np.pi * z_mod4 / 2, wires=op.wires)])(
            self
        )

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

    def pow(self, z):
        z_mod8 = z % 8
        pow_map = {
            0: lambda op: [],
            1: lambda op: [copy(op)],
            2: lambda op: [S(wires=op.wires)],
            4: lambda op: [PauliZ(wires=op.wires)],
        }
        return pow_map.get(z_mod8, lambda op: [qml.PhaseShift(np.pi * z_mod8 / 4, wires=op.wires)])(
            self
        )

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

    def pow(self, z):
        z_mod4 = z % 4
        if z_mod4 == 2:
            return [PauliX(wires=self.wires)]
        return super().pow(z_mod4)

    def single_qubit_rot_angles(self):
        # SX = RZ(-\pi/2) RY(\pi/2) RZ(\pi/2)
        return [np.pi / 2, np.pi / 2, -np.pi / 2]


class CNOT(Operation):
    r"""CNOT(wires)
    The controlled-NOT operator

    .. math:: CNOT = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & 1\\
            0 & 0 & 1 & 0
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
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

        .. seealso:: :meth:`~.CNOT.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CNOT.compute_matrix())
        [[1 0 0 0]
         [0 1 0 0]
         [0 0 0 1]
         [0 0 1 0]]
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    def adjoint(self):
        return CNOT(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)

    def _controlled(self, wire):
        return Toffoli(wires=wire + self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])

    @property
    def is_hermitian(self):
        return True


class CZ(Operation):
    r"""CZ(wires)
    The controlled-Z operator

    .. math:: CZ = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 1 & 0\\
            0 & 0 & 0 & -1
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
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

        .. seealso:: :meth:`~.CZ.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CZ.compute_matrix())
        [[ 1  0  0  0]
         [ 0  1  0  0]
         [ 0  0  1  0]
         [ 0  0  0 -1]]
        """
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    @staticmethod
    def compute_eigvals():  # pylint: disable=arguments-differ
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.CZ.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.CZ.compute_eigvals())
        [1, 1, 1, -1]
        """
        return np.array([1, 1, 1, -1])

    def adjoint(self):
        return CZ(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)

    @property
    def control_wires(self):
        return Wires(self.wires[0])

    @property
    def is_hermitian(self):
        return True


class CY(Operation):
    r"""CY(wires)
    The controlled-Y operator

    .. math:: CY = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & -i\\
            0 & 0 & i & 0
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    num_wires = 2
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

        .. seealso:: :meth:`~.CY.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CY.compute_matrix())
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -0.-1.j]
         [ 0.+0.j  0.+0.j  0.+1.j  0.+0.j]]
        """
        return np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ]
        )

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).


        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CY.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CY.compute_decomposition(0))
        [CRY(3.141592653589793, wires=[0, 1]), S(wires=[0])]

        """
        return [qml.CRY(np.pi, wires=wires), S(wires=wires[0])]

    def adjoint(self):
        return CY(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)

    @property
    def control_wires(self):
        return Wires(self.wires[0])

    @property
    def is_hermitian(self):
        return True


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

    def pow(self, z):
        return super().pow(z % 2)

    def adjoint(self):
        return SWAP(wires=self.wires)

    def _controlled(self, wire):
        return CSWAP(wires=wire + self.wires)

    @property
    def is_hermitian(self):
        return True


class ECR(Operation):
    r""" ECR(wires)

    An echoed RZX(pi/2) gate.

    .. math:: ECR = {1/\sqrt{2}} \begin{bmatrix}
            0 & 0 & 1 & i \\
            0 & 0 & i & 1 \\
            1 & -i & 0 & 0 \\
            -i & 1 & 0 & 0
        \end{bmatrix}.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 0

    Args:
        wires (int): the subsystem the gate acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional)
        id (str or None): String representing the operation (optional)
    """

    num_wires = 2
    num_params = 0

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.ECR.matrix`


        Return type: tensor_like

        **Example**

        >>> print(qml.ECR.compute_matrix())
         [[0+0.j 0.+0.j 1/sqrt(2)+0.j 0.+1j/sqrt(2)]
         [0.+0.j 0.+0.j 0.+1.j/sqrt(2) 1/sqrt(2)+0.j]
         [1/sqrt(2)+0.j 0.-1.j/sqrt(2) 0.+0.j 0.+0.j]
         [0.-1/sqrt(2)j 1/sqrt(2)+0.j 0.+0.j 0.+0.j]]
        """

        return np.array(
            [
                [0, 0, INV_SQRT2, INV_SQRT2 * 1j],
                [0, 0, INV_SQRT2 * 1j, INV_SQRT2],
                [INV_SQRT2, -INV_SQRT2 * 1j, 0, 0],
                [-INV_SQRT2 * 1j, INV_SQRT2, 0, 0],
            ]
        )

    @staticmethod
    def compute_eigvals():
        r"""Eigenvalues of the operator in the computational basis (static method).

        If :attr:`diagonalizing_gates` are specified and implement a unitary :math:`U`,
        the operator can be reconstructed as

        .. math:: O = U \Sigma U^{\dagger},

        where :math:`\Sigma` is the diagonal matrix containing the eigenvalues.

        Otherwise, no particular order for the eigenvalues is guaranteed.

        .. seealso:: :meth:`~.ECR.eigvals`


        Returns:
            array: eigenvalues

        **Example**

        >>> print(qml.ECR.compute_eigvals())
        [1, -1, 1, -1]
        """

        return np.array([1, -1, 1, -1])

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

           .. math:: O = O_1 O_2 \dots O_n.


           .. seealso:: :meth:`~.ECR.decomposition`.

           Args:
               wires (Iterable, Wires): wires that the operator acts on

           Returns:
               list[Operator]: decomposition into lower level operations

           **Example:**

           >>> print(qml.ECR.compute_decomposition((0,1)))


        [PauliZ(wires=[0]),
         CNOT(wires=[0, 1]),
         SX(wires=[1]),
         RX(1.5707963267948966, wires=[0]),
         RY(1.5707963267948966, wires=[0]),
         RX(1.5707963267948966, wires=[0])]

        """
        pi = np.pi
        return [
            PauliZ(wires=[wires[0]]),
            CNOT(wires=[wires[0], wires[1]]),
            SX(wires=[wires[1]]),
            qml.RX(pi / 2, wires=[wires[0]]),
            qml.RY(pi / 2, wires=[wires[0]]),
            qml.RX(pi / 2, wires=[wires[0]]),
        ]

    def adjoint(self):
        return ECR(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)


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

    def pow(self, z):
        z_mod2 = z % 2
        if abs(z_mod2 - 0.5) < 1e-6:
            return [SISWAP(wires=self.wires)]
        return super().pow(z_mod2)


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

    def pow(self, z):
        z_mod4 = z % 4
        return [ISWAP(wires=self.wires)] if z_mod4 == 2 else super().pow(z_mod4)


SQISW = SISWAP


class CSWAP(Operation):
    r"""CSWAP(wires)
    The controlled-swap operator

    .. math:: CSWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the wires the operation acts on
    """
    is_self_inverse = True
    num_wires = 3
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "SWAP"

    @staticmethod
    def compute_matrix():  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.CSWAP.matrix`

        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.CSWAP.compute_matrix())
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 0 1 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 0 1]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.CSWAP.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.CSWAP.compute_decomposition((0,1,2)))
        [Toffoli(wires=[0, 2, 1]), Toffoli(wires=[0, 1, 2]), Toffoli(wires=[0, 2, 1])]

        """
        decomp_ops = [
            qml.Toffoli(wires=[wires[0], wires[2], wires[1]]),
            qml.Toffoli(wires=[wires[0], wires[1], wires[2]]),
            qml.Toffoli(wires=[wires[0], wires[2], wires[1]]),
        ]
        return decomp_ops

    def pow(self, z):
        return super().pow(z % 2)

    def adjoint(self):
        return CSWAP(wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])

    @property
    def is_hermitian(self):
        return True


class Toffoli(Operation):
    r"""Toffoli(wires)
    Toffoli (controlled-controlled-X) gate.

    .. math::

        Toffoli =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
        \end{pmatrix}

    **Details:**

    * Number of wires: 3
    * Number of parameters: 0

    Args:
        wires (Sequence[int]): the subsystem the gate acts on
    """
    num_wires = 3
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

        .. seealso:: :meth:`~.Toffoli.matrix`


        Returns:
            ndarray: matrix

        **Example**

        >>> print(qml.Toffoli.compute_matrix())
        [[1 0 0 0 0 0 0 0]
         [0 1 0 0 0 0 0 0]
         [0 0 1 0 0 0 0 0]
         [0 0 0 1 0 0 0 0]
         [0 0 0 0 1 0 0 0]
         [0 0 0 0 0 1 0 0]
         [0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 0]]
        """
        return np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

    @staticmethod
    def compute_decomposition(wires):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.Toffoli.decomposition`.

        Args:
            wires (Iterable, Wires): wires that the operator acts on

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> qml.Toffoli.compute_decomposition((0,1,2))
        [Hadamard(wires=[2]),
        CNOT(wires=[1, 2]),
        Adjoint(T)(wires=[2]),
        CNOT(wires=[0, 2]),
        T(wires=[2]),
        CNOT(wires=[1, 2]),
        Adjoint(T)(wires=[2]),
        CNOT(wires=[0, 2]),
        T(wires=[2]),
        T(wires=[1]),
        CNOT(wires=[0, 1]),
        Hadamard(wires=[2]),
        T(wires=[0]),
        Adjoint(T)(wires=[1]),
        CNOT(wires=[0, 1])]

        """
        return [
            Hadamard(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(T(wires=wires[2])),
            CNOT(wires=[wires[0], wires[2]]),
            T(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            qml.adjoint(T(wires=wires[2])),
            CNOT(wires=[wires[0], wires[2]]),
            T(wires=wires[2]),
            T(wires=wires[1]),
            CNOT(wires=[wires[0], wires[1]]),
            Hadamard(wires=wires[2]),
            T(wires=wires[0]),
            qml.adjoint(T(wires=wires[1])),
            CNOT(wires=[wires[0], wires[1]]),
        ]

    def adjoint(self):
        return Toffoli(wires=self.wires)

    def pow(self, z):
        return super().pow(z % 2)

    @property
    def control_wires(self):
        return Wires(self.wires[:2])

    @property
    def is_hermitian(self):
        return True


class MultiControlledX(Operation):
    r"""MultiControlledX(control_wires, wires, control_values)
    Apply a Pauli X gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], or int]): Deprecated way to indicate the control wires.
            Now users should use "wires" to indicate both the control wires and the target wire.
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire where
            the operation acts on
        control_values (str): a string of bits representing the state of the control
            wires to control on (default is the all 1s state)
        work_wires (Union[Wires, Sequence[int], or int]): optional work wires used to decompose
            the operation into a series of Toffoli gates


    .. note::

        If ``MultiControlledX`` is not supported on the targeted device, PennyLane will decompose
        the operation into :class:`~.Toffoli` and/or :class:`~.CNOT` gates. When controlling on
        three or more wires, the Toffoli-based decompositions described in Lemmas 7.2 and 7.3 of
        `Barenco et al. <https://arxiv.org/abs/quant-ph/9503016>`__ will be used. These methods
        require at least one work wire.

        The number of work wires provided determines the decomposition method used and the resulting
        number of Toffoli gates required. When ``MultiControlledX`` is controlling on :math:`n`
        wires:

        #. If at least :math:`n - 2` work wires are provided, the decomposition in Lemma 7.2 will be
           applied using the first :math:`n - 2` work wires.
        #. If fewer than :math:`n - 2` work wires are provided, a combination of Lemmas 7.3 and 7.2
           will be applied using only the first work wire.

        These methods present a tradeoff between qubit number and depth. The method in point 1
        requires fewer Toffoli gates but a greater number of qubits.

        Note that the state of the work wires before and after the decomposition takes place is
        unchanged.

    """
    is_self_inverse = True
    num_wires = AnyWires
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        control_wires=None,
        wires=None,
        control_values=None,
        work_wires=None,
        do_queue=True,
    ):
        if wires is None:
            raise ValueError("Must specify the wires where the operation acts on")
        if control_wires is None:
            if len(wires) > 1:
                control_wires = Wires(wires[:-1])
                wires = Wires(wires[-1])
            else:
                raise ValueError(
                    "MultiControlledX: wrong number of wires. "
                    f"{len(wires)} wire(s) given. Need at least 2."
                )
        else:
            wires = Wires(wires)
            control_wires = Wires(control_wires)

            warnings.warn(
                "The control_wires keyword will be removed soon. "
                "Use wires = (control_wires, target_wire) instead. "
                "See the documentation for more information.",
                category=UserWarning,
            )

            if len(wires) != 1:
                raise ValueError("MultiControlledX accepts a single target wire.")

        work_wires = Wires([]) if work_wires is None else Wires(work_wires)
        total_wires = control_wires + wires

        if Wires.shared_wires([total_wires, work_wires]):
            raise ValueError("The work wires must be different from the control and target wires")

        if not control_values:
            control_values = "1" * len(control_wires)

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["work_wires"] = work_wires
        self.hyperparameters["control_values"] = control_values

        super().__init__(wires=total_wires, do_queue=do_queue)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "X"

    # pylint: disable=unused-argument
    @staticmethod
    def compute_matrix(
        control_wires, control_values=None, **kwargs
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.MultiControlledX.matrix`

        Args:
            control_wires (Any or Iterable[Any]): wires to place controls on
            control_values (str): string of bits determining the controls

        Returns:
           tensor_like: matrix representation

        **Example**

        >>> print(qml.MultiControlledX.compute_matrix([0], '1'))
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 0. 1.]
         [0. 0. 1. 0.]]
        >>> print(qml.MultiControlledX.compute_matrix([1], '0'))
        [[0. 1. 0. 0.]
         [1. 0. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]]

        """
        if control_values is None:
            control_values = "1" * len(control_wires)

        if isinstance(control_values, str):
            if len(control_values) != len(control_wires):
                raise ValueError("Length of control bit string must equal number of control wires.")

            # Make sure all values are either 0 or 1
            if not set(control_values).issubset({"1", "0"}):
                raise ValueError("String of control values can contain only '0' or '1'.")

            control_int = int(control_values, 2)
        else:
            raise ValueError("Control values must be passed as a string.")

        padding_left = control_int * 2
        padding_right = 2 ** (len(control_wires) + 1) - 2 - padding_left
        cx = block_diag(np.eye(padding_left), PauliX.compute_matrix(), np.eye(padding_right))
        return cx

    @property
    def control_wires(self):
        return self.wires[:~0]

    def adjoint(self):
        return MultiControlledX(
            wires=self.wires,
            control_values=self.hyperparameters["control_values"],
        )

    def pow(self, z):
        return super().pow(z % 2)

    @staticmethod
    def compute_decomposition(wires=None, work_wires=None, control_values=None, **kwargs):
        r"""Representation of the operator as a product of other operators (static method).

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.MultiControlledX.decomposition`.

        Args:
            wires (Iterable[Any] or Wires): wires that the operation acts on
            work_wires (Wires): optional work wires used to decompose
                the operation into a series of Toffoli gates.
            control_values (str): a string of bits representing the state of the control
                wires to control on (default is the all 1s state)

        Returns:
            list[Operator]: decomposition into lower level operations

        **Example:**

        >>> print(qml.MultiControlledX.compute_decomposition(wires=[0,1,2,3],control_values="111", work_wires=qml.wires.Wires("aux")))
        [Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux']),
        Toffoli(wires=[2, 'aux', 3]),
        Toffoli(wires=[0, 1, 'aux'])]

        """

        target_wire = wires[~0]
        control_wires = wires[:~0]

        if control_values is None:
            control_values = "1" * len(control_wires)

        if len(control_wires) > 2 and len(work_wires) == 0:
            raise ValueError(
                "At least one work wire is required to decompose operation: MultiControlledX"
            )

        flips1 = [
            qml.PauliX(control_wires[i]) for i, val in enumerate(control_values) if val == "0"
        ]

        if len(control_wires) == 1:
            decomp = [qml.CNOT(wires=[control_wires[0], target_wire])]
        elif len(control_wires) == 2:

            decomp = [qml.Toffoli(wires=[*control_wires, target_wire])]
        else:
            num_work_wires_needed = len(control_wires) - 2

            if len(work_wires) >= num_work_wires_needed:
                decomp = MultiControlledX._decomposition_with_many_workers(
                    control_wires, target_wire, work_wires
                )
            else:
                work_wire = work_wires[0]
                decomp = MultiControlledX._decomposition_with_one_worker(
                    control_wires, target_wire, work_wire
                )

        flips2 = [
            qml.PauliX(control_wires[i]) for i, val in enumerate(control_values) if val == "0"
        ]

        return flips1 + decomp + flips2

    @staticmethod
    def _decomposition_with_many_workers(control_wires, target_wire, work_wires):
        """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
        https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
        work wires"""
        num_work_wires_needed = len(control_wires) - 2
        work_wires = work_wires[:num_work_wires_needed]

        work_wires_reversed = list(reversed(work_wires))
        control_wires_reversed = list(reversed(control_wires))

        gates = []

        for i in range(len(work_wires)):
            ctrl1 = control_wires_reversed[i]
            ctrl2 = work_wires_reversed[i]
            t = target_wire if i == 0 else work_wires_reversed[i - 1]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

        for i in reversed(range(len(work_wires))):
            ctrl1 = control_wires_reversed[i]
            ctrl2 = work_wires_reversed[i]
            t = target_wire if i == 0 else work_wires_reversed[i - 1]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        for i in range(len(work_wires) - 1):
            ctrl1 = control_wires_reversed[i + 1]
            ctrl2 = work_wires_reversed[i + 1]
            t = work_wires_reversed[i]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

        for i in reversed(range(len(work_wires) - 1)):
            ctrl1 = control_wires_reversed[i + 1]
            ctrl2 = work_wires_reversed[i + 1]
            t = work_wires_reversed[i]
            gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

        return gates

    @staticmethod
    def _decomposition_with_one_worker(control_wires, target_wire, work_wire):
        """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
        https://arxiv.org/abs/quant-ph/9503016, which requires a single work wire"""
        tot_wires = len(control_wires) + 2
        partition = int(np.ceil(tot_wires / 2))

        first_part = control_wires[:partition]
        second_part = control_wires[partition:]

        gates = [
            MultiControlledX(
                wires=first_part + work_wire,
                work_wires=second_part + target_wire,
            ),
            MultiControlledX(
                wires=second_part + work_wire + target_wire,
                work_wires=first_part,
            ),
            MultiControlledX(
                wires=first_part + work_wire,
                work_wires=second_part + target_wire,
            ),
            MultiControlledX(
                wires=second_part + work_wire + target_wire,
                work_wires=first_part,
            ),
        ]

        return gates

    @property
    def is_hermitian(self):
        return True


class IntegerComparator(Operation):
    r"""IntegerComparator(value, geq, control_wires, wires)
    Apply a Pauli X gate controlled on the following condition:

    Given a basis state :math:`\vert \sigma \rangle`, where :math:`\sigma \in \mathbb{N}`, and a fixed positive
    integer :math:`L`, flip a target qubit if :math:`\sigma \geq L`. Alternatively, the flipping condition can
    be :math:`\sigma < L`.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        value (int): The value :math:`L` that the state's decimal representation is compared against.
        geq (bool): If set to `True`, the comparison made will be :math:`\sigma \geq L`. If `False`, the comparison
            made will be :math:`\sigma < L`.
        control_wires (Union[Wires, Sequence[int], or int]): Deprecated way to indicate the control wires.
            Now users should use "wires" to indicate both the control wires and the target wire.
        wires (Union[Wires, Sequence[int], or int]): control wire(s) followed by a single target wire where
            the operation acts on.
    """
    is_self_inverse = True
    num_wires = AnyWires
    num_params = 0
    """int: Number of trainable parameters that the operator depends on."""

    grad_method = None
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        value=None,
        geq=True,
        control_wires=None,
        wires=None,
        do_queue=True,
    ):

        if value is None:
            raise ValueError("Must specify the integer value to compare against.")
        if not isinstance(value, int):
            raise ValueError("The comparable value must be an integer.")
        if wires is None:
            raise ValueError("Must specify the target wire where the operation acts on.")
        if control_wires is None:
            if len(wires) > 1:
                control_wires = Wires(wires[:-1])
                wires = Wires(wires[-1])
            else:
                raise ValueError(
                    "IntegerComparator: wrong number of wires. "
                    f"{len(wires)} wire(s) given. Need at least 2."
                )
        else:
            wires = Wires(wires)
            control_wires = Wires(control_wires)

            if len(wires) != 1:
                raise ValueError("IntegerComparator accepts a single target wire.")

        total_wires = control_wires + wires

        self.hyperparameters["control_wires"] = control_wires
        self.hyperparameters["value"] = value
        self.hyperparameters["geq"] = geq
        self.geq = geq

        super().__init__(wires=total_wires, do_queue=do_queue)

    def label(self, decimals=None, base_label=None, cache=None):
        return base_label or "X"

    # pylint: disable=unused-argument
    @staticmethod
    def compute_matrix(
        value, control_wires, geq=True, **kwargs
    ):  # pylint: disable=arguments-differ
        r"""Representation of the operator as a canonical matrix in the computational basis (static method).

        The canonical matrix is the textbook matrix representation that does not consider wires.
        Implicitly, this assumes that the wires of the operator correspond to the global wire order.

        .. seealso:: :meth:`~.IntegerComparator.matrix`

        Args:
            value (int): The value :math:`L` that the state's decimal representation is compared against.
            control_wires (Union[Wires, Sequence[int], or int]): wires to place controls on
            geq (bool): If set to `True`, the comparison made will be :math:`\sigma \geq L`. If `False`, the comparison
                made will be :math:`\sigma < L`.

        Returns:
           tensor_like: matrix representation

        **Example**

        >>> print(qml.IntegerComparator.compute_matrix(2, [0,1]))
        [[1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]
         [0. 0. 0. 0. 0. 0. 1. 0.]]
        >>> print(qml.IntegerComparator.compute_matrix(2, [0,1], geq=False))
        [[0. 1. 0. 0. 0. 0. 0. 0.]
         [1. 0. 0. 0. 0. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0. 0. 0.]
         [0. 0. 0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 0. 0. 1.]]
        """

        if value is None:
            raise ValueError("The value to compare to must be specified.")

        if isinstance(value, int):
            paulix_mat = qml.PauliX.compute_matrix()

            if geq:  # comparison is i >= value
                if value > 2 ** len(control_wires) - 1:  # if value exceeds Hilbert space size
                    mat = np.eye(2 ** (len(control_wires) + 1))

                padding_left = value * 2
                paulix_padding = (2 ** (len(control_wires) + 1) - padding_left) // 2
                paulix_blocks = [paulix_mat for _ in range(paulix_padding)]
                mat = block_diag(np.eye(padding_left), *paulix_blocks)

            else:  # comparison is i < value

                if 2 ** (len(control_wires)) < value:  # if value exceeds Hilbert space size
                    mat = np.eye(2 ** (len(control_wires) + 1))

                padding_right = (2 ** (len(control_wires)) - value) * 2
                paulix_padding = (2 ** (len(control_wires) + 1) - padding_right) // 2
                paulix_blocks = [paulix_mat for _ in range(paulix_padding)]
                mat = block_diag(*paulix_blocks, np.eye(padding_right))

        else:
            raise ValueError("The compared value must be an int. Got {}.".format(type(value)))

        return mat

    @property
    def control_wires(self):
        return self.wires[:~0]

    def adjoint(self):
        return IntegerComparator(value=self.value, geq=self.geq, control_wires=None, wires=None)

    def pow(self, z):
        return super().pow(z % 2)


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

    def __init__(self, wires=Wires([]), only_visual=False, do_queue=True, id=None):
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
        return copy(self).queue()

    def adjoint(self):
        return copy(self)

    def pow(self, z):
        return [copy(self)]

    def simplify(self):
        if self.only_visual:
            if len(self.wires) == 1:
                return qml.Identity(self.wires[0])
            return qml.prod(*(qml.Identity(w) for w in self.wires))
        return self


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

    def __init__(self, *params, wires=None, do_queue=True, id=None):
        if wires == []:
            raise ValueError(
                f"{self.__class__.__name__}: wrong number of wires. "
                f"At least one wire has to be given."
            )
        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_decomposition(wires):  # pylint: disable=unused-argument
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

    def pow(self, z):
        return [copy(self)]
