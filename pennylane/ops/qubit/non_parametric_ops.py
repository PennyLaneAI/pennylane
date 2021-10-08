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
import numpy as np
from scipy.linalg import block_diag

import pennylane as qml
from pennylane.operation import AnyWires, DiagonalOperation, Observable, Operation
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
    num_params = 0
    num_wires = 1
    par_domain = None
    is_self_inverse = True
    eigvals = pauli_eigs(1)
    matrix = np.array([[INV_SQRT2, INV_SQRT2], [INV_SQRT2, -INV_SQRT2]])

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.eigvals

    def diagonalizing_gates(self):
        r"""Rotates the specified wires such that they
        are in the eigenbasis of the Hadamard operator.

        For the Hadamard operator,

        .. math:: H = U^\dagger Z U

        where :math:`U = R_y(-\pi/4)`.

        Returns:
            list(~.Operation): A list of gates that diagonalize Hadamard in
            the computational basis.
        """
        return [qml.RY(-np.pi / 4, wires=self.wires)]

    @staticmethod
    def decomposition(wires):
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
    num_params = 0
    num_wires = 1
    par_domain = None
    is_self_inverse = True
    basis = "X"
    eigvals = pauli_eigs(1)
    matrix = np.array([[0, 1], [1, 0]])

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.eigvals

    def diagonalizing_gates(self):
        r"""Rotates the specified wires such that they
        are in the eigenbasis of the Pauli-X operator.

        For the Pauli-X operator,

        .. math:: X = H^\dagger Z H.

        Returns:
            list(qml.Operation): A list of gates that diagonalize PauliY in the
            computational basis.
        """
        return [Hadamard(wires=self.wires)]

    @staticmethod
    def decomposition(wires):
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
    num_params = 0
    num_wires = 1
    par_domain = None
    is_self_inverse = True
    basis = "Y"
    eigvals = pauli_eigs(1)
    matrix = np.array([[0, -1j], [1j, 0]])

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.eigvals

    def diagonalizing_gates(self):
        r"""Rotates the specified wires such that they
        are in the eigenbasis of PauliY.

        For the Pauli-Y observable,

        .. math:: Y = U^\dagger Z U

        where :math:`U=HSZ`.

        Returns:
            list(~.Operation): A list of gates that diagonalize PauliY in the
                computational basis.
        """
        return [
            PauliZ(wires=self.wires),
            S(wires=self.wires),
            Hadamard(wires=self.wires),
        ]

    @staticmethod
    def decomposition(wires):
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


class PauliZ(Observable, DiagonalOperation):
    r"""PauliZ(wires)
    The Pauli Z operator

    .. math:: \sigma_z = \begin{bmatrix} 1 & 0 \\ 0 & -1\end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 0
    num_wires = 1
    par_domain = None
    is_self_inverse = True
    basis = "Z"
    eigvals = pauli_eigs(1)
    matrix = np.array([[1, 0], [0, -1]])

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.eigvals

    def diagonalizing_gates(self):
        return []

    @staticmethod
    def decomposition(wires):
        decomp_ops = [qml.PhaseShift(np.pi, wires=wires)]
        return decomp_ops

    def adjoint(self):
        return PauliZ(wires=self.wires)

    def _controlled(self, wire):
        CZ(wires=Wires(wire) + self.wires)

    def single_qubit_rot_angles(self):
        # Z = RZ(\pi) RY(0) RZ(0)
        return [np.pi, 0.0, 0.0]


class S(DiagonalOperation):
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
    num_params = 0
    num_wires = 1
    par_domain = None
    basis = "Z"
    op_eigvals = np.array([1, 1j])
    op_matrix = np.array([[1, 0], [0, 1j]])

    @classmethod
    def _matrix(cls, *params):
        return cls.op_matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.op_eigvals

    @staticmethod
    def decomposition(wires):
        decomp_ops = [qml.PhaseShift(np.pi / 2, wires=wires)]
        return decomp_ops

    def adjoint(self):
        return S(wires=self.wires).inv()

    def single_qubit_rot_angles(self):
        # S = RZ(\pi/2) RY(0) RZ(0)
        return [np.pi / 2, 0.0, 0.0]


class T(DiagonalOperation):
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
    num_params = 0
    num_wires = 1
    par_domain = None
    basis = "Z"
    op_matrix = np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]])
    op_eigvals = np.array([1, cmath.exp(1j * np.pi / 4)])

    @classmethod
    def _matrix(cls, *params):
        return cls.op_matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.op_eigvals

    @staticmethod
    def decomposition(wires):
        decomp_ops = [qml.PhaseShift(np.pi / 4, wires=wires)]
        return decomp_ops

    def adjoint(self):
        return T(wires=self.wires).inv()

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
    num_params = 0
    num_wires = 1
    par_domain = None
    basis = "X"
    op_matrix = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]])
    op_eigvals = np.array([1, 1j])

    @classmethod
    def _matrix(cls, *params):
        return cls.op_matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.op_eigvals

    @staticmethod
    def decomposition(wires):
        decomp_ops = [
            qml.RZ(np.pi / 2, wires=wires),
            qml.RY(np.pi / 2, wires=wires),
            qml.RZ(-np.pi, wires=wires),
            qml.PhaseShift(np.pi / 2, wires=wires),
        ]
        return decomp_ops

    def adjoint(self):
        return SX(wires=self.wires).inv()

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
    num_params = 0
    num_wires = 2
    par_domain = None
    is_self_inverse = True
    basis = "X"
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    @classmethod
    def _matrix(cls, *params):
        return CNOT.matrix

    def adjoint(self):
        return CNOT(wires=self.wires)

    def _controlled(self, wire):
        Toffoli(wires=Wires(wire) + self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


class CZ(DiagonalOperation):
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
    num_params = 0
    num_wires = 2
    par_domain = None
    is_self_inverse = True
    is_symmetric_over_all_wires = True
    basis = "Z"
    eigvals = np.array([1, 1, 1, -1])
    matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.eigvals

    def adjoint(self):
        return CZ(wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


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
    num_params = 0
    num_wires = 2
    par_domain = None
    is_self_inverse = True
    basis = "Y"
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0],
        ]
    )

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @staticmethod
    def decomposition(wires):
        decomp_ops = [qml.CRY(np.pi, wires=wires), S(wires=wires[0])]
        return decomp_ops

    def adjoint(self):
        return CY(wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[0])


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
    num_params = 0
    num_wires = 2
    par_domain = None
    is_self_inverse = True
    is_symmetric_over_all_wires = True
    basis = "X"
    matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @staticmethod
    def decomposition(wires):
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

    @property
    def control_wires(self):
        return Wires(self.wires[:2])


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
    num_params = 0
    num_wires = 2
    par_domain = None
    op_matrix = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
    op_eigvals = np.array([1j, -1j, 1, 1])

    @classmethod
    def _matrix(cls, *params):
        return cls.op_matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.op_eigvals

    @staticmethod
    def decomposition(wires):
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
        return ISWAP(wires=self.wires).inv()


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
    num_params = 0
    num_wires = 2
    par_domain = None
    op_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, INV_SQRT2, INV_SQRT2 * 1j, 0],
            [0, INV_SQRT2 * 1j, INV_SQRT2, 0],
            [0, 0, 0, 1],
        ]
    )
    op_eigvals = np.array([INV_SQRT2 * (1 + 1j), INV_SQRT2 * (1 - 1j), 1, 1])

    @classmethod
    def _matrix(cls, *params):
        return cls.op_matrix

    @classmethod
    def _eigvals(cls, *params):
        return cls.op_eigvals

    @staticmethod
    def decomposition(wires):
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
        return SISWAP(wires=self.wires).inv()


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
    num_params = 0
    num_wires = 3
    par_domain = None
    matrix = np.array(
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

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @staticmethod
    def decomposition(wires):
        decomp_ops = [
            qml.Toffoli(wires=[wires[0], wires[2], wires[1]]),
            qml.Toffoli(wires=[wires[0], wires[1], wires[2]]),
            qml.Toffoli(wires=[wires[0], wires[2], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        return CSWAP(wires=self.wires)


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
    num_params = 0
    num_wires = 3
    par_domain = None
    is_self_inverse = True
    is_symmetric_over_control_wires = True
    basis = "X"
    matrix = np.array(
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

    @classmethod
    def _matrix(cls, *params):
        return cls.matrix

    @staticmethod
    def decomposition(wires):
        decomp_ops = [
            Hadamard(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            T(wires=wires[2]).inv(),
            CNOT(wires=[wires[0], wires[2]]),
            T(wires=wires[2]),
            CNOT(wires=[wires[1], wires[2]]),
            T(wires=wires[2]).inv(),
            CNOT(wires=[wires[0], wires[2]]),
            T(wires=wires[2]),
            T(wires=wires[1]),
            CNOT(wires=[wires[0], wires[1]]),
            Hadamard(wires=wires[2]),
            T(wires=wires[0]),
            T(wires=wires[1]).inv(),
            CNOT(wires=[wires[0], wires[1]]),
        ]
        return decomp_ops

    def adjoint(self):
        return Toffoli(wires=self.wires)

    @property
    def control_wires(self):
        return Wires(self.wires[:2])


class MultiControlledX(Operation):
    r"""MultiControlledX(control_wires, wires, control_values)
    Apply a Pauli X gate controlled on an arbitrary computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 0
    * Gradient recipe: None

    Args:
        control_wires (Union[Wires, Sequence[int], or int]): the control wire(s)
        wires (Union[Wires or int]): a single target wire the operation acts on
        control_values (str): a string of bits representing the state of the control
            qubits to control on (default is the all 1s state)
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

    **Example**

    The ``MultiControlledX`` operation (sometimes called a mixed-polarity
    multi-controlled Toffoli) is a commonly-encountered case of the
    :class:`~.pennylane.ControlledQubitUnitary` operation wherein the applied
    unitary is the Pauli X (NOT) gate. It can be used in the same manner as
    ``ControlledQubitUnitary``, but there is no need to specify a matrix
    argument:

    >>> qml.MultiControlledX(control_wires=[0, 1, 2, 3], wires=4, control_values='1110')

    """
    num_params = 0
    num_wires = AnyWires
    par_domain = "A"
    grad_method = None

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *params,
        control_wires=None,
        wires=None,
        control_values=None,
        work_wires=None,
        do_queue=True,
    ):
        wires = Wires(wires)
        control_wires = Wires(control_wires)
        work_wires = Wires([]) if work_wires is None else Wires(work_wires)

        if len(wires) != 1:
            raise ValueError("MultiControlledX accepts a single target wire.")

        if Wires.shared_wires([wires, work_wires]) or Wires.shared_wires(
            [control_wires, work_wires]
        ):
            raise ValueError("The work wires must be different from the control and target wires")

        self._target_wire = wires[0]
        self._work_wires = work_wires
        self._control_wires = control_wires

        wires = control_wires + wires

        if not control_values:
            control_values = "1" * len(control_wires)

        control_int = self._parse_control_values(control_wires, control_values)
        self.control_values = control_values

        self._padding_left = control_int * 2
        self._padding_right = 2 ** len(wires) - 2 - self._padding_left
        self._CX = None

        super().__init__(*params, wires=wires, do_queue=do_queue)

    def _matrix(self, *params):
        if self._CX is None:
            self._CX = block_diag(
                np.eye(self._padding_left), PauliX.matrix, np.eye(self._padding_right)
            )

        return self._CX

    @property
    def control_wires(self):
        return self._control_wires

    @staticmethod
    def _parse_control_values(control_wires, control_values):
        """Ensure any user-specified control strings have the right format."""
        if isinstance(control_values, str):
            if len(control_values) != len(control_wires):
                raise ValueError("Length of control bit string must equal number of control wires.")

            # Make sure all values are either 0 or 1
            if any(x not in ["0", "1"] for x in control_values):
                raise ValueError("String of control values can contain only '0' or '1'.")

            control_int = int(control_values, 2)
        else:
            raise ValueError("Alternative control values must be passed as a binary string.")

        return control_int

    def adjoint(self):
        return MultiControlledX(
            control_wires=self.wires[:-1],
            wires=self.wires[-1],
            control_values=self.control_values,
            work_wires=self._work_wires,
        )

    # pylint: disable=unused-argument
    def decomposition(self, *args, **kwargs):

        if len(self.control_wires) > 2 and len(self._work_wires) == 0:
            raise ValueError(f"At least one work wire is required to decompose operation: {self}")

        flips1 = [
            qml.PauliX(self.control_wires[i])
            for i, val in enumerate(self.control_values)
            if val == "0"
        ]

        if len(self.control_wires) == 1:
            decomp = [qml.CNOT(wires=[self.control_wires[0], self._target_wire])]
        elif len(self.control_wires) == 2:
            decomp = [qml.Toffoli(wires=[*self.control_wires, self._target_wire])]
        else:
            num_work_wires_needed = len(self.control_wires) - 2

            if len(self._work_wires) >= num_work_wires_needed:
                decomp = self._decomposition_with_many_workers(
                    self.control_wires, self._target_wire, self._work_wires
                )
            else:
                work_wire = self._work_wires[0]
                decomp = self._decomposition_with_one_worker(
                    self.control_wires, self._target_wire, work_wire
                )

        flips2 = [
            qml.PauliX(self.control_wires[i])
            for i, val in enumerate(self.control_values)
            if val == "0"
        ]

        return flips1 + decomp + flips2

    @staticmethod
    def _decomposition_with_many_workers(control_wires, target_wire, work_wires):
        """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
        https://arxiv.org/pdf/quant-ph/9503016.pdf, which requires a suitably large register of
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
        https://arxiv.org/pdf/quant-ph/9503016.pdf, which requires a single work wire"""
        tot_wires = len(control_wires) + 2
        partition = int(np.ceil(tot_wires / 2))

        first_part = control_wires[:partition]
        second_part = control_wires[partition:]

        gates = [
            MultiControlledX(
                control_wires=first_part,
                wires=work_wire,
                work_wires=second_part + target_wire,
            ),
            MultiControlledX(
                control_wires=second_part + work_wire,
                wires=target_wire,
                work_wires=first_part,
            ),
            MultiControlledX(
                control_wires=first_part,
                wires=work_wire,
                work_wires=second_part + target_wire,
            ),
            MultiControlledX(
                control_wires=second_part + work_wire,
                wires=target_wire,
                work_wires=first_part,
            ),
        ]

        return gates
