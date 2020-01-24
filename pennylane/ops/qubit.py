# Copyright 2018-2019 Xanadu Quantum Technologies Inc.

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
This module contains the available built-in discrete-variable
quantum operations supported by PennyLane, as well as their conventions.
"""
# pylint:disable=abstract-method,arguments-differ,protected-access
import numpy as np
from scipy.linalg import block_diag

from pennylane.operation import Any, Observable, Operation
from pennylane.templates.state_preparations import (
    BasisStatePreparation,
    MottonenStatePreparation,
)
from pennylane.utils import OperationRecorder, pauli_eigs


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
    eigvals = pauli_eigs(1)

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def diagonalizing_gates(self):
        r"""Rotates the specified wires such that they
        are in the eigenbasis of the Hadamard operator.

        For the Hadamard operator,

        .. math:: H = U^\dagger Z U

        where :math:`U = R_y(-\pi/4)`.

        Returns:
            list(qml.Operation): A list of gates that diagonalize Hadamard in the
            computational basis.
        """
        return [RY(-np.pi/4, wires=self.wires)]


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
    eigvals = pauli_eigs(1)

    @staticmethod
    def _matrix(*params):
        return np.array([[0, 1], [1, 0]])

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
    eigvals = pauli_eigs(1)

    @staticmethod
    def _matrix(*params):
        return np.array([[0, -1j], [1j, 0]])

    def diagonalizing_gates(self):
        r"""Rotates the specified wires such that they
        are in the eigenbasis of PauliY.

        For the Pauli-Y observable,

        .. math:: Y = U^\dagger Z U

        where :math:`U=HSZ`.

        Returns:
            list(qml.Operation): A list of gates that diagonalize PauliY in the
            computational basis.
        """
        return [PauliZ(wires=self.wires), S(wires=self.wires), Hadamard(wires=self.wires)]


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
    num_params = 0
    num_wires = 1
    par_domain = None
    eigvals = pauli_eigs(1)

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 0], [0, -1]])

    def diagonalizing_gates(self):
        return []


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
    num_params = 0
    num_wires = 1
    par_domain = None

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 0], [0, 1j]])


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
    num_params = 0
    num_wires = 1
    par_domain = None

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 0], [0, np.exp(1j*np.pi/4)]])


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
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])


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
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])


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
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    num_wires = 2
    par_domain = None

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


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
        wires (Sequence[int] or int): the wires the operation acts on
    """
    num_params = 0
    num_wires = 3
    par_domain = None

    @staticmethod
    def _matrix(*params):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1]])


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
        wires (int): the subsystem the gate acts on
    """
    num_params = 0
    num_wires = 3
    par_domain = None

    @staticmethod
    def _matrix(*params):
        mat = np.diag([1 for i in range(8)])
        mat[6:8, 6:8] = np.array([[0, 1], [1, 0]])
        return mat


class RX(Operation):
    r"""RX(phi, wires)
    The single qubit X rotation

    .. math:: R_x(\phi) = e^{-i\phi\sigma_x/2} = \begin{bmatrix}
                \cos(\phi/2) & -i\sin(\phi/2) \\
                -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_x(\phi)) = \frac{1}{2}\left[f(R_x(\phi+\pi/2)) - f(R_x(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_x(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "A"
    generator = [PauliX, -1 / 2]

    @staticmethod
    def _matrix(*params):
        theta = params[0]
        return np.cos(theta/2) * np.eye(2) + 1j * np.sin(-theta/2) * PauliX._matrix()


class RY(Operation):
    r"""RY(phi, wires)
    The single qubit Y rotation

    .. math:: R_y(\phi) = e^{-i\phi\sigma_y/2} = \begin{bmatrix}
                \cos(\phi/2) & -\sin(\phi/2) \\
                \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_y(\phi)) = \frac{1}{2}\left[f(R_y(\phi+\pi/2)) - f(R_y(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_y(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "A"
    generator = [PauliY, -1 / 2]

    @staticmethod
    def _matrix(*params):
        theta = params[0]
        return np.cos(theta/2) * np.eye(2) + 1j * np.sin(-theta/2) * PauliY._matrix()


class RZ(Operation):
    r"""RZ(phi, wires)
    The single qubit Z rotation

    .. math:: R_z(\phi) = e^{-i\phi\sigma_z/2} = \begin{bmatrix}
                e^{-i\phi/2} & 0 \\
                0 & e^{i\phi/2}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_z(\phi)) = \frac{1}{2}\left[f(R_z(\phi+\pi/2)) - f(R_z(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_z(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "A"
    generator = [PauliZ, -1 / 2]

    @staticmethod
    def _matrix(*params):
        theta = params[0]
        return np.cos(theta/2) * np.eye(2) + 1j * np.sin(-theta/2) * PauliZ._matrix()


class PhaseShift(Operation):
    r"""PhaseShift(phi, wires)
    Arbitrary single qubit local phase shift

    .. math:: R_\phi(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\phi}
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R_\phi(\phi)) = \frac{1}{2}\left[f(R_\phi(\phi+\pi/2)) - f(R_\phi(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`R_{\phi}(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "A"
    generator = [np.array([[0, 0], [0, 1]]), 1]

    @staticmethod
    def _matrix(*params):
        phi = params[0]
        return np.array([[1, 0], [0, np.exp(1j*phi)]])


class Rot(Operation):
    r"""Rot(phi, theta, omega, wires)
    Arbitrary single qubit rotation

    .. math::

        R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
        e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
        e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Gradient recipe: :math:`\frac{d}{d\phi}f(R(\phi, \theta, \omega)) = \frac{1}{2}\left[f(R(\phi+\pi/2, \theta, \omega)) - f(R(\phi-\pi/2, \theta, \omega))\right]`
      where :math:`f` is an expectation value depending on :math:`R(\phi, \theta, \omega)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \theta, \omega\}`.

    .. note::

        If the ``Rot`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.RZ` and :class:`~.RY` gates.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 3
    num_wires = 1
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def _matrix(*params):
        a, b, c = params
        return RZ._matrix(c) @ (RY._matrix(b) @ RZ._matrix(a))

    @staticmethod
    def decomposition(phi, theta, omega, wires):
        decomp_ops = [
            RZ(phi, wires=wires),
            RY(theta, wires=wires),
            RZ(omega, wires=wires),
        ]
        return decomp_ops


class CRX(Operation):
    r"""CRX(phi, wires)
    The controlled-RX operator

    .. math::

        \begin{align}
            CRX(\phi) &= I_{1}\otimes RZ_{2}(\pi / 2) ~\cdot~ I_{1}\otimes RY_{2}(\phi/2) ~\cdot~ CNOT_{12} ~\cdot~ RY_{2}(-\phi/2) ~\cdot~ CNOT_{12} ~\cdot~ I_{1}\otimes RZ_{2}(-\pi / 2)\notag \\[10pt]
            &=
            \begin{bmatrix}
            & 1 & 0 & 0 & 0 \\
            & 0 & 1 & 0 & 0\\
            & 0 & 0 & \cos(\phi/2) & -i\sin(\phi/2)\\
            & 0 & 0 & -i\sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.
        \end{align}

    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g., 1 corresponds to the first element in ``wires`` that is the **control qubit**.


    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR_x(\phi)) = \frac{1}{2}\left[f(CR_x(\phi+\pi/2)) - f(CR_x(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`CR_x(\phi)`.

    **Decomposition**

    If the ``CRX`` gate is not supported on the targeted device, PennyLane
    will attempt to decompose the gate into :class:`~.RZ`, :class:`~.RY`
    and :class:`~.CNOT` gates the following way:


    .. image:: ../../_static/crx_circuit.png
        :align: center
        :width: 800px

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    generator = [
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        -1 / 2,
    ]

    @staticmethod
    def _matrix(*params):
        return block_diag(np.eye(2), RX._matrix(*params))

    @staticmethod
    def decomposition(theta, wires):
        decomp_ops = [
            RZ(np.pi / 2, wires=wires[1]),
            RY(theta / 2, wires=wires[1]),
            CNOT(wires=wires),
            RY(-theta / 2, wires=wires[1]),
            CNOT(wires=wires),
            RZ(-np.pi / 2, wires=wires[1]),
        ]
        return decomp_ops


class CRY(Operation):
    r"""CRY(phi, wires)
    The controlled-RY operator

    .. math::

        \begin{align}
             CRY(\phi) &= I_{1}\otimes U3_{2}(\pi / 2) ~\cdot~ CNOT_{12} ~\cdot~ I_{1}\otimes U3_{2}(-\pi / 2) ~\cdot~ CNOT_{12} \notag \\[10pt]
            &=
        \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & \cos(\phi/2) & -\sin(\phi/2)\\
            0 & 0 & \sin(\phi/2) & \cos(\phi/2)
        \end{bmatrix}.
        \end{align}

    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g. 1 corresponds to the first element in ``wires`` that is the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR_y(\phi)) = \frac{1}{2}\left[f(CR_y(\phi+\pi/2)) - f(CR_y(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`CR_y(\phi)`.

    **Decomposition**

    If the ``CRY`` gate is not supported on the targeted device, PennyLane
    will attempt to decompose the gate into :class:`~.U3` and :class:`~.CNOT` gates the following way:

    .. image:: ../../_static/cry_circuit.png
        :align: center
        :width: 650px


    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    generator = [
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        -1 / 2,
    ]

    @staticmethod
    def _matrix(*params):
        return block_diag(np.eye(2), RY._matrix(*params))

    @staticmethod
    def decomposition(theta, wires):
        decomp_ops = [
            U3(theta / 2, 0, 0, wires=wires[1]),
            CNOT(wires=wires),
            U3(-theta / 2, 0, 0, wires=wires[1]),
            CNOT(wires=wires),
        ]
        return decomp_ops


class CRZ(Operation):
    r"""CRZ(phi, wires)
    The controlled-RZ operator

    .. math::

        \begin{align}
             CRZ(\phi) &= I_{1}\otimes PhaseShift_{2}(\pi / 2) ~\cdot~ CNOT_{12} ~\cdot~ I_{1}\otimes PhaseShift_{2}(-\pi / 2) ~\cdot~ CNOT_{12} \notag \\[10pt]
            &=
         \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i\phi/2} & 0\\
            0 & 0 & 0 & e^{i\phi/2}
        \end{bmatrix}.
        \end{align}


    .. note:: The subscripts of the operations in the formula refer to the wires they act on, e.g. 1 corresponds to the first element in ``wires`` that is the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR_z(\phi)) = \frac{1}{2}\left[f(CR_z(\phi+\pi/2)) - f(CR_z(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`CR_z(\phi)`.

    **Decomposition**

    If the ``CRZ`` gate is not supported on the targeted device, PennyLane
    will attempt to decompose the gate into :class:`~.PhaseShift` and :class:`~.CNOT` gates the following way:

    .. image:: ../../_static/crz_circuit.png
        :align: center
        :width: 650px

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 2
    par_domain = "R"
    grad_method = "A"
    generator = [
        np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        -1 / 2,
    ]

    @staticmethod
    def _matrix(*params):
        return block_diag(np.eye(2), RZ._matrix(*params))

    @staticmethod
    def decomposition(lam, wires):
        decomp_ops = [
            PhaseShift(lam / 2, wires=wires[1]),
            CNOT(wires=wires),
            PhaseShift(-lam / 2, wires=wires[1]),
            CNOT(wires=wires),
        ]
        return decomp_ops


class CRot(Operation):
    r"""CRot(phi, theta, omega, wires)
    The controlled-Rot operator

    .. math:: CR(\phi, \theta, \omega) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0\\
            0 & 0 & e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2)\\
            0 & 0 & e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
        \end{bmatrix}.

    .. note:: The first wire provided corresponds to the **control qubit**.

    **Details:**

    * Number of wires: 2
    * Number of parameters: 3
    * Gradient recipe: :math:`\frac{d}{d\phi}f(CR(\phi, \theta, \omega)) = \frac{1}{2}\left[f(CR(\phi+\pi/2, \theta, \omega)) - f(CR(\phi-\pi/2, \theta, \omega))\right]`
      where :math:`f` is an expectation value depending on :math:`CR(\phi, \theta, \omega)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \theta, \omega\}`.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        omega (float): rotation angle :math:`\omega`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 3
    num_wires = 2
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def _matrix(*params):
        a, b, c = params
        return CRZ._matrix(c) @ (CRY._matrix(b) @ CRZ._matrix(a))


class U1(Operation):
    r"""U1(phi)
    U1 gate.

    .. math:: U_1(\phi) = e^{i\phi/2}R_z(\phi) = \begin{bmatrix}
                1 & 0 \\
                0 & e^{i\phi}
            \end{bmatrix}.

    .. note::

        The ``U1`` gate is an alias for the phase shift operation :class:`~.PhaseShift`.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_1(\phi)) = \frac{1}{2}\left[f(U_1(\phi+\pi/2)) - f(U_1(\phi-\pi/2))\right]`
      where :math:`f` is an expectation value depending on :math:`U_1(\phi)`.

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = "R"
    grad_method = "A"
    generator = [np.array([[0, 0], [0, 1]]), 1]

    @staticmethod
    def _matrix(*params):
        return PhaseShift._matrix(*params)

    @staticmethod
    def decomposition(phi, wires):
        return [PhaseShift(phi, wires=wires)]


class U2(Operation):
    r"""U2(phi, lambda, wires)
    U2 gate.

    .. math::

        U_2(\phi, \lambda) = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -\exp(i \lambda)
        \\ \exp(i \phi) & \exp(i (\phi + \lambda)) \end{bmatrix}

    The :math:`U_2` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_2(\phi, \lambda) = R_\phi(\phi+\lambda) R(\lambda,\pi/2,-\lambda)

    .. note::

        If the ``U2`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.Rot` and :class:`~.PhaseShift` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 2
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_2(\phi, \lambda)) = \frac{1}{2}\left[f(U_2(\phi+\pi/2, \lambda)) - f(U_2(\phi-\pi/2, \lambda))\right]`
      where :math:`f` is an expectation value depending on :math:`U_2(\phi, \lambda)`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \lambda\}`.

    Args:
        phi (float): azimuthal angle :math:`\phi`
        lambda (float): quantum phase :math:`\lambda`
        wires (Sequence[int] or int): the subsystem the gate acts on
    """
    num_params = 2
    num_wires = 1
    par_domain = "R"
    grad_method = "A"

    @staticmethod
    def _matrix(*params):
        phi, lam = params
        return PhaseShift._matrix(phi+lam) @ Rot._matrix(lam, np.pi/2, -lam)

    @staticmethod
    def decomposition(phi, lam, wires):
        decomp_ops = [
            Rot(lam, np.pi / 2, -lam, wires=wires),
            PhaseShift(lam, wires=wires),
            PhaseShift(phi, wires=wires),
        ]
        return decomp_ops


class U3(Operation):
    r"""U3(theta, phi, lambda, wires)
    Arbitrary single qubit unitary.

    .. math::

        U_3(\theta, \phi, \lambda) = \begin{bmatrix} \cos(\theta/2) & -\exp(i \lambda)\sin(\theta/2) \\
        \exp(i \phi)\sin(\theta/2) & \exp(i (\phi + \lambda))\cos(\theta/2) \end{bmatrix}

    The :math:`U_3` gate is related to the single-qubit rotation :math:`R` (:class:`Rot`) and the
    :math:`R_\phi` (:class:`PhaseShift`) gates via the following relation:

    .. math::

        U_3(\theta, \phi, \lambda) = R_\phi(\phi+\lambda) R(\lambda,\theta,-\lambda)

    .. note::

        If the ``U3`` gate is not supported on the targeted device, PennyLane
        will attempt to decompose the gate into :class:`~.PhaseShift` and :class:`~.Rot` gates.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 3
    * Gradient recipe: :math:`\frac{d}{d\phi}f(U_3(\theta, \phi, \lambda)) = \frac{1}{2}\left[f(U_3(\theta+\pi/2, \phi, \lambda)) - f(U_3(\theta-\pi/2, \phi, \lambda))\right]`
      where :math:`f` is an expectation value depending on :math:`U_3(\theta, \phi, \lambda)`.
      This gradient recipe applies for each angle argument :math:`\{\theta, \phi, \lambda\}`.

    Args:
        theta (float): polar angle :math:`\theta`
        phi (float): azimuthal angle :math:`\phi`
        lambda (float): quantum phase :math:`\lambda`
        wires (Sequence[int] or int): the subsystem the gate acts on
    """
    num_params = 3
    num_wires = 1
    par_domain = "R"
    grad_method = "A"


    @staticmethod
    def _matrix(*params):
        theta, phi, lam = params
        return PhaseShift._matrix(phi+lam) @ Rot._matrix(lam, theta, -lam)

    @staticmethod
    def decomposition(theta, phi, lam, wires):
        decomp_ops = [
            Rot(lam, theta, -lam, wires=wires),
            PhaseShift(lam, wires=wires),
            PhaseShift(phi, wires=wires),
        ]
        return decomp_ops


# =============================================================================
# Arbitrary operations
# =============================================================================


class QubitUnitary(Operation):
    r"""QubitUnitary(U, wires)
    Apply an arbitrary fixed unitary matrix.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    Args:
        U (array[complex]): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = Any
    par_domain = "A"
    grad_method = None

    @staticmethod
    def _matrix(*params):
        U = np.asarray(params[0])

        if U.shape[0] != U.shape[1]:
            raise ValueError("Operator must be a square matrix.")

        if not np.allclose(U @ U.conj().T, np.identity(U.shape[0])):
            raise ValueError("Operator must be unitary.")

        return U


# =============================================================================
# State preparation
# =============================================================================


class BasisState(Operation):
    r"""BasisState(n, wires)
    Prepares a single computational basis state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None (integer parameters not supported)

    .. note::

        If the ``BasisState`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        into :class:`~.PauliX` operations.

    Args:
        n (array): prepares the basis state :math:`\ket{n}`, where ``n`` is an
            array of integers from the set :math:`\{0, 1\}`, i.e.,
            if ``n = np.array([0, 1, 0])``, prepares the state :math:`|010\rangle`.
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = Any
    par_domain = "A"
    grad_method = None

    @staticmethod
    def decomposition(n, wires):
        with OperationRecorder() as rec:
            BasisStatePreparation(n, wires)

        return rec.queue


class QubitStateVector(Operation):
    r"""QubitStateVector(state, wires)
    Prepare subsystems using the given ket vector in the computational basis.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        If the ``QubitStateVector`` operation is not supported natively on the
        target device, PennyLane will attempt to decompose the operation
        using the method developed by Möttönen et al. (Quantum Info. Comput.,
        2005).

    Args:
        state (array[complex]): a state vector of size 2**len(wires)
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = Any
    par_domain = "A"
    grad_method = None

    @staticmethod
    def decomposition(state, wires):
        with OperationRecorder() as rec:
            MottonenStatePreparation(state, wires)

        return rec.queue


# =============================================================================
# Observables
# =============================================================================


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
    num_wires = Any
    num_params = 1
    par_domain = "A"
    grad_method = "F"
    _eigs = {}

    @staticmethod
    def _matrix(*params):
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

        Hmat = self.matrix
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
        return [
            QubitUnitary(self.eigendecomposition["eigvec"].conj().T, wires=list(self.wires)),
        ]


ops = {
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "S",
    "T",
    "CNOT",
    "CZ",
    "SWAP",
    "CSWAP",
    "Toffoli",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "Rot",
    "CRX",
    "CRY",
    "CRZ",
    "CRot",
    "U1",
    "U2",
    "U3",
    "BasisState",
    "QubitStateVector",
    "QubitUnitary",
}


obs = {"Hadamard", "PauliX", "PauliY", "PauliZ", "Hermitian"}


__all__ = list(ops | obs)
