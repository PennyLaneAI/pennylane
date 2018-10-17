# Copyright 2018 Xanadu Quantum Technologies Inc.

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
Qubit quantum operations
========================

.. currentmodule:: openqml.ops.builtins_discrete

This section contains the available built-in discrete-variable
quantum operations supported by OpenQML, as well as their conventions.

Gates
-----

.. autosummary::
    Hadamard
    PauliX
    PauliY
    PauliZ
    CNOT
    CZ
    SWAP
    RX
    RY
    RZ
    PhaseShift
    Rot
    QubitUnitary


State preparation
-----------------

.. autosummary::
    BasisState
    QubitStateVector


Details
-------
"""

from openqml.operation import Operation


class Hadamard(Operation):
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


class PauliX(Operation):
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


class PauliY(Operation):
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


class PauliZ(Operation):
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


class SWAP(Operation):
    r"""SWAP(wires)
    The swap operator

    .. math:: SWAP = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 1 & 0\\
            0 & 1 & 0 & 0\\
            0 & 0 & 0 & 1
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
    * Gradient recipe: :math:`\frac{d}{d\phi}R_x(\phi) = \frac{1}{2}\left[R_x(\phi+\pi/2)+R_x(\phi-\pi/2)\right]`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = 'R'
    grad_method = 'A'


class RY(Operation):
    r"""RY(phi, wires)
    The single qubit Y rotation.

    .. math:: R_y(\phi) = e^{-i\phi\sigma_y/2} = \begin{bmatrix}
                \cos(\phi/2) & -\sin(\phi/2) \\
                \sin(\phi/2) & \cos(\phi/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}R_y(\phi) = \frac{1}{2}\left[R_y(\phi+\pi/2)+R_y(\phi-\pi/2)\right]`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = 'R'
    grad_method = 'A'


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
    * Gradient recipe: :math:`\frac{d}{d\phi}R_z(\phi) = \frac{1}{2}\left[R_z(\phi+\pi/2)+R_z(\phi-\pi/2)\right]`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = 'R'
    grad_method = 'A'


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
    * Gradient recipe: :math:`\frac{d}{d\phi}R_\phi(\phi) = \frac{1}{2}\left[R_\phi(\phi+\pi/2)+R_\phi(\phi-\pi/2)\right]`

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 1
    num_wires = 1
    par_domain = 'R'
    grad_method = 'A'


class Rot(Operation):
    r"""Rot(phi, theta, rho, wires)
    Arbitrary single qubit rotation

    .. math:: R(\phi,\theta,\omega) = RZ(\omega)RY(\theta)RZ(\phi)= \begin{bmatrix}
                e^{-i(\phi+\omega)/2}\cos(\theta/2) & -e^{i(\phi-\omega)/2}\sin(\theta/2) \\
                e^{-i(\phi-\omega)/2}\sin(\theta/2) & e^{i(\phi+\omega)/2}\cos(\theta/2)
            \end{bmatrix}.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 1
    * Gradient recipe: :math:`\frac{d}{d\phi}R(\phi) = \frac{1}{2}\left[R(\phi+\pi/2)+R(\phi-\pi/2)\right]`.
      This gradient recipe applies for each angle argument :math:`\{\phi, \theta, \omega\}`.

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        rho (float): rotation angle :math:`\omega`
        wires (Sequence[int] or int): the wire the operation acts on
    """
    num_params = 3
    num_wires = 1
    par_domain = 'R'
    grad_method = 'A'


#=============================================================================
# State preparation
#=============================================================================

class BasisState(Operation):
    r"""BasisState(n, wires)
    Prepares a single computational basis state.

    **Details:**

    * Number of wires: None (applied to the entire system)
    * Number of parameters: 1
    * Gradient recipe: None (integer parameters not supported)

    Args:
        n (array): prepares the basis state :math:`\ket{n}`, where ``n`` is an
            array of integers from the set :math:`\{0, 1\}`, i.e.,
            if ``n = np.array([0, 1, 0])``, prepares the state :math:`|010\rangle`.
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = 0
    par_domain = 'A'
    grad_method = None


class QubitStateVector(Operation):
    r"""QubitStateVector(state, wires)
    Prepare subsystems using the given ket vector in the Fock basis.

    **Details:**

    * Number of wires: None (applied to the entire system)
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        state (array[complex]): a state vector of size 2**len(wires)
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = 0
    par_domain = 'A'
    grad_method = 'F'

#=============================================================================
# Arbitrary operations
#=============================================================================


class QubitUnitary(Operation):
    r"""QubitUnitary(U, wires)
    Apply an arbitrary unitary matrix.

    **Details:**

    * Number of wires: None (applied to the entire system)
    * Number of parameters: 1
    * Gradient recipe: None (uses finite difference)

    Args:
        U (array[complex]): square unitary matrix
        wires (Sequence[int] or int): the wire(s) the operation acts on
    """
    num_params = 1
    num_wires = 0
    par_domain = 'A'
    grad_method = 'F'


all_ops = [
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    CNOT,
    CZ,
    SWAP,
    RX,
    RY,
    RZ,
    PhaseShift,
    Rot,
    BasisState,
    QubitStateVector,
    QubitUnitary
]


__all__ = [cls.__name__ for cls in all_ops]
