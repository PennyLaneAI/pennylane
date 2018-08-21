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
"""This module contains a beamsplitter Operation"""
from .operation import Operation


class Hadamard(Operation):
    r"""The Hadamard operator.

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('Hadamard', [], wires)


class PauliX(Operation):
    r"""The Pauli X operator.

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('PauliX', [], wires)


class PauliY(Operation):
    r"""The Pauli Y operator.

    Args:
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('PauliY', [], wires)


class PauliZ(Operation):
    r"""The Pauli Z operator.

    Args:
        wires (int]): the subsystem the Operation acts on.
    """
    def __init__(self, wires):
        super().__init__('PauliZ', [], wires)


class CNOT(Operation):
    r"""The controlled-NOT operator.

    Args:
        wires (seq[int]): the two subsystems the CNOT acts on.
            The first subsystem corresponds to the control qubit.
    """
    def __init__(self, wires):
        super().__init__('CNOT', [], wires)


class CZ(Operation):
    r"""The controlled-Z operator.

    Args:
        wires (seq[int]): the two subsystems the CNOT acts on.
            The first subsystem corresponds to the control qubit.
    """
    def __init__(self, wires):
        super().__init__('CZ', [], wires)


class SWAP(Operation):
    r"""The swap operator.

    Args:
        wires (seq[int]): the two subsystems the CNOT acts on.
            The first subsystem corresponds to the control qubit.
    """
    def __init__(self, wires):
        super().__init__('SWAP', [], wires)


class RX(Operation):
    r"""The single qubit X rotation.

    .. math:: RX(\phi) = e^{-i\phi\sigma_x/2}

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, phi, wires):
        super().__init__('RX', [phi], wires)


class RY(Operation):
    r"""The single qubit Y rotation.

    .. math:: RY(\phi) = e^{-i\phi\sigma_y/2}

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, phi, wires):
        super().__init__('RY', [phi], wires)


class RZ(Operation):
    r"""The single qubit Z rotation.

    .. math:: RZ(\phi) = e^{-i\phi\sigma_z/2}

    Args:
        phi (float): rotation angle :math:`\phi`
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, phi, wires):
        super().__init__('RZ', [phi], wires)


class Rot(Operation):
    r"""Arbitrary single qubit rotation.

    .. math:: R(\phi,\theta,\rho) = RZ(\phi)RY(\theta)RZ(\rho)

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        rho (float): rotation angle :math:`\rho`
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, phi, theta, rho, wires):
        super().__init__('Rot', [phi, theta, rho], wires)


class PhaseShift(Operation):
    r"""Arbitrary single qubit local phase shift.

    Args:
        phi (float): phase shift :math:`\phi`
        wires (int): the subsystem the Operation acts on.
    """
    def __init__(self, phi, wires):
        super().__init__('PhaseShift', [phi], wires)


#=============================================================================
# State preparation
#=============================================================================


class QubitStateVector(Operation):
    r"""Prepare subsystems using the given ket vector in the Fock basis.

    Args:
        state (array): a state vector of size 2**wires.
        wires (int or seq[int]): subsystem(s) the Operation acts on.
    """
    def __init__(self, state, wires):
        super().__init__('QubitStateVector', [state], wires)


#=============================================================================
# Arbitrary operations
#=============================================================================


class QubitUnitary(Operation):
    r"""Apply an arbitrary unitary matrix.

    Args:
        U (array): square unitary matrix.
        wires (int or seq[int]): subsystem(s) the Operation acts on.
    """
    def __init__(self, U, wires):
        super().__init__('QubitUnitary', [U], wires)
