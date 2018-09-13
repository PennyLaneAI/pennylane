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


from openqml.operation import Operation


__all__ = [
    'Hadamard',
    'PauliX',
    'PauliY',
    'PauliZ',
    'CNOT',
    'CZ',
    'SWAP',
    'RX',
    'RY',
    'RZ',
    'PhaseShift',
    'Rot',
    'QubitStateVector',
    'QubitUnitary'
]



class Hadamard(Operation):
    r"""The Hadamard operator.
    """
    n_params = 0


class PauliX(Operation):
    r"""The Pauli X operator.
    """
    n_params = 0


class PauliY(Operation):
    r"""The Pauli Y operator.
    """
    n_params = 0


class PauliZ(Operation):
    r"""The Pauli Z operator.
    """
    n_params = 0


class CNOT(Operation):
    r"""The controlled-NOT operator.

    The first subsystem corresponds to the control qubit.
    """
    n_params = 0
    n_wires = 2


class CZ(Operation):
    r"""The controlled-Z operator.

    The first subsystem corresponds to the control qubit.
    """
    n_params = 0
    n_wires = 2


class SWAP(Operation):
    r"""The swap operator.
    """
    n_params = 0
    n_wires = 2


class RX(Operation):
    r"""The single qubit X rotation.

    .. math:: RX(\phi) = e^{-i\phi\sigma_x/2}

    Args:
        phi (float): rotation angle :math:`\phi`
    """


class RY(Operation):
    r"""The single qubit Y rotation.

    .. math:: RY(\phi) = e^{-i\phi\sigma_y/2}

    Args:
        phi (float): rotation angle :math:`\phi`
    """


class RZ(Operation):
    r"""The single qubit Z rotation.

    .. math:: RZ(\phi) = e^{-i\phi\sigma_z/2}

    Args:
        phi (float): rotation angle :math:`\phi`
    """


class PhaseShift(Operation):
    r"""Arbitrary single qubit local phase shift.

    Args:
        phi (float): phase shift :math:`\phi`
    """


class Rot(Operation):
    r"""Arbitrary single qubit rotation.

    .. math:: R(\phi,\theta,\rho) = RZ(\phi)RY(\theta)RZ(\rho)

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        rho (float): rotation angle :math:`\rho`
    """
    n_params = 3


#=============================================================================
# State preparation
#=============================================================================


class QubitStateVector(Operation):
    r"""Prepare subsystems using the given ket vector in the Fock basis.

    Args:
        state (array[complex]): a state vector of size 2**len(wires)
    """
    n_wires = 0


#=============================================================================
# Arbitrary operations
#=============================================================================


class QubitUnitary(Operation):
    r"""Apply an arbitrary unitary matrix.

    Args:
        U (array[complex]): square unitary matrix
    """
    n_wires = 0
