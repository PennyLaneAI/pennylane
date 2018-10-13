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
Built-in discrete quantum operations
====================================

At the moment just qubit operations.


"""

from openqml.operation import Operation



class Hadamard(Operation):
    r"""The Hadamard operator.
    """
    n_params = 0
    n_wires = 1


class PauliX(Operation):
    r"""The Pauli X operator.
    """
    n_params = 0
    n_wires = 1


class PauliY(Operation):
    r"""The Pauli Y operator.
    """
    n_params = 0
    n_wires = 1


class PauliZ(Operation):
    r"""The Pauli Z operator.
    """
    n_params = 0
    n_wires = 1


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
    n_params = 1
    n_wires = 1


class RY(Operation):
    r"""The single qubit Y rotation.

    .. math:: RY(\phi) = e^{-i\phi\sigma_y/2}

    Args:
        phi (float): rotation angle :math:`\phi`
    """
    n_params = 1
    n_wires = 1


class RZ(Operation):
    r"""The single qubit Z rotation.

    .. math:: RZ(\phi) = e^{-i\phi\sigma_z/2}

    Args:
        phi (float): rotation angle :math:`\phi`
    """
    n_params = 1
    n_wires = 1


class PhaseShift(Operation):
    r"""Arbitrary single qubit local phase shift.

    Args:
        phi (float): phase shift :math:`\phi`
    """
    n_params = 1
    n_wires = 1


class Rot(Operation):
    r"""Arbitrary single qubit rotation.

    .. math:: R(\phi,\theta,\rho) = RZ(\phi)RY(\theta)RZ(\rho)

    Args:
        phi (float): rotation angle :math:`\phi`
        theta (float): rotation angle :math:`\theta`
        rho (float): rotation angle :math:`\rho`
    """
    n_params = 3
    n_wires = 1


#=============================================================================
# State preparation
#=============================================================================

class BasisState(Operation):
    r"""Prepares a single computational basis state.

    Args:
        n (int): prepares the state :math:`\ket{n}`
    """
    n_wires = 0
    par_domain = 'N'
    grad_method = None


class QubitStateVector(Operation):
    r"""Prepare subsystems using the given ket vector in the Fock basis.

    Args:
        state (array[complex]): a state vector of size 2**len(wires)
    """
    n_wires = 0
    par_domain = 'A'
    grad_method = 'F'

#=============================================================================
# Arbitrary operations
#=============================================================================


class QubitUnitary(Operation):
    r"""Apply an arbitrary unitary matrix.

    Args:
        U (array[complex]): square unitary matrix
    """
    n_wires = 0
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
