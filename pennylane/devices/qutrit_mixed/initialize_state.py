# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to prepare a qutrit mixed state."""

from typing import Iterable, Union
import pennylane as qml
from pennylane.operation import StatePrepBase
from .utils import QUDIT_DIM


def create_initial_state(
    wires: Union[qml.wires.Wires, Iterable],
    prep_operation: StatePrepBase = None,
    like: str = None,
):
    r"""
    Returns an initial state, defaulting to :math:`\ket{0}\bra{0}` if no state-prep operator is provided.

    Args:
        wires (Union[Wires, Iterable]): The wires to be present in the initial state
        prep_operation (Optional[StatePrepBase]): An operation to prepare the initial state
        like (Optional[str]): The machine learning interface used to create the initial state.
            Defaults to None

    Returns:
        array: The initial state of a circuit
    """
    num_wires = len(wires)

    if not prep_operation:
        rho = _create_basis_state(num_wires, 0)

    else:
        rho = _apply_state_vector(prep_operation.state_vector(wire_order=wires), num_wires)

    # TODO: add instance for prep_operations as added

    return qml.math.asarray(rho, like=like)


def _apply_state_vector(state, num_wires):  # function is easy to abstract for qudit
    """Initialize the internal state in a specified pure state.

    Args:
        state (array[complex]): normalized input state of length
            ``QUDIT_DIM**num_wires``, where ``QUDIT_DIM`` is the dimension of the system.
        num_wires (int): number of wires that get initialized in the state

    Returns:
        array[complex]: complex array of shape ``[QUDIT_DIM] * (2 * num_wires)``
        representing the density matrix of this state, where ``QUDIT_DIM`` is
        the dimension of the system.
    """

    # Initialize the entire set of wires with the state
    rho = qml.math.outer(state, qml.math.conj(state))
    return qml.math.reshape(rho, [QUDIT_DIM] * 2 * num_wires)


def _create_basis_state(num_wires, index):  # function is easy to abstract for qudit
    """Return the density matrix representing a computational basis state over all wires.

    Args:
        num_wires (int): number of wires to initialize
        index (int): integer representing the computational basis state.

    Returns:
        array[complex]: complex array of shape ``[QUDIT_DIM] * (2 * num_wires)``
        representing the density matrix of the basis state, where ``QUDIT_DIM`` is
        the dimension of the system.
    """
    rho = qml.math.zeros((QUDIT_DIM**num_wires, QUDIT_DIM**num_wires))
    rho[index, index] = 1
    return qml.math.reshape(rho, [QUDIT_DIM] * (2 * num_wires))
