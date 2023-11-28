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
import numpy as np
import pennylane as qml
from pennylane import (
    QutritBasisState,
)
from pennylane.operation import (
    StatePrepBase,
    Operation,
)
from pennylane.wires import Wires

qudit_dim = 3  # specifies qudit dimension


def create_initial_state(
    wires: Union[qml.wires.Wires, Iterable],
    prep_operation: Operation = None,
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
    if isinstance(wires, Wires):
        wire_array = wires.toarray()
    else:
        wire_array = np.array(wires)
    num_wires = len(wire_array)

    if not prep_operation:
        rho = _create_basis_state(num_wires, 0)

    elif isinstance(prep_operation, QutritBasisState):
        rho = _apply_basis_state(prep_operation.parameters[0], wire_array)

    elif isinstance(prep_operation, StatePrepBase):
        rho = _apply_state_vector(
            prep_operation.state_vector(wire_order=list(wire_array)), num_wires
        )

    # TODO: add instance for prep_operations as added

    return qml.math.asarray(rho, like=like)


def _apply_state_vector(state, num_wires):  # function is easy to abstract for qudit
    """Initialize the internal state in a specified pure state.

    Args:
        state (array[complex]): normalized input state of length
            ``3**len(wires)``
        num_wires (array[int]): wires that get initialized in the state
    """

    # Initialize the entire wires with the state
    rho = qml.math.outer(state, qml.math.conj(state))
    return qml.math.reshape(rho, [3] * 2 * num_wires)


def _apply_basis_state(state, wires):  # function is easy to abstract for qudit
    """Returns initial state for a specified computational basis state.

    Args:
        state (array[int]): computational basis state of shape ``(wires,)``
            consisting of 0s and 1s.
        wires (array[int]): wires that the provided computational state should be initialized on

    """
    num_wires = len(wires)

    # get computational basis state number
    basis_states = qudit_dim ** (num_wires - 1 - wires)
    print(qml.math.dot(state, basis_states))
    num = int(qml.math.dot(state, basis_states))

    return _create_basis_state(num_wires, num)


def _create_basis_state(num_wires, index):  # function is easy to abstract for qudit
    """Return the density matrix representing a computational basis state over all wires.

    Args:
        index (int): integer representing the computational basis state.

    Returns:
        array[complex]: complex array of shape ``[3] * (2 * num_wires)``
        representing the density matrix of the basis state.
    """
    rho = np.zeros((3**num_wires, 3**num_wires))
    rho[index, index] = 1
    return qml.math.reshape(rho, [qudit_dim] * (2 * num_wires))
