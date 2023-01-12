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
"""Functions to prepare a state."""

from typing import List

import pennylane as qml
from pennylane import numpy as np
from pennylane.interfaces import INTERFACE_MAP
from pennylane.operation import Operator

INTERFACE_MAP = INTERFACE_MAP.copy()
del INTERFACE_MAP[None]
del INTERFACE_MAP["auto"]


def _get_padded_wire_order(num_wires, operations):
    """
    Gets all distinct wires in ``operations``, and pads that list
    with new labels if needed.

    Args:
        num_wires (int): The required number of wires in the returned wire_order
        operations (List[Operator]): The inputted set of operations to extract wires from

    Returns:
        wire_order (List[int]): A wire order to use in computations to ensure consistency

    Raises:
        ValueError: If too many distinct wires are found in the provided list of operations
    """
    all_wires = qml.wires.Wires.all_wires([op.wires for op in operations])
    num_prep_wires = len(all_wires)
    if num_prep_wires > num_wires:
        raise ValueError(
            f"Expected no more than {num_wires} distinct wires across all prep_operations, got {all_wires}"
        )
    wire_order = all_wires.tolist()
    if num_prep_wires < num_wires:  # pad `wire_order` to match `num_wires`
        delta = num_wires - num_prep_wires
        if int_wires := [w for w in wire_order if isinstance(w, int)]:
            first_new_wire = max(int_wires) + 1
            wire_order.extend(range(first_new_wire, first_new_wire + delta))
        else:
            wire_order.extend(range(delta))
    return wire_order


def initialize_state(
    num_wires: int, prep_operations: List[Operator] = None, ml_framework: str = "autograd"
) -> np.ndarray:
    """
    Initialize a state vector given some preparation operations.

    Args:
        num_wires (int): The number of wires in the state being initialized
        prep_operations (List[Operator]): A sequence of operations to apply to prepare an initial state
        ml_framework (str): The machine learning framework to use when preparing the initial state

    Returns:
        array[complex]: The initialized state vector

    Raises:
        QuantumFunctionError: If an unknown machine learning framework is provided
        ValueError: If too many distinct wires are found in the provided list of prep operations
    """
    if ml_framework not in INTERFACE_MAP:
        raise qml.QuantumFunctionError(
            f"Unknown framework {ml_framework}. Interface must be one of {list(INTERFACE_MAP)}."
        )
    ml_framework = INTERFACE_MAP[ml_framework]
    if ml_framework == "tf":
        ml_framework = "tensorflow"

    state = np.zeros(2**num_wires, dtype="complex128")
    state[0] = 1

    if prep_operations:
        wire_order = _get_padded_wire_order(num_wires, prep_operations)
        prep_tape = qml.tape.QuantumScript(prep=prep_operations)
        prep_matrix = qml.matrix(prep_tape, wire_order=wire_order)
        state = qml.math.matmul(prep_matrix, state)

    return qml.math.reshape(qml.math.array(state, like=ml_framework), (2,) * num_wires)
