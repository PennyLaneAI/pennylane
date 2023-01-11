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
    num_wires: int, prep_operations: List[Operator] = None, framework: str = "autograd"
) -> np.ndarray:
    """
    Initialize a state vector given some preparation operations.

    Args:
        num_wires (int): The number of wires in the state being initialized
        prep_operations (List[Operator]): A sequence of operations to apply to prepare an initial state
        framework (str): The machine learning framework to use when preparing the initial state

    Returns:
        array[complex]: The initialized state vector

    Raises:
        QuantumFunctionError: If an unknown machine learning framework is provided
        ValueError: If too many distinct wires are found in the provided list of prep operations
    """
    if framework not in INTERFACE_MAP:
        raise qml.QuantumFunctionError(
            f"Unknown framework {framework}. Interface must be one of {list(INTERFACE_MAP)}."
        )
    framework = INTERFACE_MAP[framework]
    if framework == "tf":
        framework = "tensorflow"

    shape = (2,) * num_wires
    state = qml.math.zeros(shape, dtype="complex128", like=framework)

    if prep_operations:
        wire_order = _get_padded_wire_order(num_wires, prep_operations)
        for op in prep_operations:
            state = qml.math.matmul(op.matrix(wire_order=wire_order), state, like=framework)

    return qml.math.array(state, like=framework)
