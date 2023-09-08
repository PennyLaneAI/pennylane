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

from typing import Iterable, Union

import numpy as np
import pennylane as qml


def create_initial_state(
    wires: Union[qml.wires.Wires, Iterable],
    prep_operation: qml.operation.StatePrepBase = None,
    like: str = None,
):
    r"""
    Returns an initial state, defaulting to :math:`\ket{0}` if no state-prep operator is provided.

    Args:
        wires (Union[Wires, Iterable]): The wires to be present in the initial state
        prep_operation (Optional[StatePrepBase]): An operation to prepare the initial state
        like (Optional[str]): The machine learning interface used to create the initial state.
            Defaults to None

    Returns:
        array: The initial state of a circuit
    """
    if not prep_operation:
        num_wires = len(wires)
        state = np.zeros((2,) * num_wires)
        state[(0,) * num_wires] = 1
        return qml.math.asarray(state, like=like)

    return qml.math.asarray(prep_operation.state_vector(wire_order=list(wires)), like=like)
