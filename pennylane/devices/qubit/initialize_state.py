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

from collections.abc import Iterable

import numpy as np
import scipy as sp

import pennylane as qml


def create_initial_state(
    wires: qml.wires.Wires | Iterable,
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
        state = np.zeros((2,) * num_wires, dtype=complex)
        state[(0,) * num_wires] = 1
        return qml.math.asarray(state, like=like)

    state_vector = prep_operation.state_vector(wire_order=list(wires))
    dtype = str(state_vector.dtype)
    floating_single = "float32" in dtype or "complex64" in dtype
    dtype = "complex64" if floating_single else "complex128"
    dtype = "complex128" if like == "tensorflow" else dtype
    # sparse matrix VIP tunnel
    if sp.sparse.issparse(state_vector):
        # currently, state_vector returns a flattened target shape.
        target_shape = [prep_operation.batch_size] if prep_operation.batch_size else []
        target_shape += [2] * len(wires)
        state_vector = state_vector.toarray()
        state_vector = qml.math.reshape(state_vector, target_shape)
    return qml.math.cast(qml.math.asarray(state_vector, like=like), dtype)
