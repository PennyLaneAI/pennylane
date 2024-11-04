# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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

import pennylane as qml
import pennylane.numpy as np
from pennylane import math


def create_initial_state(
    # pylint: disable=unsupported-binary-operation
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
        array: The initial density matrix (tensor form) of a circuit
    """
    num_wires = len(wires)
    num_axes = (
        2 * num_wires
    )  # we initialize the density matrix as the tensor form to keep compatibility with the rest of the module
    if not prep_operation:
        state = np.zeros((2,) * num_axes, dtype=complex)
        state[(0,) * num_axes] = 1
        return math.asarray(state, like=like)

    # Here, to avoid extending the previous class defined in `StatePrepBase`, we directly call the method `state_vector`. However, this requires some levels of abstract translation between state vectors and density matrices.
    # The concise explanation is that either state vectors or density matrices are always originally just higher-rank tensors. The only diff is that state vectors are originally of rank num_wires, while density matrices are of rank 2*num_wires. Therefore, we can always define the density matrices as the same wires, appended by a 'shifted' set of wires by num_wires. This idea is also used in the wire sewing technique in the catalyst package.
    dm_wires = qml.wires.Wires(wires + [w + num_wires for w in wires])
    density_matrix = prep_operation.state_vector(wire_order=list(dm_wires))
    density_matrix = np.reshape(density_matrix, (-1,) + (2,) * num_axes)
    dtype = str(density_matrix.dtype)
    floating_single = "float32" in dtype or "complex64" in dtype
    dtype = "complex64" if floating_single else "complex128"
    dtype = "complex128" if like == "tensorflow" else dtype
    return math.cast(math.asarray(density_matrix, like=like), dtype)
