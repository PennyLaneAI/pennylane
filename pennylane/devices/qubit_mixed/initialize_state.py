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
from typing import Union

import pennylane as qml
import pennylane.math as math
import pennylane.numpy as np


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
        array: The initial density matrix (tensor form) of a circuit
    """
    if not prep_operation:
        num_wires = len(wires)
        num_axes = (
            2 * num_wires
        )  # we initialize the density matrix as the tensor form to keep compatiblity with the rest of the module
        state = np.zeros((2,) * num_axes, dtype=complex)
        state[(0,) * num_axes] = 1
        return math.asarray(state, like=like)

    density_matrix = prep_operation.density_matrix(wire_order=list(wires))
    dtype = str(density_matrix.dtype)
    floating_single = "float32" in dtype or "complex64" in dtype
    dtype = "complex64" if floating_single else "complex128"
    dtype = "complex128" if like == "tensorflow" else dtype
    return math.cast(math.asarray(density_matrix, like=like), dtype)
