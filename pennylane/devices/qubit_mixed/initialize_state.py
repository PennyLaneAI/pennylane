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
import pennylane.numpy as np
from pennylane import math


def create_initial_state(
    # pylint: disable=unsupported-binary-operation
    wires: Union[qml.wires.Wires, Iterable],
    prep_operation: Union[qml.operation.StatePrepBase, qml.QubitDensityMatrix] = None,
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

    if isinstance(prep_operation, qml.QubitDensityMatrix):
        density_matrix = prep_operation.data

    else:  # Use pure state prep
        pure_state = prep_operation.state_vector(wire_order=list(wires))
        batch_size = math.get_batch_size(
            pure_state, expected_shape=[], expected_size=2**num_wires
        )  # don't assume the expected shape to be fixed
        if batch_size == 1:
            density_matrix = np.outer(pure_state, np.conj(pure_state))
        else:
            density_matrix = math.stack([np.outer(s, np.conj(s)) for s in pure_state])
    return _post_process(density_matrix, num_axes, like)


def _post_process(density_matrix, num_axes, like):
    r"""
    This post-processor is necessary to ensure that the density matrix is in
    the correct format, i.e. the original tensor form, instead of the pure
    matrix form, as requested by all the other more fundamental chore functions
    in the module (again from some legacy code).
    """
    density_matrix = np.reshape(density_matrix, (-1,) + (2,) * num_axes)
    dtype = str(density_matrix.dtype)
    floating_single = "float32" in dtype or "complex64" in dtype
    dtype = "complex64" if floating_single else "complex128"
    dtype = "complex128" if like == "tensorflow" else dtype
    if density_matrix.shape[0] == 1:  # non batch
        density_matrix = np.reshape(density_matrix, (2,) * num_axes)
    return math.cast(math.asarray(density_matrix, like=like), dtype)
