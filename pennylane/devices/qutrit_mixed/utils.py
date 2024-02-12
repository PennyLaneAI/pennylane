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
"""Functions and variables to be utilized by qutrit mixed state simulator."""
import functools
from string import ascii_letters as alphabet
import pennylane as qml
from pennylane import math
from pennylane import numpy as np

alphabet_array = np.array(list(alphabet))
QUDIT_DIM = 3  # specifies qudit dimension


def get_einsum_mapping(
    op: qml.operation.Operator, state, map_indices, is_state_batched: bool = False
):
    r"""Finds the indices for einsum to apply kraus operators to a mixed state

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        map_indices (function): Maps the calculated indices to an einsum indices string
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        str: Indices mapping that defines the einsum
    """
    num_ch_wires = len(op.wires)
    num_wires = int((len(qml.math.shape(state)) - is_state_batched) / 2)
    rho_dim = 2 * num_wires

    # Tensor indices of the state. For each qutrit, need an index for rows *and* columns
    state_indices = alphabet[:rho_dim]

    # row indices of the quantum state affected by this operation
    row_wires_list = op.wires.tolist()
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[rho_dim : rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires : rho_dim + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires : rho_dim + 2 * num_ch_wires + 1]

    # apply mapping function
    return map_indices(
        state_indices=state_indices,
        kraus_index=kraus_index,
        row_indices=row_indices,
        new_row_indices=new_row_indices,
        col_indices=col_indices,
        new_col_indices=new_col_indices,
    )


def reshape_state_as_matrix(state, num_wires):
    """Given a non-flat, potentially batched state, flatten it to square matrix or matrices if batched.

    Args:
        state (TensorLike): A state that needs to be reshaped to a square matrix or matrices if batched
        num_wires (int): The number of wires the state represents

    Returns:
        Tensorlike: A reshaped, square state, with an extra batch dimension if necessary
    """
    dim = QUDIT_DIM**num_wires
    batch_size = math.get_batch_size(state, ((QUDIT_DIM,) * (num_wires * 2)), dim**2)
    shape = (batch_size, dim, dim) if batch_size is not None else (dim, dim)
    return math.reshape(state, shape)


def get_num_wires(state, is_state_batched: bool = False):
    """Finds the number of wires associated with a state

    Args:
        state (TensorLike): A device compatible state that may or may not be batched
        is_state_batched (int): Boolean representing whether the state is batched or not

    Returns:
        int: Number of wires associated with state
    """
    len_row_plus_col = len(math.shape(state)) - is_state_batched
    return int(len_row_plus_col / 2)


def get_new_state_einsum_indices(old_indices, new_indices, state_indices):
    """Retrieves the einsum indices string for the new state

    Args:
        old_indices (str): indices that are summed
        new_indices (str): indices that must be replaced with sums
        state_indices (str): indices of the original state

    Returns:
        str: The einsum indices of the new state
    """
    return functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(old_indices, new_indices),
        state_indices,
    )
