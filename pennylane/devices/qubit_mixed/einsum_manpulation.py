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

alphabet_array = math.array(list(alphabet))


def get_einsum_mapping(
    op: qml.operation.Operator,
    state,
    is_state_batched: bool = False,
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
    num_wires = int((len(math.shape(state)) - is_state_batched) / 2)
    rho_dim = 2 * num_wires

    # Step 1: The first rho_dim indices are the original state indices
    state_indices = alphabet[:rho_dim]

    # Step 2: construct the indices for the Kraus operators. Note that the indices here are not necessarily consecutive any more therefore we first map them into list then pick the corresponding letters

    # row indices of the quantum state affected by this operation
    row_wires_list = op.wires.tolist()
    row_indices = "".join(alphabet_array[row_wires_list].tolist())

    # column indices are shifted by the number of wires
    col_wires_list = [w + num_wires for w in row_wires_list]
    col_indices = "".join(alphabet_array[col_wires_list].tolist())

    # Step 3: contruct the new indices for the quantum state after the operation
    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[rho_dim : rho_dim + num_ch_wires]
    new_col_indices = alphabet[rho_dim + num_ch_wires : rho_dim + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[rho_dim + 2 * num_ch_wires : rho_dim + 2 * num_ch_wires + 1]

    # apply mapping function
    return _map_indices_apply_channel(
        state_indices=state_indices,
        kraus_index=kraus_index,
        row_indices=row_indices,
        new_row_indices=new_row_indices,
        col_indices=col_indices,
        new_col_indices=new_col_indices,
    )


# pylint: disable=too-many-arguments
def _map_indices_apply_channel(
    *, state_indices, kraus_index, new_row_indices, row_indices, new_col_indices, col_indices
):
    """Map indices to einsum string
    Args:
        **kwargs (dict): Stores indices calculated in `get_einsum_mapping`

    Returns:
        String of einsum indices to complete einsum calculations
    """
    op_1_indices = f"{kraus_index}{new_row_indices}{row_indices}"
    op_2_indices = f"{kraus_index}{col_indices}{new_col_indices}"

    old_indices = col_indices + row_indices
    new_indices = new_col_indices + new_row_indices

    new_state_indices = functools.reduce(
        lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
        zip(old_indices, new_indices),
        state_indices,
    )
    # index mapping for einsum, e.g., '...iga,...abcdef,...idh->...gbchef'
    return f"...{op_1_indices},...{state_indices},...{op_2_indices}->...{new_state_indices}"
