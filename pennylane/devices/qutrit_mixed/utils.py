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
import numbers

import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from pennylane.ops.op_math import Prod, SProd

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


def get_diagonalizing_gates(obs):
    """Returns diagonalizing gates of observable or an empty list"""
    if obs is None:
        return []
    return obs.diagonalizing_gates()


def expand_qutrit_vector(vector, original_wires, expanded_wires):
    r"""Expand a vector to more wires.

    Args:
        vector (array): :math:`QUDIT_DIM^n` vector where n = len(original_wires). (Where QUDIT_DIM=3, is dimension of
            qudit being uses)
        original_wires (Sequence[int]): original wires of vector
        expanded_wires (Union[Sequence[int], int]): expanded wires of vector, can be shuffled
            If a single int m is given, corresponds to list(range(m))

    Returns:
        array: :math:`QUDIT_DIM^m` vector where m = len(expanded_wires).
    """
    if isinstance(expanded_wires, numbers.Integral):
        expanded_wires = list(range(expanded_wires))

    N = len(original_wires)
    M = len(expanded_wires)
    D = M - N

    if not set(expanded_wires).issuperset(original_wires):
        raise ValueError("Invalid target subsystems provided in 'original_wires' argument.")

    if qml.math.shape(vector) != (QUDIT_DIM**N,):
        raise ValueError("Vector parameter must be of length QUDIT_DIM**len(original_wires)")

    dims = [QUDIT_DIM] * N
    tensor = qml.math.reshape(vector, dims)

    if D > 0:
        extra_dims = [QUDIT_DIM] * D
        ones = qml.math.ones(QUDIT_DIM**D).reshape(extra_dims)
        expanded_tensor = qml.math.tensordot(tensor, ones, axes=0)
    else:
        expanded_tensor = tensor

    wire_indices = []
    for wire in original_wires:
        wire_indices.append(expanded_wires.index(wire))

    wire_indices = np.array(wire_indices)

    # Order tensor factors according to wires
    original_indices = np.array(range(N))
    expanded_tensor = qml.math.moveaxis(
        expanded_tensor, tuple(original_indices), tuple(wire_indices)
    )

    return qml.math.reshape(expanded_tensor, QUDIT_DIM**M)


@functools.singledispatch
def get_eigvals(obs: qml.operation.Observable):
    """Gets the eigenvalues of an observable"""
    return obs.eigvals()


@get_eigvals.register
def get_prod_eigvals(obs: Prod):
    """Gets the eigenvalues of an observable if type Prod, implements get_eigvals"""
    eigvals = []
    for ops in obs.overlapping_ops:
        if len(ops) == 1:
            eigvals.append(
                expand_qutrit_vector(ops[0].eigvals(), list(ops[0].wires), list(obs.wires))
            )
        else:
            tmp_composite = obs.__class__(*ops)
            eigvals.append(
                expand_qutrit_vector(
                    tmp_composite.eigendecomposition["eigval"],
                    list(tmp_composite.wires),
                    list(obs.wires),
                )
            )
    return math.prod(math.asarray(eigvals, like=math.get_deep_interface(eigvals)), axis=0)


@get_eigvals.register
def get_s_prod_eigvals(obs: SProd):
    """Gets the eigenvalues of an observable if type Prod, implements get_eigvals"""
    base_eigs = get_eigvals(obs.base)
    if qml.math.get_interface(obs.scalar) == "torch" and obs.scalar.requires_grad:
        base_eigs = qml.math.convert_like(base_eigs, obs.scalar)
    return obs.scalar * base_eigs
