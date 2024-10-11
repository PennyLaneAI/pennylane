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
"""Functions to apply operations to a qubit mixed state."""
# pylint: disable=unused-argument

from functools import singledispatch
from string import ascii_letters as alphabet

import pennylane as qml
from pennylane import math
from pennylane.devices.qubit_mixed import QUDIT_DIM
from pennylane.measurements import Shots
from pennylane.operation import Channel
from pennylane.ops.qubit.attributes import diagonal_in_z_basis

alphabet_array = math.array(list(alphabet))

SQRT2INV = 1 / math.sqrt(2)


def apply_operation(
    op: qml.operation.Operator,
    state,  # density matrix
    is_state_batched: bool = False,
    debugger=None,
    postselect_mode=None,
    rng=None,
    prng_key=None,
    tape_shots=Shots(None),
):
    """Apply an operation to a given state."""

    num_op_wires = len(op.wires)
    if isinstance(op, Channel):
        kraus = op.kraus_matrices()
    else:
        mat = op.matrix
        kraus = [mat]
        # Shape kraus operators
        kraus_shape = [len(kraus)] + [QUDIT_DIM] * num_op_wires * 2

        mat = op.matrix()
        dim = QUDIT_DIM**num_op_wires
        batch_size = math.get_batch_size(mat, (dim, dim), dim**2)
        if batch_size is not None:
            # Add broadcasting dimension to shape
            kraus_shape = [batch_size] + kraus_shape
            if op.batch_size is None:
                op._batch_size = batch_size

    interface = math.get_interface(state, *kraus)
    if (num_op_wires > 2 and interface in {"autograd", "numpy"}) or num_op_wires > 7:
        return _apply_channel_tensordot(
            kraus, state, is_state_batched, debugger, postselect_mode, rng, prng_key, tape_shots
        )
    return _apply_channel_einsum(
        kraus, state, is_state_batched, debugger, postselect_mode, rng, prng_key, tape_shots
    )


def _apply_channel_einsum(
    matrices,
    state,
    is_state_batched: bool = False,
    debugger=None,
    postselect_mode=None,
    rng=None,
    prng_key=None,
    tape_shots=Shots(None),
):
    print()
    print(f"Type of matrices: {type(matrices)}")
    print(f"Length of matrices: {len(matrices)}")
    print(f"Type of first element in matrices: {type(matrices[0])}")

    num_wires = int(math.log2(state.shape[0]) // 2)

    # Convert methods to actual matrices
    matrix_data = [m() if callable(m) else m for m in matrices]

    print(f"Shape of first matrix after conversion: {matrix_data[0].shape}")

    num_ch_wires = int(math.log2(matrix_data[0].shape[0]))

    # Compute K^\dagger, needed for the transformation K \rho K^\dagger
    matrices_dagger = [math.conj(math.transpose(k)) for k in matrix_data]

    matrices = math.stack(matrix_data)
    matrices_dagger = math.stack(matrices_dagger)

    # Shape kraus operators
    kraus_shape = [len(matrices)] + [QUDIT_DIM] * num_ch_wires * 2
    matrices = math.reshape(matrices, kraus_shape)
    matrices_dagger = math.reshape(matrices_dagger, kraus_shape)

    # Tensor indices of the state. For each qubit, need an index for rows *and* columns
    state_indices = alphabet[: 2 * num_wires]

    # row indices of the quantum state affected by this operation
    row_indices = alphabet[:num_ch_wires]

    # column indices are shifted by the number of wires
    col_indices = alphabet[num_wires : num_wires + num_ch_wires]

    # indices in einsum must be replaced with new ones
    new_row_indices = alphabet[2 * num_wires : 2 * num_wires + num_ch_wires]
    new_col_indices = alphabet[2 * num_wires + num_ch_wires : 2 * num_wires + 2 * num_ch_wires]

    # index for summation over Kraus operators
    kraus_index = alphabet[2 * num_wires + 2 * num_ch_wires]

    # new state indices replace row and column indices with new ones
    new_state_indices = get_new_state_einsum_indices(
        state_indices, col_indices + row_indices, new_col_indices + new_row_indices
    )

    # index mapping for einsum, e.g., 'iga,abcdef,idh->gbchef'
    einsum_indices = (
        f"{kraus_index}{new_row_indices}{row_indices}, {state_indices},"
        f"{kraus_index}{col_indices}{new_col_indices}->{new_state_indices}"
    )

    return math.einsum(einsum_indices, matrices, state, matrices_dagger)


def _apply_channel_tensordot(
    matrices,
    state,
    is_state_batched: bool = False,
    debugger=None,
    postselect_mode=None,
    rng=None,
    prng_key=None,
    tape_shots=Shots(None),
):
    num_wires = int(math.log2(state.shape[0]) // 2)
    num_ch_wires = int(math.log2(matrices[0].shape[0]))

    # Shape kraus operators
    kraus_shape = [QUDIT_DIM] * (num_ch_wires * 2)
    matrices = [math.reshape(k, kraus_shape) for k in matrices]

    channel_col_ids = list(range(num_ch_wires, 2 * num_ch_wires))
    axes_left = [channel_col_ids, list(range(num_ch_wires))]
    # Use column indices instead of rows to incorporate transposition of K^\dagger
    axes_right = [list(range(num_wires, num_wires + num_ch_wires)), channel_col_ids]

    # Apply the Kraus operators, and sum over all Kraus operators afterwards
    def _conjugate_state_with(k):
        """Perform the double tensor product k @ state @ k.conj()."""
        return math.tensordot(math.tensordot(k, state, axes_left), math.conj(k), axes_right)

    if len(matrices) == 1:
        _state = _conjugate_state_with(matrices[0])
    else:
        _state = math.sum(math.stack([_conjugate_state_with(k) for k in matrices]), axis=0)

    # Permute the affected axes to their destination places.
    source_left = list(range(num_ch_wires))
    dest_left = list(range(num_ch_wires))
    source_right = list(range(-num_ch_wires, 0))
    dest_right = list(range(num_wires, num_wires + num_ch_wires))
    return math.moveaxis(_state, source_left + source_right, dest_left + dest_right)


def apply_snapshot(
    op: qml.Snapshot, state, is_state_batched: bool = False, debugger=None, **execution_kwargs
):
    """Take a snapshot of the mixed state"""
    if debugger and debugger.active:
        measurement = op.hyperparameters["measurement"]

        shots = execution_kwargs.get("tape_shots")

        if isinstance(measurement, qml.measurements.StateMP) or not shots:
            snapshot = qml.devices.qubit_mixed.measure(measurement, state, is_state_batched)
        else:
            snapshot = qml.devices.qubit_mixed.measure_with_samples(
                measurement,
                state,
                shots,
                is_state_batched,
                execution_kwargs.get("rng"),
                execution_kwargs.get("prng_key"),
            )

        if op.tag:
            debugger.snapshots[op.tag] = snapshot
        else:
            debugger.snapshots[len(debugger.snapshots)] = snapshot

    return state


def apply_identity(op: qml.Identity, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state


def _map_indices_apply_channel(**kwargs):
    """Map indices to einsum string
    Args:
        **kwargs (dict): Stores indices calculated in `get_einsum_mapping`

    Returns:
        String of einsum indices to complete einsum calculations
    """
    op_1_indices = f"{kwargs['kraus_index']}{kwargs['new_row_indices']}{kwargs['row_indices']}"
    op_2_indices = f"{kwargs['kraus_index']}{kwargs['col_indices']}{kwargs['new_col_indices']}"

    new_state_indices = get_new_state_einsum_indices(
        old_indices=kwargs["col_indices"] + kwargs["row_indices"],
        new_indices=kwargs["new_col_indices"] + kwargs["new_row_indices"],
        state_indices=kwargs["state_indices"],
    )
    # index mapping for einsum, e.g., '...iga,...abcdef,...idh->...gbchef'
    return (
        f"...{op_1_indices},...{kwargs['state_indices']},...{op_2_indices}->...{new_state_indices}"
    )
