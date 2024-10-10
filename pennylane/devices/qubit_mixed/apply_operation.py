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
from pennylane.measurements import Shots
from pennylane.operation import Channel
from pennylane.ops.qubit.attributes import diagonal_in_z_basis

from .utils import QUDIT_DIM, get_einsum_mapping, get_new_state_einsum_indices

alphabet_array = math.array(list(alphabet))

SQRT2INV = 1 / math.sqrt(2)


def _get_kraus(operation):
    """Return the Kraus operators representing the operation."""
    if operation in diagonal_in_z_basis:
        return operation.eigvals()

    if isinstance(operation, Channel):
        return operation.kraus_matrices()

    return [operation.matrix()]


def apply_operation_einsum(op: qml.operation.Operator, state, is_state_batched: bool = False):
    r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
    quantum state. For a unitary gate, there is a single Kraus operator."""
    einsum_indices = get_einsum_mapping(op, state, _map_indices_apply_channel, is_state_batched)

    num_ch_wires = len(op.wires)

    if isinstance(op, Channel):
        kraus = op.kraus_matrices()
    else:
        kraus = [op.matrix()]

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * num_ch_wires * 2
    if not isinstance(op, Channel):
        mat = op.matrix()
        dim = QUDIT_DIM**num_ch_wires
        batch_size = math.get_batch_size(mat, (dim, dim), dim**2)
        if batch_size is not None:
            # Add broadcasting dimension to shape
            kraus_shape = [batch_size] + kraus_shape
            if op.batch_size is None:
                op._batch_size = batch_size  # pylint:disable=protected-access

    kraus = math.stack(kraus)
    kraus_transpose = math.stack(math.moveaxis(kraus, source=-1, destination=-2))
    kraus_dagger = math.conj(kraus_transpose)

    kraus = math.cast(math.reshape(kraus, kraus_shape), complex)
    kraus_dagger = math.cast(math.reshape(kraus_dagger, kraus_shape), complex)

    return math.einsum(einsum_indices, kraus, state, kraus_dagger)


@singledispatch
def apply_operation(
    op: qml.operation.Operator,
    state,
    is_state_batched: bool = False,
    debugger=None,
    postselect_mode=None,
    rng=None,
    prng_key=None,
    tape_shots=Shots(None),
):
    """Apply an operation to a given state."""

    num_op_wires = len(op.wires)
    matrices = _get_kraus(op)
    interface = qml.math.get_interface(state, *matrices)
    if (num_op_wires > 2 and interface in {"autograd", "numpy"}) or num_op_wires > 7:
        return _apply_channel_tensordot(matrices, state, wires)
    return _apply_channel_einsum(matrices, state, wires)


def _apply_operation_default(op, state, is_state_batched, debugger):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches.
    """
    return apply_operation_einsum(op, state, is_state_batched=is_state_batched)
    # TODO add tensordot and benchmark for performance


@apply_operation.register
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


@apply_operation.register
def apply_identity(op: qml.Identity, state, is_state_batched: bool = False, debugger=None, **_):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state


# TODO add special case speedups


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
