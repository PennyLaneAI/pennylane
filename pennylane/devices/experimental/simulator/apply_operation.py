# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""appling operation to a statevector."""
from functools import singledispatch
from string import ascii_letters as ABC

import pennylane as qml

from pennylane import math

SQRT2INV = 1 / math.sqrt(2)


def _get_slice(index, axis, num_axes):
    """docstring"""
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


def apply_operation_einsum(op: qml.operation.Operator, state):
    """Apply ``matrix`` to ``state`` at ``indices``. Uses ``np.einsum`` and is more efficent at lower qubit
    numbers.

    Args:
        state (array[complex]): input state
        mat (array): matrix to multiply
        indices (Iterable[integer]): indices to apply the matrix on

    Returns:
        array[complex]: output_state
    """
    mat = op.matrix()

    total_indices = len(state.shape)
    num_indices = len(op.wires)

    state_indices = ABC[:total_indices]
    affected_indices = "".join(ABC[i] for i in op.wires)

    new_indices = ABC[total_indices : total_indices + num_indices]

    new_state_indices = state_indices
    for old, new in zip(affected_indices, new_indices):
        new_state_indices = new_state_indices.replace(old, new)

    einsum_indices = f"{new_indices}{affected_indices},{state_indices}->{new_state_indices}"

    reshaped_mat = math.reshape(mat, [2] * (num_indices * 2))

    return math.einsum(einsum_indices, reshaped_mat, state)


def apply_matrix_tensordot(op: qml.operation.Operator, state):
    """Apply ``matrix`` to ``state`` at ``indices`` using ``np.tensordot``. More efficient at higher numbers
    of indices."""
    mat = op.matrix()
    total_indices = len(state.shape)
    num_indices = len(op.wires)
    reshaped_mat = math.reshape(mat, [2] * (num_indices * 2))
    axes = (tuple(range(num_indices, 2 * num_indices)), op.wires)

    tdot = math.tensordot(reshaped_mat, state, axes=axes)

    unused_idxs = [i for i in range(total_indices) if i not in op.wires]
    perm = list(op.wires) + unused_idxs
    inv_perm = math.argsort(perm)

    return math.transpose(tdot, inv_perm)


@singledispatch
def apply_operation(op: qml.operation.Operator, state):
    """Apply ``op`` to the ``state``.

    Args:
        op (Operator)
        state (ndarray)

    Returns:
        ndarray: output state

    This function assumes that the wires of the operator correspond to indices
    of the state.  The shape of state should be ``[2]*num_wires``.

    This is a ``functools.singledispatch`` function, so additional specialized kernels
    for apply specific operations can be register like:

    .. code-block:: python

        @apply_operation.register
        def _(op: type_op_op, state):
            # custom op application method here

    """
    if len(op.wires) < 3:
        return apply_operation_einsum(op, state)
    return apply_matrix_tensordot(op, state)


@apply_operation.register
def apply_x(op: qml.PauliX, state):
    """Apply paulix operator to state"""
    return math.roll(state, 1, op.wires[0])


@apply_operation.register
def apply_y(op: qml.PauliY, state):
    """Apply pauliy operator to state."""
    state = apply_operation(qml.PauliZ(op.wires), state)
    state = apply_operation(qml.PauliX(op.wires), state)
    return 1j * state


@apply_operation.register
def apply_pauliz(op: qml.PauliZ, state):
    """Apply phase to state."""
    ndim = math.ndim(state)
    sl_0 = _get_slice(0, op.wires[0], ndim)
    sl_1 = _get_slice(1, op.wires[0], ndim)

    state1 = math.multiply(-1, state[sl_1])
    return math.stack([state[sl_0], state1], axis=op.wires[0])


@apply_operation.register
def apply_phase(op: qml.PhaseShift, state):
    """Apply PhaseShift operator to state."""
    ndim = math.ndim(state)
    sl_0 = _get_slice(0, op.wires[0], ndim)
    sl_1 = _get_slice(1, op.wires[0], ndim)

    shift = math.exp(-1j * op.data[0])
    state1 = math.multiply(shift, state[sl_1])
    return math.stack([state[sl_0], state1], axis=op.wires[0])


@apply_operation.register
def apply_hadamard(op: qml.Hadamard, state):
    """Apply hadamard gate to state"""
    statex = apply_operation(qml.PauliX(op.wires), state)
    statez = apply_operation(qml.PauliZ(op.wires), state)
    return math.multiply(SQRT2INV, statex + statez)


@apply_operation.register
def apply_cnot(op: qml.CNOT, state):
    """Apply cnot gate to state."""
    ndim = math.ndim(state)
    sl_0 = _get_slice(0, op.wires[0], ndim)
    sl_1 = _get_slice(1, op.wires[0], ndim)

    target_axes = [op.wires[1] - 1] if op.wires[1] > op.wires[0] else [op.wires[1]]
    state_x = apply_operation(qml.PauliX(target_axes), state[sl_1])
    return math.stack([state[sl_0], state_x], axis=op.wires[0])
