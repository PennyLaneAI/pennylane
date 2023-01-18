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
    """Apply ``Operator`` to ``state`` using ``matheinsum``. This is more efficent at lower qubit
    numbers.

    Args:
        op (Operator): Operator to apply to the
        state (array[complex]): input state

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


def apply_operation_tensordot(op: qml.operation.Operator, state):
    """Apply ``Operator`` to ``state`` using ``math.tensordot``. This is more efficent at higher qubit
    numbers.

    Args:
        op (Operator): Operator to apply to the
        state (array[complex]): input state

    Returns:
        array[complex]: output_state
    """
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
    """Apply and operator to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (ndarray): The starting state.

    Returns:
        ndarray: output state

    .. warning::

        ``apply_operation`` is an internal function, and thus subject to change without a deprecation cycle.

    .. warning::
        ``apply_operation`` applies no validation to its inputs.

        This function assumes that the wires of the operator correspond to indices
        of the state. See :func:`~.map_wires` to convert operations to integer wire labels.

        The shape of state should be ``[2]*num_wires``.

    This is a ``functools.singledispatch`` function, so additional specialized kernels
    for specific operations can be registered like:

    .. code-block:: python

        @apply_operation.register
        def _(op: type_op, state):
            # custom op application method here

    **Example:**

    >>> state = np.zeros((2,2))
    >>> state[0][0] = 1
    >>> state
    tensor([[1., 0.],
        [0., 0.]], requires_grad=True)
    >>> apply_operation(qml.PauliX(0), state)
    tensor([[0., 0.],
        [1., 0.]], requires_grad=True)

    """
    if len(op.wires) < 3:
        return apply_operation_einsum(op, state)
    return apply_operation_tensordot(op, state)


@apply_operation.register
def apply_x(op: qml.PauliX, state):
    """Apply paulix operator to state"""
    return math.roll(state, 1, op.wires[0])


@apply_operation.register
def apply_pauliz(op: qml.PauliZ, state):
    """Apply pauliz to state."""
    state1 = math.multiply(-1, math.take(state, 1, axis=op.wires[0]))
    return math.stack([math.take(state, 0, axis=op.wires[0]), state1], axis=op.wires[0])


@apply_operation.register
def apply_y(op: qml.PauliY, state):
    """Apply pauliy operator to state."""
    state = apply_operation(qml.PauliZ(op.wires), state)
    state = apply_operation(qml.PauliX(op.wires), state)
    return 1j * state


@apply_operation.register
def apply_phase(op: qml.PhaseShift, state):
    """Apply PhaseShift operator to state."""
    shift = math.exp(1j * op.data[0])

    state0 = math.take(state, 0, axis=op.wires[0])
    state1 = math.multiply(shift, math.take(state, 1, axis=op.wires[0]))
    return math.stack([state0, state1], axis=op.wires[0])


@apply_operation.register
def apply_hadamard(op: qml.Hadamard, state):
    """Apply hadamard gate to state"""
    statex = apply_operation(qml.PauliX(op.wires), state)
    statez = apply_operation(qml.PauliZ(op.wires), state)
    return math.multiply(SQRT2INV, statex + statez)


@apply_operation.register
def apply_cnot(op: qml.CNOT, state):
    """Apply cnot gate to state."""

    target_axes = [op.wires[1] - 1] if op.wires[1] > op.wires[0] else [op.wires[1]]

    state0 = math.take(state, 0, axis=op.wires[0])
    state_x = math.roll(math.take(state, 1, axis=op.wires[0]), 1, target_axes)
    return math.stack([state0, state_x], axis=op.wires[0])
