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
"""Functions to apply an operation to a state vector."""
from functools import singledispatch
from string import ascii_letters as alphabet

import pennylane as qml

from pennylane import math

SQRT2INV = 1 / math.sqrt(2)

EINSUM_OP_WIRECOUNT_PERF_THRESHOLD = 3
EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD = 13


def _get_slice(index, axis, num_axes):
    """Allows slicing along an arbitrary axis of an array or tensor.

    Args:
        index (int): the index to access
        axis (int): the axis to slice into
        num_axes (int): total number of axes

    Returns:
        tuple[slice or int]: a tuple that can be used to slice into an array or tensor

    **Example:**

    Accessing the 2 index along axis 1 of a 3-axis array:

    >>> sl = _get_slice(2, 1, 3)
    >>> sl
    (slice(None, None, None), 2, slice(None, None, None))
    >>> a = np.arange(27).reshape((3, 3, 3))
    >>> a[sl]
    array([[ 6,  7,  8],
           [15, 16, 17],
           [24, 25, 26]])
    """
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


def apply_operation_einsum(op: qml.operation.Operator, state):
    """Apply ``Operator`` to ``state`` using ``einsum``. This is more efficent at lower qubit
    numbers.

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state

    Returns:
        array[complex]: output_state
    """
    mat = op.matrix()

    total_indices = len(state.shape)
    num_indices = len(op.wires)

    state_indices = alphabet[:total_indices]
    affected_indices = "".join(alphabet[i] for i in op.wires)

    new_indices = alphabet[total_indices : total_indices + num_indices]

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
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state

    Returns:
        array[complex]: output_state
    """
    mat = op.matrix()
    total_indices = len(state.shape)
    num_indices = len(op.wires)
    reshaped_mat = math.reshape(mat, [2] * (num_indices * 2))
    axes = (tuple(range(num_indices, 2 * num_indices)), tuple(op.wires))

    tdot = math.tensordot(reshaped_mat, state, axes=axes)

    # tensordot causes the axes given in `wires` to end up in the first positions
    # of the resulting tensor. This corresponds to a (partial) transpose of
    # the correct output state
    # We'll need to invert this permutation to put the indices in the correct place
    unused_idxs = [i for i in range(total_indices) if i not in op.wires]
    perm = list(op.wires) + unused_idxs
    inv_perm = math.argsort(perm)

    return math.transpose(tdot, inv_perm)


@singledispatch
def apply_operation(op: qml.operation.Operator, state):
    """Apply and operator to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (tensor_like): The starting state.

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
    if (len(op.wires) < EINSUM_OP_WIRECOUNT_PERF_THRESHOLD) and (
        math.ndim(state) < EINSUM_STATE_WIRECOUNT_PERF_THRESHOLD
    ):
        return apply_operation_einsum(op, state)
    return apply_operation_tensordot(op, state)


@apply_operation.register
def apply_paulix(op: qml.PauliX, state):
    """Apply :class:`pennylane.PauliX` operator to the quantum state"""
    return math.roll(state, 1, op.wires[0])


@apply_operation.register
def apply_pauliz(op: qml.PauliZ, state):
    """Apply pauliz to state."""
    n_wires = math.ndim(state)
    sl_0 = _get_slice(0, op.wires[0], n_wires)
    sl_1 = _get_slice(1, op.wires[0], n_wires)

    state1 = math.multiply(state[sl_1], -1)
    return math.stack([state[sl_0], state1], axis=op.wires[0])


@apply_operation.register
def apply_phase(op: qml.PhaseShift, state):
    """Apply PhaseShift operator to state."""
    shift = math.exp(math.multiply(1j, op.data[0]))

    n_wires = math.ndim(state)
    sl_0 = _get_slice(0, op.wires[0], n_wires)
    sl_1 = _get_slice(1, op.wires[0], n_wires)

    state1 = math.multiply(shift, state[sl_1])
    return math.stack([state[sl_0], state1], axis=op.wires[0])


@apply_operation.register
def apply_cnot(op: qml.CNOT, state):
    """Apply cnot gate to state."""
    target_axes = (op.wires[1] - 1) if op.wires[1] > op.wires[0] else (op.wires[1])

    n_wires = math.ndim(state)
    sl_0 = _get_slice(0, op.wires[0], n_wires)
    sl_1 = _get_slice(1, op.wires[0], n_wires)
    state_x = math.roll(state[sl_1], 1, target_axes)
    return math.stack([state[sl_0], state_x], axis=op.wires[0])
