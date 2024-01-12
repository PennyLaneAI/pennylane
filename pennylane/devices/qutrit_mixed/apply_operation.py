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
"""Functions to apply operations to a qutrit mixed state."""
# pylint: disable=unused-argument

from functools import singledispatch
from string import ascii_letters as alphabet
import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from pennylane.operation import Channel
from .utils import QUDIT_DIM, get_einsum_indices

alphabet_array = np.array(list(alphabet))


def apply_operation_einsum(op: qml.operation.Operator, state, is_state_batched: bool = False):
    r"""Apply a quantum channel specified by a list of Kraus operators to subsystems of the
    quantum state. For a unitary gate, there is a single Kraus operator.

    Args:
        op (Operator): Operator to apply to the quantum state
        state (array[complex]): Input quantum state
        is_state_batched (bool): Boolean representing whether the state is batched or not

    Returns:
        array[complex]: output_state
    """
    indices = get_einsum_indices(op, state, is_state_batched)
    # index mapping for einsum, e.g., '...iga,...abcdef,...idh->...gbchef'
    einsum_indices = (
        f"...{indices['op1']},...{indices['state']},...{indices['op2']}->...{indices['new_state']}"
    )

    num_ch_wires = len(op.wires)
    kraus = _get_kraus(op)

    # Shape kraus operators
    kraus_shape = [len(kraus)] + [QUDIT_DIM] * num_ch_wires * 2
    # Compute K^T, will be list of lists if broadcasting
    kraus_transpose = []
    if not isinstance(op, Channel):
        # TODO Channels broadcasting doesn't seem to be implemented for qubits, should they be for qutrit?
        mat = op.matrix()
        dim = QUDIT_DIM**num_ch_wires
        batch_size = math.get_batch_size(mat, (dim, dim), dim**2)
        if batch_size is not None:
            # Add broadcasting dimension to shape
            kraus_shape = [batch_size] + kraus_shape
            if op.batch_size is None:
                op._batch_size = batch_size  # pylint:disable=protected-access
            for op_mats in kraus:
                kraus_transpose.append([math.transpose(k) for k in op_mats])
    if not kraus_transpose:
        kraus_transpose = [math.transpose(k) for k in kraus]

    kraus = math.stack(kraus)
    kraus_transpose = math.stack(kraus_transpose)
    # Torch throws error if math.conj is used before stack
    kraus_dagger = math.conj(kraus_transpose)

    kraus = math.cast(math.reshape(kraus, kraus_shape), complex)
    kraus_dagger = math.cast(math.reshape(kraus_dagger, kraus_shape), complex)

    return math.einsum(einsum_indices, kraus, state, kraus_dagger)


@singledispatch
def apply_operation(
    op: qml.operation.Operator, state, is_state_batched: bool = False, debugger=None
):
    """Apply and operator to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (TensorLike): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not
        debugger (_Debugger): The debugger to use

    Returns:
        ndarray: output state

    .. warning::

        ``apply_operation`` is an internal function, and thus subject to change without a deprecation cycle.

    .. warning::
        ``apply_operation`` applies no validation to its inputs.

        This function assumes that the wires of the operator correspond to indices
        of the state. See :func:`~.map_wires` to convert operations to integer wire labels.

        The shape of state should be ``[QUDIT_DIM]*(num_wires * 2)``, where ``QUDIT_DIM`` is
        the dimension of the system.

    This is a ``functools.singledispatch`` function, so additional specialized kernels
    for specific operations can be registered like:

    .. code-block:: python

        @apply_operation.register
        def _(op: type_op, state):
            # custom op application method here

    **Example:**

    >>> state = np.zeros((3,3))
    >>> state[0][0] = 1
    >>> state
    tensor([[1., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]], requires_grad=True)
    >>> apply_operation(qml.TShift(0), state)
    tensor([[0., 0., 0.],
        [0., 1., 0],
        [0., 0., 0.],], requires_grad=True)

    """
    return _apply_operation_default(op, state, is_state_batched, debugger)


def _apply_operation_default(op, state, is_state_batched, debugger):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches.
    """

    return apply_operation_einsum(op, state, is_state_batched=is_state_batched)
    # TODO add tensordot and benchmark for performance


# TODO add diagonal for speed up.


@apply_operation.register
def apply_snapshot(op: qml.Snapshot, state, is_state_batched: bool = False, debugger=None):
    """Take a snapshot of the mixed state"""
    if debugger and debugger.active:
        measurement = op.hyperparameters["measurement"]
        if measurement:
            # TODO replace with: measure once added
            raise NotImplementedError  # TODO
        else:
            if is_state_batched:
                dim = int(np.sqrt(math.size(state[0])))
                flat_shape = [math.shape(state)[0], dim, dim]
            else:
                dim = int(np.sqrt(math.size(state)))
                flat_shape = [dim, dim]

            snapshot = math.reshape(state, flat_shape)
        if op.tag:
            debugger.snapshots[op.tag] = snapshot
        else:
            debugger.snapshots[len(debugger.snapshots)] = snapshot
    return state


# TODO add special case speedups


def _get_kraus(operation):  # pylint: disable=no-self-use
    """Return the Kraus operators representing the operation.

    Args:
        operation (.Operation): a PennyLane operation

    Returns:
        list[array[complex]]: Returns a list of 2D matrices representing the Kraus operators. If
        the operation is unitary, returns a single Kraus operator.
    """
    if isinstance(operation, Channel):
        return operation.kraus_matrices()
    return [operation.matrix()]
