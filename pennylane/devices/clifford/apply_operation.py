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
# pylint: disable=unused-argument

from functools import singledispatch
import pennylane as qml

from pennylane import math

_operations = {
    "Identity": "i",
    "Snapshot": None,
    "BasisState": None,
    "StatePrep": None,
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "Hadamard": "h",
    "S": "s",
    "SX": "sx",
    "CNOT": "cnot",
    "SWAP": "swap",
    "ISWAP": "iswap",
    "CY": "cy",
    "CZ": "cz",
    "GlobalPhase": None,
}

SQRT2INV = 1 / math.sqrt(2)


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
    return _apply_operation_default(op, state, is_state_batched, debugger)


def _apply_operation_default(op, state, is_state_batched, debugger):
    """The default behaviour of apply_operation, accessed through the standard dispatch
    of apply_operation, as well as conditionally in other dispatches."""
    pass


@apply_operation.register
def apply_identity(op: qml.Identity, state, is_state_batched: bool = False, debugger=None):
    """Applies a :class:`~.Identity` operation by just returning the input state."""
    return state


@apply_operation.register
def apply_global_phase(op: qml.GlobalPhase, state, is_state_batched: bool = False, debugger=None):
    """Applies a :class:`~.GlobalPhase` operation by multiplying the state by ``exp(1j * op.data[0])``"""
    return qml.math.exp(-1j * qml.math.cast(op.data[0], complex)) * state


@apply_operation.register
def apply_snapshot(op: qml.Snapshot, state, is_state_batched: bool = False, debugger=None):
    """Take a snapshot of the state"""
    if debugger is not None and debugger.active:
        flat_state = math.flatten(state)
        if op.tag:
            debugger.snapshots[op.tag] = flat_state
        else:
            debugger.snapshots[len(debugger.snapshots)] = flat_state
    return state
