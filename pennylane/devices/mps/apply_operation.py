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


@singledispatch
def apply_operation(op: qml.operation.Operator, state, is_state_batched: bool = False):
    """Apply and operator to a given state.

    Args:
        op (Operator): The operation to apply to ``state``
        state (tensor_like): The starting state.
        is_state_batched (bool): Boolean representing whether the state is batched or not

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
    wires = tuple(op.wires)
    gate_opts = {"contract": "swap+split", "cutoff": 0.0, "max_bond": None}
    state.gate_(op.matrix(), wires, **gate_opts)
