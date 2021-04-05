# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains a transform to make quantum function non-recordable
or invisible within a QNode or quantum tape context."""

from functools import wraps

import pennylane as qml


def invisible(fn):
    """A transform to make a quantum function non-recordable
    or invisible within a QNode or quantum tape context.

    Args:
        fn (function): A quantum function that applies quantum operations.

    Returns:
        function: The input function transformed, so that it will not be
        recorded by QNodes or quantum tapes.

    **Example**

    Consider the following quantum function:

    >>> def list_of_ops(params, wires):
    ...     return [
    ...         qml.RX(params[0], wires=wires),
    ...         qml.RY(params[1], wires=wires),
    ...         qml.RZ(params[2], wires=wires)
    ...     ]

    If executed within a QNode or a tape context, these operations will be
    recorded, even if this is not the intention:

    >>> dev = qml.device("default.qubit", wires=2)
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     ops = list_of_ops(params, wires=0)
    ...     # apply only the last operation from the list
    ...     ops[-1].queue()
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit)([1, 2, 3]))
     0: ──RX(1)──RY(2)──RZ(3)──┤ ⟨Z⟩

    Marking the quantum function as invisible will inhibit any internal
    quantum operation processing from being recorded by the QNode:

    >>> @qml.transforms.invisible
    ... def list_of_ops(params, wires):
    ...     return [
    ...         qml.RX(params[0], wires=wires),
    ...         qml.RY(params[1], wires=wires),
    ...         qml.RZ(params[2], wires=wires)
    ...     ]
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     ops = list_of_ops(params, wires=0)
    ...     # apply only the last operation from the list
    ...     ops[-1].queue()
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit)([1, 2, 3]))
     0: ──RZ(3)──┤ ⟨Z⟩
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        tape = qml.tape.get_active_tape()

        if tape is None:
            return fn(*args, **kwargs)

        with tape.stop_recording():
            return fn(*args, **kwargs)

    return wrapper
