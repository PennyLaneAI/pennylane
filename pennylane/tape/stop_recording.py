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
"""
Contains a context manager and decorator to turn off the PennyLane queuing system.
"""
import contextlib
from warnings import warn

from pennylane import QueuingManager


@contextlib.contextmanager
def stop_recording():
    """A context manager and decorator to ensure that contained logic is non-recordable
    or non-queueable within a QNode or quantum tape context.

    Deprecated in favor of :meth:`pennylane.QueuingManager.stop_recording`.

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
    ...     qml.apply(ops[-1])  # apply only the last operation from the list
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit)([1, 2, 3]))
    0: ──RX(1.00)──RY(2.00)──RZ(3.00)──RZ(3.00)─┤  <Z>

    Using the ``stop_recording`` context manager, all logic
    contained within is not queued or recorded by the QNode:

    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     with stop_recording():
    ...         ops = list_of_ops(params, wires=0)
    ...     qml.apply(ops[-1])
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit)([1, 2, 3]))
    0: ──RZ(3.00)─┤  <Z>

    ``stop_recording`` can also be used as a decorator. Decorated
    functions, when executed, will inhibit any internal
    quantum operation processing from being recorded by the QNode:

    >>> @stop_recording()
    ... def list_of_ops(params, wires):
    ...     return [
    ...         qml.RX(params[0], wires=wires),
    ...         qml.RY(params[1], wires=wires),
    ...         qml.RZ(params[2], wires=wires)
    ...     ]
    >>> @qml.qnode(dev)
    ... def circuit(params):
    ...     ops = list_of_ops(params, wires=0)
    ...     qml.apply(ops[-1])
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(qml.draw(circuit)([1, 2, 3]))
    0: ──RZ(3.00)─┤  <Z>

    """
    warn(
        "qml.tape.stop_recording has moved to qml.QueuingManager.stop_recording.", UserWarning,
    )
    with QueuingManager.stop_recording():
        yield
