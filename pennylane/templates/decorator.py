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
This module contains the template decorator.
"""
from functools import wraps


def template(func):
    """Register a quantum template with PennyLane.

    This decorator wraps the given function and makes it return a list of all queued Operations.

    **Example:**

    When defining a :doc:`template </introduction/templates>`, simply decorate
    the template function with this decorator.

    .. code-block:: python3

        @qml.template
        def bell_state_preparation(wires):
            qml.Hadamard(wires=wires[0])
            qml.CNOT(wires=wires)

    This registers the template with PennyLane, making it compatible with
    functions that act on templates, such as :func:`pennylane.inv`:

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=2)

        @qml.qnode(dev)
        def circuit():
            qml.inv(bell_state_preparation(wires=[0, 1]))
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    Args:
        func (callable): A template function

    Returns:
        callable: The wrapper function
    """
    import pennylane as qml  # pylint: disable=import-outside-toplevel

    @wraps(func)
    def wrapper(*args, **kwargs):

        with qml.tape.OperationRecorder() as rec:
            func(*args, **kwargs)

        return rec.queue

    return wrapper
