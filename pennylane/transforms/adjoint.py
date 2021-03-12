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
"""Code for the adjoint transform."""

from functools import wraps
from pennylane.tape import QuantumTape


def adjoint(fn):
    """Create a function that applies the adjoint of the provided operation or template.

    This transform can be used to apply the adjoint of an arbitrary sequence of operations.

    Args:
        fn (function): Any python function that applies pennylane operations.

    Returns:
        function: A new function that will apply the same operations but adjointed and in reverse order.

    **Example**

    .. code-block:: python3

        def my_ops():
            qml.RX(0.123, wires=0)
            qml.RY(0.456, wires=0)

        adj_my_ops = qml.adjoint(my_ops)
        adj_my_ops() # This will queue ops [qml.RY(-0.456), qml.RX(-0.123)]

    .. UsageDetails::

        **Adjoint of a function**

        Here, we apply the ``subroutine`` function, and then apply its inverse.
        Notice that in addition to adjointing all of the operations, they are also
        applied in reverse construction order.

        .. code-block:: python3

            def subroutine(wire):
                qml.RX(0.123, wires=wire)
                qml.RY(0.456, wires=wire)

            dev = qml.device('default.qubit', wires=1)
            @qml.qnode(dev)
            def circuit():
                subroutine(0)
                qml.adjoint(subroutine)(0)
                return qml.expval(qml.PauliZ(0))

        This creates the following circuit:

        >>> print(qml.draw(circuit)())
        0: --RX(0.123)--RY(0.456)--RY(-0.456)--RX(-0.123)--| <Z>

        **Single operation**

        You can also easily adjoint a single operation just by wrapping it with ``adjoint``:

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=1)
            @qml.qnode(dev)
            def circuit():
                qml.RX(0.123, wires=0)
                qml.adjoint(qml.RX)(0.123, wires=0)
                return qml.expval(qml.PauliZ(0))

        This creates the following circuit:

        >>> print(qml.draw(circuit)())
        0: --RX(0.123)--RX(-0.123)--| <Z>
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with QuantumTape(do_queue=False) as tape:
            fn(*args, **kwargs)
        for op in reversed(tape.queue):
            op.adjoint(do_queue=True)

    return wrapper
