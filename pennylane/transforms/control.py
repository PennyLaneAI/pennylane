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
Contains the control transform.
"""
from functools import wraps

import pennylane as qml
from pennylane.ops.math import Controlled

def ctrl(fn, control, control_values=None):
    """Create a method that applies a controlled version of the provided method.

    Args:
        fn (function): Any python function that applies pennylane operations.
        control (Wires): The control wire(s).
        control_values (list[int]): The values the control wire(s) should take.

    Returns:
        function: A new function that applies the controlled equivalent of ``fn``. The returned
        function takes the same input arguments as ``fn``.

    **Example**

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=4)

        def ops(params):
            qml.RX(params[0], wires=0)
            qml.RZ(params[1], wires=3)

        ops1 = qml.ctrl(ops, control=1)
        ops2 = qml.ctrl(ops, control=2)

        @qml.qnode(dev)
        def my_circuit():
            ops1(params=[0.123, 0.456])
            ops1(params=[0.789, 1.234])
            ops2(params=[2.987, 3.654])
            ops2(params=[2.321, 1.111])
            return qml.state()

    The above code would be equivalent to

    .. code-block:: python3

        @qml.qnode(dev)
        def my_circuit2():
            # ops1(params=[0.123, 0.456])
            qml.CRX(0.123, wires=[1, 0])
            qml.CRZ(0.456, wires=[1, 3])

            # ops1(params=[0.789, 1.234])
            qml.CRX(0.789, wires=[1, 0])
            qml.CRZ(1.234, wires=[1, 3])

            # ops2(params=[2.987, 3.654])
            qml.CRX(2.987, wires=[2, 0])
            qml.CRZ(3.654, wires=[2, 3])

            # ops2(params=[2.321, 1.111])
            qml.CRX(2.321, wires=[2, 0])
            qml.CRZ(1.111, wires=[2, 3])
            return qml.state()

    .. Note::

        Some devices are able to take advantage of the inherient sparsity of a
        controlled operation. In those cases, it may be more efficient to use
        this transform rather than adding controls by hand. For devices that don't
        have special control support, the operation is expanded to add control wires
        to each underlying op individually.

    .. UsageDetails::

        **Nesting Controls**

        The ``ctrl`` transform can be nested with itself arbitrarily.

        .. code-block:: python3

            # These two ops are equivalent.
            op1 = qml.ctrl(qml.ctrl(my_ops, 1), 2)
            op2 = qml.ctrl(my_ops, [2, 1])

        **Control Value Assignment**

        Control values can be assigned as follows.

        .. code-block:: python3

            op = qml.ctrl(qml.ctrl(my_ops, 1), 2, control_values=0)
            op()

        This is equivalent to the following.

        .. code-block:: python3

            qml.PauliX(wires=2)
            op = qml.ctrl(qml.ctrl(my_ops, 1), 2)
            op()
            qml.PauliX(wires=2)

    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with QuantumTape(do_queue=False) as tape:
            fn(*args, **kwargs)
        return [Controlled(op, control, control_values=control_values) for op in tape.operations+tape.measurements]   
    return wrapper
