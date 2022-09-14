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

from pennylane.operation import Operator
from pennylane.ops.op_math import Controlled


def ctrl(op, control, control_values=None, work_wires=None):
    """Create a method that applies a controlled version of the provided op.

    Args:
        op (function or :class:`~.operation.Operator`): A single operator or a function that applies pennylane operators.
        control (Wires): The control wire(s).
        control_values (bool or list[bool]): The value(s) the control wire(s) should take.
            Integers other than 0 or 1 will be treated as ``int(bool(x))``.
        work_wires (Any): Any auxiliary wires that can be used in the decomposition

    Returns:
        (function or :class:`~.operation.Operator`): If an Operator is provided, returns a Contolled version of the Operator.
        If a function is provided, returns a function with the same call signature that creates a controlled version of the
        provided function.

    .. seealso:: :class:`~.Controlled`.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device('default.qubit', wires=range(4)))
        def circuit(x):
            qml.PauliX(2)
            qml.ctrl(qml.RX, (1,2,3), control_values=(0,1,0))(x, wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> print(qml.draw(circuit)("x"))
    0: ────╭RX(x)─┤  <Z>
    1: ────├○─────┤
    2: ──X─├●─────┤
    3: ────╰○─────┤
    >>> x = np.array(1.2)
    >>> circuit(x)
    tensor(0.36235775, requires_grad=True)
    >>> qml.grad(circuit)(x)
    -0.9320390859672264

    :func:`~.ctrl` works on both callables like ``qml.RX`` or a quantum function
    and individual :class:`~.operation.Operator`'s.

    >>> qml.ctrl(qml.PauliX(0), (1,2))
    Controlled(PauliX(wires=[0]), control_wires=[1, 2])
    >>> qml.ctrl(qml.PauliX(0), (1,2)).decomposition()
    [Toffoli(wires=[1, 2, 0])]

    Controlled operations work with all other forms of operator math and simplification:

    >>> op = qml.ctrl(qml.RX(1.2, wires=0) ** 2 @ qml.RY(0.1, wires=0), control=1)
    >>> qml.simplify(qml.adjoint(op))
    Controlled(RY(12.466370614359173, wires=[0]) @ RX(10.166370614359172, wires=[0]), control_wires=[1])

    """
    control_values = [control_values] if isinstance(control_values, int) else control_values
    control = qml.wires.Wires(control)

    if isinstance(op, Operator):
        return Controlled(
            op, control_wires=control, control_values=control_values, work_wires=work_wires
        )
    if not callable(op):
        raise ValueError(
            f"The object {op} of type {type(op)} is not an Operator or callable. "
            "This error might occur if you apply ctrl to a list "
            "of operations instead of a function or Operator."
        )

    @wraps(op)
    def wrapper(*args, **kwargs):
        tape = qml.transforms.make_tape(op)(*args, **kwargs)

        # flip control_values == 0 wires here, so we don't have to do it for each individual op.
        flip_control_on_zero = (len(tape) > 1) and (control_values is not None)
        op_control_values = None if flip_control_on_zero else control_values
        if flip_control_on_zero:
            _ = [qml.PauliX(w) for w, val in zip(control, control_values) if not val]

        _ = [
            Controlled(
                op, control_wires=control, control_values=op_control_values, work_wires=work_wires
            )
            for op in tape.operations
        ]

        if flip_control_on_zero:
            _ = [qml.PauliX(w) for w, val in zip(control, control_values) if not val]

        if qml.queuing.QueuingManager.recording():
            _ = [qml.apply(m) for m in tape.measurements]

        return tape.measurements

    return wrapper
