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
from pennylane.tape import QuantumTape, get_active_tape
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires
from pennylane.transforms.adjoint import adjoint


def requeue_ops_in_tape(tape):
    """Requeue all of the operations in a tape directly to the current tape context"""
    for op in tape.operations:
        op.queue()


def expand_with_control(tape, control_wire):
    """Expand a tape to include a control wire on all queued operations.

    Args:
        tape (.QuantumTape): quantum tape to be controlled
        control_wire (int): a single wire to use as the control wire

    Returns:
        .QuantumTape: A new QuantumTape with the controlled operations.
    """
    with QuantumTape(do_queue=False) as new_tape:
        for op in tape.operations:
            if hasattr(op, "_controlled"):
                # Execute the controlled version of the operation
                # and add that the to the tape context.
                # pylint: disable=protected-access
                op._controlled(control_wire)
            else:
                # Attempt to decompose the operation and apply
                # controls to each gate in the decomposition.
                with new_tape.stop_recording():
                    try:
                        tmp_tape = op.expand()

                    except NotImplementedError:
                        # No decomposition is defined. Create a
                        # ControlledQubitUnitary gate using the operation
                        # matrix representation.
                        with QuantumTape() as tmp_tape:
                            qml.ControlledQubitUnitary(
                                op.matrix, control_wires=control_wire, wires=op.wires
                            )

                tmp_tape = expand_with_control(tmp_tape, control_wire)
                requeue_ops_in_tape(tmp_tape)

    return new_tape


class ControlledOperation(Operation):
    """A Controlled Operation.

    Unless you are a Pennylane plugin developer, **you should NOT directly use this class**,
    instead, use the :func:`qml.ctrl <.ctrl>` function.

    The ``ControlledOperation`` class is a container class that defines a set of operations that
    should by applied relative to a single control wire or a list of control wires.

    Certain simulators and quantum computers can take advantage of the controlled gate sparsity,
    while other devices must rely on the op-by-op decomposition defined by the ``op.expand``
    method.

    Args:
        tape: A QuantumTape. This tape defines the unitary that should be applied relative
            to the control wires.
        control_wires: A wire or set of wires.
    """

    num_wires = AnyWires

    def __init__(self, tape, control_wires, do_queue=True):
        self.subtape = tape
        """QuantumTape: The tape that defines the underlying operation."""

        self._control_wires = Wires(control_wires)
        """Wires: The control wires."""

        wires = self.control_wires + tape.wires
        super().__init__(*tape.get_parameters(), wires=wires, do_queue=do_queue)

    @property
    def num_params(self):
        return self.subtape.num_params

    @property
    def control_wires(self):
        return self._control_wires

    def expand(self):
        tape = self.subtape
        for wire in self.control_wires:
            tape = expand_with_control(tape, wire)
        return tape

    def adjoint(self):
        """Returns a new ControlledOperation that is equal to the adjoint of `self`"""

        active_tape = get_active_tape()

        if active_tape is not None:
            with get_active_tape().stop_recording(), QuantumTape() as new_tape:
                # Execute all ops adjointed.
                adjoint(requeue_ops_in_tape)(self.subtape)

        else:
            # Not within a queuing context
            with QuantumTape() as new_tape:
                # Execute all ops adjointed.
                ops = adjoint(requeue_ops_in_tape)(self.subtape)

        return ControlledOperation(new_tape, self.control_wires)

    def _controlled(self, wires):
        ControlledOperation(tape=self.subtape, control_wires=Wires(wires) + self.control_wires)


def ctrl(fn, control):
    """Create a method that applies a controlled version of the provided method.

    Args:
        fn (function): Any python function that applies pennylane operations.
        control (Wires): The control wire(s).

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
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with QuantumTape(do_queue=False) as tape:
            fn(*args, **kwargs)
        return ControlledOperation(tape, control)

    return wrapper
