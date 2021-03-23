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
from pennylane.tape import QuantumTape
from pennylane.operation import Operation, AnyWires
from pennylane.wires import Wires
from pennylane.transforms.registrations import register_control, CONTROL_MAPS
from pennylane.transforms.adjoint import adjoint


def expand_with_control(tape, control_wire):
    with QuantumTape(do_queue=False) as new_tape:
        for op in tape.operations:
            if op.__class__ in CONTROL_MAPS:
                # Create the controlled version of the operation
                # add add that the to the tape context.
                # NOTE(chase): Should we use type(op) here instead?
                CONTROL_MAPS[op.__class__](op, control_wire)
            else:
                raise NotImplementedError(
                    f"Control transform of operation {type(op)} is not registered."
                    "You can define the control transform by using qml.register_control"
                    "See documentation for more details."
                )
    return new_tape


class ControlledOperation(Operation):
    par_domain = "A"
    num_wires = AnyWires
    num_params = property(lambda self: self._tape.num_params)

    def __init__(self, tape, control_wires, do_queue=True):
        self._tape = tape
        self._control_wires = Wires(control_wires)
        wires = self._control_wires + tape.wires
        super().__init__(*tape.get_parameters(), wires=wires, do_queue=do_queue)

    def expand(self):
        tape = self._tape
        for wire in self._control_wires:
            tape = expand_with_control(tape, wire)
        # TODO(chase): Do we need to re-queue these ops?
        return tape

    def adjoint(self, do_queue=False):
        def requeue_tape(tape):
            for op in tape.operations:
                op.queue()

        with QuantumTape(do_queue=False) as new_tape:
            # Execute all ops adjointed.
            adjoint(requeue_tape)(self._tape)
        return ControlledOperation(new_tape, self._control_wires, do_queue=do_queue)

register_control(
    ControlledOperation, 
    lambda op, wires: ControlledOperation(
        tape=op._tape, 
        control_wires=Wires(wires) + op._control_wires, 
        do_queue=True))


def ctrl(fn, control_wires):
    """Create a method that applies a controlled version of the provided method.

    TODO(chase): documentation.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with QuantumTape(do_queue=False) as tape:
            fn(*args, **kwargs)
        return ControlledOperation(tape, control_wires)

    return wrapper
