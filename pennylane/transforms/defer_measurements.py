# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Code for the tape transform implementing the deferred measurement principle."""
from pennylane.wires import Wires

import pennylane as qml
from pennylane.transforms import qfunc_transform, ctrl
from pennylane.queuing import apply
from pennylane.tape import QuantumTape, get_active_tape

class Aux:
    """
    A wire label for a wire holding the state of the `self.base_wire` just before it was mid-circuit measured,
    for the `self.count` time.
    """

    def __init__(self, wire_or_aux, count=0):
        if isinstance(wire_or_aux, Aux):
            self.base_wire = wire_or_aux.base_wire
            self.count = wire_or_aux.count + count + 1
        else:
            self.base_wire = wire_or_aux
            self.count = count

    def __hash__(self):
        return hash((Aux, self.base_wire, self.count))

    def __eq__(self, other):
        if isinstance(other, Aux):
            return self.base_wire == other.base_wire and self.count == other.count
        return False

    def __str__(self):
        return f"Aux({repr(self.base_wire)},{repr(self.count)})"

    def __repr__(self):
        return f"Aux({repr(self.base_wire)},{repr(self.count)})"


class WireRemapper:

    def __init__(self):
        self._altered_wires = set()
        self._measured_wires = {}

    def mark_altered(self, wire):
        self._altered_wires.add(wire)

    def get_mapped(self, base_wire):
        assert not isinstance(base_wire, Aux)
        if base_wire in self._measured_wires.keys():
            return Aux(base_wire, self._measured_wires[base_wire])
        return base_wire

    def mark_altered_and_get_mapped(self, base_wire):
        assert not isinstance(base_wire, Aux)
        mapped = self.get_mapped(base_wire)
        self.mark_altered(mapped)
        return mapped

    def mark_measured_and_get_mapped(self, base_wire):
        assert not isinstance(base_wire, Aux)
        if base_wire not in self._measured_wires.keys():
            if base_wire in self._altered_wires:
                self._measured_wires[base_wire] = 0
                return Aux(base_wire)
            else:
                # self._altered_wires.add(base_wire)  # not sure if this should be here?
                return base_wire
        else:
            if Aux(base_wire, self._measured_wires[base_wire]) in self._altered_wires:
                self._measured_wires[base_wire] += 1
                return Aux(base_wire, self._measured_wires[base_wire])



def make_mid_circuit_measurements_terminal(tape):

    ops = []
    for op in tape.queue:
        ops.append(op)

    wr = WireRemapper()
    new_ops_reversed = []
    for op in reversed(ops):
        # need to work in the reverse direction for this tape transform
        if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
            wire = op.wires[0]
            mapped_wire = wr.mark_measured_and_get_mapped(wire)
            op._wires = Wires([mapped_wire])
        else:
            new_wires = []
            for wire in op.wires:
                new_wires.append(wr.mark_altered_and_get_mapped(wire))
            op._wires = Wires(new_wires)

        new_ops_reversed.append(op)

    # reverse the tape back to original form
    with QuantumTape() as new_tape:
        for op in reversed(new_ops_reversed):
            apply(op)

    return new_tape


def defer_measurements_on_mid_circuit_measured_terminal_tape(tape):

    with QuantumTape() as new_tape:
        measured_wires = {}
        for op in tape.queue:
            if any([wire in measured_wires.values() for wire in op.wires]):
                raise ValueError("cannot reuse measured wires.")

            if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
                measured_wires[op.measurement_id] = op.wires[0]

            elif op.__class__.__name__ == "_IfOp":
                control = [measured_wires[m_id] for m_id in op.dependant_measurements]
                flipped = [False] * len(control)
                for branch, value in op.branches.items():
                    if value:
                        for i, wire_val in enumerate(branch):
                            if wire_val and flipped[i] or not wire_val and not flipped[i]:
                                qml.PauliX(control[i])
                                flipped[i] = not flipped[i]
                        ctrl(lambda: apply(op.then_op), control=Wires(control))()
                for i, flip in enumerate(flipped):
                    if flip:
                        qml.PauliX(control[i])

            elif op.__class__.__name__ == "_ConditionOp":
                control = [measured_wires[m_id] for m_id in op.dependant_measurements]
                flipped = [False] * len(control)
                for branch, branch_op in op.branches.items():
                    for i, wire_val in enumerate(branch):
                        if wire_val and flipped[i] or not wire_val and not flipped[i]:
                            qml.PauliX(control[i])
                            flipped[i] = not flipped[i]
                    ctrl(lambda: apply(branch_op), control=Wires(control))()
                for i, flip in enumerate(flipped):
                    if flip:
                        qml.PauliX(control[i])

            else:
                apply(op)

    return new_tape

@qfunc_transform
def defer_measurements(tape):
    active_tape = get_active_tape()
    with get_active_tape().stop_recording():
        tape = make_mid_circuit_measurements_terminal(tape)
    defer_measurements_on_mid_circuit_measured_terminal_tape(tape)
