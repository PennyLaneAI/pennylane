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
from pennylane.tape import QuantumTape


@qfunc_transform
def extend_qubits(tape):

    measured_wires = {}

    ops = []
    for op in tape.queue:
        ops.append(op)

    new_ops_reversed = []
    for op in reversed(ops):

        if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
            if op.wires[0] not in measured_wires.keys():
                measured_wires[op.wires[0]] = 0
            else:
                wire = op.wires[0]
                op._wires = Wires([("aux", wire, measured_wires[wire])])
                measured_wires[wire] += 1
        else:
            for i, wire in enumerate(op.wires):
                new_wires = []
                if wire in measured_wires.keys():
                    new_wires.append(("aux", wire, measured_wires[wire]))
                else:
                    new_wires.append(wire)
                op._wires = Wires(new_wires)

        new_ops_reversed.append(op)

    with QuantumTape() as new_tape:
        for op in reversed(new_ops_reversed):
            apply(op)

    return new_tape


@qfunc_transform
def defer_measurements(tape):

    with QuantumTape() as new_tape:
        measured_wires = []
        for op in tape.queue:
            if any([wire in measured_wires for wire in op.wires]):
                raise ValueError("cannot reuse measured wires.")

            if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
                measured_wires.append(op.wires[0])

            elif op.__class__.__name__ == "_IfOp":
                control = op.dependant_measurements
                flipped = [False] * len(control)
                for branch, value in op.branches.items():
                    if value:
                        for i, wire_val in enumerate(branch):
                            if wire_val and flipped[i] or not wire_val and not flipped[i]:
                                qml.PauliX(wires=control[i])
                                flipped[i] = not flipped[i]
                        ctrl(lambda: apply(op.then_op), control=control)()
                for i, flip in enumerate(flipped):
                    if flip:
                        qml.PauliX(wires=control[i])

            elif op.__class__.__name__ == "_ConditionOp":
                control = op.dependant_measurements
                flipped = [False] * len(control)
                for branch, branch_op in op.branches.items():
                    for i, wire_val in enumerate(branch):
                        if wire_val and flipped[i] or not wire_val and not flipped[i]:
                            qml.PauliX(wires=control[i])
                            flipped[i] = not flipped[i]
                    ctrl(lambda: apply(branch_op), control=control)()
                for i, flip in enumerate(flipped):
                    if flip:
                        qml.PauliX(wires=control[i])

            else:
                apply(op)

    return new_tape
