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
import pennylane as qml
from pennylane.transforms import qfunc_transform, ctrl
from pennylane.queuing import apply
from pennylane.tape import QuantumTape

@qfunc_transform
def defer_measurements(tape):

    with QuantumTape() as new_tape:
        for op in tape.queue:
            if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
                pass

            elif op.__class__.__name__ == "_IfOp":
                control = op.dependent_on
                for branch in op.branch:
                    op_class = op.then_op.__class__
                    if op.data:
                        controlled_op = ctrl(op_class, control=control)(*op.data, wires=op.wires)

                    else:
                        controlled_op = ctrl(op_class, control=control)(wires=op.wires)
            else:
                apply(op)

    return new_tape
