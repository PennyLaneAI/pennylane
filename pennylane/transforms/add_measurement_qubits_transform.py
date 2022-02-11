"""Code for the tape transform implementing the deferred measurement principle."""
import pennylane as qml
from pennylane.transforms import qfunc_transform, ctrl
from pennylane.queuing import apply
from pennylane.tape import QuantumTape

@qfunc_transform
def add_measurement_wires(tape):

    with QuantumTape() as new_tape:
        for op in tape.queue:
            if isinstance(op, qml.ops.mid_circuit_measure._MidCircuitMeasure):
                pass

            elif op.__class__.__name__ == "_IfOp":
                control = op.measured_qubit._dependent_on
                op_class = op.then_op.__class__
                if op.data:
                    controlled_op = ctrl(op_class, control=control)(*op.data, wires=op.wires)

                else:
                    controlled_op = ctrl(op_class, control=control)(wires=op.wires)
            else:
                apply(op)

    return new_tape