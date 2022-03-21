import pennylane as qml

from pennylane.tape.tape import QuantumTape
from pennylane.script.tape import IfTape, WhileTape, FunctionTape


@qml.qfunc_transform
def remove_control_flow_transform(tape: QuantumTape):
    with QuantumTape() as new_tape:
        _remove_control_flow(tape)
    return new_tape

def _remove_control_flow(tape: QuantumTape):
        for op in tape.queue:
            if isinstance(op, IfTape):
                if op.expr():
                    for if_op in op.queue:
                        if isinstance(if_op, QuantumTape):
                            _remove_control_flow(if_op)
                        else:
                            qml.apply(if_op)
            elif isinstance(op, WhileTape):
                while op.expr():
                    for while_op in op.queue:
                        if isinstance(while_op, QuantumTape):
                            _remove_control_flow(while_op)
                        else:
                            qml.apply(while_op)
            elif isinstance(op, FunctionTape):

            qml.apply(op)