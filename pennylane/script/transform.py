import pennylane as qml

from pennylane.tape.tape import QuantumTape
from pennylane.script.tape import IfTape, WhileTape, FunctionTape

def remove_control_flow_transform(tape: QuantumTape):
    with QuantumTape() as new_tape:
        for op in tape.queue:
            if isinstance(op, IfTape):
                if op.expr():
                    for if_op in op.queue:
                        qml.apply(if_op)
            elif isinstance(op, WhileTape):
                while op.expr():
                    for while_op in op.queue:
                        qml.apply(while_op)
            if isinstance(op, FunctionTape):
                pass
            qml.apply(op)
    return new_tape