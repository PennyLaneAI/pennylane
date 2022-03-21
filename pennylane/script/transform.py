import pennylane as qml

from pennylane.tape.tape import QuantumTape
from pennylane.script.tape import IfTape, WhileTape, FunctionTape


@qml.qfunc_transform
def remove_control_flow_transform(tape: QuantumTape):
    with QuantumTape() as new_tape:
        _remove_control_flow(tape)
    return new_tape

def _remove_control_flow(tape: QuantumTape):

        if isinstance(tape, IfTape):
            if tape.expr():
                for if_op in tape.queue:
                    if isinstance(if_op, QuantumTape):
                        _remove_control_flow(if_op)
                    else:
                        qml.apply(if_op)
        elif isinstance(tape, WhileTape):
            while tape.expr():
                for while_op in tape.queue:
                    if isinstance(while_op, QuantumTape):
                        _remove_control_flow(while_op)
                    else:
                        qml.apply(while_op)
        elif isinstance(tape, FunctionTape):
            for fun_op in tape.queue:
                if isinstance(fun_op, QuantumTape):
                    _remove_control_flow(fun_op)
                else:
                    qml.apply(fun_op)
        else:
            for op in tape.queue:
                qml.apply(op)