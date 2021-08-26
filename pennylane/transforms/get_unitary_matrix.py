import pennylane as qml
from pennylane.wires import Wires
from functools import wraps, reduce
from itertools import repeat
from pennylane.tape import QuantumTape, get_active_tape
import numpy as np


def get_unitary_matrix(fn, wires):
    # wires or wire_order?
    wires = Wires(wires)
    n_wires = len(wires)
    # print(wires)

    mat = np.eye(2 ** n_wires)

    @wraps(fn)
    def wrapper(*args, **kwargs):
            ######### NOTE BELOW
        active_tape = get_active_tape()

        if active_tape is not None:
            with active_tape.stop_recording(), QuantumTape() as tape:
                fn(*args, **kwargs)
        else:
            # Not within a queuing context
            with QuantumTape() as tape:
                fn(*args, **kwargs)

        if not tape.operations:
            # we called op.expand(): get the outputted tape
            tape = fn(*args, **kwargs)
        #### NOTE took the above from qml.adjoint. Appropriate here?

        mat = np.eye(2 ** n_wires)

        for op in tape.operations:

            # Single qubit gates
            if len(op.wires) == 1:
                wire_pos = wires.index(op.wires)
                I_left = np.eye(2 ** wire_pos)
                I_right = np.eye(2 ** (n_wires - wire_pos - 1))
                mat = np.dot(np.kron(I_left, np.kron(op.matrix, I_right)), mat)

        return mat

    return wrapper
