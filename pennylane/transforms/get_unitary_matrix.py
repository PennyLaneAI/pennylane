import pennylane as qml
from pennylane.wires import Wires
from functools import wraps, reduce
from itertools import repeat
from pennylane.tape import QuantumTape, get_active_tape
import numpy as np
from difflib import SequenceMatcher


def get_unitary_matrix(fn, wire_order):
    # wires or wire_order?
    wire_order = Wires(wire_order)
    n_wires = len(wire_order)
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

            # ASSUME ADJACENT WIRES

            # check if the operator wires match the wire order
            s = SequenceMatcher(lambda x: x == " ", op.wires.tolist(), wire_order.tolist())

            match = s.find_longest_match()
            print(match, match[2])
            print(wire_order, op.wires)

            # if all operator wires are adjacent
            # match [2] is the length of the matching sequence
            if match[2] == len(op.wires):

                I_left = np.eye(2 ** match[1])
                I_right = np.eye(2 ** (n_wires - len(op.wires) - match[1]))

                mat = np.dot(np.kron(I_left, np.kron(op.matrix, I_right)), mat)

        return mat

    return wrapper
