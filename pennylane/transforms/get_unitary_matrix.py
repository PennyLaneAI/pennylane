import pennylane as qml
from pennylane.wires import Wires
from functools import wraps, reduce
from itertools import repeat
from pennylane.tape import QuantumTape, get_active_tape
import numpy as np
from difflib import SequenceMatcher


def get_unitary_matrix(fn, wire_order):

    wire_order = Wires(wire_order)
    n_wires = len(wire_order)

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

        # initialize the unitary matrix
        unitary_matrix = np.eye(2 ** n_wires)

        for op in tape.operations:

            # get the matrix for the operator op
            op_matrix = _get_op_matrix(op, wire_order, np.eye(2 ** n_wires), np.eye(2 ** n_wires))
            # add the single operator matrix to the full unitary matrix
            unitary_matrix = np.dot(op_matrix, unitary_matrix)

        return unitary_matrix

    return wrapper


def _get_op_matrix(op, wire_order, matrix, SWAP):

    # check if the operator wires match the wire order
    s = SequenceMatcher(lambda x: x == " ", op.wires.tolist(), wire_order.tolist())
    match = s.find_longest_match()

    n_wires = len(wire_order)

    # if all operator wires are adjacent, just perform the operation
    # match [2] is the length of the matching sequence
    if match[2] == len(op.wires):

        # Operate with the identity on wires not affected by op
        # match[1] is the index of the first match in wire_order
        I_left = np.eye(2 ** match[1])
        I_right = np.eye(2 ** (n_wires - len(op.wires) - match[1]))

        # matrix representation of the operator op on the full space
        matrix = np.dot(np.kron(I_left, np.kron(op.matrix, I_right)), matrix)

        # Swap back if wires were swapped
        matrix = SWAP @ matrix

        return matrix

    else:  # if there are non-adjacent wires

        # get all matching wires
        all_matches = s.get_matching_blocks()
        for m in all_matches:
            if m[2] > 0:  # For some reason, matches of length 0 are otherwise included
                # if indices don't match, swap wire positions
                if m[0] != m[1]:
                    swaps = np.sort([m[0], m[1]])

                    # print("swap qubit index %d with qubit index %d" % (swaps[0], swaps[1]))
                    # construct SWAP operator to swap wires m[0] and m[1]
                    I_left = np.eye(2 ** (swaps[0]))
                    I_right = np.eye(2 ** (n_wires - swaps[1] - 1))
                    I_middle = np.eye(2 ** (swaps[1] - swaps[0] - 1))

                    state0 = [1, 0]
                    state1 = [0, 1]
                    op00 = np.outer(state0, state0)
                    op11 = np.outer(state1, state1)
                    op01 = np.outer(state0, state1)
                    op10 = np.outer(state1, state0)

                    SWAP1 = reduce(np.kron, [I_left, op00, I_middle, op00, I_right])
                    SWAP1 += reduce(np.kron, [I_left, op11, I_middle, op11, I_right])
                    SWAP1 += reduce(np.kron, [I_left, op10, I_middle, op01, I_right])
                    SWAP1 += reduce(np.kron, [I_left, op01, I_middle, op10, I_right])

                    # Combine SWAPS in case there are multiple
                    SWAP = np.matmul(SWAP, SWAP1)

                    # SWAP wires in the operator matrix
                    matrix = SWAP @ matrix

                    # Swap wires in wire_order
                    wire_order = wire_order.tolist()
                    wire_order[swaps[0]], wire_order[swaps[1]] = (
                        wire_order[swaps[1]],
                        wire_order[swaps[0]],
                    )
                    wire_order = Wires(wire_order)

                    # call function recursively to check if all wires are now in order
                    return _get_op_matrix(op, wire_order, matrix, SWAP)
