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

    matrix = np.eye(2 ** n_wires)

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

        matrix = np.eye(2 ** n_wires)

        for op in tape.operations:

            tmpmat = _get_op_matrix(op, wire_order, np.eye(2 ** n_wires), np.eye(2 ** n_wires))
            matrix = np.dot(tmpmat, matrix)

        return matrix

    return wrapper


def _get_op_matrix(op, wire_order, matrix, SWAP):
    # check if the operator wires match the wire order
    s = SequenceMatcher(lambda x: x == " ", op.wires.tolist(), wire_order.tolist())
    match = s.find_longest_match()
    n_wires = len(wire_order)
    print(s.get_matching_blocks())

    # if all operator wires are adjacent (match [2] is the length of the matching sequence)
    print("OP", op)
    if match[2] == len(op.wires):

        # match[1] is the index of the first match in wire_order
        I_left = np.eye(2 ** match[1])
        I_right = np.eye(2 ** (n_wires - len(op.wires) - match[1]))
        # print("MAT before op\n", matrix)

        # matrix representation of the operator op on the full space
        matrix = np.dot(np.kron(I_left, np.kron(op.matrix, I_right)), matrix)

        # Swap back if wires were swapped
        matrix = SWAP @ matrix

        return matrix

    # if there are non-adjacent wires
    else:

        all_matches = s.get_matching_blocks()
        for m in all_matches:
            if m[2] > 0:  # don't know why "matches" of length 0 are included
                print(m)
                if m[0] != m[1]:  # if indices match, do nothing

                    # wire_order_indices = wire_order.toarray().searchsorted(op.wires.toarray())
                    # print("WIRE ORDER INDICES", wire_order_indices)
                    # print("wire order", wire_order)
                    # print("match", match)
                    # for idx, position in enumerate(wire_order_indices[1:]):
                    swaps = np.sort([m[0], m[1]])

                    print("swap qubit index %d with qubit index %d" % (swaps[0], swaps[1]))
                    # print(swaps)
                    # construct SWAP operator working on the full space
                    I_left = np.eye(2 ** (swaps[0]))
                    # print("LEFT",I_left)
                    I_right = np.eye(2 ** (n_wires - swaps[1] - 1))
                    # print("RIGHT",I_right)
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

                    # print("SWAP1 in swap\n", SWAP)
                    # SWAP1 = swap(3, targets=[2,0]).full().real

                    ## IS THIS RIGHT? check how it works with multiple swaps (i.e. multiple non-adjacent wires)
                    SWAP = np.matmul(SWAP, SWAP1)
                    # SWAP =  np.kron(np.eye(2), qml.SWAP.matrix)

                    matrix = SWAP @ matrix

                    wire_order = wire_order.tolist()
                    print("wire order before swap", wire_order)
                    wire_order[swaps[0]], wire_order[swaps[1]] = (
                        wire_order[swaps[1]],
                        wire_order[swaps[0]],
                    )
                    print("order after swap", wire_order)
                    wire_order = Wires(wire_order)

                    # print("mat in swap\n", matrix)
                    # print("SWAP in swap\n", SWAP)

                    print("recursion")
                    return _get_op_matrix(op, wire_order, matrix, SWAP)
