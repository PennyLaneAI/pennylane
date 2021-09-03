# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from difflib import SequenceMatcher
from functools import wraps, reduce

from pennylane.wires import Wires
from pennylane.tape import QuantumTape, get_active_tape
import numpy as np


def get_unitary_matrix(fn, wire_order):

    n_wires = len(wire_order)
    wire_order = Wires(wire_order)

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

        unitary_matrix = np.eye(2 ** n_wires)

        for op in tape.operations:

            op_wire_pos = wire_order.indices(op.wires)

            I = np.reshape(np.eye(2 ** n_wires), [2] * n_wires * 2)

            axes = (np.arange(len(op.wires), 2 * len(op.wires)), op_wire_pos)

            U_op_reshaped = np.reshape(op.matrix, [2] * len(op.wires) * 2)

            U_tensordot = np.tensordot(U_op_reshaped, I, axes=axes)

            unused_idxs = [idx for idx in range(n_wires) if idx not in op_wire_pos]
            perm = op_wire_pos + unused_idxs
            inv_perm = np.argsort(perm)

            print("PRINT", op, wire_order.indices(wire_order), axes, perm, op_wire_pos)

            U = np.moveaxis(U_tensordot, wire_order.indices(wire_order), perm)

            U = np.reshape(U, ((2 ** n_wires, 2 ** n_wires)))

            unitary_matrix = np.dot(U, unitary_matrix)
        return unitary_matrix

    return wrapper


def get_unitary_matrix1(fn, wire_order):
    """Given a quantum circuit and a list of wire ordering, construct the matrix representation"""

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
            op_matrix = _get_op_matrix(op, wire_order, [np.eye(2 ** n_wires)])
            # add the single operator matrix to the full unitary matrix
            unitary_matrix = np.dot(op_matrix, unitary_matrix)

        return unitary_matrix

    return wrapper


def _get_op_matrix(op, wire_order, SWAP_list):
    """Called by get_unitary_matrix(fn, wire_order) to construct the matrix representation for an operator"""
    # check if the operator wires match the wire order
    s = SequenceMatcher(lambda x: x == " ", op.wires.tolist(), wire_order.tolist())
    match = s.find_longest_match()
    n_wires = len(wire_order)

    # Initialize matrix
    matrix = np.eye(2 ** n_wires)

    # if all operator wires are adjacent, just perform the operation
    # match [2] is the length of the matching sequence
    if match[2] == len(op.wires):

        # Operate with the identity on wires not affected by op
        # match[1] is the index of the first match in wire_order
        I_left = np.eye(2 ** match[1])
        I_right = np.eye(2 ** (n_wires - len(op.wires) - match[1]))

        # matrix representation of the operator op on the full space
        matrix = np.kron(I_left, np.kron(op.matrix, I_right))

        # Compose SWAP and re-SWAP operators
        SWAP1 = reduce(np.dot, SWAP_list)
        SWAP2 = reduce(np.dot, reversed(SWAP_list))

        # Swap if wires were non-adjacent
        matrix = SWAP1 @ matrix @ SWAP2

        return matrix

    # else if there are non-adjacent wires
    all_matches = s.get_matching_blocks()
    for m in all_matches:
        if m[2] > 0:  # For some reason, matches of length 0 are otherwise included
            # if indices don't match, swap wire positions
            if m[0] != m[1]:
                swaps = np.sort([m[0], m[1]])

                # construct SWAP operator to swap wires m[0] and m[1]
                reshapedI = np.reshape(np.eye(2 ** n_wires), [2] * n_wires * 2)

                SWAP = np.reshape(
                    np.swapaxes(reshapedI, swaps[0], swaps[1]), (2 ** n_wires, 2 ** n_wires)
                )

                # Append to list of all SWAPs
                SWAP_list.append(SWAP)

                # Swap wires in wire_order
                wire_order = wire_order.tolist()
                wire_order[swaps[0]], wire_order[swaps[1]] = (
                    wire_order[swaps[1]],
                    wire_order[swaps[0]],
                )
                wire_order = Wires(wire_order)

                # call recursively until all wires are ordered
                return _get_op_matrix(op, wire_order, SWAP_list)

    return None  # Because pylint wants it
