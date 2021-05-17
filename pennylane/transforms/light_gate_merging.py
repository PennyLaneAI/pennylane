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
"""
Contains the measurement grouping transform
"""
import pennylane as qml
import numpy as np
from copy import copy

def get_flipped_control_target_mx(matrix):

    new_mx = np.copy(matrix) # matrix is a ref to the gate matrix, if we swap rows and cols we also change the original matrix
    new_mx[[2, 1]] = new_mx[[1, 2]]
    new_mx[:,[2, 1]] = new_mx[:,[1, 2]]
    return new_mx

def get_merged_op(neighbour_gate, other_gate):
    first_wires = neighbour_gate.wires
    second_wires = other_gate.wires
    num_wires = len(first_wires)

    first_matrix = neighbour_gate.matrix
    second_matrix = other_gate.matrix

    print(len(first_wires) == 2, first_wires[0]>first_wires[1])
    if len(first_wires) == 2 and first_wires[0]>first_wires[1]:
        first_matrix = get_flipped_control_target_mx(first_matrix)

    if len(second_wires) == 2 and second_wires[0]>second_wires[1]:
        print(second_matrix)
        second_matrix = get_flipped_control_target_mx(second_matrix)
        print(second_matrix)

    new_mat = first_matrix.reshape([2] * num_wires * 2)
    mat = second_matrix.reshape([2] * len(second_wires) * 2)
    all_first_axes = np.arange(2 * len(first_wires)) # len(second_wires) is the qubit number, we have twice as many axes
    first_axes_used = all_first_axes[-len(second_wires):] # select the last len(first_wires) many axes

    axes = (2, 1)
    print(axes)
    new_mat = np.tensordot(new_mat, mat, axes=axes)

    mat_dim = 2 ** num_wires
    return qml.QubitUnitary(new_mat.reshape(mat_dim, mat_dim), wires=first_wires)

def merge(neighbour_gate, other_gate):
    """
    Note: this function as of now assumes consecutive integer wire ordering for operations.
    """
    # There are 3 cases as we assumed that there's a subset relation between the sets of wires
    first_wires = neighbour_gate.wires
    second_wires = other_gate.wires
    if neighbour_gate.wires == other_gate.wires:
        # 1. case: target qubits of neighbour gate = target qubits of current gate
        wires = neighbour_gate.wires

        new_matrix = neighbour_gate.matrix @ other_gate.matrix
        new_gate = qml.QubitUnitary(new_matrix, wires=wires)

    else:
        # 2. case: target qubits of neighbour gate ⊆ target qubits of current gate
        new_gate = get_merged_op(neighbour_gate, other_gate)

        # 3. case: target qubits of neighbour gate ⊄ target qubits of the other
        # gate: handled previously, not merging in this case so it cannot
        # happen here

    return new_gate


def lightweight_optimize(circuit):
    """
    Greedy algorithm as described in the Qulacs ref. and implemented:

    https://github.com/qulacs/qulacs/blob/8cd29d4c1d7836c37b32b42a2516d1fbcd41535a/src/cppsim/circuit_optimizer.cpp#L141

    Note: this function as of now assumes consecutive integer wire ordering for operations.


    **Examples**

     >>> dev = qml.device('default.qubit', wires=2)
     >>> def circuit():
     ... qml.RY(0.3, wires=0)
     ... qml.RY(0.6, wires=1)
     ... qml.RZ(0.3, wires=0)
     ... return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
     >>> qnode = qml.QNode(circuit, dev)
     >>> qnode()
     >>> print(qnode.qtape.draw())
      0: ──RY(0.3)──RZ(0.3)──╭┤ ⟨Z ⊗ Z⟩
      1: ──RY(0.6)───────────╰┤ ⟨Z ⊗ Z⟩

     >>> merged_qt = lightweight_optimize(qnode.qtape)
     >>> print(merged_qt.draw())
      0: ──U0───────╭┤ ⟨Z ⊗ Z⟩
      1: ──RY(0.6)──╰┤ ⟨Z ⊗ Z⟩
     U0 =
     [[ 0.97766824-0.1477601j  -0.1477601 -0.02233176j]
      [ 0.1477601 -0.02233176j  0.97766824+0.1477601j ]]
    """
    circuit = circuit.copy()
    qubit_count = len(circuit.wires)
    current_step = [(-1, 0)] * qubit_count
    gate_list = circuit._ops
    num_gates = len(circuit._ops)

    ind = 0
    while(ind<num_gates):

        gate = gate_list[ind]

        target_qubits = qml.wires.Wires(sorted(gate.wires))

        neighbour_qubits = qml.wires.Wires([])

        neighbour_index = -1 # pos var in qulacs
        # The index of the neighbouring gate to our current gate
        # Note: it might be that more than one gate is neighbouring.
        # In such cases gate merging will not happen: we only merge if
        # the target qubits of the "neighbour" gate are a subset of the
        # target qubits of the current gate

        hit = -1
        # The qubit that indexes the neighbour, a common target qubit for both the neighbour and the current gate
        # TODO: Why do we not store the target qubits straight away? perhaps performance, cheaper to set an int?

        for target_qubit in target_qubits:
            if not isinstance(target_qubit, int):
                if len(target_qubit) > 1:
                    raise ValueError

                target_qubit = (target_qubit.tolist())[0]
            if current_step[target_qubit][0] > neighbour_index:


                neighbour_index = current_step[target_qubit][0]

                hit = target_qubit

        # This update only happens if we in the previous case update hit
        # We gather the target qubits of the "neighbour" gate
        if hit!=-1:
            neighbour_qubits = current_step[hit][1]

        if neighbour_qubits.contains_wires(target_qubits):
            # Merge step
            # We check that the target qubits of the current gate are a subset of
            # the qubits from the "neighbour" gate

            merged_gate = merge(gate_list[neighbour_index], gate)

            del gate_list[ind]
            gate_list.insert(neighbour_index + 1,merged_gate)
            del gate_list[neighbour_index]

        else:
            # Update step
            # We checked the target qubits of the neighbour and the current
            # gate, and we're not merging. This means, that we need to store
            # the info of the current gate.

            # We update all the qubits the current gate acts on, so that the following are stored:
            # 1. the index of the gate and
            # 2. all the target qubits of the gate are stored

            for target_qubit in target_qubits:
                if not isinstance(target_qubit, int) and len(target_qubit) > 1:
                    raise ValueError
                if not isinstance(target_qubit, int):
                    target_qubit = (target_qubit.tolist())[0]
                current_step[target_qubit] = (ind, target_qubits)

            ind+=1

        num_gates = len(circuit._ops)

    return circuit
