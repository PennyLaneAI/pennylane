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

    new_mx = np.copy(matrix)
    new_mx[[2, 1]] = new_mx[[1, 2]]
    new_mx[:,[2, 1]] = new_mx[:,[1, 2]]
    return new_mx

def get_merged_op(gate_first, gate_second):
    first_wires = gate_first.wires
    second_wires = gate_second.wires
    num_wires = len(first_wires)
    new_mat = gate_first.matrix.reshape([2] * num_wires * 2)
    mat = gate_second.matrix.reshape([2] * len(second_wires) * 2)
    all_first_axes = np.arange(2 * len(first_wires)) # len(second_wires) is the qubit number, we have twice as many axes
    first_axes_used = all_first_axes[-len(second_wires):] # select the last len(first_wires) many axes

    axes = (first_axes_used, second_wires)
    new_mat = np.tensordot(new_mat, mat, axes=axes)

    mat_dim = 2 ** num_wires
    return qml.QubitUnitary(new_mat.reshape(mat_dim, mat_dim), wires=first_wires)

def merge(gate_first, gate_second):
    """
    Note: this function as of now assumes consecutive integer wire ordering for operations.
    """
    # There are 3 cases as we assumed that there's a subset relation between the sets of wires
    # 1. case: both gates act exactly on the same wires
    first_wires = gate_first.wires
    second_wires = gate_second.wires
    if gate_first.wires == gate_second.wires:
        wires = gate_first.wires

        new_matrix = gate_first.matrix @ gate_second.matrix
        new_gate = qml.QubitUnitary(new_matrix, wires=wires)

    # 2. case: gate_first acts on all wires gate_second acts and more
    elif first_wires.contains_wires(second_wires):
        new_gate = get_merged_op(gate_first, gate_second)

    # 3. case: gate_second acts on all wires gate_first acts and more
    else:
        new_gate = get_merged_op(gate_second, gate_first)

    return new_gate


def lightweight_optimize(circuit):
    """
    Greedy algorithm as described in the Qulacs ref. and implemented:

    https://github.com/qulacs/qulacs/blob/8cd29d4c1d7836c37b32b42a2516d1fbcd41535a/src/cppsim/circuit_optimizer.cpp#L141

    Note: this function as of now assumes consecutive integer wire ordering for operations.
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
        parent_qubits = qml.wires.Wires([])
        pos = -1;
        hit = -1;

        for target_qubit in target_qubits:
            if not isinstance(target_qubit, int):
                if len(target_qubit) > 1:
                    raise ValueError

                target_qubit = (target_qubit.tolist())[0]
            if current_step[target_qubit][0] > pos:
                pos = current_step[target_qubit][0]
                hit = target_qubit

        if hit!=-1:
            parent_qubits = current_step[hit][1]

        if parent_qubits.contains_wires(target_qubits):

            merged_gate = merge(gate_list[pos], gate)

            del gate_list[ind]
            gate_list.insert(pos + 1,merged_gate)
            del gate_list[pos]

        else:

            for target_qubit in target_qubits:
                if not isinstance(target_qubit, int) and len(target_qubit) > 1:
                    raise ValueError
                if not isinstance(target_qubit, int):
                    target_qubit = (target_qubit.tolist())[0]
                current_step[target_qubit] = (ind, target_qubits)

            ind+=1

        num_gates = len(circuit._ops)

    return circuit
