# Copyright 2021-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform finding all maximal matches of a pattern in a quantum circuit."""

import itertools
import pennylane as qml
from pennylane import apply
from pennylane.transforms import qfunc_transform, get_dag_commutation, make_tape
from pennylane.ops.qubit.attributes import (
    self_inverses,
    symmetric_over_all_wires,
    symmetric_over_control_wires,
)


@qfunc_transform
def pattern_matching(tape, pattern_tapes):
    r"""Quantum function transform to optimize a circuit given a list of patterns.

    Args:
        qfunc (function): A quantum function.
        pattern_tapes(list(.QuantumTape)): List of quantum tapes or quantum functions without measurement.

    Returns:
        function: the transformed quantum function

    **Example**

    Consider the following quantum function.

    .. code-block:: python

        def qfunc(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.PauliZ(0))

    The circuit before optimization:

    """
    for pattern in pattern_tapes:
        # Check the validity of the pattern
        if not isinstance(pattern, qml.tape.QuantumTape):
            raise qml.QuantumFunctionError(
                f"The pattern {pattern}, does not appear "
                "to be a valid quantum tape"
            )

        # Check that it does not contain a measurement.
        if pattern.measurements:
            raise qml.QuantumFunctionError(
                f"The pattern {pattern}, contains measurements. "
            )

        # Verify that the pattern is implementing the identity

        # Verify that the pattern has less qubits and less gates

        # Construct Dag representation of the circuit and the pattern.
        circuit_dag = qml.transforms.get_dag_commutation(tape)()
        pattern_dag = qml.transforms.get_dag_commutation(pattern)()

        # Loop through all possible initial matches
        for node_c in circuit_dag.get_nodes():
            for node_p in pattern_dag.get_nodes():
                # Initial matches between two identical gates (No qubits comparison)
                if _compare_operation_without_qubits(node_c[1], node_p[1]):
                    node_c_id = node_c[0]
                    node_p_id = node_p[0]
                    print("Match between circuit op", node_c_id)
                    print("And pattern op", node_p_id)
                    print("__________________")

                    circuit_range = range(0, circuit_dag.num_wires)

                    # Fix qubits from the first (target fixed and control restrained)
                    not_fixed_qubits_confs = _not_fixed_qubits(circuit_range, node_c[1].wires, pattern_dag.num_wires - len(node_p[1].wires))

                    # Loop over all possible qubits configurations given the first match constrains
                    for not_fixed_qubits_conf in not_fixed_qubits_confs:
                        for not_fixed_qubits_conf_permuted in itertools.permutations(not_fixed_qubits_conf):
                            not_fixed_qubits_conf_permuted = list(not_fixed_qubits_conf_permuted)
                            for first_match_qubits_conf in _first_match_qubits(node_c[1], node_p[1], pattern_dag.num_wires):
                                qubit_conf = _merge_first_match_permutation(first_match_qubits_conf, not_fixed_qubits_conf_permuted)
                                print(qubit_conf)

        # Different qubit configurations
        # Forward Match
        # Backward match
        # Compatible and optimized maximal matches
        # Create optimized tape

    # Construct optimized circuit
    # for op in tape.operations:
    #    apply(op)

    # Queue the measurements normally
    # for m in tape.measurements:
    #    apply(m)


def _compare_operation_without_qubits(node_1, node_2):
    operation_1 = node_1.op
    operation_2 = node_2.op
    if operation_1.name == operation_2.name:
        if operation_1.num_params == operation_2.num_params:
            if operation_1.data == operation_2.data:
                return True
    return False


def _compare_operation(node_1, node_2):
    operation_1 = node_1.op
    operation_2 = node_2.op
    if operation_1.name == operation_2.name:
        if operation_1.num_params == operation_2.num_params:
            if operation_1.data == operation_2.data:
                if operation_1.is_controlled:
                    if node_1.control_wires == node_2.control_wires:
                        if operation_1.symmetric_over_all_wires:
                            if set(node_1.target_wires) == set(node_2.target_wires):
                                return True
                        else:
                            if node_1.target_wires == node_2.target_wires:
                                return True
                else:
                    if operation_1 in symmetric_over_all_wires:
                        if set(node_1.wires) == set(node_2.wires):
                            return True
                    else:
                        if node_1.wires == node_2.wires:
                            return True
    return False

def _first_match_qubits(node_c, node_p, n_qubits_p):
    """
    Returns the list of qubit for circuit given the first match, the unknown qubit are
    replaced by -1.
    Args:
        node_c (): First match node in the circuit.
        node_p (): First match node in the template.
        n_qubits_p (int): number of qubit in the template.
    Returns:
        list: list of qubits to consider in circuit (with specific order).
    """
    first_match_qubits = []

    # Controlled gate with more than 1 control wire
    if len(node_c.op.control_wires) > 1:
        circuit_control = node_c.op.control_wires
        circuit_target = node_c.op.target_wires
        # Not symmetric target gate (target wires cannot be permuted)
        if node_p.op.is_controlled not in symmetric_over_all_wires:
            # Permute control
            for control_permuted in itertools.permutations(circuit_control):
                control_permuted = list(control_permuted)
                first_match_qubits_sub = [-1] * n_qubits_p
                for q in node_p.wires:
                    node_circuit_perm = control_permuted + circuit_target
                    first_match_qubits_sub[q] = node_circuit_perm[node_p.wires.index(q)]
                first_match_qubits.append(first_match_qubits_sub)
        # Symmetric target gate (target wires cannot be permuted)
        else:
            for control_permuted in itertools.permutations(circuit_control):
                control_permuted = list(control_permuted)
                for target_permuted in itertools.permutations(circuit_target):
                    target_permuted = list(target_permuted)
                    first_match_qubits_sub = [-1] * n_qubits_p
                    for q in node_p.wires:
                        node_circuit_perm = control_permuted + target_permuted
                        first_match_qubits_sub[q] = node_circuit_perm[node_p.wires.index(q)]
                    first_match_qubits.append(first_match_qubits)
    # Not controlled
    else:
        # Not symmetric gate (target wires cannot be permuted)
        if node_p.op.name not in symmetric_over_all_wires:
            first_match_qubits_sub = [-1] * n_qubits_p
            for q in node_p.wires:
                first_match_qubits_sub[q] = node_c.wires[node_p.wires.index(q)]
            first_match_qubits.append(first_match_qubits_sub)
        # Symmetric target gate (target wires cannot be permuted)
        else:
            for perm_q in itertools.permutations(node_c.wires):
                first_match_qubits_sub = [-1] * n_qubits_p
                for q in node_p.wires:
                    first_match_qubits_sub[q] = perm_q[node_p.wires.index(q)]
                first_match_qubits.append(first_match_qubits_sub)
    return first_match_qubits

def _not_fixed_qubits(lst, exclude, length):
    """
    Function that returns all possible combinations of a given length, considering an
    excluded list of elements.
    Args:
        lst (list): list of qubits indices from the circuit.
        exclude (list): list of qubits from the first matched circuit gate.
        length (int): length of the list to be returned (number of template qubit -
        number of qubit from the first matched template gate).
    Yield:
        iterator: Iterator of the possible lists.
    """
    for sublist in itertools.combinations([e for e in lst if e not in exclude], length):
        yield list(sublist)

def _merge_first_match_permutation(list_first_match, permutation):
    """
    Function that returns the list of the circuit qubits and clbits give a permutation
    and an initial match.
    Args:
        list_first_match (list): list of qubits indices for the initial match.
        permutation (list): possible permutation for the circuit qubit.
    Returns:
        list: list of circuit qubit for the given permutation and initial match.
    """
    list_circuit = []

    counter = 0

    for elem in list_first_match:
        if elem == -1:
            list_circuit.append(permutation[counter])
            counter = counter + 1
        else:
            list_circuit.append(elem)

    return list_circuit

