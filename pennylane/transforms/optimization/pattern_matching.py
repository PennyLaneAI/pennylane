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
                f"The pattern {pattern}, does not appear " "to be a valid quantum tape"
            )

        # Check that it does not contain a measurement.
        if pattern.measurements:
            raise qml.QuantumFunctionError(f"The pattern {pattern}, contains measurements. ")

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
                    not_fixed_qubits_confs = _not_fixed_qubits(
                        circuit_range, node_c[1].wires, pattern_dag.num_wires - len(node_p[1].wires)
                    )

                    # Loop over all possible qubits configurations given the first match constrains
                    for not_fixed_qubits_conf in not_fixed_qubits_confs:
                        for not_fixed_qubits_conf_permuted in itertools.permutations(
                            not_fixed_qubits_conf
                        ):
                            not_fixed_qubits_conf_permuted = list(not_fixed_qubits_conf_permuted)
                            for first_match_qubits_conf in _first_match_qubits(
                                node_c[1], node_p[1], pattern_dag.num_wires
                            ):
                                qubit_conf = _merge_first_match_permutation(
                                    first_match_qubits_conf, not_fixed_qubits_conf_permuted
                                )
                                print(qubit_conf)
                                wires, target_wires, control_wires = _update_qubits(
                                    circuit_dag, qubit_conf
                                )
                                forward = ForwardMatch(
                                    circuit_dag,
                                    pattern_dag,
                                    node_c[0],
                                    node_p[0],
                                    wires,
                                    target_wires,
                                    control_wires,
                                )
                                forward.run_forward_match()

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


def _update_qubits(circuit_dag, qubits_conf):
    """
    Change qubits indices of the current circuit node in order to
    be comparable with the indices of the template qubits list.
    Args:
        qarg (list): list of qubits indices from the circuit for a given node.
    """
    wires = []
    control_wires = []
    target_wires = []

    for i, node in enumerate(circuit_dag.get_nodes()):
        # Wires
        wires.append([])
        for q in node[1].wires:
            if q in qubits_conf:
                wires[i].append(qubits_conf.index(q))
        if len(node[1].wires) != len(wires[i]):
            wires[i] = []

        # Control wires
        control_wires.append([])
        for q in node[1].control_wires:
            if q in qubits_conf:
                control_wires[i].append(qubits_conf.index(q))
        if len(node[1].control_wires) != len(control_wires[i]):
            control_wires[i] = []

        # Target wires
        target_wires.append([])
        for q in node[1].target_wires:
            if q in qubits_conf:
                target_wires[i].append(qubits_conf.index(q))
        if len(node[1].target_wires) != len(target_wires[i]):
            target_wires[i] = []

    return wires, target_wires, control_wires


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
                        if operation_1.name in symmetric_over_all_wires:
                            if set(node_1.target_wires) == set(node_2.target_wires):
                                return True
                        else:
                            if node_1.target_wires == node_2.target_wires:
                                return True
                else:
                    if operation_1.name in symmetric_over_all_wires:
                        if set(node_1.wires) == set(node_2.wires):
                            return True
                    else:
                        if node_1.wires == node_2.wires:
                            return True
    return False


def _compare_qubits(node1, wires1, control1, target1, wires2, control2, target2):

    if control1:
        if set(control1) == set(control2):
            if node1.op.name in symmetric_over_all_wires:
                if set(target1) == set(target2):
                    return True
            else:
                if target1 == target2:
                    return True
    else:
        if node1.op.name in symmetric_over_all_wires:
            if set(wires1) == set(wires2):
                return True
        else:
            if wires1 == wires2:
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


class ForwardMatch:
    """
    Object to apply template matching in the forward direction.
    """

    def __init__(
        self, circuit_dag, pattern_dag, node_id_c, node_id_p, wires, control_wires, target_wires
    ):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag (): circuit in the dag dependency form.
            template_dag (): template in the dag dependency form.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_p (int): index of the first gate matched in the pattern.
        """

        # The dag dependency representation of the circuit
        self.circuit_dag = circuit_dag

        # The dag dependency representation of the template
        self.pattern_dag = pattern_dag

        # Id of the node in the circuit
        self.node_id_c = node_id_c

        # Id of the node in the template
        self.node_id_p = node_id_p

        # Successors to visit attribute in the circuit
        self.successors_to_visit = [None] * circuit_dag.size

        # Blocked nodes in the circuit
        self.circuit_blocked = [None] * circuit_dag.size

        # Blocked nodes in the pattern
        self.pattern_blocked = [None] * pattern_dag.size

        # Matched nodes circuit
        self.circuit_matched_with = [None] * circuit_dag.size

        # Matched nodes pattern
        self.pattern_matched_with = [None] * pattern_dag.size

        # Updated qubits for each node
        self.updated_qubits = []

        # List of match
        self.match = []

        # List of candidates for the forward match
        self.candidates = []

        # List of nodes in circuit which are matched
        self.matched_nodes_list = []

        self.wires = wires

        self.control_wires = control_wires

        self.target_wires = target_wires

    def _init_successors_to_visit(self):
        """
        Initialize the list of successors to visit.
        """
        for i in range(0, self.circuit_dag.size):
            if i == self.node_id_c:
                self.successors_to_visit[i] = self.circuit_dag.direct_successors(i)
            else:
                self.successors_to_visit[i] = []

    def _init_matched_with_circuit(self):
        """
        Initialize the list of corresponding matches in the pattern for the circuit.
        """
        for i in range(0, self.circuit_dag.size):
            if i == self.node_id_c:
                self.circuit_matched_with[i] = [self.node_id_p]
            else:
                self.circuit_matched_with[i] = []

    def _init_matched_with_template(self):
        """
        Initialize the list of corresponding matches in the circuit for the pattern.
        """
        for i in range(0, self.pattern_dag.size):
            if i == self.node_id_p:
                self.pattern_matched_with[i] = [self.node_id_c]
            else:
                self.pattern_matched_with[i] = []

    def _init_is_blocked_circuit(self):
        """
        Initialize the list of blocked nodes in the circuit.
        """
        for i in range(0, self.circuit_dag.size):
            self.circuit_blocked[i] = False

    def _init_is_blocked_template(self):
        """
        Initialize the list of blocked nodes in the pattern.
        """
        for i in range(0, self.pattern_dag.size):
            self.pattern_blocked[i] = False

    def _init_list_match(self):
        """
        Initialize the list of matched nodes between the circuit and the template
        with the first match found.
        """
        self.match.append([self.node_id_p, self.node_id_c])

    def _find_forward_candidates(self, node_id_p):
        """
        Find the candidate nodes to be matched in the template for a given node.
        Args:
            node_id_p (int): considered node id.
        """
        matches = []

        for i in range(0, len(self.match)):
            matches.append(self.match[i][0])

        pred = matches.copy()
        if len(pred) > 1:
            pred.sort()
        pred.remove(node_id_p)

        if self.template_dag.direct_successors(node_id_p):
            maximal_index = self.template_dag.direct_successors(node_id_p)[-1]
            for elem in pred:
                if elem > maximal_index:
                    pred.remove(elem)

        block = []
        for node_id in pred:
            for dir_succ in self.template_dag.direct_successors(node_id):
                if dir_succ not in matches:
                    succ = self.template_dag.successors(dir_succ)
                    block = block + succ
        self.candidates = list(
            set(self.template_dag.direct_successors(node_id_p)) - set(matches) - set(block)
        )

    def _init_matched_nodes(self):
        """
        Initialize the list of current matched nodes.
        """
        self.matched_nodes_list.append(
            [
                self.node_id_c,
                self.circuit_dag.get_node(self.node_id_c),
                self.successors_to_visit[self.node_id_c],
            ]
        )

    def _get_node_forward(self, list_id):
        """
        Return a node from the matched_node_list for a given list id.
        Args:
            list_id (int): considered list id of the desired node.
        Returns:
            DAGDepNode: DAGDepNode object corresponding to i-th node of the matched_node_list.
        """
        node = self.matched_nodes_list[list_id][1]
        succ = self.matched_nodes_list[list_id][2]
        return node, succ

    def _remove_node_forward(self, list_id):
        """
        Remove a node of the current matched list for a given list id.
        Args:
            list_id (int): considered list id of the desired node.
        """
        self.matched_nodes_list.pop(list_id)

    def run_forward_match(self):
        """
        Apply the forward match algorithm and returns the list of matches given an initial match
        and a circuit qubits configuration.
        """

        # Initialize the new attributes of the DAGDepNodes of the DAGDependency object
        self._init_successors_to_visit()

        self._init_matched_with_circuit()
        self._init_matched_with_template()

        self._init_is_blocked_circuit()
        self._init_is_blocked_template()

        # Initialize the list of matches and the stack of matched nodes (circuit)
        self._init_list_match()
        self._init_matched_nodes()

        # While the list of matched nodes is not empty
        while self.matched_nodes_list:

            # Return first element of the matched_nodes_list and removes it from the list
            v_first, successors_to_visit = self._get_node_forward(0)
            self._remove_node_forward(0)

            # If there is no successors to visit go to the end
            if not successors_to_visit:
                continue

            # Get the label and the node of the first successor to visit
            label = successors_to_visit[0]
            v = [label, self.circuit_dag.get_node(label)]

            # Update of the SuccessorsToVisit attribute
            successors_to_visit.pop(0)

            # Update the matched_nodes_list with new attribute successor to visit and sort the list.
            self.matched_nodes_list.append([v_first.node_id, v_first, successors_to_visit])
            self.matched_nodes_list.sort(key=lambda x: x[2])

            # If the node is blocked and already matched go to the end
            if self.circuit_blocked[v[0]] | (self.circuit_matched_with[v[0]] != []):
                continue

            # Search for potential candidates in the template
            self._find_forward_candidates(successors_to_visit[0])

            match = False

            # For loop over the candidates (template) to find a match.
            for i in self.candidates:

                # Break the for loop if a match is found.
                if match:
                    break

                # Compare the indices of qubits and the operation,
                # if True; a match is found
                node_circuit = self.circuit_dag.get_node(label)
                node_template = self.template_dag.get_node(i)

                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(self.wires[label]) != len(node_template.wires)
                    or set(self.wires[label]) != set(node_template.wires)
                    or node_circuit.name != node_template.name
                ):
                    continue

                # Check if the qubit, clbit configuration are compatible for a match,
                # Check if the operations are the same.
                if _compare_operation_without_qubits(node_circuit, node_template):
                    if _compare_qubits(
                        node_circuit,
                        self.wires[label],
                        self.target_wires[label],
                        self.control_wires[label],
                        node_template.wires,
                        node_template.control_wires,
                        node_template.trget_wires,
                    ):
                        # Check if the qubit, clbit configuration are compatible for a match,
                        self.circuit_matched_with[label] = [i]

                        self.pattern_matched_with[i] = [label]

                        # Append the new match to the list of matches.
                        self.match.append([i, label])

                        # Potential successors to visit (circuit) for a given match.
                        potential = self.circuit_dag.direct_successors(label)

                        # If the potential successors to visit are blocked or match, it is removed.
                        for potential_id in potential:
                            if self.circuit_blocked[potential_id] | (
                                self.circuit_matched_with[potential_id] != []
                            ):
                                potential.remove(potential_id)

                        sorted_potential = sorted(potential)

                        #  Update the successor to visit attribute
                        successorstovisit = sorted_potential

                        # Add the updated node to the stack.
                        self.matched_nodes_list.append([v[0], v[1], successorstovisit])
                        self.matched_nodes_list.sort(key=lambda x: x[2])
                        match = True
                        continue

            # If no match is found, block the node and all the successors.
            if not match:
                self.circuit_blocked[label] = True
                for succ in v[1].successors:
                    self.circuit_blocked[succ] = True
                    if self.circuit_matched_with[succ]:
                        self.match.remove([self.circuit_matched_with[succ][0], succ])
                        match_id = self.circuit_matched_with[succ][0]
                        self.pattern_matched_with[match_id] = []
                        self.circuit_matched_with[succ] = []
