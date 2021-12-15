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
                                qubits_conf = _merge_first_match_permutation(
                                    first_match_qubits_conf, not_fixed_qubits_conf_permuted
                                )
                                wires, target_wires, control_wires = _update_qubits(
                                    circuit_dag, qubits_conf
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

                                backward = BackwardMatch(
                                    circuit_dag,
                                    pattern_dag,
                                    qubits_conf,
                                    forward.match,
                                    forward.circuit_matched_with,
                                    forward.circuit_blocked,
                                    forward.pattern_matched_with,
                                    node_c[0],
                                    node_p[0],
                                    wires,
                                    control_wires,
                                    target_wires,
                                )
                                backward.run_backward_match()
                                if len(backward.match_final[0].match) == 10:
                                    print(backward.match_final[0].match)
                                    print(backward.match_final[1].match)

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
            pattern_dag (): pattern in the dag dependency form.
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

    def _init_matched_with_pattern(self):
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

        if self.pattern_dag.direct_successors(node_id_p):
            maximal_index = self.pattern_dag.direct_successors(node_id_p)[-1]
            for elem in pred:
                if elem > maximal_index:
                    pred.remove(elem)

        block = []
        for node_id in pred:
            for dir_succ in self.pattern_dag.direct_successors(node_id):
                if dir_succ not in matches:
                    succ = self.pattern_dag.successors(dir_succ)
                    block = block + succ
        self.candidates = list(
            set(self.pattern_dag.direct_successors(node_id_p)) - set(matches) - set(block)
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
        self._init_matched_with_pattern()

        self._init_is_blocked_circuit()

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
            self._find_forward_candidates(self.circuit_matched_with[v_first.node_id][0])

            match = False

            # For loop over the candidates (template) to find a match.
            for i in self.candidates:

                # Break the for loop if a match is found.
                if match:
                    break

                # Compare the indices of qubits and the operation,
                # if True; a match is found
                node_circuit = self.circuit_dag.get_node(label)
                node_template = self.pattern_dag.get_node(i)

                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(self.wires[label]) != len(node_template.wires)
                    or set(self.wires[label]) != set(node_template.wires)
                    or node_circuit.op.name != node_template.op.name
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
                        node_template.target_wires,
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


class Match:
    """
    Object to represent a match and its qubit configurations.
    """

    def __init__(self, match, qubit):
        """
        Create a Match class with necessary arguments.
        Args:
            match (list): list of matched gates.
            qubit (list): list of qubits configuration.
        """
        # Match list
        self.match = match
        # Qubits list for circuit
        self.qubit = [qubit]


class MatchingScenarios:
    """
    Class to represent a matching scenario.
    """

    def __init__(
        self, circuit_matched, circuit_blocked, template_matched, template_blocked, matches, counter
    ):
        """
        Create a MatchingScenarios class with necessary arguments.
        Args:
            circuit_matched (list): list of matchedwith attributes in the circuit.
            circuit_blocked (list): list of isblocked attributes in the circuit.
            template_matched (list): list of matchedwith attributes in the template.
            template_blocked (list): list of isblocked attributes in the template.
            matches (list): list of matches.
            counter (int): counter of the number of circuit gates already considered.
        """
        self.circuit_matched = circuit_matched
        self.template_matched = template_matched
        self.circuit_blocked = circuit_blocked
        self.template_blocked = template_blocked
        self.matches = matches
        self.counter = counter


class MatchingScenariosList:
    """
    Object to define a list of MatchingScenarios, with method to append
    and pop elements.
    """

    def __init__(self):
        """
        Create an empty MatchingScenariosList.
        """
        self.matching_scenarios_list = []

    def append_scenario(self, matching):
        """
        Append a scenario to the list.
        Args:
            matching (MatchingScenarios): a scenario of match.
        """
        self.matching_scenarios_list.append(matching)

    def pop_scenario(self):
        """
        Pop the first scenario of the list.
        Returns:
            MatchingScenarios: a scenario of match.
        """
        # Pop the first MatchingScenario and returns it
        first = self.matching_scenarios_list[0]
        self.matching_scenarios_list.pop(0)
        return first


class BackwardMatch:
    """
    Class BackwardMatch allows to run backward direction part of template
    matching algorithm.
    """

    def __init__(
        self,
        circuit_dag,
        pattern_dag,
        qubits_conf,
        forward_matches,
        circuit_matched,
        circuit_blocked,
        pattern_matched,
        node_id_c,
        node_id_p,
        wires,
        control_wires,
        target_wires,
    ):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag (DAGDependency): circuit in the dag dependency form.
            pattern_dag (DAGDependency): template in the dag dependency form.
            forward_matches (list): list of match obtained in the forward direction.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_p (int): index of the first gate matched in the template.
            wires (list):
            control_wires (list):
            target_wires (list):
        """
        self.circuit_dag = circuit_dag
        self.pattern_dag = pattern_dag
        self.circuit_matched = circuit_matched
        self.circuit_blocked = circuit_blocked
        self.pattern_matched = pattern_matched
        self.qubits_conf = qubits_conf
        self.wires = wires
        self.control_wires = control_wires
        self.target_wires = target_wires
        self.node_id_c = node_id_c
        self.node_id_p = node_id_p
        self.forward_matches = forward_matches
        self.match_final = []
        self.matching_list = MatchingScenariosList()

    def _gate_indices(self, circuit_matched, circuit_blocked):
        """
        Function which returns the list of gates that are not match and not
        blocked for the first scenario.
        Returns:
            list: list of gate id.
        """
        gate_indices = []

        for i, (matched, blocked) in enumerate(zip(circuit_matched, circuit_blocked)):
            if (not matched) and (not blocked):
                gate_indices.append(i)
        gate_indices.reverse()
        return gate_indices

    def _find_backward_candidates(self, template_blocked, matches):
        """
        Function which returns the list possible backward candidates in the template dag.
        Args:
            template_blocked (list): list of attributes isblocked in the template circuit.
            matches (list): list of matches.
        Returns:
            list: list of backward candidates (id).
        """
        template_block = []

        for node_id in range(self.node_id_p, self.pattern_dag.size):
            if template_blocked[node_id]:
                template_block.append(node_id)

        matches_template = sorted(match[0] for match in matches)

        successors = self.pattern_dag.get_node(self.node_id_p).successors
        potential = []
        for index in range(self.node_id_p + 1, self.pattern_dag.size):
            if (index not in successors) and (index not in template_block):
                potential.append(index)

        candidates_indices = list(set(potential) - set(matches_template))
        candidates_indices = sorted(candidates_indices)
        candidates_indices.reverse()

        return candidates_indices

    def _backward_metrics(self, scenario):
        """
        Heuristics to cut the tree in the backward match algorithm.
        Args:
            scenario (MatchingScenarios): scenario for the given match.
        Returns:
            int: length of the match for the given scenario.
        """
        return len(scenario.matches)

    def run_backward_match(self):
        """
        Apply the forward match algorithm and returns the list of matches given an initial match
        and a circuit qubits configuration.
        """
        match_store_list = []

        counter = 1

        # Initialize the list of attributes matchedwith and isblocked.

        pattern_blocked = [False] * self.pattern_dag.size

        # First Scenario is stored in the MatchingScenariosList().
        first_match = MatchingScenarios(
            self.circuit_matched,
            self.circuit_blocked,
            self.pattern_matched,
            pattern_blocked,
            self.forward_matches,
            counter,
        )

        self.matching_list = MatchingScenariosList()
        self.matching_list.append_scenario(first_match)

        # Set the circuit indices that can be matched.
        gate_indices = self._gate_indices(self.circuit_matched, self.circuit_blocked)

        number_of_gate_to_match = (
            self.pattern_dag.size - (self.node_id_p - 1) - len(self.forward_matches)
        )

        # While the scenario stack is not empty.
        while self.matching_list.matching_scenarios_list:

            scenario = self.matching_list.pop_scenario()

            circuit_matched = scenario.circuit_matched
            circuit_blocked = scenario.circuit_blocked
            template_matched = scenario.template_matched
            template_blocked = scenario.template_blocked
            matches_scenario = scenario.matches
            counter_scenario = scenario.counter

            # Part of the match list coming from the backward match.
            match_backward = [
                match for match in matches_scenario if match not in self.forward_matches
            ]

            # Matches are stored if the counter is bigger than the length of the list of
            # candidates in the circuit. Or if number of gate left to match is the same as
            # the length of the backward part of the match.
            if (
                counter_scenario > len(gate_indices)
                or len(match_backward) == number_of_gate_to_match
            ):
                matches_scenario.sort(key=lambda x: x[0])
                match_store_list.append(Match(matches_scenario, self.qubits_conf))
                continue

            # First circuit candidate.
            circuit_id = gate_indices[counter_scenario - 1]
            node_circuit = self.circuit_dag.get_node(circuit_id)

            # If the circuit candidate is blocked, only the counter is changed.
            if circuit_blocked[circuit_id]:
                matching_scenario = MatchingScenarios(
                    circuit_matched,
                    circuit_blocked,
                    template_matched,
                    template_blocked,
                    matches_scenario,
                    counter_scenario + 1,
                )
                self.matching_list.append_scenario(matching_scenario)
                continue

            # The candidates in the template.
            candidates_indices = self._find_backward_candidates(template_blocked, matches_scenario)
            # Update of the qubits/clbits indices in the circuit in order to be
            # comparable with the one in the template.

            wires1 = self.wires[circuit_id]
            control_wires1 = self.control_wires[circuit_id]
            target_wires1 = self.target_wires[circuit_id]

            global_match = False
            global_broken = []

            # Loop over the template candidates.
            for template_id in candidates_indices:

                node_template = self.pattern_dag.get_node(template_id)
                wires2 = self.pattern_dag.get_node(template_id).wires
                control_wires2 = self.pattern_dag.get_node(template_id).control_wires
                target_wires2 = self.pattern_dag.get_node(template_id).target_wires
                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(wires1) != len(wires2)
                    or set(wires1) != set(wires2)
                    or node_circuit.op.name != node_template.op.name
                ):
                    continue

                # Check if the qubit, clbit configuration are compatible for a match,
                # also check if the operation are the same.
                if _compare_operation_without_qubits(node_circuit, node_template):
                    if _compare_qubits(
                        node_circuit,
                        wires1,
                        control_wires1,
                        target_wires1,
                        wires2,
                        control_wires2,
                        target_wires2,
                    ):

                        # If there is a match the attributes are copied.
                        circuit_matched_match = circuit_matched.copy()
                        circuit_blocked_match = circuit_blocked.copy()

                        template_matched_match = template_matched.copy()
                        template_blocked_match = template_blocked.copy()

                        matches_scenario_match = matches_scenario.copy()

                        block_list = []
                        broken_matches_match = []

                        # Loop to check if the match is not connected, in this case
                        # the successors matches are blocked and unmatched.
                        for potential_block in self.pattern_dag.successors(template_id):
                            if not template_matched_match[potential_block]:
                                template_blocked_match[potential_block] = True
                                block_list.append(potential_block)
                                for block_id in block_list:
                                    for succ_id in self.pattern_dag.successors(block_id):
                                        template_blocked_match[succ_id] = True
                                        if template_matched_match[succ_id]:
                                            new_id = template_matched_match[succ_id][0]
                                            circuit_matched_match[new_id] = []
                                            template_matched_match[succ_id] = []
                                            broken_matches_match.append(succ_id)

                        if broken_matches_match:
                            global_broken.append(True)
                        else:
                            global_broken.append(False)

                        new_matches_scenario_match = [
                            elem
                            for elem in matches_scenario_match
                            if elem[0] not in broken_matches_match
                        ]

                        condition = True

                        for back_match in match_backward:
                            if back_match not in new_matches_scenario_match:
                                condition = False
                                break

                        # First option greedy match.
                        if ([self.node_id_p, self.node_id_c] in new_matches_scenario_match) and (
                            condition or not match_backward
                        ):
                            template_matched_match[template_id] = [circuit_id]
                            circuit_matched_match[circuit_id] = [template_id]
                            new_matches_scenario_match.append([template_id, circuit_id])

                            new_matching_scenario = MatchingScenarios(
                                circuit_matched_match,
                                circuit_blocked_match,
                                template_matched_match,
                                template_blocked_match,
                                new_matches_scenario_match,
                                counter_scenario + 1,
                            )
                            self.matching_list.append_scenario(new_matching_scenario)

                            global_match = True

            if global_match:
                circuit_matched_block_s = circuit_matched.copy()
                circuit_blocked_block_s = circuit_blocked.copy()

                template_matched_block_s = template_matched.copy()
                template_blocked_block_s = template_blocked.copy()

                matches_scenario_block_s = matches_scenario.copy()

                circuit_blocked_block_s[circuit_id] = True

                broken_matches = []

                # Second option, not a greedy match, block all successors (push the gate
                # to the right).
                for succ in self.circuit_dag.get_node(circuit_id).successors:
                    circuit_blocked_block_s[succ] = True
                    if circuit_matched_block_s[succ]:
                        broken_matches.append(succ)
                        new_id = circuit_matched_block_s[succ][0]
                        template_matched_block_s[new_id] = []
                        circuit_matched_block_s[succ] = []

                new_matches_scenario_block_s = [
                    elem for elem in matches_scenario_block_s if elem[1] not in broken_matches
                ]

                condition_not_greedy = True

                for back_match in match_backward:
                    if back_match not in new_matches_scenario_block_s:
                        condition_not_greedy = False
                        break

                if ([self.node_id_p, self.node_id_c] in new_matches_scenario_block_s) and (
                    condition_not_greedy or not match_backward
                ):
                    new_matching_scenario = MatchingScenarios(
                        circuit_matched_block_s,
                        circuit_blocked_block_s,
                        template_matched_block_s,
                        template_blocked_block_s,
                        new_matches_scenario_block_s,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(new_matching_scenario)

                # Third option: if blocking the succesors breaks a match, we consider
                # also the possibility to block all predecessors (push the gate to the left).
                if broken_matches and all(global_broken):

                    circuit_matched_block_p = circuit_matched.copy()
                    circuit_blocked_block_p = circuit_blocked.copy()

                    template_matched_block_p = template_matched.copy()
                    template_blocked_block_p = template_blocked.copy()

                    matches_scenario_block_p = matches_scenario.copy()

                    circuit_blocked_block_p[circuit_id] = True

                    for pred in self.circuit_dag.get_node(circuit_id).predecessors:
                        circuit_blocked_block_p[pred] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched_block_p,
                        circuit_blocked_block_p,
                        template_matched_block_p,
                        template_blocked_block_p,
                        matches_scenario_block_p,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

            # If there is no match then there are three options.
            if not global_match:

                circuit_blocked[circuit_id] = True

                following_matches = []

                successors = self.circuit_dag.get_node(circuit_id).successors
                for succ in successors:
                    if circuit_matched[succ]:
                        following_matches.append(succ)

                # First option, the circuit gate is not disturbing because there are no
                # following match and no predecessors.
                predecessors = self.circuit_dag.get_node(circuit_id).predecessors

                if not predecessors or not following_matches:

                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        template_matched,
                        template_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

                else:

                    circuit_matched_nomatch = circuit_matched.copy()
                    circuit_blocked_nomatch = circuit_blocked.copy()

                    template_matched_nomatch = template_matched.copy()
                    template_blocked_nomatch = template_blocked.copy()

                    matches_scenario_nomatch = matches_scenario.copy()

                    # Second option, all predecessors are blocked (circuit gate is
                    # moved to the left).
                    for pred in predecessors:
                        circuit_blocked[pred] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        template_matched,
                        template_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

                    # Third option, all successors are blocked (circuit gate is
                    # moved to the right).

                    broken_matches = []

                    successors = self.circuit_dag.get_node(circuit_id).successors

                    for succ in successors:
                        circuit_blocked_nomatch[succ] = True
                        if circuit_matched_nomatch[succ]:
                            broken_matches.append(succ)
                            circuit_matched_nomatch[succ] = []

                    new_matches_scenario_nomatch = [
                        elem for elem in matches_scenario_nomatch if elem[1] not in broken_matches
                    ]

                    condition_block = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario_nomatch:
                            condition_block = False
                            break

                    if ([self.node_id_p, self.node_id_c] in matches_scenario_nomatch) and (
                        condition_block or not match_backward
                    ):
                        new_matching_scenario = MatchingScenarios(
                            circuit_matched_nomatch,
                            circuit_blocked_nomatch,
                            template_matched_nomatch,
                            template_blocked_nomatch,
                            new_matches_scenario_nomatch,
                            counter_scenario + 1,
                        )
                        self.matching_list.append_scenario(new_matching_scenario)

        length = max(len(m.match) for m in match_store_list)

        # Store the matches with maximal length.
        for scenario in match_store_list:
            if (len(scenario.match) == length) and not any(
                scenario.match == x.match for x in self.match_final
            ):
                self.match_final.append(scenario)
