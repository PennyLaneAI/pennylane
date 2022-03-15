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
"""Transform finding all maximal matches of a pattern in a quantum circuit and optimizing the circuit by
substitution."""

import itertools
import copy
from collections import OrderedDict

import pennylane as qml
from pennylane import apply
from pennylane.transforms import qfunc_transform
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
from pennylane.transforms.decompositions import two_qubit_decomposition, zyz_decomposition


@qfunc_transform
def sequences_optimization(tape, n_qubits):
    r"""Quantum function transform to optimize a circuit given a list of patterns (templates).

    Args:
        qfunc (function): A quantum function to be optimized.
        n_qubits(list(int)): List of number of qubits [1], [2], [1, 2], [2, 1]

    Returns:
        function: the transformed quantum function

    Raises:
        QuantumFunctionError: The pattern provided is not a valid QuantumTape or the pattern contains measurements or
            the pattern does not implement identity or the circuit has less qubits than the pattern.

    **Example**

    Consider the following quantum circuit to be optimized

    .. code-block:: python

        def circuit():
            qml.PauliX(wires=2)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires = 1)
            qml.CNOT(wires=[1, 2])
            qml.Hadamard(wires = 1)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[0, 2])
            qml.Hadamard(wires = 2)
            qml.CNOT(wires=[2, 1])
            qml.Hadamard(wires = 2)
            qml.CNOT(wires=[2, 1])
            qml.PauliX(wires=1)
            return qml.expval(qml.PauliX(wires=0))

    For optimizing the circuit given the given following template of CNOTs we apply the `pattern_matching`
    transform.

    >>> dev = qml.device('default.qubit', wires=3)
    >>> qnode = qml.QNode(circuit, dev)
    >>> optimized_qfunc = sequences_optimization(n_qubit=[2])(circuit)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

    In our case, it is possible to reduce 4 CNOTs to only two CNOTs and therefore
    optimizing the circuit. The number of CNOTs in the circuit is reduced by one.

    >>> qml.specs(qnode)()["gate_types"]["CNOT"]
    4

    >>> qml.specs(optimized_qnode)()["gate_types"]["CNOT"]
    2

    >>> print(qml.draw(qnode)())
    0: ────────────────╭C─╭C────────────────┤  <X>
    1: ────╭C──H─╭C──H─╰X─│─────╭X────╭X──X─┤
    2: ──X─╰X────╰X───────╰X──H─╰C──H─╰C────┤

    >>> print(qml.draw(optimized_qnode)())
    0: ─╭C───────────────────────────────────────────────────────╭C─┤  <X>
    1: ─│───RZ(1.57)─────────────╭C──SX─╭C──Rot(-1.57,1.57,1.57)─╰X─┤
    2: ─╰X──Rot(0.00,1.57,-1.57)─╰X──S──╰X──Rot(0.00,1.57,-1.57)────┤

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2022.
    Exact and practical pattern matching for quantum circuit optimization.
    `doi.org/10.1145/3498325 <https://dl.acm.org/doi/abs/10.1145/3498325>`_
    """
    # pylint: disable=protected-access, too-many-branches

    measurements = tape.measurements
    observables = tape.observables

    consecutive_wires = Wires(range(len(tape.wires)))
    inverse_wires_map = OrderedDict(zip(consecutive_wires, tape.wires))

    # Check the validity of number of qubits
    if n_qubits not in [[1], [2], [1, 2], [2, 1]]:
        raise qml.QuantumFunctionError(
            "The list of number of qubits is not a valid. It should be [1] or [2] or ["
            "1, 2] or [2, 1]"
        )

    for n_qubit in n_qubits:
        # Construct Dag representation of the circuit and the pattern.
        circuit_dag = commutation_dag(tape)()

        max_sequences = maximal_sequences(circuit_dag, n_qubit)

        # Optimizes the circuit for compatible maximal matches
        if max_sequences:
            # Initialize the optimization by substitution of the different matches
            substitution = SequencesSubstitution(max_sequences, circuit_dag, n_qubit)
            substitution.substitution()
            already_sub = []

            # If some substitutions are possible, we create an optimized circuit.
            if substitution.substitution_list:
                # Create a tape that does not affect the outside context.
                with qml.tape.QuantumTape(do_queue=False) as tape_inside:
                    # Loop over all possible substitutions
                    for group in substitution.substitution_list:

                        circuit_sub = group.circuit_config
                        subtape_operations = group.subtape_opt.operations

                        pred = group.pred_block

                        # Choose the first configuration
                        qubit = group.qubit_config[0]

                        # First add all the predecessors of the given match.
                        for elem in pred:
                            node = circuit_dag.get_node(elem)
                            inst = copy.deepcopy(node.op)
                            apply(inst)
                            already_sub.append(elem)

                        already_sub = already_sub + circuit_sub

                        # Then add the inverse of the template.
                        for op in subtape_operations:
                            apply(op)

                    # Add the unmatched gates.
                    for node_id in substitution.unmatched_list:
                        node = circuit_dag.get_node(node_id)
                        inst = copy.deepcopy(node.op)
                        apply(inst)

                tape = tape_inside

    for op in tape.operations:
        op._wires = Wires([inverse_wires_map[wire] for wire in op.wires.tolist()])
        apply(op)

        # After optimization, simply apply the measurements
    for obs in observables:
        obs._wires = Wires([inverse_wires_map[wire] for wire in obs.wires.tolist()])

    for m in measurements:
        apply(m)


def maximal_sequences(circuit_dag, n_qubit_subset):
    r"""Function that applies the pattern matching algorithm and returns the list of maximal matches.

    Args:
        circuit_dag (.CommutationDAG): A commutation DAG representing the circuit to be optimized.
        n_qubit_subset(int): Number of qubits that should be considered in the subset.

    Returns:
        list(Match): the list of maximal sequences.
    """
    # Check the validity of number of qubits
    if n_qubit_subset >= circuit_dag.num_wires:
        raise qml.QuantumFunctionError(
            "The qubits subset considered must be smaller than the number of qubit in the "
            "circuit."
        )

    # Match list
    sequence_list = []

    # Loop through all possible initial matches
    for node_c in circuit_dag.get_nodes():
        if _compare_operation_qubits_number(node_c[1], n_qubit_subset):
            # Fix qubits from the first (target fixed and control restrained)
            not_fixed_qubits_confs = _not_fixed_qubits(
                circuit_dag.num_wires, node_c[1].wires, n_qubit_subset - len(node_c[1].wires)
            )
            # Loop over all possible qubits configurations given the first match constrains
            for not_fixed_qubits_conf in not_fixed_qubits_confs:
                for not_fixed_qubits_conf_permuted in itertools.permutations(not_fixed_qubits_conf):
                    first_match_qubits_conf = _first_match_qubits(node_c[1], n_qubit_subset)
                    # Qubit configuration given the first operation and number of qubits.
                    qubits_conf = _merge_first_match_and_permutation(
                        first_match_qubits_conf, not_fixed_qubits_conf_permuted
                    )

                    qubits_conf = set(qubits_conf)
                    # Forward match part of the algorithm
                    forward = ForwardSequence(
                        circuit_dag,
                        node_c[0],
                        qubits_conf,
                    )
                    forward.run_forward_match()

                    # Backward match part of the algorithm
                    backward = BackwardSequence(
                        circuit_dag,
                        forward.sequence,
                        forward.circuit_matched,
                        forward.circuit_blocked,
                        node_c[0],
                        qubits_conf,
                    )
                    backward.run_backward_match()

                    _add_sequence(sequence_list, backward.final_sequences)

    sequence_list.sort(key=lambda x: len(x.sequence), reverse=True)

    # Extract maximal matches and optimizes the circuit for compatible maximal matches
    if sequence_list:
        maximal = MaximalSequences(sequence_list)
        maximal.run_maximal_sequences()
        max_matches = maximal.max_sequences_list
        return max_matches

    return sequence_list


def _compare_operation_qubits_number(node_1, n_qubit):
    """Compare two operations without taking the qubits into account.

    Args:
        node_1 (.CommutationDAGNode): First operation.
        n_qubit (int): Number of qubits, one or two.
    Return:
        Bool: True if the operation has as much or less qubits than the desired number.
    """
    return len(node_1.op.wires) <= n_qubit


def _not_fixed_qubits(n_qubits_circuit, exclude, length):
    """
    Function that returns all possible combinations of qubits given some restrictions and using itertools.
    Args:
        n_qubits_circuit (int): Number of qubit in the circuit.
        exclude (list): list of qubits from the first matched circuit operation that needs to be excluded.
        length (int): number of qubits.
    Yield:
        iterator: Iterator of the possible lists.
    """
    circuit_range = range(0, n_qubits_circuit)
    for sublist in itertools.combinations([e for e in circuit_range if e not in exclude], length):
        yield list(sublist)


def _first_match_qubits(node_c, n_qubit):
    """
    Returns the list of qubits for circuit given the first match, the unknown qubit are
    replaced by -1.
    Args:
        node_c (.CommutationDAGNode): First node in the circuit.
        n_qubit (int):

    Returns:
        list: list of qubits to consider in circuit (with specific order).
    """
    wires = node_c.wires
    list_first_node = [-1] * n_qubit
    for q in range(0, len(wires)):
        list_first_node[q] = wires[q]
    return list_first_node


def _merge_first_match_and_permutation(list_first_match, permutation):
    """
    Function that returns the final qubits configuration given the first match constraints and the permutation of
    qubits not in the first match.

    Args:
        list_first_match (list): list of qubits indices for the first match.
        permutation (list): possible permutation for the circuit qubits not in the first match.

    Returns:
        list: list of circuit qubits for the given permutation and constraints from the initial match.
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


def _add_sequence(sequence_list, backward_sequence_list):
    """
    Add a match configuration found by pattern matching if it is not already in final list of matches.
    If the match is already in the final list, the qubit configuration is added to the existing Match.
    Args:
        match_list (list): match from the backward part of the algorithm.
        backward_match_list (list): List of matches found by the algorithm for a given configuration.
    """

    already_in = False

    for b_match in backward_sequence_list:
        for l_match in sequence_list:
            if b_match.sequence == l_match.sequence:
                index = sequence_list.index(l_match)
                sequence_list[index].qubit.append(b_match.qubit[0])
                already_in = True

        if not already_in:
            sequence_list.append(b_match)


class ForwardSequence:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Class to apply pattern matching in the forward direction.
    """

    def __init__(self, circuit_dag, node_id_c, qubits_conf):
        """
        Create the ForwardMatch class.
        Args:
            circuit_dag (.CommutationDAG): circuit as commutation DAG.

        """
        # pylint: disable=too-many-branches, too-many-arguments

        # Commutation DAG of the circuit
        self.circuit_dag = circuit_dag

        # Node ID in the circuit
        self.node_id_c = node_id_c

        # List of wires
        self.wires = qubits_conf

        # Successors to visit
        self.successors_to_visit = [None] * circuit_dag.size

        # Blocked nodes in the circuit
        self.circuit_blocked = [None] * circuit_dag.size

        # Matched nodes circuit
        self.circuit_matched = [None] * circuit_dag.size

        # List of sequence
        self.sequence = []

        # List of nodes in circuit which are in the sequence
        self.sequence_nodes_list = []

    def _init_successors_to_visit(self):
        """
        Initialize the list of successors to visit.
        """
        for i in range(0, self.circuit_dag.size):
            if i == self.node_id_c:
                self.successors_to_visit[i] = self.circuit_dag.direct_successors(i)
            else:
                self.successors_to_visit[i] = []

    def _init_circuit_matched(self):
        """
        Initialize the list of corresponding matches in the pattern for the circuit.
        """
        for i in range(0, self.circuit_dag.size):
            if i == self.node_id_c:
                self.circuit_matched[i] = True
            else:
                self.circuit_matched[i] = False

    def _init_circuit_blocked(self):
        """
        Initialize the list of blocked nodes in the circuit.
        """
        for i in range(0, self.circuit_dag.size):
            self.circuit_blocked[i] = False

    def _init_list_sequence(self):
        """
        Initialize the list of matched nodes between the circuit and the pattern with the first match found.
        """
        self.sequence.append(self.node_id_c)

    def _init_sequence_nodes(self):
        """
        Initialize the list of current matched nodes.
        """
        self.sequence_nodes_list.append(
            [
                self.node_id_c,
                self.circuit_dag.get_node(self.node_id_c),
                self.successors_to_visit[self.node_id_c],
            ]
        )

    def _get_node_forward(self, list_id):
        """
        Return node and successors from the matched_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        Returns:
            CommutationDAGNode: Node from the matched_node_list.
            list(int): List of successors.
        """
        node = self.sequence_nodes_list[list_id][1]
        succ = self.sequence_nodes_list[list_id][2]
        return node, succ

    def _remove_node_forward(self, list_id):
        """Remove a node of the current matched_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        """
        self.sequence_nodes_list.pop(list_id)

    def run_forward_match(self):
        """Apply the forward match algorithm and returns the list of matches given an initial match
        and a qubits configuration.
        """
        # Initialization
        self._init_successors_to_visit()
        self._init_circuit_matched()
        self._init_circuit_blocked()

        # Initialize the list of matches and the stack of matched nodes (circuit)
        self._init_list_sequence()
        self._init_sequence_nodes()

        # While the list of matched nodes is not empty
        while self.sequence_nodes_list:
            # Return first element of the matched_nodes_list and removes it from the list
            v_first, successors_to_visit = self._get_node_forward(0)
            self._remove_node_forward(0)

            # If there is no successors to visit go to the end
            if not successors_to_visit:
                continue

            # Get the label and the node of the first successor to visit
            label = successors_to_visit[0]
            v = [label, self.circuit_dag.get_node(label)]

            # Update of the successors to visit.
            successors_to_visit.pop(0)

            # Update the matched_nodes_list with new attribute successor to visit and sort the list.
            self.sequence_nodes_list.append([v_first.node_id, v_first, successors_to_visit])
            self.sequence_nodes_list.sort(key=lambda x: x[2])

            # If the node is blocked and already matched go to the end
            if self.circuit_blocked[label] or self.circuit_matched[v[0]]:
                continue

            if set(v[1].wires).issubset(self.wires):
                # A match happens
                self.circuit_matched[label] = True

                # Append the new match to the list of matches.
                self.sequence.append(label)

                # Potential successors to visit (circuit) for a given match.
                potential = self.circuit_dag.direct_successors(label)
                # If the potential successors to visit are blocked or match, it is removed.
                for potential_id in potential:
                    if self.circuit_blocked[potential_id] or (self.circuit_matched[potential_id]):
                        potential.remove(potential_id)

                sorted_potential = sorted(potential)

                #  Update the successor to visit attribute
                successorstovisit = sorted_potential

                # Add the updated node to the stack.
                self.sequence_nodes_list.append([v[0], v[1], successorstovisit])
                self.sequence_nodes_list.sort(key=lambda x: x[2])

            # If no match is found, block the node and all the successors.
            else:
                self.circuit_blocked[label] = True
                for succ in v[1].successors:
                    self.circuit_blocked[succ] = True


class Sequence:  # pylint: disable=too-few-public-methods
    """
    Object to represent a match and its qubits configurations.
    """

    def __init__(self, sequence, qubit):
        """Create a Match class with necessary arguments.
        Args:
            match (list): list of gates in the sequence.
            qubit (list): list of qubits configuration.
        """
        # Match list
        self.sequence = sequence
        # Qubits list for circuit
        if any(isinstance(el, list) for el in qubit):
            self.qubit = qubit
        else:
            self.qubit = [qubit]


class SequenceScenarios:
    """
    Class to represent a matching scenario in the Backward part of the algorithm.
    """

    def __init__(self, circuit_matched, circuit_blocked, sequence, counter):
        """Create a SequenceScenarios class for the Backward match.
        Args:
            circuit_matched (list): list representing the matched gates in the circuit.
            circuit_blocked (list): list representing the blocked gates in the circuit.
            sequence (list): list of matches.
            counter (int): counter of the number of circuit gates already considered.
        """

        self.circuit_matched = circuit_matched
        self.circuit_blocked = circuit_blocked
        self.sequence = sequence
        self.counter = counter


class SequenceScenariosList:
    """
    Object to define a list of MatchingScenarios, with method to append
    and pop elements.
    """

    def __init__(self):
        """
        Create an empty MatchingScenariosList.
        """
        self.sequence_scenarios_list = []

    def append_scenario(self, matching):
        """
        Append a scenario to the list.
        Args:
            matching (MatchingScenarios): a scenario of match.
        """
        self.sequence_scenarios_list.append(matching)

    def pop_scenario(self):
        """
        Pop the first scenario of the list.
        Returns:
            MatchingScenarios: a scenario of match.
        """
        # Pop the first MatchingScenario and returns it
        first = self.sequence_scenarios_list[0]
        self.sequence_scenarios_list.pop(0)
        return first


class BackwardSequence:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Class BackwardMatch allows to run backward direction part of the pattern matching algorithm.
    """

    def __init__(
        self,
        circuit_dag,
        forward_sequence,
        circuit_matched,
        circuit_blocked,
        node_id_c,
        qubits_conf,
    ):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag (DAGDependency): circuit in the dag dependency form.
            forward_sequence (list): list of match obtained in the forward direction.
            node_id_c (int): index of the first gate matched in the circuit.
        """
        # pylint: disable=too-many-arguments

        self.circuit_dag = circuit_dag
        self.circuit_matched = circuit_matched
        self.circuit_blocked = circuit_blocked
        self.wires = qubits_conf
        self.node_id_c = node_id_c
        self.forward_sequence = forward_sequence
        self.final_sequences = []
        self.sequence_list = SequenceScenariosList()

    def run_backward_match(self):
        """Run the backward match algorithm and returns the list of matches given an initial match, a forward
        scenario and a circuit qubits configuration.
        """
        # pylint: disable=too-many-branches, too-many-statements
        sequence_store_list = []

        counter = 1

        # First Scenario is stored in the MatchingScenariosList().
        first_match = SequenceScenarios(
            self.circuit_matched,
            self.circuit_blocked,
            self.forward_sequence,
            counter,
        )

        self.sequence_list = SequenceScenariosList()
        self.sequence_list.append_scenario(first_match)

        # Set the circuit indices that can be matched.
        gate_indices = _gate_indices(self.circuit_matched, self.circuit_blocked)

        # While the scenario stack is not empty.
        while self.sequence_list.sequence_scenarios_list:

            scenario = self.sequence_list.pop_scenario()
            circuit_matched = scenario.circuit_matched
            circuit_blocked = scenario.circuit_blocked
            sequence_scenario = scenario.sequence
            counter_scenario = scenario.counter

            # Part of the match list coming from the backward match.
            sequence_backward = [
                match for match in sequence_scenario if match not in self.forward_sequence
            ]

            # Sequences are stored
            if counter_scenario > len(gate_indices):
                sequence_scenario.sort()
                sequence_store_list.append(Sequence(sequence_scenario, self.wires))
                continue

            circuit_id = gate_indices[counter_scenario - 1]
            node_circuit = self.circuit_dag.get_node(circuit_id)

            # If the circuit candidate is blocked, only the counter is changed.
            if circuit_blocked[circuit_id]:
                sequence_scenarios = SequenceScenarios(
                    circuit_matched,
                    circuit_blocked,
                    sequence_scenario,
                    counter_scenario + 1,
                )
                self.sequence_list.append_scenario(sequence_scenarios)
                continue

            add = False

            if set(node_circuit.wires).issubset(self.wires):
                # A match happens.
                # If there is a match the attributes are copied.
                circuit_matched_match = circuit_matched.copy()
                circuit_blocked_match = circuit_blocked.copy()

                sequence_scenario_add = sequence_scenario.copy()

                circuit_matched_match[circuit_id] = True
                sequence_scenario_add.append(circuit_id)

                new_sequence_scenario = SequenceScenarios(
                    circuit_matched_match,
                    circuit_blocked_match,
                    sequence_scenario_add,
                    counter_scenario + 1,
                )
                self.sequence_list.append_scenario(new_sequence_scenario)

                add = True

            if add:
                circuit_matched_block_s = circuit_matched.copy()
                circuit_blocked_block_s = circuit_blocked.copy()

                sequence_scenario_block_s = sequence_scenario.copy()

                circuit_blocked_block_s[circuit_id] = True

                broken_matches = []

                # Second option, not a greedy match, block all successors (push the gate
                # to the right).
                for succ in self.circuit_dag.get_node(circuit_id).successors:
                    circuit_blocked_block_s[succ] = True
                    if circuit_matched_block_s[succ]:
                        broken_matches.append(succ)
                        circuit_matched_block_s[succ] = False

                new_sequence_scenario_block_s = [
                    elem for elem in sequence_scenario_block_s if elem not in broken_matches
                ]

                condition_not_greedy = True

                for back_match in sequence_backward:
                    if back_match not in new_sequence_scenario_block_s:
                        condition_not_greedy = False
                        break

                if (self.node_id_c in new_sequence_scenario_block_s) and (
                    condition_not_greedy or not sequence_backward
                ):
                    new_sequence_scenario = SequenceScenarios(
                        circuit_matched_block_s,
                        circuit_blocked_block_s,
                        new_sequence_scenario_block_s,
                        counter_scenario + 1,
                    )
                    self.sequence_list.append_scenario(new_sequence_scenario)

            # If there is no match then there are three options.
            if not add:

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
                    sequence_scenarios = SequenceScenarios(
                        circuit_matched,
                        circuit_blocked,
                        sequence_scenario,
                        counter_scenario + 1,
                    )
                    self.sequence_list.append_scenario(sequence_scenarios)

                else:

                    circuit_matched_nomatch = circuit_matched.copy()
                    circuit_blocked_nomatch = circuit_blocked.copy()

                    sequence_scenario_nomatch = sequence_scenario.copy()

                    # Second option, all predecessors are blocked (circuit gate is
                    # moved to the left).
                    for pred in predecessors:
                        circuit_blocked[pred] = True

                    sequence_scenarios = SequenceScenarios(
                        circuit_matched,
                        circuit_blocked,
                        sequence_scenario,
                        counter_scenario + 1,
                    )
                    self.sequence_list.append_scenario(sequence_scenarios)

                    # Third option, all successors are blocked (circuit gate is
                    # moved to the right).

                    broken_matches = []

                    successors = self.circuit_dag.get_node(circuit_id).successors

                    for succ in successors:
                        circuit_blocked_nomatch[succ] = True
                        if circuit_matched_nomatch[succ]:
                            broken_matches.append(succ)
                            circuit_matched_nomatch[succ] = False

                    new_sequence_scenario_no_add = [
                        elem for elem in sequence_scenario_nomatch if elem not in broken_matches
                    ]

                    condition_block = True

                    for back_match in sequence_backward:
                        if back_match not in new_sequence_scenario_no_add:
                            condition_block = False
                            break

                    if (self.node_id_c in new_sequence_scenario_no_add) and (
                        condition_block or not sequence_backward
                    ):
                        new_sequence_scenario = SequenceScenarios(
                            circuit_matched_nomatch,
                            circuit_blocked_nomatch,
                            new_sequence_scenario_no_add,
                            counter_scenario + 1,
                        )
                        self.sequence_list.append_scenario(new_sequence_scenario)

        length = max(len(m.sequence) for m in sequence_store_list)

        # Store the matches with maximal length.
        for scenario in sequence_store_list:
            if (len(scenario.sequence) == length) and not any(
                scenario.sequence == x.sequence for x in self.final_sequences
            ):
                self.final_sequences.append(scenario)


def _gate_indices(circuit_matched, circuit_blocked):
    """Function which returns the list of gates that are not matched and not blocked for the first scenario.
    Returns:
        list(int): list of gate id.
    """
    gate_indices = []

    for i, (matched, blocked) in enumerate(zip(circuit_matched, circuit_blocked)):
        if (not matched) and (not blocked):
            gate_indices.append(i)
    gate_indices.reverse()
    return gate_indices


class MaximalSequences:  # pylint: disable=too-few-public-methods
    """
    Class MaximalMatches allows to sort and store the maximal matches from the list
    of matches obtained with the pattern matching algorithm.
    """

    def __init__(self, sequences):
        """Initialize MaximalMatches with the necessary arguments.
        Args:
            pattern_matches (list): list of matches obtained from running the algorithm.
        """
        self.sequences = sequences

        self.max_sequences_list = []

    def run_maximal_sequences(self):
        """Method that extracts and stores maximal matches in decreasing length order."""

        self.max_sequences_list = [
            Sequence(
                sorted(self.sequences[0].sequence),
                self.sequences[0].qubit,
            )
        ]

        for sequence in self.sequences[1::]:
            present = False
            for max_sequence in self.max_sequences_list:
                for elem in sequence.sequence:
                    if elem in max_sequence.sequence and len(sequence.sequence) <= len(
                        max_sequence.sequence
                    ):
                        present = True
            if not present:
                self.max_sequences_list.append(Sequence(sorted(sequence.sequence), sequence.qubit))


class SubstitutionConfig:
    """Class to store the configuration of a given match substitution, which circuit gates, template gates,
    qubits and predecessors of the match in the circuit.
    """

    def __init__(
        self,
        circuit_config,
        pred_block,
        qubit_config,
        subtape_opt,
    ):
        self.circuit_config = circuit_config
        self.qubit_config = qubit_config
        self.pred_block = pred_block
        self.subtape_opt = subtape_opt


class SequencesSubstitution:  # pylint: disable=too-few-public-methods
    """Class to run the substitution algorithm from the list of maximal matches."""

    def __init__(self, max_sequences, circuit_dag, n_qubits_subset):
        """
        Initialize TemplateSubstitution with necessary arguments.
        Args:
            max_matches (list(int)): list of maximal matches obtained from the running the pattern matching algorithm.
            circuit_dag (.CommutationDAG): circuit in the dag dependency form.
        """

        self.sequence_stack = max_sequences
        self.circuit_dag = circuit_dag
        self.n_qubits_subset = n_qubits_subset

        self.substitution_list = []
        self.unmatched_list = []

    def _pred_block(self, circuit_sublist, index):
        """It returns the predecessors of a given part of the circuit.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
            index (int): Index of the group of matches.
        Returns:
            list: List of predecessors of the current match circuit configuration.
        """
        predecessors = set()
        for node_id in circuit_sublist:
            predecessors = predecessors | set(self.circuit_dag.get_node(node_id).predecessors)

        exclude = set()
        for elem in self.substitution_list[:index]:
            exclude = exclude | set(elem.circuit_config) | set(elem.pred_block)

        pred = list(predecessors - set(circuit_sublist) - exclude)
        pred.sort()

        return pred

    def _substitution_sort(self):
        """Sort the substitution list."""
        ordered = False
        while not ordered:
            ordered = self._permutation()

    def _permutation(self):  # pragma: no cover
        """Permute two groups of matches if first one has predecessors in the second one.
        Returns:
            bool: True if the matches groups are in the right order, False otherwise.
        """
        for scenario in self.substitution_list:
            predecessors = set()
            for match in scenario.circuit_config:
                predecessors = predecessors | set(self.circuit_dag.get_node(match).predecessors)
            predecessors = predecessors - set(scenario.circuit_config)
            index = self.substitution_list.index(scenario)
            for scenario_b in self.substitution_list[index::]:
                if set(scenario_b.circuit_config) & predecessors:
                    index1 = self.substitution_list.index(scenario)
                    index2 = self.substitution_list.index(scenario_b)

                    scenario_pop = self.substitution_list.pop(index2)
                    self.substitution_list.insert(index1, scenario_pop)
                    return False
        return True

    def _remove_impossible(self):  # pragma: no cover
        """Remove matched groups if they both have predecessors in the other one, they are not compatible."""
        list_predecessors = []
        remove_list = []

        # Initialize predecessors for each group of matches.
        for scenario in self.substitution_list:
            predecessors = set()
            for index in scenario.circuit_config:
                predecessors = predecessors | set(self.circuit_dag.get_node(index).predecessors)
            list_predecessors.append(predecessors)

        # Check if two groups of matches are incompatible.
        for scenario_a in self.substitution_list:
            if scenario_a in remove_list:
                continue
            index_a = self.substitution_list.index(scenario_a)
            circuit_a = scenario_a.circuit_config
            for scenario_b in self.substitution_list[index_a + 1 : :]:
                if scenario_b in remove_list:
                    continue
                index_b = self.substitution_list.index(scenario_b)
                circuit_b = scenario_b.circuit_config
                if (set(circuit_a) & list_predecessors[index_b]) and (
                    set(circuit_b) & list_predecessors[index_a]
                ):
                    remove_list.append(scenario_b)

        # Remove the incompatible groups from the list.
        if remove_list:
            self.substitution_list = [
                scenario for scenario in self.substitution_list if scenario not in remove_list
            ]

    def substitution(self):
        """From the list of maximal matches, it chooses which one will be used and gives the necessary details for
        each substitution(template inverse, predecessors of the match).
        """

        while self.sequence_stack:

            # Get the first match scenario of the list
            current = self.sequence_stack.pop(0)

            circuit_sublist = current.sequence
            current_qubit = current.qubit

            with qml.tape.QuantumTape(do_queue=False) as subtape:
                for id in circuit_sublist:
                    apply(self.circuit_dag.get_node(id).op)

            subtape_matrix = qml.matrix(subtape)

            if self.n_qubits_subset == 1:
                subtape_opt = zyz_decomposition(subtape_matrix)

            if self.n_qubits_subset == 2:
                with qml.tape.QuantumTape(do_queue=False) as subtape_opt:
                    two_qubit_decomposition(subtape_matrix, subtape.wires)

            # If the match obey the rule then it is added to the list.
            if len(subtape_opt.operations) < len(subtape.operations):
                config = SubstitutionConfig(
                    circuit_sublist,
                    [],
                    current_qubit,
                    subtape_opt,
                )
                self.substitution_list.append(config)

        # Remove incompatible matches.
        self._remove_impossible()

        # First sort the matches according to the smallest index in the matches (circuit).
        self.substitution_list.sort(key=lambda x: x.circuit_config[0])

        # Change position of the groups due to predecessors of other groups.
        self._substitution_sort()

        for scenario in self.substitution_list:
            index = self.substitution_list.index(scenario)
            scenario.pred_block = self._pred_block(scenario.circuit_config, index)

        circuit_list = []
        for elem in self.substitution_list:
            circuit_list = circuit_list + elem.circuit_config + elem.pred_block

        # Unmatched gates that are not predecessors of any group of matches.
        self.unmatched_list = sorted(list(set(range(0, self.circuit_dag.size)) - set(circuit_list)))
