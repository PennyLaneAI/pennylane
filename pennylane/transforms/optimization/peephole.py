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
"""Function finding all maximal sequences of gates acting on a subset of qubits considering pairwise commutation of gates.
"""

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
def peephole_optimization(tape, size_qubits_subsets, custom_quantum_cost=None):
    r"""Quantum function transform to optimize a circuit given a list of qubits subset (peephole) size (1, 2). First the
    algorithm finds all maximal sequences of gates acting on a all subset of qubits of a given size. Then all
    sequences are optimized by using 1 qubits and 2 qubits optimal decompositions.

    Args:
        qfunc (function): A quantum function to be optimized.
        n_qubits(list(int)): List of size of subset qubits [1], [2], [1, 2], [2, 1]. The order in the list matters as
                            it will be optimized sequentially.
        custom_quantum_cost (dict): Optional, custom quantum cost dictionary.

    Returns:
        function: the transformed quantum function

    Raises:
        QuantumFunctionError: The subset size list is not of the right format or contains unsupported sizes.

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
    >>> optimized_qfunc = peephole_optimization(n_qubit=[2])(circuit)
    >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

    In our case, it is possible to reduce 4 CNOTs to only two CNOTs and therefore
    optimizing the circuit. The number of CNOTs in the circuit is reduced by one.

    >>> qml.specs(qnode)()["gate_types"]["CNOT"]
    6

    >>> qml.specs(optimized_qnode)()["gate_types"]["CNOT"]
    4

    >>> print(qml.draw(qnode)())
    0: ────────────────╭C─╭C────────────────┤  <X>
    1: ────╭C──H─╭C──H─╰X─│─────╭X────╭X──X─┤
    2: ──X─╰X────╰X───────╰X──H─╰C──H─╰C────┤

    >>> print(qml.draw(optimized_qnode)())
    0: ─╭C───────────────────────────────────────────────────────╭C─┤  <X>
    1: ─│───RZ(1.57)─────────────╭C──SX─╭C──Rot(-1.57,1.57,1.57)─╰X─┤
    2: ─╰X──Rot(0.00,1.57,-1.57)─╰X──S──╰X──Rot(0.00,1.57,-1.57)────┤

    The algortihm finds the sequence [0, 1, 2, 3, 4, 7, 8, 9, 10, 11] (acting on qubits 1 and 2) as we can push CNOT
    5 to the right and CNOT 6 to the left. The longest sequence contains 4 CNOT which is not optimal and can bbe
    reduced to two by optimal synthesis of 4x4 unitaries. Therefore the sequences of gates is replaced and therefore
    the whole circuit is optimized.

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

    # Check the validity of the size of qubits subsets.
    if size_qubits_subsets not in [[1], [2], [1, 2], [2, 1]]:
        raise qml.QuantumFunctionError(
            "The list of number of qubits is not a valid. It should be a list containing 1 and or 2."
        )

    for size_qubits_subset in size_qubits_subsets:
        # Construct Dag representation of the circuit and the pattern.
        circuit_dag = commutation_dag(tape)()

        # FInd all maximal seuqences for a circuit with a given qubity subset size.
        max_sequences = maximal_sequences(circuit_dag, size_qubits_subset)

        # Optimizes the circuit for compatible maximal sequences
        if max_sequences:
            # Initialize the optimization by substitution of the different sequences
            substitution = SequencesSubstitution(
                max_sequences, circuit_dag, size_qubits_subset, custom_quantum_cost
            )
            substitution.substitution()
            already_sub = []

            # If some substitutions are possible, we create an optimized circuit.
            if substitution.substitution_list:
                # Create a tape that does not affect the outside context.
                with qml.tape.QuantumTape(do_queue=False) as tape_inside:
                    # Loop over all possible substitutions and create the optimized circuit.
                    for group in substitution.substitution_list:

                        sequence = group.sequence
                        sequence_opt_operations = group.sequence_opt.operations
                        pred = group.pred_block

                        # First add all the predecessors of the given match.
                        for elem in pred:
                            node = circuit_dag.get_node(elem)
                            inst = copy.deepcopy(node.op)
                            apply(inst)
                            already_sub.append(elem)

                        already_sub = already_sub + sequence

                        # Then replace the sequence by the optimized version.
                        for op in sequence_opt_operations:
                            apply(op)

                    # Add the gates not found in sequences.
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


def maximal_sequences(circuit_dag, size_qubit_subset):
    r"""Function that applies an algorithm to find all maximal sequences of gates acting on a given qubits subset size.

    Args:
        circuit_dag (.CommutationDAG): A commutation DAG representing the circuit to be optimized.
        size_qubit_subset(int): Size of the qubits subset that are considered.

    Returns:
        list(Sequence): the list of maximal sequences.

    Raises:
        QuantumFunctionError: The subset size is greater than the number of qubits in the circuit.
    """
    # Check the validity of the qubits subset size.
    if size_qubit_subset > circuit_dag.num_wires:
        raise qml.QuantumFunctionError(
            "The qubits subset considered must be smaller or equal than the number of qubits in the "
            "circuit."
        )

    # Sequence list
    sequence_list = []

    # Loop through all possible initial matches
    for node_c in circuit_dag.get_nodes():
        # Check that the gate acts on less or equal number of qubits than the given size.
        if _compare_operation_qubits_number(node_c[1], size_qubit_subset):
            # List the not fixed qubits
            not_fixed_qubits_conf = _not_fixed_qubits(circuit_dag.num_wires, node_c[1].wires)

            # Fix the qubits from the first gate
            first_match_qubits_conf = _first_gate_qubits(node_c[1])

            # Qubit configuration given the first operation and number of qubits.
            qubits_confs = _merge_first_match_and_not_fixed(
                first_match_qubits_conf, not_fixed_qubits_conf, size_qubit_subset
            )

            # Loops over the diffrent qubits configurations
            for qubits_conf in qubits_confs:
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

    # Extract maximal sequences
    if sequence_list:
        maximal = MaximalSequences(sequence_list)
        maximal.run_maximal_sequences()
        max_matches = maximal.max_sequences_list
        return max_matches

    return sequence_list


def _compare_operation_qubits_number(node_1, qubits_subset_size):
    """Compare the number of qubits in the operation with the qubits subset size.

    Args:
        node_1 (.CommutationDAGNode): First operation.
        qubits_subset_size (int): Qubits subset size.
    Return:
        Bool: True if the operation has as much or less qubits than the qubits subset size.
    """
    return len(node_1.op.wires) <= qubits_subset_size


def _not_fixed_qubits(n_qubits_circuit, exclude_qubits):
    """
    Function that returns all possible combinations of qubits given some restrictions and using itertools.
    Args:
        n_qubits_circuit (int): Number of qubit in the circuit.
        exclude_qubits (list): list of qubits from the first matched circuit operation that needs to be excluded.
    Returns:
        list: List of not fixed qubits.
    """
    circuit_range = range(0, n_qubits_circuit)
    return [e for e in circuit_range if e not in exclude_qubits]


def _first_gate_qubits(node_c):
    """
    Returns the list of qubits of the operation.
    Args:
        node_c (.CommutationDAGNode): First node in the circuit.

    Returns:
        list: list of qubits to consider in circuit.
    """
    return node_c.wires


def _merge_first_match_and_not_fixed(list_first_match, list_not_fixed, size_qubits_subset):
    """
    Function that returns the final qubits configuration given the first match constraints and the permutation of
    qubits not in the first match.

    Args:
        list_first_match (list): list of qubits indices for the first match.
        list_not_fixed (list): possible permutation for the circuit qubits not in the first match.
        size_qubits_subset (int): Size of the qubits subset.

    Returns:
        list(set(int)): list of qubits configurations to consider.
    """
    list_circuit = []

    combinations = itertools.combinations(
        list_not_fixed, size_qubits_subset - len(list_first_match)
    )

    for combination in combinations:
        current_set = set(list_first_match)
        combination = set(combination)
        current_set = current_set.union(combination)
        list_circuit.append(current_set)

    return list_circuit


def _add_sequence(sequences_list, backward_sequences):
    """
    Add a sequence configuration found by the algorithm if it is not already in final list of matches.
    If the match is already in the final sequences list, the qubit configuration is added to the existing Sequence.
    Args:
        sequences_list (list(.Sequence)): Sequence from the backward part of the algorithm.
        backward_sequences (list(.Sequence)): List of Sequences found by the algorithm for a given configuration.
    """

    already_in = False
    for backward_sequence in backward_sequences:
        for sequence in sequences_list:
            if backward_sequence.sequence == sequence.sequence:
                index = sequences_list.index(sequence)
                if backward_sequence.qubit[0] not in sequences_list[index].qubit:
                    sequences_list[index].qubit.append(backward_sequence.qubit[0])
                already_in = True

        if not already_in:
            sequences_list.append(backward_sequence)


class ForwardSequence:
    """
    Class to apply the forward part of the sequence finding algorithm.
    """

    def __init__(self, circuit_dag, node_id_c, qubits_conf):
        """
        Create the ForwardMatch class.
        Args:
            circuit_dag (.CommutationDAG): Circuit as commutation DAG.
            node_id_c (int): ID of the given node.
            qubits_conf (set(int)): Qubits configuration.

        """

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

        # Sequence
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
        Initialize the nodes that are in the sequence.
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
        Initialize the list of nodes in the sequence (first matched gate).
        """
        self.sequence.append(self.node_id_c)

    def _init_sequence_nodes(self):
        """
        Initialize the list of current nodes to be considered for addition in the sequence.
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
        Return node and successors from the sequence_nodes_list for a given ID.
        Args:
            list_id (int): list ID of the desired node.
        Returns:
            CommutationDAGNode: Node from the sequence_nodes_list.
            list(int): List of successors.
        """
        node = self.sequence_nodes_list[list_id][1]
        succ = self.sequence_nodes_list[list_id][2]
        return node, succ

    def _remove_node_forward(self, list_id):
        """Remove a node of the current sequence_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        """
        self.sequence_nodes_list.pop(list_id)

    def run_forward_match(self):
        """Apply the forward match sequence finding algorithm and returns the list of matches given an initial sequence
        and a qubits configuration.
        """
        # Initialization
        self._init_successors_to_visit()
        self._init_circuit_matched()
        self._init_circuit_blocked()

        # Initialize the sequence of gates and the stack of nodes to consider.
        self._init_list_sequence()
        self._init_sequence_nodes()

        # While the list of nodes to be considered is not empty
        while self.sequence_nodes_list:
            # Return first element of the sequence_nodes_list and removes it from the list
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

            # Update the seequence_nodes_list with new attribute successor to visit and sort the list.
            self.sequence_nodes_list.append([v_first.node_id, v_first, successors_to_visit])
            self.sequence_nodes_list.sort(key=lambda x: x[2])

            # If the node is blocked and already matched go to the end
            if self.circuit_blocked[label] or self.circuit_matched[v[0]]:
                continue

            # Match condition is the node acting on the subset.
            if set(v[1].wires).issubset(self.wires):
                # A match happens
                self.circuit_matched[label] = True

                # Append the gate to the sequence.
                self.sequence.append(label)

                # Potential successors to visit for a given matched node.
                potential = self.circuit_dag.direct_successors(label)
                # If the potential successors to visit are blocked or matched, it is removed.
                for potential_id in potential:
                    if self.circuit_blocked[potential_id] or (self.circuit_matched[potential_id]):
                        potential.remove(potential_id)

                sorted_potential = sorted(potential)

                #  Update the successor to visit attribute
                successorstovisit = sorted_potential

                # Add the updated node to the stack.
                self.sequence_nodes_list.append([v[0], v[1], successorstovisit])
                self.sequence_nodes_list.sort(key=lambda x: x[2])

            # If no match is found, block the node and all its successors.
            else:
                self.circuit_blocked[label] = True
                for succ in v[1].successors:
                    self.circuit_blocked[succ] = True


class Sequence:
    """
    Object to represent a sequence and its qubits configurations.
    """

    def __init__(self, sequence, qubit):
        """Create a Match class with necessary arguments.
        Args:
            sequence (list): list of gates in the sequence.
            qubit (list): list of qubits configuration.
        """
        # Match list
        self.sequence = sequence
        # Qubits list for circuit
        if isinstance(qubit, list):
            self.qubit = qubit
        else:
            self.qubit = [qubit]


class SequenceScenarios:
    """
    Class to represent a sequence scenario in the Backward part of the algorithm.
    """

    def __init__(self, circuit_matched, circuit_blocked, sequence, counter):
        """Create a SequenceScenarios class for the Backward match.
        Args:
            circuit_matched (list): list representing the matched gates.
            circuit_blocked (list): list representing the blocked gates.
            sequence (list): list of nodes in the sequence.
            counter (int): counter of the number of circuit gates already considered.
        """

        self.circuit_matched = circuit_matched
        self.circuit_blocked = circuit_blocked
        self.sequence = sequence
        self.counter = counter


class SequenceScenariosList:
    """
    Object to define a list of SequenceScenarios, with method to append
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


class BackwardSequence:
    """
    Class to apply the backward part of the sequence finding algorithm.
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
        Create a BackwardSequence class with necessary arguments.
        Args:
            circuit_dag (DAGDependency): circuit in the dag dependency form.
            forward_sequence (list): list of match obtained in the forward direction.
            circuit_matched (list): List of node ID belonging to the sequence.
            circuit_blocked (list): List of node ID blocked in the circuit.
            node_id_c (int): index of the first gate matched in the circuit.
            qubits_conf (set(int)): Set of qubits.
        """

        self.circuit_dag = circuit_dag
        self.forward_sequence = forward_sequence
        self.circuit_matched = circuit_matched
        self.circuit_blocked = circuit_blocked
        self.node_id_c = node_id_c
        self.wires = qubits_conf
        self.final_sequences = []
        self.sequence_list = SequenceScenariosList()

    def run_backward_match(self):
        """Run the backward sequence algorithm and returns the sequence of gates given an initial gate, the forward
        results and a circuit qubits configuration.
        """

        sequence_store_list = []

        counter = 1

        # First Scenario is stored in the SequenceScenariosList().
        first_match = SequenceScenarios(
            self.circuit_matched,
            self.circuit_blocked,
            self.forward_sequence,
            counter,
        )

        self.sequence_list.append_scenario(first_match)

        # Set the circuit ID that can be potentially added to the sequence.
        gate_indices = _gate_indices(self.circuit_matched, self.circuit_blocked)

        # While the scenario stack is not empty.
        while self.sequence_list.sequence_scenarios_list:

            scenario = self.sequence_list.pop_scenario()
            circuit_matched = scenario.circuit_matched
            circuit_blocked = scenario.circuit_blocked
            sequence_scenario = scenario.sequence
            counter_scenario = scenario.counter

            # Part of the sequence coming from the backward match.
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
                circuit_matched_add = circuit_matched.copy()
                circuit_blocked_add = circuit_blocked.copy()

                sequence_scenario_add = sequence_scenario.copy()

                circuit_matched_add[circuit_id] = True
                sequence_scenario_add.append(circuit_id)

                # First option: greedy matching
                new_sequence_scenario = SequenceScenarios(
                    circuit_matched_add,
                    circuit_blocked_add,
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
                    new_sequence_scenario = SequenceScenarios(
                        circuit_matched,
                        circuit_blocked,
                        sequence_scenario,
                        counter_scenario + 1,
                    )
                    self.sequence_list.append_scenario(new_sequence_scenario)

                else:

                    circuit_matched_no_add = circuit_matched.copy()
                    circuit_blocked_no_add = circuit_blocked.copy()

                    sequence_scenario_no_add = sequence_scenario.copy()

                    # Second option, all predecessors are blocked (circuit gate is
                    # moved to the left).
                    for pred in predecessors:
                        circuit_blocked[pred] = True

                    new_sequence_scenario = SequenceScenarios(
                        circuit_matched,
                        circuit_blocked,
                        sequence_scenario,
                        counter_scenario + 1,
                    )
                    self.sequence_list.append_scenario(new_sequence_scenario)

                    # Third option, all successors are blocked (circuit gate is
                    # moved to the right).

                    broken_matches = []

                    successors = self.circuit_dag.get_node(circuit_id).successors

                    for succ in successors:
                        circuit_blocked_no_add[succ] = True
                        if circuit_matched_no_add[succ]:
                            broken_matches.append(succ)
                            circuit_matched_no_add[succ] = False

                    new_sequence_scenario_no_add = [
                        elem for elem in sequence_scenario_no_add if elem not in broken_matches
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
                            circuit_matched_no_add,
                            circuit_blocked_no_add,
                            new_sequence_scenario_no_add,
                            counter_scenario + 1,
                        )
                        self.sequence_list.append_scenario(new_sequence_scenario)

        length = max(len(m.sequence) for m in sequence_store_list)

        # Store the sequences with maximal length.
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
    Class MaximalSequences allows to sort and store the maximal sequences from the list
    of sequences obtained with the sequence finding algorithm.
    """

    def __init__(self, sequences):
        """Initialize MaximalSequences with the necessary arguments.
        Args:
            sequences (list): list of sequences obtained from running the algorithm.
        """
        self.sequences = sequences

        self.max_sequences_list = []

    def run_maximal_sequences(self):
        """Method that extracts and stores maximal sequences in decreasing length order."""

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
    """Class to store the configuration of a given sequence, its qubits configurations and predecessors and the
    optimized sequences.
    """

    def __init__(
        self,
        sequence,
        qubits_conf,
        sequence_opt,
        pred_block,
    ):
        """Initialize MaximalSequences with the necessary arguments.
        Args:
            sequence (list(int)): sequence of gates in the circuit.
            qubits_conf (list(int)): Qubits configuration.
            sequence_opt (.QuantumTape): Quantum tape of the optimized sequence.
            pred_block (list(int)): List of predecessors.
        """
        self.sequence = sequence
        self.qubits_conf = qubits_conf
        self.sequence_opt = sequence_opt
        self.pred_block = pred_block


class SequencesSubstitution:  # pylint: disable=too-few-public-methods
    """Class to run the substitution algorithm from the list of maximal sequences."""

    def __init__(self, max_sequences, circuit_dag, size_qubits_subset, custom_quantum_cost=None):
        """
        Initialize TemplateSubstitution with necessary arguments.
        Args:
            max_sequences (list(int)): list of maximal matches obtained from the running the pattern matching algorithm.
            circuit_dag (.CommutationDAG): circuit in the dag dependency form.
            size_qubits_subset (int): Size of the qubits subset.
            custom_quantum_cost (dict): Optional, dictionary containing gate names and their respective costs.
        """

        self.sequence_stack = max_sequences
        self.circuit_dag = circuit_dag
        self.size_qubits_subset = size_qubits_subset

        self.substitution_list = []
        self.unmatched_list = []

        if custom_quantum_cost is not None:
            self.quantum_cost = dict(custom_quantum_cost)
        else:
            self.quantum_cost = {
                "Identity": 0,
                "PauliX": 1,
                "PauliY": 1,
                "PauliZ": 1,
                "RX": 1,
                "RY": 1,
                "RZ": 1,
                "Rot": 1,
                "Hadamard": 1,
                "T": 1,
                "S": 1,
                "SX": 1,
                "CNOT": 2,
                "CZ": 4,
                "SWAP": 6,
            }

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
            exclude = exclude | set(elem.sequence) | set(elem.pred_block)

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
            for match in scenario.sequence:
                predecessors = predecessors | set(self.circuit_dag.get_node(match).predecessors)
            predecessors = predecessors - set(scenario.sequence)
            index = self.substitution_list.index(scenario)
            for scenario_b in self.substitution_list[index::]:
                if set(scenario_b.sequence) & predecessors:
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
            for index in scenario.sequence:
                predecessors = predecessors | set(self.circuit_dag.get_node(index).predecessors)
            list_predecessors.append(predecessors)

        # Check if two groups of matches are incompatible.
        for scenario_a in self.substitution_list:
            if scenario_a in remove_list:
                continue
            index_a = self.substitution_list.index(scenario_a)
            circuit_a = scenario_a.sequence
            for scenario_b in self.substitution_list[index_a + 1 : :]:
                if scenario_b in remove_list:
                    continue
                index_b = self.substitution_list.index(scenario_b)
                circuit_b = scenario_b.sequence
                if (set(circuit_a) & list_predecessors[index_b]) and (
                    set(circuit_b) & list_predecessors[index_a]
                ):
                    remove_list.append(scenario_b)

        # Remove the incompatible groups from the list.
        if remove_list:
            self.substitution_list = [
                scenario for scenario in self.substitution_list if scenario not in remove_list
            ]

    def _quantum_cost(self, sequence, sequence_opt):
        """Compare the quantum cost between the original sequence and the optimized sequence.
        Args:
            sequence (.QuantumTape): tape of the sequence.
            sequence_opt (.QuantumTape): tape of the optimized sequencee.
        Returns:
            bool: True if the quantum cost is reduced.
        """
        cost_sequence = 0
        for gate in sequence:
            cost_sequence += self.quantum_cost[gate.name]

        cost_sequence_opt = 0
        for gate in sequence_opt:
            cost_sequence_opt += self.quantum_cost[gate.name]
        return cost_sequence > cost_sequence_opt

    def substitution(self):
        """From the list of maximal sequences, it creates all subsitution configurations necessary to create the
        optimized version of the circuit.
        """

        while self.sequence_stack:

            # Get the first match scenario of the list
            current = self.sequence_stack.pop(0)

            circuit_sequence = current.sequence
            qubits_conf = current.qubit

            with qml.tape.QuantumTape(do_queue=False) as sequence:
                for id in circuit_sequence:
                    apply(self.circuit_dag.get_node(id).op)

            sequence_matrix = qml.matrix(sequence)

            if self.size_qubits_subset == 1:
                with qml.tape.QuantumTape(do_queue=False) as sequence_opt:
                    zyz_decomposition(sequence_matrix, sequence.wires)

            if self.size_qubits_subset == 2:
                with qml.tape.QuantumTape(do_queue=False) as sequence_opt:
                    two_qubit_decomposition(sequence_matrix, sequence.wires)

            # If the sequence is optimized
            if self._quantum_cost(sequence.operations, sequence_opt.operations):
                config = SubstitutionConfig(
                    circuit_sequence,
                    qubits_conf,
                    sequence_opt,
                    [],
                )
                self.substitution_list.append(config)

        # Remove incompatible matches.
        self._remove_impossible()

        # First sort the matches according to the smallest index in the sequences.
        self.substitution_list.sort(key=lambda x: x.sequence[0])

        # Change position of the groups due to predecessors of other groups.
        self._substitution_sort()

        for scenario in self.substitution_list:
            index = self.substitution_list.index(scenario)
            scenario.pred_block = self._pred_block(scenario.sequence, index)

        circuit_list = []
        for elem in self.substitution_list:
            circuit_list = circuit_list + elem.sequence + elem.pred_block

        # Not listed gates that are not predecessors of any group of sequences.
        self.unmatched_list = sorted(list(set(range(0, self.circuit_dag.size)) - set(circuit_list)))
