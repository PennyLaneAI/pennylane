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

import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable

import numpy as np

import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires


# pylint: disable=too-many-statements
@transform
def pattern_matching_optimization(
    tape: QuantumTape, pattern_tapes, custom_quantum_cost=None
) -> (Sequence[QuantumTape], Callable):
    r"""Quantum function transform to optimize a circuit given a list of patterns (templates).

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit to be optimized.
        pattern_tapes(list(.QuantumTape)): List of quantum tapes that implement the identity.
        custom_quantum_cost (dict): Optional, quantum cost that overrides the default cost dictionary.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    Raises:
        QuantumFunctionError: The pattern provided is not a valid QuantumTape or the pattern contains measurements or
            the pattern does not implement identity or the circuit has less qubits than the pattern.

    **Example**

    >>> dev = qml.device('default.qubit', wires=5)

    You can apply the transform directly on a :class:`QNode`. For that, you need first to define a pattern that is to be
    found in the circuit. We use the following pattern that implements the identity:

    .. code-block:: python

        ops = [qml.S(0), qml.S(0), qml.Z(0)]
        pattern = qml.tape.QuantumTape(ops)


    Let's consider the following circuit where we want to replace a sequence of two ``pennylane.S`` gates with a
    ``pennylane.PauliZ`` gate.

    .. code-block:: python

        @partial(pattern_matching_optimization, pattern_tapes = [pattern])
        @qml.qnode(device=dev)
        def circuit():
            qml.S(wires=0)
            qml.Z(0)
            qml.S(wires=1)
            qml.CZ(wires=[0, 1])
            qml.S(wires=1)
            qml.S(wires=2)
            qml.CZ(wires=[1, 2])
            qml.S(wires=2)
            return qml.expval(qml.X(0))

    During the call of the circuit, it is first optimized (if possible) and then executed.

    >>> circuit()

    .. details::
        :title: Usage Details

        .. code-block:: python

            def circuit():
                qml.S(wires=0)
                qml.Z(0)
                qml.S(wires=1)
                qml.CZ(wires=[0, 1])
                qml.S(wires=1)
                qml.S(wires=2)
                qml.CZ(wires=[1, 2])
                qml.S(wires=2)
                return qml.expval(qml.X(0))

        For optimizing the circuit given the following template of CNOTs we apply the ``pattern_matching``
        transform.

        >>> qnode = qml.QNode(circuit, dev)
        >>> optimized_qfunc = pattern_matching_optimization(pattern_tapes=[pattern])(circuit)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

        >>> print(qml.draw(qnode)())
        0: ──S──Z─╭●──────────┤  <X>
        1: ──S────╰Z──S─╭●────┤
        2: ──S──────────╰Z──S─┤

        >>> print(qml.draw(optimized_qnode)())
        0: ──S†─╭●────┤  <X>
        1: ──Z──╰Z─╭●─┤
        2: ──Z─────╰Z─┤

        Note that with this pattern we also replace a ``pennylane.S``, ``pennylane.PauliZ`` sequence by
        ``Adjoint(S)``. If one would like avoiding this, it possible to give a custom
        quantum cost dictionary with a negative cost for ``pennylane.PauliZ``.

        >>> my_cost = {"PauliZ": -1 , "S": 1, "Adjoint(S)": 1}
        >>> optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[pattern], custom_quantum_cost=my_cost)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

        >>> print(qml.draw(optimized_qnode)())
        0: ──S──Z─╭●────┤  <X>
        1: ──Z────╰Z─╭●─┤
        2: ──Z───────╰Z─┤

        Now we can consider a more complicated example with the following quantum circuit to be optimized

        .. code-block:: python

            def circuit():
                qml.Toffoli(wires=[3, 4, 0])
                qml.CNOT(wires=[1, 4])
                qml.CNOT(wires=[2, 1])
                qml.Hadamard(wires=3)
                qml.Z(1)
                qml.CNOT(wires=[2, 3])
                qml.Toffoli(wires=[2, 3, 0])
                qml.CNOT(wires=[1, 4])
                return qml.expval(qml.X(0))

        We define a pattern that implement the identity:

        .. code-block:: python

            ops = [
                qml.CNOT(wires=[1, 2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[1, 2]),
                qml.CNOT(wires=[0, 1]),
                qml.CNOT(wires=[0, 2]),
            ]
            tape = qml.tape.QuantumTape(ops)

        For optimizing the circuit given the following pattern of CNOTs we apply the ``pattern_matching``
        transform.

        >>> dev = qml.device('default.qubit', wires=5)
        >>> qnode = qml.QNode(circuit, dev)
        >>> optimized_qfunc = pattern_matching_optimization(circuit, pattern_tapes=[pattern])
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)

        In our case, it is possible to find three CNOTs and replace this pattern with only two CNOTs and therefore
        optimizing the circuit. The number of CNOTs in the circuit is reduced by one.

        >>> qml.specs(qnode)()["resources"].gate_types["CNOT"]
        4

        >>> qml.specs(optimized_qnode)()["resources"].gate_types["CNOT"]
        3

        >>> print(qml.draw(qnode)())
        0: ─╭X──────────╭X────┤  <X>
        1: ─│──╭●─╭X──Z─│──╭●─┤
        2: ─│──│──╰●─╭●─├●─│──┤
        3: ─├●─│───H─╰X─╰●─│──┤
        4: ─╰●─╰X──────────╰X─┤

        >>> print(qml.draw(optimized_qnode)())
        0: ─╭X──────────╭X─┤  <X>
        1: ─│─────╭X──Z─│──┤
        2: ─│──╭●─╰●─╭●─├●─┤
        3: ─├●─│───H─╰X─╰●─┤
        4: ─╰●─╰X──────────┤

    .. seealso:: :func:`~.pattern_matching`

    **References**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2022.
    Exact and practical pattern matching for quantum circuit optimization.
    `doi.org/10.1145/3498325 <https://dl.acm.org/doi/abs/10.1145/3498325>`_
    """
    # pylint: disable=too-many-branches
    consecutive_wires = Wires(range(len(tape.wires)))
    inverse_wires_map = OrderedDict(zip(consecutive_wires, tape.wires))

    for pattern in pattern_tapes:
        # Check the validity of the pattern
        if not isinstance(pattern, QuantumScript):
            raise qml.QuantumFunctionError("The pattern is not a valid quantum tape.")

        # Check that it does not contain a measurement.
        if pattern.measurements:
            raise qml.QuantumFunctionError("The pattern contains measurements.")

        # Verify that the pattern is implementing the identity
        if not np.allclose(
            qml.matrix(pattern, wire_order=pattern.wires), np.eye(2**pattern.num_wires)
        ):
            raise qml.QuantumFunctionError("Pattern is not valid, it does not implement identity.")

        # Verify that the pattern has less qubits or same number of qubits
        if tape.num_wires < pattern.num_wires:
            raise qml.QuantumFunctionError("Circuit has less qubits than the pattern.")

        # Construct Dag representation of the circuit and the pattern.
        circuit_dag = commutation_dag(tape)
        pattern_dag = commutation_dag(pattern)

        max_matches = pattern_matching(circuit_dag, pattern_dag)

        # Optimizes the circuit for compatible maximal matches
        if max_matches:
            # Initialize the optimization by substitution of the different matches
            substitution = TemplateSubstitution(
                max_matches, circuit_dag, pattern_dag, custom_quantum_cost
            )
            substitution.substitution()
            already_sub = []

            # If some substitutions are possible, we create an optimized circuit.
            if substitution.substitution_list:
                # Create a tape that does not affect the outside context.
                with qml.queuing.AnnotatedQueue() as q_inside:
                    # Loop over all possible substitutions
                    for group in substitution.substitution_list:
                        circuit_sub = group.circuit_config
                        template_inverse = group.template_config

                        pred = group.pred_block

                        # Choose the first configuration
                        qubit = group.qubit_config[0]

                        # First add all the predecessors of the given match.
                        for elem in pred:
                            node = circuit_dag.get_node(elem)  # pylint: disable=no-member
                            inst = copy.deepcopy(node.op)
                            qml.apply(inst)
                            already_sub.append(elem)

                        already_sub = already_sub + circuit_sub

                        # Then add the inverse of the template.
                        for index in template_inverse:
                            all_qubits = tape.wires.tolist()
                            all_qubits.sort()
                            wires_t = group.template_dag.get_node(index).wires
                            wires_c = [qubit[x] for x in wires_t]
                            wires = [all_qubits[x] for x in wires_c]

                            node = group.template_dag.get_node(index)
                            inst = copy.deepcopy(node.op)

                            inst = qml.map_wires(inst, wire_map=dict(zip(inst.wires, wires)))
                            adjoint(qml.apply, lazy=False)(inst)

                    # Add the unmatched gates.
                    for node_id in substitution.unmatched_list:
                        node = circuit_dag.get_node(node_id)  # pylint: disable=no-member
                        inst = copy.deepcopy(node.op)
                        qml.apply(inst)

                qscript = QuantumScript.from_queue(q_inside)
                [tape], _ = qml.map_wires(input=qscript, wire_map=inverse_wires_map)

    new_tape = type(tape)(tape.operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


def pattern_matching(circuit_dag, pattern_dag):
    r"""Function that applies the pattern matching algorithm and returns the list of maximal matches.

    Args:
        circuit_dag (.CommutationDAG): A commutation DAG representing the circuit to be optimized.
        pattern_dag(.CommutationDAG): A commutation DAG representing the pattern.

    Returns:
        list(Match): the list of maximal matches.

    **Example**

    First let's consider the following circuit

    .. code-block:: python

        def circuit():
            qml.S(wires=0)
            qml.Z(0)
            qml.S(wires=1)
            qml.CZ(wires=[0, 1])
            qml.S(wires=1)
            qml.S(wires=2)
            qml.CZ(wires=[1, 2])
            qml.S(wires=2)
            return qml.expval(qml.X(0))

    Assume that we want to find all maximal matches of a pattern containing a sequence of two :class:`~.S` gates and
    a :class:`~.PauliZ` gate:

    .. code-block:: python

        def pattern():
            qml.S(wires=0)
            qml.S(wires=0)
            qml.Z(0)


    >>> circuit_dag = qml.commutation_dag(circuit)()
    >>> pattern_dag = qml.commutation_dag(pattern)()
    >>> all_max_matches = qml.pattern_matching(circuit_dag, pattern_dag)

    The matches are accessible by looping through the list outputted by ``qml.pattern_matching``. This output is a list
    of lists containing indices. Each list represents a match between a gate in the pattern with a gate in the circuit.
    The first indices represent the gates in the pattern and the second indices provide indices for the gates in the
    circuit (by order of appearance).

    >>> for match_conf in all_max_matches:
    ...     print(match_conf.match)
    [[0, 0], [2, 1]]
    [[0, 2], [1, 4]]
    [[0, 4], [1, 2]]
    [[0, 5], [1, 7]]
    [[0, 7], [1, 5]]

    The first match of this list corresponds to match the first gate (:class:`~.S`) in the pattern with the first gate
    in the circuit and also the third gate in the pattern (:class:`~.PauliZ`) with the second circuit gate.

    .. seealso:: :func:`~.pattern_matching_optimization`

    **Reference:**

    [1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2022.
    Exact and practical pattern matching for quantum circuit optimization.
    `doi.org/10.1145/3498325 <https://dl.acm.org/doi/abs/10.1145/3498325>`_

    """
    # Match list
    match_list = []

    # Loop through all possible initial matches
    for node_c, node_p in itertools.product(circuit_dag.get_nodes(), pattern_dag.get_nodes()):
        # Initial matches between two identical gates (No qubits comparison)
        if _compare_operation_without_qubits(node_c[1], node_p[1]):
            # Fix qubits from the first (target fixed and control restrained)
            not_fixed_qubits_confs = _not_fixed_qubits(
                circuit_dag.num_wires, node_c[1].wires, pattern_dag.num_wires - len(node_p[1].wires)
            )

            # Loop over all possible qubits configurations given the first match constrains
            for not_fixed_qubits_conf in not_fixed_qubits_confs:
                for not_fixed_qubits_conf_permuted in itertools.permutations(not_fixed_qubits_conf):
                    for first_match_qubits_conf in _first_match_qubits(
                        node_c[1], node_p[1], pattern_dag.num_wires
                    ):
                        # Qubits mapping between circuit and pattern
                        qubits_conf = _merge_first_match_and_permutation(
                            first_match_qubits_conf, not_fixed_qubits_conf_permuted
                        )
                        # Update wires, target_wires, control_wires
                        wires, target_wires, control_wires = _update_qubits(
                            circuit_dag, qubits_conf
                        )

                        # Forward match part of the algorithm
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

                        # Backward match part of the algorithm
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

                        _add_match(match_list, backward.match_final)

    match_list.sort(key=lambda x: len(x.match), reverse=True)

    # Extract maximal matches and optimizes the circuit for compatible maximal matches
    if match_list:
        maximal = MaximalMatches(match_list)
        maximal.run_maximal_matches()
        max_matches = maximal.max_match_list
        return max_matches

    return match_list


def _compare_operation_without_qubits(node_1, node_2):
    """Compare two operations without taking the qubits into account.

    Args:
        node_1 (.CommutationDAGNode): First operation.
        node_2 (.CommutationDAGNode): Second operation.
    Return:
        Bool: True if similar operation (no qubits comparison) and False otherwise.
    """
    return (node_1.op.name == node_2.op.name) and (node_1.op.data == node_2.op.data)


def _not_fixed_qubits(n_qubits_circuit, exclude, length):
    """
    Function that returns all possible combinations of qubits given some restrictions and using itertools.
    Args:
        n_qubits_circuit (int): Number of qubit in the circuit.
        exclude (list): list of qubits from the first matched circuit operation that needs to be excluded.
        length (int): length of the list to be returned (number of qubits in pattern -
        number of qubits from the first matched operation).
    Yield:
        iterator: Iterator of the possible lists.
    """
    circuit_range = range(0, n_qubits_circuit)
    for sublist in itertools.combinations([e for e in circuit_range if e not in exclude], length):
        yield list(sublist)


def _first_match_qubits(node_c, node_p, n_qubits_p):
    """
    Returns the list of qubits for circuit given the first match, the unknown qubit are
    replaced by -1.
    Args:
        node_c (.CommutationDAGNode): First matched node in the circuit.
        node_p (.CommutationDAGNode): First matched node in the pattern.
        n_qubits_p (int): number of qubit in the pattern.
    Returns:
        list: list of qubits to consider in circuit (with specific order).
    """
    # pylint: disable=too-many-branches
    control_base = {
        "CNOT": "PauliX",
        "CZ": "PauliZ",
        "CCZ": "PauliZ",
        "CY": "PauliY",
        "CH": "Hadamard",
        "CSWAP": "SWAP",
        "Toffoli": "PauliX",
        "ControlledPhaseShift": "PhaseShift",
        "CRX": "RX",
        "CRY": "RY",
        "CRZ": "RZ",
        "CRot": "Rot",
        "MultiControlledX": "PauliX",
        "ControlledOperation": "ControlledOperation",
    }

    first_match_qubits = []

    # Controlled gate
    if len(node_c.op.control_wires) >= 1:
        circuit_control = node_c.op.control_wires
        circuit_target = Wires([w for w in node_c.op.wires if w not in node_c.op.control_wires])
        # Not symmetric target gate or acting on 1 wire (target wires cannot be permuted) (For example Toffoli)
        if control_base[node_p.op.name] not in symmetric_over_all_wires:
            # Permute control
            for control_permuted in itertools.permutations(circuit_control):
                control_permuted = list(control_permuted)
                first_match_qubits_sub = [-1] * n_qubits_p
                for q in node_p.wires:
                    node_circuit_perm = control_permuted + circuit_target
                    first_match_qubits_sub[q] = node_circuit_perm[node_p.wires.index(q)]
                first_match_qubits.append(first_match_qubits_sub)
        # Symmetric target gate (target wires can be permuted) (For example CSWAP)
        else:
            for control_permuted in itertools.permutations(circuit_control):
                control_permuted = list(control_permuted)
                for target_permuted in itertools.permutations(circuit_target):
                    target_permuted = list(target_permuted)
                    first_match_qubits_sub = [-1] * n_qubits_p
                    for q in node_p.wires:
                        node_circuit_perm = control_permuted + target_permuted
                        first_match_qubits_sub[q] = node_circuit_perm[node_p.wires.index(q)]
                    first_match_qubits.append(first_match_qubits_sub)
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


def _update_qubits(circuit_dag, qubits_conf):
    """
    Update the qubits, target qubits and the control qubits given the mapping configuration between the circuit
    and the pattern.

    Args:
        circuit_dag (.CommutationDAG): the DAG representation of the circuit.
        qubits_conf (list): list of qubits of the mapping configuration.
    Return:
        list(list(int)): Wires
        list(list(int)): Target wires
        list(list(int)): Control wires
    """
    # pylint: disable=too-many-arguments
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


def _add_match(match_list, backward_match_list):
    """
    Add a match configuration found by pattern matching if it is not already in final list of matches.
    If the match is already in the final list, the qubit configuration is added to the existing Match.
    Args:
        match_list (list): match from the backward part of the algorithm.
        backward_match_list (list): List of matches found by the algorithm for a given configuration.
    """

    already_in = False

    for b_match in backward_match_list:
        for l_match in match_list:
            if b_match.match == l_match.match:
                index = match_list.index(l_match)
                match_list[index].qubit.append(b_match.qubit[0])
                already_in = True

        if not already_in:
            match_list.append(b_match)


def _compare_qubits(node1, wires1, control1, target1, wires2, control2, target2):
    """Compare the qubit configurations of two operations. The operations are supposed to be similar up to their
    qubits configuration.
    Args:
        node1 (.CommutationDAGNode): First node.
        wires1 (list(int)): Wires of the first node.
        control1 (list(int)): Control wires of the first node.
        target1 (list(int)): Target wires of the first node.
        wires2 (list(int)): Wires of the second node.
        control2 (list(int)): Control wires of the second node.
        target2 (list(int)): Target wires of the second node.
    """
    # pylint: disable=too-many-instance-attributes, too-many-arguments

    control_base = {
        "CNOT": "PauliX",
        "CZ": "PauliZ",
        "CY": "PauliY",
        "CSWAP": "SWAP",
        "Toffoli": "PauliX",
        "ControlledPhaseShift": "PhaseShift",
        "CRX": "RX",
        "CRY": "RY",
        "CRZ": "RZ",
        "CRot": "Rot",
        "MultiControlledX": "PauliX",
        "ControlledOperation": "ControlledOperation",
    }

    if control1 and set(control1) == set(control2):
        if control_base[node1.op.name] in symmetric_over_all_wires and set(target1) == set(target2):
            return True
        if target1 == target2:
            return True
    else:
        if node1.op.name in symmetric_over_all_wires and set(wires1) == set(wires2):
            return True
        if wires1 == wires2:
            return True
    return False


class ForwardMatch:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """
    Class to apply pattern matching in the forward direction.
    """

    def __init__(
        self, circuit_dag, pattern_dag, node_id_c, node_id_p, wires, control_wires, target_wires
    ):
        """
        Create the ForwardMatch class.
        Args:
            circuit_dag (.CommutationDAG): circuit as commutation DAG.
            pattern_dag (.CommutationDAG): pattern as commutation DAG.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_p (int): index of the first gate matched in the pattern.
        """
        # pylint: disable=too-many-branches, too-many-arguments

        # Commutation DAG of the circuit
        self.circuit_dag = circuit_dag

        # Commutation DAG of the pattern
        self.pattern_dag = pattern_dag

        # Node ID in the circuit
        self.node_id_c = node_id_c

        # Node ID in the pattern
        self.node_id_p = node_id_p

        # List of mapped wires for each node
        self.wires = wires

        # List of mapped control wires for each node
        self.control_wires = control_wires

        # List of mapped target wires for each node
        self.target_wires = target_wires

        # Successors to visit
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
        Initialize the list of matched nodes between the circuit and the pattern with the first match found.
        """
        self.match.append([self.node_id_p, self.node_id_c])

    def _find_forward_candidates(self, node_id_p):
        """Find the candidate nodes to be matched in the pattern for a given node in the pattern.
        Args:
            node_id_p (int): Node ID in pattern.
        """
        matches = [i[0] for i in self.match]

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
        Return node and successors from the matched_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        Returns:
            CommutationDAGNode: Node from the matched_node_list.
            list(int): List of successors.
        """
        node = self.matched_nodes_list[list_id][1]
        succ = self.matched_nodes_list[list_id][2]
        return node, succ

    def _remove_node_forward(self, list_id):
        """Remove a node of the current matched_nodes_list for a given ID.
        Args:
            list_id (int): considered list id of the desired node.
        """
        self.matched_nodes_list.pop(list_id)

    def run_forward_match(self):
        """Apply the forward match algorithm and returns the list of matches given an initial match
        and a qubits configuration.
        """
        # pylint: disable=too-many-branches,too-many-nested-blocks

        # Initialization
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

            # Update of the successors to visit.
            successors_to_visit.pop(0)

            # Update the matched_nodes_list with new attribute successor to visit and sort the list.
            self.matched_nodes_list.append([v_first.node_id, v_first, successors_to_visit])
            self.matched_nodes_list.sort(key=lambda x: x[2])

            # If the node is blocked and already matched go to the end
            if self.circuit_blocked[v[0]] | (self.circuit_matched_with[v[0]] != []):
                continue

            # Search for potential candidates in the pattern
            self._find_forward_candidates(self.circuit_matched_with[v_first.node_id][0])

            match = False

            # For loop over the candidates from the pattern to find a match.
            for i in self.candidates:
                # Break the for loop if a match is found.
                if match:
                    break

                node_circuit = self.circuit_dag.get_node(label)
                node_pattern = self.pattern_dag.get_node(i)

                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(self.wires[label]) != len(node_pattern.wires)
                    or set(self.wires[label]) != set(node_pattern.wires)
                    or node_circuit.op.name != node_pattern.op.name
                ):
                    continue

                # Compare two operations
                if _compare_operation_without_qubits(node_circuit, node_pattern):
                    if _compare_qubits(
                        node_circuit,
                        self.wires[label],
                        self.target_wires[label],
                        self.control_wires[label],
                        node_pattern.wires,
                        node_pattern.control_wires,
                        node_pattern.target_wires,
                    ):
                        # A match happens
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


class Match:  # pylint: disable=too-few-public-methods
    """
    Object to represent a match and its qubits configurations.
    """

    def __init__(self, match, qubit):
        """Create a Match class with necessary arguments.
        Args:
            match (list): list of matched gates.
            qubit (list): list of qubits configuration.
        """
        # Match list
        self.match = match
        # Qubits list for circuit
        if any(isinstance(el, list) for el in qubit):
            self.qubit = qubit
        else:
            self.qubit = [qubit]


class MatchingScenarios:  # pylint: disable=too-few-public-methods
    """
    Class to represent a matching scenario in the Backward part of the algorithm.
    """

    def __init__(
        self, circuit_matched, circuit_blocked, pattern_matched, pattern_blocked, matches, counter
    ):
        """Create a MatchingScenarios class for the Backward match.
        Args:
            circuit_matched (list): list representing the matched gates in the circuit.
            circuit_blocked (list): list representing the blocked gates in the circuit.
            pattern_matched (list): list representing the matched gates in the pattern.
            pattern_blocked (list): list representing the blocked gates in the pattern.
            matches (list): list of matches.
            counter (int): counter of the number of circuit gates already considered.
        """
        # pylint: disable=too-many-arguments

        self.circuit_matched = circuit_matched
        self.pattern_matched = pattern_matched
        self.circuit_blocked = circuit_blocked
        self.pattern_blocked = pattern_blocked
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


class BackwardMatch:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """
    Class BackwardMatch allows to run backward direction part of the pattern matching algorithm.
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
            pattern_dag (DAGDependency): pattern in the dag dependency form.
            forward_matches (list): list of match obtained in the forward direction.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_p (int): index of the first gate matched in the pattern.
            wires (list):
            control_wires (list):
            target_wires (list):
        """
        # pylint: disable=too-many-arguments

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

    def _find_backward_candidates(self, pattern_blocked, matches):
        """Function which returns the list possible backward candidates in the pattern dag.
        Args:
            pattern_blocked (list(bool)): list of blocked nodes in the pattern circuit.
            matches (list(int)): list of matches.
        Returns:
            list(int): list of backward candidates ID.
        """
        pattern_block = []

        for node_id in range(self.node_id_p, self.pattern_dag.size):
            if pattern_blocked[node_id]:
                pattern_block.append(node_id)

        matches_pattern = sorted(match[0] for match in matches)

        successors = self.pattern_dag.get_node(self.node_id_p).successors
        potential = []
        for index in range(self.node_id_p + 1, self.pattern_dag.size):
            if (index not in successors) and (index not in pattern_block):
                potential.append(index)

        candidates_indices = list(set(potential) - set(matches_pattern))
        candidates_indices = sorted(candidates_indices)
        candidates_indices.reverse()

        return candidates_indices

    def run_backward_match(self):
        """Run the backward match algorithm and returns the list of matches given an initial match, a forward
        scenario and a circuit qubits configuration.
        """
        # pylint: disable=too-many-branches,too-many-statements,too-many-nested-blocks
        match_store_list = []

        counter = 1

        # Initialize the blocked pattern list
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
        gate_indices = _gate_indices(self.circuit_matched, self.circuit_blocked)

        number_of_gate_to_match = (
            self.pattern_dag.size - (self.node_id_p - 1) - len(self.forward_matches)
        )

        # While the scenario stack is not empty.
        while self.matching_list.matching_scenarios_list:
            scenario = self.matching_list.pop_scenario()

            circuit_matched = scenario.circuit_matched
            circuit_blocked = scenario.circuit_blocked
            pattern_matched = scenario.pattern_matched
            pattern_blocked = scenario.pattern_blocked
            matches_scenario = scenario.matches
            counter_scenario = scenario.counter

            # Part of the match list coming from the backward match.
            match_backward = [
                match for match in matches_scenario if match not in self.forward_matches
            ]

            # Matches are stored
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
                    pattern_matched,
                    pattern_blocked,
                    matches_scenario,
                    counter_scenario + 1,
                )
                self.matching_list.append_scenario(matching_scenario)
                continue

            # The backward candidates in the pattern.
            candidates_indices = self._find_backward_candidates(pattern_blocked, matches_scenario)

            # Get the different wires variables
            wires1 = self.wires[circuit_id]
            control_wires1 = self.control_wires[circuit_id]
            target_wires1 = self.target_wires[circuit_id]

            global_match = False
            global_broken = []

            # Loop over the pattern candidates.
            for pattern_id in candidates_indices:
                node_pattern = self.pattern_dag.get_node(pattern_id)
                wires2 = self.pattern_dag.get_node(pattern_id).wires
                control_wires2 = self.pattern_dag.get_node(pattern_id).control_wires
                target_wires2 = self.pattern_dag.get_node(pattern_id).target_wires

                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(wires1) != len(wires2)
                    or set(wires1) != set(wires2)
                    or node_circuit.op.name != node_pattern.op.name
                ):
                    continue
                # Compare two operations
                if _compare_operation_without_qubits(node_circuit, node_pattern):
                    if _compare_qubits(
                        node_circuit,
                        wires1,
                        control_wires1,
                        target_wires1,
                        wires2,
                        control_wires2,
                        target_wires2,
                    ):
                        # A match happens.
                        # If there is a match the attributes are copied.
                        circuit_matched_match = circuit_matched.copy()
                        circuit_blocked_match = circuit_blocked.copy()

                        pattern_matched_match = pattern_matched.copy()
                        pattern_blocked_match = pattern_blocked.copy()

                        matches_scenario_match = matches_scenario.copy()

                        block_list = []
                        broken_matches_match = []

                        # Loop to check if the match is not connected, in this case
                        # the successors matches are blocked and unmatched.
                        for potential_block in self.pattern_dag.successors(pattern_id):
                            if not pattern_matched_match[potential_block]:
                                pattern_blocked_match[potential_block] = True
                                block_list.append(potential_block)
                                for block_id in block_list:
                                    for succ_id in self.pattern_dag.successors(block_id):
                                        pattern_blocked_match[succ_id] = True
                                        if pattern_matched_match[succ_id]:
                                            new_id = pattern_matched_match[succ_id][0]
                                            circuit_matched_match[new_id] = []
                                            pattern_matched_match[succ_id] = []
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
                            if back_match not in new_matches_scenario_match:  # pragma: no cover
                                condition = False
                                break

                        # First option greedy match.
                        if ([self.node_id_p, self.node_id_c] in new_matches_scenario_match) and (
                            condition or not match_backward
                        ):
                            pattern_matched_match[pattern_id] = [circuit_id]
                            circuit_matched_match[circuit_id] = [pattern_id]
                            new_matches_scenario_match.append([pattern_id, circuit_id])

                            new_matching_scenario = MatchingScenarios(
                                circuit_matched_match,
                                circuit_blocked_match,
                                pattern_matched_match,
                                pattern_blocked_match,
                                new_matches_scenario_match,
                                counter_scenario + 1,
                            )
                            self.matching_list.append_scenario(new_matching_scenario)

                            global_match = True

            if global_match:
                circuit_matched_block_s = circuit_matched.copy()
                circuit_blocked_block_s = circuit_blocked.copy()

                pattern_matched_block_s = pattern_matched.copy()
                pattern_blocked_block_s = pattern_blocked.copy()

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
                        pattern_matched_block_s[new_id] = []
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
                        pattern_matched_block_s,
                        pattern_blocked_block_s,
                        new_matches_scenario_block_s,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(new_matching_scenario)

                # Third option: if blocking the succesors breaks a match, we consider
                # also the possibility to block all predecessors (push the gate to the left).
                if broken_matches and all(global_broken):
                    circuit_matched_block_p = circuit_matched.copy()
                    circuit_blocked_block_p = circuit_blocked.copy()

                    pattern_matched_block_p = pattern_matched.copy()
                    pattern_blocked_block_p = pattern_blocked.copy()

                    matches_scenario_block_p = matches_scenario.copy()

                    circuit_blocked_block_p[circuit_id] = True

                    for pred in self.circuit_dag.get_node(circuit_id).predecessors:
                        circuit_blocked_block_p[pred] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched_block_p,
                        circuit_blocked_block_p,
                        pattern_matched_block_p,
                        pattern_blocked_block_p,
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
                        pattern_matched,
                        pattern_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

                else:
                    circuit_matched_nomatch = circuit_matched.copy()
                    circuit_blocked_nomatch = circuit_blocked.copy()

                    pattern_matched_nomatch = pattern_matched.copy()
                    pattern_blocked_nomatch = pattern_blocked.copy()

                    matches_scenario_nomatch = matches_scenario.copy()

                    # Second option, all predecessors are blocked (circuit gate is
                    # moved to the left).
                    for pred in predecessors:
                        circuit_blocked[pred] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        pattern_matched,
                        pattern_blocked,
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
                            pattern_matched_nomatch,
                            pattern_blocked_nomatch,
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


class MaximalMatches:  # pylint: disable=too-few-public-methods
    """
    Class MaximalMatches allows to sort and store the maximal matches from the list
    of matches obtained with the pattern matching algorithm.
    """

    def __init__(self, pattern_matches):
        """Initialize MaximalMatches with the necessary arguments.
        Args:
            pattern_matches (list): list of matches obtained from running the algorithm.
        """
        self.pattern_matches = pattern_matches

        self.max_match_list = []

    def run_maximal_matches(self):
        """Method that extracts and stores maximal matches in decreasing length order."""

        self.max_match_list = [
            Match(
                sorted(self.pattern_matches[0].match),
                self.pattern_matches[0].qubit,
            )
        ]

        for matches in self.pattern_matches[1::]:
            present = False
            for max_match in self.max_match_list:
                for elem in matches.match:
                    if elem in max_match.match and len(matches.match) <= len(max_match.match):
                        present = True
            if not present:
                self.max_match_list.append(Match(sorted(matches.match), matches.qubit))


class SubstitutionConfig:  # pylint: disable=too-many-arguments, too-few-public-methods
    """Class to store the configuration of a given match substitution, which circuit gates, template gates,
    qubits and predecessors of the match in the circuit.
    """

    def __init__(
        self,
        circuit_config,
        template_config,
        pred_block,
        qubit_config,
        template_dag,
    ):
        self.template_dag = template_dag
        self.circuit_config = circuit_config
        self.template_config = template_config
        self.qubit_config = qubit_config
        self.pred_block = pred_block


class TemplateSubstitution:  # pylint: disable=too-few-public-methods
    """Class to run the substitution algorithm from the list of maximal matches."""

    def __init__(self, max_matches, circuit_dag, template_dag, custom_quantum_cost=None):
        """
        Initialize TemplateSubstitution with necessary arguments.
        Args:
            max_matches (list(int)): list of maximal matches obtained from the running the pattern matching algorithm.
            circuit_dag (.CommutationDAG): circuit in the dag dependency form.
            template_dag (.CommutationDAG): template in the dag dependency form.
            custom_quantum_cost (dict): Optional, quantum cost that overrides the default cost dictionnary.
        """

        self.match_stack = max_matches
        self.circuit_dag = circuit_dag
        self.template_dag = template_dag

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
                "Hadamard": 1,
                "T": 1,
                "Adjoint(T)": 1,
                "S": 1,
                "Adjoint(S)": 1,
                "CNOT": 2,
                "CZ": 4,
                "SWAP": 6,
                "CSWAP": 63,
                "Toffoli": 21,
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
            exclude = exclude | set(elem.circuit_config) | set(elem.pred_block)

        pred = list(predecessors - set(circuit_sublist) - exclude)
        pred.sort()

        return pred

    def _quantum_cost(self, left, right):
        """Compare the two parts of the template and returns True if the quantum cost is reduced.
        Args:
            left (list): list of matched nodes in the template.
            right (list): list of nodes to be replaced.
        Returns:
            bool: True if the quantum cost is reduced
        """
        cost_left = 0
        for i in left:
            cost_left += self.quantum_cost[self.template_dag.get_node(i).op.name]

        cost_right = 0
        for j in right:
            cost_right += self.quantum_cost[self.template_dag.get_node(j).op.name]

        return cost_left > cost_right

    def _rules(self, circuit_sublist, template_sublist, template_complement):
        """Set of rules to decide whether the match is to be substitute or not.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
            template_sublist (list): list of matched nodes in the template.
            template_complement (list): list of gates not matched in the template.
        Returns:
            bool: True if the match respects the given rule for replacement, False otherwise.
        """

        if self._quantum_cost(template_sublist, template_complement):
            for elem in circuit_sublist:
                for config in self.substitution_list:
                    if any(elem == x for x in config.circuit_config):
                        return False
            return True
        return False

    def _template_inverse(self, template_list, template_sublist, template_complement):
        """The template circuit realizes the identity operator, then given the list of matches in the template,
        it returns the inverse part of the template that will be replaced.
        Args:
            template_list (list): list of all gates in the template.
            template_sublist (list): list of the gates matched in the circuit.
            template_complement  (list): list of gates not matched in the template.
        Returns:
            list(int): the template inverse part that will substitute the circuit match.
        """
        inverse = template_complement
        left = []
        right = []

        pred = set()
        for index in template_sublist:
            pred = pred | set(self.template_dag.get_node(index).predecessors)
        pred = list(pred - set(template_sublist))

        succ = set()
        for index in template_sublist:
            succ = succ | set(self.template_dag.get_node(index).successors)
        succ = list(succ - set(template_sublist))

        comm = list(set(template_list) - set(pred) - set(succ))

        for elem in inverse:
            if elem in pred:
                left.append(elem)
            elif elem in succ:
                right.append(elem)
            elif elem in comm:
                right.append(elem)

        left.sort()
        right.sort()

        left.reverse()
        right.reverse()

        total = left + right
        return total

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

        while self.match_stack:
            # Get the first match scenario of the list
            current = self.match_stack.pop(0)

            current_match = current.match
            current_qubit = current.qubit

            template_sublist = [x[0] for x in current_match]
            circuit_sublist = [x[1] for x in current_match]
            circuit_sublist.sort()

            template_list = range(0, self.template_dag.size)
            template_complement = list(set(template_list) - set(template_sublist))

            # If the match obey the rule then it is added to the list.
            if self._rules(circuit_sublist, template_sublist, template_complement):
                template_sublist_inverse = self._template_inverse(
                    template_list, template_sublist, template_complement
                )

                config = SubstitutionConfig(
                    circuit_sublist,
                    template_sublist_inverse,
                    [],
                    current_qubit,
                    self.template_dag,
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
