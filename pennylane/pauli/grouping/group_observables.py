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
This module contains the high-level Pauli-word-partitioning functionality used in measurement optimization.
"""

from copy import copy

import numpy as np
import pennylane as qml

from pennylane.ops import Prod, SProd
from pennylane.pauli.utils import (
    are_identical_pauli_words,
    binary_to_pauli,
    observables_to_binary_matrix,
    qwc_complement_adj_matrix,
)
from pennylane.wires import Wires

from .graph_colouring import largest_first, recursive_largest_first


GROUPING_TYPES = frozenset(["qwc", "commuting", "anticommuting"])
GRAPH_COLOURING_METHODS = {"lf": largest_first, "rlf": recursive_largest_first}


class PauliGroupingStrategy:  # pylint: disable=too-many-instance-attributes
    """
    Class for partitioning a list of Pauli words according to some binary symmetric relation.

    Partitions are defined by the binary symmetric relation of interest, e.g., all Pauli words in a
    partition being mutually commuting. The partitioning is accomplished by formulating the list of
    Pauli words as a graph where nodes represent Pauli words and edges between nodes denotes that
    the two corresponding Pauli words satisfy the symmetric binary relation.

    Obtaining the fewest number of partitions such that all Pauli terms within a partition mutually
    satisfy the binary relation can then be accomplished by finding a partition of the graph nodes
    such that each partition is a fully connected subgraph (a "clique"). The problem of finding the
    optimal partitioning, i.e., the fewest number of cliques, is the minimum clique cover (MCC)
    problem. The solution of MCC may be found by graph colouring on the complementary graph. Both
    MCC and graph colouring are NP-Hard, so heuristic graph colouring algorithms are employed to
    find approximate solutions in polynomial time.

    Args:
        observables (list[Observable]): a list of Pauli words to be partitioned according to a
            grouping strategy
        grouping_type (str): the binary relation used to define partitions of
            the Pauli words, can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or
            ``'anticommuting'``.
        graph_colourer (str): the heuristic algorithm to employ for graph
            colouring, can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive
            Largest First)

    Raises:
        ValueError: if arguments specified for ``grouping_type`` or
            ``graph_colourer`` are not recognized
    """

    def __init__(self, observables, grouping_type="qwc", graph_colourer="rlf"):
        if grouping_type.lower() not in GROUPING_TYPES:
            raise ValueError(
                f"Grouping type must be one of: {GROUPING_TYPES}, instead got {grouping_type}."
            )

        self.grouping_type = grouping_type.lower()

        if graph_colourer.lower() not in GRAPH_COLOURING_METHODS:
            raise ValueError(
                f"Graph colouring method must be one of: {list(GRAPH_COLOURING_METHODS)}, "
                f"instead got {graph_colourer}."
            )

        self.graph_colourer = GRAPH_COLOURING_METHODS[graph_colourer.lower()]
        self.observables = observables
        self._wire_map = None
        self._n_qubits = None
        self.binary_observables = None
        self.adj_matrix = None
        self.grouped_paulis = None

    def binary_repr(self, n_qubits=None, wire_map=None):
        """Converts the list of Pauli words to a binary matrix.

        Args:
            n_qubits (int): number of qubits to specify dimension of binary vector representation
            wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys,
                and unique integer labels as their values

        Returns:
            array[int]: a column matrix of the Pauli words in binary vector representation
        """

        if wire_map is None:
            self._wire_map = {
                wire: c
                for c, wire in enumerate(
                    Wires.all_wires([obs.wires for obs in self.observables]).tolist()
                )
            }

        else:
            self._wire_map = wire_map

        self._n_qubits = n_qubits

        return observables_to_binary_matrix(self.observables, n_qubits, self._wire_map)

    def complement_adj_matrix_for_operator(self):
        """Constructs the adjacency matrix for the complement of the Pauli graph.

        The adjacency matrix for an undirected graph of N vertices is an N by N symmetric binary
        matrix, where matrix elements of 1 denote an edge, and matrix elements of 0 denote no edge.

        Returns:
            array[int]: the square and symmetric adjacency matrix
        """

        if self.binary_observables is None:
            self.binary_observables = self.binary_repr()

        n_qubits = int(np.shape(self.binary_observables)[1] / 2)

        if self.grouping_type == "qwc":
            adj = qwc_complement_adj_matrix(self.binary_observables)

        elif self.grouping_type in frozenset(["commuting", "anticommuting"]):
            symplectic_form = np.block(
                [
                    [np.zeros((n_qubits, n_qubits)), np.eye(n_qubits)],
                    [np.eye(n_qubits), np.zeros((n_qubits, n_qubits))],
                ]
            )
            mat_prod = (
                self.binary_observables @ symplectic_form @ np.transpose(self.binary_observables)
            )

            if self.grouping_type == "commuting":
                adj = mat_prod % 2

            elif self.grouping_type == "anticommuting":
                adj = (mat_prod + 1) % 2
                np.fill_diagonal(adj, 0)

        return adj

    def colour_pauli_graph(self):
        """
        Runs the graph colouring heuristic algorithm to obtain the partitioned Pauli words.

        Returns:
            list[list[Observable]]: a list of the obtained groupings. Each grouping is itself a
            list of Pauli word ``Observable`` instances
        """

        if self.adj_matrix is None:
            self.adj_matrix = self.complement_adj_matrix_for_operator()

        coloured_binary_paulis = self.graph_colourer(self.binary_observables, self.adj_matrix)

        self.grouped_paulis = [
            [binary_to_pauli(pauli_word, wire_map=self._wire_map) for pauli_word in grouping]
            for grouping in coloured_binary_paulis.values()
        ]

        return self.grouped_paulis


def group_observables(observables, coefficients=None, grouping_type="qwc", method="rlf"):
    """Partitions a list of observables (Pauli operations and tensor products thereof) into
    groupings according to a binary relation (qubit-wise commuting, fully-commuting, or
    anticommuting).

    Partitions are found by 1) mapping the list of observables to a graph where vertices represent
    observables and edges encode the binary relation, then 2) solving minimum clique cover for the
    graph using graph-colouring heuristic algorithms.

    Args:
        observables (list[Observable]): a list of Pauli word ``Observable`` instances (Pauli
            operation instances and :class:`~.Tensor` instances thereof)
        coefficients (tensor_like): A tensor or list of coefficients. If not specified,
            output ``partitioned_coeffs`` is not returned.
        grouping_type (str): The type of binary relation between Pauli words.
            Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
        method (str): the graph coloring heuristic to use in solving minimum clique cover, which
            can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest First)

    Returns:
       tuple:

           * list[list[Observable]]: A list of the obtained groupings. Each grouping
             is itself a list of Pauli word ``Observable`` instances.
           * list[tensor_like]: A list of coefficient groupings. Each coefficient
             grouping is itself a tensor or list of the grouping's corresponding coefficients. This is only
             returned if coefficients are specified.

    Raises:
        IndexError: if the input list of coefficients is not of the same length as the input list
            of Pauli words

    **Example**

    >>> obs = [qml.Y(0), qml.X(0) @ qml.X(1), qml.Z(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'anticommuting', 'lf')
    >>> obs_groupings
    [[Z(1), X(0) @ X(1)],
     [Y(0)]]
    >>> coeffs_groupings
    [[0.97, 4.21], [1.43]]
    """

    if coefficients is not None:
        if qml.math.shape(coefficients)[0] != len(observables):
            raise IndexError(
                "The coefficients list must be the same length as the observables list."
            )

    no_wires_obs = []
    wires_obs = []
    for ob in observables:
        if len(ob.wires) == 0:
            no_wires_obs.append(ob)
        else:
            wires_obs.append(ob)
    if not wires_obs:
        if coefficients is None:
            return [no_wires_obs]
        return [no_wires_obs], [coefficients]

    pauli_grouping = PauliGroupingStrategy(
        wires_obs, grouping_type=grouping_type, graph_colourer=method
    )

    temp_opmath = not qml.operation.active_new_opmath() and any(
        isinstance(o, (Prod, SProd)) for o in observables
    )
    if temp_opmath:
        qml.operation.enable_new_opmath(warn=False)

    try:
        partitioned_paulis = pauli_grouping.colour_pauli_graph()
    finally:
        if temp_opmath:
            qml.operation.disable_new_opmath(warn=False)

    partitioned_paulis[0].extend(no_wires_obs)

    if coefficients is None:
        return partitioned_paulis

    partitioned_coeffs = _partition_coeffs(partitioned_paulis, observables, coefficients)

    return partitioned_paulis, partitioned_coeffs


def _partition_coeffs(partitioned_paulis, observables, coefficients):
    """Partition the coefficients according to the Pauli word groupings."""

    partitioned_coeffs = [
        qml.math.cast_like([0] * len(g), coefficients) for g in partitioned_paulis
    ]

    observables = copy(observables)
    # we cannot delete elements from the coefficients tensor, so we
    # use a proxy list memorising the indices for this logic
    coeff_indices = list(range(qml.math.shape(coefficients)[0]))
    for i, partition in enumerate(partitioned_paulis):  # pylint:disable=too-many-nested-blocks
        indices = []
        for pauli_word in partition:
            # find index of this pauli word in remaining original observables,
            for ind, observable in enumerate(observables):
                if isinstance(observable, qml.ops.Hamiltonian):
                    # Converts single-term Hamiltonian to SProd because
                    # are_identical_pauli_words cannot handle Hamiltonian
                    coeffs, ops = observable.terms()
                    # Assuming the Hamiltonian has only one term
                    observable = qml.s_prod(coeffs[0], ops[0])
                if are_identical_pauli_words(pauli_word, observable):
                    indices.append(coeff_indices[ind])
                    observables.pop(ind)
                    coeff_indices.pop(ind)
                    break

        # add a tensor of coefficients to the grouped coefficients
        partitioned_coeffs[i] = qml.math.take(coefficients, indices, axis=0)

    # make sure the output is of the same format as the input
    # for these two frequent cases
    if isinstance(coefficients, list):
        partitioned_coeffs = [list(p) for p in partitioned_coeffs]

    return partitioned_coeffs
