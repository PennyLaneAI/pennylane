# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
This module contains the high-level Pauli word partitioning functionality used in measurement optimization.
"""

from pennylane.wires import Wires
from pennylane.grouping.utils import (
    convert_observables_to_binary_matrix,
    binary_to_pauli,
    are_identical_pauli_words,
    get_qwc_complement_adj_matrix,
)
from pennylane.grouping.graph_colouring import largest_first, recursive_largest_first
import numpy as np

GROUPING_TYPES = frozenset(["qwc", "commuting", "anticommuting"])
GRAPH_COLOURING_METHODS = {"lf": largest_first, "rlf": recursive_largest_first}


class PauliGroupingStrategy:  # pylint: disable=too-many-instance-attributes
    """
    Class for partitioning a list of Pauli words according to some binary symmetric relation.

    Partitions are defined by the binary symmetric relation of interest, e.g. all Pauli words in a
    partition being mutually commuting. The partitioning is accomplished by formulating the list of
    Pauli words as a graph where nodes represent Pauli words and edges between nodes denotes that
    the two corresponding Pauli words satisfy the symmetric binary relation.

    Obtaining the fewest number of partitions such that all Pauli terms within a partition mutually
    satisfy the binary relation can then be accomplished by finding a partition of the graph nodes
    such that each partition is a fully connected subgraph (a "clique"). The problem of finding the
    optimal partitioning, i.e. the fewest number of cliques, is the minimum clique cover (MCC)
    problem. The solution of MCC may be found by graph colouring on the complementary graph. Both
    MCC and graph colouring are NP-Hard, so heuristic graph colouring algorithms are employed to
    find approximate solutions in polynomial time.

    Args:
        observables (list[Observable]): A list of Pauli words to be partitioned according to a
        grouping strategy.

    Keyword Args:
        grouping_type (str): The binary relation used to define partitions of the Pauli words.
        graph_colourer (str): The heuristic algorithm to employ for graph colouring.

    Raises:
        ValueError: if `grouping_type` or `graph_colourer` are not recognized as elements of
        `GROUPING_TYPES` or `GRAPH_COLOURING_METHODS` respectively.

    """

    def __init__(self, observables, grouping_type="qwc", graph_colourer="rlf"):

        if grouping_type.lower() not in GROUPING_TYPES:
            raise ValueError(
                "Grouping type must be one of: {}, instead got {}.".format(
                    GROUPING_TYPES, grouping_type
                )
            )

        self.grouping_type = grouping_type.lower()

        if graph_colourer.lower() not in GRAPH_COLOURING_METHODS.keys():
            raise ValueError(
                "Graph colouring method must be one of: {}, instead got {}.".format(
                    list(GRAPH_COLOURING_METHODS.keys()), graph_colourer
                )
            )

        self.graph_colourer = GRAPH_COLOURING_METHODS[graph_colourer.lower()]
        self.observables = observables
        self.__wire_map = None
        self.__n_qubits = None
        self.binary_observables = None
        self.adj_matrix = None
        self.grouped_paulis = None

    def obtain_binary_repr(self, n_qubits=None, wire_map=None):
        """Converts the list of Pauli words to a binary matrix.

        Keyword args:
            n_qubits (int): number of qubits to specify dimension of binary vector representation.
            wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys,
                and unique integer labels as their values

        Returns:
            array[bool]: a column matrix of the Pauli words in binary vector representation.

        """

        if wire_map is None:
            self.__wire_map = {
                wire: c
                for c, wire in enumerate(
                    Wires.all_wires([obs.wires for obs in self.observables]).tolist()
                )
            }

        else:
            self.__wire_map = wire_map

        self.binary_observables = convert_observables_to_binary_matrix(
            self.observables, n_qubits, self.__wire_map
        )

        self.__n_qubits = n_qubits

        return self.binary_observables

    def construct_complement_adj_matrix_for_operator(self):
        """Constructs the adjacency matrix for complement of Pauli graph.

        The adjacency matrix for an undirected graph of N vertices is an N by N symmetric binary
        matrix, where matrix elements of 1 denote an edge, and matrix elements of 0 denote no edge.

        Returns:
            array[bool]: the square and symmetric adjacency matrix.

        """

        if self.binary_observables is None:
            self.obtain_binary_repr()

        n_qubits = int(np.shape(self.binary_observables)[0] / 2)

        if self.grouping_type == "qwc":
            adj = get_qwc_complement_adj_matrix(self.binary_observables)

        elif self.grouping_type in ["commuting", "anticommuting"]:
            symplectic_form = np.block(
                [
                    [np.zeros((n_qubits, n_qubits)), np.eye(n_qubits)],
                    [np.eye(n_qubits), np.zeros((n_qubits, n_qubits))],
                ]
            )
            mat_prod = (
                np.transpose(self.binary_observables) @ symplectic_form @ self.binary_observables
            )

            if self.grouping_type == "commuting":

                adj = mat_prod % 2

            elif self.grouping_type == "anticommuting":

                adj = (mat_prod + 1) % 2
                np.fill_diagonal(adj, 0)

        self.adj_matrix = adj
        return adj

    def colour_pauli_graph(self):
        """
        Runs the graph colouring heuristic algorithm to obtain the partitioned Pauli words.

        Returns:
            list[list[Observable]]: a list of the obtained groupings. Each grouping is itself a
            list of Pauli word `Observable` instances.

        """

        if self.adj_matrix is None:
            self.construct_complement_adj_matrix_for_operator()

        coloured_binary_paulis = self.graph_colourer(self.binary_observables, self.adj_matrix)

        self.grouped_paulis = [
            [binary_to_pauli(pauli_word, wire_map=self.__wire_map) for pauli_word in grouping]
            for grouping in coloured_binary_paulis.values()
        ]

        return self.grouped_paulis


def group_observables(observables, coefficients=None, grouping_type="qwc", method="rlf"):
    """Partitions a list of observables (Pauli operations and tensor products thereof) into
    groupings according to a binary relation (qubit-wise commuting, fully commuting, or
    anticommuting).

    Partitions are found by 1) mapping the list of observables to a graph where vertices represent
    observables and edges encode the binary relation, then 2) solving minimum clique cover for the
    graph using graph-coloring heuristic algorithms.

    **Example usage:**

    >>> observables = [qml.PauliY(0), qml.PauliX(0) @ qml.PauliX(1), qml.PauliZ(1)]
    >>> coefficients = [1.43, 4.21, 0.97]
    >>> obs_groupings, coeffs_groupings = group_observables(
                                                            observables,
                                                            coefficients,
                                                            'anticommuting',
                                                            'lf')
    >>> obs_groupings
    [[Tensor(PauliZ(wires=[1])),
      Tensor(PauliX(wires=[0]), PauliX(wires=[1]))],
     [Tensor(PauliY(wires=[0]))]]
    >>> coeffs_groupings
    [[0.97, 4.21], [1.43]]

    Args:
        observables (list[Observable]): a list of Pauli word `Observable` instances (Pauli
            operation instances and Tensor instances thereof).

    Keyword args:
        coefficients (list[scalar]): a list of scalar coefficients. If not specified,
            output `partitioned_coeffs` is not returned.
        grouping_type (str): the type of binary relation between Pauli words, can be 'qwc',
            'commuting', or 'anticommuting'.
        method (str): the graph coloring heuristic to use in solving minimum clique cover, which
            can be 'lf' (Largest First) or 'rlf' (Recursive Largest First).

    Returns:
       partitioned_paulis (list[list[Observable]]): a list of the obtained groupings. Each grouping
            is itself a list of Pauli word `Observable` instances.
       partitioned_coeffs (list[list[scalar]]): a list of coefficient groupings. Each coefficient
           grouping is itself a list of the groupings corresponding coefficients. (This is only
           output if coefficients are specified.)

    Raises:
        IndexError: if the input list of coefficients is not of the same length as the input list
            of Pauli words.
    """

    if coefficients is not None:
        if len(coefficients) != len(observables):
            raise IndexError(
                "The coefficients list must be the same length as the observables list."
            )

    pauli_grouping = PauliGroupingStrategy(
        observables, grouping_type=grouping_type, graph_colourer=method
    )
    partitioned_paulis = pauli_grouping.colour_pauli_graph()

    if coefficients is None:
        return partitioned_paulis

    partitioned_coeffs = [[0] * len(g) for g in partitioned_paulis]

    for i, partition in enumerate(partitioned_paulis):
        for j, pauli_word in enumerate(partition):
            for observable in observables:
                if are_identical_pauli_words(pauli_word, observable):
                    partitioned_coeffs[i][j] = coefficients[observables.index(observable)]
                    break

    return partitioned_paulis, partitioned_coeffs
