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

from collections import defaultdict
from copy import copy
from functools import cached_property
from operator import itemgetter
from typing import Sequence

import numpy as np
import rustworkx as rx

import pennylane as qml
from pennylane.ops import Prod, SProd
from pennylane.pauli.utils import (
    are_identical_pauli_words,
    binary_to_pauli,
    observables_to_binary_matrix,
)
from pennylane.wires import Wires

from .graph_colouring import recursive_largest_first

GROUPING_TYPES = frozenset(["qwc", "commuting", "anticommuting"])

RX_STRATEGIES = {
    "lf": rx.ColoringStrategy.Degree,
    "dsatur": rx.ColoringStrategy.Saturation,
    "gis": rx.ColoringStrategy.IndependentSet,
}

GRAPH_COLOURING_METHODS = frozenset(RX_STRATEGIES.keys()).union({"rlf"})


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
        observables (list[Observable]): A list of Pauli words to be partitioned according to a
            grouping strategy.
        grouping_type (str): The binary relation used to define partitions of
            the Pauli words, can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or
            ``'anticommuting'``.
        graph_colourer (str): The heuristic algorithm to employ for graph
                colouring, can be ``'lf'`` (Largest First), ``'rlf'`` (Recursive
                Largest First), `dsatur` (DSATUR), or `gis` (IndependentSet). Defaults to ``'lf'``.

    See Also:
        `rustworkx.ColoringStrategy <https://www.rustworkx.org/apiref/rustworkx.ColoringStrategy.html#coloringstrategy>`_
        for more information on the ``('lf', 'dsatur', 'gis')`` strategies.

    Raises:
        ValueError: if arguments specified for ``grouping_type`` or ``graph_colourer``
        are not recognized.
    """

    def __init__(self, observables, grouping_type="qwc", graph_colourer="lf"):
        if grouping_type.lower() not in GROUPING_TYPES:
            raise ValueError(
                f"Grouping type must be one of: {GROUPING_TYPES}, instead got {grouping_type}."
            )

        if graph_colourer.lower() not in GRAPH_COLOURING_METHODS:
            raise ValueError(
                f"Graph colouring method must be one of: {GRAPH_COLOURING_METHODS}, "
                f"instead got {graph_colourer}."
            )

        self.graph_colourer = graph_colourer.lower()
        self.grouping_type = grouping_type.lower()
        self.observables = observables
        self._wire_map = None

    @cached_property
    def binary_observables(self):
        """Binary Matrix corresponding to the symplectic representation of ``self.observables``.

        It is an m x n matrix where each row is the symplectic (binary) representation of
        ``self.observables``, with ``m = len(self.observables)`` and n the
        number of qubits acted on by the observables.
        """
        return self.binary_repr()

    def binary_repr(self, n_qubits=None, wire_map=None):
        """Converts the list of Pauli words to a binary matrix,
        i.e. a matrix where row m is the symplectic representation of ``self.observables[m]``.

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

        return observables_to_binary_matrix(self.observables, n_qubits, self._wire_map)

    @cached_property
    def adj_matrix(self) -> np.ndarray:
        """Adjacency matrix for the complement of the Pauli graph determined by the ``grouping_type``.

        The adjacency matrix for an undirected graph of N nodes is an N by N symmetric binary
        matrix, where matrix elements of 1 denote an edge, and matrix elements of 0 denote no edge.
        """
        return adj_matrix_from_symplectic(self.binary_observables, grouping_type=self.grouping_type)

    def colour_pauli_graph(self):
        """
        Runs the graph colouring heuristic algorithm to obtain the partitioned Pauli observables.

        Returns:
            list[list[Observable]]: List of partitions of the Pauli observables made up of mutually (anti-)commuting observables.
        """
        if self.graph_colourer == "rlf":
            coloured_binary_paulis = recursive_largest_first(
                self.binary_observables, self.adj_matrix
            )

            # Need to convert back from the symplectic representation
            pauli_groups = [
                [binary_to_pauli(pauli_word, wire_map=self._wire_map) for pauli_word in grouping]
                for grouping in coloured_binary_paulis.values()
            ]

        else:
            pauli_groups = self.pauli_partitions_from_graph()

        return pauli_groups

    def idx_partitions_from_graph(self) -> list[list]:
        """Colours the complement graph using a greedy colouring algorithm and groups indices by colour.

        This function uses the `graph_greedy_color` function from `rx` to colour the graph defined by
        `self.complement_graph` using a specified strategy from `RX_STRATEGIES`. It then groups the indices
        (nodes) of the graph by their assigned colours.

        Returns:
            dict[int, list[int]]: A dictionary where the keys are colours (integers) and the values are lists
            of indices (nodes) that have been assigned that colour.
        """
        # A dictionary where keys are node indices and the value is the color
        colouring_dict = rx.graph_greedy_color(
            self.complement_graph, strategy=RX_STRATEGIES[self.graph_colourer]
        )
        # group together indices (values) of the same colour (keys)
        groups = defaultdict(list)
        for idx, colour in sorted(colouring_dict.items()):
            groups[colour].append(idx)

        return groups

    def pauli_partitions_from_graph(self) -> list[list]:
        """Partition Pauli observables into lists of (anti-)commuting observables
        using Rustworkx graph colouring algorithms.

        Returns:
            list[list[Observable]]: List of partitions of the Pauli observables made up of mutually (anti-)commuting terms.
        """
        groups = self.idx_partitions_from_graph()

        # Get the observables from the indices. itemgetter outperforms list comprehension
        pauli_groups = obs_partitions_from_idx_partitions(self.observables, groups.values())
        return pauli_groups

    @property
    def complement_graph(self) -> rx.PyGraph:
        """
        Complement graph of the (anti-)commutation graph constructed from the Pauli observables.

        Edge (i,j) is present in the graph if observable[i] and observable[j] do NOT commute under
        the ``grouping_type`` strategy.

        The nodes are the observables (can only be accesssed through their integer index).
        """
        # Use upper triangle since adjacency matrix is symmetric and we have an undirected graph
        edges = list(zip(*np.where(np.triu(self.adj_matrix, k=1))))
        # Create complement graph
        graph = rx.PyGraph()
        graph.add_nodes_from(self.observables)
        graph.add_edges_from_no_data(edges)
        return graph


def obs_partitions_from_idx_partitions(
    observables: list, idx_partitions: Sequence[Sequence[int]]
) -> list[list]:
    """Get the partitions of the observables corresponding to the partitions of the indices.

    Args:
        observables (list[Observable]): A list of Pauli words to be partitioned according to a
            grouping strategy.
        idx_partitions (Sequence[Sequence[int]]): Sequence of sequences containing the indices of the partitioned observables.

    Returns:
        list[list[Observable]]: List of partitions of the Pauli observables made up of mutually (anti-)commuting terms.
    """
    pauli_groups = [
        (
            list(itemgetter(*indices)(observables))
            if len(indices) > 1
            else [itemgetter(*indices)(observables)]
        )
        for indices in idx_partitions
    ]

    return pauli_groups


def adj_matrix_from_symplectic(symplectic_matrix: np.ndarray, grouping_type: str):
    """Get adjacency matrix of (anti-)commuting graph based on grouping type.

    This is the adjacency matrix of the complement graph. Based on symplectic representations and inner product of [1].

    [1] Andrew Jena (2019). Partitioning Pauli Operators: in Theory and in Practice.
    UWSpace. http://hdl.handle.net/10012/15017

    Args:
        symplectic_matrix (np.ndarray): 2D symplectic matrix. Each row corresponds to the
        symplectic representation of the Pauli observables.
        grouping_type (str): the binary relation used to define partitions of
            the Pauli words, can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or
            ``'anticommuting'``.

    Returns:
        np.ndarray: Adjacency matrix. Binary matrix such that adj_matrix[i,j] = 1 if observables[i]
        observables[j] do NOT (anti-)commute, as determined by the ``grouping_type``.
    """

    n_qubits = symplectic_matrix.shape[1] // 2

    # Convert symplectic representation to integer format.
    # This is equivalent to the map: {0: I, 1: X, 2:Y, Z:3}
    pauli_matrix_int = np.array(
        [2 * row[:n_qubits] + row[n_qubits:] for row in symplectic_matrix], dtype=np.int8
    )
    # Broadcast the second dimension, sucht that pauli_matrix_broad.shape = (m, 1, n_qubits)
    # with m = len(observables). This allows for calculation of all possible combinations of Pauli observable pairs (Pi, Pj).
    # Something like: result[i, j, k] = pauli_matrix_int[i, k] * pauli_matrix_int[j, k]
    pauli_matrix_broad = pauli_matrix_int[:, None]
    # Calculate the symplectic inner product in [1], using the integer representation - hence the difference in the equation form.
    # qubit_anticommutation_mat[i,j, k] is k=0 if Pi and Pj commute, else k!=0 if they anticommute.
    qubit_anticommutation_mat = (pauli_matrix_int * pauli_matrix_broad) * (
        pauli_matrix_int - pauli_matrix_broad
    )
    # 'adjacency_mat[i, j]' is True iff Paulis 'i' and 'j' do not (anti-)commute under given grouping_type.
    if grouping_type == "qwc":
        # True if any term anti commutes
        adj_matrix = np.logical_or.reduce(qubit_anticommutation_mat, axis=2)
    elif grouping_type == "commuting":
        # True if the number of anti commuting terms is odd (anti commte)
        adj_matrix = np.logical_xor.reduce(qubit_anticommutation_mat, axis=2)
    else:
        # True if the number of anti commuting terms is even (commute)
        adj_matrix = ~np.logical_xor.reduce(qubit_anticommutation_mat, axis=2)

    return adj_matrix


def compute_partition_indices(
    observables: list, grouping_type: str = "qwc", method: str = "lf"
) -> tuple[tuple[int]]:
    """
    Computes the partition indices of a list of observables using a specified grouping type
    and graph colouring method.

    Args:
        observables (list[Observable]): A list of Pauli Observables to be partitioned.
        grouping_type (str): The type of binary relation between Pauli observables.
            Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``. Defaults to ``'qwc'``.
        method (str): The graph colouring heuristic to use in solving minimum clique cover.
            Can be ``'lf'`` (Largest First), ``'dsatur'`` (Degree of Saturation), or ``'gis'`` (Greedy Independent Set).
            Defaults to ``'lf'``.

    Returns:
        tuple[tuple[int]]: A tuple of tuples where each inner tuple contains the indices of
        observables that are grouped together according to the specified grouping type and
        graph colouring method.

    Raises:
        ValueError: If method is ``'rlf'`` as it is not a supported heuristic for this implementation.

    **Example**

        >>> observables = [qml.X(0) @ qml.Z(1), qml.Z(0), qml.X(1)]
        >>> compute_partition_indices(observables, grouping_type="qwc", method="lf")
        ((0,), (1, 2))
    """
    if method in RX_STRATEGIES.keys():
        idx_no_wires = []
        has_obs_with_wires = False
        for idx, obs in enumerate(observables):
            if len(obs.wires) == 0:
                idx_no_wires.append(idx)
            else:
                has_obs_with_wires = True
                break

        if not has_obs_with_wires:
            return (tuple(idx_no_wires),)

        pauli_groupper = PauliGroupingStrategy(
            observables, grouping_type=grouping_type, graph_colourer=method
        )

        idx_dictionary = pauli_groupper.idx_partitions_from_graph()

        partition_indices = tuple(tuple(indices) for indices in idx_dictionary.values())
    elif method == "rlf":
        # 'rlf' method is not compatible with the rx implementation.
        partition_indices = _compute_partition_indices_rlf(observables, grouping_type=grouping_type)
    else:
        raise ValueError(
            f"Graph colouring method must be one of: {GRAPH_COLOURING_METHODS}, "
            f"instead got {method}."
        )

    return partition_indices


def _compute_partition_indices_rlf(observables: list, grouping_type: str):
    """Computes the partition indices of a list of observables using a specified grouping type and 'rlf' method.

    This option is much less efficient so should be avoided.
    """

    with qml.QueuingManager.stop_recording():
        obs_groups = qml.pauli.group_observables(
            observables, grouping_type=grouping_type, method="rlf"
        )

    observables = copy(observables)

    indices = []
    available_indices = list(range(len(observables)))
    for partition in obs_groups:  # pylint:disable=too-many-nested-blocks
        indices_this_group = []
        for pauli_word in partition:
            # find index of this pauli word in remaining original observables,
            for ind, observable in enumerate(observables):
                if qml.pauli.are_identical_pauli_words(pauli_word, observable):
                    indices_this_group.append(available_indices[ind])
                    # delete this observable and its index, so it cannot be found again
                    observables.pop(ind)
                    available_indices.pop(ind)
                    break
        indices.append(tuple(indices_this_group))

    return tuple(indices)


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

    pauli_groupper = PauliGroupingStrategy(
        wires_obs, grouping_type=grouping_type, graph_colourer=method
    )

    temp_opmath = not qml.operation.active_new_opmath() and any(
        isinstance(o, (Prod, SProd)) for o in observables
    )
    if temp_opmath:
        qml.operation.enable_new_opmath(warn=False)

    try:
        partitioned_paulis = pauli_groupper.colour_pauli_graph()
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
