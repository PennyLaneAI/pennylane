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
from collections.abc import Sequence
from copy import copy
from functools import cached_property
from operator import itemgetter
from typing import Literal

import numpy as np
import rustworkx as rx

import pennylane as qml
from pennylane.pauli.utils import (
    are_identical_pauli_words,
    binary_to_pauli,
    observables_to_binary_matrix,
)
from pennylane.typing import TensorLike
from pennylane.wires import Wires

from .graph_colouring import recursive_largest_first

GROUPING_TYPES = frozenset(["qwc", "commuting", "anticommuting"])

# ColoringStrategy is only available from version 0.15.0
new_rx = True
try:
    RX_STRATEGIES = {
        "lf": rx.ColoringStrategy.Degree,
        "dsatur": rx.ColoringStrategy.Saturation,
        "gis": rx.ColoringStrategy.IndependentSet,
    }
except AttributeError:  # pragma: no cover
    new_rx = False  # pragma: no cover. # This error is raised for versions lower than 0.15.0
    RX_STRATEGIES = {"lf": None}  # pragma: no cover # Only "lf" can be used without a strategy

GRAPH_COLOURING_METHODS = frozenset(RX_STRATEGIES.keys()).union({"rlf"})


class PauliGroupingStrategy:
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
        observables (list[Operator]): A list of Pauli words to be partitioned according to a
            grouping strategy.
        grouping_type (str): The binary relation used to define partitions of
            the Pauli words, can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or
            ``'anticommuting'``.
        graph_colourer (str): The heuristic algorithm to employ for graph
            colouring, can be ``'lf'`` (Largest First), ``'rlf'`` (Recursive
            Largest First), ``'dsatur'`` (Degree of Saturation), or ``'gis'`` (IndependentSet). Defaults to ``'lf'``.

    Raises:
        ValueError: If arguments specified for ``grouping_type`` or ``graph_colourer``
            are not recognized.

    .. seealso:: `rustworkx.ColoringStrategy <https://www.rustworkx.org/apiref/rustworkx.ColoringStrategy.html#coloringstrategy>`_
        for more information on the ``('lf', 'dsatur', 'gis')`` strategies.
    """

    def __init__(
        self,
        observables,
        grouping_type: Literal["qwc", "commuting", "anticommuting"] = "qwc",
        graph_colourer: Literal["lf", "rlf", "dsatur", "gis"] = "lf",
    ):
        self.graph_colourer = graph_colourer.lower()
        self.grouping_type = grouping_type.lower()

        if self.grouping_type not in GROUPING_TYPES:
            raise ValueError(
                f"Grouping type must be one of: {GROUPING_TYPES}, instead got {grouping_type}."
            )

        if self.graph_colourer in ["dsatur", "gis"] and not new_rx:
            raise ValueError(
                f"The strategy '{graph_colourer}' is not supported in this version of Rustworkx. "
                "Please install rustworkx>=0.15.0 to access the 'dsatur' and 'gis' colouring strategies."
            )

        if self.graph_colourer not in GRAPH_COLOURING_METHODS:
            raise ValueError(
                f"Graph colouring method must be one of: {GRAPH_COLOURING_METHODS}, "
                f"instead got '{graph_colourer}'."
            )

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

        The adjacency matrix for an undirected graph of N nodes is an N x N symmetric binary
        matrix, where matrix elements of 1 denote an edge (grouping strategy is **not** satisfied), and
        matrix elements of 0 denote no edge (grouping strategy is satisfied).
        """
        return _adj_matrix_from_symplectic(
            self.binary_observables, grouping_type=self.grouping_type
        )

    @property
    def complement_graph(self) -> rx.PyGraph:
        """
        Complement graph of the (anti-)commutation graph constructed from the Pauli observables.

        Edge ``(i,j)`` is present in the graph if ``observable[i]`` and ``observable[j]`` do **not** satisfy
        the ``grouping_type`` strategy.

        The nodes are the observables (can only be accessed through their integer index).
        """
        # Use upper triangle since adjacency matrix is symmetric and we have an undirected graph
        edges = list(zip(*np.where(np.triu(self.adj_matrix, k=1))))
        # Create complement graph
        if new_rx:
            # node/edge hinting was introduced on version 0.15
            graph = rx.PyGraph(
                node_count_hint=len(self.observables),
                edge_count_hint=len(edges),
            )
        else:
            graph = rx.PyGraph()

        graph.add_nodes_from(self.observables)
        graph.add_edges_from_no_data(edges)

        return graph

    def partition_observables(self) -> list[list]:
        """
        Partition the observables into groups of observables mutually satisfying the binary relation determined
        by ``self.grouping_type``.

        Returns:
            list[list[Operator]]: List of partitions of the Pauli observables made up of mutually (anti-)commuting
            observables.
        """
        if self.graph_colourer != "rlf":
            return self.pauli_partitions_from_graph()
        coloured_binary_paulis = recursive_largest_first(self.binary_observables, self.adj_matrix)

        # Need to convert back from the symplectic representation
        return [
            [binary_to_pauli(pauli_word, wire_map=self._wire_map) for pauli_word in grouping]
            for grouping in coloured_binary_paulis.values()
        ]

    @cached_property
    def _idx_partitions_dict_from_graph(self) -> dict[int, list[int]]:
        """Dictionary containing the solution to the graph colouring problem of ``self.complement_graph``.

        Colours the complement graph using a greedy colouring algorithm and groups indices by colour.

        Uses the ``graph_greedy_color`` function from ``Rustworkx`` to colour the graph defined by
        ``self.complement_graph`` using a specified strategy from ``RX_STRATEGIES``. It then groups the indices
        (nodes) of the graph by their assigned colours.

        Returns:
            dict[int, list[int]]: A dictionary where the keys are colours (integers) and the values are lists
                of indices (nodes) that have been assigned that colour.
        """
        # A dictionary where keys are node indices and the value is the colour
        if new_rx:
            # 'strategy' kwarg was implemented in Rustworkx 0.15
            colouring_dict = rx.graph_greedy_color(
                self.complement_graph, strategy=RX_STRATEGIES[self.graph_colourer]
            )
        else:
            # Default value for <0.15.0 was 'lf'.
            colouring_dict = rx.graph_greedy_color(self.complement_graph)

        # group together indices (values) of the same colour (keys)
        groups = defaultdict(list)
        for idx, colour in sorted(colouring_dict.items()):
            groups[colour].append(idx)

        return groups

    def idx_partitions_from_graph(self, observables_indices=None) -> tuple[tuple[int, ...], ...]:
        """Use ``Rustworkx`` graph colouring algorithms to partition the indices of the Pauli observables into
        tuples containing the indices of observables satisying the binary relation determined by ``self.grouping_type``.

        Args:
            observables_indices (Optional[TensorLike]): A tensor or list of indices associated to each observable.
                This argument is helpful when the observables used in the graph colouring are part of a bigger set of observables.
                Defaults to None. If ``None``, the partitions are made up of the relative indices, i.e. assuming ``self.observables``
                have indices in [0, len(observables)-1].

        Raises:
            IndexError: When ``observables_indices`` is not of the same length as the observables.

        Returns:
            tuple[tuple[int]]: Tuple of tuples containing the indices of the partitioned observables.
        """
        if observables_indices is None:
            return tuple(
                tuple(indices) for indices in self._idx_partitions_dict_from_graph.values()
            )
        if len(observables_indices) != len(self.observables):
            raise ValueError(
                f"The length of the list of indices: {len(observables_indices)} does not "
                f"match the length of the list of observables: {len(self.observables)}. "
            )
        return self._partition_custom_indices(observables_indices)

    def _partition_custom_indices(self, observables_indices) -> list[list]:
        """Compute the indices of the partititions of the observables when these have custom indices.

        TODO: Use this function to calculate custom indices instead of calculating observables first.

        Args:
            observables_indices (Optional[TensorLike]): A tensor or list of indices associated to each observable.
                This argument is helpful when the observables used in the graph colouring are part of a bigger set of observables.
                Defaults to None.

        Returns:
            tuple[tuple[int]]: Tuple of tuples containing the indices of the observables on each partition.
        """
        partition_indices = items_partitions_from_idx_partitions(
            observables_indices, self._idx_partitions_dict_from_graph.values(), return_tuples=True
        )

        return partition_indices

    def pauli_partitions_from_graph(self) -> list[list]:
        """Partition Pauli observables into lists of (anti-)commuting observables
        using ``Rustworkx`` graph colouring algorithms based on binary relation determined by  ``self.grouping_type``.

        Returns:
            list[list[Operator]]]: List of partitions of the Pauli observables made up of mutually (anti-)commuting terms.
        """
        # Get the observables from the indices. itemgetter outperforms list comprehension
        pauli_partitions = items_partitions_from_idx_partitions(
            self.observables, self._idx_partitions_dict_from_graph.values()
        )
        return pauli_partitions


def items_partitions_from_idx_partitions(
    items: Sequence, idx_partitions: Sequence[Sequence[int]], return_tuples: bool = False
) -> Sequence[Sequence]:
    """Get the partitions of the items corresponding to the partitions of the indices.

    Args:
        items (Sequence): A Sequence of items to be partitioned according to the partition of the indices.
        idx_partitions (Sequence[Sequence[int]]): Sequence of sequences containing the indices of the partitioned items.
        return_tuples (bool): Whether to return tuples of tuples or list of lists.
            Useful when dealing with indices or observables.
    Returns:
        Sequence[Sequence]: Sequence of partitions of the items according to the partition of the indices.
    """
    if return_tuples:
        items_partitioned = tuple(
            (
                tuple(itemgetter(*indices)(items))
                if len(indices) > 1
                else (itemgetter(*indices)(items),)
            )
            for indices in idx_partitions
        )
    else:
        items_partitioned = [
            (
                list(itemgetter(*indices)(items))
                if len(indices) > 1
                else [itemgetter(*indices)(items)]
            )
            for indices in idx_partitions
        ]

    return items_partitioned


def _adj_matrix_from_symplectic(symplectic_matrix: np.ndarray, grouping_type: str) -> np.ndarray:
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
        observables[j] do **not** (anti-)commute, as determined by the ``grouping_type``.
    """

    n_qubits = symplectic_matrix.shape[1] // 2

    # Convert symplectic representation to integer format.
    # This is equivalent to the map: {0: I, 1: X, 2:Y, Z:3}
    pauli_matrix_int = 2 * symplectic_matrix[:, :n_qubits] + symplectic_matrix[:, n_qubits:]
    pauli_matrix_int = pauli_matrix_int.astype(np.int8)
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
        observables (list[Operator]): A list of Pauli operators to be partitioned.
        grouping_type (str): The type of binary relation between Pauli observables.
            It can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``. Defaults to ``'qwc'``.
        method (str): The graph colouring heuristic to use in solving minimum clique cover.
            It can be ``'lf'`` (Largest First), ``'rlf'`` (Recursive Largest First), ``'dsatur'`` (Degree of Saturation),
            or ``'gis'`` (Greedy Independent Set). Defaults to ``'lf'``.

    Returns:
        tuple[tuple[int]]: A tuple of tuples where each inner tuple contains the indices of
        observables that are grouped together according to the specified grouping type and
        graph colouring method.

    .. seealso:: `rustworkx.ColoringStrategy <https://www.rustworkx.org/apiref/rustworkx.ColoringStrategy.html#coloringstrategy>`_
        for more information on the ``('lf', 'dsatur', 'gis')`` strategies.

    **Example**

    >>> from pennylane.pauli import compute_partition_indices
    >>> observables = [qml.X(0) @ qml.Z(1), qml.Z(0), qml.X(1)]
    >>> compute_partition_indices(observables, grouping_type="qwc", method="lf")
    ((0,), (1, 2))
    """
    if method != "rlf":

        idx_no_wires = [idx for idx, obs in enumerate(observables) if len(obs.wires) == 0]

        if len(idx_no_wires) == len(observables):
            return (tuple(idx_no_wires),)

        pauli_groupper = PauliGroupingStrategy(
            observables, grouping_type=grouping_type, graph_colourer=method
        )

        return pauli_groupper.idx_partitions_from_graph()
    # 'rlf' method is not compatible with the rx implementation.
    return _compute_partition_indices_rlf(observables, grouping_type=grouping_type)


def _compute_partition_indices_rlf(observables: list, grouping_type: str):
    """Computes the partition indices of a list of observables using a specified grouping type and 'rlf' method.

    This option is much less efficient so should be avoided.
    """

    with qml.QueuingManager.stop_recording():
        obs_groups = group_observables(observables, grouping_type=grouping_type, method="rlf")

    observables = copy(observables)

    indices = []
    available_indices = list(range(len(observables)))
    for partition in obs_groups:
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


def group_observables(
    observables: list["qml.operation.Operator"],
    coefficients: TensorLike | None = None,
    grouping_type: Literal["qwc", "commuting", "anticommuting"] = "qwc",
    method: Literal["lf", "rlf", "dsatur", "gis"] = "lf",
):
    """Partitions a list of observables (Pauli operations and tensor products thereof) into
    groupings according to a binary relation (qubit-wise commuting, fully-commuting, or
    anticommuting).

    Partitions are found by 1) mapping the list of observables to a graph where vertices represent
    observables and edges encode the binary relation, then 2) solving minimum clique cover for the
    graph using graph-colouring heuristic algorithms.

    Args:
        observables (list[Operator]): a list of Pauli word ``Operator`` instances (Pauli
            operation instances and tensor products thereof)
        coefficients (TensorLike): A tensor or list of coefficients. If not specified,
            output ``partitioned_coeffs`` is not returned.
        grouping_type (str): The type of binary relation between Pauli words.
            It can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``.
        method (str): The graph colouring heuristic to use in solving minimum clique cover, which
            can be ``'lf'`` (Largest First), ``'rlf'`` (Recursive Largest First),
            ``'dsatur'`` (Degree of Saturation), or ``'gis'`` (IndependentSet). Defaults to ``'lf'``.

    Returns:
       tuple:

           * list[list[Operator]]: A list of the obtained groupings. Each grouping
             is itself a list of Pauli word ``Operator`` instances.
           * list[TensorLike]: A list of coefficient groupings. Each coefficient
             grouping is itself a tensor or list of the grouping's corresponding coefficients. This is only
             returned if coefficients are specified.

    Raises:
        IndexError: if the input list of coefficients is not of the same length as the input list
            of Pauli words

    .. seealso:: `rustworkx.ColoringStrategy <https://www.rustworkx.org/apiref/rustworkx.ColoringStrategy.html#coloringstrategy>`_
        for more information on the ``('lf', 'dsatur', 'gis')`` strategies.

    **Example**

    >>> from pennylane.pauli import group_observables
    >>> obs = [qml.Y(0), qml.X(0) @ qml.X(1), qml.Z(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'anticommuting', 'lf')
    >>> obs_groupings
    [[Y(0), X(0) @ X(1)], [Z(1)]]
    >>> coeffs_groupings
    [[np.float64(1.43), np.float64(4.21)], [np.float64(0.97)]]
    """

    if coefficients is not None and qml.math.shape(coefficients)[0] != len(observables):
        raise IndexError("The coefficients list must be the same length as the observables list.")

    # Separate observables based on whether they have wires or not.
    no_wires_obs, wires_obs = [], []

    for ob in observables:
        if len(ob.wires) == 0:
            no_wires_obs.append(ob)
        else:
            wires_obs.append(ob)

    # Handle case where all observables have no wires
    if not wires_obs:
        if coefficients is None:
            return [observables]
        return [observables], [coefficients]

    # Initialize PauliGroupingStrategy
    pauli_groupper = PauliGroupingStrategy(
        wires_obs, grouping_type=grouping_type, graph_colourer=method
    )

    partitioned_paulis = pauli_groupper.partition_observables()

    # Add observables without wires back to the first partition
    partitioned_paulis[0].extend(no_wires_obs)

    if coefficients is None:
        return partitioned_paulis

    partitioned_coeffs = _partition_coeffs(partitioned_paulis, observables, coefficients)

    return partitioned_paulis, partitioned_coeffs


def _partition_coeffs(partitioned_paulis, observables, coefficients):
    """Partition the coefficients according to the Pauli word groupings.

    This function is necessary in the cases where the coefficients are not the trivial
    integers range(0, len(observables)). In the latter case, using `compute_partition_indices`
    is recommended.
    """

    partitioned_coeffs = [
        qml.math.cast_like([0] * len(g), coefficients) for g in partitioned_paulis
    ]

    observables = copy(observables)
    # we cannot delete elements from the coefficients tensor, so we
    # use a proxy list memorising the indices for this logic
    coeff_indices = list(range(qml.math.shape(coefficients)[0]))
    for i, partition in enumerate(partitioned_paulis):
        indices = []
        for pauli_word in partition:
            # find index of this pauli word in remaining original observables,
            for ind, observable in enumerate(observables):
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
