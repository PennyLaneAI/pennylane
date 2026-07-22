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

import pennylane as qp
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
        edges = list(zip(*np.where(np.triu(self.adj_matrix, k=1)), strict=True))
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

        For the default ``'lf'`` (Largest First) strategy, the colouring is computed directly
        from the bit-packed symplectic representation of the observables, without building
        the graph. This lowers the memory usage from quadratic in the number of observables
        to linear, and, for the ``'qwc'`` grouping type, also avoids all pairwise relation
        checks during the greedy colouring by testing candidates against per-group
        aggregates.

        The remaining strategies use the ``graph_greedy_color`` function from ``Rustworkx``
        to colour the graph defined by ``self.complement_graph`` using a specified strategy
        from ``RX_STRATEGIES``. The indices (nodes) of the graph are then grouped by their
        assigned colours.

        Returns:
            dict[int, list[int]]: A dictionary where the keys are colours (integers) and the values are lists
                of indices (nodes) that have been assigned that colour.
        """
        if self.graph_colourer == "lf":
            x_words, z_words = _pack_symplectic(self.binary_observables)
            colours = _colour_packed_lf(x_words, z_words, self.grouping_type)
            colouring_items = enumerate(colours)
        else:
            # A dictionary where keys are node indices and the value is the colour.
            # The remaining strategies ('dsatur', 'gis') require Rustworkx >= 0.15,
            # which is enforced at construction time.
            colouring_dict = rx.graph_greedy_color(
                self.complement_graph, strategy=RX_STRATEGIES[self.graph_colourer]
            )
            colouring_items = sorted(colouring_dict.items())

        # group together indices (values) of the same colour (keys)
        groups = defaultdict(list)
        for idx, colour in colouring_items:
            groups[int(colour)].append(idx)

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


# Number of uint64 elements allowed in the pairwise temporaries of the blocked
# kernels below (2**23 words = 64 MB), which bounds their peak memory usage.
_BLOCK_WORD_BUDGET = 2**23


def _pack_symplectic(symplectic_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack the X and Z halves of a symplectic binary matrix into 64-bit words.

    Each row of the ``m x 2n`` symplectic matrix is the binary vector ``(x | z)`` of a Pauli
    word. The ``n`` X-bits and ``n`` Z-bits of every row are packed (little-endian within
    each word) into ``ceil(n / 64)`` unsigned 64-bit integers, so that pairwise relations
    between Pauli words can be evaluated with a constant number of bitwise word operations
    per 64 qubits.

    Args:
        symplectic_matrix (np.ndarray): 2D binary matrix. Each row is the symplectic
            representation of a Pauli word.

    Returns:
        tuple[np.ndarray, np.ndarray]: arrays of shape ``(m, ceil(n / 64))`` and dtype
        ``uint64`` holding the packed X-bits and Z-bits, respectively.
    """
    n_rows, two_n = np.shape(symplectic_matrix)
    n_qubits = two_n // 2
    bits = np.asarray(symplectic_matrix, dtype=np.uint8)

    n_bytes = -(-n_qubits // 8)
    n_words = max(1, -(-n_bytes // 8))

    packed = np.zeros((2, n_rows, n_words * 8), dtype=np.uint8)
    if n_qubits:
        packed[0, :, :n_bytes] = np.packbits(bits[:, :n_qubits], axis=1, bitorder="little")
        packed[1, :, :n_bytes] = np.packbits(bits[:, n_qubits:], axis=1, bitorder="little")

    x_words = packed[0].view(np.uint64)
    z_words = packed[1].view(np.uint64)
    return x_words, z_words


def _conflict_block(
    x_blk: np.ndarray,
    z_blk: np.ndarray,
    x_all: np.ndarray,
    z_all: np.ndarray,
    grouping_type: str,
) -> np.ndarray:
    """Complement-graph edges between a block of Pauli words and all Pauli words.

    Evaluates, for every pair ``(i, j)`` with ``i`` in the block, whether the pair violates
    the binary relation selected by ``grouping_type``, directly on the packed symplectic
    representation:

    - two Pauli words qubit-wise commute iff on every qubit occupied by both, their X- and
      Z-bits agree;
    - two Pauli words commute (anticommute) iff their symplectic inner product
      ``x_i . z_j + z_i . x_j`` is 0 (1) modulo 2 [1]. The parity of the total popcount is
      obtained as the popcount parity of the XOR-fold of the per-word products, since
      ``popcount(a ^ b) = popcount(a) + popcount(b) (mod 2)``.

    [1] Andrew Jena (2019). Partitioning Pauli Operators: in Theory and in Practice.
    UWSpace. http://hdl.handle.net/10012/15017

    Args:
        x_blk (np.ndarray): packed X-bits of the block, shape ``(b, n_words)``.
        z_blk (np.ndarray): packed Z-bits of the block, shape ``(b, n_words)``.
        x_all (np.ndarray): packed X-bits of all words, shape ``(m, n_words)``.
        z_all (np.ndarray): packed Z-bits of all words, shape ``(m, n_words)``.
        grouping_type (str): the binary relation used to define partitions of
            the Pauli words, can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or
            ``'anticommuting'``.

    Returns:
        np.ndarray: boolean matrix of shape ``(b, m)`` where ``True`` marks a pair that does
        **not** satisfy the relation (an edge of the complement graph).
    """
    if grouping_type == "qwc":
        # Conflict on a qubit occupied by both words whose X- or Z-bits differ.
        occupied = (x_blk | z_blk)[:, None, :] & (x_all | z_all)[None, :, :]
        conflict = occupied & (
            (x_blk[:, None, :] ^ x_all[None, :, :]) | (z_blk[:, None, :] ^ z_all[None, :, :])
        )
        return np.bitwise_or.reduce(conflict, axis=2).astype(bool)

    fold = np.bitwise_xor.reduce(
        (x_blk[:, None, :] & z_all[None, :, :]) ^ (z_blk[:, None, :] & x_all[None, :, :]),
        axis=2,
    )
    anticommutes = (np.bitwise_count(fold) & 1).astype(bool)
    if grouping_type == "commuting":
        return anticommutes
    return ~anticommutes


def _block_rows(n_cols: int, n_words: int) -> int:
    """Number of block rows keeping the pairwise temporaries within the word budget."""
    return int(max(1, _BLOCK_WORD_BUDGET // max(1, n_cols * n_words)))


def _adj_matrix_from_symplectic(symplectic_matrix: np.ndarray, grouping_type: str) -> np.ndarray:
    """Get adjacency matrix of (anti-)commuting graph based on grouping type.

    This is the adjacency matrix of the complement graph. Based on symplectic representations and inner product of [1].

    The pairwise relations are evaluated on a bit-packed symplectic representation in blocks
    of rows, so the peak memory beyond the output matrix is constant instead of the
    ``O(m^2 n)`` of a dense per-qubit broadcast.

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
    x_words, z_words = _pack_symplectic(symplectic_matrix)
    n_rows, n_words = x_words.shape

    adj_matrix = np.empty((n_rows, n_rows), dtype=bool)
    block = _block_rows(n_rows, n_words)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        adj_matrix[start:stop] = _conflict_block(
            x_words[start:stop], z_words[start:stop], x_words, z_words, grouping_type
        )

    return adj_matrix


def _complement_graph_degrees(
    x_words: np.ndarray, z_words: np.ndarray, grouping_type: str
) -> np.ndarray:
    """Degrees of the complement-graph nodes, computed in blocks with ``O(m)`` memory."""
    n_rows, n_words = x_words.shape
    degrees = np.zeros(n_rows, dtype=np.int64)

    block = _block_rows(n_rows, n_words)
    for start in range(0, n_rows, block):
        stop = min(start + block, n_rows)
        conflicts = _conflict_block(
            x_words[start:stop], z_words[start:stop], x_words, z_words, grouping_type
        )
        degrees[start:stop] = conflicts.sum(axis=1)

    if grouping_type == "anticommuting":
        # A word commutes with itself, so the diagonal is a spurious self-conflict.
        degrees -= 1

    return degrees


def _smallest_missing_colour(blocked: np.ndarray, n_colours: int) -> int:
    """Smallest colour in ``[0, n_colours]`` that does not appear in ``blocked``."""
    if blocked.size == 0:
        return 0
    used = np.zeros(n_colours + 1, dtype=bool)
    used[blocked] = True
    return int(np.nonzero(~used)[0][0])


def _greedy_colour_qwc(x_words: np.ndarray, z_words: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Greedy colouring for the qubit-wise commuting relation using group aggregates.

    Qubit-wise commutativity with every member of a group is equivalent to consistency
    with the union of the members' qubit assignments, because mutually consistent partial
    assignments (qubit -> Pauli letter) remain consistent under unions. Each colour class
    is therefore summarized by the OR-aggregate of its members' packed X- and Z-bits, and
    testing a candidate word against a whole group costs ``O(n / 64)`` regardless of the
    group size. This makes the colouring run in ``O(m * g * n / 64)`` time and ``O(m)``
    memory, with ``g`` the number of groups, without materializing the conflict graph.

    Args:
        x_words (np.ndarray): packed X-bits of the Pauli words.
        z_words (np.ndarray): packed Z-bits of the Pauli words.
        order (np.ndarray): order in which the words are greedily coloured.

    Returns:
        np.ndarray: the colour assigned to each Pauli word (indexed as ``x_words``).
    """
    n_rows, n_words = x_words.shape
    colours = np.empty(n_rows, dtype=np.int64)

    capacity = 16
    group_x = np.zeros((capacity, n_words), dtype=np.uint64)
    group_z = np.zeros((capacity, n_words), dtype=np.uint64)
    n_groups = 0

    for v in order:
        x_v, z_v = x_words[v], z_words[v]
        occupied_v = x_v | z_v

        colour = n_groups
        if n_groups:
            g_x, g_z = group_x[:n_groups], group_z[:n_groups]
            conflict = ((g_x | g_z) & occupied_v) & ((g_x ^ x_v) | (g_z ^ z_v))
            compatible = np.nonzero(~conflict.any(axis=1))[0]
            if compatible.size:
                colour = int(compatible[0])

        if colour == n_groups:
            if n_groups == capacity:
                group_x = np.concatenate([group_x, np.zeros_like(group_x)])
                group_z = np.concatenate([group_z, np.zeros_like(group_z)])
                capacity *= 2
            n_groups += 1

        group_x[colour] |= x_v
        group_z[colour] |= z_v
        colours[v] = colour

    return colours


def _greedy_colour_pairwise(
    x_words: np.ndarray, z_words: np.ndarray, order: np.ndarray, grouping_type: str
) -> np.ndarray:
    """Greedy colouring for the (anti)commuting relations from the packed representation.

    General (anti)commutativity is a parity condition that is not closed under unions, so
    a group cannot be summarized by an aggregate. Instead, each candidate is tested with a
    single vectorized symplectic-parity kernel against all previously coloured words, and
    the smallest colour with no conflicting member is selected. This preserves the
    ``O(m^2)`` pair checks of the graph-based approach but runs them 64 qubits per word
    operation with ``O(m)`` memory instead of building an explicit graph.

    Args:
        x_words (np.ndarray): packed X-bits of the Pauli words.
        z_words (np.ndarray): packed Z-bits of the Pauli words.
        order (np.ndarray): order in which the words are greedily coloured.
        grouping_type (str): either ``'commuting'`` or ``'anticommuting'``.

    Returns:
        np.ndarray: the colour assigned to each Pauli word (indexed as ``x_words``).
    """
    n_rows = x_words.shape[0]
    x_sorted, z_sorted = x_words[order], z_words[order]

    colours_in_order = np.empty(n_rows, dtype=np.int64)
    colours = np.empty(n_rows, dtype=np.int64)
    n_groups = 0

    for t in range(n_rows):
        colour = 0
        if t:
            fold = np.bitwise_xor.reduce(
                (x_sorted[t] & z_sorted[:t]) ^ (z_sorted[t] & x_sorted[:t]), axis=1
            )
            anticommutes = (np.bitwise_count(fold) & 1).astype(bool)
            conflict = anticommutes if grouping_type == "commuting" else ~anticommutes
            colour = _smallest_missing_colour(colours_in_order[:t][conflict], n_groups)

        n_groups = max(n_groups, colour + 1)
        colours_in_order[t] = colour
        colours[order[t]] = colour

    return colours


def _colour_packed_lf(
    x_words: np.ndarray, z_words: np.ndarray, grouping_type: str
) -> np.ndarray:
    """Largest-First greedy colouring of the complement graph, computed directly from the
    packed symplectic representation without materializing the graph.

    Nodes are coloured in order of decreasing complement-graph degree (ties broken by node
    index, matching the stable Degree strategy of ``rustworkx``), each receiving the
    smallest colour not taken by a conflicting, already-coloured node.

    Args:
        x_words (np.ndarray): packed X-bits of the Pauli words.
        z_words (np.ndarray): packed Z-bits of the Pauli words.
        grouping_type (str): the binary relation used to define partitions of
            the Pauli words, can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or
            ``'anticommuting'``.

    Returns:
        np.ndarray: the colour assigned to each Pauli word.
    """
    degrees = _complement_graph_degrees(x_words, z_words, grouping_type)
    order = np.argsort(-degrees, kind="stable")

    if grouping_type == "qwc":
        return _greedy_colour_qwc(x_words, z_words, order)
    return _greedy_colour_pairwise(x_words, z_words, order, grouping_type)


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
    >>> observables = [qp.X(0) @ qp.Z(1), qp.Z(0), qp.X(1)]
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

    with qp.QueuingManager.stop_recording():
        obs_groups = group_observables(observables, grouping_type=grouping_type, method="rlf")

    observables = copy(observables)

    indices = []
    available_indices = list(range(len(observables)))
    for partition in obs_groups:
        indices_this_group = []
        for pauli_word in partition:
            # find index of this pauli word in remaining original observables,
            for ind, observable in enumerate(observables):
                if qp.pauli.are_identical_pauli_words(pauli_word, observable):
                    indices_this_group.append(available_indices[ind])
                    # delete this observable and its index, so it cannot be found again
                    observables.pop(ind)
                    available_indices.pop(ind)
                    break
        indices.append(tuple(indices_this_group))

    return tuple(indices)


def group_observables(
    observables: list["qp.operation.Operator"],
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
    >>> obs = [qp.Y(0), qp.X(0) @ qp.X(1), qp.Z(1)]
    >>> coeffs = [1.43, 4.21, 0.97]
    >>> obs_groupings, coeffs_groupings = group_observables(obs, coeffs, 'anticommuting', 'lf')
    >>> obs_groupings
    [[Y(0), X(0) @ X(1)], [Z(1)]]
    >>> coeffs_groupings
    [[np.float64(1.43), np.float64(4.21)], [np.float64(0.97)]]
    """

    if coefficients is not None and qp.math.shape(coefficients)[0] != len(observables):
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

    partitioned_coeffs = [qp.math.cast_like([0] * len(g), coefficients) for g in partitioned_paulis]

    observables = copy(observables)
    # we cannot delete elements from the coefficients tensor, so we
    # use a proxy list memorising the indices for this logic
    coeff_indices = list(range(qp.math.shape(coefficients)[0]))
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
        partitioned_coeffs[i] = qp.math.take(coefficients, indices, axis=0)

    # make sure the output is of the same format as the input
    # for these two frequent cases
    if isinstance(coefficients, list):
        partitioned_coeffs = [list(p) for p in partitioned_coeffs]

    return partitioned_coeffs
