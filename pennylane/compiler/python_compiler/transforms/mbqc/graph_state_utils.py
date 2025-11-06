# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for graph state representations"""

import math
from collections.abc import Generator, Sequence
from typing import TypeAlias

from pennylane.exceptions import CompileError

DenselyPackedAdjMatrix: TypeAlias = Sequence[int] | Sequence[bool]


_MBQC_GATE_SET = {"Hadamard", "S", "RZ", "RotXZX", "CNOT"}
_SINGLE_QUBIT_AUX_WIRE_NUM = 4
_CNOT_AUX_WIRE_NUM = 13


def get_num_aux_wires(gate_name: str) -> int:
    """
    Return the number of auxiliary wires required for gates from the MBQC gate set.
    The number of auxiliary qubits for a single qubit gate is 4, while it is 13 for a
    CNOT gate.

    Args:
        gate_name (str): The name of a gate.

    Returns:
        The number of auxiliary wires.
    """
    if gate_name == "CNOT":
        return _CNOT_AUX_WIRE_NUM
    if gate_name in _MBQC_GATE_SET:
        return _SINGLE_QUBIT_AUX_WIRE_NUM
    raise ValueError(f"{gate_name} is not supported in the MBQC formalism.")


def get_graph_state_edges(gate_name: str) -> list[tuple[int, int]]:
    """
    Return a list of edges information in the graph state of a gate.

    -  The connectivity of the target qubits in the register and auxiliary qubits for a single-qubit gate is:

        tgt --  0  --  1  --  2  --  3

        Note that the target qubit is not in the adjacency matrix and the connectivity
        of the auxiliary qubits is:
        edges_in_adj_matrix = [
        (0, 1),
        (1, 2),
        (2, 3),
        ]

        Wire 1 in the above isn't the target wire described in the Fig.2 of [`arXiv:quant-ph/0301052 <https://arxiv.org/abs/quant-ph/0301052>`_],
        1 in the above maps to 3 in the figure.

    - The connectivity of the ctrl/target qubits in the register and auxiliary qubits for a CNOT gate is:

        ctl --  0  --  1  --  2  --  3  --  4  -- 5
                            |
                            6
                            |
        tgt --  7  --  8  --  9  -- 10  -- 11  -- 12

        Note that both ctrl and target qubits are not in the adjacency matrix and the connectivity
        of the auxiliary qubits is:
        edges_in_adj_matrix = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (2, 6),
            (7, 8),
            (6, 9),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ]

        This graph is labelled based on the rows and columns of the adjacent matrix, but maps on to the graph described in
        the Fig.2 of [`arXiv:quant-ph/0301052 <https://arxiv.org/abs/quant-ph/0301052>`_], where wire 1 is the control and
        wire 9 is the target.

        Args:
            gate_name (str): The name of a gate.

        Returns:
            A list of edges information in the graph state for the given gate.
    """

    if gate_name == "CNOT":
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (2, 6),
            (7, 8),
            (6, 9),
            (8, 9),
            (9, 10),
            (10, 11),
            (11, 12),
        ]
    if gate_name in _MBQC_GATE_SET:
        return [
            (0, 1),
            (1, 2),
            (2, 3),
        ]
    raise ValueError(f"{gate_name} is not supported in the MBQC formalism.")


def n_vertices_from_packed_adj_matrix(adj_matrix: DenselyPackedAdjMatrix) -> int:
    """Returns the number of vertices in the graph represented by the given densely packed adjacency
    matrix.

    Args:
        adj_matrix (DenselyPackedAdjMatrix): The densely packed adjacency matrix, given as a
            sequence of bools or ints. See the note in the module documentation for a description of
            this format.

    Raises:
        CompileError: If the number of elements in `adj_matrix` is not compatible with the number of
            elements in the lower-triangular part of a square matrix, excluding the elements along
            the diagonal.

    Returns:
        int: The number of vertices in the graph.

    Example:
        >>> _n_vertices_from_packed_adj_matrix([1, 1, 0, 0, 1, 1])
        4
    """
    assert isinstance(
        adj_matrix, Sequence
    ), f"Expected `adj_matrix` to be a sequence, but got {type(adj_matrix).__name__}"

    m = len(adj_matrix)

    # The formula to compute the number of vertices, N, in the graph from the number elements in the
    # densely packed adjacency matrix, m, is
    #   N = (1 + sqrt(1 + 8m)) / 2
    # To avoid floating-point errors in the sqrt function, we break it down into integer-arithmetic
    # operations and ensure that the solution is one where N is mathematically a true integer.

    discriminant = 1 + 8 * m
    sqrt_discriminant = math.isqrt(discriminant)

    # Check if it's a perfect square
    if sqrt_discriminant * sqrt_discriminant != discriminant:
        raise CompileError(
            f"The number of elements in the densely packed adjacency matrix is {m}, which does not "
            f"correspond to an integer number of graph vertices"
        )

    # The numerator, 1 + sqrt(1 + 8m), must be even for the result to be an integer. The quantity
    # sqrt(1 + 8m) will always be odd if it's a perfect square, so the quantity (1 + sqrt(1 + 8m))
    # will always be even. We can therefore safely divide (using integer division).
    return (1 + sqrt_discriminant) // 2


def edge_iter(adj_matrix: DenselyPackedAdjMatrix) -> Generator[tuple[int, int], None, None]:
    """Generate an iterator over the edges in a graph represented by the given densely packed
    adjacency matrix.

    Args:
        adj_matrix (DenselyPackedAdjMatrix): The densely packed adjacency matrix, given as a
            sequence of bools or ints. See the note in the module documentation for a description of
            this format.

    Yields:
        tuple[int, int]: The next edge in the graph, represented as the pair of vertices labelled
            according to their indices in the adjacency matrix.

    Example:
        >>> for edge in _edge_iter([1, 1, 0, 0, 1, 1]):
        ...     print(edge)
        (0, 1)
        (0, 2)
        (1, 3)
        (2, 3)
    """
    # Calling `_n_vertices_from_packed_adj_matrix()` asserts that the input `adj_matrix` is in the
    # correct format and is valid.
    n_vertices_from_packed_adj_matrix(adj_matrix)

    j = 1
    k = 0
    for entry in adj_matrix:
        if entry:
            yield (k, j)
        k += 1
        if k == j:
            k = 0
            j += 1


def _adj_matrix_generation_helper(
    num_vertices: int, edges_in_adj_matrix: list[tuple[int, int]]
) -> list:
    """Helper function to generate an adjacency matrix to represent the connectivity of auxiliary qubits in
    a graph state for a gate operation with the number of vertices and edges information.
    Note that the adjacency matrix here means the lower triangular part of the full adjacency matrix.
    It can be represented as below and `x` marks here denotes the matrix diagonal.
    x
    + x
    + + x
    .
    ........
    .
    + + + + + x

    Args:
      num_vertices (int) : Number of vertices in the adjacency matrix.
      edges_in_adj_matrix (list[tuple[int, int]]): List of edges in the adjacency matrix.

    Return:
      An adjacency matrix represents the connectivity of vertices.
    """
    adj_matrix_length = num_vertices * (num_vertices - 1) // 2
    adj_matrix = [0] * adj_matrix_length
    for edge in edges_in_adj_matrix:
        col, row = edge
        n = col + (row - 1) * row // 2
        adj_matrix[n] = 1

    return adj_matrix


def generate_adj_matrix(op_name: str) -> list:
    """Generate an adjacency matrix represents the connectivity of auxiliary qubits in a
    graph state for a gate operation.
    Args:
        op_name (str): The gate name. Note that only a gate in the MBQC gate set is supported.
    Returns:
        An adjacent matrix represents the connectivity of auxiliary qubits.
    """
    num_aux_wires = get_num_aux_wires(op_name)
    edges_list = get_graph_state_edges(op_name)
    return _adj_matrix_generation_helper(num_aux_wires, edges_list)
