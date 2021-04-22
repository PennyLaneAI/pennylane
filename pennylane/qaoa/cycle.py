# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Methods for finding max weighted cycle of weighted directed graphs
"""

from typing import Dict, Tuple
import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix, kron
import pennylane as qml


def edges_to_wires(graph: nx.Graph) -> Dict[Tuple[int], int]:
    r"""Maps the edges of a graph to corresponding wires.

    **Example**

    >>> g = nx.complete_graph(4).to_directed()
    >>> edges_to_wires(g)
    {(0, 1): 0,
     (0, 2): 1,
     (0, 3): 2,
     (1, 0): 3,
     (1, 2): 4,
     (1, 3): 5,
     (2, 0): 6,
     (2, 1): 7,
     (2, 3): 8,
     (3, 0): 9,
     (3, 1): 10,
     (3, 2): 11}

    Args:
        graph (nx.Graph): the graph specifying possible edges

    Returns:
        Dict[Tuple[int], int]: a mapping from graph edges to wires
    """
    return {edge: i for i, edge in enumerate(graph.edges)}


def wires_to_edges(graph: nx.Graph) -> Dict[int, Tuple[int]]:
    r"""Maps the wires of a register of qubits to corresponding edges.

    **Example**

    >>> g = nx.complete_graph(4).to_directed()
    >>> wires_to_edges(g)
    {0: (0, 1),
     1: (0, 2),
     2: (0, 3),
     3: (1, 0),
     4: (1, 2),
     5: (1, 3),
     6: (2, 0),
     7: (2, 1),
     8: (2, 3),
     9: (3, 0),
     10: (3, 1),
     11: (3, 2)}

    Args:
        graph (nx.Graph): the graph specifying possible edges

    Returns:
        Dict[Tuple[int], int]: a mapping from wires to graph edges
    """
    return {i: edge for i, edge in enumerate(graph.edges)}


def cycle_mixer(graph: nx.DiGraph) -> qml.Hamiltonian:
    r"""Calculates the cycle-mixer Hamiltonian.

    The cycle-mixer Hamiltonian preserves the set of valid cycles. This function returns:

    .. math::

        \frac{1}{4}\sum_{(i, j)\in E}
        \left(\sum_{k \in V, k\neq i, k\neq j, (i, k) \in E, (k, j) \in E}
        \left[X_{ij}X_{ik}X_{kj} +Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\right]
        \right)

    Args:
        graph (nx.DiGraph): the graph specifying possible edges

    Returns:
        qml.Hamiltonian: the cycle-mixer Hamiltonian
    """
    hamiltonian = qml.Hamiltonian([], [])

    for edge in graph.edges:
        hamiltonian += _cycle_mixer_on_edge(graph, edge)

    return hamiltonian


def _cycle_mixer_on_edge(graph: nx.DiGraph, edge: Tuple[int]) -> qml.Hamiltonian:
    r"""Calculates the partial cycle-mixer Hamiltonian for a specific edge.

    For an edge :math:`(i, j)`, this function returns:

    .. math::

        \sum_{k \in V, k\neq i, k\neq j, (i, k) \in E, (k, j) \in E}\left[
        X_{ij}X_{ik}X_{kj} + Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\right]

    Args:
        graph (nx.DiGraph): the graph specifying possible edges
        edge (Tuple[Int]): a fixed edge

    Returns:
        qml.Hamiltonian: the partial cycle-mixer Hamiltonian
    """
    coeffs = []
    ops = []

    edges_to_qubits = edges_to_wires(graph)

    for node in graph.nodes:
        out_edge = (edge[0], node)
        in_edge = (node, edge[1])
        if node not in edge and out_edge in graph.edges and in_edge in graph.edges:
            wire = edges_to_qubits[edge]
            out_wire = edges_to_qubits[out_edge]
            in_wire = edges_to_qubits[in_edge]

            t = qml.PauliX(wires=wire) @ qml.PauliX(wires=out_wire) @ qml.PauliX(wires=in_wire)
            ops.append(t)

            t = qml.PauliY(wires=wire) @ qml.PauliY(wires=out_wire) @ qml.PauliX(wires=in_wire)
            ops.append(t)

            t = qml.PauliY(wires=wire) @ qml.PauliX(wires=out_wire) @ qml.PauliY(wires=in_wire)
            ops.append(t)

            t = qml.PauliX(wires=wire) @ qml.PauliY(wires=out_wire) @ qml.PauliY(wires=in_wire)
            ops.append(t)

            coeffs.extend([0.25, 0.25, 0.25, -0.25])

    return qml.Hamiltonian(coeffs, ops)


def matrix(hamiltonian: qml.Hamiltonian, n_wires: int) -> csc_matrix:
    r"""Calculates the matrix representation of an input Hamiltonian in the standard basis.

    Args:
        hamiltonian (qml.Hamiltonian): the input Hamiltonian
        n_wires (int): the total number of wires

    Returns:
        csc_matrix: a sparse matrix representation
    """
    ops_matrices = []

    for op in hamiltonian.ops:
        op_wires = np.array(op.wires.tolist())
        op_list = op.non_identity_obs if isinstance(op, qml.operation.Tensor) else [op]
        op_matrices = []

        for wire in range(n_wires):
            loc = np.argwhere(op_wires == wire).flatten()
            mat = np.eye(2) if len(loc) == 0 else op_list[loc[0]].matrix
            mat = csc_matrix(mat)
            op_matrices.append(mat)

        op_matrix = op_matrices.pop(0)

        for mat in op_matrices:
            op_matrix = kron(op_matrix, mat)

        ops_matrices.append(op_matrix)

    mat = sum(coeff * op_mat for coeff, op_mat in zip(hamiltonian.coeffs, ops_matrices))
    return csc_matrix(mat)
