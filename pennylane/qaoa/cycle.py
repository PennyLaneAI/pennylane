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
import pennylane as qml
import numpy as np


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


def edge_weight(graph: nx.DiGraph) -> qml.Hamiltonian:
    r"""Calculates the product of edge weights Hamiltonian.

    The product of weights of a subset of edges in a graph is given by

    .. math:: \prod_{(i, j) \in E} x_{ij} c_{ij}

    where :math:`E` are the edges of the graph, :math:`x_{ij}` is a binary number that selects
    whether to include the edge :math:`(i, j)` and :math:`c_{ij}` is the corresponding edge weight.

    The product of weights is maximized by equivalently considering

    .. math:: \sum_{(i, j) \in E} x_{ij}\log c_{ij},

    assuming :math:`c_{ij} > 0`.

    This can be restated as a minimization over the following qubit Hamiltonian:

    .. math::

        \sum_{(i, j) \in E} Z_{ij}\log c_{ij}.

    where :math:`Z_{ij}` is a qubit Pauli-Z matrix acting upon the qubit specified by the pair
    :math:`(i, j)`.

    Args:
        graph (nx.DiGraph): the graph specifying possible edges

    Returns:
        qml.Hamiltonian: the product of edge weights Hamiltonian

    Raises:
        ValueError: if the graph contains parallel edges or self-loops
        KeyError: if one or more edges do not contain weight data
    """
    edges_to_qubits = edges_to_wires(graph)
    coeffs = []
    ops = []

    edges_data = graph.edges(data=True)
    edges = [edge[:2] for edge in graph.edges]

    if len(edges) != len(set(edges)):
        raise ValueError("Graph contains parallel edges")

    for edge_data in edges_data:
        edge = edge_data[:2]

        if edge[0] == edge[1]:
            raise ValueError("Graph contains self-loops")

        try:
            weight = edge_data[2]["weight"]
        except KeyError as e:
            raise ValueError(f"Edge {edge} does not contain weight data") from e

        coeffs.append(np.log(weight))
        wires = (edges_to_qubits[edge],)
        ops.append(qml.PauliZ(wires=wires))

    return qml.Hamiltonian(coeffs, ops)
