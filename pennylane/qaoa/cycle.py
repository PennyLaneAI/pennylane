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
Functionality for finding the maximum weighted cycle of directed graphs.
"""

from typing import Dict, Tuple
import networkx as nx
import numpy as np
import pennylane as qml


def edges_to_wires(graph: nx.Graph) -> Dict[Tuple, int]:
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
        Dict[Tuple, int]: a mapping from graph edges to wires
    """
    return {edge: i for i, edge in enumerate(graph.edges)}


def wires_to_edges(graph: nx.Graph) -> Dict[int, Tuple]:
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
        Dict[Tuple, int]: a mapping from wires to graph edges
    """
    return {i: edge for i, edge in enumerate(graph.edges)}


def loss_hamiltonian(graph: nx.Graph) -> qml.Hamiltonian:
    r"""Calculates the loss Hamiltonian for the maximum-weighted cycle problem.

    We consider the problem of selecting a cycle from a graph that has the greatest product of edge
    weights, as outlined `here <https://1qbit.com/whitepaper/arbitrage/>`__. The product of weights
    of a subset of edges in a graph is given by

    .. math:: P = \prod_{(i, j) \in E} x_{ij} c_{ij}

    where :math:`E` are the edges of the graph, :math:`x_{ij}` is a binary number that selects
    whether to include the edge :math:`(i, j)` and :math:`c_{ij}` is the corresponding edge weight.
    Our objective is to maximimize :math:`P`, subject to selecting the :math:`x_{ij}` so that
    our subset of edges composes a cycle.

    The product of edge weights is maximized by equivalently considering

    .. math:: \sum_{(i, j) \in E} x_{ij}\log c_{ij},

    assuming :math:`c_{ij} > 0`.

    This can be restated as a minimization of the expectation value of the following qubit
    Hamiltonian:

    .. math::

        H = \sum_{(i, j) \in E} Z_{ij}\log c_{ij}.

    where :math:`Z_{ij}` is a qubit Pauli-Z matrix acting upon the wire specified by the edge
    :math:`(i, j)`. Mapping from edges to wires can be achieved using :func:`~.edges_to_wires`.

    .. note::
        The expectation value of the returned Hamiltonian :math:`H` is not equal to :math:`P`, but
        minimizing the expectation value of :math:`H` is equivalent to maximizing :math:`P`.

        Also note that the returned Hamiltonian does not impose that the selected set of edges is
        a cycle. This constraint can be enforced using a penalty term or by selecting a QAOA
        mixer Hamiltonian that only transitions between states that correspond to cycles.

    **Example**

    >>> import networkx as nx
    >>> g = nx.complete_graph(3).to_directed()
    >>> edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(g.edges)}
    >>> for k, v in edge_weight_data.items():
            g[k[0]][k[1]]["weight"] = v
    >>> h = loss_hamiltonian(g)
    >>> print(h)
      (-0.6931471805599453) [Z0]
    + (0.0) [Z1]
    + (0.4054651081081644) [Z2]
    + (0.6931471805599453) [Z3]
    + (0.9162907318741551) [Z4]
    + (1.0986122886681098) [Z5]

    Args:
        graph (nx.Graph): the graph specifying possible edges

    Returns:
        qml.Hamiltonian: the loss Hamiltonian

    Raises:
        ValueError: if the graph contains self-loops
        KeyError: if one or more edges do not contain weight data
    """
    edges_to_qubits = edges_to_wires(graph)
    coeffs = []
    ops = []

    edges_data = graph.edges(data=True)

    for edge_data in edges_data:
        edge = edge_data[:2]

        if edge[0] == edge[1]:
            raise ValueError("Graph contains self-loops")

        try:
            weight = edge_data[2]["weight"]
        except KeyError as e:
            raise KeyError(f"Edge {edge} does not contain weight data") from e

        coeffs.append(np.log(weight))
        ops.append(qml.PauliZ(wires=edges_to_qubits[edge]))

    return qml.Hamiltonian(coeffs, ops)
