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
# pylint: disable=unnecessary-comprehension, unnecessary-lambda-assignment
import itertools
from typing import (
    Dict,
    Tuple,
    Iterable,
    List,
    Union,
)

import networkx as nx
import rustworkx as rx

import numpy as np
import pennylane as qml


def edges_to_wires(graph: Union[nx.Graph, rx.PyGraph, rx.PyDiGraph]) -> Dict[Tuple, int]:
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

    >>> g = rx.generators.directed_mesh_graph(4, [0,1,2,3])
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
        graph (nx.Graph or rx.PyGraph or rx.PyDiGraph): the graph specifying possible edges

    Returns:
        Dict[Tuple, int]: a mapping from graph edges to wires
    """
    if isinstance(graph, nx.Graph):
        return {edge: i for i, edge in enumerate(graph.edges)}
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        gnodes = graph.nodes()
        return {
            (gnodes.index(e[0]), gnodes.index(e[1])): i
            for i, e in enumerate(sorted(graph.edge_list()))
        }
    raise ValueError(
        f"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph, got {type(graph).__name__}"
    )


def wires_to_edges(graph: Union[nx.Graph, rx.PyGraph, rx.PyDiGraph]) -> Dict[int, Tuple]:
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

    >>> g = rx.generators.directed_mesh_graph(4, [0,1,2,3])
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
        graph (nx.Graph or rx.PyGraph or rx.PyDiGraph): the graph specifying possible edges

    Returns:
        Dict[Tuple, int]: a mapping from wires to graph edges
    """
    if isinstance(graph, nx.Graph):
        return {i: edge for i, edge in enumerate(graph.edges)}
    if isinstance(graph, (rx.PyGraph, rx.PyDiGraph)):
        gnodes = graph.nodes()
        return {
            i: (gnodes.index(e[0]), gnodes.index(e[1]))
            for i, e in enumerate(sorted(graph.edge_list()))
        }
    raise ValueError(
        f"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph, got {type(graph).__name__}"
    )


def cycle_mixer(graph: Union[nx.DiGraph, rx.PyDiGraph]) -> qml.operation.Operator:
    r"""Calculates the cycle-mixer Hamiltonian.

    Following methods outlined `here <https://arxiv.org/abs/1709.03489>`__, the
    cycle-mixer Hamiltonian preserves the set of valid cycles:

    .. math::
        \frac{1}{4}\sum_{(i, j)\in E}
        \left(\sum_{k \in V, k\neq i, k\neq j, (i, k) \in E, (k, j) \in E}
        \left[X_{ij}X_{ik}X_{kj} +Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\right]
        \right)

    where :math:`E` are the edges of the directed graph. A valid cycle is defined as a subset of
    edges in :math:`E` such that all of the graph's nodes :math:`V` have zero net flow (see the
    :func:`~.net_flow_constraint` function).

    **Example**

    >>> import networkx as nx
    >>> g = nx.complete_graph(3).to_directed()
    >>> h_m = cycle_mixer(g)
    >>> print(h_m)
      (-0.25) [X0 Y1 Y5]
    + (-0.25) [X1 Y0 Y3]
    + (-0.25) [X2 Y3 Y4]
    + (-0.25) [X3 Y2 Y1]
    + (-0.25) [X4 Y5 Y2]
    + (-0.25) [X5 Y4 Y0]
    + (0.25) [X0 X1 X5]
    + (0.25) [Y0 Y1 X5]
    + (0.25) [Y0 X1 Y5]
    + (0.25) [X1 X0 X3]
    + (0.25) [Y1 Y0 X3]
    + (0.25) [Y1 X0 Y3]
    + (0.25) [X2 X3 X4]
    + (0.25) [Y2 Y3 X4]
    + (0.25) [Y2 X3 Y4]
    + (0.25) [X3 X2 X1]
    + (0.25) [Y3 Y2 X1]
    + (0.25) [Y3 X2 Y1]
    + (0.25) [X4 X5 X2]
    + (0.25) [Y4 Y5 X2]
    + (0.25) [Y4 X5 Y2]
    + (0.25) [X5 X4 X0]
    + (0.25) [Y5 Y4 X0]
    + (0.25) [Y5 X4 Y0]

    >>> import rustworkx as rx
    >>> g = rx.generators.directed_mesh_graph(3, [0,1,2])
    >>> h_m = cycle_mixer(g)
    >>> print(h_m)
      (-0.25) [X0 Y1 Y5]
    + (-0.25) [X1 Y0 Y3]
    + (-0.25) [X2 Y3 Y4]
    + (-0.25) [X3 Y2 Y1]
    + (-0.25) [X4 Y5 Y2]
    + (-0.25) [X5 Y4 Y0]
    + (0.25) [X0 X1 X5]
    + (0.25) [Y0 Y1 X5]
    + (0.25) [Y0 X1 Y5]
    + (0.25) [X1 X0 X3]
    + (0.25) [Y1 Y0 X3]
    + (0.25) [Y1 X0 Y3]
    + (0.25) [X2 X3 X4]
    + (0.25) [Y2 Y3 X4]
    + (0.25) [Y2 X3 Y4]
    + (0.25) [X3 X2 X1]
    + (0.25) [Y3 Y2 X1]
    + (0.25) [Y3 X2 Y1]
    + (0.25) [X4 X5 X2]
    + (0.25) [Y4 Y5 X2]
    + (0.25) [Y4 X5 Y2]
    + (0.25) [X5 X4 X0]
    + (0.25) [Y5 Y4 X0]
    + (0.25) [Y5 X4 Y0]

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges

    Returns:
        qml.Hamiltonian: the cycle-mixer Hamiltonian
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    hamiltonian = qml.Hamiltonian([], [])
    graph_edges = sorted(graph.edge_list()) if isinstance(graph, rx.PyDiGraph) else graph.edges

    for edge in graph_edges:
        hamiltonian += _partial_cycle_mixer(graph, edge)

    return hamiltonian


def _partial_cycle_mixer(
    graph: Union[nx.DiGraph, rx.PyDiGraph], edge: Tuple
) -> qml.operation.Operator:
    r"""Calculates the partial cycle-mixer Hamiltonian for a specific edge.

    For an edge :math:`(i, j)`, this function returns:

    .. math::

        \sum_{k \in V, k\neq i, k\neq j, (i, k) \in E, (k, j) \in E}\left[
        X_{ij}X_{ik}X_{kj} + Y_{ij}Y_{ik}X_{kj} + Y_{ij}X_{ik}Y_{kj} - X_{ij}Y_{ik}Y_{kj}\right]

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges
        edge (tuple): a fixed edge

    Returns:
        qml.Hamiltonian: the partial cycle-mixer Hamiltonian
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    coeffs = []
    ops = []

    is_rx = isinstance(graph, rx.PyDiGraph)
    edges_to_qubits = edges_to_wires(graph)
    graph_nodes = graph.node_indexes() if is_rx else graph.nodes
    graph_edges = sorted(graph.edge_list()) if is_rx else graph.edges

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalues = lambda T: (graph.nodes().index(T[0]), graph.nodes().index(T[1])) if is_rx else T

    for node in graph_nodes:
        out_edge = (edge[0], node)
        in_edge = (node, edge[1])
        if node not in edge and out_edge in graph_edges and in_edge in graph_edges:
            wire = edges_to_qubits[get_nvalues(edge)]
            out_wire = edges_to_qubits[get_nvalues(out_edge)]
            in_wire = edges_to_qubits[get_nvalues(in_edge)]

            t = qml.X(wire) @ qml.X(out_wire) @ qml.X(in_wire)
            ops.append(t)

            t = qml.Y(wire) @ qml.Y(out_wire) @ qml.X(in_wire)
            ops.append(t)

            t = qml.Y(wire) @ qml.X(out_wire) @ qml.Y(in_wire)
            ops.append(t)

            t = qml.X(wire) @ qml.Y(out_wire) @ qml.Y(in_wire)
            ops.append(t)

            coeffs.extend([0.25, 0.25, 0.25, -0.25])

    return qml.Hamiltonian(coeffs, ops)


def loss_hamiltonian(graph: Union[nx.Graph, rx.PyGraph, rx.PyDiGraph]) -> qml.operation.Operator:
    r"""Calculates the loss Hamiltonian for the maximum-weighted cycle problem.

    We consider the problem of selecting a cycle from a graph that has the greatest product of edge
    weights, as outlined `here <https://1qbit.com/whitepaper/arbitrage/>`__. The product of weights
    of a subset of edges in a graph is given by

    .. math:: P = \prod_{(i, j) \in E} [(c_{ij} - 1)x_{ij} + 1]

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
    >>> h
    (
        -0.6931471805599453 * Z(0)
      + 0.0 * Z(1)
      + 0.4054651081081644 * Z(2)
      + 0.6931471805599453 * Z(3)
      + 0.9162907318741551 * Z(4)
      + 1.0986122886681098 * Z(5)
    )

    >>> import rustworkx as rx
    >>> g = rx.generators.directed_mesh_graph(3, [0, 1, 2])
    >>> edge_weight_data = {edge: (i + 1) * 0.5 for i, edge in enumerate(sorted(g.edge_list()))}
    >>> for k, v in edge_weight_data.items():
            g.update_edge(k[0], k[1], {"weight": v})
    >>> h = loss_hamiltonian(g)
    >>> print(h)
    (
        -0.6931471805599453 * Z(0)
      + 0.0 * Z(1)
      + 0.4054651081081644 * Z(2)
      + 0.6931471805599453 * Z(3)
      + 0.9162907318741551 * Z(4)
      + 1.0986122886681098 * Z(5)
    )

    Args:
        graph (nx.Graph or rx.PyGraph or rx.PyDiGraph): the graph specifying possible edges

    Returns:
        qml.Hamiltonian: the loss Hamiltonian

    Raises:
        ValueError: if the graph contains self-loops
        KeyError: if one or more edges do not contain weight data
    """
    if not isinstance(graph, (nx.Graph, rx.PyGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    edges_to_qubits = edges_to_wires(graph)

    coeffs = []
    ops = []

    is_rx = isinstance(graph, (rx.PyGraph, rx.PyDiGraph))
    edges_data = sorted(graph.weighted_edge_list()) if is_rx else graph.edges(data=True)

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalues = lambda T: (graph.nodes().index(T[0]), graph.nodes().index(T[1])) if is_rx else T

    for edge_data in edges_data:
        edge = edge_data[:2]

        if edge[0] == edge[1]:
            raise ValueError("Graph contains self-loops")

        try:
            weight = edge_data[2]["weight"]
        except KeyError as e:
            raise KeyError(f"Edge {edge} does not contain weight data") from e
        except TypeError as e:
            raise TypeError(f"Edge {edge} does not contain weight data") from e

        coeffs.append(np.log(weight))
        ops.append(qml.Z(edges_to_qubits[get_nvalues(edge)]))

    H = qml.Hamiltonian(coeffs, ops)
    # store the valuable information that all observables are in one commuting group
    H.grouping_indices = [list(range(len(H.ops)))]

    return H


def _square_hamiltonian_terms(
    coeffs: Iterable[float], ops: Iterable[qml.operation.Observable]
) -> Tuple[List[float], List[qml.operation.Observable]]:
    """Calculates the coefficients and observables that compose the squared Hamiltonian.

    Args:
        coeffs (Iterable[float]): coeffients of the input Hamiltonian
        ops (Iterable[qml.operation.Observable]): observables of the input Hamiltonian

    Returns:
        Tuple[List[float], List[qml.operation.Observable]]: The list of coefficients and list of observables
        of the squared Hamiltonian.
    """
    squared_coeffs, squared_ops = [], []
    pairs = [(coeff, op) for coeff, op in zip(coeffs, ops)]
    products = itertools.product(pairs, repeat=2)

    for (coeff1, op1), (coeff2, op2) in products:
        squared_coeffs.append(coeff1 * coeff2)

        if isinstance(op1, qml.Identity):
            squared_ops.append(op2)
        elif isinstance(op2, qml.Identity):
            squared_ops.append(op1)
        # pylint: disable=unidiomatic-typecheck
        elif op1.wires == op2.wires and isinstance(op1, type(op2)):
            squared_ops.append(qml.Identity(0))
        elif op2.wires[0] < op1.wires[0]:
            squared_ops.append(op2 @ op1)
        else:
            squared_ops.append(op1 @ op2)

    return squared_coeffs, squared_ops


def out_flow_constraint(graph: Union[nx.DiGraph, rx.PyDiGraph]) -> qml.operation.Operator:
    r"""Calculates the `out flow constraint <https://1qbit.com/whitepaper/arbitrage/>`__
    Hamiltonian for the maximum-weighted cycle problem.

    Given a subset of edges in a directed graph, the out-flow constraint imposes that at most one
    edge can leave any given node, i.e., for all :math:`i`:

    .. math:: \sum_{j,(i,j)\in E}x_{ij} \leq 1,

    where :math:`E` are the edges of the graph and :math:`x_{ij}` is a binary number that selects
    whether to include the edge :math:`(i, j)`.

    A set of edges satisfies the out-flow constraint whenever the following Hamiltonian is minimized:

    .. math::

        \sum_{i\in V}\left(d_{i}^{out}(d_{i}^{out} - 2)\mathbb{I}
        - 2(d_{i}^{out}-1)\sum_{j,(i,j)\in E}\hat{Z}_{ij} +
        \left( \sum_{j,(i,j)\in E}\hat{Z}_{ij} \right)^{2}\right)


    where :math:`V` are the graph vertices, :math:`d_{i}^{\rm out}` is the outdegree of node
    :math:`i`, and :math:`Z_{ij}` is a qubit Pauli-Z matrix acting
    upon the qubit specified by the pair :math:`(i, j)`. Mapping from edges to wires can be achieved
    using :func:`~.edges_to_wires`.

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges

    Returns:
        qml.Hamiltonian: the out flow constraint Hamiltonian

    Raises:
        ValueError: if the input graph is not directed
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    if isinstance(graph, (nx.DiGraph, rx.PyDiGraph)) and not hasattr(graph, "out_edges"):
        raise ValueError("Input graph must be directed")

    hamiltonian = qml.Hamiltonian([], [])
    graph_nodes = graph.node_indexes() if isinstance(graph, rx.PyDiGraph) else graph.nodes

    for node in graph_nodes:
        hamiltonian += _inner_out_flow_constraint_hamiltonian(graph, node)

    return hamiltonian


def net_flow_constraint(graph: Union[nx.DiGraph, rx.PyDiGraph]) -> qml.operation.Operator:
    r"""Calculates the `net flow constraint <https://doi.org/10.1080/0020739X.2010.526248>`__
    Hamiltonian for the maximum-weighted cycle problem.

    Given a subset of edges in a directed graph, the net-flow constraint imposes that the number of
    edges leaving any given node is equal to the number of edges entering the node, i.e.,

    .. math:: \sum_{j, (i, j) \in E} x_{ij} = \sum_{j, (j, i) \in E} x_{ji},

    for all nodes :math:`i`, where :math:`E` are the edges of the graph and :math:`x_{ij}` is a
    binary number that selects whether to include the edge :math:`(i, j)`.

    A set of edges has zero net flow whenever the following Hamiltonian is minimized:

    .. math::

        \sum_{i \in V} \left((d_{i}^{\rm out} - d_{i}^{\rm in})\mathbb{I} -
        \sum_{j, (i, j) \in E} Z_{ij} + \sum_{j, (j, i) \in E} Z_{ji} \right)^{2},

    where :math:`V` are the graph vertices, :math:`d_{i}^{\rm out}` and :math:`d_{i}^{\rm in}` are
    the outdegree and indegree, respectively, of node :math:`i` and :math:`Z_{ij}` is a qubit
    Pauli-Z matrix acting upon the wire specified by the pair :math:`(i, j)`. Mapping from edges to
    wires can be achieved using :func:`~.edges_to_wires`.


    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges

    Returns:
        qml.Hamiltonian: the net-flow constraint Hamiltonian

    Raises:
        ValueError: if the input graph is not directed
    """
    if isinstance(graph, (nx.DiGraph, rx.PyDiGraph)) and (
        not hasattr(graph, "in_edges") or not hasattr(graph, "out_edges")
    ):
        raise ValueError("Input graph must be directed")

    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    hamiltonian = qml.Hamiltonian([], [])
    graph_nodes = graph.node_indexes() if isinstance(graph, rx.PyDiGraph) else graph.nodes

    for node in graph_nodes:
        hamiltonian += _inner_net_flow_constraint_hamiltonian(graph, node)

    return hamiltonian


def _inner_out_flow_constraint_hamiltonian(
    graph: Union[nx.DiGraph, rx.PyDiGraph], node: int
) -> qml.operation.Operator:
    r"""Calculates the inner portion of the Hamiltonian in :func:`out_flow_constraint`.
    For a given :math:`i`, this function returns:

    .. math::

        d_{i}^{out}(d_{i}^{out} - 2)\mathbb{I}
        - 2(d_{i}^{out}-1)\sum_{j,(i,j)\in E}\hat{Z}_{ij} +
        ( \sum_{j,(i,j)\in E}\hat{Z}_{ij} )^{2}

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges
        node: a fixed node

    Returns:
        qml.Hamiltonian: The inner part of the out-flow constraint Hamiltonian.
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    coeffs = []
    ops = []

    is_rx = isinstance(graph, rx.PyDiGraph)

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalues = lambda T: (graph.nodes().index(T[0]), graph.nodes().index(T[1])) if is_rx else T

    edges_to_qubits = edges_to_wires(graph)
    out_edges = graph.out_edges(node)
    d = len(out_edges)

    # To ensure the out_edges method in both RX and NX returns
    # the list of edges in the same order, we sort results.
    if is_rx:
        out_edges = sorted(out_edges)

    for edge in out_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wire = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(1)
        ops.append(qml.Z(wire))

    coeffs, ops = _square_hamiltonian_terms(coeffs, ops)

    for edge in out_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wire = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(-2 * (d - 1))
        ops.append(qml.Z(wire))

    coeffs.append(d * (d - 2))
    ops.append(qml.Identity(0))

    H = qml.Hamiltonian(coeffs, ops)
    H.simplify()
    # store the valuable information that all observables are in one commuting group
    H.grouping_indices = [list(range(len(H.ops)))]

    return H


def _inner_net_flow_constraint_hamiltonian(
    graph: Union[nx.DiGraph, rx.PyDiGraph], node: int
) -> qml.operation.Operator:
    r"""Calculates the squared inner portion of the Hamiltonian in :func:`net_flow_constraint`.


    For a given :math:`i`, this function returns:

    .. math::

        \left((d_{i}^{\rm out} - d_{i}^{\rm in})\mathbb{I} -
        \sum_{j, (i, j) \in E} Z_{ij} + \sum_{j, (j, i) \in E} Z_{ji} \right)^{2}.

    Args:
        graph (nx.DiGraph or rx.PyDiGraph): the directed graph specifying possible edges
        node: a fixed node

    Returns:
        qml.Hamiltonian: The inner part of the net-flow constraint Hamiltonian.
    """
    if not isinstance(graph, (nx.DiGraph, rx.PyDiGraph)):
        raise ValueError(
            f"Input graph must be a nx.DiGraph or rx.PyDiGraph, got {type(graph).__name__}"
        )

    edges_to_qubits = edges_to_wires(graph)

    coeffs = []
    ops = []

    is_rx = isinstance(graph, rx.PyDiGraph)

    out_edges = graph.out_edges(node)
    in_edges = graph.in_edges(node)

    # To ensure out_edges and in_edges methods in both RX and NX return
    # the lists of edges in the same order, we sort results.
    if is_rx:
        out_edges = sorted(out_edges)
        in_edges = sorted(in_edges)

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalues = lambda T: (graph.nodes().index(T[0]), graph.nodes().index(T[1])) if is_rx else T

    coeffs.append(len(out_edges) - len(in_edges))
    ops.append(qml.Identity(0))

    for edge in out_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wires = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(-1)
        ops.append(qml.Z(wires))

    for edge in in_edges:
        if len(edge) > 2:
            edge = tuple(edge[:2])
        wires = (edges_to_qubits[get_nvalues(edge)],)
        coeffs.append(1)
        ops.append(qml.Z(wires))

    coeffs, ops = _square_hamiltonian_terms(coeffs, ops)
    H = qml.Hamiltonian(coeffs, ops)
    H = H.simplify()
    # store the valuable information that all observables are in one commuting group
    H.grouping_indices = [list(range(len(H.ops)))]
    return H
