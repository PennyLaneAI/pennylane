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
r"""
Methods for constructing QAOA mixer Hamiltonians.
"""
# pylint: disable=unnecessary-lambda-assignment
import itertools
import functools
from typing import Iterable, Union

import networkx as nx
import rustworkx as rx

import pennylane as qml
from pennylane.wires import Wires


def x_mixer(wires: Union[Iterable, Wires]):
    r"""Creates a basic Pauli-X mixer Hamiltonian.

    This Hamiltonian is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{i} X_{i},

    where :math:`i` ranges over all wires, and :math:`X_i`
    denotes the Pauli-X operator on the :math:`i`-th wire.

    This is mixer is used in *A Quantum Approximate Optimization Algorithm*
    by Edward Farhi, Jeffrey Goldstone, Sam Gutmann [`arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`__].

    Args:
        wires (Iterable or Wires): The wires on which the Hamiltonian is applied

    Returns:
        Hamiltonian: Mixer Hamiltonian

    **Example**

    The mixer Hamiltonian can be called as follows:

    >>> from pennylane import qaoa
    >>> wires = range(3)
    >>> mixer_h = qaoa.x_mixer(wires)
    >>> print(mixer_h)
    1 * X(0) + 1 * X(1) + 1 * X(2)
    """

    wires = Wires(wires)

    coeffs = [1 for w in wires]
    obs = [qml.X(w) for w in wires]

    H = qml.Hamiltonian(coeffs, obs)
    # store the valuable information that all observables are in one commuting group
    H.grouping_indices = [list(range(len(H.ops)))]
    return H


def xy_mixer(graph: Union[nx.Graph, rx.PyGraph]):
    r"""Creates a generalized SWAP/XY mixer Hamiltonian.

    This mixer Hamiltonian is defined as:

    .. math:: H_M \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} X_i X_j \ + \ Y_i Y_j,

    for some graph :math:`G`. :math:`X_i` and :math:`Y_i` denote the Pauli-X and Pauli-Y operators on the :math:`i`-th
    wire respectively.

    This mixer was introduced in *From the Quantum Approximate Optimization Algorithm
    to a Quantum Alternating Operator Ansatz* by Stuart Hadfield, Zhihui Wang, Bryan O'Gorman,
    Eleanor G. Rieffel, Davide Venturelli, and Rupak Biswas `Algorithms 12.2 (2019) <https://doi.org/10.3390/a12020034>`__.

    Args:
        graph (nx.Graph or rx.PyGraph): A graph defining the collections of wires on which the Hamiltonian acts.

    Returns:
        Hamiltonian: Mixer Hamiltonian

    **Example**

    The mixer Hamiltonian can be called as follows:

    >>> from pennylane import qaoa
    >>> from networkx import Graph
    >>> graph = Graph([(0, 1), (1, 2)])
    >>> mixer_h = qaoa.xy_mixer(graph)
    >>> print(mixer_h)
      (0.5) [X0 X1]
    + (0.5) [Y0 Y1]
    + (0.5) [X1 X2]
    + (0.5) [Y1 Y2]

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1, ""), (1, 2, "")])
    >>> mixer_h = xy_mixer(graph)
    >>> print(mixer_h)
      (0.5) [X0 X1]
    + (0.5) [Y0 Y1]
    + (0.5) [X1 X2]
    + (0.5) [Y1 Y2]
    """

    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph object, got {type(graph).__name__}"
        )

    is_rx = isinstance(graph, rx.PyGraph)
    edges = graph.edge_list() if is_rx else graph.edges

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph.nodes()[i] if is_rx else i

    coeffs = 2 * [0.5 for e in edges]

    obs = []
    for node1, node2 in edges:
        obs.append(qml.X(get_nvalue(node1)) @ qml.X(get_nvalue(node2)))
        obs.append(qml.Y(get_nvalue(node1)) @ qml.Y(get_nvalue(node2)))

    return qml.Hamiltonian(coeffs, obs)


def bit_flip_mixer(graph: Union[nx.Graph, rx.PyGraph], b: int):
    r"""Creates a bit-flip mixer Hamiltonian.

    This mixer is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{v \in V(G)} \frac{1}{2^{d(v)}} X_{v}
              \displaystyle\prod_{w \in N(v)} (\mathbb{I} \ + \ (-1)^b Z_w)

    where :math:`V(G)` is the set of vertices of some graph :math:`G`, :math:`d(v)` is the
    `degree <https://en.wikipedia.org/wiki/Degree_(graph_theory)>`__ of vertex :math:`v`, and
    :math:`N(v)` is the `neighbourhood <https://en.wikipedia.org/wiki/Neighbourhood_(graph_theory)>`__
    of vertex :math:`v`. In addition, :math:`Z_v` and :math:`X_v`
    are the Pauli-Z and Pauli-X operators on vertex :math:`v`, respectively,
    and :math:`\mathbb{I}` is the identity operator.

    This mixer was introduced in `Hadfield et al. (2019) <https://doi.org/10.3390/a12020034>`__.

    Args:
         graph (nx.Graph or rx.PyGraph): A graph defining the collections of wires on which the Hamiltonian acts.
         b (int): Either :math:`0` or :math:`1`. When :math:`b=0`, a bit flip is performed on
             vertex :math:`v` only when all neighbouring nodes are in state :math:`|0\rangle`.
             Alternatively, for :math:`b=1`, a bit flip is performed only when all the neighbours of
             :math:`v` are in the state :math:`|1\rangle`.

    Returns:
        Hamiltonian: Mixer Hamiltonian

    **Example**

    The mixer Hamiltonian can be called as follows:

    >>> from pennylane import qaoa
    >>> from networkx import Graph
    >>> graph = Graph([(0, 1), (1, 2)])
    >>> mixer_h = qaoa.bit_flip_mixer(graph, 0)
    >>> mixer_h
    (
        0.5 * X(0)
      + 0.5 * (X(0) @ Z(1))
      + 0.25 * X(1)
      + 0.25 * (X(1) @ Z(2))
      + 0.25 * (X(1) @ Z(0))
      + 0.25 * (X(1) @ Z(0) @ Z(2))
      + 0.5 * X(2)
      + 0.5 * (X(2) @ Z(1))
    )

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1, ""), (1, 2, "")])
    >>> mixer_h = qaoa.bit_flip_mixer(graph, 0)
    >>> print(mixer_h)
    (
        0.5 * X(0)
      + 0.5 * (X(0) @ Z(1))
      + 0.25 * X(1)
      + 0.25 * (X(1) @ Z(2))
      + 0.25 * (X(1) @ Z(0))
      + 0.25 * (X(1) @ Z(0) @ Z(2))
      + 0.5 * X(2)
      + 0.5 * (X(2) @ Z(1))
    )
    """

    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(
            f"Input graph must be a nx.Graph or rx.PyGraph object, got {type(graph).__name__}"
        )

    if b not in [0, 1]:
        raise ValueError(f"'b' must be either 0 or 1, got {b}")

    sign = 1 if b == 0 else -1

    coeffs = []
    terms = []

    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.node_indexes() if is_rx else graph.nodes

    # In RX each node is assigned to an integer index starting from 0;
    # thus, we use the following lambda function to get node-values.
    get_nvalue = lambda i: graph.nodes()[i] if is_rx else i

    for i in graph_nodes:
        neighbours = sorted(graph.neighbors(i)) if is_rx else list(graph.neighbors(i))
        degree = len(neighbours)

        n_terms = [[qml.X(get_nvalue(i))]] + [
            [qml.Identity(get_nvalue(n)), qml.Z(get_nvalue(n))] for n in neighbours
        ]
        n_coeffs = [[1, sign] for n in neighbours]

        final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
        final_coeffs = [
            (0.5**degree) * functools.reduce(lambda x, y: x * y, list(m), 1)
            for m in itertools.product(*n_coeffs)
        ]

        coeffs.extend(final_coeffs)
        terms.extend(final_terms)

    return qml.Hamiltonian(coeffs, terms)
