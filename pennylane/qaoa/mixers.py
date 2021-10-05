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
import itertools
import functools
import networkx as nx
import pennylane as qml
from pennylane.wires import Wires


def x_mixer(wires):
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
      (1) [X0]
    + (1) [X1]
    + (1) [X2]
    """

    wires = Wires(wires)

    coeffs = [1 for w in wires]
    obs = [qml.PauliX(w) for w in wires]

    return qml.Hamiltonian(coeffs, obs)


def xy_mixer(graph):
    r"""Creates a generalized SWAP/XY mixer Hamiltonian.

    This mixer Hamiltonian is defined as:

    .. math:: H_M \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} X_i X_j \ + \ Y_i Y_j,

    for some graph :math:`G`. :math:`X_i` and :math:`Y_i` denote the Pauli-X and Pauli-Y operators on the :math:`i`-th
    wire respectively.

    This mixer was introduced in *From the Quantum Approximate Optimization Algorithm
    to a Quantum Alternating Operator Ansatz* by Stuart Hadfield, Zhihui Wang, Bryan O'Gorman,
    Eleanor G. Rieffel, Davide Venturelli, and Rupak Biswas [`arXiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__].

    Args:
        graph (nx.Graph): A graph defining the collections of wires on which the Hamiltonian acts.

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
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError(
            "Input graph must be a nx.Graph object, got {}".format(type(graph).__name__)
        )

    edges = graph.edges
    coeffs = 2 * [0.5 for e in edges]

    obs = []
    for node1, node2 in edges:
        obs.append(qml.PauliX(node1) @ qml.PauliX(node2))
        obs.append(qml.PauliY(node1) @ qml.PauliY(node2))

    return qml.Hamiltonian(coeffs, obs)


def bit_flip_mixer(graph, b):
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

    This mixer was introduced in [`arXiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__].

    Args:
         graph (nx.Graph): A graph defining the collections of wires on which the Hamiltonian acts.
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
    >>> print(mixer_h)
      (0.25) [X1]
    + (0.5) [X0]
    + (0.5) [X2]
    + (0.25) [X1 Z2]
    + (0.25) [X1 Z0]
    + (0.5) [X0 Z1]
    + (0.5) [X2 Z1]
    + (0.25) [X1 Z0 Z2]
    """

    if not isinstance(graph, nx.Graph):
        raise ValueError(
            "Input graph must be a nx.Graph object, got {}".format(type(graph).__name__)
        )

    if b not in [0, 1]:
        raise ValueError("'b' must be either 0 or 1, got {}".format(b))

    sign = 1 if b == 0 else -1

    coeffs = []
    terms = []

    for i in graph.nodes:

        neighbours = list(graph.neighbors(i))
        degree = len(neighbours)

        n_terms = [[qml.PauliX(i)]] + [[qml.Identity(n), qml.PauliZ(n)] for n in neighbours]
        n_coeffs = [[1, sign] for n in neighbours]

        final_terms = [qml.operation.Tensor(*list(m)).prune() for m in itertools.product(*n_terms)]
        final_coeffs = [
            (0.5 ** degree) * functools.reduce(lambda x, y: x * y, list(m), 1)
            for m in itertools.product(*n_coeffs)
        ]

        coeffs.extend(final_coeffs)
        terms.extend(final_terms)

    return qml.Hamiltonian(coeffs, terms)
