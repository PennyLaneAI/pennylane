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

"""
This file contains the classes and functions to create lattice structure object.
This object stores all the topological connectivity information of a lattice for FTQC.
"""

import networkx as nx


class Lattice:
    """Constructs a Lattice object for measurement base quantum computing (MBQC).

    Lattices, representing qubit connectivity, are crucial in measurement-based quantum computing (MBQC) and fault-tolerant quantum computing (FTQC).  MBQC often utilizes cluster states,
    typically generated on qubit lattices, as a computational resource [Entanglement in Graph States and its Applications, arXiv:quant-ph/0602096].  As discussed in [Measurement-based quantum
    computation with cluster states, arXiv:quant-ph/0301052], single-qubit gates can be implemented with a 1D chain of five entangled qubits, while two-qubit CNOT gates require 15 qubits
    arranged in a 2D lattice.  Furthermore, 3D lattice connectivity may be necessary to incorporate quantum error correction (QEC) into MBQC [A fault-tolerant one-way quantum computer,
    arxiv.org:quant-ph/0510135].

    Serving as a fundamental substrate for MBQC and FTQC, these lattices allow users to define indexing strategies, data distribution, measurement of qubits and defects, and other relevant parameters.

    This Lattice class, inspired by the design of ~pennylane.spin.Lattice, leverages NetworkX to represent the relationships within the lattice structure.

        Args:
            graph: A network undirected graph object.
    """

    _short_name = "ftqc_lattice"

    def __init__(self, graph: nx.Graph = None):
        self._graph = graph

    def get_neighbors(self, node):
        r"""Returns the neighbors of a given node in the lattice."""
        return self._graph.neighbors(node)

    def get_nodes(self):
        r"""Returns all nodes in the lattice."""
        return self._graph.nodes

    def get_edges(self):
        r"""Returns all edges in the lattice."""
        return self._graph.edges

    def get_graph(self):
        r"""Returns the underlying NetworkX graph object representing the lattice."""
        return self._graph


def generate_lattice(lattice, dims: list):
    r"""Generates a :class:`~pennylane.ftqc.Lattice` object for a given lattice shape and dimensions.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'triangle'``, ``'honeycomb'``.
        dims(list[int]): Number of nodes in each direction of the lattice.

    Returns:
        ~pennylane.ftqc.Lattice: lattice object.
    """

    lattice_shape = lattice.strip().lower()

    if lattice_shape not in ["chain", "rectangle", "honeycomb", "triangle", "cubic"]:
        raise ValueError(
            f"Lattice shape, '{lattice}' is not supported."
            f"Please set lattice to: 'chain', 'rectangle', 'honeycomb', 'triangle', 'cubic'."
        )

    if lattice_shape in ["chain", "rectangle", "cubic"]:
        if lattice_shape == "chain" and len(dims) != 1:
            raise ValueError(
                f"For a chain lattice, the length of dims should 1 instead of {len(dims)}"
            )

        if lattice_shape == "rectangle" and len(dims) != 2:
            raise ValueError(
                f"For a chain rectangle, the length of dims should 2 instead of {len(dims)}"
            )

        if lattice_shape == "cubic" and len(dims) != 3:
            raise ValueError(
                f"For a cubic lattice, the length of dims should 3 instead of {len(dims)}"
            )

        lattice_obj = Lattice(nx.grid_graph(dims))
        return lattice_obj

    if lattice_shape == "triangle":
        if len(dims) != 2:
            raise ValueError(
                f"For a triangle lattice, the length of dims should 2 instead of {len(dims)}"
            )
        lattice_obj = Lattice(nx.triangular_lattice_graph(dims[0], dims[1]))
        return lattice_obj

    if lattice_shape == "honeycomb":
        if len(dims) != 2:
            raise ValueError(
                f"For a honeycomb lattice, the length of dims should 2 instead of {len(dims)}"
            )
        lattice_obj = Lattice(nx.hexagonal_lattice_graph(dims[0], dims[1]))
        return lattice_obj
