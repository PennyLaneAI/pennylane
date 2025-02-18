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

from typing import Dict, List, Union

import networkx as nx


class Lattice:
    """Represents a qubit lattice for measurement-based quantum computing (MBQC) and fault-tolerant quantum computing (FTQC).

    Lattices, representing qubit connectivity, are crucial in measurement-based quantum computing (MBQC) and fault-tolerant quantum computing (FTQC).  MBQC often utilizes cluster states,
    typically generated on qubit lattices, as a computational resource [Entanglement in Graph States and its Applications, arXiv:quant-ph/0602096].  As discussed in [Measurement-based quantum
    computation with cluster states, arXiv:quant-ph/0301052], single-qubit gates can be implemented with a 1D chain of five entangled qubits, while two-qubit CNOT gates require 15 qubits
    arranged in a 2D lattice.  Furthermore, 3D lattice connectivity may be necessary to incorporate quantum error correction (QEC) into MBQC [A fault-tolerant one-way quantum computer,
    arxiv.org:quant-ph/0510135].

    This Lattice class, inspired by the design of ~pennylane.spin.Lattice, leverages NetworkX to represent the relationships within the lattice structure.

        Args:
            lattice_shape: Name of the lattice shape.
            graph: A NetworkX undirected graph object. If provided, `nodes` and `edges` are ignored.
            nodes: Nodes to construct a graph object. Ignored if `graph` is provided.
            egdes: Edges to construct the graph. Ignored if `graph` is provided.
        Raises:
            ValueError: If neither `graph` nor both `nodes` and `edges` are provided.
    """

    _short_name = "ftqc_lattice"

    def __init__(
        self, lattice_shape: str, graph: nx.Graph = None, nodes: List = None, edges: List = None
    ):
        self._lattice_shape = lattice_shape
        self._graph = graph
        if self._graph is None:
            if nodes is None and edges is None:
                raise ValueError(
                    "Neither a networkx Graph object nor nodes together with egdes are provided."
                )
            self._graph = nx.Graph()
            self._graph.add_nodes_from(nodes)
            self._graph.add_edges_from(edges)

    def get_lattice_shape(self):
        r"""Returns the lattice shape name."""
        return self._lattice_shape

    def relabel_nodes(self, mapping: Dict):
        r"""Relabel nodes of the NetworkX graph.
        #TODO: This method could be renamed later as it could be used for the node indexing only.

        Args:
            mapping: A dict with the old labels as keys and new labels as values.
        """
        nx.relabel_nodes(self._graph, mapping, copy=False)

    def set_node_attributes(self, attribute_name: str, attributes: Dict):
        r"""Add attributes to the nodes of the Networkx graph.
        #TODO: This method could be renamed later as it's possible that this method is only for stablizers setup.
        Args:
            attribute_name: Name of the node attribute to set.
            attributes: A dict with node labels as keys and attributes as values.
        """
        nx.set_node_attributes(self._graph, attributes, attribute_name)

    def get_node_attributes(self, attribute_name: str):
        r"""Return node attributes.
        Args:
            attribute_name: Name of the node attribute
        """
        return nx.get_node_attributes(self._graph, attribute_name)

    def set_edge_attributes(self, attribute_name: str, attributes: Union[str, Dict]):
        r"""Add attributes to the edges of the Network graph.
        #TODO: This method could be renamed later as it's possible that this method is only for the entanglement setup.
        Args:
            attribute_name: Name of the edge attribute to set.
            attributes: Edge attributes to set. It accepts a dict with node labels as keys and attributes as values or a scalar to set the new attribute of egdes with.
        """
        nx.set_edge_attributes(self._graph, attributes, attribute_name)

    def get_edge_attributes(self, attribute_name: str):
        r"""Add attributes to the edges of the Network graph.
        #TODO: This method could be renamed later as it's possible that this method is only for the entanglement setup.
        Args:
            attribute_name: Name of the edge attribute to set.
        """
        return nx.get_edge_attributes(self._graph, attribute_name)

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

        lattice_obj = Lattice(lattice_shape, nx.grid_graph(dims))
        return lattice_obj

    if lattice_shape == "triangle":
        if len(dims) != 2:
            raise ValueError(
                f"For a triangle lattice, the length of dims should 2 instead of {len(dims)}"
            )
        lattice_obj = Lattice(lattice_shape, nx.triangular_lattice_graph(dims[0], dims[1]))
        return lattice_obj

    if lattice_shape == "honeycomb":
        if len(dims) != 2:
            raise ValueError(
                f"For a honeycomb lattice, the length of dims should 2 instead of {len(dims)}"
            )
        lattice_obj = Lattice(lattice_shape, nx.hexagonal_lattice_graph(dims[0], dims[1]))
        return lattice_obj
