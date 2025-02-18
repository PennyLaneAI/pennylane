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

    Lattices which can be used to represent connectitivity of qubits play an important role in MBQC as well as fault-tolerant quantum computing (FTQC). 
    Cluster states are usually generated in lattices of qubits and used as resources for MBQC [Entanglement in Graph States and its Applications, arXiv:quant-ph/0602096].
    As mentioned in [Measurement-based quantum computation with cluster states, arXiv:quant-ph/0301052], 1-qubit gate operators can be represented by a 1D chain of 5 qubits
    entangled with each other, while 2-qubit CNOT gate can be represented by 15-qubits arranged in a 2D lattice. To support quantum error corrections (QEC), 3D lattices connectivity
    might be required to add QEC support to MBQC [A fault-tolerant one-way quantum computer, arxiv.org:quant-ph/0510135].

    As a fundamental stabstrate of MBQC and FTQC, users can define the indexing strategy as well as define the distribution of data and measure qubits as well as defects and so on.

    This class follows the design of `~pennylane.spin.Lattice`. Lattice is built on top of the networkx to represent the one-many relationship within a lattice.

    Args:
        graph: A network undirected graph object.
    """
    _short_name = "ftqc_lattice"

    def __init__(self, graph: nx.Graph = None):
        self._graph = graph

    def get_neighbors(self, node):
        return self._graph.neighbors(node)

    def get_nodes(self):
        return self._graph.nodes
    
    def get_edges(self):
        return self._graph.edges

    def get_graph(self):
        return self._graph

def generate_lattice(lattice, dims:list):
    r"""Generates a :class:`~pennylane.ftqc.Lattice` object for a given lattice shape and dimensions.

    Args:
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``,
            ``'rectangle'``, ``'triangle'``, ``'honeycomb'``.
        dims(list[int]): Number of nodes in each direction of the lattice.

    Returns:
        ~pennylane.ftqc.Lattice: lattice object.
    """

    lattice_shape = lattice.strip().lower()

    if lattice_shape not in [
        "chain",
        "rectangle",
        "honeycomb",
        "triangle",
        "cubic"
    ]:
        raise ValueError(
            f"Lattice shape, '{lattice}' is not supported."
            f"Please set lattice to: 'chain', 'rectangle', 'honeycomb', 'triangle', 'cubic'."
        )

    if lattice_shape in [
        "chain",
        "rectangle",
        "cubic"
    ]:
        if lattice_shape == "chain" and len(dims) != 1:
            raise ValueError(f"For a chain lattice, the length of dims should 1 instead of {len(dims)}")

        if lattice_shape == "rectangle" and len(dims) != 2:
            raise ValueError(f"For a chain rectangle, the length of dims should 2 instead of {len(dims)}")

        if lattice_shape == "cubic" and len(dims) != 3:
            raise ValueError(f"For a cubic lattice, the length of dims should 3 instead of {len(dims)}")

        lattice_obj = Lattice(
            nx.grid_graph(dims)
        )
        return lattice_obj
    
    if lattice_shape == "triangle":
        if len(dims) != 2:
            raise ValueError(f"For a triangle lattice, the length of dims should 2 instead of {len(dims)}")
        lattice_obj = Lattice(
            nx.triangular_lattice_graph(dims[0], dims[1])
        )
        return lattice_obj
    
    if lattice_shape == "honeycomb":
        if len(dims) != 2:
            raise ValueError(f"For a honeycomb lattice, the length of dims should 2 instead of {len(dims)}")
        lattice_obj = Lattice(
            nx.hexagonal_lattice_graph(dims[0], dims[1])
        )
        return lattice_obj
