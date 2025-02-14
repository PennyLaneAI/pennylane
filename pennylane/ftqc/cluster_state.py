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
This object stores all the topological information of a lattice.
"""

import networkx as nx
from pennylane import numpy as np

class ClusterStateLattice:
    """Class of cluster states lattice creation for measurement base quantum computing (MBQC). 

    Cluster states are usually generated in lattices of qubits and used as resources for MBQC [Entanglement in Graph States and its Applications, arXiv:quant-ph/0602096].
    Cluster state lattices can be in 1D, 2D or 3D.

    Args:
        dims: dimensions
        graph:
    """
    _short_name = "cluster_state_lattice"

    def __init__(self, dims: list = None, graph: nx.Graph = None):
        if dims is None and graph is None:
            raise ValueError("Please provide either lattice dimensions or a networkx graph object to create a lattice structure.")
        if dims is not None and graph is not None:
            raise ValueError("Please provide either lattice dimensions or a networkx graph object to create a lattice structure.")
        self._graph = graph
        if self._graph is not None:
            self._set_grid_graph(dims)

    def _set_grid_graph(self, dims: list):
        assert len(dims) > 0 and len(dims) < 4, f"{len(dims)}-dimension lattice is not supported."
        self._graph = nx.grid_graph(dims)
    
    def get_nodes(self):
        return self._graph.nodes
    

    def get_graph(self):
        return self._graph


