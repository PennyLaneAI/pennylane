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

"""This module contains the classes and functions for creating a cluster state."""


import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class ClusterState:
    """Class of stabilizer-formalism cluster state for measurement-based quantum computing(MBQC).

    Cluster states is a specific instance of a graph state as illustated in [Entanglement in Graph States and its Applications, arXiv:quant-ph/0602096].
    With cluster states, univeral computation can be realized without the need of making use of any controlled two-system quantum gates. 

    Vertex connectivity:
    A cluster state can be defined in 1,2 and 3 dimensions, correponding to 1-D chain lattice, 2-D rectangular lattice and 3-D cubic lattices. 
    
    Stabilizer:
    Each vertex is a cluster state represents a qubit and there is one stabilizer generator S_j per graph vertex j of the form
    S_j = X_{j} \prod_{k\in N(j)} Z_k
    where the neighborhood N(j) is the set of vertices which share an edge with j [Cluster-state code, https://errorcorrectionzoo.org/c/cluster_state].

    #TODO: do we need 

    

    Args:

    graph: nx.Graph

    """
    _short_name = "cluster_state"

    def __init__(self):
        self._graph = None
        self._graph_type = None
        self._num_vertex = None

    def set_graph(self, graph: nx.Graph):
        self._graph = graph

    def set_grid_graph(self, dims: list):
        assert len(dims) > 0 and len(dims) < 4, f"{len(dims)}-dimension lattice is not supported."
        self._graph = nx.grid_graph(dims)
        # map n-dimension index labels to 1d index
        mapping = {}
        for node in self._graph:
            mapping[node] = np.ravel_multi_index(node, list(reversed(dims)), order='F')
        self._graph = nx.relabel_nodes(self._graph, mapping)

        # Add stabilizer attirbutes
        stabilizers = {}
        #Loop over all the neigbors and add it the stabilizers
        for node in self._graph:
            neighbors = nx.all_neighbors(self._graph, node)
            stabilizer = ["X" + str(node)]
            for neighbor in neighbors:
                stabilizer.append("Z"+str(neighbor))
            stabilizers[node] = stabilizer
    
        # Add stabilizers
        nx.set_node_attributes(self._graph, stabilizers, "stabilizers")

        # set attribute to edges
        edge_labels = "CNOT"
        nx.set_edge_attributes(self._graph, edge_labels, "ops")
    
    def draw(self):
        pos = nx.spring_layout(self._graph)
        nx.draw(self._graph, pos, with_labels=True)
        #nx.draw_networkx_labels(self._graph, pos, labels=nx.get_node_attributes(self._graph,'stabilizers')) # Draw edge labels
        nx.draw_networkx_edge_labels(self._graph, pos, edge_labels=nx.get_edge_attributes(self._graph,'ops')) # Draw edge labels
        plt.show()



