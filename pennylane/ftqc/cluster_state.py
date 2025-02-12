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

"""This module contains the classes and functions for creating and diagonalizing
midcircuit measurements with a parameterized measurement axis."""


import networkx as nx
import math
import matplotlib.pyplot as plt
from networkx import Graph


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

    def __init__(self, dims: list = None, graph: Graph = None):
        assert len(dims) <= 3, f"{len(dims)}-dimensional cluster state is not supported."
        self._graph = len(dims)
        
    
    def _create_1d_chain(dims):
        labels = list(range(dims[0]))
        G = nx.Graph()
        for label in labels:
            G.add_node(label)
    
        for label in labels[1:]:
            G.add_edge(label - 1, label, label='CNOT')
    
        return G

    def _create_2d_rectangle(dims):
        assert len(dims) == 2, f"Rectangle lattice requires a 2 dims input, but {len(dims)} dim is provided"
        num_labels = math.prod(dims)
        labels = list(range(num_labels))
        G = nx.Graph()
        # Add nodes to the graph
        for label in labels:
            G.add_node(label)
    
        # Add row edges to the graph
        for j in range(dims[1]):
            for i in range(dims[0] - 1):
                start = j*dims[0] + i
                end = start + 1
                G.add_edge(start, end, lable='CNOT')
    
        for j in range(dims[1] - 1):
            for i in range(dims[0]):
                start = j*dims[0] + i
                end = start + dims[0]
                G.add_edge(start, end, label='CNOT')
    
        return G
    
    def _create_3d_cubic(dims):
        assert len(dims) == 3, f"cubic lattice requires a 3 dims input, but {len(dims)} dim is provided"
        num_labels = math.prod(dims)
        labels = list(range(num_labels))
        G = nx.Graph()
        # Add nodes to the graph
        for label in labels:
            G.add_node(label)
    
        # Add row edges to the graph
        for k in range(dims[2]):
            for j in range(dims[1]):
                for i in range(dims[0] - 1):
                    start = k*dims[1]*dims[0]+ j*dims[0] + i
                    end = start + 1
                    G.add_edge(start, end, label='CNOT')
    
        for k in range(dims[2]):
            for j in range(dims[1] - 1):
                for i in range(dims[0]):
                    start = k*dims[1]*dims[0]+ j*dims[0] + i
                    end = start + dims[0]
                    G.add_edge(start, end, label='CNOT')
    
        for k in range(dims[2]-1):
            for j in range(dims[1]):
                for i in range(dims[0]):
                    start = k*dims[1]*dims[0]+ j*dims[0] + i
                    end = start + dims[1]*dims[0]
                    G.add_edge(start, end, label='CNOT')
    
        return G



