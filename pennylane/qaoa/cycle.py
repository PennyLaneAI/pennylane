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
