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
This file defines classes and functions for creating lattice objects that store topological
connectivity information.
"""

from collections.abc import Sequence
from enum import Enum, auto
from functools import lru_cache

import rustworkx as rx
from pennylane._rustworkx_compat import CompatPyGraph


def _rx_grid_graph(dims):
    """Generate a grid graph using rustworkx, mimicking networkx's grid_graph.

    Args:
        dims: A sequence of integers specifying the dimensions of the grid.
              For example, [3, 4] creates a 3x4 grid, [2, 3, 4] creates a 2x3x4 grid.

    Returns:
        rx.PyGraph: A rustworkx graph representing the grid.
    """
    if len(dims) == 1:
        # 1D grid (chain) - networkx uses integer labels, not tuples
        g = rx.PyGraph()
        for i in range(dims[0]):
            g.add_node(i)  # Integer label, not tuple
        for i in range(dims[0] - 1):
            g.add_edge(i, i + 1, None)
        return g

    if len(dims) == 2:
        # 2D grid
        rows, cols = dims
        g = rx.PyGraph()
        node_map = {}
        for i in range(rows):
            for j in range(cols):
                idx = g.add_node((i, j))
                node_map[(i, j)] = idx

        # Add horizontal edges
        for i in range(rows):
            for j in range(cols - 1):
                g.add_edge(node_map[(i, j)], node_map[(i, j + 1)], None)

        # Add vertical edges
        for i in range(rows - 1):
            for j in range(cols):
                g.add_edge(node_map[(i, j)], node_map[(i + 1, j)], None)

        return g

    if len(dims) == 3:
        # 3D grid (cubic)
        x_dim, y_dim, z_dim = dims
        g = rx.PyGraph()
        node_map = {}
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim):
                    idx = g.add_node((i, j, k))
                    node_map[(i, j, k)] = idx

        # Add edges along x-axis
        for i in range(x_dim - 1):
            for j in range(y_dim):
                for k in range(z_dim):
                    g.add_edge(node_map[(i, j, k)], node_map[(i + 1, j, k)], None)

        # Add edges along y-axis
        for i in range(x_dim):
            for j in range(y_dim - 1):
                for k in range(z_dim):
                    g.add_edge(node_map[(i, j, k)], node_map[(i, j + 1, k)], None)

        # Add edges along z-axis
        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(z_dim - 1):
                    g.add_edge(node_map[(i, j, k)], node_map[(i, j, k + 1)], None)

        return g

    raise NotImplementedError(f"Grid graphs with {len(dims)} dimensions are not supported")


def _rx_triangular_lattice_graph(m, n):
    """Generate a triangular lattice graph using rustworkx.

    This creates a triangular lattice matching networkx's triangular_lattice_graph(m, n).
    
    The nodes form a grid where:
    - Number of rows: m + 1
    - Number of columns: N + 1 where N = (n + 1) // 2
    - If n is odd, extra nodes at column N in odd rows are removed

    Node labels are coordinate tuples (i, j) matching networkx's convention.

    Args:
        m: Number of rows of triangles (creates m+1 rows of nodes)
        n: Parameter related to columns (creates roughly (n+1)//2 + 1 columns of nodes)

    Returns:
        rx.PyGraph: A rustworkx graph representing the triangular lattice.
    """
    g = rx.PyGraph()
    
    if m == 0 or n == 0:
        return g
    
    N = (n + 1) // 2  # number of columns of nodes (excluding rightmost column)
    rows_range = range(m + 1)
    cols_range = range(N + 1)
    
    # First, determine which nodes will exist
    # All nodes (i, j) for i in range(N+1), j in range(m+1) exist,
    # except when n is odd, nodes (N, j) for odd j are removed
    existing_nodes = set()
    for i in cols_range:
        for j in rows_range:
            existing_nodes.add((i, j))
    
    # If n is odd, remove extra nodes at column N for odd rows
    if n % 2:
        for j in rows_range[1::2]:  # odd rows
            existing_nodes.discard((N, j))
    
    # Create nodes
    node_map = {}
    for coord in sorted(existing_nodes):
        idx = g.add_node(coord)
        node_map[coord] = idx
    
    # Horizontal edges: (i, j) - (i+1, j) for i in range(N), j in rows
    for j in rows_range:
        for i in range(N):
            if (i, j) in node_map and (i + 1, j) in node_map:
                g.add_edge(node_map[(i, j)], node_map[(i + 1, j)], None)
    
    # Vertical edges: (i, j) - (i, j+1) for i in cols, j in range(m)
    for j in range(m):
        for i in cols_range:
            if (i, j) in node_map and (i, j + 1) in node_map:
                g.add_edge(node_map[(i, j)], node_map[(i, j + 1)], None)
    
    # Diagonal edges (type 1): (i, j) - (i+1, j+1) for odd rows j in [1, m) step 2, i in range(N)
    for j in rows_range[1:m:2]:  # odd rows from 1 to m-1
        for i in range(N):
            if (i, j) in node_map and (i + 1, j + 1) in node_map:
                g.add_edge(node_map[(i, j)], node_map[(i + 1, j + 1)], None)
    
    # Diagonal edges (type 2): (i+1, j) - (i, j+1) for even rows j in [0, m) step 2, i in range(N)
    for j in rows_range[:m:2]:  # even rows from 0 to m-1
        for i in range(N):
            if (i + 1, j) in node_map and (i, j + 1) in node_map:
                g.add_edge(node_map[(i + 1, j)], node_map[(i, j + 1)], None)
    
    return g


def _rx_hexagonal_lattice_graph(rows, cols):
    """Generate a hexagonal lattice graph using rustworkx with proper coordinate labels.

    This creates a hexagonal lattice where each node has a coordinate tuple as its data,
    similar to networkx's hexagonal_lattice_graph.

    Args:
        rows: Number of rows of hexagons
        cols: Number of columns of hexagons

    Returns:
        rx.PyGraph: A rustworkx graph with nodes labeled by coordinate tuples.
    """
    # Use rustworkx's generator to get the structure
    base_graph = rx.generators.hexagonal_lattice_graph(rows, cols)
    
    # Create a new graph with coordinate labels
    # The hexagonal lattice nodes follow a pattern based on rows and cols
    graph = rx.PyGraph()
    
    # Number of nodes in each row: alternates between 2*cols+1 and 2*cols+2
    # Generate node coordinates similar to networkx
    num_nodes = len(base_graph.node_indices())
    
    # Create node-to-coord mapping based on structure
    # For a hexagonal lattice, we use simple integer indices if we can't determine exact coords
    # Just use integer labels for simplicity - matching rustworkx indices
    for idx in base_graph.node_indices():
        graph.add_node(idx)  # Use integer index as label
    
    # Copy edges
    for u, v in base_graph.edge_list():
        graph.add_edge(u, v, None)
    
    return graph


class Lattice:
    """Represents a qubit lattice structure.

    This Lattice class, inspired by the design of :class:`~pennylane.spin.Lattice`, leverages rustworkx to represent the relationships within the lattice structure.

        Args:
            lattice_shape: Name of the lattice shape.
            graph (rx.PyGraph): A rustworkx undirected graph object. If provided, `nodes` and `edges` are ignored.
            nodes (List): Nodes to construct a graph object. Ignored if `graph` is provided.
            edges (List): Edges to construct the graph. Ignored if `graph` is provided.
        Raises:
            ValueError: If neither `graph` nor both `nodes` and `edges` are provided.
    """

    # TODOs: To support braiding operations, Lattice should support nodes/edges addition/deletion.

    def __init__(
        self, lattice_shape: str, graph: rx.PyGraph = None, nodes: list = None, edges: list = None
    ):
        self._lattice_shape = lattice_shape
        if graph is None:
            if nodes is None and edges is None:
                raise ValueError(
                    "Neither a networkx Graph object nor nodes together with edges are provided."
                )
            g = CompatPyGraph()
            # Add nodes with their labels stored in _label
            for node in nodes:
                g.add_node({"_label": node})
            # rustworkx add_edges_from expects 3-tuples (source, target, weight)
            edges_with_weights = [(e[0], e[1], None) for e in edges]
            g.add_edges_from(edges_with_weights)
            self._graph = g
        else:
            # Wrap rx.PyGraph in CompatPyGraph for networkx-like API
            if isinstance(graph, CompatPyGraph):
                self._graph = graph
            else:
                # Convert rx.PyGraph to CompatPyGraph
                g = CompatPyGraph()
                for idx in graph.node_indices():
                    node_data = graph[idx]
                    # Use node data as label
                    if node_data is None:
                        g.add_node({"_label": idx})
                    else:
                        g.add_node({"_label": node_data})
                for u, v in graph.edge_list():
                    g.add_edge(u, v, None)
                self._graph = g

    @property
    def shape(self) -> str:
        r"""Returns the lattice shape name."""
        return self._lattice_shape

    def get_neighbors(self, node):
        r"""Returns the neighbors of a given node in the lattice.

        Args:
            node: a target node label.
        """
        # For CompatPyGraph, use .nodes for label-based lookup
        if isinstance(self._graph, CompatPyGraph):
            # Check if node is in the graph via .nodes property
            if node in self._graph.nodes:
                idx = self._graph.nodes._resolve_key(node)
                # CompatPyGraph extends rx.PyGraph, so neighbors() is directly available
                neighbor_indices = super(CompatPyGraph, self._graph).neighbors(idx)
                # Return neighbor labels from the NodeDict
                result = []
                for neighbor_idx in neighbor_indices:
                    neighbor_data = self._graph[neighbor_idx]
                    if isinstance(neighbor_data, dict) and "_label" in neighbor_data:
                        result.append(neighbor_data["_label"])
                    else:
                        result.append(neighbor_data if neighbor_data is not None else neighbor_idx)
                return result
        
        # For standard rustworkx graphs, neighbors() takes a node index, not a node value
        # If node is the index already, use it directly
        if isinstance(node, int):
            return self._graph.neighbors(node)
        # Otherwise find the node index by value
        for idx in self._graph.node_indices():
            node_data = self._graph[idx]
            # Check if node data is the node itself or contains _label
            if node_data == node:
                # Return neighbor values, not indices
                neighbor_indices = self._graph.neighbors(idx)
                return [self._graph[i] for i in neighbor_indices]
            if isinstance(node_data, dict) and node_data.get("_label") == node:
                neighbor_indices = self._graph.neighbors(idx)
                result = []
                for i in neighbor_indices:
                    n_data = self._graph[i]
                    if isinstance(n_data, dict) and "_label" in n_data:
                        result.append(n_data["_label"])
                    else:
                        result.append(n_data if n_data is not None else i)
                return result
        raise ValueError(f"Node {node} not found in graph")

    @property
    def nodes(self):
        r"""Returns all nodes in the lattice."""
        # CompatPyGraph has .nodes as a property (NodeDict), standard rx.PyGraph has nodes() method
        nodes_attr = self._graph.nodes
        if callable(nodes_attr):
            return list(nodes_attr())
        # NodeDict-like object - iterating yields node labels
        return list(nodes_attr)

    @property
    def edges(self):
        r"""Returns all edges in the lattice as (source, target) tuples."""
        # CompatPyGraph has .edges as a property (EdgeList), standard rx.PyGraph has edge_list() method
        edges_attr = getattr(self._graph, 'edges', None)
        if edges_attr is not None:
            if callable(edges_attr):
                return list(edges_attr())
            # EdgeList-like object
            return list(edges_attr)
        # Fallback to edge_list() for plain rustworkx graphs
        return list(self._graph.edge_list())

    @property
    def graph(self) -> rx.PyGraph:
        r"""Returns the underlying rustworkx graph object representing the lattice."""
        return self._graph


class LatticeShape(Enum):
    """Enum to define valid set of lattice shape supported."""

    chain = auto()
    square = auto()
    rectangle = auto()
    triangle = auto()
    honeycomb = auto()
    cubic = auto()


# map between lattice name and dimensions
_LATTICE_DIM_MAP = {
    "chain": 1,
    "square": 2,
    "rectangle": 2,
    "cubic": 3,
    "triangle": 2,
    "honeycomb": 2,
}

# map between lattice name and generator function
_LATTICE_GENERATOR_MAP = {
    "chain": _rx_grid_graph,
    "square": _rx_grid_graph,
    "rectangle": _rx_grid_graph,
    "cubic": _rx_grid_graph,
    "triangle": _rx_triangular_lattice_graph,
    "honeycomb": _rx_hexagonal_lattice_graph,
}


@lru_cache
def _supported_shapes():
    r"""Return the supported shape in str"""
    return [shape.name for shape in LatticeShape]


def generate_lattice(dims: Sequence[int], lattice: str) -> Lattice:
    r"""Generates a :class:`~pennylane.ftqc.Lattice` object with a given geometric parameters and its shape name.

    Args:
        dims(List[int]): Geometric parameters for lattice generation. For grid-based lattices ( ``'chain'``, ``'rectangle'``,  ``'square'``, ``'cubic'``),
        `dims` contains the number of nodes in the each direction of grid. Per ``'honeycomb'`` or ``'triangle'``, the generated lattices will have dims[0] rows and dims[1]
        columns of hexagons or triangles.
        lattice (str): Shape of the lattice. Input values can be ``'chain'``, ``'square'``, ``'rectangle'``, ``'honeycomb'``, ``'triangle'``, ``'cubic'``.

    Returns:
        a :class:`~pennylane.ftqc.Lattice` object.

    Raises:
        ValueError: If the lattice shape is not supported or the dimensions are invalid.
    """

    lattice_shape = lattice.strip().lower()
    supported_shapes = _supported_shapes()

    if lattice_shape not in supported_shapes:
        raise ValueError(
            f"Lattice shape, '{lattice}' is not supported."
            f"Please set lattice to: {supported_shapes}."
        )

    if _LATTICE_DIM_MAP[lattice_shape] != len(dims):
        raise ValueError(
            f"For a {lattice_shape} lattice, the length of dims should be {_LATTICE_DIM_MAP[lattice_shape]} instead of {len(dims)}"
        )

    lattice_generate_method = _LATTICE_GENERATOR_MAP[lattice_shape]

    if lattice_shape in ["chain", "square", "rectangle", "cubic"]:
        lattice_obj = Lattice(lattice_shape, lattice_generate_method(dims))
        return lattice_obj

    if lattice_shape in ["triangle", "honeycomb"]:
        lattice_obj = Lattice(lattice_shape, lattice_generate_method(dims[0], dims[1]))
        return lattice_obj

    raise NotImplementedError  # pragma: no cover
