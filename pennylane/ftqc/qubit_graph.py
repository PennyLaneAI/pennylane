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

"""This module contains the data structures to represent qubits using a hierarchical memory model
and defines the API to access and manipulate these data structures.
"""

import warnings
from typing import Optional, Union

import networkx as nx

# Fail-safe for algorithms that traverse the nested graph structure
# In other words, we should never expect more than this many layers of nested QubitGraphs
MAX_TRAVERSAL_DEPTH = 100


class QubitGraph:
    """A class to represent a hierarchical qubit memory model as nested graphs of qubits.

    A QubitGraph is a qubit that contains a graph of underlying qubits, where each underlying qubit
    is itself a QubitGraph. This representation allows for nesting of lower-level qubits with
    arbitrary depth to allow easy insertion of arbitrarily many levels of abstractions between
    logical qubits and physical qubits.

    Args:
        id (Any): An identifier for this QubitGraph object.
        graph (graph-like, optional): The graph to use as the QubitGraph's underlying qubits.
            Inputting None (the default), leaves the underlying qubit graph in an uninitialized
            state, in which case one of the graph-initialization methods may be used to define the
            structure of the underlying qubit graph. QubitGraph expects an undirected graph as
            input, specifically an instance of the ``networkx.Graph`` class, although other networkx
            graphs and graph-like types are also permitted. An object is considered "graph-like" if
            it has both a 'nodes' and an 'edges' attribute.

    TODO:

        * How to represent operations on qubits?
            * We should be able to broadcast operations to underlying qubits, assuming operations
              are transversal.
            * Recall that a _transversal operation_ is defined as a logical operator that is formed
              by applying the individual physical operators to each qubit in a QEC code block.
        * Implement tensor-like indexing and slicing.
    """

    def __init__(self, id, graph: Optional[nx.Graph] = None):
        if id is None:
            raise TypeError("'None' is not a valid QubitGraph ID.")

        self._id = id  # The identifier for this QubitGraph, e.g. a number, string, tuple, etc.

        if graph is not None:
            self._check_graph_type_supported_and_raise_or_warn(graph)

        self._graph_qubits = graph  # The qubits underlying (nested within) the current qubit
        self._parent = None  # The parent QubitGraph object under which this QubitGraph is nested

        # Initialize each node in the graph to store an empty QubitGraph object
        if self._graph_qubits is not None:
            self._initialize_all_nodes_as_qubit_graph()

    def __getitem__(self, key) -> "QubitGraph":
        """QubitGraph subscript operator for read access.

        Currently only basic, linear indexing and slicing is supported.

        Args:
            key (Any): Node label in the underlying qubit graph.

        TODO: Allow for more advanced tensor-like indexing and slicing.
        """
        if not self.is_initialized:
            self._warn_uninitialized()
            return None

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._graph_qubits.nodes))
            return [self._graph_qubits.nodes[node]["qubits"] for node in range(start, stop, step)]

        return self._graph_qubits.nodes[key]["qubits"]

    def __setitem__(self, key, value: "QubitGraph"):
        """QubitGraph subscript operator for assignment.

        Currently only basic, linear indexing is supported. Slicing is not supported.

        The QubitGraph assignment operator transfers ownership of the new QubitGraph object passed
        as the parameter `value` to the parent QubitGraph object. It does so by updating two of the
        new object's attributes:

            1. It updates the new object's `id` to be equal to the label of the node to which it has
               been assigned, as given by the `key` parameter.
            2. It updates the new object's `parent` attribute to be the current QubitGraph object.

        Args:
            key (Any): Node label in the underlying qubit graph.
            value (QubitGraph): The QubitGraph object to assign to the node with the given key.

        Example:

            >>> graph = nx.Graph()
            >>> graph.add_node(0)
            >>> q_top = QubitGraph("top", graph)
            >>> print(f"{q_top}; {q_top.nodes}")
            QubitGraph<top>; [0]
            >>> q_new = QubitGraph("new", graph)
            >>> print(f"{q_new}; {q_new.nodes}")
            QubitGraph<new>; [0]
            >>> q_top[0] = q_new
            >>> print(f"{q_top[0]}; {q_top[0].nodes}")
            QubitGraph<top, 0>; [0]

        TODO: Allow for more advanced tensor-like indexing and slicing.
        """
        if not isinstance(value, QubitGraph):
            raise TypeError(
                f"'QubitGraph' item assignment type must also be a QubitGraph, but got type "
                f"'{type(value).__name__}'"
            )

        if not self.is_initialized:
            self._warn_uninitialized()
            return

        value._id = key
        value._parent = self

        self._graph_qubits.nodes[key]["qubits"] = value

    def __iter__(self):
        """Dummy QubitGraph iterator method that yields itself.

        This method ensures that no implicit iteration takes place over the nodes in the underlying
        qubit graph, which is possible since we have defined __getitem__().

        Consider the example where we wrap a QubitGraph object in a tuple. With this dummy
        __iter__() method defined, the resulting tuple contains the top-level QubitGraph object:

            >>> q = QubitGraph(0)
            >>> q.init_graph_nd_grid((2,))
            >>> tuple(q)
            (<QubitGraph object>,)

        Without this dummy method defined, the tuple() constructor attempts to iterate over the
        values that the __getitem__() method yield, which creates a tuple of the QubitGraph objects
        contained in the underlying qubit graph:

            >>> q = QubitGraph(0)
            >>> q.init_graph_nd_grid((2,))
            >>> tuple(q)
            >>> (<QubitGraph object>, <QubitGraph object>)

        Making QubitGraph non-iterable is especially important when using it as input to the Wires
        class, which checks if the input is iterable by wrapping it in a tuple.
        """
        yield self

    def __repr__(self) -> str:
        """Representation of a QubitGraph object.

        This representation displays the full index hierarchy of a nested QubitGraph object.

        Examples:

            >>> QubitGraph(0)
            QubitGraph<0>

            >>> q = QubitGraph(0)
            >>> q.init_graph_nd_grid((2, 2))
            >>> q
            QubitGraph<0>
            >>> q[(0, 1)]
            QubitGraph<0, (0, 1)>
            >>> graph = nx.Graph()
            >>> graph.add_node("aux")
            >>> q[(0, 1)].init_graph(graph)
            >>> q[(0, 1)]["aux"]
            QubitGraph<0, (0, 1), aux>
        """
        if self.is_root:
            return f"QubitGraph<{self._id}>"

        depth_counter = 1

        ids = [self.id]
        parent = self.parent
        while parent is not None:
            # TODO: For now, we check against the fail-safe to prevent infinite traversal through
            # nested graph structure. This might arise if a user accidentally creates a cyclical
            # nesting structure, e.g. q0 -> q1 -> q0. In the future, we should explicitly check for
            # cycles, or better yet, check for cycles on assignment and prevent the user from
            # creating such a structure.
            if depth_counter >= MAX_TRAVERSAL_DEPTH:
                self._warn_max_traversal_depth_reached("__repr__")
                break

            ids.append(parent.id)
            parent = parent.parent

            depth_counter += 1

        ids_str = ", ".join(str(id) for id in ids[::-1])
        return f"QubitGraph<{ids_str}>"

    @property
    def id(self):
        """Gets the QubitGraph ID."""
        return self._id

    @property
    def graph(self):
        """Gets the underlying qubit graph."""
        return self._graph_qubits

    @property
    def nodes(self):
        """Gets the set of nodes in the underlying qubit graph.

        If the underlying qubit graph has not been initialized, emit a UserWarning and return None.

        Returns:
            networkx.NodeView: The set of nodes, with native support for operations such as
                `len(g.nodes)`, `n in g.nodes`, `g.nodes & h.nodes`, etc. See the networkx
                documentation for more information.
        """
        if self._graph_qubits is None:
            self._warn_uninitialized()
            return None

        return self._graph_qubits.nodes

    @property
    def edges(self):
        """Gets the set of edges in the underlying qubit graph.

        If the underlying qubit graph has not been initialized, emit a UserWarning and return None.

        Returns:
            networkx.EdgeView: The set of edges, with native support for operations such as
                `len(g.edges)`, `e in g.edges`, `g.edges & h.edges`, etc. See the networkx
                documentation for more information.
        """
        if self._graph_qubits is None:
            self._warn_uninitialized()
            return None

        return self._graph_qubits.edges

    @property
    def is_initialized(self) -> bool:
        """Checks if the underlying qubits have been initialized.

        The underlying qubit graph is considered uninitialized if and only if it is NoneType. A
        QubitGraph consisting of a null graph (one with zero nodes) is considered initialized. A
        QubitGraph may be uninitialized if it is a leaf node in the hierarchical graph structure.

        Returns:
            bool: Returns True if the underlying qubits have been initialized, False otherwise.
        """
        return self._graph_qubits is not None

    @property
    def is_leaf(self) -> bool:
        """Checks if this QubitGraph object is a leaf node in the hierarchical graph structure.

        A QubitGraph node is a leaf when it has no underlying qubit graph, either if the underlying
        qubit graph has not been initialized (i.e. it is NoneType) or if the underlying qubits graph
        has been initialized but is a null graph (one with zero nodes).

        Returns:
            bool: Returns True if this QubitGraph object is a leaf node.
        """
        return (self._graph_qubits is None) or (len(self.nodes) == 0)

    @property
    def parent(self) -> "QubitGraph":
        """Gets the parent QubitGraph of this QubitGraph object.

        Returns:
            QubitGraph: The parent QubitGraph object.
        """
        return self._parent

    @property
    def is_root(self) -> bool:
        """Checks if this QubitGraph object is a root node in the hierarchical graph structure.

        A QubitGraph node is a root when it has no parent QubitGraph object.

        Returns:
            bool: Returns True if this QubitGraph object is a root node.
        """
        return self._parent is None

    def clear(self):
        """Clears the graph of underlying qubits."""
        self._graph_qubits = None

    def connected_qubits(self, node):
        """Returns an iterator over all of the qubits connected to the qubit with label ``node``.

        Args:
            node (node-like): The label of a node in the qubit graph.

        Returns:
            iterator: An iterator over all QubitGraph objects connected to the qubit with label
                ``node``.
        """
        if not self.is_initialized:
            self._warn_uninitialized()
            return

        for neighbor in self._graph_qubits.neighbors(node):
            yield self[neighbor]

    def init_graph(self, graph: nx.Graph):
        """Initialize the QubitGraph's underlying qubits with the given graph.

        Args:
            graph (networkx.Graph): The undirected graph to use as the QubitGraph's underlying
                qubits. This object must not be None.

        Example:

            >>> graph = networkx.hexagonal_lattice_graph(3, 2)
            >>> q = QubitGraph(0, graph)
        """
        if graph is None:
            raise TypeError("QubitGraph requires a graph-like input, got NoneType.")

        self._check_graph_type_supported_and_raise_or_warn(graph)

        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph_qubits = graph
        self._initialize_all_nodes_as_qubit_graph()

    def init_graph_2d_grid(self, m: int, n: int):
        """Initialize the QubitGraph's underlying qubits as a 2-dimensional Cartesian grid of other
        QubitGraphs.

        Args:
            m, n (int): The number of rows, m, and columns, n, in the grid.

        Example:

            >>> q = QubitGraph(0)
            >>> q.init_graph_2d_grid(2, 3)

            This example initializes the underlying qubits as a 2x3 2-dimensional Cartesian grid
            with graph structure and qubit indexing below:

                (0,0) --- (0,1) --- (0,2)
                  |         |         |
                (1,0) --- (1,1) --- (1,2)
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph_qubits = nx.grid_2d_graph(m, n)
        self._initialize_all_nodes_as_qubit_graph()

    def init_graph_nd_grid(self, dim: Union[list[int], tuple[int]]):
        """Initialize the QubitGraph's underlying qubits as an n-dimensional Cartesian grid of other
        QubitGraphs.

        Args:
            dim (list or tuple of ints): The size of each dimension.

        Example:

            >>> q = QubitGraph(0)
            >>> q.init_graph_2d_grid(2, 2, 3)

            This example initializes the underlying qubits as a 2x2x3 3-dimensional Cartesian grid
            with graph structure and qubit indexing below:

                      (2,0,0) ------------- (2,0,1)
                     /|                    /|
                   (1,0,0) ------------- (1,0,1)
                  /|  |                 /|  |
                (0,0,0) ------------- (0,0,1)
                |  |  |               |  |  |
                |  |  (2,1,0) --------|--|- (2,1,1)
                |  | /                |  | /
                |  (1,1,0) -----------|- (1,1,1)
                | /                   | /
                (0,1,0) ------------- (0,1,1)
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph_qubits = nx.grid_graph(dim)
        self._initialize_all_nodes_as_qubit_graph()

    def init_graph_surface_code_17(self):
        r"""Initialize the QubitGraph's underlying qubits as the 17-qubit surface code graph from

            Tomita & Svore, 2014, Low-distance Surface Codes under Realistic Quantum Noise.
                https://arxiv.org/abs/1404.3747.

        This graph structure is commonly referred to as the "ninja star" surface code given its
        shape.

        The nodes are indexed as follows, where 'd' refers to data qubits and 'a' to auxiliary
        qubits:

                          a9
                         /   \
                d0     d1     d2
               /  \   /  \   /
            a10    a11    a12
               \  /   \  /   \
                d3     d4     d5
                  \   /  \   /  \
                   a13    a14    a15
                  /   \  /   \  /
                d6     d7     d8
                  \   /
                   a16
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        data_qubits = [("data", i) for i in range(9)]  # 9 data qubits, indexed 0, 1, ..., 8
        aux_qubits = [("aux", i) for i in range(9, 17)]  # 8 aux qubits, indexed 9, 10, ..., 16

        self._graph_qubits = nx.Graph()
        self._graph_qubits.add_nodes_from(data_qubits)
        self._graph_qubits.add_nodes_from(aux_qubits)

        # Adjacency list showing the connectivity of each auxiliary qubit to its neighbouring data qubits
        aux_adjacency_list = {
            9: [1, 2],
            10: [0, 3],
            11: [0, 1, 3, 4],
            12: [1, 2, 4, 5],
            13: [3, 4, 6, 7],
            14: [4, 5, 7, 8],
            15: [5, 8],
            16: [6, 7],
        }

        for aux_node, data_nodes in aux_adjacency_list.items():
            for data_node in data_nodes:
                self._graph_qubits.add_edge(("aux", aux_node), ("data", data_node))

        self._initialize_all_nodes_as_qubit_graph()

    def _initialize_all_nodes_as_qubit_graph(self):
        """Helper function to initialize all nodes in the underlying qubit graph as uninitialized
        QubitGraph objects. This functions also sets the _parent attribute appropriately.
        """
        assert self._graph_qubits is not None, "Underlying qubit graph object must not be None"
        assert hasattr(
            self._graph_qubits, "nodes"
        ), "Underlying qubit graph object must have 'nodes' attribute"

        for node in self._graph_qubits.nodes:
            q = QubitGraph(id=node)
            q._parent = self  # pylint: disable=protected-access
            self[node] = q

    @staticmethod
    def _warn_uninitialized():
        """Emit a UserWarning when attempting to access an uninitialized graph."""
        warnings.warn("Attempting to access an uninitialized QubitGraph.", UserWarning)

    @staticmethod
    def _warn_reinitialization():
        """Emit a UserWarning when attempting to initialize an already-initialized graph."""
        warnings.warn(
            "Attempting to re-initialize a QubitGraph. If you wish to initialize the underlying "
            "qubits with a new graph structure, you must first call QubitGraph.clear() and then "
            "call the initialization method.",
            UserWarning,
        )

    @staticmethod
    def _warn_max_traversal_depth_reached(algo_name: str):
        """Emit a UserWarning when an algorithm traversing through the layers of a nested QubitGraph
        surpasses the maximum traversal depth.

        Args:
            algo_name (str): The name of the algorithm that triggered the max traversal-depth
                warning.
        """
        warnings.warn(
            f"Maximum traversal depth reached in '{algo_name}' "
            f"(traversal depth > {MAX_TRAVERSAL_DEPTH})",
            UserWarning,
        )

    @staticmethod
    def _check_graph_type_supported_and_raise_or_warn(graph):
        """Check that the input type is graph-like and raise a TypeError if not, and then check that
        the graph type is one that is supported and emit a UserWarning if not.

        The input is considered "graph-like" if it has both a 'nodes' and an 'edges' attribute.

        Currently, QubitGraph only supports networkx graphs; specifically, graph objects that are
        instances of the `networkx.Graph
        <https://networkx.org/documentation/stable/reference/classes/graph.html>` class, or
        subclasses thereof.

        Note that other networkx graph types, including ``DiGraph``, ``MultiGraph`` and
        ``MultiDiGraph`` are all subclasses of the ``Graph`` class and are therefore permitted,
        although their usage is discouraged since they store additional information that is not used
        by QubitGraph.

        Args:
            graph: The graph object used for type-checking. This object must not be None.
        """
        assert graph is not None, "Graph object used for type-checking must not be None"

        if not hasattr(graph, "nodes") or not hasattr(graph, "edges"):
            raise TypeError(
                "QubitGraph requires a graph-like input, i.e. an object having both a 'nodes' and "
                "an 'edges' attribute."
            )

        if not isinstance(graph, nx.Graph):
            warnings.warn(
                f"QubitGraph expects an input graph of type 'networkx.Graph', but got "
                f"'{type(graph).__name__}'. Using a graph of another type may result in unexpected "
                f"behaviour.",
                UserWarning,
            )
