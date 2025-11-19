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

import uuid
import warnings
from typing import Any

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
        graph (graph-like, optional): The graph structure to use for the QubitGraph's underlying
            qubits. The graph must be "flat" (unnested). An object is considered "graph-like" if it
            has both a 'nodes' and an 'edges' attribute. Defaults to None, which leaves the
            QubitGraph in an uninitialized state.
        id (Any, optional): An identifier for this QubitGraph object. The identifier is generally an
            integer, string, or a tuple of integers and strings, but it may be any object. Inputting
            None (the default), assigns a random universally unique identifier (uuid) to ``id``.

    .. note::
        The input graph defines the structure of the underlying qubit graph only; only the set of
        nodes and the set of edges defining the connectivity between nodes are used to construct the
        underlying qubit graph. Any data contained within the input graph's nodes, including nested
        graphs, are ignored.

    .. note::
        QubitGraph expects an undirected graph as input, specifically an instance of the
        ``networkx.Graph`` class, although other NetworkX graphs and graph-like types are also
        permitted.

    **Examples**

    Create a QubitGraph (with id=0) using a 2x2 Cartesian grid to define the structure of its
    underlying qubits:

    >>> import networkx as nx
    >>> from pennylane.ftqc import QubitGraph
    >>> q = QubitGraph(nx.grid_graph((2,2)), id=0)
    >>> q
    QubitGraph<id=0, loc=[]>
    >>> for child in q.children:
    ...     print(child)
    QubitGraph<id=(0, 0), loc=[0]>
    QubitGraph<id=(0, 1), loc=[0]>
    QubitGraph<id=(1, 0), loc=[0]>
    QubitGraph<id=(1, 1), loc=[0]>

    Using the QubitGraph object defined above as a starting point, it's possible to nest other
    QubitGraph objects in any of the nodes of its underlying qubit graph:

    >>> q[(0,0)].init_graph(nx.grid_graph((2,)))
    >>> q[(0,0)]
    QubitGraph<id=(0, 0), loc=[0]>
    >>> for child in q[(0,0)].children:
    ...     print(child)
    QubitGraph<id=0, loc=[(0, 0), 0]>
    QubitGraph<id=1, loc=[(0, 0), 0]>

    Notice that you do not have to provide an ID when constructing the QubitGraph; when a
    QubitGraph object is assigned to the node of another QubitGraph, its ID takes on the label
    of the node it was assigned to.

    .. details::
        :title: How to Read the QubitGraph String Representation

        The examples above showed that any QubitGraph in the hierarchical structure can be displayed
        with information on its ID and its location in the hierarchy. For example, the string
        representation

        ::

            QubitGraph<id=1, loc=[(0, 0), 0]>

        indicates that this QubitGraph has an ID of ``1``, and it is nested within a parent node
        labelled ``(0, 0)``, which is nested within a parent node labelled ``0``, which is a root
        node. If a QubitGraph is itself a root node (meaning it is not nested within a parent
        QubitGraph), its location is displayed as an empty list, for example:

        ::

            QubitGraph<id=0, loc=[]>

        If no ID parameter was given upon construction of the QubitGraph, you will notice that a
        random UUID has been assigned to it:

        >>> q = QubitGraph()
        >>> q # doctest: +SKIP
        QubitGraph<id=7491161c, loc=[]>

        This ID is truncated for brevity; the full ID is a 32-digit hexadecimal number:

        >>> q.id # doctest: +SKIP
        '7491161c-ca7e-42cc-af3e-8ca6250a370e'

    .. details::
        :title: Initializing a QubitGraph

        The examples above showed how to initialize the graph structure of a QubitGraph by passing
        in a NetworkX graph. It is also possible to construct a QubitGraph object with no graph
        structure:

        >>> q = QubitGraph(id=0)
        >>> q
        QubitGraph<id=0, loc=[]>

        In this case, the QubitGraph is still valid for use, but it is in an *uninitialized* state.
        You can check the initialization state of a QubitGraph with its
        :attr:`~QubitGraph.is_initialized` attribute:

        >>> q.is_initialized
        False

        You may not want to initialize a graph of underlying qubits when representing physical
        qubits at the hardware layer, for example.

        An uninitialized QubitGraph can always be initialized later using one of its
        graph-initialization methods. The most general of these is :meth:`~QubitGraph.init_graph`,
        which accepts an arbitrary NetworkX graph as input:

        >>> graph = nx.grid_graph((2,))
        >>> q.init_graph(graph)
        >>> q.is_initialized
        True

        The :meth:`~QubitGraph.init_graph` method behaves in the same way as passing the graph input
        directly to the QubitGraph constructor, and the same requirements and caveats listed above
        apply here as well (the input must be graph-like and any data annotated on the nodes are
        ignored).

        Other graph-initialization methods that automatically construct common graph structures are
        also available. For example, :meth:`~QubitGraph.init_graph_nd_grid` initializes a
        QubitGraph's underlying qubits as an n-dimensional Cartesian grid:

        >>> q_3d_grid = QubitGraph(id=0)
        >>> q_3d_grid.init_graph_nd_grid((3, 4, 5))

    .. details::
        :title: Traversing the Hierarchical Graph Structure

        The graph hierarchy is structured as a tree data type, with "root" nodes representing the
        highest-level qubits (for example, the logical qubits in a quantum circuit), down to "leaf"
        nodes representing the lowest-level qubits (for example, the physical qubits at the hardware
        level). The :attr:`~QubitGraph.is_root` and :attr:`~QubitGraph.is_leaf` attributes are
        available to quickly determine if a QubitGraph is a root or leaf node, respectively.
        Consider the following example with a three-layer nesting structure:

        >>> single_node_graph = nx.grid_graph((1,))
        >>> q = QubitGraph(single_node_graph, id=0)
        >>> q[0].init_graph(single_node_graph)
        >>> print(f"is root? {q.is_root}, is leaf? {q.is_leaf}")
        is root? True, is leaf? False
        >>> print(f"is root? {q[0].is_root}, is leaf? {q[0].is_leaf}")
        is root? False, is leaf? False
        >>> print(f"is root? {q[0][0].is_root}, is leaf? {q[0][0].is_leaf}")
        is root? False, is leaf? True

        The :attr:`~QubitGraph.parent` and :attr:`~QubitGraph.children` attributes are also
        available to access the parent of a given QubitGraph and its set of children, respectively:

        >>> q[0].parent is q
        True
        >>> q[0][0].parent is q[0]
        True
        >>> list(q[0].children)
        [QubitGraph<id=0, loc=[0, 0]>]

    ..  TODO:

        - How to represent operations on qubits?
            - We should be able to broadcast operations to underlying qubits, assuming operations
              are transversal.
            - Recall that a *transversal operation* is defined as a logical operator that is formed
              by applying the individual physical operators to each qubit in a QEC code block.
        - Implement tensor-like indexing and slicing.
    """

    def __init__(self, graph: nx.Graph | None = None, id: Any | None = None):
        # The identifier for this QubitGraph, e.g. a number, string, tuple, etc.
        # Generate a random uuid if the input is None
        self._id = uuid.uuid4() if id is None else id

        graph_copy = None
        if graph is not None:
            self._check_graph_type_supported_and_raise_or_warn(graph)
            graph_copy = self._copy_graph_structure(graph)

        # The graph structure of the qubits underlying (nested within) the current qubit
        self._graph = graph_copy

        # The parent QubitGraph object under which this QubitGraph is nested
        self._parent = None

        # Initialize each node in the graph to store an empty QubitGraph object
        if self._graph is not None:
            self._initialize_all_nodes_as_qubit_graph()

    def __getitem__(self, key: Any) -> "QubitGraph":
        """QubitGraph subscript operator for read access.

        Currently only basic, linear indexing and slicing is supported.

        Args:
            key (Any): Node label in the underlying qubit graph.

        ..  TODO:

            - Allow for more advanced tensor-like indexing and slicing.
        """
        if not self.is_initialized:
            self._warn_uninitialized()
            return None

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self._graph.nodes))
            return [self._graph.nodes[node]["qubits"] for node in range(start, stop, step)]

        return self._graph.nodes[key]["qubits"]

    def __setitem__(self, key: Any, value: "QubitGraph"):
        """QubitGraph subscript operator for assignment.

        Currently only basic, linear indexing is supported. Slicing is not supported.

        The QubitGraph assignment operator transfers ownership of the new QubitGraph object passed
        as the parameter ``value`` to the parent QubitGraph object. It does so by updating two of
        the new object's attributes:

            1. It updates the new object's ``id`` to be equal to the label of the node to which it
               has been assigned, as given by the ``key`` parameter.
            2. It updates the new object's ``parent`` attribute to be the current QubitGraph object.

        Args:
            key (Any): Node label in the underlying qubit graph.
            value (QubitGraph): The QubitGraph object to assign to the node with the given key.

        **Example**

        >>> graph = nx.Graph()
        >>> graph.add_node(0)
        >>> q_top = QubitGraph(graph, id="top")
        >>> print(f"{q_top}; {q_top.node_labels}")
        QubitGraph<id=top, loc=[]>; [0]
        >>> q_new = QubitGraph(graph, "new")
        >>> print(f"{q_new}; {q_new.node_labels}")
        QubitGraph<id=new, loc=[]>; [0]
        >>> q_top[0] = q_new
        >>> print(f"{q_top[0]}; {q_top[0].node_labels}")
        QubitGraph<id=0, loc=[top]>; [0]

        ..  TODO:

            - Allow for more advanced tensor-like indexing and slicing.
            - Explicitly disallow assigning a QubitGraph object that would make the nesting
              structure cyclic.
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

        self._graph.nodes[key]["qubits"] = value

    def __iter__(self):
        """Dummy QubitGraph iterator method that yields itself.

        This method ensures that no implicit iteration takes place over the nodes in the underlying
        qubit graph, which is possible since we have defined ``__getitem__()``.

        Consider the example where we wrap a QubitGraph object in a tuple. With this dummy
        ``__iter__()`` method defined, the resulting tuple contains the top-level QubitGraph object:

        >>> q = QubitGraph(id=0)
        >>> q.init_graph_nd_grid((2,))
        >>> tuple(q)
        (QubitGraph<id=0, loc=[]>,)

        Without this dummy method defined, the ``tuple()`` constructor attempts to iterate over the
        values that the ``__getitem__()`` method yield, which creates a tuple of the QubitGraph
        objects contained in the underlying qubit graph:

        >>> # Without QubitGraph.__iter__() defined
        >>> q = QubitGraph(id=0)
        >>> q.init_graph_nd_grid((2,))
        >>> tuple(q) # doctest: +SKIP
        (QubitGraph<id=0, loc=[0]>, QubitGraph<id=1, loc=[0]>)

        Making QubitGraph non-iterable is especially important when using it as input to the
        :class:`~.Wires` class, which checks if the input is iterable by wrapping it in a tuple.
        """
        yield self

    def __repr__(self) -> str:
        """Representation of a QubitGraph object.

        This representation displays the full index hierarchy of a nested QubitGraph object.

        **Examples**

        >>> QubitGraph(id=0)
        QubitGraph<id=0, loc=[]>

        >>> q = QubitGraph(id=0)
        >>> q.init_graph_nd_grid((2, 2))
        >>> q
        QubitGraph<id=0, loc=[]>
        >>> q[(0, 1)]
        QubitGraph<id=(0, 1), loc=[0]>
        >>> graph = nx.Graph()
        >>> graph.add_node("aux")
        >>> q[(0, 1)].init_graph(graph)
        >>> q[(0, 1)]["aux"]
        QubitGraph<id=aux, loc=[(0, 1), 0]>
        """
        # Truncate the ID representation if it is a UUID
        id_repr = self._truncate_id_if_uuid(self._id)

        if self.is_root:
            return f"QubitGraph<id={id_repr}, loc=[]>"

        if self.has_cycle():
            # NOTE: For now, we check if the QubitGraph nesting structure is cyclic and break early,
            # rather than iterating up through the parents until we reach the max-traversal depth.
            # Eventually, it would be better to explicitly disallow constructing a cyclically nested
            # QubitGraph.
            self._warn_cyclically_nested_graph()
            return f"QubitGraph<id={id_repr}; cyclic>"

        depth_counter = 1

        ids = []
        parent = self.parent
        while parent is not None:
            # To prevent extremely long string representations, we break early once we reach the
            # max-traversal depth. Generally, we should never expect more than this many layers of
            # nested QubitGraphs, so emit a warning.
            if depth_counter >= MAX_TRAVERSAL_DEPTH:
                self._warn_max_traversal_depth_reached("__repr__")
                ids.append("...")
                break

            ids.append(self._truncate_id_if_uuid(parent._id))
            parent = parent.parent

            depth_counter += 1

        ids_str = ", ".join(str(id) for id in ids)
        return f"QubitGraph<id={id_repr}, loc=[{ids_str}]>"

    @property
    def id(self):
        """Gets the QubitGraph ID."""
        return str(self._id) if isinstance(self._id, uuid.UUID) else self._id

    @property
    def graph(self):
        """Gets the underlying qubit graph."""
        return self._graph

    @property
    def parent(self) -> "QubitGraph":
        """Gets the parent QubitGraph of this QubitGraph object.

        Returns:
            QubitGraph: The parent QubitGraph object.
        """
        return self._parent

    @property
    def is_initialized(self) -> bool:
        """Checks if the underlying qubits have been initialized.

        The underlying qubit graph is considered uninitialized if and only if it is NoneType. A
        QubitGraph consisting of a null graph (one with zero nodes) is considered initialized. A
        QubitGraph may be uninitialized if it is a leaf node in the hierarchical graph structure.

        Returns:
            bool: Returns True if the underlying qubits have been initialized, False otherwise.
        """
        return self._graph is not None

    @property
    def is_leaf(self) -> bool:
        """Checks if this QubitGraph object is a leaf node in the hierarchical graph structure.

        A QubitGraph node is a leaf when it has no underlying qubit graph, either if the underlying
        qubit graph has not been initialized (i.e. it is NoneType) or if the underlying qubits graph
        has been initialized but is a null graph (one with zero nodes).

        Returns:
            bool: Returns True if this QubitGraph object is a leaf node.
        """
        return (not self.is_initialized) or (len(self._graph.nodes) == 0)

    @property
    def is_root(self) -> bool:
        """Checks if this QubitGraph object is a root node in the hierarchical graph structure.

        A QubitGraph node is a root when it has no parent QubitGraph object.

        Returns:
            bool: Returns True if this QubitGraph object is a root node.
        """
        return self._parent is None

    @property
    def node_labels(self):
        """Gets the set of node labels in the underlying qubit graph.

        If the underlying qubit graph has not been initialized, emit a UserWarning and return None.

        Accessing ``QubitGraph.node_labels`` is equivalent to accessing the ``nodes`` attribute of
        the NetworkX graph:

        >>> g = nx.grid_graph((2,))
        >>> g.nodes
        NodeView((0, 1))
        >>> q = QubitGraph(g, id=0)
        >>> q.node_labels
        NodeView((0, 1))

        To access the underlying QubitGraph *objects*, rather than their labels, use the
        :attr:`~QubitGraph.children` attribute.

        Returns:
            networkx.NodeView: A view of the set of nodes, with native support for operations such
            as ``len(g.nodes)``, ``n in g.nodes``, ``g.nodes & h.nodes``, etc. See the NetworkX
            documentation for more information.
        """
        if not self.is_initialized:
            self._warn_uninitialized()
            return None

        return self._graph.nodes

    @property
    def edge_labels(self):
        """Gets the set of edge labels in the underlying qubit graph.

        If the underlying qubit graph has not been initialized, emit a UserWarning and return None.

        Accessing ``QubitGraph.edges_labels`` is equivalent to accessing the ``edges`` attribute of
        the NetworkX graph:

        >>> g = nx.grid_graph((2,))
        >>> g.edges
        EdgeView([(0, 1)])
        >>> q = QubitGraph(g, id=0)
        >>> q.edge_labels
        EdgeView([(0, 1)])

        Returns:
            networkx.EdgeView: The set of edges, with native support for operations such as
            ``len(g.edges)``, ``e in g.edges``, ``g.edges & h.edges``, etc. See the NetworkX
            documentation for more information.
        """
        if not self.is_initialized:
            self._warn_uninitialized()
            return None

        return self._graph.edges

    @property
    def children(self):
        """Gets an iterator over the set of children QubitGraph objects.

        To access the node labels of the underlying qubit graph, rather than the QubitGraph objects
        themselves, use the :attr:`~QubitGraph.node_labels` attribute.

        Yields:
            QubitGraph: The next QubitGraph object in the set of children QubitGraphs.

        **Example**

        >>> q = QubitGraph(nx.grid_graph((2,)), id=0)
        >>> set(q.node_labels)
        {0, 1}
        >>> list(q.children)
        [QubitGraph<id=0, loc=[0]>, QubitGraph<id=1, loc=[0]>]
        """
        if not self.is_initialized:
            self._warn_uninitialized()
            return

        for node in self.node_labels:
            yield self[node]

    @property
    def neighbors(self):
        """Gets an iterator over all of the QubitGraph objects connected to this QubitGraph (its
        *neighbors*).

        A QubitGraph does not have to be initialized for it to have neighbors. Similarly, a
        root-level QubitGraph does not have any neighboring qubits, by construction.

        Yields:
            QubitGraph: The next QubitGraph object in the set of neighboring QubitGraphs.

        **Example**

        >>> q = QubitGraph(nx.grid_graph((2, 2)), id=0)
        >>> list(q[(0,0)].neighbors)
        [QubitGraph<id=(1, 0), loc=[0]>, QubitGraph<id=(0, 1), loc=[0]>]
        """
        if self.is_root:
            return

        for neighbor_id in self._parent.graph.neighbors(self._id):
            yield self._parent[neighbor_id]

    def clear(self):
        """Clears the graph of underlying qubits."""
        self._graph = None

    def has_cycle(self) -> bool:
        """Checks if the QubitGraph contains a cycle in its nesting structure.

        This method uses Floyd's cycle-finding algorithm (also known as Floyd's tortoise and hare
        algorithm) by iterating up through the QubitGraphs' parents.

        Returns:
            bool: Returns True if this QubitGraph has a cycle in its nesting structure.
        """
        if self._parent is None:
            return False

        slow = self._parent
        fast = self._parent

        while (fast is not None) and (fast.parent is not None):
            slow = slow.parent
            fast = fast.parent.parent

            if slow == fast:
                return True

        return False

    def init_graph(self, graph: nx.Graph):
        """Initialize the QubitGraph's underlying qubits with the given graph.

        Args:
            graph (networkx.Graph): The undirected graph to use as the QubitGraph's underlying
                qubits. This object must not be None.

        **Example**

        This example creates a NetworkX graph with two nodes, labelled 0 and 1, and one edge
        between them, and uses this graph to initialize the graph structure of a QubitGraph:

        >>> import networkx as nx
        >>> graph = nx.Graph()
        >>> graph.add_edge(0, 1)
        >>> q = QubitGraph(id=0)
        >>> q.init_graph(graph)
        >>> list(q.children)
        [QubitGraph<id=0, loc=[0]>, QubitGraph<id=1, loc=[0]>]
        """
        if graph is None:
            raise TypeError("QubitGraph requires a graph-like input, got NoneType.")

        self._check_graph_type_supported_and_raise_or_warn(graph)

        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph = self._copy_graph_structure(graph)
        self._initialize_all_nodes_as_qubit_graph()

    def init_graph_2d_grid(self, m: int, n: int):
        """Initialize the QubitGraph's underlying qubits as a 2-dimensional Cartesian grid of other
        QubitGraphs.

        Args:
            m (int): The number of rows in the grid.
            n (int): The number of columns in the grid.

        **Example**

        This example initializes the underlying qubits as a 2x3 2-dimensional Cartesian grid
        with graph structure and qubit indexing below:

        ::

            (0,0) --- (0,1) --- (0,2)
              |         |         |
            (1,0) --- (1,1) --- (1,2)

        >>> q = QubitGraph(id=0)
        >>> q.init_graph_2d_grid(2, 3)
        >>> list(q.children)
        [QubitGraph<id=(0, 0), loc=[0]>, QubitGraph<id=(0, 1), loc=[0]>,
        QubitGraph<id=(0, 2), loc=[0]>, QubitGraph<id=(1, 0), loc=[0]>,
        QubitGraph<id=(1, 1), loc=[0]>, QubitGraph<id=(1, 2), loc=[0]>]
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph = nx.grid_2d_graph(m, n)
        self._initialize_all_nodes_as_qubit_graph()

    def init_graph_nd_grid(self, dim: list[int] | tuple[int]):
        """Initialize the QubitGraph's underlying qubits as an n-dimensional Cartesian grid of other
        QubitGraphs.

        Args:
            dim (list or tuple of ints): The size of each dimension.

        **Example**

        This example initializes the underlying qubits as a 2x2x3 3-dimensional Cartesian grid
        with graph structure and qubit indexing below:

        ::

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

        >>> q = QubitGraph(id=0)
        >>> q.init_graph_nd_grid((2, 2, 3))
        >>> list(q.children)
        [QubitGraph<id=(0, 0, 0), loc=[0]>, QubitGraph<id=(0, 0, 1), loc=[0]>,
        QubitGraph<id=(0, 1, 0), loc=[0]>, QubitGraph<id=(0, 1, 1), loc=[0]>,
        QubitGraph<id=(1, 0, 0), loc=[0]>, QubitGraph<id=(1, 0, 1), loc=[0]>,
        QubitGraph<id=(1, 1, 0), loc=[0]>, QubitGraph<id=(1, 1, 1), loc=[0]>,
        QubitGraph<id=(2, 0, 0), loc=[0]>, QubitGraph<id=(2, 0, 1), loc=[0]>,
        QubitGraph<id=(2, 1, 0), loc=[0]>, QubitGraph<id=(2, 1, 1), loc=[0]>]
        """
        if self.is_initialized:
            self._warn_reinitialization()
            return

        self._graph = nx.grid_graph(dim)
        self._initialize_all_nodes_as_qubit_graph()

    def init_graph_surface_code_17(self):
        r"""Initialize the QubitGraph's underlying qubits as the 17-qubit surface code graph from

            Y. Tomita & K. Svore, 2014, *Low-distance Surface Codes under Realistic Quantum Noise*.
            `arXiv:1404.3747 <https://arxiv.org/abs/1404.3747>`_.

        This graph structure is commonly referred to as the "ninja star" surface code given its
        shape.

        The nodes are indexed as follows, where 'd' refers to data qubits and 'a' to auxiliary
        qubits:

        ::

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

        self._graph = nx.Graph()
        self._graph.add_nodes_from(data_qubits)
        self._graph.add_nodes_from(aux_qubits)

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
                self._graph.add_edge(("aux", aux_node), ("data", data_node))

        self._initialize_all_nodes_as_qubit_graph()

    def _initialize_all_nodes_as_qubit_graph(self):
        """Helper function to initialize all nodes in the underlying qubit graph as uninitialized
        QubitGraph objects. This function also sets the ``parent`` attribute appropriately.
        """
        assert self._graph is not None, "Underlying qubit graph object must not be None"
        assert hasattr(
            self._graph, "nodes"
        ), "Underlying qubit graph object must have 'nodes' attribute"

        for node in self._graph.nodes:
            q = QubitGraph(id=node)
            q._parent = self  # pylint: disable=protected-access
            self[node] = q

    @staticmethod
    def _copy_graph_structure(graph: nx.Graph):
        """Creates a copy of a NetworkX graph, but only the graph structure (nodes and edges), without
        copying the node data.

        Args:
            graph (networkx.Graph): The graph to copy. Must not be None.

        Returns:
            networkx.Graph: A new graph with the same structure but empty node attributes.
        """
        assert graph is not None, "Graph object for copying must not be None"

        if isinstance(graph, nx.Graph):
            # This allows support for other networkx graph types, which all inherit from nx.Graph
            graph_type = type(graph)
            new_graph = graph_type()
        else:
            new_graph = nx.Graph()

        new_graph.add_nodes_from(graph.nodes)
        new_graph.add_edges_from(graph.edges)

        return new_graph

    @staticmethod
    def _truncate_id_if_uuid(id: Any):
        """Truncate an ID if it is a UUID, otherwise return the input unmodified.

        Args:
            id (Any): The ID to truncate.

        Returns:
            str: The truncated UUID.
            type(id): If the id is not a UUID, return the unmodified input.

        **Examples**

        >>> id = uuid.uuid4()
        >>> id # doctest: +SKIP
        UUID('2b03b2f5-5d36-4dfe-997e-01d2b01556c8')
        >>> QubitGraph._truncate_id_if_uuid(id) # doctest: +SKIP
        '2b03b2f5'

        >>> QubitGraph._truncate_id_if_uuid("abcdefghijkl")
        'abcdefghijkl'
        """
        if isinstance(id, uuid.UUID):
            return str(id).split("-", maxsplit=1)[0]

        return id

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
    def _warn_cyclically_nested_graph():
        """Emit a UserWarning when a cyclically nested graph structure was detected in a QubitGraph
        object.
        """
        warnings.warn(
            "A cyclically nested graph structure was detected in a QubitGraph object. QubitGraph "
            "objects should generally not be cyclically nested since such structures have neither "
            "well-defined root nor leaf nodes.",
            UserWarning,
        )

    @classmethod
    def _check_graph_type_supported_and_raise_or_warn(cls, candidate: Any):
        """Check that the input type is graph-like and raise a TypeError if not, and then check that
        the graph type is one that is supported and emit a UserWarning if not.

        The input is considered "graph-like" according to the definition in the
        ``QubitGraph._is_graph_like()`` method.

        Currently, QubitGraph only supports NetworkX graphs; specifically, graph objects that are
        instances of the `networkx.Graph
        <https://networkx.org/documentation/stable/reference/classes/graph.html>` class, or
        subclasses thereof.

        Note that other NetworkX graph types, including ``DiGraph``, ``MultiGraph`` and
        ``MultiDiGraph`` are all subclasses of the ``Graph`` class and are therefore permitted,
        although their usage is discouraged since they store additional information that is not used
        by QubitGraph.

        Args:
            candidate: The candidate graph object used for type-checking. This object must not be
                None.
        """
        assert candidate is not None, "Graph object used for type-checking must not be None"

        if not cls._is_graph_like(candidate):
            raise TypeError(
                "QubitGraph requires a graph-like input, i.e. an object having both a 'nodes' and "
                "an 'edges' attribute."
            )

        if not isinstance(candidate, nx.Graph):
            warnings.warn(
                f"QubitGraph expects an input graph of type 'networkx.Graph', but got "
                f"'{type(candidate).__name__}'. Using a graph of another type may result in "
                f"unexpected behaviour.",
                UserWarning,
            )

    @classmethod
    def _is_graph_like(cls, candidate: Any) -> bool:
        """Check if the input is a graph-like object.

        An object is considered "graph-like" if it has both a 'nodes' and an 'edges' attribute.

        Args:
            candidate: The candidate graph object used for type-checking. This object must not be
                None.
        """
        assert candidate is not None, "Graph object used for type-checking must not be None"

        return hasattr(candidate, "nodes") and hasattr(candidate, "edges")
