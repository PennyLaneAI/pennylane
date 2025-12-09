# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Compatibility layer for rustworkx to provide networkx-like MultiDiGraph API.
"""

import rustworkx as rx
import numpy as np


class MultiDiGraph:
    """
    A directed multigraph implementation using rustworkx.PyDiGraph as backend.
    
    This class provides a networkx.MultiDiGraph-like interface for compatibility
    with existing PennyLane code while using rustworkx as the underlying graph library.
    """

    def __init__(self, incoming_graph_data=None):
        self._graph = rx.PyDiGraph(multigraph=True)
        self._node_to_index = {}  # Map from node object to integer index
        self._index_to_node = {}  # Map from integer index to node object
        
        # Support initialization from edge list like networkx
        if incoming_graph_data is not None:
            if isinstance(incoming_graph_data, list):
                # Assume it's an edge list: [(u, v), (u, v, data), ...]
                self.add_edges_from(incoming_graph_data)
            else:
                raise ValueError("MultiDiGraph initialization only supports edge lists")

    def add_node(self, node, **attr):
        """Add a single node and optionally attach attributes."""
        if node not in self._node_to_index:
            idx = self._graph.add_node((node, attr))
            self._node_to_index[node] = idx
            self._index_to_node[idx] = node

    def add_nodes_from(self, nodes):
        """Add multiple nodes."""
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u, v, key=None, **attr):
        """Add an edge between u and v with optional attributes."""
        # Ensure nodes exist
        if u not in self._node_to_index:
            self.add_node(u)
        if v not in self._node_to_index:
            self.add_node(v)

        u_idx = self._node_to_index[u]
        v_idx = self._node_to_index[v]
        
        edge_data = attr.copy()
        if key is not None:
            edge_data['key'] = key
            
        self._graph.add_edge(u_idx, v_idx, edge_data)

    def add_edges_from(self, edges):
        """Add multiple edges."""
        for edge in edges:
            if len(edge) == 2:
                u, v = edge
                self.add_edge(u, v)
            elif len(edge) == 3:
                u, v, data = edge
                if isinstance(data, dict):
                    self.add_edge(u, v, **data)
                else:
                    self.add_edge(u, v, key=data)

    def remove_edge(self, u, v, key=None):
        """Remove an edge between u and v."""
        u_idx = self._node_to_index[u]
        v_idx = self._node_to_index[v]
        
        # Find and remove the specific edge
        edge_list = self._graph.get_all_edge_data(u_idx, v_idx)
        for edge_idx, edge_data in enumerate(edge_list):
            if key is None or edge_data.get('key') == key:
                self._graph.remove_edge(u_idx, v_idx)
                break

    def remove_node(self, node):
        """Remove a node."""
        if node in self._node_to_index:
            idx = self._node_to_index[node]
            self._graph.remove_node(idx)
            del self._index_to_node[idx]
            del self._node_to_index[node]

    def has_node(self, node):
        """Return True if the graph contains the node."""
        return node in self._node_to_index

    def nodes(self, data=False):
        """Return a view of nodes. If data is True or a string, include node attributes."""
        if data is False:
            return list(self._node_to_index.keys())
        elif data is True:
            result = []
            for idx in self._graph.node_indices():
                if idx in self._index_to_node:
                    node = self._index_to_node[idx]
                    node_data, attrs = self._graph[idx]
                    result.append((node, attrs))
            return result
        else:  # data is a string key
            result = []
            for idx in self._graph.node_indices():
                if idx in self._index_to_node:
                    node = self._index_to_node[idx]
                    node_data, attrs = self._graph[idx]
                    result.append((node, attrs.get(data)))
            return result

    def edges(self, data=False):
        """Return a view of edges. If data is True, include edge attributes."""
        result = []
        for u_idx in self._graph.node_indices():
            if u_idx not in self._index_to_node:
                continue
            u = self._index_to_node[u_idx]
            for v_idx in self._graph.successor_indices(u_idx):
                if v_idx not in self._index_to_node:
                    continue
                v = self._index_to_node[v_idx]
                edge_list = self._graph.get_all_edge_data(u_idx, v_idx)
                for edge_data in edge_list:
                    if data is False:
                        result.append((u, v))
                    elif data is True:
                        result.append((u, v, edge_data))
                    else:  # data is a key
                        result.append((u, v, edge_data.get(data)))
        return result

    def copy(self):
        """Return a copy of the graph."""
        new_graph = MultiDiGraph()
        new_graph._graph = self._graph.copy()
        new_graph._node_to_index = self._node_to_index.copy()
        new_graph._index_to_node = self._index_to_node.copy()
        return new_graph

    def subgraph(self, nodes):
        """Return the subgraph induced by the specified nodes."""
        node_indices = [self._node_to_index[n] for n in nodes if n in self._node_to_index]
        rx_subgraph = self._graph.subgraph(node_indices)
        
        new_graph = MultiDiGraph()
        new_graph._graph = rx_subgraph
        
        # Rebuild the node mappings for the subgraph
        for idx in rx_subgraph.node_indices():
            node_data, _ = rx_subgraph[idx]
            node = node_data  # The node object is stored as first element
            new_graph._node_to_index[node] = idx
            new_graph._index_to_node[idx] = node
            
        return new_graph

    def get_edge_data(self, u, v, key=None, default=None):
        """Return the attribute dictionary associated with edge (u, v)."""
        if u not in self._node_to_index or v not in self._node_to_index:
            return default
            
        u_idx = self._node_to_index[u]
        v_idx = self._node_to_index[v]
        
        try:
            edge_list = self._graph.get_all_edge_data(u_idx, v_idx)
            if not edge_list:
                return default
            if key is None:
                return edge_list[0] if edge_list else default
            for edge_data in edge_list:
                if edge_data.get('key') == key:
                    return edge_data
            return default
        except rx.NoEdgeBetweenNodes:
            return default

    @property
    def succ(self):
        """Graph adjacency object holding the successors of each node."""
        class SuccDict:
            def __init__(self, graph_wrapper):
                self._graph_wrapper = graph_wrapper
                
            def __getitem__(self, node):
                if node not in self._graph_wrapper._node_to_index:
                    return {}
                u_idx = self._graph_wrapper._node_to_index[node]
                successors = {}
                for v_idx in self._graph_wrapper._graph.successor_indices(u_idx):
                    if v_idx in self._graph_wrapper._index_to_node:
                        v = self._graph_wrapper._index_to_node[v_idx]
                        successors[v] = {}
                return successors
                
        return SuccDict(self)


# Compatibility functions

def weakly_connected_components(G):
    """
    Return weakly connected components of a directed graph.
    
    Args:
        G (MultiDiGraph): A rustworkx-based MultiDiGraph
        
    Yields:
        set: A set of nodes for each weakly connected component
    """
    components = rx.weakly_connected_components(G._graph)
    for component_indices in components:
        yield {G._index_to_node[idx] for idx in component_indices if idx in G._index_to_node}


def has_path(G, source, target):
    """
    Returns True if there exists a path from source to target in graph G.
    
    Args:
        G (MultiDiGraph): A rustworkx-based MultiDiGraph
        source: Source node
        target: Target node
        
    Returns:
        bool: True if a path exists
    """
    if source not in G._node_to_index or target not in G._node_to_index:
        return False
    
    source_idx = G._node_to_index[source]
    target_idx = G._node_to_index[target]
    
    return rx.digraph_has_path(G._graph, source_idx, target_idx)


def number_of_selfloops(G):
    """
    Returns the number of selfloop edges in the graph.
    
    Args:
        G (MultiDiGraph): A rustworkx-based MultiDiGraph
        
    Returns:
        int: Number of self-loops
    """
    count = 0
    for u, v in G.edges():
        if u == v:
            count += 1
    return count


# Note: For to_pydot and steiner_tree, these would need to be implemented
# or we'd need to convert to networkx temporarily. Let's implement them as needed.

def descendants(G, source):
    """
    Returns all nodes reachable from source in G.
    
    Args:
        G (MultiDiGraph): A rustworkx-based MultiDiGraph
        source: Source node
        
    Returns:
        set: Set of nodes reachable from source
    """
    if source not in G._node_to_index:
        return set()
    
    source_idx = G._node_to_index[source]
    
    # Use BFS/DFS to find all descendants
    try:
        descendants_indices = rx.descendants(G._graph, source_idx)
        return {G._index_to_node[idx] for idx in descendants_indices if idx in G._index_to_node}
    except AttributeError:
        # If rx.descendants doesn't exist, implement manually
        visited = set()
        stack = [source_idx]
        
        while stack:
            node_idx = stack.pop()
            if node_idx in visited:
                continue
            visited.add(node_idx)
            
            for succ_idx in G._graph.successor_indices(node_idx):
                if succ_idx not in visited:
                    stack.append(succ_idx)
        
        visited.discard(source_idx)  # Don't include source itself
        return {G._index_to_node[idx] for idx in visited if idx in G._index_to_node}


def graph_from_adj_dict(adj_dict):
    """
    Create a rustworkx CompatPyGraph from an adjacency dictionary.
    
    This is a helper function for test compatibility, as networkx allows
    creating graphs like: nx.Graph({0: {1, 2}, 1: {0}, 2: {0}})
    
    Args:
        adj_dict (dict): Adjacency dictionary where keys are nodes and values are 
                        sets/tuples/lists of neighbors
        
    Returns:
        CompatPyGraph: A rustworkx undirected graph with networkx-like API
    """
    g = CompatPyGraph()
    node_to_idx = {}
    
    # Add all nodes first - store the label in the node data dict
    for node in adj_dict.keys():
        if node not in node_to_idx:
            idx = g.add_node({"_label": node})
            node_to_idx[node] = idx
    
    # Add edges (avoiding duplicates in undirected graph)
    edges_added = set()
    for node, neighbors in adj_dict.items():
        node_idx = node_to_idx[node]
        # Handle sets, tuples, and lists of neighbors
        if not isinstance(neighbors, (set, list, tuple)):
            neighbors = [neighbors]
        for neighbor in neighbors:
            if neighbor not in node_to_idx:
                neighbor_idx = g.add_node({"_label": neighbor})
                node_to_idx[neighbor] = neighbor_idx
            else:
                neighbor_idx = node_to_idx[neighbor]
            
            # Create a sorted edge tuple to avoid duplicates
            edge_tuple = tuple(sorted([node_idx, neighbor_idx]))
            if edge_tuple not in edges_added:
                g.add_edge(node_idx, neighbor_idx, None)
                edges_added.add(edge_tuple)
    
    return g


def digraph_from_adj_dict(adj_dict):
    """
    Create a rustworkx PyDiGraph from an adjacency dictionary.
    
    Args:
        adj_dict (dict): Adjacency dictionary where keys are nodes and values are sets of neighbors
        
    Returns:
        rx.PyDiGraph: A rustworkx directed graph
    """
    g = rx.PyDiGraph()
    node_to_idx = {}
    
    # Add all nodes first
    for node in adj_dict.keys():
        if node not in node_to_idx:
            idx = g.add_node(node)
            node_to_idx[node] = idx
    
    # Add edges
    for node, neighbors in adj_dict.items():
        node_idx = node_to_idx[node]
        for neighbor in neighbors:
            if neighbor not in node_to_idx:
                neighbor_idx = g.add_node(neighbor)
                node_to_idx[neighbor] = neighbor_idx
            else:
                neighbor_idx = node_to_idx[neighbor]
            
            g.add_edge(node_idx, neighbor_idx, None)
    
    return g


def _is_numpy_array_sequence(data):
    """Check if data is a tuple/list of numpy arrays representing edges."""
    if not isinstance(data, (tuple, list)):
        return False
    if len(data) == 0:
        return False
    # Check if all elements are numpy arrays or array-like with at least 2 elements
    for item in data:
        if hasattr(item, '__len__') and hasattr(item, '__getitem__'):
            if len(item) < 2:
                return False
        else:
            return False
    return True


def _edges_from_numpy_arrays(data):
    """
    Convert a tuple/list of numpy arrays to edge tuples.
    
    Each array should contain exactly 2 elements: [source, target]
    
    Args:
        data: Tuple or list of numpy arrays/sequences
        
    Returns:
        list: List of (source, target) edge tuples
    """
    edges = []
    for item in data:
        if len(item) >= 2:
            edges.append((item[0], item[1]))
    return edges


class NodeDict:
    """
    A dict-like wrapper for rustworkx PyGraph nodes that provides networkx-like access.
    
    This allows code like `graph.nodes[key]["attr"]` to work with rustworkx graphs.
    
    Supports both integer index-based access and label-based access where labels
    are either the node data values (for simple graphs) or the "_label" key in
    node data dicts (for networkx-compatible graphs).
    """
    
    def __init__(self, graph, label_to_index=None):
        self._graph = graph
        # Build mapping from node labels to indices for label-based access
        if label_to_index is not None:
            self._label_to_index = label_to_index
        else:
            self._label_to_index = {}
            for idx in graph.node_indices():
                node_data = graph[idx]
                # If node data is a dict with "_label" key, use that as the label
                if isinstance(node_data, dict) and "_label" in node_data:
                    try:
                        self._label_to_index[node_data["_label"]] = idx
                    except TypeError:
                        pass  # Not hashable
                # If node data is a hashable non-dict value, use it as the label
                elif node_data is not None and not isinstance(node_data, dict):
                    try:
                        self._label_to_index[node_data] = idx
                    except TypeError:
                        pass  # Not hashable
    
    def _resolve_key(self, key):
        """Resolve a key to an integer index.
        
        First checks if key is a known label, then checks if it's a valid index.
        This allows integer labels that differ from their indices.
        """
        # First check if key is a known label
        if key in self._label_to_index:
            return self._label_to_index[key]
        # Then check if it's a valid integer index
        if isinstance(key, int) and key in self._graph.node_indices():
            return key
        raise KeyError(f"Node {key} not found")
    
    def __getitem__(self, key):
        """Get node data by index or label."""
        idx = self._resolve_key(key)
        try:
            return self._graph[idx]
        except IndexError as e:
            raise KeyError(f"Node {key} not found") from e
    
    def __setitem__(self, key, value):
        """Set node data by index or label."""
        idx = self._resolve_key(key)
        self._graph[idx] = value
    
    def __len__(self):
        """Return the number of nodes."""
        return len(self._graph.node_indices())
    
    def __iter__(self):
        """Iterate over node labels (or indices if no labels)."""
        if self._label_to_index:
            return iter(self._label_to_index.keys())
        return iter(self._graph.node_indices())
    
    def __contains__(self, key):
        """Check if node exists by index or label."""
        # Check label first, then index
        if key in self._label_to_index:
            return True
        if isinstance(key, int):
            return key in self._graph.node_indices()
        return False


class EdgeList:
    """
    A list-like wrapper for rustworkx PyGraph edges that provides networkx-like access.
    
    This allows code like `for edge in graph.edges` to work with rustworkx graphs.
    """
    
    def __init__(self, graph):
        self._graph = graph
    
    def _index_to_label(self, idx):
        """Convert an integer index to its label (if any)."""
        node_data = self._graph[idx]
        if isinstance(node_data, dict) and "_label" in node_data:
            return node_data["_label"]
        return node_data if node_data is not None else idx
    
    def __iter__(self):
        """Iterate over edges as (source_label, target_label) tuples."""
        for u_idx, v_idx in self._graph.edge_list():
            yield (self._index_to_label(u_idx), self._index_to_label(v_idx))
    
    def __len__(self):
        """Return the number of edges."""
        return self._graph.num_edges()
    
    def __contains__(self, edge):
        """Check if edge exists."""
        if len(edge) >= 2:
            return self._graph.has_edge(edge[0], edge[1])
        return False


class CompatPyGraph(rx.PyGraph):
    """
    A rustworkx PyGraph subclass that provides networkx-like node access via a `.nodes` property.
    
    This allows existing code that uses `graph.nodes[key]["attr"]` to work with rustworkx.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._edges_view = None
        self._label_to_index = None
    
    def _get_label_to_index(self):
        """Build and return a mapping from labels to indices."""
        if self._label_to_index is None:
            self._label_to_index = {}
            for idx in self.node_indices():
                node_data = self[idx]
                if isinstance(node_data, dict) and "_label" in node_data:
                    try:
                        self._label_to_index[node_data["_label"]] = idx
                    except TypeError:
                        pass
                elif node_data is not None and not isinstance(node_data, dict):
                    try:
                        self._label_to_index[node_data] = idx
                    except TypeError:
                        pass
        return self._label_to_index
    
    def _resolve_to_index(self, node):
        """Resolve a node label or index to an integer index."""
        if isinstance(node, int):
            return node
        label_map = self._get_label_to_index()
        if node in label_map:
            return label_map[node]
        raise KeyError(f"Node {node} not found")
    
    def _index_to_label(self, idx):
        """Convert an integer index to its label (if any)."""
        node_data = self[idx]
        if isinstance(node_data, dict) and "_label" in node_data:
            return node_data["_label"]
        return node_data if node_data is not None else idx
    
    def neighbors(self, node):
        """Get neighbors of a node, accepting either index or label.
        
        Returns neighbor labels (not indices) for consistency with networkx.
        """
        idx = self._resolve_to_index(node)
        for neighbor_idx in super().neighbors(idx):
            yield self._index_to_label(neighbor_idx)
    
    @property
    def adj(self):
        """Return an adjacency dict view for networkx compatibility.
        
        Returns a dict where keys are node labels and values are dicts of neighbor labels.
        This mimics networkx's adj property.
        """
        result = {}
        for idx in self.node_indices():
            label = self._index_to_label(idx)
            neighbors_dict = {}
            for neighbor_idx in super().neighbors(idx):
                neighbor_label = self._index_to_label(neighbor_idx)
                neighbors_dict[neighbor_label] = {}
            result[label] = neighbors_dict
        return result
    
    def __iter__(self):
        """Iterate over node labels (like networkx)."""
        for idx in self.node_indices():
            yield self._index_to_label(idx)
    
    @property  
    def nodes(self):
        """Return a dict-like view of nodes.
        
        Note: Creates a fresh view each time to ensure label mappings are current.
        """
        return NodeDict(self)
    
    @property
    def edges(self):
        """Return a list-like view of edges."""
        if self._edges_view is None:
            self._edges_view = EdgeList(self)
        return self._edges_view
        return self._edges_view


class Graph:
    """
    A networkx.Graph-like wrapper around rustworkx.PyGraph.
    
    Provides a more familiar API for users coming from networkx.
    
    Supported initialization formats:
    - Empty graph: Graph()
    - Edge list: Graph([(0, 1), (1, 2)])
    - Edge list with data: Graph([(0, 1, {'weight': 1}), (1, 2, {})])
    - Numpy array edges: Graph((np.array([0, 1]), np.array([1, 2])))
    
    Examples:
        >>> g = Graph()  # Empty graph
        >>> g = Graph([(0, 1), (1, 2)])  # Graph from edge list
        >>> g = Graph((np.array([0, 1]), np.array([1, 2])))  # From numpy arrays
    """
    
    def __init__(self, incoming_graph_data=None):
        self._graph = rx.PyGraph(multigraph=False)
        self._node_to_index = {}  # Map from node object to integer index
        self._index_to_node = {}  # Map from integer index to node object
        
        if incoming_graph_data is not None:
            if isinstance(incoming_graph_data, list):
                # Standard edge list: [(0, 1), (1, 2), ...]
                self.add_edges_from(incoming_graph_data)
            elif _is_numpy_array_sequence(incoming_graph_data):
                # Tuple/list of numpy arrays: (np.array([0, 1]), np.array([1, 2]))
                edges = _edges_from_numpy_arrays(incoming_graph_data)
                self.add_edges_from(edges)
            elif isinstance(incoming_graph_data, dict):
                # Adjacency dictionary: {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
                self._init_from_adj_dict(incoming_graph_data)
            else:
                raise ValueError(
                    f"Graph initialization does not support type {type(incoming_graph_data)}. "
                    "Supported formats: edge list [(u, v), ...], "
                    "numpy array tuple (np.array([u, v]), ...), "
                    "or adjacency dict {u: {v, w}, ...}"
                )
    
    def _init_from_adj_dict(self, adj_dict):
        """Initialize from an adjacency dictionary."""
        # Add all nodes first
        for node in adj_dict.keys():
            self.add_node(node)
        
        # Add edges (avoiding duplicates in undirected graph)
        edges_added = set()
        for node, neighbors in adj_dict.items():
            node_idx = self._node_to_index[node]
            # Handle sets, tuples, and lists of neighbors
            if not isinstance(neighbors, (set, list, tuple)):
                neighbors = [neighbors]
            for neighbor in neighbors:
                if neighbor not in self._node_to_index:
                    self.add_node(neighbor)
                neighbor_idx = self._node_to_index[neighbor]
                
                # Create a sorted edge tuple to avoid duplicates
                edge_tuple = tuple(sorted([node_idx, neighbor_idx]))
                if edge_tuple not in edges_added:
                    self._graph.add_edge(node_idx, neighbor_idx, None)
                    edges_added.add(edge_tuple)
    
    def add_node(self, node, **attr):
        """Add a single node."""
        if node not in self._node_to_index:
            idx = self._graph.add_node(node)
            self._node_to_index[node] = idx
            self._index_to_node[idx] = node
    
    def add_nodes_from(self, nodes):
        """Add multiple nodes."""
        for node in nodes:
            if isinstance(node, tuple) and len(node) == 2:
                # (node, attr_dict) format
                self.add_node(node[0])
            else:
                self.add_node(node)
    
    def add_edge(self, u, v, **attr):
        """Add an edge between u and v."""
        if u not in self._node_to_index:
            self.add_node(u)
        if v not in self._node_to_index:
            self.add_node(v)
        
        u_idx = self._node_to_index[u]
        v_idx = self._node_to_index[v]
        
        # Check if edge already exists (undirected)
        if not self._graph.has_edge(u_idx, v_idx):
            self._graph.add_edge(u_idx, v_idx, attr if attr else None)
    
    def add_edges_from(self, edges):
        """Add multiple edges."""
        for edge in edges:
            if len(edge) == 2:
                u, v = edge
                self.add_edge(u, v)
            elif len(edge) == 3:
                u, v, data = edge
                if isinstance(data, dict):
                    self.add_edge(u, v, **data)
                else:
                    self.add_edge(u, v)
    
    def nodes(self, data=False):
        """Return nodes."""
        if data:
            return [(self._index_to_node[idx], {}) for idx in self._graph.node_indices()]
        return list(self._node_to_index.keys())
    
    def edges(self, data=False):
        """Return edges."""
        result = []
        seen = set()
        for u_idx, v_idx, edge_data in self._graph.weighted_edge_list():
            # Normalize edge representation for undirected graph
            edge_key = tuple(sorted([u_idx, v_idx]))
            if edge_key in seen:
                continue
            seen.add(edge_key)
            
            u = self._index_to_node[u_idx]
            v = self._index_to_node[v_idx]
            if data:
                result.append((u, v, edge_data if edge_data else {}))
            else:
                result.append((u, v))
        return result
    
    def __len__(self):
        """Return number of nodes."""
        return len(self._node_to_index)
    
    def __iter__(self):
        """Iterate over nodes."""
        return iter(self._node_to_index.keys())
    
    def __contains__(self, node):
        """Check if node is in graph."""
        return node in self._node_to_index
    
    def number_of_nodes(self):
        """Return the number of nodes."""
        return len(self._node_to_index)
    
    def number_of_edges(self):
        """Return the number of edges."""
        return self._graph.num_edges()
    
    def has_edge(self, u, v):
        """Return True if edge (u, v) exists."""
        if u not in self._node_to_index or v not in self._node_to_index:
            return False
        u_idx = self._node_to_index[u]
        v_idx = self._node_to_index[v]
        return self._graph.has_edge(u_idx, v_idx)
    
    def neighbors(self, node):
        """Return neighbors of node."""
        if node not in self._node_to_index:
            raise KeyError(f"Node {node} not in graph")
        node_idx = self._node_to_index[node]
        return [self._index_to_node[idx] for idx in self._graph.neighbors(node_idx)]
    
    def degree(self, node=None):
        """Return degree of node(s)."""
        if node is not None:
            if node not in self._node_to_index:
                raise KeyError(f"Node {node} not in graph")
            node_idx = self._node_to_index[node]
            return self._graph.degree(node_idx)
        # Return dict of all degrees
        return {self._index_to_node[idx]: self._graph.degree(idx) 
                for idx in self._graph.node_indices()}
    
    def copy(self):
        """Return a copy of the graph."""
        new_graph = Graph()
        new_graph._graph = self._graph.copy()
        new_graph._node_to_index = self._node_to_index.copy()
        new_graph._index_to_node = self._index_to_node.copy()
        return new_graph

    def edge_list(self):
        """Return list of edges as (source_idx, target_idx) tuples.
        
        This provides compatibility with rustworkx.PyGraph.edge_list().
        """
        return list(self._graph.edge_list())

    def node_indices(self):
        """Return list of node indices.
        
        This provides compatibility with rustworkx.PyGraph.node_indices().
        """
        return list(self._graph.node_indices())

    def __getitem__(self, idx):
        """Get node data at index.
        
        This provides compatibility with rustworkx.PyGraph.__getitem__().
        """
        return self._graph[idx]
