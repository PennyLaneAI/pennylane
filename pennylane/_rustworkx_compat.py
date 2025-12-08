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
    Create a rustworkx PyGraph from an adjacency dictionary.
    
    This is a helper function for test compatibility, as networkx allows
    creating graphs like: nx.Graph({0: {1, 2}, 1: {0}, 2: {0}})
    
    Args:
        adj_dict (dict): Adjacency dictionary where keys are nodes and values are 
                        sets/tuples/lists of neighbors
        
    Returns:
        rx.PyGraph: A rustworkx undirected graph
    """
    g = rx.PyGraph()
    node_to_idx = {}
    
    # Add all nodes first
    for node in adj_dict.keys():
        if node not in node_to_idx:
            idx = g.add_node(node)
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
                neighbor_idx = g.add_node(neighbor)
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


# Graph class that wraps PyGraph for networkx-like initialization
def Graph(incoming_graph_data=None, multigraph=True):
    """
    Create a rustworkx PyGraph with networkx-like initialization support.
    
    Examples:
        >>> g = Graph()  # Empty graph
        >>> g = Graph([(0, 1), (1, 2)])  # Graph from edge list
    
    Args:
        incoming_graph_data: Optional edge list to initialize graph
        multigraph: Whether to create a multigraph
        
    Returns:
        rx.PyGraph: A rustworkx graph
    """
    g = rx.PyGraph(multigraph=multigraph)
    
    if incoming_graph_data is not None:
        if isinstance(incoming_graph_data, list):
            # Assume it's an edge list
            nodes = set()
            for edge in incoming_graph_data:
                if len(edge) >= 2:
                    nodes.add(edge[0])
                    nodes.add(edge[1])
            
            # Add nodes first
            node_to_idx = {}
            for node in sorted(nodes):
                idx = g.add_node(node)
                node_to_idx[node] = idx
            
            # Add edges
            for edge in incoming_graph_data:
                if len(edge) >= 2:
                    u_idx = node_to_idx[edge[0]]
                    v_idx = node_to_idx[edge[1]]
                    edge_data = edge[2] if len(edge) > 2 else None
                    g.add_edge(u_idx, v_idx, edge_data)
        else:
            raise ValueError("Graph initialization only supports edge lists")
    
    return g
