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
"""File that defines the PyDotDAGBuilder subclass of DAGBuilder."""

from collections import ChainMap
from typing import Any

from .dag_builder import DAGBuilder

has_pydot = True
try:
    import pydot
except ImportError:
    has_pydot = False


class PyDotDAGBuilder(DAGBuilder):
    """A Directed Acyclic Graph builder for the PyDot backend."""

    def __init__(
        self,
        default_attrs: dict | None = None,
        default_node_attrs: dict | None = None,
        default_edge_attrs: dict | None = None,
        default_cluster_attrs: dict | None = None,
    ) -> None:
        """Initialize PyDotDAGBuilder instance.

        Args:
            default_attrs (dict | None): Default attributes to be used for all elements (nodes, edges, clusters) in the graph.
            default_node_attrs (dict | None): Default attributes for a node.
            default_edge_attrs (dict | None): Default attributes for an edge.
            default_cluster_attrs (dict | None): Default attributes for a cluster.

        """
        # Initialize the pydot graph:
        # - graph_type="digraph": Create a directed graph (edges have arrows).
        # - rankdir="TB": Set layout direction from Top to Bottom.
        # - compound="true": Allow edges to connect directly to clusters/subgraphs.
        # - strict=True": Prevent duplicate edges (e.g., A -> B added twice).
        self.graph: pydot.Dot = pydot.Dot(
            graph_type="digraph", rankdir="TB", compound="true", strict=True
        )
        # Create context variable to map IDs to Graph objects
        self._subgraphs: dict[str, pydot.Graph] = {}
        self._subgraphs["__base__"] = self.graph

        _default_attrs: dict = (
            {"fontname": "Helvetica", "penwidth": 2} if default_attrs is None else default_attrs
        )
        self._default_node_attrs: dict = (
            {
                **_default_attrs,
                "shape": "ellipse",
                "style": "filled",
                "fillcolor": "lightblue",
                "color": "lightblue4",
                "penwidth": 3,
            }
            if default_node_attrs is None
            else default_node_attrs
        )
        self._default_edge_attrs: dict = (
            {
                "color": "lightblue4",
                "penwidth": 3,
            }
            if default_edge_attrs is None
            else default_edge_attrs
        )
        self._default_cluster_attrs: dict = (
            {
                **_default_attrs,
                "shape": "rectangle",
                "style": "solid",
            }
            if default_cluster_attrs is None
            else default_cluster_attrs
        )

    def add_node(
        self,
        node_id: str,
        node_label: str,
        parent_graph_id: str | None = None,
        **node_attrs: Any,
    ) -> None:
        """Add a single node to the graph.

        Args:
            node_id (str): Unique node ID to identify this node.
            node_label (str): The text to display on the node when rendered.
            parent_graph_id (str | None): Optional ID of the cluster this node belongs to.
            **node_attrs (Any): Any additional styling keyword arguments.

        """
        node_attrs = ChainMap(node_attrs, self._default_node_attrs)
        node = pydot.Node(node_id, label=node_label, **node_attrs)
        parent_graph_id = "__base__" if parent_graph_id is None else parent_graph_id

        self._subgraphs[parent_graph_id].add_node(node)

    def add_edge(self, from_node_id: str, to_node_id: str, **edge_attrs: Any) -> None:
        """Add a single directed edge between nodes in the graph.

        Args:
            from_node_id (str): The unique ID of the source node.
            to_node_id (str): The unique ID of the destination node.
            **edge_attrs (Any): Any additional styling keyword arguments.

        """
        edge_attrs = ChainMap(edge_attrs, self._default_edge_attrs)
        edge = pydot.Edge(from_node_id, to_node_id, **edge_attrs)
        self.graph.add_edge(edge)

    def add_cluster(
        self,
        cluster_id: str,
        cluster_label: str,
        parent_graph_id: str | None = None,
        **cluster_attrs: Any,
    ) -> None:
        """Add a single cluster to the graph.

        A cluster is a specific type of subgraph where the nodes and edges contained
        within it are visually and logically grouped.

        Args:
            cluster_id (str): Unique cluster ID to identify this cluster.
            cluster_label (str): The text to display on the cluster when rendered.
            parent_graph_id (str | None): Optional ID of the cluster this cluster belongs to.
            **cluster_attrs (Any): Any additional styling keyword arguments.

        """
        cluster_attrs = ChainMap(cluster_attrs, self._default_cluster_attrs)
        cluster = pydot.Cluster(graph_name=cluster_id, **cluster_attrs)

        # Puts the label in a node within the cluster.
        # Ensures that any edges connecting nodes through the cluster
        # boundary don't block the label.
        # ┌───────────┐
        # │ ┌───────┐ │
        # │ │ label │ │
        # │ └───────┘ │
        # │           │
        # └───────────┘
        if cluster_label:
            node_id = f"{cluster_id}_info_node"
            rank_subgraph = pydot.Subgraph()
            node = pydot.Node(
                node_id,
                label=cluster_label,
                shape="rectangle",
                style="dashed",
                fontname="Helvetica",
                penwidth=2,
            )
            rank_subgraph.add_node(node)
            cluster.add_subgraph(rank_subgraph)
            cluster.add_node(node)

        self._subgraphs[cluster_id] = cluster

        parent_graph_id = "__base__" if parent_graph_id is None else parent_graph_id
        self._subgraphs[parent_graph_id].add_subgraph(cluster)

    def render(self, output_filename: str) -> None:
        """Render the graph to a file

        The implementation should ideally infer the output format
        (e.g., 'png', 'svg') from this filename's extension.

        Args:
            output_filename (str): Desired filename for the rendered graph. If no file extension is
                provided, it will default to a `.png` file.

        """
        format = output_filename.split(".")[-1].lower()
        if not format:
            format = "png"
            output_filename += ".png"

        self.graph.write(output_filename, format=format)

    def to_string(self) -> str:
        """Render the graph as a string.

        This is typically used to get the graph's representation in a standard string format like DOT.

        Returns:
            str: A string representation of the graph.
        """
        return self.graph.to_string()
