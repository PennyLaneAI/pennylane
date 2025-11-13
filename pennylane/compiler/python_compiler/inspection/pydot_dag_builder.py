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

from typing import Any

from .dag_builder import DAGBuilder

has_pydot = True
try:
    import pydot
except ImportError:
    has_pydot = False


class PyDotDAGBuilder(DAGBuilder):
    """A Directed Acyclic Graph builder for the PyDot backend."""

    def __init__(self) -> None:
        self.graph: pydot.Graph = pydot.Dot(
            graph_type="digraph", rankdir="TB", compound="true", strict=True
        )
        self._clusters: dict[str, pydot.Cluster] = {}
        self._clusters["__root__"] = self.graph

        _default_attrs = {"fontname": "Helvetica", "penwidth": 2}
        self._default_node_attrs: dict = {
            **_default_attrs,
            "shape": "ellipse",
            "style": "filled",
            "fillcolor": "lightblue",
            "color": "lightblue4",
            "penwidth": 3,
        }
        self._default_edge_attrs: dict = {"color": "lightblue4", "penwidth": 3}
        self._default_cluster_attrs: dict = {
            **_default_attrs,
            "shape": "rectangle",
            "style": "solid",
        }

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
        node_attrs = {**self._default_node_attrs, **node_attrs}
        node = pydot.Node(node_id, label=node_label, **node_attrs)
        parent_graph_id = "__root__" if parent_graph_id is None else parent_graph_id

        self._clusters[parent_graph_id].add_node(node)

    def add_edge(self, from_node_id: str, to_node_id: str, **edge_attrs: Any) -> None:
        """Add a single directed edge between nodes in the graph.

        Args:
            from_node_id (str): The unique ID of the source node.
            to_node_id (str): The unique ID of the destination node.
            **edge_attrs (Any): Any additional styling keyword arguments.

        """
        edge_attrs = {**self._default_edge_attrs, **edge_attrs}
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

        Args:
            cluster_id (str): Unique cluster ID to identify this cluster.
            cluster_label (str): The text to display on the cluster when rendered.
            parent_graph_id (str | None): Optional ID of the cluster this cluster belongs to.
            **cluster_attrs (Any): Any additional styling keyword arguments.

        """
        cluster_attrs = {**self._default_cluster_attrs, **cluster_attrs}
        cluster = pydot.Cluster(graph_name=cluster_id, **cluster_attrs)

        # Puts the label in a node within the cluster.
        # Ensures that any edges connecting nodes through the cluster
        # boundary don't block the label.
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

        self._clusters[cluster_id] = cluster

        parent_graph_id = "__root__" if parent_graph_id is None else parent_graph_id
        self._clusters[parent_graph_id].add_subgraph(cluster)

    def render(self, output_filename: str) -> None:
        """Render the graph to a file

        The implementation should ideally infer the output format
        (e.g., 'png', 'svg') from this filename's extension.

        Args:
            output_filename (str): Desired filename for the rendered graph.

        """
        format = output_filename.split(".")[-1].lower()
        if not format:
            format = "png"
            output_filename += ".png"

        self.graph.write(output_filename, format=format)
        print(f"Successfully rendered graph to: {output_filename}")

    def to_string(self) -> str:
        """Render the graph as a string.

        This is typically used to get the graph's representation in a standard string format like DOT.

        Returns:
            str: A string representation of the graph.
        """
        return self.graph.to_string()
