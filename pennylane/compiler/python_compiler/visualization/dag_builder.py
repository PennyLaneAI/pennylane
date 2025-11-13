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
"""File that defines the DAGBuilder abstract base class."""

import abc
from typing import Any


class DAGBuilder(abc.ABC):
    """An abstract base class for building Directed Acyclic Graphs (DAGs).

    This class provides a simple interface with three core methods (`add_node`, `add_edge` and `add_cluster`). You can override these methods to implement any backend, like `pydot` or `graphviz` or even `matplotlib`.

    Outputting your graph can be done by overriding `render` and `to_string`.
    """

    @abc.abstractmethod
    def add_node(
        self, node_id: str, node_label: str, parent_graph_id: str | None = None, **node_attrs: Any
    ) -> None:
        """Add a single node to the graph.

        Args:
            node_id (str): Unique node ID to identify this node.
            node_label (str): The text to display on the node when rendered.
            parent_graph_id (str | None): Optional ID of the cluster this node belongs to.
            **node_attrs (Any): Any additional styling keyword arguments.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_edge(self, from_node_id: str, to_node_id: str, **edge_attrs: Any) -> None:
        """Add a single directed edge between nodes in the graph.

        Args:
            from_node_id (str): The unique ID of the source node.
            to_node_id (str): The unique ID of the destination node.
            **edge_attrs (Any): Any additional styling keyword arguments.

        """
        raise NotImplementedError

    @abc.abstractmethod
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
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, output_filename: str) -> None:
        """Render the graph to a file.

        The implementation should ideally infer the output format
        (e.g., 'png', 'svg') from this filename's extension.

        Args:
            output_filename (str): Desired filename for the rendered graph.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_string(self) -> str:
        """Render the graph as a string.

        This is typically used to get the graph's representation in a standard string format like DOT.

        Returns:
            str: A string representation of the graph.
        """
        raise NotImplementedError
