import abc
from typing import Any


class GraphBuilder(abc.ABC):
    """
    An Abstract Base Class that defines a generic interface for building
    a graph. An interpreter can use this interface without knowing the
    specifics of the underlying graphing library.
    """

    @abc.abstractmethod
    def add_cluster(self, cluster_id: str, label: str, **attrs: Any):
        """Adds a subgraph or cluster to the main graph."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_node(self, node_id: str, label: str, cluster_id: str = None, **attrs: Any):
        """Adds a node, optionally placing it inside a cluster."""
        raise NotImplementedError

    @abc.abstractmethod
    def add_edge(self, from_node_id: str, to_node_id: str, **attrs: Any):
        """Adds a directed edge between two nodes."""
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, output_filename: str, view: bool = False):
        """Renders the final graph to a file."""
        raise NotImplementedError
