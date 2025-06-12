import uuid
from typing import Sequence, Union

import pydot

# pylint: disable=no-member


class DeviceNode(pydot.Node):
    """Node representing a quantum device in the graph."""

    def __init__(self, wires: Sequence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_shape("box")
        self.set_style("filled")
        self.set_fillcolor("cornsilk")
        self.set_color("cornsilk4")

        self.set_penwidth(2)
        self.set_fontname("Helvetica")

        self.wires = wires


class OperatorNode(pydot.Node):
    """Node representing a quantum operator in the graph."""

    _counter = 1

    def __init__(self, wires: Sequence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_shape(kwargs.get("shape", "ellipse"))
        self.set_style("filled")
        self.set_fillcolor(kwargs.get("fillcolor", "lightblue"))
        self.set_color(kwargs.get("color", "lightblue4"))
        self.set_penwidth(2)
        self.set_fontname("Helvetica")

        cur_name = self.get_name()
        new_name = f"{cur_name}{OperatorNode._counter}"
        OperatorNode._counter += 1
        self.set_name(new_name)

        # Store the wires associated with this operator
        self.wires = wires


class MeasurementNode(pydot.Node):
    """Node representing a measurement operation in the graph."""

    _counter = 1

    def __init__(self, wires: Sequence, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_shape("ellipse")
        self.set_style("filled")
        self.set_fillcolor("lightpink")
        self.set_color("lightpink4")
        self.set_penwidth(2)
        self.set_fontname("Helvetica")

        cur_name = self.get_name()
        new_name = f"{cur_name}{MeasurementNode._counter}"
        MeasurementNode._counter += 1
        self.set_name(new_name)

        # Store the wires associated with this measurement
        self.wires = wires


class ClassicalNode(pydot.Node):
    """Node representing a classical operation in the graph."""


class ControlFlowCluster(pydot.Cluster):
    """Cluster representing a control flow structure in the graph, such as a loop or conditional."""

    _counter = 1

    def __init__(self, info_label="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_penwidth(2)
        self.set_fontname("Helvetica")

        cur_name = self.get_name()
        new_name = f"{cur_name}{ControlFlowCluster._counter}"
        ControlFlowCluster._counter += 1
        self.set_name(new_name)

        # Increment the counter and use it for a unique node name
        unique_name = f"{new_name}_info_node"
        rank_subgraph = pydot.Subgraph()
        node = pydot.Node(
            unique_name,
            label=info_label,
            shape="rectangle",
            style="dashed",
            fontname="Helvetica",
            penwidth=2,
        )
        rank_subgraph.add_node(node)
        self.add_subgraph(rank_subgraph)
        self.add_node(node)


class QNodeCluster(pydot.Cluster):
    """Cluster representing a QNode in the graph"""

    _counter = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_penwidth(2)
        self.set_fontname("Helvetica")

        cur_name = self.get_name()
        new_name = f"{cur_name}{QNodeCluster._counter}"
        QNodeCluster._counter += 1
        self.set_name(new_name)


class PyDotGraphBuilder:
    """Object to build and store the graph representation of a captured PennyLane circuit."""

    def __init__(self, **attrs):
        super().__init__()

        # type: dict[Union[int, str], list[pydot.Node]]
        self.wires_to_nodes = {}

        self.graph = pydot.Dot(
            graph_type=attrs.pop("graph_type", "digraph"),
            rankdir=attrs.pop("rankdir", "TB"),
            **attrs,
        )

        self.current_cluster = self.graph

    def __copy__(self):

        new_builder = PyDotGraphBuilder()
        new_builder.graph = self.graph
        new_builder.current_cluster = self.current_cluster
        new_builder.wires_to_nodes = self.wires_to_nodes
        return new_builder

    def add_cluster_to_graph(
        self, cluster: pydot.Cluster, graph: Union[pydot.Graph, pydot.Cluster] = None
    ) -> None:
        """Adds a cluster to the main graph or another cluster."""
        if graph:
            graph.add_subgraph(cluster)
        else:
            self.current_cluster.add_subgraph(cluster)

    def add_quantum_node_to_graph(
        self,
        node: Union[OperatorNode, MeasurementNode, DeviceNode],
        cluster: pydot.Cluster = None,
        auto_connect: bool = True,
    ) -> None:
        """Adds a node to a cluster."""

        # Step 1: Add the node to the graph
        if cluster:
            cluster.add_node(node)
        else:
            self.current_cluster.add_node(node)

        if auto_connect:
            # Step 2: Connect the node to any previously seen nodes with the same wires
            if isinstance(node, DeviceNode):
                # Special node since it just needs to "register" all available wires
                # in the wires_to_nodes mapping
                self._connect_device_node(node)
                return
            self._connect_quantum_node(node)

    def add_classical_node_to_graph(
        self, node: ClassicalNode, cluster: pydot.Cluster = None
    ) -> None:
        """Adds a classical node to a cluster."""
        raise NotImplementedError("Classical nodes are not yet implemented in PyDotGraphBuilder.")

    def render_graph(self, output_filename: str, view: bool = False) -> None:
        """Renders the graph to a file."""
        self.graph.write(output_filename, format="png")
        if view:
            from PIL import Image

            img = Image.open(output_filename)
            img.show()

    def add_edge(self, from_node: pydot.Node, to_node: pydot.Node, **attrs) -> None:
        """Adds an edge between two nodes in the graph."""
        edge = pydot.Edge(from_node, to_node, **attrs)
        self.graph.add_edge(edge)

    def _connect_device_node(self, node: DeviceNode) -> None:
        """Connects a device node to the graph based on its wires."""

        for wire in node.wires:
            self.wires_to_nodes[wire] = [node]

    def _connect_quantum_node(self, node: Union[OperatorNode, MeasurementNode]) -> None:
        """Connects an operator node to the graph based on its wires."""

        # Sort strings first so we prioritize dynamic wires
        sorted_node_wires = sorted(node.wires, key=lambda x: (isinstance(x, int), x))

        color = "lightpink4" if isinstance(node, MeasurementNode) else "lightblue4"
        for wire in sorted_node_wires:
            if isinstance(wire, int):
                seen_wire_before = wire in self.wires_to_nodes
                if seen_wire_before:
                    # We've seen this wire before
                    # Connect all previous nodes with this wire to the current node
                    prev_nodes = self.wires_to_nodes[wire]
                    for prev_node in prev_nodes:
                        if prev_node != node and not self.graph.get_edge(
                            prev_node.get_name(), node.get_name()
                        ):
                            self.graph.add_edge(
                                pydot.Edge(prev_node, node, style="solid", color=color)
                            )
                        if isinstance(node, OperatorNode):
                            # If it's an operator node, we need to keep track of it
                            self.wires_to_nodes[wire] = [node]
                else:
                    if self.wires_to_nodes:
                        # Never seen this operator's wire before
                        # but we have seen other wires

                        # Look through all previously seen wires
                        # and connect them to this node if they are dynamic wires (string)
                        for different_wire, different_prev_nodes in self.wires_to_nodes.items():
                            if isinstance(different_wire, str):
                                for prev_node in different_prev_nodes:
                                    if prev_node != node and not self.graph.get_edge(
                                        prev_node.get_name(), node.get_name()
                                    ):  # Avoid self-loop
                                        self.graph.add_edge(
                                            pydot.Edge(prev_node, node, color=color, style="dashed")
                                        )

                    if isinstance(node, (OperatorNode, DeviceNode)):
                        # If this is the first time we see this wire, just store the node
                        self.wires_to_nodes[wire] = [node]
            else:
                # Encoutered a wire that is not an int
                # This will represent a dynamic wire that could be any value
                # Therefore, we need to connect all previous nodes to this one
                # with a dashed line to indicate uncertainty
                for _, prev_nodes in self.wires_to_nodes.items():
                    for prev_node in prev_nodes:
                        if prev_node != node and not self.graph.get_edge(
                            prev_node.get_name(), node.get_name()
                        ):
                            self.graph.add_edge(
                                pydot.Edge(prev_node, node, color=color, style="dashed")
                            )
                self.wires_to_nodes.clear()

                if isinstance(node, OperatorNode):
                    self.wires_to_nodes[wire] = [node]
