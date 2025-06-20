import uuid
from copy import copy
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
        new_name = f"{cur_name}_control_flow_{ControlFlowCluster._counter}"
        ControlFlowCluster._counter += 1
        self.set_name(new_name)

        # Increment the counter and use it for a unique node name
        if info_label:
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
        new_name = f"{cur_name}_qnode_{QNodeCluster._counter}"
        QNodeCluster._counter += 1
        self.set_name(new_name)


class ControlCluster(pydot.Cluster):
    """Cluster representing a control flow structure in the graph, such as a loop or conditional."""

    _counter = 1

    def __init__(self, *args, info_label="", **kwargs):
        super().__init__(*args, **kwargs)
        self.set_penwidth(2)
        self.set_fontname("Helvetica")
        self.set_style("filled")
        self.set_fillcolor(kwargs.get("fillcolor", "darkgoldenrod1"))
        self.set_color(kwargs.get("color", "darkgoldenrod3"))

        cur_name = self.get_name()
        new_name = f"{cur_name}_control_{ControlCluster._counter}"
        ControlCluster._counter += 1
        self.set_name(new_name)

        # Increment the counter and use it for a unique node name
        if info_label:
            unique_name = f"{new_name}_info_node"
            rank_subgraph = pydot.Subgraph()
            node = pydot.Node(
                unique_name,
                label=info_label,
                shape="ellipse",
                style="dashed",
                fontname="Helvetica",
                penwidth=2,
                color="darkgoldenrod3",
            )
            rank_subgraph.add_node(node)
            self.add_subgraph(rank_subgraph)
            self.add_node(node)


class AdjointCluster(pydot.Cluster):
    """Cluster representing a control flow structure in the graph, such as a loop or conditional."""

    _counter = 1

    def __init__(self, *args, info_label="", **kwargs):
        super().__init__(*args, **kwargs)
        self.set_penwidth(2)
        self.set_fontname("Helvetica")
        self.set_style("filled")
        self.set_fillcolor(kwargs.get("fillcolor", "hotpink1"))
        self.set_color(kwargs.get("color", "hotpink3"))

        cur_name = self.get_name()
        new_name = f"{cur_name}_adjoint_{AdjointCluster._counter}"
        AdjointCluster._counter += 1
        self.set_name(new_name)

        # Increment the counter and use it for a unique node name
        if info_label:
            unique_name = f"{new_name}_info_node"
            rank_subgraph = pydot.Subgraph()
            node = pydot.Node(
                unique_name,
                label=info_label,
                shape="ellipse",
                style="dashed",
                fontname="Helvetica",
                penwidth=2,
                color="hotpink3",
            )
            rank_subgraph.add_node(node)
            self.add_subgraph(rank_subgraph)
            self.add_node(node)


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

    def render_graph(self, output_filename: str) -> None:
        """Renders the graph to a file."""
        shade_pydot_clusters_by_depth(self.graph)
        self.graph.write(output_filename, format="png")

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


# --- Color Helper Functions (ensure these are defined as in previous examples) ---
def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError("Invalid HEX color format. Expected #RRGGBB or RRGGBB.")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb_tuple: tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(rgb_tuple[0]))),
        max(0, min(255, int(rgb_tuple[1]))),
        max(0, min(255, int(rgb_tuple[2]))),
    )


def _get_color_shade_for_depth(
    base_color_hex: str,
    current_depth: int,
    max_overall_depth: int,
    start_factor: float = 1.0,
    end_factor: float = 0.5,
) -> str:
    if max_overall_depth == 0:
        current_shade_factor = start_factor
    else:
        current_shade_factor = start_factor - (
            (start_factor - end_factor) * (current_depth / max_overall_depth)
        )

    lower_bound = min(start_factor, end_factor)
    upper_bound = max(start_factor, end_factor)
    current_shade_factor = max(lower_bound, min(upper_bound, current_shade_factor))

    r_base, g_base, b_base = _hex_to_rgb(base_color_hex)
    new_r = r_base * current_shade_factor
    new_g = g_base * current_shade_factor
    new_b = b_base * current_shade_factor
    return _rgb_to_hex((new_r, new_g, new_b))


def _collect_clusters_recursive(
    graph_element, current_host_cluster_depth: int, collected_cluster_info: list
):
    subgraphs = graph_element.get_subgraphs()

    for sg in subgraphs:
        if "cluster_" in sg.get_name():

            collected_cluster_info.append({"obj": sg, "depth": current_host_cluster_depth})
            _collect_clusters_recursive(sg, current_host_cluster_depth + 1, collected_cluster_info)
        else:
            _collect_clusters_recursive(sg, current_host_cluster_depth, collected_cluster_info)


# --- Main Shading Function ---
def shade_pydot_clusters_by_depth(
    graph: pydot.Graph,
    base_bg_color_hex: str = "#ECF3ED",
    start_factor: float = 1.0,
    end_factor: float = 0.6,
    default_cluster_border_color: str = "#000000",
) -> None:
    if not isinstance(graph, pydot.Graph):
        raise TypeError("Input must be a pydot.Graph object.")

    all_clusters_with_depths = []
    _collect_clusters_recursive(graph, 0, all_clusters_with_depths)

    if not all_clusters_with_depths:
        return

    max_depth = 0
    if len(all_clusters_with_depths) > 0:
        max_depth = max(item["depth"] for item in all_clusters_with_depths)

    for item in all_clusters_with_depths:
        cluster_obj = item["obj"]
        depth = item["depth"]
        if "ctrl" in cluster_obj.get_name():
            base_bg_color_hex = "#F9F1A7"  # Khaki for control clusters
        elif "adjoint" in cluster_obj.get_name():
            base_bg_color_hex = "#FFB6C1"
        else:
            base_bg_color_hex = "#ECF3ED"
        shaded_color = _get_color_shade_for_depth(
            base_bg_color_hex, depth, max_depth, start_factor, end_factor
        )
        cluster_obj.set("style", "filled")
        cluster_obj.set("fillcolor", shaded_color)
        cluster_obj.set("bgcolor", shaded_color)
        if default_cluster_border_color:
            cluster_obj.set("color", default_cluster_border_color)
        try:
            r_f, g_f, b_f = _hex_to_rgb(shaded_color)
            brightness = (r_f * 299 + g_f * 587 + b_f * 114) / 1000
            current_fontcolor = cluster_obj.get_fontcolor()
            if current_fontcolor is None:
                if brightness < 128:
                    cluster_obj.set("fontcolor", "white")
                else:
                    cluster_obj.set("fontcolor", "black")
        except (ValueError, TypeError):
            pass
