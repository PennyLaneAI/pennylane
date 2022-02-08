# Copyright 2022 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module provides the circuit cutting functionality that allows large
circuits to be distributed across multiple devices.
"""

from functools import partial
from typing import Tuple, Union

from networkx import MultiDiGraph, weakly_connected_components
from pennylane import apply
from pennylane.measure import MeasurementProcess
from pennylane.operation import Operation, Operator, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape
from pennylane.wires import Wires


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None


SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def replace_wire_cut_node(node: WireCut, graph: MultiDiGraph):
    """
    Replace a :class:`~.WireCut` node in the graph with a :class:`~.MeasureNode`
    and :class:`~.PrepareNode`.

    Args:
        node (WireCut): the  :class:`~.WireCut` node to be replaced with a :class:`~.MeasureNode`
            and :class:`~.PrepareNode`
        graph (MultiDiGraph): the graph containing the node to be replaced

    **Example**

    Consider the following circuit with a manually-placed wire cut:

    .. code-block:: python

        wire_cut = qml.WireCut(wires=0)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.apply(wire_cut)
            qml.RY(0.5, wires=0)
            qml.expval(qml.PauliZ(0))

    We can find the circuit graph and remove the wire cut node using:

    >>> graph = qml.transforms.tape_to_graph(tape)
    >>> qml.transforms.replace_wire_cut_node(wire_cut, graph)
    """
    predecessors = graph.pred[node]
    successors = graph.succ[node]

    predecessor_on_wire = {}
    for op, data in predecessors.items():
        for d in data.values():
            wire = d["wire"]
            predecessor_on_wire[wire] = op

    successor_on_wire = {}
    for op, data in successors.items():
        for d in data.values():
            wire = d["wire"]
            successor_on_wire[wire] = op

    order = graph.nodes[node]["order"]
    graph.remove_node(node)

    for wire in node.wires:
        predecessor = predecessor_on_wire.get(wire, None)
        successor = successor_on_wire.get(wire, None)

        meas = MeasureNode(wires=wire)
        prep = PrepareNode(wires=wire)

        # We are introducing a degeneracy in the order of the measure and prepare nodes
        # here but the order can be inferred as MeasureNode always precedes
        # the corresponding PrepareNode
        graph.add_node(meas, order=order)
        graph.add_node(prep, order=order)

        graph.add_edge(meas, prep, wire=wire)

        if predecessor is not None:
            graph.add_edge(predecessor, meas, wire=wire)
        if successor is not None:
            graph.add_edge(prep, successor, wire=wire)


def replace_wire_cut_nodes(graph: MultiDiGraph):
    """
    Replace each :class:`~.WireCut` node in the graph with a
    :class:`~.MeasureNode` and :class:`~.PrepareNode`.

    Args:
        graph (MultiDiGraph): The graph containing the :class:`~.WireCut` nodes
            to be replaced

    **Example**

    Consider the following circuit with manually-placed wire cuts:

    .. code-block:: python

        wire_cut_0 = qml.WireCut(wires=0)
        wire_cut_1 = qml.WireCut(wires=1)
        multi_wire_cut = qml.WireCut(wires=[0, 1])

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.apply(wire_cut_0)
            qml.RY(0.5, wires=0)
            qml.apply(wire_cut_1)
            qml.CNOT(wires=[0, 1])
            qml.apply(multi_wire_cut)
            qml.RZ(0.6, wires=1)
            qml.expval(qml.PauliZ(0))

    We can find the circuit graph and remove all the wire cut nodes using:

    >>> graph = qml.transforms.tape_to_graph(tape)
    >>> qml.transforms.replace_wire_cut_nodes(graph)
    """
    for op in list(graph.nodes):
        if isinstance(op, WireCut):
            replace_wire_cut_node(op, graph)


def _add_operator_node(graph: MultiDiGraph, op: Operator, order: int, wire_latest_node: dict):
    """
    Helper function to add operators as nodes during tape to graph conversion.
    """
    graph.add_node(op, order=order)
    for wire in op.wires:
        if wire_latest_node[wire] is not None:
            parent_op = wire_latest_node[wire]
            graph.add_edge(parent_op, op, wire=wire)
        wire_latest_node[wire] = op


def tape_to_graph(tape: QuantumTape) -> MultiDiGraph:
    """
    Converts a quantum tape to a directed multigraph.

    Args:
        tape (QuantumTape): tape to be converted into a directed multigraph

    Returns:
        MultiDiGraph: a directed multigraph that captures the circuit structure
        of the input tape

    **Example**

    Consider the following tape:

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.9, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(1))

    Its corresponding circuit graph can be found using

    >>> qml.transforms.tape_to_graph(tape)
    <networkx.classes.multidigraph.MultiDiGraph at 0x7fe41cbd7210>
    """
    graph = MultiDiGraph()

    wire_latest_node = {w: None for w in tape.wires}

    for order, op in enumerate(tape.operations):
        _add_operator_node(graph, op, order, wire_latest_node)

    order += 1  # pylint: disable=undefined-loop-variable
    for m in tape.measurements:
        obs = getattr(m, "obs", None)
        if obs is not None and isinstance(obs, Tensor):
            for o in obs.obs:
                m_ = MeasurementProcess(m.return_type, obs=o)

                _add_operator_node(graph, m_, order, wire_latest_node)

        else:
            _add_operator_node(graph, m, order, wire_latest_node)
            order += 1

    return graph


def fragment_graph(graph: MultiDiGraph) -> Tuple[Tuple[MultiDiGraph], MultiDiGraph]:
    """
    Fragments a graph into a collection of subgraphs as well as returning
    the communication/`quotient <https://en.wikipedia.org/wiki/Quotient_graph>`__
    graph. Each node of the communication graph represents a fragment and the edges

    denote the flow of qubits between fragments.

    Args:
        graph (MultiDiGraph): directed multigraph containing measure and prepare
            nodes at cut locations

    Returns:
        Tuple[Tuple[MultiDiGraph], MultiDiGraph]: the subgraphs of the cut graph
        and the communication graph.

    **Example**

    Consider the following circuit with manually-placed wire cuts:

    .. code-block:: python

        wire_cut_0 = qml.WireCut(wires=0)
        wire_cut_1 = qml.WireCut(wires=1)
        multi_wire_cut = qml.WireCut(wires=[0, 1])

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.apply(wire_cut_0)
            qml.RY(0.5, wires=0)
            qml.apply(wire_cut_1)
            qml.CNOT(wires=[0, 1])
            qml.apply(multi_wire_cut)
            qml.RZ(0.6, wires=1)
            qml.expval(qml.PauliZ(0))

    We can find the corresponding graph, remove all the wire cut nodes, and
    find the subgraphs and communication graph by using:

    >>> graph = qml.transforms.tape_to_graph(tape)
    >>> qml.transforms.replace_wire_cut_nodes(graph)
    >>> qml.transforms.fragment_graph(graph)
    ((<networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311940>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311c10>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e2820>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e27f0>),
     <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e26a0>)
    """

    graph_copy = graph.copy()

    cut_edges = []

    for node1, node2, wire in graph.edges:
        if isinstance(node1, MeasureNode):
            assert isinstance(node2, PrepareNode)
            cut_edges.append((node1, node2, wire))
            graph_copy.remove_edge(node1, node2, key=wire)

    subgraph_nodes = weakly_connected_components(graph_copy)
    subgraphs = tuple(graph_copy.subgraph(n) for n in subgraph_nodes)

    communication_graph = MultiDiGraph()
    communication_graph.add_nodes_from(range(len(subgraphs)))

    for node1, node2, wire in cut_edges:
        for i, subgraph in enumerate(subgraphs):
            if subgraph.has_node(node1):
                start_fragment = i
            if subgraph.has_node(node2):
                end_fragment = i

        communication_graph.add_edge(start_fragment, end_fragment, pair=(node1, node2, wire))

    return subgraphs, communication_graph


def _subscripted(inp: Union[int, str], subscript: int) -> str:
    """Returns a subscripted version of ``inp`` with an integer
    subscript."""
    return str(inp) + str(subscript).translate(SUB)


def _find_new_wire(target, wires: Wires) -> int:
    """Finds a new wire label that is not in ``wires`` based
    upon a ``target`` label. Subscripts are used to find a
    unique new label."""
    if target not in wires:
        return target

    if isinstance(target, (int, str)):
        wire_func = partial(_subscripted, target)
    else:
        wire_func = lambda ctr: ctr - 1

    ctr = 1
    new_wire = wire_func(ctr)

    while new_wire in wires:
        ctr += 1
        new_wire = wire_func(ctr)
    return new_wire


def graph_to_tape(graph: MultiDiGraph) -> QuantumTape:
    """
    Converts a directed multigraph to the corresponding quantum tape. Each node
    in graph should have an order attribute specifying the topological order of
    the operations. This allows for the support of mid circuit measurements
    when speficied as `MeasureNode` within a  directed multigraph.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :doc:`transforms </code/qml_transforms>` page for more details.

    Args:
        graph (MultiDiGraph): directed multigraph to be converted to a tape

    Returns:
        QuantumTape: the quantum tape corresponding to the input graph

    **Example**

    Consider the following where ``graph`` contains a three pairs of :class:`~.MeasureNode` and
    :class:`~.PrepareNode` which divides the full circuit graph into five subgraphs.
    We can find the circuit fragments by using:

    .. code-block:: python

        >>> subgraphs, communication_graph = qml.transforms.fragment_graph(graph)
        >>> tapes = [qml.transforms.graph_to_tape(sg) for sg in subgraphs]
        >>> tapes
        [<QuantumTape: wires=[0], params=1>, <QuantumTape: wires=[0, 1], params=1>,
         <QuantumTape: wires=[1], params=1>, <QuantumTape: wires=[0], params=0>,
         <QuantumTape: wires=[1], params=0>]
    """
    wires = Wires.all_wires([n.wires for n in graph.nodes])

    ordered_ops = sorted(
        [(order, op) for op, order in graph.nodes(data="order")], key=lambda x: x[0]
    )
    wire_map = {w: w for w in wires}

    with QuantumTape() as tape:
        for _, op in ordered_ops:
            original_op_wires = op._wires
            new_wires = [wire_map[w] for w in op.wires]
            op._wires = Wires(new_wires)  # TODO: find a better way to update operation wires
            apply(op)
            op._wires = original_op_wires

            if isinstance(op, MeasureNode):
                assert len(op.wires) == 1
                measured_wire = op.wires[0]
                new_wire = _find_new_wire(measured_wire, wires)
                wires += new_wire
                wire_map[measured_wire] = new_wire

    return tape
