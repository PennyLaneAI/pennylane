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
Support functions for cut_circuit and cut_circuit_mc.
"""


import uuid
from typing import Any, Callable, Sequence, Tuple
import warnings
import numpy as np
from networkx import MultiDiGraph, has_path, weakly_connected_components

import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.meta import WireCut
from pennylane.queuing import WrappedObj
from pennylane.operation import Operation

from .kahypar import kahypar_cut
from .cutstrategy import CutStrategy


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None
    num_params = 0

    def __init__(self, wires=None, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None
    num_params = 0

    def __init__(self, wires=None, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(wires=wires, id=id)

    def label(self, decimals=None, base_label=None, cache=None):
        op_label = base_label or self.__class__.__name__
        return op_label


def _prep_zero_state(wire):
    return [qml.Identity(wire)]


def _prep_one_state(wire):
    return [qml.X(wire)]


def _prep_plus_state(wire):
    return [qml.Hadamard(wire)]


def _prep_minus_state(wire):
    return [qml.X(wire), qml.Hadamard(wire)]


def _prep_iplus_state(wire):
    return [qml.Hadamard(wire), qml.S(wires=wire)]


def _prep_iminus_state(wire):
    return [qml.X(wire), qml.Hadamard(wire), qml.S(wires=wire)]


def find_and_place_cuts(
    graph: MultiDiGraph,
    cut_method: Callable = kahypar_cut,
    cut_strategy: CutStrategy = None,
    replace_wire_cuts=False,
    local_measurement=False,
    **kwargs,
) -> MultiDiGraph:
    """Automatically finds and places optimal :class:`~.WireCut` nodes into a given tape-converted graph
    using a customizable graph partitioning function. Preserves existing placed cuts.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.
        cut_method (Callable): A graph partitioning function that takes an input graph and returns
            a list of edges to be cut based on a given set of constraints and objective. Defaults
            to :func:`kahypar_cut` which requires KaHyPar to be installed using
            ``pip install kahypar`` for Linux and Mac users or visiting the
            instructions `here <https://kahypar.org>`__ to compile from
            source for Windows users.
        cut_strategy (CutStrategy): Strategy for optimizing cutting parameters based on device
            constraints. Defaults to ``None`` in which case ``kwargs`` must be fully specified
            for passing to the ``cut_method``.
        replace_wire_cuts (bool): Whether to replace :class:`~.WireCut` nodes with
            :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs. Defaults to ``False``.
        local_measurement (bool): Whether to use the local-measurement circuit-cutting objective,
            i.e. the maximum node-degree of the communication graph, for cut evaluation. Defaults
            to ``False`` which assumes global measurement and uses the total number of cuts as the
            cutting objective.
        kwargs: Additional keyword arguments to be passed to the callable ``cut_method``.

    Returns:
        nx.MultiDiGraph: Copy of the input graph with :class:`~.WireCut` nodes inserted.

    **Example**

    Consider the following 4-wire circuit with a single CNOT gate connecting the top (wires
    ``[0, 1]``) and bottom (wires ``["a", "b"]``) halves of the circuit. Note there's a
    :class:`~.WireCut` manually placed into the circuit already.

    .. code-block:: python

        ops = [
            qml.RX(0.1, wires=0),
            qml.RY(0.2, wires=1),
            qml.RX(0.3, wires="a"),
            qml.RY(0.4, wires="b"),
            qml.CNOT(wires=[0, 1]),
            qml.WireCut(wires=1),
            qml.CNOT(wires=["a", "b"]),
            qml.CNOT(wires=[1, "a"]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=["a", "b"]),
            qml.RX(0.5, wires="a"),
            qml.RY(0.6, wires="b"),
        ]
        measurements = [qml.expval(qml.X(0) @ qml.Y("a") @ qml.Z("b"))]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape, decimals=1))
    0: ──RX(0.1)─╭●────────╭●──────────┤ ╭<X@Y@Z>
    1: ──RY(0.2)─╰X──//─╭●─╰X──────────┤ │
    a: ──RX(0.3)─╭●─────╰X─╭●──RX(0.5)─┤ ├<X@Y@Z>
    b: ──RY(0.4)─╰X────────╰X──RY(0.6)─┤ ╰<X@Y@Z>

    Since the existing :class:`~.WireCut` doesn't sufficiently fragment the circuit, we can find the
    remaining cuts using the default KaHyPar partitioner:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> cut_graph = qml.qcut.find_and_place_cuts(
    ...     graph=graph,
    ...     num_fragments=2,
    ...     imbalance=0.5,
    ... )

    Visualizing the newly-placed cut:

    >>> print(qml.qcut.graph_to_tape(cut_graph).draw(decimals=1))
    0: ──RX(0.1)─╭●────────────╭●───────┤ ╭<X@Y@Z>
    1: ──RY(0.2)─╰X──//─╭●──//─╰X───────┤ │
    a: ──RX(0.3)─╭●─────╰X─╭●───RX(0.5)─┤ ├<X@Y@Z>
    b: ──RY(0.4)─╰X────────╰X───RY(0.6)─┤ ╰<X@Y@Z>

    We can then proceed with the usual process of replacing :class:`~.WireCut` nodes with
    pairs of :class:`~.MeasureNode` and :class:`~.PrepareNode`, and then break the graph
    into fragments. Or, alternatively, we can directly get such processed graph by passing
    ``replace_wire_cuts=True``:

    >>> cut_graph = qml.qcut.find_and_place_cuts(
    ...     graph=graph,
    ...     num_fragments=2,
    ...     imbalance=0.5,
    ...     replace_wire_cuts=True,
    ... )
    >>> frags, comm_graph = qml.qcut.fragment_graph(cut_graph)
    >>> for t in frags:
    ...     print(qml.qcut.graph_to_tape(t).draw())

    .. code-block::

         0: ──RX(0.1)──────╭●───────────────╭●──┤ ⟨X⟩
         1: ──RY(0.2)──────╰X──MeasureNode──│───┤
         2: ──PrepareNode───────────────────╰X──┤

         a: ──RX(0.3)──────╭●──╭X──╭●────────────RX(0.5)──╭┤ ⟨Y ⊗ Z⟩
         b: ──RY(0.4)──────╰X──│───╰X────────────RY(0.6)──╰┤ ⟨Y ⊗ Z⟩
         1: ──PrepareNode──────╰●───MeasureNode────────────┤

    Alternatively, if all we want to do is to find the optimal way to fit a circuit onto a smaller
    device, a :class:`~.CutStrategy` can be used to populate the necessary explorations of cutting
    parameters. As an extreme example, if the only device at our disposal is a 2-qubit device, a
    simple cut strategy is to simply specify the the ``max_free_wires`` argument (or equivalently
    directly passing a :class:`pennylane.Device` to the ``device`` argument):

    >>> cut_strategy = qml.qcut.CutStrategy(max_free_wires=2)
    >>> cut_strategy.get_cut_kwargs(graph)
    [{'num_fragments': 2, 'imbalance': 0.2857142857142858},
     {'num_fragments': 3, 'imbalance': 0.2857142857142858},
     {'num_fragments': 4, 'imbalance': 0.2857142857142858},
     {'num_fragments': 5, 'imbalance': 0.2857142857142858},
     {'num_fragments': 6, 'imbalance': 0.2857142857142858},
     {'num_fragments': 7, 'imbalance': 0.2857142857142858},
     {'num_fragments': 8, 'imbalance': 0.2857142857142858},
     {'num_fragments': 9, 'imbalance': 0.2857142857142858},
     {'num_fragments': 10, 'imbalance': 0.2857142857142858},
     {'num_fragments': 11, 'imbalance': 0.2857142857142858},
     {'num_fragments': 12, 'imbalance': 0.2857142857142858},
     {'num_fragments': 13, 'imbalance': 0.0},
     {'num_fragments': 14, 'imbalance': 0.0}]

    The printed list above shows all the possible cutting configurations one can attempt to perform
    in order to search for the optimal cut. This is done by directly passing a
    :class:`~.CutStrategy` to :func:`~.find_and_place_cuts`:

    >>> cut_graph = qml.qcut.find_and_place_cuts(
            graph=graph,
            cut_strategy=cut_strategy,
        )
    >>> print(qml.qcut.graph_to_tape(cut_graph).draw())
    0: ──RX──//─╭●──//────────╭●──//────────┤ ╭<X@Y@Z>
    1: ──RY──//─╰X──//─╭●──//─╰X────────────┤ │
    a: ──RX──//─╭●──//─╰X──//─╭●──//──RX─//─┤ ├<X@Y@Z>
    b: ──RY──//─╰X──//────────╰X──//──RY────┤ ╰<X@Y@Z>

    As one can tell, quite a few cuts have to be made in order to execute the circuit on solely
    2-qubit devices. To verify, let's print the fragments:

    >>> qml.qcut.replace_wire_cut_nodes(cut_graph)
    >>> frags, comm_graph = qml.qcut.fragment_graph(cut_graph)
    >>> for t in frags:
    ...     print(qml.qcut.graph_to_tape(t).draw())

    .. code-block::

         0: ──RX──MeasureNode─┤

         1: ──RY──MeasureNode─┤

         a: ──RX──MeasureNode─┤

         b: ──RY──MeasureNode─┤

         0: ──PrepareNode─╭●──MeasureNode─┤
         1: ──PrepareNode─╰X──MeasureNode─┤

         a: ──PrepareNode─╭●──MeasureNode─┤
         b: ──PrepareNode─╰X──MeasureNode─┤

         1: ──PrepareNode─╭●──MeasureNode─┤
         a: ──PrepareNode─╰X──MeasureNode─┤

         0: ──PrepareNode─╭●──MeasureNode─┤
         1: ──PrepareNode─╰X──────────────┤

         b: ──PrepareNode─╭X──MeasureNode─┤
         a: ──PrepareNode─╰●──MeasureNode─┤

         a: ──PrepareNode──RX──MeasureNode─┤

         b: ──PrepareNode──RY─┤  <Z>

         0: ──PrepareNode─┤  <X>

         a: ──PrepareNode─┤  <Y>

    """

    cut_graph = _remove_existing_cuts(graph)

    if isinstance(cut_strategy, CutStrategy):
        cut_kwargs_probed = cut_strategy.get_cut_kwargs(cut_graph)

        # Need to reseed if a seed is passed:
        seed = kwargs.pop("seed", None)
        seeds = np.random.default_rng(seed).choice(2**15, cut_strategy.trials_per_probe).tolist()

        cut_edges_probed = {
            (cut_kwargs["num_fragments"], trial_id): cut_method(
                cut_graph,
                **{
                    **cut_kwargs,
                    **kwargs,
                    "seed": seed,
                },  # kwargs has higher precedence for colliding keys
            )
            for cut_kwargs in cut_kwargs_probed
            for trial_id, seed in zip(range(cut_strategy.trials_per_probe), seeds)
        }

        valid_cut_edges = {}
        for (num_partitions, _), cut_edges in cut_edges_probed.items():
            # The easiest way to tell if a cut is valid is to just do the fragment graph.

            cut_graph = place_wire_cuts(graph=graph, cut_edges=cut_edges)
            num_cuts = sum(isinstance(n.obj, WireCut) for n in cut_graph.nodes)

            replace_wire_cut_nodes(cut_graph)
            frags, comm = fragment_graph(cut_graph)

            max_frag_degree = max(dict(comm.degree()).values())

            if _is_valid_cut(
                fragments=frags,
                num_cuts=num_cuts,
                max_frag_degree=max_frag_degree,
                num_fragments_requested=num_partitions,
                cut_candidates=valid_cut_edges,
                max_free_wires=cut_strategy.max_free_wires,
            ):
                key = (len(frags), max_frag_degree)
                valid_cut_edges[key] = cut_edges

        if len(valid_cut_edges) < 1:
            raise ValueError(
                "Unable to find a circuit cutting that satisfies all constraints. "
                "Are the constraints too strict?"
            )

        cut_edges = _get_optim_cut(valid_cut_edges, local_measurement=local_measurement)

    else:
        cut_edges = cut_method(cut_graph, **kwargs)

    cut_graph = place_wire_cuts(graph=graph, cut_edges=cut_edges)

    if replace_wire_cuts:
        replace_wire_cut_nodes(cut_graph)

    return cut_graph


def replace_wire_cut_node(node: WireCut, graph: MultiDiGraph):
    """
    Replace a :class:`~.WireCut` node in the graph with a :class:`~.MeasureNode`
    and :class:`~.PrepareNode`.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        node (WireCut): the  :class:`~.WireCut` node to be replaced with a :class:`~.MeasureNode`
            and :class:`~.PrepareNode`
        graph (nx.MultiDiGraph): the graph containing the node to be replaced

    **Example**

    Consider the following circuit with a manually-placed wire cut:

    .. code-block:: python

        wire_cut = qml.WireCut(wires=0)

        ops = [
            qml.RX(0.4, wires=0),
            wire_cut,
            qml.RY(0.5, wires=0),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can find the circuit graph and remove the wire cut node using:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_node(wire_cut, graph)
    """
    node_obj = WrappedObj(node)
    predecessors = graph.pred[node_obj]
    successors = graph.succ[node_obj]

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

    order = graph.nodes[node_obj]["order"]
    graph.remove_node(node_obj)

    for wire in node.wires:
        predecessor = predecessor_on_wire.get(wire, None)
        successor = successor_on_wire.get(wire, None)

        meas = MeasureNode(wires=wire)
        prep = PrepareNode(wires=wire)

        # We are introducing a degeneracy in the order of the measure and prepare nodes
        # here but the order can be inferred as MeasureNode always precedes
        # the corresponding PrepareNode
        meas_node = WrappedObj(meas)
        prep_node = WrappedObj(prep)
        graph.add_node(meas_node, order=order)
        graph.add_node(prep_node, order=order)

        graph.add_edge(meas_node, prep_node, wire=wire)

        if predecessor is not None:
            graph.add_edge(predecessor, meas_node, wire=wire)
        if successor is not None:
            graph.add_edge(prep_node, successor, wire=wire)


def replace_wire_cut_nodes(graph: MultiDiGraph):
    """
    Replace each :class:`~.WireCut` node in the graph with a
    :class:`~.MeasureNode` and :class:`~.PrepareNode`.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        graph (nx.MultiDiGraph): The graph containing the :class:`~.WireCut` nodes
            to be replaced

    **Example**

    Consider the following circuit with manually-placed wire cuts:

    .. code-block:: python

        wire_cut_0 = qml.WireCut(wires=0)
        wire_cut_1 = qml.WireCut(wires=1)
        multi_wire_cut = qml.WireCut(wires=[0, 1])

        ops = [
            qml.RX(0.4, wires=0),
            wire_cut_0,
            qml.RY(0.5, wires=0),
            wire_cut_1,
            qml.CNOT(wires=[0, 1]),
            multi_wire_cut,
            qml.RZ(0.6, wires=1),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can find the circuit graph and remove all the wire cut nodes using:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_nodes(graph)
    """
    for op in list(graph.nodes):
        if isinstance(op.obj, WireCut):
            replace_wire_cut_node(op.obj, graph)


def place_wire_cuts(
    graph: MultiDiGraph, cut_edges: Sequence[Tuple[Operation, Operation, Any]]
) -> MultiDiGraph:
    """Inserts a :class:`~.WireCut` node for each provided cut edge into a circuit graph.

    Args:
        graph (nx.MultiDiGraph): The original (tape-converted) graph to be cut.
        cut_edges (Sequence[Tuple[Operation, Operation, Any]]): List of ``MultiDiGraph`` edges
            to be replaced with a :class:`~.WireCut` node. Each 3-tuple represents the source node, the
            target node, and the wire key of the (multi)edge.

    Returns:
        MultiDiGraph: Copy of the input graph with :class:`~.WireCut` nodes inserted.

    **Example**

    Consider the following 2-wire circuit with one CNOT gate connecting the wires:

    .. code-block:: python

        ops = [
            qml.RX(0.432, wires=0),
            qml.RY(0.543, wires="a"),
            qml.CNOT(wires=[0, "a"]),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape_text(tape, decimals=3))
    0: ──RX(0.432)─╭●─┤  <Z>
    a: ──RY(0.543)─╰X─┤

    If we know we want to place a :class:`~.WireCut` node between the nodes corresponding to the
    ``RY(0.543, wires=["a"])`` and ``CNOT(wires=[0, 'a'])`` operations after the tape is constructed,
    we can first find the edge in the graph:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> op0, op1 = tape.operations[1], tape.operations[2]
    >>> cut_edges = [e for e in graph.edges if e[0].obj is op0 and e[1].obj is op1]
    >>> cut_edges
    [(Wrapped(RY(0.543, wires=['a'])), Wrapped(CNOT(wires=[0, 'a'])), 0)]

    Then feed it to this function for placement:

    >>> cut_graph = qml.qcut.place_wire_cuts(graph=graph, cut_edges=cut_edges)
    >>> cut_graph
    <networkx.classes.multidigraph.MultiDiGraph at 0x7f7251ac1220>

    And visualize the cut by converting back to a tape:

    >>> print(qml.qcut.graph_to_tape(cut_graph).draw(decimals=3))
    0: ──RX(0.432)─────╭●─┤  <Z>
    a: ──RY(0.543)──//─╰X─┤
    """
    cut_graph = graph.copy()

    for op0, op1, wire_key in cut_edges:
        # Get info:
        order = cut_graph.nodes[op0]["order"] + 1
        wire = cut_graph.edges[(op0, op1, wire_key)]["wire"]
        # Apply cut:
        cut_graph.remove_edge(op0, op1, wire_key)
        # Increment order for all subsequent gates:
        for op, o in cut_graph.nodes(data="order"):
            if o >= order:
                cut_graph.nodes[op]["order"] += 1
        # Add WireCut
        wire_cut = WireCut(wires=wire)
        wire_cut_node = WrappedObj(wire_cut)
        cut_graph.add_node(wire_cut_node, order=order)
        cut_graph.add_edge(op0, wire_cut_node, wire=wire)
        cut_graph.add_edge(wire_cut_node, op1, wire=wire)

    return cut_graph


def _remove_existing_cuts(graph: MultiDiGraph) -> MultiDiGraph:
    """Removes all existing, manually or automatically placed, cuts from a circuit graph, be it
    ``WireCut``s or ``MeasureNode``-``PrepareNode`` pairs.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.

    Returns:
        (MultiDiGraph): Copy of the input graph with all its existing cuts removed.
    """
    uncut_graph = graph.copy()
    for node in list(graph.nodes):
        if isinstance(node.obj, WireCut):
            uncut_graph.remove_node(node)
        elif isinstance(node.obj, MeasureNode):
            for node1 in graph.neighbors(node):
                if isinstance(node1.obj, PrepareNode):
                    uncut_graph.remove_node(node)
                    uncut_graph.remove_node(node1)

    if len([n for n in uncut_graph.nodes if isinstance(n.obj, (MeasureNode, PrepareNode))]) > 0:
        warnings.warn(
            "The circuit contains `MeasureNode` or `PrepareNode` operations that are "
            "not paired up correctly. Please check.",
            UserWarning,
        )
    return uncut_graph


# pylint: disable=too-many-branches
def fragment_graph(graph: MultiDiGraph) -> Tuple[Tuple[MultiDiGraph], MultiDiGraph]:
    """
    Fragments a graph into a collection of subgraphs as well as returning
    the communication (`quotient <https://en.wikipedia.org/wiki/Quotient_graph>`__)
    graph.

    The input ``graph`` is fragmented by disconnecting each :class:`~.MeasureNode` and
    :class:`~.PrepareNode` pair and finding the resultant disconnected subgraph fragments.
    Each node of the communication graph represents a subgraph fragment and the edges
    denote the flow of qubits between fragments due to the removed :class:`~.MeasureNode` and
    :class:`~.PrepareNode` pairs.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        graph (nx.MultiDiGraph): directed multigraph containing measure and prepare
            nodes at cut locations

    Returns:
        Tuple[Tuple[nx.MultiDiGraph], nx.MultiDiGraph]: the subgraphs of the cut graph
        and the communication graph.

    **Example**

    Consider the following circuit with manually-placed wire cuts:

    .. code-block:: python

        wire_cut_0 = qml.WireCut(wires=0)
        wire_cut_1 = qml.WireCut(wires=1)
        multi_wire_cut = qml.WireCut(wires=[0, 1])

        ops = [
            qml.RX(0.4, wires=0),
            wire_cut_0,
            qml.RY(0.5, wires=0),
            wire_cut_1,
            qml.CNOT(wires=[0, 1]),
            multi_wire_cut,
            qml.RZ(0.6, wires=1),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can find the corresponding graph, remove all the wire cut nodes, and
    find the subgraphs and communication graph by using:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_nodes(graph)
    >>> qml.qcut.fragment_graph(graph)
    ([<networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311940>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311c10>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e2820>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e27f0>],
     <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e26a0>)
    """

    graph_copy = graph.copy()

    cut_edges = []
    measure_nodes = [n for n in graph.nodes if isinstance(n.obj, MeasurementProcess)]

    for node1, node2, wire_key in graph.edges:
        if isinstance(node1.obj, MeasureNode):
            assert isinstance(node2.obj, PrepareNode)
            cut_edges.append((node1, node2, wire_key))
            graph_copy.remove_edge(node1, node2, key=wire_key)

    subgraph_nodes = weakly_connected_components(graph_copy)
    subgraphs = tuple(MultiDiGraph(graph_copy.subgraph(n)) for n in subgraph_nodes)

    communication_graph = MultiDiGraph()
    communication_graph.add_nodes_from(range(len(subgraphs)))

    for node1, node2, _ in cut_edges:
        for i, subgraph in enumerate(subgraphs):
            if subgraph.has_node(node1):
                start_fragment = i
            if subgraph.has_node(node2):
                end_fragment = i

        if start_fragment != end_fragment:
            communication_graph.add_edge(start_fragment, end_fragment, pair=(node1, node2))
        else:
            # The MeasureNode and PrepareNode pair live in the same fragment and did not result
            # in a disconnection. We can therefore remove these nodes. Note that we do not need
            # to worry about adding back an edge between the predecessor to node1 and the successor
            # to node2 because our next step is to convert the fragment circuit graphs to tapes,
            # a process that does not depend on edge connections in the subgraph.
            subgraphs[start_fragment].remove_node(node1)
            subgraphs[end_fragment].remove_node(node2)

    terminal_indices = [i for i, s in enumerate(subgraphs) for n in measure_nodes if s.has_node(n)]

    subgraphs_connected_to_measurements = []
    subgraphs_indices_to_remove = []
    prepare_nodes_removed = []

    for i, s in enumerate(subgraphs):
        if any(has_path(communication_graph, i, t) for t in terminal_indices):
            subgraphs_connected_to_measurements.append(s)
        else:
            subgraphs_indices_to_remove.append(i)
            prepare_nodes_removed.extend([n for n in s.nodes if isinstance(n.obj, PrepareNode)])

    measure_nodes_to_remove = [
        m for p in prepare_nodes_removed for m, p_, _ in cut_edges if p is p_
    ]
    communication_graph.remove_nodes_from(subgraphs_indices_to_remove)

    for m in measure_nodes_to_remove:
        for s in subgraphs_connected_to_measurements:
            if s.has_node(m):
                s.remove_node(m)

    return subgraphs_connected_to_measurements, communication_graph


def _is_valid_cut(
    fragments,
    num_cuts,
    max_frag_degree,
    num_fragments_requested,
    cut_candidates,
    max_free_wires,
):
    """Helper function for determining if a cut is a valid canditate."""
    # pylint: disable=too-many-arguments

    k = len(fragments)
    key = (k, max_frag_degree)

    correct_num_fragments = k <= num_fragments_requested
    best_candidate_yet = (key not in cut_candidates) or (len(cut_candidates[key]) > num_cuts)
    # pylint: disable=no-member
    all_fragments_fit = all(
        len(qml.qcut.graph_to_tape(f).wires) <= max_free_wires for j, f in enumerate(fragments)
    )

    return correct_num_fragments and best_candidate_yet and all_fragments_fit


def _get_optim_cut(valid_cut_edges, local_measurement=False):
    """Picks out the best cut from a dict of valid candidate cuts."""

    if local_measurement:
        min_max_node_degree = min(max_node_degree for _, max_node_degree in valid_cut_edges)
        optim_cuts = {
            k: cut_edges
            for (k, max_node_degree), cut_edges in valid_cut_edges.items()
            if (max_node_degree == min_max_node_degree)
        }
    else:
        min_cuts = min(len(cut_edges) for cut_edges in valid_cut_edges.values())
        optim_cuts = {
            k: cut_edges
            for (k, _), cut_edges in valid_cut_edges.items()
            if (len(cut_edges) == min_cuts)
        }

    return optim_cuts[min(optim_cuts)]  # choose the lowest num_fragments among best ones.
