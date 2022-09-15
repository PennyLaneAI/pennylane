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
Functions for performing quantum circuit cutting.
"""

import copy
import inspect
import string
import uuid
import warnings
from collections.abc import Sequence as SequenceType
from dataclasses import InitVar, dataclass
from functools import partial
from itertools import compress, product
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from networkx import MultiDiGraph, has_path, weakly_connected_components

import pennylane as qml
from pennylane import apply, expval
from pennylane import numpy as np
from pennylane.grouping import string_to_pauli_word
from pennylane.measurements import Expectation, MeasurementProcess, Sample
from pennylane.operation import Operation, Operator, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape
from pennylane.wires import Wires

from .batch_transform import batch_transform


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None

    def __init__(self, *params, wires=None, do_queue=True, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None

    def __init__(self, *params, wires=None, do_queue=True, id=None):
        id = id or str(uuid.uuid4())

        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)


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

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.apply(wire_cut)
            qml.RY(0.5, wires=0)
            qml.expval(qml.PauliZ(0))

    We can find the circuit graph and remove the wire cut node using:

    >>> graph = qml.transforms.qcut.tape_to_graph(tape)
    >>> qml.transforms.qcut.replace_wire_cut_node(wire_cut, graph)
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

    >>> graph = qml.transforms.qcut.tape_to_graph(tape)
    >>> qml.transforms.qcut.replace_wire_cut_nodes(graph)
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

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        tape (QuantumTape): tape to be converted into a directed multigraph

    Returns:
        nx.MultiDiGraph: a directed multigraph that captures the circuit structure
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

    >>> qml.transforms.qcut.tape_to_graph(tape)
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
            if m.return_type is Sample:
                raise ValueError(
                    "Sampling from tensor products of observables "
                    "is not supported in circuit cutting"
                )
            for o in obs.obs:
                m_ = MeasurementProcess(m.return_type, obs=o)

                _add_operator_node(graph, m_, order, wire_latest_node)
        elif m.return_type is Sample and obs is None:
            for w in m.wires:
                s_ = qml.sample(qml.Projector([1], wires=w))
                _add_operator_node(graph, s_, order, wire_latest_node)
        else:
            _add_operator_node(graph, m, order, wire_latest_node)
            order += 1

    return graph


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

    >>> graph = qml.transforms.qcut.tape_to_graph(tape)
    >>> qml.transforms.qcut.replace_wire_cut_nodes(graph)
    >>> qml.transforms.qcut.fragment_graph(graph)
    ((<networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311940>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311c10>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e2820>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e27f0>),
     <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e26a0>)
    """

    graph_copy = graph.copy()

    cut_edges = []
    measure_nodes = [n for n in graph.nodes if isinstance(n, MeasurementProcess)]

    for node1, node2, wire_key in graph.edges:
        if isinstance(node1, MeasureNode):
            assert isinstance(node2, PrepareNode)
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
            prepare_nodes_removed.extend([n for n in s.nodes if isinstance(n, PrepareNode)])

    measure_nodes_to_remove = [
        m for p in prepare_nodes_removed for m, p_, _ in cut_edges if p is p_
    ]
    communication_graph.remove_nodes_from(subgraphs_indices_to_remove)

    for m in measure_nodes_to_remove:
        for s in subgraphs_connected_to_measurements:
            if s.has_node(m):
                s.remove_node(m)

    return subgraphs_connected_to_measurements, communication_graph


def _find_new_wire(wires: Wires) -> int:
    """Finds a new wire label that is not in ``wires``."""
    ctr = 0
    while ctr in wires:
        ctr += 1
    return ctr


# pylint: disable=protected-access
def graph_to_tape(graph: MultiDiGraph) -> QuantumTape:
    """
    Converts a directed multigraph to the corresponding :class:`~.QuantumTape`.

    To account for the possibility of needing to perform mid-circuit measurements, if any operations
    follow a :class:`MeasureNode` operation on a given wire then these operations are mapped to a
    new wire.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        graph (nx.MultiDiGraph): directed multigraph to be converted to a tape

    Returns:
        QuantumTape: the quantum tape corresponding to the input graph

    **Example**

    Consider the following circuit:

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.5, wires=1)
            qml.CNOT(wires=[0, 1])
            qml.transforms.qcut.MeasureNode(wires=1)
            qml.transforms.qcut.PrepareNode(wires=1)
            qml.CNOT(wires=[1, 0])
            qml.expval(qml.PauliZ(0))

    This circuit contains operations that follow a :class:`~.MeasureNode`. These operations will
    subsequently act on wire ``2`` instead of wire ``1``:

    >>> graph = qml.transforms.qcut.tape_to_graph(tape)
    >>> tape = qml.transforms.qcut.graph_to_tape(graph)
    >>> print(tape.draw())
    0: ──RX──────────╭●──────────────╭X─┤  <Z>
    1: ──RY──────────╰X──MeasureNode─│──┤
    2: ──PrepareNode─────────────────╰●─┤

    """

    wires = Wires.all_wires([n.wires for n in graph.nodes])

    ordered_ops = sorted(
        [(order, op) for op, order in graph.nodes(data="order")], key=lambda x: x[0]
    )
    wire_map = {w: w for w in wires}
    reverse_wire_map = {v: k for k, v in wire_map.items()}

    copy_ops = [copy.copy(op) for _, op in ordered_ops if not isinstance(op, MeasurementProcess)]
    copy_meas = [copy.copy(op) for _, op in ordered_ops if isinstance(op, MeasurementProcess)]
    observables = []

    with QuantumTape() as tape:
        for op in copy_ops:
            new_wires = Wires([wire_map[w] for w in op.wires])

            # TODO: find a better way to update operation wires
            op._wires = new_wires
            apply(op)

            if isinstance(op, MeasureNode):
                assert len(op.wires) == 1
                measured_wire = op.wires[0]

                new_wire = _find_new_wire(wires)
                wires += new_wire

                original_wire = reverse_wire_map[measured_wire]
                wire_map[original_wire] = new_wire
                reverse_wire_map[new_wire] = original_wire

        if copy_meas:
            return_types = set(meas.return_type for meas in copy_meas)
            if len(return_types) > 1:
                raise ValueError(
                    "Only a single return type can be used for measurement "
                    "nodes in graph_to_tape"
                )
            return_type = return_types.pop()

            if return_type not in {Sample, Expectation}:
                raise ValueError(
                    "Invalid return type. Only expectation value and sampling measurements "
                    "are supported in graph_to_tape"
                )

            for meas in copy_meas:
                obs = meas.obs
                obs._wires = Wires([wire_map[w] for w in obs.wires])
                observables.append(obs)

                if return_type is Sample:
                    apply(meas)

            if return_type is Expectation:
                if len(observables) > 1:
                    qml.expval(Tensor(*observables))
                else:
                    qml.expval(obs)

    return tape


def _get_measurements(
    group: Sequence[Operator], measurements: Sequence[MeasurementProcess]
) -> List[MeasurementProcess]:
    """Pairs each observable in ``group`` with the circuit ``measurements``.

    Only a single measurement of an expectation value is currently supported
    in ``measurements``.

    Args:
        group (Sequence[Operator]): a collection of observables
        measurements (Sequence[MeasurementProcess]): measurements from the circuit

    Returns:
        List[MeasurementProcess]: the expectation values of ``g @ obs``, where ``g`` is iterated
        over ``group`` and ``obs`` is the observable composing the single measurement
        in ``measurements``
    """
    if len(group) == 0:
        # This ensures the measurements of the original tape are carried over to the
        # following tape configurations in the absence of any MeasureNodes in the fragment
        return measurements

    n_measurements = len(measurements)
    if n_measurements > 1:
        raise ValueError(
            "The circuit cutting workflow only supports circuits with a single output "
            "measurement"
        )
    if n_measurements == 0:
        return [expval(g) for g in group]

    measurement = measurements[0]

    if measurement.return_type is not Expectation:
        raise ValueError(
            "The circuit cutting workflow only supports circuits with expectation "
            "value measurements"
        )

    obs = measurement.obs

    return [expval(copy.copy(obs) @ g) for g in group]


def _prep_zero_state(wire):
    qml.Identity(wire)


def _prep_one_state(wire):
    qml.PauliX(wire)


def _prep_plus_state(wire):
    qml.Hadamard(wire)


def _prep_minus_state(wire):
    qml.PauliX(wire)
    qml.Hadamard(wire)


def _prep_iplus_state(wire):
    qml.Hadamard(wire)
    qml.S(wires=wire)


def _prep_iminus_state(wire):
    qml.PauliX(wire)
    qml.Hadamard(wire)
    qml.S(wires=wire)


PREPARE_SETTINGS = [_prep_zero_state, _prep_one_state, _prep_plus_state, _prep_iplus_state]


def expand_fragment_tape(
    tape: QuantumTape,
) -> Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]:
    """
    Expands a fragment tape into a sequence of tapes for each configuration of the contained
    :class:`MeasureNode` and :class:`PrepareNode` operations.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        tape (QuantumTape): the fragment tape containing :class:`MeasureNode` and
            :class:`PrepareNode` operations to be expanded

    Returns:
        Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]: the
        tapes corresponding to each configuration and the order of preparation nodes and
        measurement nodes used in the expansion

    **Example**

    Consider the following circuit, which contains a :class:`~.MeasureNode` and
    :class:`~.PrepareNode` operation:

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.transforms.qcut.PrepareNode(wires=0)
            qml.RX(0.5, wires=0)
            qml.transforms.qcut.MeasureNode(wires=0)

    We can expand over the measurement and preparation nodes using:

    >>> tapes, prep, meas = qml.transforms.qcut.expand_fragment_tape(tape)
    >>> for t in tapes:
    ...     print(qml.drawer.tape_text(t, decimals=1))
    0: ──I──RX(0.5)─┤  <I>  <Z>
    0: ──I──RX(0.5)─┤  <X>
    0: ──I──RX(0.5)─┤  <Y>
    0: ──X──RX(0.5)─┤  <I>  <Z>
    0: ──X──RX(0.5)─┤  <X>
    0: ──X──RX(0.5)─┤  <Y>
    0: ──H──RX(0.5)─┤  <I>  <Z>
    0: ──H──RX(0.5)─┤  <X>
    0: ──H──RX(0.5)─┤  <Y>
    0: ──H──S──RX(0.5)─┤  <I>  <Z>
    0: ──H──S──RX(0.5)─┤  <X>
    0: ──H──S──RX(0.5)─┤  <Y>
    """
    prepare_nodes = [o for o in tape.operations if isinstance(o, PrepareNode)]
    measure_nodes = [o for o in tape.operations if isinstance(o, MeasureNode)]

    wire_map = {mn.wires[0]: i for i, mn in enumerate(measure_nodes)}

    n_meas = len(measure_nodes)
    if n_meas >= 1:
        measure_combinations = qml.grouping.partition_pauli_group(len(measure_nodes))
    else:
        measure_combinations = [[""]]

    tapes = []

    for prepare_settings in product(range(len(PREPARE_SETTINGS)), repeat=len(prepare_nodes)):
        for measure_group in measure_combinations:
            if n_meas >= 1:
                group = [
                    string_to_pauli_word(paulis, wire_map=wire_map) for paulis in measure_group
                ]
            else:
                group = []

            prepare_mapping = {
                n: PREPARE_SETTINGS[s] for n, s in zip(prepare_nodes, prepare_settings)
            }

            with QuantumTape() as tape_:
                for op in tape.operations:
                    if isinstance(op, PrepareNode):
                        w = op.wires[0]
                        prepare_mapping[op](w)
                    elif not isinstance(op, MeasureNode):
                        apply(op)

                with qml.queuing.stop_recording():
                    measurements = _get_measurements(group, tape.measurements)
                for meas in measurements:
                    apply(meas)

                tapes.append(tape_)

    return tapes, prepare_nodes, measure_nodes


MC_STATES = [
    _prep_zero_state,
    _prep_one_state,
    _prep_plus_state,
    _prep_minus_state,
    _prep_iplus_state,
    _prep_iminus_state,
    _prep_zero_state,
    _prep_one_state,
]


def _identity(wire):
    qml.sample(qml.Identity(wires=wire))


def _pauliX(wire):
    qml.sample(qml.PauliX(wires=wire))


def _pauliY(wire):
    qml.sample(qml.PauliY(wires=wire))


def _pauliZ(wire):
    qml.sample(qml.PauliZ(wires=wire))


MC_MEASUREMENTS = [
    _identity,
    _identity,
    _pauliX,
    _pauliX,
    _pauliY,
    _pauliY,
    _pauliZ,
    _pauliZ,
]


def expand_fragment_tapes_mc(
    tapes: Sequence[QuantumTape], communication_graph: MultiDiGraph, shots: int
) -> Tuple[List[QuantumTape], np.ndarray]:
    """
    Expands fragment tapes into a sequence of random configurations of the contained pairs of
    :class:`MeasureNode` and :class:`PrepareNode` operations.

    For each pair, a measurement is sampled from
    the Pauli basis and a state preparation is sampled from the corresponding pair of eigenstates.
    A settings array is also given which tracks the configuration pairs. Since each of the 4
    measurements has 2 possible eigenvectors, all configurations can be uniquely identified by
    8 values. The number of rows is determined by the number of cuts and the number of columns
    is determined by the number of shots.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`~.cut_circuit_mc` transform for more details.

    Args:
        tapes (Sequence[QuantumTape]): the fragment tapes containing :class:`MeasureNode` and
            :class:`PrepareNode` operations to be expanded
        communication_graph (nx.MultiDiGraph): the communication (quotient) graph of the fragmented
            full graph
        shots (int): number of shots

    Returns:
        Tuple[List[QuantumTape], np.ndarray]: the tapes corresponding to each configuration and the
        settings that track each configuration pair

    **Example**

    Consider the following circuit that contains a sample measurement:

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])
            qml.sample(wires=[0, 1, 2])

    We can generate the fragment tapes using the following workflow:

    >>> g = qml.transforms.qcut.tape_to_graph(tape)
    >>> qml.transforms.qcut.replace_wire_cut_nodes(g)
    >>> subgraphs, communication_graph = qml.transforms.qcut.fragment_graph(g)
    >>> tapes = [qml.transforms.qcut.graph_to_tape(sg) for sg in subgraphs]

    We can then expand over the measurement and preparation nodes to generate random
    configurations using:

    .. code-block:: python

        >>> configs, settings = qml.transforms.qcut.expand_fragment_tapes_mc(tapes, communication_graph, 3)
        >>> print(settings)
        [[1 6 2]]
        >>> for i, (c1, c2) in enumerate(zip(configs[0], configs[1])):
        ...     print(f"config {i}:")
        ...     print(c1.draw())
        ...     print("")
        ...     print(c2.draw())
        ...     print("")
        ...

        config 0:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Z]

        1: ──I─╭●─┤  Sample[|1⟩⟨1|]
        2: ────╰X─┤  Sample[|1⟩⟨1|]

        config 1:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Y]

        1: ──H──S─╭●─┤  Sample[|1⟩⟨1|]
        2: ───────╰X─┤  Sample[|1⟩⟨1|]

        config 2:
        0: ──H─╭●─┤  Sample[|1⟩⟨1|]
        1: ────╰X─┤  Sample[Y]

        1: ──X──H──S─╭●─┤  Sample[|1⟩⟨1|]
        2: ──────────╰X─┤  Sample[|1⟩⟨1|]

    """
    pairs = [e[-1] for e in communication_graph.edges.data("pair")]
    settings = np.random.choice(range(8), size=(len(pairs), shots), replace=True)

    meas_settings = {pair[0].id: setting for pair, setting in zip(pairs, settings)}
    prep_settings = {pair[1].id: setting for pair, setting in zip(pairs, settings)}

    all_configs = []
    for tape in tapes:
        frag_config = []
        for shot in range(shots):
            with qml.tape.QuantumTape() as new_tape:
                for op in tape.operations:
                    w = op.wires[0]
                    if isinstance(op, PrepareNode):
                        MC_STATES[prep_settings[op.id][shot]](w)
                    elif not isinstance(op, MeasureNode):
                        qml.apply(op)

                for meas in tape.measurements:
                    qml.apply(meas)
                for op in tape.operations:
                    meas_w = op.wires[0]
                    if isinstance(op, MeasureNode):
                        MC_MEASUREMENTS[meas_settings[op.id][shot]](meas_w)

            frag_config.append(new_tape)

        all_configs.append(frag_config)

    return all_configs, settings


def _reshape_results(results: Sequence, shots: int) -> List[List]:
    """
    Helper function to reshape ``results`` into a two-dimensional nested list whose number of rows
    is determined by the number of shots and whose number of columns is determined by the number of
    cuts.
    """
    results = [qml.math.flatten(r) for r in results]
    results = [results[i : i + shots] for i in range(0, len(results), shots)]
    results = list(map(list, zip(*results)))  # calculate list-based transpose

    return results


def qcut_processing_fn_sample(
    results: Sequence, communication_graph: MultiDiGraph, shots: int
) -> List:
    """
    Function to postprocess samples for the :func:`cut_circuit_mc() <pennylane.cut_circuit_mc>`
    transform. This removes superfluous mid-circuit measurement samples from fragment
    circuit outputs.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`qml.cut_circuit_mc() <pennylane.cut_circuit_mc>` transform for more details.

    Args:
        results (Sequence): a collection of sample-based execution results generated from the
            random expansion of circuit fragments over measurement and preparation node configurations
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        shots (int): the number of shots

    Returns:
        List[tensor_like]: the sampled output for all terminal measurements over the number of shots given
    """
    res0 = results[0]
    results = _reshape_results(results, shots)
    out_degrees = [d for _, d in communication_graph.out_degree]

    samples = []
    for result in results:
        sample = []
        for fragment_result, out_degree in zip(result, out_degrees):
            sample.append(fragment_result[: -out_degree or None])
        samples.append(np.hstack(sample))
    return [qml.math.convert_like(np.array(samples), res0)]


def qcut_processing_fn_mc(
    results: Sequence,
    communication_graph: MultiDiGraph,
    settings: np.ndarray,
    shots: int,
    classical_processing_fn: callable,
):
    """
    Function to postprocess samples for the :func:`cut_circuit_mc() <pennylane.cut_circuit_mc>`
    transform. This takes a user-specified classical function to act on bitstrings and
    generates an expectation value.

    .. note::

        This function is designed for use as part of the sampling-based circuit cutting workflow.
        Check out the :func:`qml.cut_circuit_mc() <pennylane.cut_circuit_mc>` transform for more details.

    Args:
        results (Sequence): a collection of sample-based execution results generated from the
            random expansion of circuit fragments over measurement and preparation node configurations
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        settings (np.ndarray): Each element is one of 8 unique values that tracks the specific
            measurement and preparation operations over all configurations. The number of rows is determined
            by the number of cuts and the number of columns is determined by the number of shots.
        shots (int): the number of shots
        classical_processing_fn (callable): A classical postprocessing function to be applied to
            the reconstructed bitstrings. The expected input is a bitstring; a flat array of length ``wires``
            and the output should be a single number within the interval :math:`[-1, 1]`.

    Returns:
        float or tensor_like: the expectation value calculated in accordance to Eq. (35) of
        `Peng et al. <https://arxiv.org/abs/1904.00102>`__
    """
    res0 = results[0]
    results = _reshape_results(results, shots)
    out_degrees = [d for _, d in communication_graph.out_degree]

    evals = (0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5)
    expvals = []
    for result, setting in zip(results, settings.T):
        sample_terminal = []
        sample_mid = []

        for fragment_result, out_degree in zip(result, out_degrees):
            sample_terminal.append(fragment_result[: -out_degree or None])
            sample_mid.append(fragment_result[-out_degree or len(fragment_result) :])

        sample_terminal = np.hstack(sample_terminal)
        sample_mid = np.hstack(sample_mid)

        assert set(sample_terminal).issubset({np.array(0), np.array(1)})
        assert set(sample_mid).issubset({np.array(-1), np.array(1)})
        # following Eq.(35) of Peng et.al: https://arxiv.org/abs/1904.00102
        f = classical_processing_fn(sample_terminal)
        if not -1 <= f <= 1:
            raise ValueError(
                "The classical processing function supplied must "
                "give output in the interval [-1, 1]"
            )
        sigma_s = np.prod(sample_mid)
        t_s = f * sigma_s
        c_s = np.prod([evals[s] for s in setting])
        K = len(sample_mid)
        expvals.append(8**K * c_s * t_s)

    return qml.math.convert_like(np.mean(expvals), res0)


@batch_transform
def cut_circuit_mc(
    tape: QuantumTape,
    classical_processing_fn: Optional[callable] = None,
    auto_cutter: Union[bool, Callable] = False,
    max_depth: int = 1,
    shots: Optional[int] = None,
    device_wires: Optional[Wires] = None,
    **kwargs,
) -> Tuple[Tuple[QuantumTape], Callable]:
    """
    Cut up a circuit containing sample measurements into smaller fragments using a
    Monte Carlo method.

    Following the approach of `Peng et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.150504>`__,
    strategic placement of :class:`~.WireCut` operations can allow a quantum circuit to be split
    into disconnected circuit fragments. A circuit containing sample measurements can be cut and
    processed using Monte Carlo (MC) methods. This transform employs MC methods to allow for sampled measurement
    outcomes to be recombined to full bitstrings and, if a classical processing function is supplied,
    an expectation value will be evaluated.

    Args:
        tape (QuantumTape): the tape of the full circuit to be cut
        classical_processing_fn (callable): A classical postprocessing function to be applied to
            the reconstructed bitstrings. The expected input is a bitstring; a flat array of length ``wires``,
            and the output should be a single number within the interval :math:`[-1, 1]`.
            If not supplied, the transform will output samples.
        auto_cutter (Union[bool, Callable]): Toggle for enabling automatic cutting with the default
            :func:`~.kahypar_cut` partition method. Can also pass a graph partitioning function that
            takes an input graph and returns a list of edges to be cut based on a given set of
            constraints and objective. The default :func:`~.kahypar_cut` function requires KaHyPar to
            be installed using ``pip install kahypar`` for Linux and Mac users or visiting the
            instructions `here <https://kahypar.org>`__ to compile from source for Windows users.
        max_depth (int): The maximum depth used to expand the circuit while searching for wire cuts.
            Only applicable when transforming a QNode.
        shots (int): Number of shots. When transforming a QNode, this argument is
            set by the device's ``shots`` value or at QNode call time (if provided).
            Required when transforming a tape.
        device_wires (Wires): Wires of the device that the cut circuits are to be run on.
            When transforming a QNode, this argument is optional and will be set to the
            QNode's device wires. Required when transforming a tape.
        kwargs: Additional keyword arguments to be passed to a callable ``auto_cutter`` argument.
            For the default KaHyPar cutter, please refer to the docstring of functions
            :func:`~.find_and_place_cuts` and :func:`~.kahypar_cut` for the available arguments.

    Returns:
        Callable: Function which accepts the same arguments as the QNode.
        When called, this function will sample from the partitioned circuit fragments
        and combine the results using a Monte Carlo method.

    **Example**

    The following :math:`3`-qubit circuit contains a :class:`~.WireCut` operation and a :func:`~.sample`
    measurement. When decorated with ``@qml.cut_circuit_mc``, we can cut the circuit into two
    :math:`2`-qubit fragments:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2, shots=1000)

        @qml.cut_circuit_mc
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(0.89, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

    we can then execute the circuit as usual by calling the QNode:

    >>> x = 0.3
    >>> circuit(x)
    tensor([[1, 1],
            [0, 1],
            [0, 1],
            ...,
            [0, 1],
            [0, 1],
            [0, 1]], requires_grad=True)

    Furthermore, the number of shots can be temporarily altered when calling
    the qnode:

    >>> results = circuit(x, shots=123)
    >>> results.shape
    (123, 2)

    Alternatively, if the optimal wire-cut placement is unknown for an arbitrary circuit, the
    ``auto_cutter`` option can be enabled to make attempts in finding such a optimal cut. The
    following examples shows this capability on the same circuit as above but with the
    :class:`~.WireCut` removed:

    .. code-block:: python

        @qml.cut_circuit_mc(auto_cutter=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(0.89, wires=0)
            qml.RY(0.5, wires=1)
            qml.RX(1.3, wires=2)

            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])

            qml.RX(x, wires=0)
            qml.RY(0.7, wires=1)
            qml.RX(2.3, wires=2)
            return qml.sample(wires=[0, 2])

    >>> results = circuit(x, shots=123)
    >>> results.shape
    (123, 2)


    .. details::
        :title: Usage Details

        Manually placing :class:`~.WireCut` operations and decorating the QNode with the
        ``cut_circuit_mc()`` batch transform is the suggested entrypoint into sampling-based
        circuit cutting using the Monte Carlo method. However,
        advanced users also have the option to work directly with a :class:`~.QuantumTape` and
        manipulate the tape to perform circuit cutting using the below functionality:

        .. autosummary::
            :toctree:

            ~transforms.qcut.tape_to_graph
            ~transforms.qcut.find_and_place_cuts
            ~transforms.qcut.replace_wire_cut_nodes
            ~transforms.qcut.fragment_graph
            ~transforms.qcut.graph_to_tape
            ~transforms.qcut.remap_tape_wires
            ~transforms.qcut.expand_fragment_tapes_mc
            ~transforms.qcut.qcut_processing_fn_sample
            ~transforms.qcut.qcut_processing_fn_mc

        The following shows how these elementary steps are combined as part of the
        ``cut_circuit_mc()`` transform.

        Consider the circuit below:

        .. code-block:: python

            np.random.seed(42)

            with qml.tape.QuantumTape() as tape:
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=1)
                qml.WireCut(wires=1)
                qml.CNOT(wires=[1, 2])
                qml.sample(wires=[0, 1, 2])

        >>> print(tape.draw())
        0: ──H─╭●───────────┤ ╭Sample
        1: ────╰X──X──//─╭●─┤ ├Sample
        2: ──────────────╰X─┤ ╰Sample

        To cut the circuit, we first convert it to its graph representation:

        >>> graph = qml.transforms.qcut.tape_to_graph(tape)

        If, however, the optimal location of the :class:`~.WireCut` is unknown, we can use
        :func:`~.find_and_place_cuts` to make attempts in automatically finding such a cut
        given the device constraints. Using the same circuit as above but with the
        :class:`~.WireCut` removed, a slightly different cut with identical cost can be discovered
        and placed into the circuit with automatic cutting:

        .. code-block:: python

            with qml.tape.QuantumTape() as uncut_tape:
                qml.Hadamard(wires=0)
                qml.CNOT(wires=[0, 1])
                qml.PauliX(wires=1)
                qml.CNOT(wires=[1, 2])
                qml.sample(wires=[0, 1, 2])

        >>> cut_graph = qml.transforms.qcut.find_and_place_cuts(
        ...     graph=qml.transforms.qcut.tape_to_graph(uncut_tape),
        ...     cut_strategy=qml.transforms.qcut.CutStrategy(max_free_wires=2),
        ... )
        >>> print(qml.transforms.qcut.graph_to_tape(cut_graph).draw())
         0: ──H─╭●───────────┤  Sample[|1⟩⟨1|]
         1: ────╰X──//──X─╭●─┤  Sample[|1⟩⟨1|]
         2: ──────────────╰X─┤  Sample[|1⟩⟨1|]

        Our next step, using the original manual cut placement, is to remove the :class:`~.WireCut`
        nodes in the graph and replace with :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs.

        >>> qml.transforms.qcut.replace_wire_cut_nodes(graph)

        The :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs are placeholder operations that
        allow us to cut the circuit graph and then randomly select measurement and preparation
        configurations at cut locations. First, the :func:`~.fragment_graph` function pulls apart
        the graph into disconnected components as well as returning the
        `communication_graph <https://en.wikipedia.org/wiki/Quotient_graph>`__
        detailing the connectivity between the components.

        >>> fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)

        We now convert the ``fragments`` back to :class:`~.QuantumTape` objects

        >>> fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]

        The circuit fragments can now be visualized:

        >>> print(fragment_tapes[0].draw())
        0: ──H─╭●─────────────────┤  Sample[|1⟩⟨1|]
        1: ────╰X──X──MeasureNode─┤

        >>> print(fragment_tapes[1].draw())
        1: ──PrepareNode─╭●─┤  Sample[|1⟩⟨1|]
        2: ──────────────╰X─┤  Sample[|1⟩⟨1|]

        Additionally, we must remap the tape wires to match those available on our device.

        >>> dev = qml.device("default.qubit", wires=2, shots=1)
        >>> fragment_tapes = [
        ...     qml.transforms.qcut.remap_tape_wires(t, dev.wires) for t in fragment_tapes
        ... ]

        Note that the number of shots on the device is set to :math:`1` here since we
        will only require one execution per fragment configuration. In the
        following steps we introduce a shots value that will determine the number
        of fragment configurations. When using the ``cut_circuit_mc()`` decorator
        with a QNode, this shots value is automatically inferred from the provided
        device.

        Next, each circuit fragment is randomly expanded over :class:`~.MeasureNode` and
        :class:`~.PrepareNode` configurations. For each pair, a measurement is sampled from
        the Pauli basis and a state preparation is sampled from the corresponding pair of eigenstates.

        A settings array is also given which tracks the configuration pairs. Since each of the 4
        measurements has 2 possible eigenvectors, all configurations can be uniquely identified by
        8 values. The number of rows is determined by the number of cuts and the number of columns
        is determined by the number of shots.

        >>> shots = 3
        >>> configurations, settings = qml.transforms.qcut.expand_fragment_tapes_mc(
        ...     fragment_tapes, communication_graph, shots=shots
        ... )
        >>> tapes = tuple(tape for c in configurations for tape in c)
        >>> settings
        tensor([[6, 3, 4]], requires_grad=True)

        Each configuration is drawn below:

        >>> for t in tapes:
        ...     print(qml.drawer.tape_text(t))
        ...     print("")

        .. code-block::

            0: ──H─╭●────┤  Sample[|1⟩⟨1|]
            1: ────╰X──X─┤  Sample[Z]

            0: ──H─╭●────┤  Sample[|1⟩⟨1|]
            1: ────╰X──X─┤  Sample[X]

            0: ──H─╭●────┤  Sample[|1⟩⟨1|]
            1: ────╰X──X─┤  Sample[Y]

            0: ──I─╭●─┤  Sample[|1⟩⟨1|]
            1: ────╰X─┤  Sample[|1⟩⟨1|]

            0: ──X──S─╭●─┤  Sample[|1⟩⟨1|]
            1: ───────╰X─┤  Sample[|1⟩⟨1|]

            0: ──H─╭●─┤  Sample[|1⟩⟨1|]
            1: ────╰X─┤  Sample[|1⟩⟨1|]

        The last step is to execute the tapes and postprocess the results using
        :func:`~.qcut_processing_fn_sample`, which processes the results to approximate the original full circuit
        output bitstrings.

        >>> results = qml.execute(tapes, dev, gradient_fn=None)
        >>> qml.transforms.qcut.qcut_processing_fn_sample(
        ...     results,
        ...     communication_graph,
        ...     shots=shots,
        ... )
        [array([[0., 0., 0.],
                [1., 0., 0.],
                [1., 0., 0.]])]

        Alternatively, it is possible to calculate an expectation value if a classical
        processing function is provided that will accept the reconstructed circuit bitstrings
        and return a value in the interval :math:`[-1, 1]`:

        .. code-block::

            def fn(x):
                if x[0] == 0:
                    return 1
                if x[0] == 1:
                    return -1

        >>> qml.transforms.qcut.qcut_processing_fn_mc(
        ...     results,
        ...     communication_graph,
        ...     settings,
        ...     shots,
        ...     fn
        ... )
        array(-4.)

        Using the Monte Carlo approach of [Peng et. al](https://arxiv.org/abs/1904.00102), the
        `cut_circuit_mc` transform also supports returning sample-based expectation values of
        observables that are diagonal in the computational basis, as shown below for a `ZZ` measurement
        on wires `0` and `2`:

        .. code-block::

            dev = qml.device("default.qubit", wires=2, shots=10000)

            def observable(bitstring):
                return (-1) ** np.sum(bitstring)

            @qml.cut_circuit_mc(classical_processing_fn=observable)
            @qml.qnode(dev)
            def circuit(x):
                qml.RX(0.89, wires=0)
                qml.RY(0.5, wires=1)
                qml.RX(1.3, wires=2)

                qml.CNOT(wires=[0, 1])
                qml.WireCut(wires=1)
                qml.CNOT(wires=[1, 2])

                qml.RX(x, wires=0)
                qml.RY(0.7, wires=1)
                qml.RX(2.3, wires=2)
                return qml.sample(wires=[0, 2])

        We can now approximate the expectation value of the observable using

        >>> circuit(x)
        tensor(-0.776, requires_grad=True)
    """
    # pylint: disable=unused-argument, too-many-arguments

    if len(tape.measurements) != 1:
        raise ValueError(
            "The Monte Carlo circuit cutting workflow only supports circuits "
            "with a single output measurement"
        )

    if not all(m.return_type is Sample for m in tape.measurements):
        raise ValueError(
            "The Monte Carlo circuit cutting workflow only supports circuits "
            "with sampling-based measurements"
        )

    for meas in tape.measurements:
        if meas.obs is not None:
            raise ValueError(
                "The Monte Carlo circuit cutting workflow only "
                "supports measurements in the computational basis. Please only specify "
                "wires to be sampled within qml.sample(), do not pass observables."
            )

    g = tape_to_graph(tape)

    if auto_cutter is True or callable(auto_cutter):

        cut_strategy = kwargs.pop("cut_strategy", None) or CutStrategy(
            max_free_wires=len(device_wires)
        )

        g = find_and_place_cuts(
            graph=g,
            cut_method=auto_cutter if callable(auto_cutter) else kahypar_cut,
            cut_strategy=cut_strategy,
            **kwargs,
        )

    replace_wire_cut_nodes(g)
    fragments, communication_graph = fragment_graph(g)
    fragment_tapes = [graph_to_tape(f) for f in fragments]
    fragment_tapes = [remap_tape_wires(t, device_wires) for t in fragment_tapes]

    configurations, settings = expand_fragment_tapes_mc(
        fragment_tapes, communication_graph, shots=shots
    )

    tapes = tuple(tape for c in configurations for tape in c)

    if classical_processing_fn:
        return tapes, partial(
            qcut_processing_fn_mc,
            communication_graph=communication_graph,
            settings=settings,
            shots=shots,
            classical_processing_fn=classical_processing_fn,
        )

    return tapes, partial(
        qcut_processing_fn_sample, communication_graph=communication_graph, shots=shots
    )


@cut_circuit_mc.custom_qnode_wrapper
def qnode_execution_wrapper_mc(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to replace execution variables"""

    transform_max_diff = tkwargs.pop("max_diff", None)
    tkwargs.setdefault("device_wires", qnode.device.wires)

    if "shots" in inspect.signature(qnode.func).parameters:
        raise ValueError(
            "Detected 'shots' as an argument of the quantum function to transform. "
            "The 'shots' argument name is reserved for overriding the number of shots "
            "taken by the device."
        )

    def _wrapper(*args, **kwargs):
        if tkwargs.get("shots", False):
            raise ValueError(
                "Cannot provide a 'shots' value directly to the cut_circuit_mc "
                "decorator when transforming a QNode. Please provide the number of shots in "
                "the device or when calling the QNode."
            )

        shots = kwargs.pop("shots", False)
        shots = shots or qnode.device.shots

        if shots is None:
            raise ValueError(
                "A shots value must be provided in the device "
                "or when calling the QNode to be cut"
            )

        qnode.construct(args, kwargs)
        tapes, processing_fn = self.construct(qnode.qtape, *targs, **tkwargs, shots=shots)

        interface = qnode.interface
        execute_kwargs = getattr(qnode, "execute_kwargs", {}).copy()
        max_diff = execute_kwargs.pop("max_diff", 2)
        max_diff = transform_max_diff or max_diff

        gradient_fn = getattr(qnode, "gradient_fn", qnode.diff_method)
        gradient_kwargs = getattr(qnode, "gradient_kwargs", {})

        if interface is None or not self.differentiable:
            gradient_fn = None

        execute_kwargs["cache"] = False

        res = qml.execute(
            tapes,
            device=qnode.device,
            gradient_fn=gradient_fn,
            interface=interface,
            max_diff=max_diff,
            override_shots=1,
            gradient_kwargs=gradient_kwargs,
            **execute_kwargs,
        )

        out = processing_fn(res)
        if isinstance(out, list) and len(out) == 1:
            return out[0]
        return out

    return _wrapper


def _get_symbol(i):
    """Finds the i-th ASCII symbol. Works for lowercase and uppercase letters, allowing i up to
    51."""
    if i >= len(string.ascii_letters):
        raise ValueError(
            "Set the use_opt_einsum argument to True when applying more than "
            f"{len(string.ascii_letters)} wire cuts to a circuit"
        )
    return string.ascii_letters[i]


# pylint: disable=too-many-branches
def contract_tensors(
    tensors: Sequence,
    communication_graph: MultiDiGraph,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
    use_opt_einsum: bool = False,
):
    r"""Contract tensors according to the edges specified in the communication graph.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Consider the three tensors :math:`T^{(1)}`, :math:`T^{(2)}`, and :math:`T^{(3)}`, along with
    their contraction equation

    .. math::

        \sum_{ijklmn} T^{(1)}_{ij,km} T^{(2)}_{kl,in} T^{(3)}_{mn,jl}

    Each tensor is the result of the tomography of a circuit fragment and has some indices
    corresponding to state preparations (marked by the indices before the comma) and some indices
    corresponding to measurements (marked by the indices after the comma).

    An equivalent representation of the contraction equation is to use a directed multigraph known
    as the communication/quotient graph. In the communication graph, each tensor is assigned a node
    and edges are added between nodes to mark a contraction along an index. The communication graph
    resulting from the above contraction equation is a complete directed graph.

    In the communication graph provided by :func:`fragment_graph`, edges are composed of
    :class:`PrepareNode` and :class:`MeasureNode` pairs. To correctly map back to the contraction
    equation, we must keep track of the order of preparation and measurement indices in each tensor.
    This order is specified in the ``prepare_nodes`` and ``measure_nodes`` arguments.

    Args:
        tensors (Sequence): the tensors to be contracted
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between the tensors
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of preparation indices in
            each tensor
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of measurement indices in
            each tensor
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.

    Returns:
        float or tensor_like: the result of contracting the tensor network

    **Example**

    We first set up the tensors and their corresponding :class:`~.PrepareNode` and
    :class:`~.MeasureNode` orderings:

    .. code-block:: python

        from pennylane.transforms import qcut
        import networkx as nx
        import numpy as np

        tensors = [np.arange(4), np.arange(4, 8)]
        prep = [[], [qcut.PrepareNode(wires=0)]]
        meas = [[qcut.MeasureNode(wires=0)], []]

    The communication graph describing edges in the tensor network must also be constructed:

    .. code-block:: python

        graph = nx.MultiDiGraph([(0, 1, {"pair": (meas[0][0], prep[1][0])})])

    The network can then be contracted using:

    >>> qml.transforms.qcut.contract_tensors(tensors, graph, prep, meas)
    38
    """
    # pylint: disable=import-outside-toplevel
    if use_opt_einsum:
        try:
            from opt_einsum import contract, get_symbol
        except ImportError as e:
            raise ImportError(
                "The opt_einsum package is required when use_opt_einsum is set to "
                "True in the contract_tensors function. This package can be "
                "installed using:\npip install opt_einsum"
            ) from e
    else:
        contract = qml.math.einsum
        get_symbol = _get_symbol

    ctr = 0
    tensor_indxs = [""] * len(communication_graph.nodes)

    meas_map = {}

    for i, (node, prep) in enumerate(zip(communication_graph.nodes, prepare_nodes)):
        predecessors = communication_graph.pred[node]

        for p in prep:
            for _, pred_edges in predecessors.items():
                for pred_edge in pred_edges.values():
                    meas_op, prep_op = pred_edge["pair"]

                    if p.id is prep_op.id:
                        symb = get_symbol(ctr)
                        ctr += 1
                        tensor_indxs[i] += symb
                        meas_map[meas_op] = symb

    for i, (node, meas) in enumerate(zip(communication_graph.nodes, measure_nodes)):
        successors = communication_graph.succ[node]

        for m in meas:
            for _, succ_edges in successors.items():
                for succ_edge in succ_edges.values():
                    meas_op, _ = succ_edge["pair"]

                    if m.id is meas_op.id:
                        symb = meas_map[meas_op]
                        tensor_indxs[i] += symb

    eqn = ",".join(tensor_indxs)
    kwargs = {} if use_opt_einsum else {"like": tensors[0]}

    return contract(eqn, *tensors, **kwargs)


CHANGE_OF_BASIS = qml.math.array(
    [[1.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 2.0, 0.0], [-1.0, -1.0, 0.0, 2.0], [1.0, -1.0, 0.0, 0.0]]
)


def _process_tensor(results, n_prep: int, n_meas: int):
    """Convert a flat slice of an individual circuit fragment's execution results into a tensor.

    This function performs the following steps:

    1. Reshapes ``results`` into the intermediate shape ``(4,) * n_prep + (4**n_meas,)``
    2. Shuffles the final axis to follow the standard product over measurement settings. E.g., for
      ``n_meas = 2`` the standard product is: II, IX, IY, IZ, XI, ..., ZY, ZZ while the input order
      will be the result of ``qml.grouping.partition_pauli_group(2)``, i.e., II, IZ, ZI, ZZ, ...,
      YY.
    3. Reshapes into the final target shape ``(4,) * (n_prep + n_meas)``
    4. Performs a change of basis for the preparation indices (the first ``n_prep`` indices) from
       the |0>, |1>, |+>, |+i> basis to the I, X, Y, Z basis using ``CHANGE_OF_BASIS``.

    Args:
        results (tensor_like): the input execution results
        n_prep (int): the number of preparation nodes in the corresponding circuit fragment
        n_meas (int): the number of measurement nodes in the corresponding circuit fragment

    Returns:
        tensor_like: the corresponding fragment tensor
    """
    n = n_prep + n_meas
    dim_meas = 4**n_meas

    # Step 1
    intermediate_shape = (4,) * n_prep + (dim_meas,)
    intermediate_tensor = qml.math.reshape(results, intermediate_shape)

    # Step 2
    grouped = qml.grouping.partition_pauli_group(n_meas)
    grouped_flat = [term for group in grouped for term in group]
    order = qml.math.argsort(grouped_flat)

    if qml.math.get_interface(intermediate_tensor) == "tensorflow":
        # TensorFlow does not support slicing
        intermediate_tensor = qml.math.gather(intermediate_tensor, order, axis=-1)
    else:
        sl = [slice(None)] * n_prep + [order]
        intermediate_tensor = intermediate_tensor[tuple(sl)]

    # Step 3
    final_shape = (4,) * n
    final_tensor = qml.math.reshape(intermediate_tensor, final_shape)

    # Step 4
    change_of_basis = qml.math.convert_like(CHANGE_OF_BASIS, intermediate_tensor)

    for i in range(n_prep):
        axes = [[1], [i]]
        final_tensor = qml.math.tensordot(change_of_basis, final_tensor, axes=axes)

    axes = list(reversed(range(n_prep))) + list(range(n_prep, n))

    # Use transpose to reorder indices. We must do this because tensordot returns a tensor whose
    # indices are ordered according to the uncontracted indices of the first tensor, followed
    # by the uncontracted indices of the second tensor. For example, calculating C_kj T_ij returns
    # a tensor T'_ki rather than T'_ik.
    final_tensor = qml.math.transpose(final_tensor, axes=axes)

    final_tensor *= qml.math.power(2, -(n_meas + n_prep) / 2)
    return final_tensor


def _to_tensors(
    results,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
) -> List:
    """Process a flat list of execution results from all circuit fragments into the corresponding
    tensors.

    This function slices ``results`` according to the expected size of fragment tensors derived from
    the ``prepare_nodes`` and ``measure_nodes`` and then passes onto ``_process_tensor`` for further
    transformation.

    Args:
        results (tensor_like): A collection of execution results, provided as a flat tensor,
            corresponding to the expansion of circuit fragments in the communication graph over
            measurement and preparation node configurations. These results are processed into
            tensors by this function.
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence whose length is equal to the
            number of circuit fragments, with each element used here to determine the number of
            preparation nodes in a given fragment
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence whose length is equal to the
            number of circuit fragments, with each element used here to determine the number of
            measurement nodes in a given fragment

    Returns:
        List[tensor_like]: the tensors for each circuit fragment in the communication graph
    """
    ctr = 0
    tensors = []

    for p, m in zip(prepare_nodes, measure_nodes):
        n_prep = len(p)
        n_meas = len(m)
        n = n_prep + n_meas

        dim = 4**n
        results_slice = results[ctr : dim + ctr]

        tensors.append(_process_tensor(results_slice, n_prep, n_meas))

        ctr += dim

    if results.shape[0] != ctr:
        raise ValueError(f"The results argument should be a flat list of length {ctr}")

    return tensors


def qcut_processing_fn(
    results: Sequence[Sequence],
    communication_graph: MultiDiGraph,
    prepare_nodes: Sequence[Sequence[PrepareNode]],
    measure_nodes: Sequence[Sequence[MeasureNode]],
    use_opt_einsum: bool = False,
):
    """Processing function for the :func:`cut_circuit() <pennylane.cut_circuit>` transform.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        results (Sequence[Sequence]): A collection of execution results generated from the
            expansion of circuit fragments over measurement and preparation node configurations.
            These results are processed into tensors and then contracted.
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of preparation indices in
            each tensor
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of measurement indices in
            each tensor
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.

    Returns:
        float or tensor_like: the output of the original uncut circuit arising from contracting
        the tensor network of circuit fragments
    """
    flat_results = qml.math.concatenate(results)

    tensors = _to_tensors(flat_results, prepare_nodes, measure_nodes)
    result = contract_tensors(
        tensors, communication_graph, prepare_nodes, measure_nodes, use_opt_einsum
    )
    return result


@batch_transform
def cut_circuit(
    tape: QuantumTape,
    auto_cutter: Union[bool, Callable] = False,
    use_opt_einsum: bool = False,
    device_wires: Optional[Wires] = None,
    max_depth: int = 1,
    **kwargs,
) -> Tuple[Tuple[QuantumTape], Callable]:
    """
    Cut up a quantum circuit into smaller circuit fragments.

    Following the approach outlined in Theorem 2 of
    `Peng et al. <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.150504>`__,
    strategic placement of :class:`~.WireCut` operations can allow a quantum circuit to be split
    into disconnected circuit fragments. Each circuit fragment is then executed multiple times by
    varying the state preparations and measurements at incoming and outgoing cut locations,
    respectively, resulting in a process tensor describing the action of the fragment. The process
    tensors are then contracted to provide the result of the original uncut circuit.

    .. note::

        Only circuits that return a single expectation value are supported.

    Args:
        tape (QuantumTape): the tape of the full circuit to be cut
        auto_cutter (Union[bool, Callable]): Toggle for enabling automatic cutting with the default
            :func:`~.kahypar_cut` partition method. Can also pass a graph partitioning function that
            takes an input graph and returns a list of edges to be cut based on a given set of
            constraints and objective. The default :func:`~.kahypar_cut` function requires KaHyPar to
            be installed using ``pip install kahypar`` for Linux and Mac users or visiting the
            instructions `here <https://kahypar.org>`__ to compile from source for Windows users.
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.
        device_wires (Wires): Wires of the device that the cut circuits are to be run on.
            When transforming a QNode, this argument is optional and will be set to the
            QNode's device wires. Required when transforming a tape.
        max_depth (int): The maximum depth used to expand the circuit while searching for wire cuts.
            Only applicable when transforming a QNode.
        kwargs: Additional keyword arguments to be passed to a callable ``auto_cutter`` argument.
            For the default KaHyPar cutter, please refer to the docstring of functions
            :func:`~.find_and_place_cuts` and :func:`~.kahypar_cut` for the available arguments.

    Returns:
        Callable: Function which accepts the same arguments as the QNode.
        When called, this function will perform a process tomography of the
        partitioned circuit fragments and combine the results via tensor
        contractions.

    **Example**

    The following :math:`3`-qubit circuit contains a :class:`~.WireCut` operation. When decorated
    with ``@qml.cut_circuit``, we can cut the circuit into two :math:`2`-qubit fragments:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.cut_circuit
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.9, wires=1)
            qml.RX(0.3, wires=2)

            qml.CZ(wires=[0, 1])
            qml.RY(-0.4, wires=0)

            qml.WireCut(wires=1)

            qml.CZ(wires=[1, 2])

            return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))

    Executing ``circuit`` will run multiple configurations of the :math:`2`-qubit fragments which
    are then postprocessed to give the result of the original circuit:

    >>> x = np.array(0.531, requires_grad=True)
    >>> circuit(x)
    0.47165198882111165

    Futhermore, the output of the cut circuit is also differentiable:

    >>> qml.grad(circuit)(x)
    -0.276982865449393

    Alternatively, if the optimal wire-cut placement is unknown for an arbitrary circuit, the
    ``auto_cutter`` option can be enabled to make attempts in finding such an optimal cut. The
    following examples shows this capability on the same circuit as above but with the
    :class:`~.WireCut` removed:

    .. code-block:: python

        @qml.cut_circuit(auto_cutter=True)
        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.9, wires=1)
            qml.RX(0.3, wires=2)

            qml.CZ(wires=[0, 1])
            qml.RY(-0.4, wires=0)

            qml.CZ(wires=[1, 2])

            return qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))

    >>> x = np.array(0.531, requires_grad=True)
    >>> circuit(x)
    0.47165198882111165
    >>> qml.grad(circuit)(x)
    -0.276982865449393

    .. details::
        :title: Usage Details

        Manually placing :class:`~.WireCut` operations and decorating the QNode with the
        ``cut_circuit()`` batch transform is the suggested entrypoint into circuit cutting. However,
        advanced users also have the option to work directly with a :class:`~.QuantumTape` and
        manipulate the tape to perform circuit cutting using the below functionality:

        .. autosummary::
            :toctree:

            ~transforms.qcut.tape_to_graph
            ~transforms.qcut.find_and_place_cuts
            ~transforms.qcut.replace_wire_cut_nodes
            ~transforms.qcut.fragment_graph
            ~transforms.qcut.graph_to_tape
            ~transforms.qcut.remap_tape_wires
            ~transforms.qcut.expand_fragment_tape
            ~transforms.qcut.qcut_processing_fn
            ~transforms.qcut.CutStrategy

        The following shows how these elementary steps are combined as part of the
        ``cut_circuit()`` transform.

        Consider the circuit below:

        .. code-block:: python

            with qml.tape.QuantumTape() as tape:
                qml.RX(0.531, wires=0)
                qml.RY(0.9, wires=1)
                qml.RX(0.3, wires=2)

                qml.CZ(wires=[0, 1])
                qml.RY(-0.4, wires=0)

                qml.WireCut(wires=1)

                qml.CZ(wires=[1, 2])

                qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))

        >>> print(qml.drawer.tape_text(tape))
        0: ──RX─╭●──RY────┤ ╭<Z@Z@Z>
        1: ──RY─╰Z──//─╭●─┤ ├<Z@Z@Z>
        2: ──RX────────╰Z─┤ ╰<Z@Z@Z>

        To cut the circuit, we first convert it to its graph representation:

        >>> graph = qml.transforms.qcut.tape_to_graph(tape)

        .. figure:: ../../_static/qcut_graph.svg
            :align: center
            :width: 60%
            :target: javascript:void(0);

        If, however, the optimal location of the :class:`~.WireCut` is unknown, we can use
        :func:`~.find_and_place_cuts` to make attempts in automatically finding such a cut
        given the device constraints. Using the same circuit as above but with the
        :class:`~.WireCut` removed, the same (optimal) cut can be recovered with automatic
        cutting:

        .. code-block:: python

            with qml.tape.QuantumTape() as uncut_tape:
                qml.RX(0.531, wires=0)
                qml.RY(0.9, wires=1)
                qml.RX(0.3, wires=2)

                qml.CZ(wires=[0, 1])
                qml.RY(-0.4, wires=0)

                qml.CZ(wires=[1, 2])

                qml.expval(qml.grouping.string_to_pauli_word("ZZZ"))

        >>> cut_graph = qml.transforms.qcut.find_and_place_cuts(
                graph = qml.transforms.qcut.tape_to_graph(uncut_tape),
                cut_strategy = qml.transforms.qcut.CutStrategy(max_free_wires=2),
            )
        >>> print(qml.transforms.qcut.graph_to_tape(cut_graph).draw())
        0: ──RX─╭●──RY────┤ ╭<Z@Z@Z>
        1: ──RY─╰Z──//─╭●─┤ ├<Z@Z@Z>
        2: ──RX────────╰Z─┤ ╰<Z@Z@Z>

        Our next step is to remove the :class:`~.WireCut` nodes in the graph and replace with
        :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs.

        >>> qml.transforms.qcut.replace_wire_cut_nodes(graph)

        The :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs are placeholder operations that
        allow us to cut the circuit graph and then iterate over measurement and preparation
        configurations at cut locations. First, the :func:`~.fragment_graph` function pulls apart
        the graph into disconnected components as well as returning the
        `communication_graph <https://en.wikipedia.org/wiki/Quotient_graph>`__
        detailing the connectivity between the components.

        >>> fragments, communication_graph = qml.transforms.qcut.fragment_graph(graph)

        We now convert the ``fragments`` back to :class:`~.QuantumTape` objects

        >>> fragment_tapes = [qml.transforms.qcut.graph_to_tape(f) for f in fragments]

        The circuit fragments can now be visualized:

        >>> print(fragment_tapes[0].draw())
         0: ──RX(0.531)──╭●──RY(-0.4)─────┤ ⟨Z⟩
         1: ──RY(0.9)────╰Z──MeasureNode──┤

        >>> print(fragment_tapes[1].draw())
         2: ──RX(0.3)──────╭Z──╭┤ ⟨Z ⊗ Z⟩
         1: ──PrepareNode──╰●──╰┤ ⟨Z ⊗ Z⟩

        Additionally, we must remap the tape wires to match those available on our device.

        >>> dev = qml.device("default.qubit", wires=2)
        >>> fragment_tapes = [
        ...     qml.transforms.qcut.remap_tape_wires(t, dev.wires) for t in fragment_tapes
        ... ]

        Next, each circuit fragment is expanded over :class:`~.MeasureNode` and
        :class:`~.PrepareNode` configurations and a flat list of tapes is created:

        .. code-block::

            expanded = [qml.transforms.qcut.expand_fragment_tape(t) for t in fragment_tapes]

            configurations = []
            prepare_nodes = []
            measure_nodes = []
            for tapes, p, m in expanded:
                configurations.append(tapes)
                prepare_nodes.append(p)
                measure_nodes.append(m)

            tapes = tuple(tape for c in configurations for tape in c)

        Each configuration is drawn below:

        >>> for t in tapes:
        ...     print(qml.drawer.tape_text(t))

        .. code-block::

             0: ──RX(0.531)──╭●──RY(-0.4)──╭┤ ⟨Z ⊗ I⟩ ╭┤ ⟨Z ⊗ Z⟩
             1: ──RY(0.9)────╰Z────────────╰┤ ⟨Z ⊗ I⟩ ╰┤ ⟨Z ⊗ Z⟩

             0: ──RX(0.531)──╭●──RY(-0.4)──╭┤ ⟨Z ⊗ X⟩
             1: ──RY(0.9)────╰Z────────────╰┤ ⟨Z ⊗ X⟩

             0: ──RX(0.531)──╭●──RY(-0.4)──╭┤ ⟨Z ⊗ Y⟩
             1: ──RY(0.9)────╰Z────────────╰┤ ⟨Z ⊗ Y⟩

             0: ──RX(0.3)──╭Z──╭┤ ⟨Z ⊗ Z⟩
             1: ──I────────╰●──╰┤ ⟨Z ⊗ Z⟩

             0: ──RX(0.3)──╭Z──╭┤ ⟨Z ⊗ Z⟩
             1: ──X────────╰●──╰┤ ⟨Z ⊗ Z⟩

             0: ──RX(0.3)──╭Z──╭┤ ⟨Z ⊗ Z⟩
             1: ──H────────╰●──╰┤ ⟨Z ⊗ Z⟩

             0: ──RX(0.3)─────╭Z──╭┤ ⟨Z ⊗ Z⟩
             1: ──H────────S──╰●──╰┤ ⟨Z ⊗ Z⟩

        The last step is to execute the tapes and postprocess the results using
        :func:`~.qcut_processing_fn`, which processes the results to the original full circuit
        output via a tensor network contraction

        >>> results = qml.execute(tapes, dev, gradient_fn=None)
        >>> qml.transforms.qcut.qcut_processing_fn(
        ...     results,
        ...     communication_graph,
        ...     prepare_nodes,
        ...     measure_nodes,
        ... )
        0.47165198882111165
    """
    # pylint: disable=unused-argument
    if len(tape.measurements) != 1:
        raise ValueError(
            "The circuit cutting workflow only supports circuits with a single output "
            "measurement"
        )

    if not all(m.return_type is Expectation for m in tape.measurements):
        raise ValueError(
            "The circuit cutting workflow only supports circuits with expectation "
            "value measurements"
        )

    if use_opt_einsum:
        try:
            import opt_einsum  # pylint: disable=import-outside-toplevel,unused-import
        except ImportError as e:
            raise ImportError(
                "The opt_einsum package is required when use_opt_einsum is set to "
                "True in the cut_circuit function. This package can be "
                "installed using:\npip install opt_einsum"
            ) from e

    g = tape_to_graph(tape)

    if auto_cutter is True or callable(auto_cutter):

        cut_strategy = kwargs.pop("cut_strategy", None) or CutStrategy(
            max_free_wires=len(device_wires)
        )

        g = find_and_place_cuts(
            graph=g,
            cut_method=auto_cutter if callable(auto_cutter) else kahypar_cut,
            cut_strategy=cut_strategy,
            **kwargs,
        )

    replace_wire_cut_nodes(g)
    fragments, communication_graph = fragment_graph(g)
    fragment_tapes = [graph_to_tape(f) for f in fragments]
    fragment_tapes = [remap_tape_wires(t, device_wires) for t in fragment_tapes]
    expanded = [expand_fragment_tape(t) for t in fragment_tapes]

    configurations = []
    prepare_nodes = []
    measure_nodes = []
    for tapes, p, m in expanded:
        configurations.append(tapes)
        prepare_nodes.append(p)
        measure_nodes.append(m)

    tapes = tuple(tape for c in configurations for tape in c)

    return tapes, partial(
        qcut_processing_fn,
        communication_graph=communication_graph,
        prepare_nodes=prepare_nodes,
        measure_nodes=measure_nodes,
        use_opt_einsum=use_opt_einsum,
    )


@cut_circuit.custom_qnode_wrapper
def qnode_execution_wrapper(self, qnode, targs, tkwargs):
    """Here, we overwrite the QNode execution wrapper in order
    to access the device wires."""
    # pylint: disable=function-redefined

    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_wrapper(qnode, targs, tkwargs)


def _qcut_expand_fn(
    tape: QuantumTape,
    max_depth: int = 1,
    auto_cutter: Union[bool, Callable] = False,
):
    """Expansion function for circuit cutting.

    Expands operations until reaching a depth that includes :class:`~.WireCut` operations.
    """

    for op in tape.operations:
        if isinstance(op, WireCut):
            return tape

    if max_depth > 0:
        return _qcut_expand_fn(tape.expand(), max_depth=max_depth - 1, auto_cutter=auto_cutter)

    if not (auto_cutter is True or callable(auto_cutter)):
        raise ValueError(
            "No WireCut operations found in the circuit. Consider increasing the max_depth value if"
            " operations or nested tapes contain WireCut operations."
        )

    return tape


def _cut_circuit_expand(
    tape: QuantumTape,
    use_opt_einsum: bool = False,
    device_wires: Optional[Wires] = None,
    max_depth: int = 1,
    auto_cutter: Union[bool, Callable] = False,
    **kwargs,
):
    """Main entry point for expanding operations until reaching a depth that
    includes :class:`~.WireCut` operations."""
    # pylint: disable=unused-argument
    return _qcut_expand_fn(tape, max_depth, auto_cutter)


def _cut_circuit_mc_expand(
    tape: QuantumTape,
    classical_processing_fn: Optional[callable] = None,
    max_depth: int = 1,
    shots: Optional[int] = None,
    device_wires: Optional[Wires] = None,
    auto_cutter: Union[bool, Callable] = False,
    **kwargs,
):
    """Main entry point for expanding operations in sample-based tapes until
    reaching a depth that includes :class:`~.WireCut` operations."""
    # pylint: disable=unused-argument, too-many-arguments
    return _qcut_expand_fn(tape, max_depth, auto_cutter)


cut_circuit.expand_fn = _cut_circuit_expand
cut_circuit_mc.expand_fn = _cut_circuit_mc_expand


def remap_tape_wires(tape: QuantumTape, wires: Sequence) -> QuantumTape:
    """Map the wires of a tape to a new set of wires.

    Given an :math:`n`-wire ``tape``, this function returns a new :class:`~.QuantumTape` with
    operations and measurements acting on the first :math:`n` wires provided in the ``wires``
    argument. The input ``tape`` is left unmodified.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        tape (QuantumTape): the quantum tape whose wires should be remapped
        wires (Sequence): the new set of wires to map to

    Returns:
        QuantumTape: A remapped copy of the input tape

    Raises:
        ValueError: if the number of wires in ``tape`` exceeds ``len(wires)``

    **Example**

    Consider the following circuit that operates on wires ``[2, 3]``:

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, wires=2)
            qml.RY(0.6, wires=3)
            qml.CNOT(wires=[2, 3])
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))

    We can map from wires ``[2, 3]`` to ``[0, 1]`` using:

    >>> new_wires = [0, 1]
    >>> new_tape = qml.transforms.qcut.remap_tape_wires(tape, new_wires)
    >>> print(qml.drawer.tape_text(new_tape))
     0: ──RX(0.5)──╭●──╭┤ ⟨Z ⊗ Z⟩
     1: ──RY(0.6)──╰X──╰┤ ⟨Z ⊗ Z⟩
    """
    if len(tape.wires) > len(wires):
        raise ValueError(
            f"Attempting to run a {len(tape.wires)}-wire circuit on a "
            f"{len(wires)}-wire device. Consider increasing the number of wires in "
            f"your device."
        )

    wire_map = dict(zip(tape.wires, wires))
    copy_ops = [copy.copy(op) for op in tape.operations]
    copy_meas = [copy.copy(op) for op in tape.measurements]

    with QuantumTape() as new_tape:
        for op in copy_ops:
            new_wires = Wires([wire_map[w] for w in op.wires])
            op._wires = new_wires
            apply(op)
        for meas in copy_meas:
            obs = meas.obs

            if isinstance(obs, Tensor):
                for obs in obs.obs:
                    new_wires = Wires([wire_map[w] for w in obs.wires])
                    obs._wires = new_wires
            else:
                new_wires = Wires([wire_map[w] for w in obs.wires])
                obs._wires = new_wires
            apply(meas)

    return new_tape


@dataclass()
class CutStrategy:
    """
    A circuit-cutting distribution policy for executing (large) circuits on available (comparably
    smaller) devices.

    .. note::

        This class is part of a work-in-progress feature to support automatic cut placement in the
        circuit cutting workflow. Currently only manual placement of cuts is supported,
        check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        devices (Union[qml.Device, Sequence[qml.Device]]): Single, or Sequence of, device(s).
            Optional only when ``max_free_wires`` is provided.
        max_free_wires (int): Number of wires for the largest available device. Optional only when
            ``devices`` is provided where it defaults to the maximum number of wires among
            ``devices``.
        min_free_wires (int): Number of wires for the smallest available device, or, equivalently,
            the smallest max fragment-wire-size that the partitioning is allowed to explore.
            When provided, this parameter will be used to derive an upper-bound to the range of
            explored number of fragments.  Optional, defaults to 2 which corresponds to attempting
            the most granular partitioning of max 2-wire fragments.
        num_fragments_probed (Union[int, Sequence[int]]): Single, or 2-Sequence of, number(s)
            specifying the potential (range of) number of fragments for the partitioner to attempt.
            Optional, defaults to probing all valid strategies derivable from the circuit and
            devices. When provided, has precedence over all other arguments affecting partitioning
            exploration, such as ``max_free_wires``, ``min_free_wires``, or ``exhaustive``.
        max_free_gates (int): Maximum allowed circuit depth for the deepest available device.
            Optional, defaults to unlimited depth.
        min_free_gates (int): Maximum allowed circuit depth for the shallowest available device.
            Optional, defaults to ``max_free_gates``.
        imbalance_tolerance (float): The global maximum allowed imbalance for all partition trials.
            Optional, defaults to unlimited imbalance. Used only if there's a known hard balancing
            constraint on the partitioning problem.
        trials_per_probe (int): Number of repeated partitioning trials for a random automatic
            cutting method to attempt per set of partitioning parameters. For a deterministic
            cutting method, this can be set to 1. Defaults to 4.

    **Example**

    The following cut strategy specifies that a circuit should be cut into between
    ``2`` to ``5`` fragments, with each fragment having at most ``6`` wires and
    at least ``4`` wires:

    >>> cut_strategy = qml.transforms.CutStrategy(
    ...     max_free_wires=6,
    ...     min_free_wires=4,
    ...     num_fragments_probed=(2, 5),
    ... )

    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes

    #: Initialization argument only, used to derive ``max_free_wires`` and ``min_free_wires``.
    devices: InitVar[Union[qml.Device, Sequence[qml.Device]]] = None

    #: Number of wires for the largest available device.
    max_free_wires: int = None
    #: Number of wires for the smallest available device.
    min_free_wires: int = None
    #: The potential (range of) number of fragments for the partitioner to attempt.
    num_fragments_probed: Union[int, Sequence[int]] = None
    #: Maximum allowed circuit depth for the deepest available device.
    max_free_gates: int = None
    #: Maximum allowed circuit depth for the shallowest available device.
    min_free_gates: int = None
    #: The global maximum allowed imbalance for all partition trials.
    imbalance_tolerance: float = None
    #: Number of trials to repeat for per set of partition parameters probed.
    trials_per_probe: int = 4

    #: Class attribute, threshold for warning about too many fragments.
    HIGH_NUM_FRAGMENTS: ClassVar[int] = 20
    #: Class attribute, threshold for warning about too many partition attempts.
    HIGH_PARTITION_ATTEMPTS: ClassVar[int] = 20

    def __post_init__(
        self,
        devices,
    ):
        """Deriving cutting constraints from given devices and parameters."""

        self.max_free_wires = self.max_free_wires
        if isinstance(self.num_fragments_probed, int):
            self.num_fragments_probed = [self.num_fragments_probed]
        if isinstance(self.num_fragments_probed, (list, tuple)):
            self.num_fragments_probed = sorted(self.num_fragments_probed)
            self.k_lower = self.num_fragments_probed[0]
            self.k_upper = self.num_fragments_probed[-1]
            if self.k_lower <= 0:
                raise ValueError("`num_fragments_probed` must be positive int(s)")
        else:
            self.k_lower, self.k_upper = None, None

        if devices is None and self.max_free_wires is None:
            raise ValueError("One of arguments `devices` and max_free_wires` must be provided.")

        if isinstance(devices, qml.Device):
            devices = (devices,)

        if devices is not None:
            if not isinstance(devices, SequenceType) or any(
                (not isinstance(d, qml.Device) for d in devices)
            ):
                raise ValueError(
                    "Argument `devices` must be a list or tuple containing elements of type "
                    "`qml.Device`"
                )

            device_wire_sizes = [len(d.wires) for d in devices]

            self.max_free_wires = self.max_free_wires or max(device_wire_sizes)
            self.min_free_wires = self.min_free_wires or min(device_wire_sizes)

        if (self.imbalance_tolerance is not None) and not (
            isinstance(self.imbalance_tolerance, (float, int)) and self.imbalance_tolerance >= 0
        ):
            raise ValueError(
                "The overall `imbalance_tolerance` is expected to be a non-negative number, "
                f"got {type(self.imbalance_tolerance)} with value {self.imbalance_tolerance}."
            )

        self.min_free_wires = self.min_free_wires or 1

    def get_cut_kwargs(
        self,
        tape_dag: MultiDiGraph,
        max_wires_by_fragment: Sequence[int] = None,
        max_gates_by_fragment: Sequence[int] = None,
        exhaustive: bool = True,
    ) -> List[Dict[str, Any]]:
        """Derive the complete set of arguments, based on a given circuit, for passing to a graph
        partitioner.

        Args:
            tape_dag (nx.MultiDiGraph): Graph representing a tape, typically the output of
                :func:`tape_to_graph`.
            max_wires_by_fragment (Sequence[int]): User-predetermined list of wire limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            max_gates_by_fragment (Sequence[int]): User-predetermined list of gate limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            exhaustive (bool): Toggle for an exhaustive search which will attempt all potentially
                valid numbers of fragments into which the circuit is partitioned. If ``True``,
                for a circuit with N gates, N - 1 attempts will be made with ``num_fragments``
                ranging from [2, N], i.e. from bi-partitioning to complete partitioning where each
                fragment has exactly a single gate. Defaults to ``True``.

        Returns:
            List[Dict[str, Any]]: A list of minimal kwargs being passed to a graph
            partitioner method.

        **Example**

        Deriving kwargs for a given circuit and feeding them to a custom partitioner, along with
        extra parameters specified using ``extra_kwargs``:

        >>> cut_strategy = qcut.CutStrategy(devices=dev)
        >>> cut_kwargs = cut_strategy.get_cut_kwargs(tape_dag)
        >>> cut_trials = [
        ...     my_partition_fn(tape_dag, **kwargs, **extra_kwargs) for kwargs in cut_kwargs
        ... ]

        """
        wire_depths = {}
        for g in tape_dag.nodes:
            if not isinstance(g, WireCut):
                for w in g.wires:
                    wire_depths[w] = wire_depths.get(w, 0) + 1 / len(g.wires)
        self._validate_input(max_wires_by_fragment, max_gates_by_fragment)

        probed_cuts = self._infer_probed_cuts(
            wire_depths=wire_depths,
            max_wires_by_fragment=max_wires_by_fragment,
            max_gates_by_fragment=max_gates_by_fragment,
            exhaustive=exhaustive,
        )

        return probed_cuts

    @staticmethod
    def _infer_imbalance(k, wire_depths, free_wires, free_gates, imbalance_tolerance=None) -> float:
        """Helper function for determining best imbalance limit."""
        num_wires = len(wire_depths)
        num_gates = sum(wire_depths.values())

        avg_fragment_wires = (num_wires - 1) // k + 1
        avg_fragment_gates = (num_gates - 1) // k + 1
        if free_wires < avg_fragment_wires:
            raise ValueError(
                "`free_wires` should be no less than the average number of wires per fragment. "
                f"Got {free_wires} >= {avg_fragment_wires} ."
            )
        if free_gates < avg_fragment_gates:
            raise ValueError(
                "`free_gates` should be no less than the average number of gates per fragment. "
                f"Got {free_gates} >= {avg_fragment_gates} ."
            )
        if free_gates > num_gates - k:
            # Case where gate depth not limited (`-k` since each fragments has to have >= 1 gates):
            free_gates = num_gates
            # A small adjustment is added to the imbalance factor to prevents small ks from resulting
            # in extremely unbalanced fragments. It will heuristically force the smallest fragment size
            # to be >= 3 if the average fragment size is greater than 5. In other words, tiny fragments
            # are only allowed when average fragmeng size is small in the first place.
            balancing_adjustment = 2 if avg_fragment_gates > 5 else 0
            free_gates = free_gates - (k - 1 + balancing_adjustment)

        depth_imbalance = max(wire_depths.values()) * num_wires / num_gates - 1
        max_imbalance = free_gates / avg_fragment_gates - 1
        imbalance = min(depth_imbalance, max_imbalance)
        if imbalance_tolerance is not None:
            imbalance = min(imbalance, imbalance_tolerance)

        return imbalance

    @staticmethod
    def _validate_input(
        max_wires_by_fragment,
        max_gates_by_fragment,
    ):
        """Helper parameter checker."""
        if max_wires_by_fragment is not None:
            if not isinstance(max_wires_by_fragment, (list, tuple)):
                raise ValueError(
                    "`max_wires_by_fragment` is expected to be a list or tuple, but got "
                    f"{type(max_gates_by_fragment)}."
                )
            if any(not (isinstance(i, int) and i > 0) for i in max_wires_by_fragment):
                raise ValueError(
                    "`max_wires_by_fragment` is expected to contain positive integers only."
                )
        if max_gates_by_fragment is not None:
            if not isinstance(max_gates_by_fragment, (list, tuple)):
                raise ValueError(
                    "`max_gates_by_fragment` is expected to be a list or tuple, but got "
                    f"{type(max_gates_by_fragment)}."
                )
            if any(not (isinstance(i, int) and i > 0) for i in max_gates_by_fragment):
                raise ValueError(
                    "`max_gates_by_fragment` is expected to contain positive integers only."
                )
        if max_wires_by_fragment is not None and max_gates_by_fragment is not None:
            if len(max_wires_by_fragment) != len(max_gates_by_fragment):
                raise ValueError(
                    "The lengths of `max_wires_by_fragment` and `max_gates_by_fragment` should be "
                    f"equal, but got {len(max_wires_by_fragment)} and {len(max_gates_by_fragment)}."
                )

    def _infer_probed_cuts(
        self,
        wire_depths,
        max_wires_by_fragment=None,
        max_gates_by_fragment=None,
        exhaustive=True,
    ) -> List[Dict[str, Any]]:
        """
        Helper function for deriving the minimal set of best default partitioning constraints
        for the graph partitioner.

        Args:
            num_tape_wires (int): Number of wires in the circuit tape to be partitioned.
            num_tape_gates (int): Number of gates in the circuit tape to be partitioned.
            max_wires_by_fragment (Sequence[int]): User-predetermined list of wire limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            max_gates_by_fragment (Sequence[int]): User-predetermined list of gate limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            exhaustive (bool): Toggle for an exhaustive search which will attempt all potentially
                valid numbers of fragments into which the circuit is partitioned. If ``True``,
                ``num_tape_gates - 1`` attempts will be made with ``num_fragments`` ranging from
                [2, ``num_tape_gates``], i.e. from bi-partitioning to complete partitioning where
                each fragment has exactly a single gate. Defaults to ``True``.

        Returns:
            List[Dict[str, Any]]: A list of minimal set of kwargs being passed to a graph
                partitioner method.
        """

        num_tape_wires = len(wire_depths)
        num_tape_gates = int(sum(wire_depths.values()))

        # Assumes unlimited width/depth if not supplied.
        max_free_wires = self.max_free_wires or num_tape_wires
        max_free_gates = self.max_free_gates or num_tape_gates

        # Assumes same number of wires/gates across all devices if min_free_* not provided.
        min_free_wires = self.min_free_wires or max_free_wires
        min_free_gates = self.min_free_gates or max_free_gates

        # The lower bound of k corresponds to executing each fragment on the largest available
        # device.
        k_lb = 1 + max(
            (num_tape_wires - 1) // max_free_wires,  # wire limited
            (num_tape_gates - 1) // max_free_gates,  # gate limited
        )
        # The upper bound of k corresponds to executing each fragment on the smallest available
        # device.
        k_ub = 1 + max(
            (num_tape_wires - 1) // min_free_wires,  # wire limited
            (num_tape_gates - 1) // min_free_gates,  # gate limited
        )

        if exhaustive:
            k_lb = max(2, k_lb)
            k_ub = num_tape_gates

        # The global imbalance tolerance, if not given, defaults to a very loose upper bound:
        imbalance_tolerance = k_ub if self.imbalance_tolerance is None else self.imbalance_tolerance

        probed_cuts = []

        if max_gates_by_fragment is None and max_wires_by_fragment is None:

            # k_lower, when supplied by a user, can be higher than k_lb if the the desired k is known:
            k_lower = self.k_lower if self.k_lower is not None else k_lb
            # k_upper, when supplied by a user, can be higher than k_ub to encourage exploration:
            k_upper = self.k_upper if self.k_upper is not None else k_ub

            if k_lower < k_lb:
                warnings.warn(
                    f"The provided `k_lower={k_lower}` is less than the lowest allowed value, "
                    f"will override and set `k_lower={k_lb}`."
                )
                k_lower = k_lb

            if k_lower > self.HIGH_NUM_FRAGMENTS:
                warnings.warn(
                    f"The attempted number of fragments seems high with lower bound at {k_lower}."
                )

            # Prepare the list of ks to explore:
            ks = list(range(k_lower, k_upper + 1))

            if len(ks) > self.HIGH_PARTITION_ATTEMPTS:
                warnings.warn(f"The numer of partition attempts seems high ({len(ks)}).")
        else:
            # When the by-fragment wire and/or gate limits are supplied, derive k and imbalance and
            # return a single partition config.
            ks = [len(max_wires_by_fragment or max_gates_by_fragment)]

        for k in ks:
            imbalance = self._infer_imbalance(
                k,
                wire_depths,
                max_free_wires if max_wires_by_fragment is None else max(max_wires_by_fragment),
                max_free_gates if max_gates_by_fragment is None else max(max_gates_by_fragment),
                imbalance_tolerance,
            )
            cut_kwargs = {
                "num_fragments": k,
                "imbalance": imbalance,
            }
            if max_wires_by_fragment is not None:
                cut_kwargs["max_wires_by_fragment"] = max_wires_by_fragment
            if max_gates_by_fragment is not None:
                cut_kwargs["max_gates_by_fragment"] = max_gates_by_fragment

            probed_cuts.append(cut_kwargs)

        return probed_cuts


def _graph_to_hmetis(
    graph: MultiDiGraph,
    hyperwire_weight: int = 0,
    edge_weights: Sequence[int] = None,
) -> Tuple[List[int], List[int], List[Union[int, float]]]:
    """Converts a ``MultiDiGraph`` into the
    `hMETIS hypergraph input format <http://glaros.dtc.umn.edu/gkhome/fetch/sw/hmetis/manual.pdf>`__
    conforming to KaHyPar's calling signature.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.
        hyperwire_weight (int): Weight on the artificially appended hyperedges representing wires.
            Defaults to 0 which leads to no such insertion. If greater than 0, hyperedges will be
            appended with the provided weight, to encourage the resulting fragments to cluster gates
            on the same wire together.
        edge_weights (Sequence[int]): Weights for regular edges in the graph. Defaults to ``None``,
            which leads to unit-weighted edges.

    Returns:
        Tuple[List,List,List]: The 3 lists representing an (optionally weighted) hypergraph:
        - Flattened list of adjacent node indices.
        - List of starting indices for edges in the above adjacent-nodes-list.
        - Optional list of edge weights. ``None`` if ``hyperwire_weight`` is equal to 0.
    """

    nodes = list(graph.nodes)
    edges = graph.edges(data="wire")
    wires = {w for _, _, w in edges}

    adj_nodes = [nodes.index(v) for ops in graph.edges(keys=False) for v in ops]
    edge_splits = qml.math.cumsum([0] + [len(e) for e in graph.edges(keys=False)]).tolist()
    edge_weights = (
        edge_weights if edge_weights is not None and len(edges) == len(edge_weights) else None
    )

    if hyperwire_weight:
        hyperwires = {w: set() for w in wires}
        num_wires = len(hyperwires)

        for v0, v1, wire in edges:
            hyperwires[wire].update([nodes.index(v0), nodes.index(v1)])

        for wire, nodes_on_wire in hyperwires.items():
            nwv = len(nodes_on_wire)
            edge_splits.append(nwv + edge_splits[-1])
            adj_nodes = adj_nodes + list(nodes_on_wire)
        assert len(edge_splits) == len(edges) + num_wires + 1

        if isinstance(hyperwire_weight, (int, float)):
            # assumes original edges having unit weights by default:
            edge_weights = edge_weights or ([1] * len(edges))
            wire_weights = [hyperwire_weight] * num_wires
            edge_weights = edge_weights + wire_weights

    return adj_nodes, edge_splits, edge_weights


def kahypar_cut(
    graph: MultiDiGraph,
    num_fragments: int,
    imbalance: int = None,
    edge_weights: List[Union[int, float]] = None,
    node_weights: List[Union[int, float]] = None,
    fragment_weights: List[Union[int, float]] = None,
    hyperwire_weight: int = 1,
    seed: int = None,
    config_path: Union[str, Path] = None,
    trial: int = None,
    verbose: bool = False,
) -> List[Tuple[Operation, Operation, Any]]:
    """Calls `KaHyPar <https://kahypar.org/>`__ to partition a graph.

    .. warning::
        Requires KaHyPar to be installed separately. For Linux and Mac users,
        KaHyPar can be installed using ``pip install kahypar``. Windows users
        can follow the instructions
        `here <https://kahypar.org>`__ to compile from source.

    Args:
        graph (nx.MultiDiGraph): The graph to be partitioned.
        num_fragments (int): Desired number of fragments.
        imbalance (int): Imbalance factor of the partitioning. Defaults to KaHyPar's determination.
        edge_weights (List[Union[int, float]]): Weights for edges. Defaults to unit-weighted edges.
        node_weights (List[Union[int, float]]): Weights for nodes. Defaults to unit-weighted nodes.
        fragment_weights (List[Union[int, float]]): Maximum size constraints by fragment. Defaults
            to no such constraints, with ``imbalance`` the only parameter affecting fragment sizes.
        hyperwire_weight (int): Weight on the artificially appended hyperedges representing wires.
            Setting it to 0 leads to no such insertion. If greater than 0, hyperedges will be
            appended with the provided weight, to encourage the resulting fragments to cluster gates
            on the same wire together. Defaults to 1.
        seed (int): KaHyPar's seed. Defaults to the seed in the config file which defaults to -1,
            i.e. unfixed seed.
        config_path (str): KaHyPar's ``.ini`` config file path. Defaults to its SEA20 paper config.
        trial (int): trial id for summary label creation. Defaults to ``None``.
        verbose (bool): Flag for printing KaHyPar's output summary. Defaults to ``False``.

    Returns:
        List[Union[int, Any]]: List of cut edges.

    **Example**

    Consider the following 2-wire circuit with one CNOT gate connecting the wires:

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires="a")
            qml.CNOT(wires=[0, "a"])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires="a")
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires="a")
            qml.expval(qml.PauliZ(wires=[0]))

    We can let KaHyPar automatically find the optimal edges to place cuts:

    >>> graph = qml.transforms.qcut.tape_to_graph(tape)
    >>> cut_edges = qml.transforms.qcut.kahypar_cut(
            graph=graph,
            num_fragments=2,
        )
    >>> cut_edges
    [(CNOT(wires=[0, 'a']), RZ(0.24, wires=[0]), 0)]
    """
    # pylint: disable=too-many-arguments, import-outside-toplevel
    try:
        import kahypar
    except ImportError as e:
        raise ImportError(
            "KaHyPar must be installed to use this method for automatic "
            "cut placement. Try pip install kahypar or visit "
            "https://kahypar.org/ for installation instructions."
        ) from e

    adjacent_nodes, edge_splits, edge_weights = _graph_to_hmetis(
        graph=graph, hyperwire_weight=hyperwire_weight, edge_weights=edge_weights
    )

    trial = 0 if trial is None else trial
    ne = len(edge_splits) - 1
    nv = max(adjacent_nodes) + 1

    if edge_weights is not None or node_weights is not None:
        edge_weights = edge_weights or [1] * ne
        node_weights = node_weights or [1] * nv
        hypergraph = kahypar.Hypergraph(
            nv,
            ne,
            edge_splits,
            adjacent_nodes,
            num_fragments,
            edge_weights,
            node_weights,
        )

    else:
        hypergraph = kahypar.Hypergraph(nv, ne, edge_splits, adjacent_nodes, num_fragments)

    context = kahypar.Context()

    config_path = config_path or str(Path(__file__).parent / "_cut_kKaHyPar_sea20.ini")
    context.loadINIconfiguration(config_path)

    context.setK(num_fragments)

    if isinstance(imbalance, float):
        context.setEpsilon(imbalance)
    if isinstance(fragment_weights, SequenceType) and (len(fragment_weights) == num_fragments):
        context.setCustomTargetBlockWeights(fragment_weights)
    if not verbose:
        context.suppressOutput(True)

    # KaHyPar fixes seed to 42 by default, need to manually sample seed to randomize:
    kahypar_seed = np.random.default_rng(seed).choice(2**15)
    context.setSeed(kahypar_seed)

    kahypar.partition(hypergraph, context)

    cut_edge_mask = [hypergraph.connectivity(e) > 1 for e in hypergraph.edges()]

    # compress() ignores the extra hyperwires at the end if there is any.
    cut_edges = list(compress(graph.edges, cut_edge_mask))

    if verbose:
        fragment_sizes = [hypergraph.blockSize(p) for p in range(num_fragments)]
        print(len(fragment_sizes), fragment_sizes)

    return cut_edges


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

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.432, wires=0)
            qml.RY(0.543, wires="a")
            qml.CNOT(wires=[0, "a"])
            qml.expval(qml.PauliZ(wires=[0]))

    >>> print(qml.drawer.tape_text(tape))
     0: ──RX(0.432)──╭●──┤ ⟨Z⟩
     a: ──RY(0.543)──╰X──┤

    If we know we want to place a :class:`~.WireCut` node between nodes ``RY(0.543, wires=["a"])`` and
    ``CNOT(wires=[0, 'a'])`` after the tape is constructed, we can first find the edge in the graph:

    >>> graph = qml.transforms.qcut.tape_to_graph(tape)
    >>> op0, op1 = tape.operations[1], tape.operations[2]
    >>> cut_edges = [e for e in graph.edges if e[0] is op0 and e[1] is op1]
    >>> cut_edges
    [(RY(0.543, wires=['a']), CNOT(wires=[0, 'a']), 0)]

    Then feed it to this function for placement:

    >>> cut_graph = qml.transforms.qcut.place_wire_cuts(graph=graph, cut_edges=cut_edges)
    >>> cut_graph
    <networkx.classes.multidigraph.MultiDiGraph at 0x7f7251ac1220>

    And visualize the cut by converting back to a tape:

    >>> print(qml.transforms.qcut.graph_to_tape(cut_graph).draw())
     0: ──RX(0.432)──────╭●──┤ ⟨Z⟩
     a: ──RY(0.543)──//──╰X──┤
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
        cut_graph.add_node(wire_cut, order=order)
        cut_graph.add_edge(op0, wire_cut, wire=wire)
        cut_graph.add_edge(wire_cut, op1, wire=wire)

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
    for op in list(graph.nodes):
        if isinstance(op, WireCut):
            uncut_graph.remove_node(op)
        elif isinstance(op, MeasureNode):
            for op1 in graph.neighbors(op):
                if isinstance(op1, PrepareNode):
                    uncut_graph.remove_node(op)
                    uncut_graph.remove_node(op1)

    if len([n for n in uncut_graph.nodes if isinstance(n, (MeasureNode, PrepareNode))]) > 0:
        warnings.warn(
            "The circuit contains `MeasureNode` or `PrepareNode` operations that are "
            "not paired up correctly. Please check.",
            UserWarning,
        )
    return uncut_graph


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

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.1, wires=0)
            qml.RY(0.2, wires=1)
            qml.RX(0.3, wires="a")
            qml.RY(0.4, wires="b")
            qml.CNOT(wires=[0, 1])
            qml.WireCut(wires=1)
            qml.CNOT(wires=["a", "b"])
            qml.CNOT(wires=[1, "a"])
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=["a", "b"])
            qml.RX(0.5, wires="a")
            qml.RY(0.6, wires="b")
            qml.expval(qml.PauliX(wires=[0]) @ qml.PauliY(wires=["a"]) @ qml.PauliZ(wires=["b"]))

    >>> print(qml.drawer.tape.text(tape))
     0: ──RX(0.1)──╭●──────────╭●───────────╭┤ ⟨X ⊗ Y ⊗ Z⟩
     1: ──RY(0.2)──╰X──//──╭●──╰X───────────│┤
     a: ──RX(0.3)──╭●──────╰X──╭●──RX(0.5)──├┤ ⟨X ⊗ Y ⊗ Z⟩
     b: ──RY(0.4)──╰X──────────╰X──RY(0.6)──╰┤ ⟨X ⊗ Y ⊗ Z⟩

    Since the existing :class:`~.WireCut` doesn't sufficiently fragment the circuit, we can find the
    remaining cuts using the default KaHyPar partitioner:

    >>> graph = qml.transforms.qcut.tape_to_graph(tape)
    >>> cut_graph = qml.transforms.qcut.find_and_place_cuts(
            graph=graph,
            num_fragments=2,
            imbalance=0.5,
        )

    Visualizing the newly-placed cut:

    >>> print(qml.transforms.qcut.graph_to_tape(cut_graph).draw())
     0: ──RX(0.1)──╭●───────────────╭●────────╭┤ ⟨X ⊗ Y ⊗ Z⟩
     1: ──RY(0.2)──╰X──//──╭●───//──╰X────────│┤
     a: ──RX(0.3)──╭●──────╰X──╭●────RX(0.5)──├┤ ⟨X ⊗ Y ⊗ Z⟩
     b: ──RY(0.4)──╰X──────────╰X────RY(0.6)──╰┤ ⟨X ⊗ Y ⊗ Z⟩

    We can then proceed with the usual process of replacing :class:`~.WireCut` nodes with
    pairs of :class:`~.MeasureNode` and :class:`~.PrepareNode`, and then break the graph
    into fragments. Or, alternatively, we can directly get such processed graph by passing
    ``replace_wire_cuts=True``:

    >>> cut_graph = qml.transforms.qcut.find_and_place_cuts(
            graph=graph,
            num_fragments=2,
            imbalance=0.5,
            replace_wire_cuts=True,
        )
    >>> frags, comm_graph = qml.transforms.qcut.fragment_graph(cut_graph)
    >>> for t in frags:
    ...     print(qml.transforms.qcut.graph_to_tape(t).draw())

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
    directly passing a :class:`~.Device` to the ``device`` argument):

    >>> cut_strategy = qml.transforms.qcut.CutStrategy(max_free_wires=2)
    >>> print(cut_strategy.get_cut_kwargs(graph))
     [{'num_fragments': 2, 'imbalance': 0.5714285714285714},
      {'num_fragments': 3, 'imbalance': 1.4},
      {'num_fragments': 4, 'imbalance': 1.75},
      {'num_fragments': 5, 'imbalance': 2.3333333333333335},
      {'num_fragments': 6, 'imbalance': 2.0},
      {'num_fragments': 7, 'imbalance': 3.0},
      {'num_fragments': 8, 'imbalance': 2.5},
      {'num_fragments': 9, 'imbalance': 2.0},
      {'num_fragments': 10, 'imbalance': 1.5},
      {'num_fragments': 11, 'imbalance': 1.0},
      {'num_fragments': 12, 'imbalance': 0.5},
      {'num_fragments': 13, 'imbalance': 0.05},
      {'num_fragments': 14, 'imbalance': 0.1}]

    The printed list above shows all the possible cutting configurations one can attempt to perform
    in order to search for the optimal cut. This is done by directly passing a
    :class:`~.CutStrategy` to :func:`~.find_and_place_cuts`:

    >>> cut_graph = qml.transforms.qcut.find_and_place_cuts(
            graph=graph,
            cut_strategy=cut_strategy,
        )
    >>> print(qml.transforms.qcut.graph_to_tape(cut_graph).draw())
     0: ──RX──//─╭●──//────────╭●──//─────────┤ ╭<X@Y@Z>
     1: ──RY──//─╰X──//─╭●──//─╰X─────────────┤ │
     a: ──RX──//─╭●──//─╰X──//─╭●──//──RX──//─┤ ├<X@Y@Z>
     b: ──RY──//─╰X──//────────╰X──//──RY─────┤ ╰<X@Y@Z>

    As one can tell, quite a few cuts have to be made in order to execute the circuit on solely
    2-qubit devices. To verify, let's print the fragments:

    >>> qml.transforms.qcut.replace_wire_cut_nodes(cut_graph)
    >>> frags, comm_graph = qml.transforms.qcut.fragment_graph(cut_graph)
    >>> for t in frags:
    ...     print(qml.transforms.qcut.graph_to_tape(t).draw())

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
            num_cuts = sum(isinstance(n, WireCut) for n in cut_graph.nodes)

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
        len(graph_to_tape(f).wires) <= max_free_wires for j, f in enumerate(fragments)
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
