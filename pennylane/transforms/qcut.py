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

import copy
import string
import uuid
import warnings
from dataclasses import InitVar, dataclass
from functools import partial
from itertools import product
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

from networkx import MultiDiGraph, weakly_connected_components, has_path

import pennylane as qml
from pennylane import apply, expval
from pennylane.grouping import string_to_pauli_word
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Expectation, Operation, Operator, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape
from pennylane.wires import Wires

from .batch_transform import batch_transform


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None

    def __init__(self, *params, wires=None, do_queue=True, id=None):
        id = str(uuid.uuid4())

        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None

    def __init__(self, *params, wires=None, do_queue=True, id=None):
        id = str(uuid.uuid4())

        super().__init__(*params, wires=wires, do_queue=do_queue, id=id)


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


# pylint: disable=too-many-branches
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
    measure_nodes = [n for n in graph.nodes if isinstance(n, MeasurementProcess)]

    for node1, node2, wire in graph.edges:
        if isinstance(node1, MeasureNode):
            assert isinstance(node2, PrepareNode)
            cut_edges.append((node1, node2))
            graph_copy.remove_edge(node1, node2, key=wire)

    subgraph_nodes = weakly_connected_components(graph_copy)
    subgraphs = tuple(MultiDiGraph(graph_copy.subgraph(n)) for n in subgraph_nodes)

    communication_graph = MultiDiGraph()
    communication_graph.add_nodes_from(range(len(subgraphs)))

    for node1, node2 in cut_edges:
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

    measure_nodes_to_remove = [m for p in prepare_nodes_removed for m, p_ in cut_edges if p is p_]
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

    Each node in the graph should have an order attribute specifying the topological order of
    the operations.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :doc:`transforms </code/qml_transforms>` page for more details.

    Args:
        graph (MultiDiGraph): directed multigraph to be converted to a tape

    Returns:
        QuantumTape: the quantum tape corresponding to the input graph

    **Example**

    Consider the following, where ``graph`` contains :class:`~.MeasureNode` and
    :class:`~.PrepareNode` pairs that divide the full circuit graph into five subgraphs.
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

        for meas in copy_meas:
            obs = meas.obs
            obs._wires = Wires([wire_map[w] for w in obs.wires])
            observables.append(obs)

        # We assume that each MeasurementProcess node in the graph contributes to a single
        # expectation value of an observable, given by the tensor product over the observables of
        # each MeasurementProcess.
        if len(observables) > 1:
            qml.expval(Tensor(*observables))
        elif len(observables) == 1:
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


def _prep_iplus_state(wire):
    qml.Hadamard(wire)
    qml.S(wires=wire)


PREPARE_SETTINGS = [_prep_zero_state, _prep_one_state, _prep_plus_state, _prep_iplus_state]


def expand_fragment_tapes(
    tape: QuantumTape,
) -> Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]:
    """
    Expands a fragment tape into a collection of tapes for each configuration of the
    :class:`MeasureNode` and :class:`PrepareNode` operations.

    .. note::

        This function is designed for use as part of the circuit cutting workflow. Check out the
        :doc:`transforms </code/qml_transforms>` page for more details.

    Args:
        tape (QuantumTape): the fragment tape to be expanded.

    Returns:
        Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]: the
        tapes corresponding to each configuration, the preparation nodes and
        the measurement nodes.

    **Example**

    Consider the following circuit, which contains a :class:`~.MeasureNode` and :class:`~.PrepareNode`
    operation:

    .. code-block:: python

        from pennylane.transforms import qcut

        with qml.tape.QuantumTape() as tape:
            qcut.PrepareNode(wires=0)
            qml.RX(0.5, wires=0)
            qcut.MeasureNode(wires=0)

    We can expand over the measurement and preparation nodes using:

    .. code-block:: python

        >>> tapes, prep, meas = qml.transforms.expand_fragment_tapes(tape)
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

                with qml.tape.stop_recording():
                    measurements = _get_measurements(group, tape.measurements)
                for meas in measurements:
                    apply(meas)

                tapes.append(tape_)

    return tapes, prepare_nodes, measure_nodes


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

        This function is designed for use as part of the circuit cutting workflow. Check out the
        :doc:`transforms </code/qml_transforms>` page for more details.

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
        communication_graph (MultiDiGraph): the communication graph determining connectivity between
            the tensors
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

    >>> qml.transforms.contract_tensors(tensors, graph, prep, meas)
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
    """Processing function for the :func:`cut_circuit` transform.

    .. note::

        This function is designed for use as part of the circuit cutting workflow. Check out the
        :doc:`transforms </code/qml_transforms>` page for more details.

    Args:
        results (Sequence[Sequence]): A collection of execution results corresponding to the
            expansion of circuit fragments in the ``communication_graph`` over measurement and
            preparation node configurations. These results are processed into tensors and then
            contracted.
        communication_graph (MultiDiGraph): the communication graph determining connectivity between
            circuit fragments
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
    tape: QuantumTape, use_opt_einsum: bool = False, device_wires: Optional[Wires] = None
) -> Tuple[Tuple[QuantumTape], Callable]:
    """
    Batch transform for circuit cutting.

    .. note::

        This function is designed for use as part of the circuit cutting workflow. Check out the
        :doc:`transforms </code/qml_transforms>` page for more details.

    Args:
        tape (QuantumTape): The tape of the full circuit to be cut.
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.
        device_wires (.wires.Wires): Wires of the device that the cut circuits are to be run on

    Returns:
        Tuple[Tuple[QuantumTape], Callable]: the tapes corresponding to the circuit fragments as a
        result of cutting and a post-processing function which combines the results via tensor
        contractions.

    **Example**

    Consider the following circuit containing a :class:`~.WireCut` operation:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.RY(0.543, wires=1)
            qml.WireCut(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(0.240, wires=0)
            qml.RZ(0.133, wires=1)
            return qml.expval(qml.PauliZ(wires=[0]))

    >>> x = 0.531
    >>> print(circuit(x))
    0.8623011058543121
    >>> print(qml.grad(circuit)(x))
    -0.506395895364911

    This can be cut using the following transform

    >>> x = 0.531
    >>> cut_circuit = qcut.cut_circuit(circuit)
    >>> cut_circuit(x)
    0.8623011058543121

    Futhermore, the output of the cut circuit is also differentiable:

    .. code-block:: python

        >>> qml.grad(cut_circuit)(x)
        -0.506395895364911
    """
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

    num_cut = len([op for op in tape.operations if isinstance(op, WireCut)])
    if num_cut == 0:
        raise ValueError("Cannot apply the circuit cutting workflow to a circuit without any cuts")

    g = tape_to_graph(tape)
    replace_wire_cut_nodes(g)
    fragments, communication_graph = fragment_graph(g)
    fragment_tapes = [graph_to_tape(f) for f in fragments]
    fragment_tapes = [remap_tape_wires(t, device_wires) for t in fragment_tapes]
    expanded = [expand_fragment_tapes(t) for t in fragment_tapes]

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

    tkwargs.setdefault("device_wires", qnode.device.wires)
    return self.default_qnode_wrapper(qnode, targs, tkwargs)


def remap_tape_wires(tape: QuantumTape, wires: Sequence) -> QuantumTape:
    """Map the wires of a tape to a new set of wires.

    Given an :math:`n`-wire ``tape``, this function returns a new :class:`~.QuantumTape` with
    operations and measurements acting on the first :math:`n` wires provided in the ``wires``
    argument. The input ``tape`` is left unmodified.

    .. note::

        This function is designed for use as part of the circuit cutting workflow. Check out the
        :doc:`transforms </code/qml_transforms>` page for more details.

    Args:
        tape (QuantumTape): the quantum tape whose wires should be remapped
        wires (Sequence): the new set of wires to map to

    Returns:
        QuantumTape: A remapped copy of the input tape

    Raises:
        ValueError: if the number of wires in ``tape`` exceeds ``len(wires)``

    **Example**

    .. code-block:: python

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.5, wires=2)
            qml.RY(0.6, wires=3)
            qml.CNOT(wires=[2, 3])
            qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))

        new_wires = [0, 1]
        new_tape = qml.transforms.remap_tape_wires(tape, new_wires)

    >>> print(new_tape.draw())
     0: ──RX(0.5)──╭C──╭┤ ⟨Z ⊗ Z⟩
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

    Args:
        devices (Union[qml.Device, Sequence[qml.Device]]): Single, or Sequence of, device(s).
            Optional only when ``max_free_wires`` is provided.
        max_free_wires (int): Number of wires for the largest available device. Optional only when
            ``devices`` is provided where it defaults to the maximum number of wires among
            ``devices``.
        min_free_wires (int): Number of wires for the smallest available device, or, equivalently,
            the smallest max fragment-wire-size that the partitioning is allowed to explore.
            When provided, this parameter will be used to derive an upper-bound to the range of
            explored number of fragments.  Optional, defaults to ``max_free_wires``.
        num_fragments_probed (Union[int, Sequence[int]]): Single, or 2-Sequence of, number(s)
            specifying the potential (range of) number of fragments for the partitioner to attempt.
            Optional, defaults to probing all valid strategies derivable from the circuit and
            devices.
        max_free_gates (int): Maximum allowed circuit depth for the deepest available device.
            Optional, defaults to unlimited depth.
        min_free_gates (int): Maximum allowed circuit depth for the shallowest available device.
            Optional, defaults to ``max_free_gates``.
        imbalance_tolerance (float): The global maximum allowed imbalance for all partition trials.
            Optional, defaults to unlimited imbalance. Used only if there's a known hard balancing
            constraint on the partitioning problem.

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

    #: Class attribute, threshold for warning about too many fragments.
    HIGH_NUM_FRAGMENTS: ClassVar[int] = 20
    #: Class attribute, threshold for warning about too many partition attempts.
    HIGH_PARTITION_ATTEMPTS: ClassVar[int] = 20

    def __post_init__(
        self,
        devices,
    ):
        """Deriving cutting constraints from given devices and parameters."""

        self.max_free_wires = self.max_free_wires or self.min_free_wires
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
            if not isinstance(devices, Sequence) or any(
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

    def get_cut_kwargs(
        self,
        tape_dag: MultiDiGraph,
        max_wires_by_fragment: Sequence[int] = None,
        max_gates_by_fragment: Sequence[int] = None,
    ) -> List[Dict[str, Any]]:
        """Derive the complete set of arguments, based on a given circuit, for passing to a graph
        partitioner.

        Args:
            tape_dag (MultiDiGraph): Graph representing a tape, typically the output of
                :func:`tape_to_graph`.
            max_wires_by_fragment (Sequence[int]): User-predetermined list of wire limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.
            max_gates_by_fragment (Sequence[int]): User-predetermined list of gate limits by
                fragment. If supplied, the number of fragments will be derived from it and
                exploration of other choices will not be made.

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
        tape_wires = set(w for _, _, w in tape_dag.edges.data("wire"))
        num_tape_wires = len(tape_wires)
        num_tape_gates = tape_dag.order()
        self._validate_input(max_wires_by_fragment, max_gates_by_fragment)

        probed_cuts = self._infer_probed_cuts(
            num_tape_wires=num_tape_wires,
            num_tape_gates=num_tape_gates,
            max_wires_by_fragment=max_wires_by_fragment,
            max_gates_by_fragment=max_gates_by_fragment,
        )

        return probed_cuts

    @staticmethod
    def _infer_imbalance(
        k, num_wires, num_gates, free_wires, free_gates, imbalance_tolerance=None
    ) -> float:
        """Helper function for determining best imbalance limit."""
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

        wire_imbalance = free_wires / avg_fragment_wires - 1
        gate_imbalance = free_gates / avg_fragment_gates - 1
        imbalance = min(gate_imbalance, wire_imbalance)
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
        num_tape_wires,
        num_tape_gates,
        max_wires_by_fragment=None,
        max_gates_by_fragment=None,
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

        Returns:
            List[Dict[str, Any]]: A list of minimal set of kwargs being passed to a graph
                partitioner method.
        """

        # Assumes unlimited width/depth if not supplied.
        max_free_wires = self.max_free_wires or num_tape_wires
        max_free_gates = self.max_free_gates or num_tape_gates

        # Assumes same number of wires/gates across all devices if min_free_* not provided.
        min_free_wires = self.min_free_wires or max_free_wires
        min_free_gates = self.min_free_gates or max_free_gates

        # The lower bound of k corresponds to executing each fragment on the largest available device.
        k_lb = 1 + max(
            (num_tape_wires - 1) // max_free_wires,  # wire limited
            (num_tape_gates - 1) // max_free_gates,  # gate limited
        )
        # The upper bound of k corresponds to executing each fragment on the smallest available device.
        k_ub = 1 + max(
            (num_tape_wires - 1) // min_free_wires,  # wire limited
            (num_tape_gates - 1) // min_free_gates,  # gate limited
        )

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
                num_tape_wires,
                num_tape_gates,
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
