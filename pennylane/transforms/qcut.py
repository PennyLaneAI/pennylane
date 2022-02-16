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
import itertools
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import pennylane as qml
from networkx import MultiDiGraph, weakly_connected_components
from pennylane import Hadamard, Identity, PauliX, PauliY, PauliZ, S, apply, expval
from pennylane.grouping import string_to_pauli_word
from pennylane.measure import MeasurementProcess
from pennylane.operation import AnyWires, Expectation, Operation, Operator, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape, stop_recording
from pennylane.wires import Wires


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None


SUBS = "₀₁₂₃₄₅₆₇₈₉"
SUB = str.maketrans("0123456789", SUBS)


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
    the operations. This maintains ordering of operations and allows for the
    deferred measurement principle to be applied when necessary.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :doc:`transforms </code/qml_transforms>` page for more details.

    Args:
        graph (MultiDiGraph): directed multigraph to be converted to a tape

    Returns:
        QuantumTape: the quantum tape corresponding to the input graph

    **Example**

    Consider the following, where ``graph`` contains three :class:`~.MeasureNode` and
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

    copy_ordered_ops = copy.deepcopy(ordered_ops)

    updated = {}
    with QuantumTape() as tape:
        for _, op in copy_ordered_ops:
            name_and_wire = op.name + str(op.wires)

            new_wires = [wire_map[w] for w in op.wires]
            op._wires = Wires(new_wires)  # TODO: find a better way to update operation wires
            apply(op)

            if isinstance(op, MeasureNode):
                assert len(op.wires) == 1
                # if a wire has already been updated we still
                # want to map *from* the original wire
                if name_and_wire in updated:
                    new_wire = updated[name_and_wire] + 1
                    wire_map[measured_wire] = new_wire
                else:
                    measured_wire = op.wires[0]
                    new_wire = _find_new_wire(wires)
                    wires += new_wire
                    wire_map[measured_wire] = new_wire
                updated[name_and_wire] = new_wire

    return tape


def _prep_zero_state(wire):
    Identity(wire)


def _prep_one_state(wire):
    PauliX(wire)


def _prep_plus_state(wire):
    Hadamard(wire)


def _prep_iplus_state(wire):
    Hadamard(wire)
    S(wires=wire)


PREPARE_SETTINGS = [_prep_zero_state, _prep_one_state, _prep_plus_state, _prep_iplus_state]


def expand_fragment_tapes(
    tape: QuantumTape,
) -> Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]:
    """
    Expands a fragment tape into a tape for each configuration.

    Args:
        tape (QuantumTape): the fragment tape to be expanded.

    Returns:
        Tuple[List[QuantumTape], List[PrepareNode], List[MeasureNode]]: the
        tapes corresponding to each configration, the preparation nodes and
        the measurement nodes.

    **Example**

    Consider the following where ``graph`` contains a single
    :class:`~.MeasureNode` and :class:`~.PrepareNode` within the full circuit
    graph:

    .. code-block:: python

        from pennylane.transforms import qcut

        >>> subgraphs, communication_graph = qcut.fragment_graph(graph)
        >>> tapes = [qcut.graph_to_tape(sg) for sg in subgraphs]

        >>> fragment_configurations = [qcut.expand_fragment_tapes(tape) for tape in tapes]
        >>> fragment_configurations
        [([<QuantumTape: wires=[0, 1], params=4>,
           <QuantumTape: wires=[0, 1], params=4>,
           <QuantumTape: wires=[0, 1], params=4>,
           <QuantumTape: wires=[0, 1], params=4>],
          [],
          [MeasureNode(wires=[1])]),
         ([<QuantumTape: wires=[1, 2], params=2>,
           <QuantumTape: wires=[1, 2], params=2>,
           <QuantumTape: wires=[1, 2], params=2>,
           <QuantumTape: wires=[1, 2], params=2>],
          [PrepareNode(wires=[1])],
          [])]

    """
    obs_map = {"Identity": Identity, "PauliX": PauliX, "PauliY": PauliY, "PauliZ": PauliZ}
    # meas_map = {"I": Identity, "X": PauliX, "Y": PauliY, "Z": PauliZ}

    prepare_nodes = [o for o in tape.operations if isinstance(o, PrepareNode)]
    measure_nodes = [o for o in tape.operations if isinstance(o, MeasureNode)]
    wire_map = {mn.wires.tolist()[0]: i for i, mn in enumerate(measure_nodes)}
    # wire_map = [{mn.wires.tolist()[0]: 0} for mn in measure_nodes]
        
    # prepare_combinations = product(range(len(PREPARE_SETTINGS)), repeat=len(prepare_nodes))
    # pauli_group_strs = partition_pauli_group(len(measure_nodes))
    # measure_combinations = [string_to_pauli_word(obs) for comb in pauli_group_strs for obs in comb]
    n_qubits = len(measure_nodes)
    if n_qubits >= 1:
        measure_combinations = partition_pauli_group(len(measure_nodes))
    else:
        measure_combinations = [((),)] # this helps `product()` give the desired effect in following loop
        # measure_combinations = product(range(len(prepare_nodes)), repeat=len(measure_nodes))
    # MEASURE_SETTINGS = [meas_map[x] for b in pauli_group_strs for x in b]
    # measure_combinations = product(range(len(MEASURE_SETTINGS)), repeat=len(measure_nodes))
    tapes = []

    for measure_group in measure_combinations:
        prepare_combinations = product(range(len(PREPARE_SETTINGS)), repeat=len(prepare_nodes))
        if any(isinstance(e, str) for e in measure_group):
            group = [string_to_pauli_word(paulis, wire_map=wire_map) for paulis in measure_group]
        else:
            group = []
        for prepare_settings, measure_settings in product(prepare_combinations, [measure_group]):
            prepare_mapping = {n: PREPARE_SETTINGS[s] for n, s in zip(prepare_nodes, prepare_settings)}

            # if any(isinstance(e, str) for e in measure_settings):
            #     comb_ops = [string_to_pauli_word(obs, wire_map=wire_map[i]) for i, obs in enumerate(measure_settings)]
            # else: 
            #     comb_ops = []
            
            # measure_mapping = {n: s for n, s in zip(measure_nodes, comb_ops)}
            meas = []

            with QuantumTape() as tape_:
                for op in tape.operations:
                    if isinstance(op, PrepareNode):
                        w = op.wires[0]
                        prepare_mapping[op](w)
                    elif isinstance(op, MeasureNode):
                        meas.append(op)
                    else:
                        apply(op)

                for meas_op in group:
                    with stop_recording():
                        tens_ops = [meas_op]
                        # for op in meas:
                        #     tens_ops.append(meas_op)
                        op_tensor = Tensor(*tens_ops)
                    
                    if len(tape.measurements) > 0:
                        for m in tape.measurements:
                            if m.return_type is not Expectation:
                                raise ValueError("Only expectation values supported for now")
                            with stop_recording():
                                m_obs = obs_map[m.obs.name](wires=m.obs.wires)
                                if isinstance(m_obs, Tensor):
                                    terms = m_obs.obs
                                    for t in terms:
                                        if not isinstance(t, (Identity, PauliX, PauliY, PauliY)):
                                            raise ValueError("Only tensor products of Paulis for now")
                                    op_tensor_wires = [(t.wires.tolist()[0], t) for t in op_tensor.obs]
                                    m_obs_wires = [(t.wires.tolist()[0], t) for t in terms]
                                    all_wires = sorted(op_tensor_wires + m_obs_wires)
                                    all_terms = [t[1] for t in all_wires]
                                    full_tensor = Tensor(*all_terms)
                                else:
                                    if not isinstance(m_obs, (Identity, PauliX, PauliY, PauliZ)):
                                        raise ValueError("Only tensor products of Paulis for now")

                                    op_tensor_wires = [(t.wires.tolist()[0], t) for t in op_tensor.obs]
                                    m_obs_wires = [(m_obs.wires.tolist()[0], m_obs)]
                                    all_wires = sorted(op_tensor_wires + m_obs_wires)
                                    all_terms = [t[1] for t in all_wires]
                                    full_tensor = Tensor(*all_terms)
                            expval(full_tensor)
                    elif len(op_tensor.name) > 0:
                        expval(op_tensor)
                    else:
                        expval(Identity(tape.wires[0]))

                tapes.append(tape_)

    return tapes, prepare_nodes, measure_nodes


def partition_pauli_group(n_qubits: int) -> List[List[str]]:
    """Partitions the :math:`n`-qubit Pauli group into qubit-wise commuting terms.
    The :math:`n`-qubit Pauli group is composed of :math:`4^{n}` terms that can be partitioned into
    :math:`3^{n}` qubit-wise commuting groups.
    Args:
        n_qubits (int): number of qubits
    Returns:
        List[List[str]]: A collection of qubit-wise commuting groups containing Pauli words as
        simple strings
    **Example**
    >>> qml.grouping.partition_pauli_group(3)
    [['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ'],
     ['IIX', 'IZX', 'ZIX', 'ZZX'],
     ['IIY', 'IZY', 'ZIY', 'ZZY'],
     ['IXI', 'IXZ', 'ZXI', 'ZXZ'],
     ['IXX', 'ZXX'],
     ['IXY', 'ZXY'],
     ['IYI', 'IYZ', 'ZYI', 'ZYZ'],
     ['IYX', 'ZYX'],
     ['IYY', 'ZYY'],
     ['XII', 'XIZ', 'XZI', 'XZZ'],
     ['XIX', 'XZX'],
     ['XIY', 'XZY'],
     ['XXI', 'XXZ'],
     ['XXX'],
     ['XXY'],
     ['XYI', 'XYZ'],
     ['XYX'],
     ['XYY'],
     ['YII', 'YIZ', 'YZI', 'YZZ'],
     ['YIX', 'YZX'],
     ['YIY', 'YZY'],
     ['YXI', 'YXZ'],
     ['YXX'],
     ['YXY'],
     ['YYI', 'YYZ'],
     ['YYX'],
     ['YYY']]
    """
    # Cover the case where n_qubits may be passed as a float
    if isinstance(n_qubits, float):
        if n_qubits.is_integer():
            n_qubits = int(n_qubits)

    # If not an int, or a float representing a int, raise an error
    if not isinstance(n_qubits, int):
        raise TypeError("Must specify an integer number of qubits.")

    if n_qubits <= 0:
        raise ValueError("Number of qubits must be at least 1.")

    strings = set()  # tracks all the strings that have already been grouped
    groups = []

    # We know that I and Z always commute on a given qubit. The following generates all product
    # sequences of len(n_qubits) over "FXYZ", with F indicating a free slot that can be swapped for
    # the product over I and Z, and all other terms fixed to the given X/Y/Z. For example, if
    # ``n_qubits = 3`` our first value for ``string`` will be ``('F', 'F', 'F')``. We then expand
    # the product of I and Z over the three free slots, giving
    # ``['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ']``, which is our first group. The
    # next element of ``string`` will be ``('F', 'F', 'X')`` which we use to generate our second
    # group ``['IIX', 'IZX', 'ZIX', 'ZZX']``.

    for string in itertools.product("FXYZ", repeat=n_qubits):
        if string not in strings:
            num_free_slots = string.count("F")

            group = []
            commuting = itertools.product("IZ", repeat=num_free_slots)

            for commuting_string in commuting:
                commuting_string = list(commuting_string)
                new_string = tuple(commuting_string.pop(0) if s == "F" else s for s in string)

                if new_string not in strings:  # only add if string has not already been grouped
                    group.append("".join(new_string))
                    strings |= {new_string}

            if len(group) > 0:
                groups.append(group)

    return groups

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
            for tensor contractions of large networks but must be installed separately using, e.g.,
            ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.

    Returns:
        float or array-like: the result of contracting the tensor network

    **Example**

    We first set up the tensors and their corresponding :class:`~.PrepareNode` and
    :class:`~.MeasureNode` orderings:
    
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

                    if p is prep_op:
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

                    if m is meas_op:
                        symb = meas_map[meas_op]
                        tensor_indxs[i] += symb

    eqn = ",".join(tensor_indxs)
    kwargs = {} if use_opt_einsum else {"like": tensors[0]}

    return contract(eqn, *tensors, **kwargs)
