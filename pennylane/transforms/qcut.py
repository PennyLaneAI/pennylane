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
from itertools import product
from typing import List, Sequence, Tuple

import pennylane as qml
from networkx import MultiDiGraph, weakly_connected_components
from pennylane import apply, expval
from pennylane.grouping import string_to_pauli_word
from pennylane.measure import MeasurementProcess
from pennylane.operation import Expectation, Operation, Operator, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape
from pennylane.wires import Wires


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
            cut_edges.append((node1, node2))
            graph_copy.remove_edge(node1, node2, key=wire)

    subgraph_nodes = weakly_connected_components(graph_copy)
    subgraphs = tuple(graph_copy.subgraph(n) for n in subgraph_nodes)

    communication_graph = MultiDiGraph()
    communication_graph.add_nodes_from(range(len(subgraphs)))

    for node1, node2 in cut_edges:
        for i, subgraph in enumerate(subgraphs):
            if subgraph.has_node(node1):
                start_fragment = i
            if subgraph.has_node(node2):
                end_fragment = i

        communication_graph.add_edge(start_fragment, end_fragment, pair=(node1, node2))

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

    copy_ops = [copy.copy(op) for _, op in ordered_ops]

    with QuantumTape() as tape:
        for op in copy_ops:
            new_wires = [wire_map[w] for w in op.wires]
            op._wires = Wires(new_wires)  # TODO: find a better way to update operation wires
            apply(op)

            if isinstance(op, MeasureNode):
                assert len(op.wires) == 1
                measured_wire = op.wires[0]

                new_wire = _find_new_wire(wires)
                wires += new_wire

                original_wire = reverse_wire_map[measured_wire]
                wire_map[original_wire] = new_wire
                reverse_wire_map[new_wire] = original_wire

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

    return [expval(obs @ g) for g in group]


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
        ...     print(t.draw())
         0: ──I──RX(0.5)──┤ ⟨I⟩ ┤ ⟨Z⟩

         0: ──I──RX(0.5)──┤ ⟨X⟩

         0: ──I──RX(0.5)──┤ ⟨Y⟩

         0: ──X──RX(0.5)──┤ ⟨I⟩ ┤ ⟨Z⟩

         0: ──X──RX(0.5)──┤ ⟨X⟩

         0: ──X──RX(0.5)──┤ ⟨Y⟩

         0: ──H──RX(0.5)──┤ ⟨I⟩ ┤ ⟨Z⟩

         0: ──H──RX(0.5)──┤ ⟨X⟩

         0: ──H──RX(0.5)──┤ ⟨Y⟩

         0: ──H──S──RX(0.5)──┤ ⟨I⟩ ┤ ⟨Z⟩

         0: ──H──S──RX(0.5)──┤ ⟨X⟩

         0: ──H──S──RX(0.5)──┤ ⟨Y⟩
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

    if len(results) != ctr:
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
