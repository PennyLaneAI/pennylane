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
import string
import warnings
from typing import Sequence, Tuple, List, Dict, Any, Union, ClassVar
from dataclasses import dataclass, InitVar

from networkx import MultiDiGraph, weakly_connected_components
import pennylane as qml
from pennylane.measure import MeasurementProcess
from pennylane.operation import Operation, Operator, Tensor
from pennylane.ops.qubit.non_parametric_ops import WireCut
from pennylane.tape import QuantumTape


class MeasureNode(Operation):
    """Placeholder node for measurement operations"""

    num_wires = 1
    grad_method = None


class PrepareNode(Operation):
    """Placeholder node for state preparations"""

    num_wires = 1
    grad_method = None


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
        min_free_wires (int): Number of wires for the smallest available device.
            Optional, defaults to ``max_device_wires``.
        max_fragments_probed (Union[int, Sequence[int]]): Single, or 2-Sequence of, number(s)
            specifying the potential (range of) number of fragments for the partitioner to attampt.
            Optional, defaults to probing all valid strategies derivable from the circuit and
            devices.
        max_free_gates (int): Maximum allowed circuit depth for the deepest available device.
            Optional, defaults to unlimited depth.
        min_free_gates (int): Maximum allowed circuit depth for the shallowest available device.
            Optional, defaults to ``max_device_gates``.

    **Example**

    .. code-block:: python

        import pennylane as qml
        from pennylane.transforms import qcut

        dev_a = qml.device('default.qubit', wires=4)
        dev_b = qml.device('default.qubit', wires=6)

        with qml.tape.QuantumTape() as tape:
            qml.RX(0.4, wires=0)
            qml.RY(0.9, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.expval(qml.PauliZ(1))

        tape_dag = qcut.tape_to_graph(tape)

    Cut circuit with single-device-based strategy using the default 'kahypar' cutter:
    >>> cut_strategy = qcut.CutStrategy(devices=dev_a)
    >>> qcut.cut_circuit(tape_dag, method='kahypar', strategy=cut_strategy)

    Cut circuit with multi-device-based strategy using the default 'kahypar' cutter:
    >>> cut_strategy = qcut.CutStrategy(devices=(dev_a, dev_b))
    >>> qcut.cut_circuit(tape_dag, method='kahypar', strategy=cut_strategy)

    Cut circuit with user-supplied strategy using user-supplied partitioning callable:
    >>> cut_strategy = qcut.CutStrategy(
            max_free_wires=6,
            min_free_wires=4,
            num_fragments_probed=(2, 5),
        )
    >>> qcut.cut_circuit(tape_dag, method=my_partitioner, strategy=cut_strategy, **my_kwargs)

    """

    # pylint: disable=too-many-arguments, too-many-instance-attributes

    devices: InitVar[Sequence[qml.Device]] = None
    max_free_wires: int = None
    min_free_wires: int = None
    num_fragments_probed: Union[int, Sequence[int]] = None
    max_free_gates: int = None
    min_free_gates: int = None
    imbalance_tolerance: float = None

    HIGH_NUM_FRAGMENTS: ClassVar[int] = 20
    HIGH_PARTITION_ATTEMPS: ClassVar[int] = 20

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
            if not isinstance(devices, (list, tuple)) or any(
                (not isinstance(d, qml.Device) for d in devices)
            ):
                raise ValueError(
                    f"Argument `devices` must be a list of `Device` instances, got {type(devices)}."
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

        **Example**

        .. code-block:: python

            import pennylane as qml
            from pennylane.transforms import qcut

            dev = qml.device('default.qubit', wires=4)

            with qml.tape.QuantumTape() as tape:
                qml.RX(0.4, wires=0)
                qml.RY(0.9, wires=0)
                qml.CNOT(wires=[0, 1])
                qml.expval(qml.PauliZ(1))

            tape_dag = qcut.tape_to_graph(tape)

        Deriving kwargs for a given circuit and feed it to a custom partitioner along with an extra
        custom parameter:
        >>> cut_strategy = qcut.CutStrategy(devices=dev)
        >>> cut_kwargs = cut_strategy.get_cut_kwargs(tape_dag)
        >>> cut_kwargs.update({'extra_param': 0})
        >>> my_partitioner(tape_dag, **cut_kwargs)

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
            List[Dict[str, Any]]: A list of minimal set of kwargs being passed to a graph
                partitioner method.
        """
        tape_wires = set(w for _, _, w in tape_dag.edges.data("wire"))
        assert all((w is not None for w in tape_wires))
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
        assert free_wires >= avg_fragment_wires
        assert free_gates >= avg_fragment_gates

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
            assert isinstance(max_wires_by_fragment, (list, tuple))
            assert all(isinstance(i, int) and i > 0 for i in max_wires_by_fragment)
        if max_gates_by_fragment is not None:
            assert isinstance(max_gates_by_fragment, (list, tuple))
            assert all(isinstance(i, int) and i > 0 for i in max_gates_by_fragment)
        if max_wires_by_fragment is not None and max_gates_by_fragment is not None:
            assert len(max_wires_by_fragment) == len(max_gates_by_fragment)

    def _infer_probed_cuts(
        self,
        num_tape_wires,
        num_tape_gates,
        max_wires_by_fragment=None,
        max_gates_by_fragment=None,
    ) -> List[Dict[str, Any]]:
        """
        Helper function for deriving the minimal set of best default partitioning constraints
        for the a graph partitioner.

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
                    f"The attempted number of fragments seems high with lower bound at {k_lower}, "
                    "are you sure?"
                )

            # Prepare the list of ks to explore:
            ks = list(range(k_lower, k_upper + 1))

            if len(ks) > self.HIGH_PARTITION_ATTEMPS:
                warnings.warn(
                    f"The numer of partition attempts seems high ({len(ks)}), are you sure?"
                )
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
