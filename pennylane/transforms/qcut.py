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
from multiprocessing.sharedctypes import Value
import string
import warnings
from typing import Sequence, Tuple, List, Dict, Any, Union
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
class CutConstraints:
    """
    Container wrapping partitioning parameters for passing to the automatic cutter.

    Args:
        devices (List[qml.Device]): A list of (or single) device(s), Optional.
        max_free_wires (int): Number of wires for the largest available device.
        min_free_wires (int): Number of wires for the smallest available device.
            Optional, defaults to `max_device_wires`.
        max_fragments_probed (List[int]): List of potential number of fragments to try for the
            partitioner. Optional, defaults to probing all valid choices that are derived from
            the circuit and devices.
        max_free_gates (int): Maximum allowed circuit depth for the deepest available device.
            Optional, defaults to unlimited depth.
        min_free_gates (int): Maximum allowed circuit depth for the shallowest available device.
            Optional, defaults to `max_device_gates`.
    """

    # pylint: disable=too-many-arguments

    devices: InitVar[List[qml.Device]] = None
    max_free_wires: int = None
    min_free_wires: int = None
    num_fragments_probed: Union[int, List[int], Tuple[int]] = None
    max_free_gates: int = None
    min_free_gates: int = None
    imbalance_tolerance: float = None

    def __post_init__(
        self,
        devices,
    ):

        self.max_free_wires = self.max_free_wires or self.min_free_gates
        if isinstance(self.num_fragments_probed, int):
            self.num_fragments_probed = [self.num_fragments_probed]
        if isinstance(self.num_fragments_probed, (list, tuple)):
            self.num_fragments_probed = sorted(self.num_fragments_probed)
            self.k_lower = self.num_fragments_probed[0]
            self.k_upper = self.num_fragments_probed[-1]
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
            else:
                device_wire_sizes = [len(d.wires) for d in devices]

                self.max_free_wires = self.max_free_wires or max(device_wire_sizes)
                self.min_free_wires = self.min_free_wires or min(device_wire_sizes)

    def get_cut_kwargs(
        self,
        tape_dag: MultiDiGraph,
        max_wires_by_fragment: List[int] = None,
        max_gates_by_fragment: List[int] = None,
    ):
        tape_wires = set(tape_dag.edges.data("wire"))
        assert all((w is not None for w in tape_wires))
        num_tape_wires = len(tape_wires)
        num_tape_gates = tape_dag.order()
        self._validate_dag(
            num_tape_wires, num_tape_gates, max_wires_by_fragment, max_gates_by_fragment
        )

        cut_kwargs = self._infer_default_configs(
            num_tape_wires=num_tape_wires,
            num_tape_gates=num_tape_gates,
            max_wires_by_fragment=max_wires_by_fragment,
            max_gates_by_fragment=max_gates_by_fragment,
        )

        return cut_kwargs

    @staticmethod
    def infer_imbalance(k, num_wires, num_gates, free_wires, free_gates, imbalance_tolerance=None):
        """Helper function for determining best imbalance limit."""
        avg_fragment_wires = (num_wires - 1) // k + 1
        avg_fragment_gates = (num_gates - 1) // k + 1
        wire_imbalance = free_wires / avg_fragment_wires - 1
        gate_imbalance = free_gates / avg_fragment_gates - 1
        imbalance = min(gate_imbalance, wire_imbalance)
        if imbalance_tolerance is not None:
            imbalance = min(imbalance, imbalance_tolerance)
        return imbalance

    def _validate_dag(
        self,
        num_tape_wires,
        num_tape_gates,
        max_wires_by_fragment,
        max_gates_by_fragment,
    ):
        """Helper parameter checker."""
        if max_wires_by_fragment is not None:
            assert isinstance(max_wires_by_fragment, (list, tuple))
            assert all(isinstance(i, int) and i <= num_tape_wires for i in max_wires_by_fragment)
            if self.max_free_wires is not None:
                assert all(i <= self.max_free_wires for i in max_wires_by_fragment)
        if max_gates_by_fragment is not None:
            assert isinstance(max_gates_by_fragment, (list, tuple))
            assert all(isinstance(i, int) and i <= num_tape_gates for i in max_gates_by_fragment)
            if self.max_free_gates is not None:
                assert all(i <= self.max_free_gates for i in max_gates_by_fragment)
        if max_wires_by_fragment is not None and max_gates_by_fragment is not None:
            assert len(max_wires_by_fragment) == len(max_gates_by_fragment)

    def _infer_default_configs(
        self,
        num_tape_wires,
        num_tape_gates,
        max_wires_by_fragment=None,
        max_gates_by_fragment=None,
    ):
        """
        Helper function for deriving the best default partitioning parameters for the automatic
        circuit cutter. Arguments can be either user provided or inferred from devices.

        Args:
            num_tape_wires (int): Number of wires in the circuit tape to be partitioned.
            num_tape_gates (int): Number of gates in the circuit tape to be partitioned.
            k_lower (int): Lowest number of fragments to attempt for the automatic circuit cutter.
                Optional, defaults to the scenario where each fragment is executed on the largest
                available device, satisfying both wire and gate constraints.
                If supplied, can be higher than said default if the the desired k is known.
            k_upper (int): Largest number of fragments to attempt for the automatic circuit cutter.
                Optional, defaults to the scenario where each fragment is executed on the smallest
                available device, satisfying both wire and gate constraints.
                If supplied, can be higher than said default to encourage exploration.
            imbalance_tolerance (float): Maximum allowed imbalance for all partition attempts.
                Can be used to globally override the imbalance constraint parameter.
                Optional, defaults to a loose upper bound to avoid being too restrictive.
            fragment_wires (List[int]): User provided predetermined list of fragment wire limits.
                If supplied, k will be derived from it and exploration of other ks will not be made.
            fragment_gates (List[int]): User provided predetermined list of fragment gate limits.
                If supplied, k will be derived from it and exploration of other ks will not be made.

        Returns:
            cut_configs (List[CutConfig]): A list of cut configurations for passing to the automatic
                circuit cutter to attempt.

        """

        # Assumes unlimited width/depth if not supplied.
        max_free_wires = self.max_free_wires or num_tape_wires
        max_free_gates = self.max_free_gates or num_tape_gates

        # Assumes same number of wires/gates across all devices if min_free_* not provided.
        min_free_wires = self.min_free_wires or max_free_wires
        min_free_gates = self.min_free_gates or max_free_gates

        # The lower bound of k corresponds to executing each fragment on the largest available device.
        k_lb = 1 + min(
            (num_tape_wires - 1) // max_free_wires,  # wire limited
            (num_tape_gates - 1) // max_free_gates,  # gate limited
        )
        # The upper bound of k corresponds to executing each fragment on the smallest available device.
        k_ub = 1 + max(
            (num_tape_wires - 1) // min_free_wires,  # wire limited
            (num_tape_gates - 1) // min_free_gates,  # gate limited
        )

        # The global imbalance tolerance, if not given, defaults to a very loose upper bound:
        imbalance_tolerance = self.imbalance_tolerance or k_ub
        if not (isinstance(imbalance_tolerance, (float, int)) and imbalance_tolerance >= 0):
            raise ValueError(
                "The global `imbalance_tolerance` is expected to be a non-negative number, "
                f"got {type(imbalance_tolerance)} with value {imbalance_tolerance}."
            )

        cut_kwargs = []

        if max_gates_by_fragment is None and max_wires_by_fragment is None:

            # k_lower, when supplied by a user, can be higher than k_lb if the the desired k is known:
            k_lower = int(self.k_lower or k_lb)
            # k_upper, when supplied by a user, can be higher than k_ub to encourage exploration:
            k_upper = int(self.k_upper or k_ub)

            if k_lower < k_lb:
                warnings.warn(
                    f"The provided `k_lower={k_lower}` is less than the lowest allowed value, "
                    f"will override and set `k_lower={k_lb}`."
                )
                k_lower = k_lb
            if k_upper < k_lower:
                warnings.warn(
                    f"The provided `k_upper={k_upper}` is less than `k_lower={k_lower}`, "
                    f"will override and set `k_upper={k_lower}`. "
                    "Note this will result in only one partitioning attempt, with the number of fragments "
                    f"fixed at {k_lower}, rather than exploring into higher number of fragments."
                )
                k_upper = k_lower

            HIGH_NUM_FRAGMENTS = 20
            if k_lower > HIGH_NUM_FRAGMENTS:
                warnings.warn(
                    f"The attempted number of fragments seems high with lower bound at {k_lower}, "
                    "are you sure?"
                )

            # Prepare the list of ks to explore:
            ks = list(range(k_lower, k_upper + 1))

            HIGH_PARTITION_ATTEMPS = 20
            if len(ks) > HIGH_PARTITION_ATTEMPS:
                warnings.warn(
                    f"The numer of partition attempts seems high ({len(ks)}), are you sure?"
                )
        else:
            # When the by-fragment wire and/or gate limits are supplied, derive k and imbalance and
            # return a single partition config.
            ks = [len(max_wires_by_fragment)]

        for k in ks:
            imbalance = self.infer_imbalance(
                k,
                num_tape_wires,
                num_tape_gates,
                max_free_wires if max_wires_by_fragment is None else max(max_wires_by_fragment),
                max_free_gates if max_gates_by_fragment is None else max(max_gates_by_fragment),
                imbalance_tolerance,
            )
            cut_kwargs.append(
                {
                    "num_fragments": k,
                    "imbalance": imbalance,
                    "max_wires_by_fragment": max_wires_by_fragment,
                    "max_gates_by_fragment": max_gates_by_fragment,
                }
            )

        return cut_kwargs


@dataclass(frozen=True)
class CutSpec:
    """
    Container wrapping partitioning results returned from the automatic cutter.
    NOTE: Incomplete.
    """

    trial_id: Tuple[int, int]
    num_wires: int
    num_gates: int
    config: CutConfig
    fragments: Dict[Any, Any]
    raw_cuts: List[Any]  # i.e. including "hyper-wires"
    raw_cost: Union[int, float]
    # mode: str = "wire"

    @property
    def k(self) -> int:
        """Property: the number of partitions."""
        return self.config.k

    @property
    def cuts(self):
        """Property: the list of cut edges."""
        # removes hyperedges, assuming they are at the end of raw_cuts.
        return self.raw_cuts[: (self.num_gates if self.mode == "gate" else self.num_wires)]

    @property
    def num_cuts(self) -> int:
        """Property: the number of cut edges."""
        return len(self.cuts)

    @property
    def cost(self) -> int:
        """Property: cost of circuit cut"""
        # TODO: should be different for gate cut
        return self.num_cuts / 2

    @property
    def fragment_gates(self):
        """Property: dict of fragment sizes {fragment_id: fragment_size}"""
        return {k: len(gates) for k, gates in self.fragments.items()}

    @staticmethod
    def collect_fragment_wires(gates):
        """Collects wires from a fragment"""
        # TODO: implement this.
        return {w for g in gates for w in g.wires}

    @property
    def fragment_wires(self):
        """Property: dict of number of wires by fragment {fragment_id: wire_size}"""
        return {k: len(self.collect_wires(gates)) for k, gates in self.fragments.items()}
